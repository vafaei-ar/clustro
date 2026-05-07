"""Optional Optuna integration for candidate evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import optuna
import pandas as pd

from clustro.config.schema import ExperimentConfig, OptunaConfig
from clustro.evaluation.acceptance import compute_weighted_score
from clustro.search.pruners import should_prune
from clustro.search.scheduler import (
    CandidateExecution,
    evaluate_candidate_full,
    evaluate_candidate_pilot,
)
from clustro.search.search_space import Candidate
from clustro.utils.io import ensure_directory


@dataclass(slots=True)
class TrialBundle:
    trial: optuna.trial.Trial
    candidate: Candidate


def create_study(name: str, optuna_config: OptunaConfig, *, random_seed: int) -> optuna.Study:
    return optuna.create_study(
        direction="maximize",
        study_name=name,
        sampler=_build_sampler(optuna_config, random_seed=random_seed),
        pruner=_build_pruner(optuna_config),
    )


def optimize_family_candidates(
    *,
    family: str,
    candidates: list[Candidate],
    matrix,
    config: ExperimentConfig,
    study_name: str,
    output_dir: Path,
) -> list[CandidateExecution]:
    if not candidates:
        return []

    max_trials = min(len(candidates), config.search.optuna.n_trials_per_family)
    selected_candidates = sorted(candidates, key=lambda candidate: candidate.candidate_id)[
        :max_trials
    ]
    study = create_study(
        study_name, config.search.optuna, random_seed=config.experiment.random_seed
    )
    executions: list[CandidateExecution] = []
    trial_rows: list[dict[str, object]] = []

    for candidate in selected_candidates:
        trial = study.ask()
        pilot_metrics, pilot_runtime = evaluate_candidate_pilot(candidate, matrix, config)
        pilot_score = compute_weighted_score(pilot_metrics, config)
        trial.report(pilot_score, step=0)

        prune, prune_reasons = should_prune(pilot_metrics, runtime_seconds=pilot_runtime)
        should_optuna_prune = trial.should_prune()
        if prune or should_optuna_prune:
            reasons = list(prune_reasons)
            if should_optuna_prune:
                reasons.append("optuna_pruned")
            metrics = {
                **pilot_metrics,
                "runtime_seconds": pilot_runtime,
                "final_weighted_score": pilot_score,
            }
            execution = CandidateExecution(
                candidate=candidate,
                labels=_invalid_labels(len(matrix)),
                seed_label_runs=[],
                perturbation_label_runs=[],
                metrics=metrics,
                accepted=False,
                rejection_reasons=reasons,
                runtime_seconds=pilot_runtime,
            )
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            trial_rows.append(
                {
                    "trial_number": trial.number,
                    "candidate_id": candidate.candidate_id,
                    "family": family,
                    "status": "pruned",
                    "pilot_score": pilot_score,
                    "final_weighted_score": pilot_score,
                    "reasons": ";".join(reasons),
                }
            )
            executions.append(execution)
            continue

        execution = evaluate_candidate_full(candidate, matrix, config)
        study.tell(trial, execution.metrics.get("final_weighted_score", 0.0))
        trial_rows.append(
            {
                "trial_number": trial.number,
                "candidate_id": candidate.candidate_id,
                "family": family,
                "status": "completed",
                "pilot_score": pilot_score,
                "final_weighted_score": execution.metrics.get("final_weighted_score", 0.0),
                "reasons": ";".join(execution.rejection_reasons),
            }
        )
        executions.append(execution)

    _write_study_outputs(output_dir, family, trial_rows, study)
    return executions


def _write_study_outputs(
    output_dir: Path,
    family: str,
    rows: list[dict[str, object]],
    study: optuna.Study,
) -> None:
    studies_dir = ensure_directory(output_dir / "optuna")
    if rows:
        pd.DataFrame(rows).to_csv(studies_dir / f"{family}_trials.csv", index=False)
    best_payload: dict[str, object] = {
        "family": family,
        "best_trial_number": None,
        "best_value": None,
    }
    if study.best_trials:
        best_trial = max(
            study.best_trials,
            key=lambda trial: trial.value if trial.value is not None else float("-inf"),
        )
        best_payload["best_trial_number"] = best_trial.number
        best_payload["best_value"] = best_trial.value
    (studies_dir / f"{family}_study_summary.json").write_text(
        json.dumps(best_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_sampler(optuna_config: OptunaConfig, *, random_seed: int) -> optuna.samplers.BaseSampler:
    if optuna_config.sampler == "RandomSampler":
        return optuna.samplers.RandomSampler(seed=random_seed)
    return optuna.samplers.TPESampler(seed=random_seed)


def _build_pruner(optuna_config: OptunaConfig) -> optuna.pruners.BasePruner:
    if optuna_config.pruner == "NopPruner":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0)


def _invalid_labels(length: int):
    import numpy as np

    return np.full(length, -1, dtype=int)
