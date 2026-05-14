"""Optional Optuna integration for candidate evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from clustro.config.schema import ExperimentConfig, MethodConfig, OptunaConfig
from clustro.evaluation.acceptance import compute_weighted_score
from clustro.search.compatibility import validate_candidate
from clustro.search.pruners import should_prune
from clustro.search.scheduler import (
    CandidateExecution,
    evaluate_candidate_full,
    evaluate_candidate_pilot,
)
from clustro.search.search_space import Candidate
from clustro.utils.hashing import stable_hash
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
    matrix: Any | None = None,
    matrices: dict[str, Any] | None = None,
    config: ExperimentConfig,
    study_name: str,
    output_dir: Path,
    dataset_fingerprint: dict[str, Any] | None = None,
    raw_frame: pd.DataFrame | None = None,
) -> list[CandidateExecution]:
    if not candidates:
        return []

    max_trials = config.search.optuna.n_trials_per_family
    study = create_study(
        study_name, config.search.optuna, random_seed=config.experiment.random_seed
    )
    executions: list[CandidateExecution] = []
    trial_rows: list[dict[str, object]] = []

    for _ in range(max_trials):
        trial = study.ask()
        candidate = suggest_candidate_for_family(
            trial,
            family,
            config,
            dataset_fingerprint or {},
            candidate_pool=candidates,
        )
        candidate_matrix = _matrix_for_candidate(candidate, matrix=matrix, matrices=matrices)
        if candidate_matrix is None:
            reasons = ["sampled_preprocessing_unavailable"]
            execution = _rejected_execution(candidate, 0, reasons)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            trial_rows.append(
                _trial_row(trial, candidate, family, "failed", np.nan, np.nan, reasons)
            )
            executions.append(execution)
            continue

        compatibility = validate_candidate(
            candidate,
            n_rows=candidate_matrix.shape[0],
            n_features=candidate_matrix.shape[1],
        )
        if not compatibility.allowed:
            execution = _rejected_execution(candidate, len(candidate_matrix), compatibility.reasons)
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            trial_rows.append(
                _trial_row(
                    trial,
                    candidate,
                    family,
                    "invalid",
                    np.nan,
                    np.nan,
                    compatibility.reasons,
                )
            )
            executions.append(execution)
            continue

        pilot_metrics, pilot_runtime = evaluate_candidate_pilot(candidate, candidate_matrix, config)
        pilot_score = _pilot_score(pilot_metrics, config)
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
                labels=_invalid_labels(len(candidate_matrix)),
                seed_label_runs=[],
                perturbation_label_runs=[],
                metrics=metrics,
                accepted=False,
                rejection_reasons=reasons,
                runtime_seconds=pilot_runtime,
                search_stage="pilot_pruned",
            )
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            trial_rows.append(
                _trial_row(trial, candidate, family, "pruned", pilot_score, pilot_score, reasons)
            )
            executions.append(execution)
            continue

        execution = evaluate_candidate_full(
            candidate, candidate_matrix, config, raw_frame=raw_frame
        )
        study.tell(trial, execution.metrics.get("final_weighted_score", 0.0))
        trial_rows.append(
            _trial_row(
                trial,
                candidate,
                family,
                "completed",
                pilot_score,
                execution.metrics.get("final_weighted_score", 0.0),
                execution.rejection_reasons,
            )
        )
        executions.append(execution)

    _write_study_outputs(output_dir, family, trial_rows, study)
    return executions


def suggest_candidate_for_family(
    trial: optuna.trial.Trial,
    family: str,
    config: ExperimentConfig,
    dataset_fingerprint: dict[str, Any],
    *,
    candidate_pool: list[Candidate] | None = None,
) -> Candidate:
    transforms = _pool_values(
        candidate_pool,
        lambda candidate: str(candidate.preprocessing["continuous_transform"]),
        config.preprocessing.continuous_transforms,
    )
    transform = trial.suggest_categorical("continuous_transform", transforms)
    encodings = _pool_values(
        [
            candidate
            for candidate in candidate_pool or []
            if candidate.preprocessing["continuous_transform"] == transform
        ],
        lambda candidate: str(candidate.preprocessing["categorical_encoding"]),
        config.preprocessing.categorical_encoding,
    )
    encoding = trial.suggest_categorical("categorical_encoding", encodings)

    representation_names = _pool_values(
        [
            candidate
            for candidate in candidate_pool or []
            if candidate.preprocessing["continuous_transform"] == transform
            and candidate.preprocessing["categorical_encoding"] == encoding
        ],
        lambda candidate: str(candidate.representation["name"]),
        [method.name for method in config.representation.methods],
    )
    representation_name = trial.suggest_categorical("representation", representation_names)
    representation_method = _method_by_name(config.representation.methods, representation_name)
    clustering_method = _method_by_name(config.clustering.methods, family)
    representation_params = _suggest_params(
        trial, representation_method, prefix=representation_name
    )
    clustering_params = _suggest_params(trial, clustering_method, prefix=family)

    payload = {
        "preprocessing": {
            "continuous_transform": transform,
            "categorical_encoding": encoding,
        },
        "representation": {"name": representation_name, "params": representation_params},
        "clustering": {"name": family, "params": clustering_params},
        "dataset": dataset_fingerprint,
    }
    return Candidate(
        candidate_id=stable_hash(payload),
        preprocessing={
            "continuous_transform": transform,
            "categorical_encoding": encoding,
        },
        representation={"name": representation_name, "params": representation_params},
        clustering={"name": family, "params": clustering_params},
        family=family,
    )


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


def _invalid_labels(length: int) -> np.ndarray:
    return np.full(length, -1, dtype=int)


def _rejected_execution(
    candidate: Candidate,
    n_rows: int,
    reasons: list[str],
) -> CandidateExecution:
    return CandidateExecution(
        candidate=candidate,
        labels=_invalid_labels(n_rows),
        seed_label_runs=[],
        perturbation_label_runs=[],
        metrics={"final_weighted_score": float("nan")},
        accepted=False,
        rejection_reasons=reasons,
        runtime_seconds=0.0,
        search_stage="compatibility_rejected",
    )


def _trial_row(
    trial: optuna.trial.Trial,
    candidate: Candidate,
    family: str,
    status: str,
    pilot_score: float,
    final_weighted_score: float,
    reasons: list[str],
) -> dict[str, object]:
    return {
        "trial_number": trial.number,
        "candidate_id": candidate.candidate_id,
        "family": family,
        "status": status,
        "continuous_transform": candidate.preprocessing.get("continuous_transform"),
        "categorical_encoding": candidate.preprocessing.get("categorical_encoding"),
        "representation_name": candidate.representation.get("name"),
        "representation_params_json": json.dumps(
            candidate.representation.get("params", {}), sort_keys=True
        ),
        "clustering_name": candidate.clustering.get("name"),
        "clustering_params_json": json.dumps(
            candidate.clustering.get("params", {}), sort_keys=True
        ),
        "pilot_score": pilot_score,
        "final_weighted_score": final_weighted_score,
        "reasons": ";".join(reasons),
        "trial_params_json": json.dumps(trial.params, sort_keys=True),
    }


def _pilot_score(pilot_metrics: dict[str, float], config: ExperimentConfig) -> float:
    available_weights = {
        metric_name: weight
        for metric_name, weight in config.evaluation.acceptance.weighted_score.items()
        if _raw_metric_available(metric_name, pilot_metrics)
    }
    if not available_weights:
        return 0.0
    pilot_config = config.model_copy(
        update={
            "evaluation": config.evaluation.model_copy(
                update={
                    "acceptance": config.evaluation.acceptance.model_copy(
                        update={"weighted_score": available_weights}
                    )
                }
            )
        }
    )
    return compute_weighted_score(pilot_metrics, pilot_config)


def _raw_metric_available(metric_name: str, metrics: dict[str, float]) -> bool:
    aliases = {
        "runtime": "runtime_seconds",
        "runtime_seconds": "runtime_seconds",
        "parsimony": "parsimony_penalty",
        "parsimony_penalty": "parsimony_penalty",
    }
    return aliases.get(metric_name, metric_name) in metrics


def _matrix_for_candidate(
    candidate: Candidate,
    *,
    matrix: Any | None,
    matrices: dict[str, Any] | None,
) -> Any | None:
    if matrices is None:
        return matrix
    key = _preprocessing_key(
        candidate.preprocessing["continuous_transform"],
        candidate.preprocessing["categorical_encoding"],
    )
    preprocessed = matrices.get(key)
    if preprocessed is None:
        return None
    return getattr(preprocessed, "evaluation_matrix", preprocessed)


def _preprocessing_key(continuous_transform: object, categorical_encoding: object) -> str:
    return f"{continuous_transform}__{categorical_encoding}"


def _pool_values(
    candidates: list[Candidate] | None,
    selector: Any,
    fallback: list[str],
) -> list[str]:
    values = sorted({selector(candidate) for candidate in candidates or []})
    return values or list(fallback)


def _method_by_name(methods: list[MethodConfig], name: str) -> MethodConfig:
    for method in methods:
        if method.name == name:
            return method
    raise ValueError(f"No configured method named {name!r}.")


def _suggest_params(
    trial: optuna.trial.Trial,
    method: MethodConfig,
    *,
    prefix: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, raw_values in method.params.items():
        choices = raw_values if isinstance(raw_values, list) else [raw_values]
        params[name] = trial.suggest_categorical(f"{prefix}_{name}", choices)
    return params
