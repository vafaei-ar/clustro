"""Candidate execution scheduler."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from clustro.clustering.wrappers import fit_predict_clusterer
from clustro.config.schema import ExperimentConfig
from clustro.evaluation.acceptance import evaluate_acceptance
from clustro.evaluation.metrics_internal import compute_internal_metrics
from clustro.evaluation.metrics_stability import (
    PerturbationLabelRun,
    cluster_balance,
    summarize_perturbation_stability,
    summarize_seed_stability,
)
from clustro.evaluation.metrics_structure import structure_summary
from clustro.repr.ae_repr import AutoencoderRepresentation
from clustro.repr.none_repr import IdentityRepresentation
from clustro.repr.pca_repr import PcaRepresentation
from clustro.repr.umap_repr import UmapRepresentation
from clustro.search.early_screen import pilot_subset
from clustro.search.pruners import should_prune
from clustro.search.search_space import Candidate


@dataclass(slots=True)
class CandidateExecution:
    candidate: Candidate
    labels: np.ndarray
    seed_label_runs: list[np.ndarray]
    perturbation_label_runs: list[PerturbationLabelRun]
    metrics: dict[str, float]
    accepted: bool
    rejection_reasons: list[str]
    runtime_seconds: float
    search_stage: str = "full_evaluated"


def evaluate_candidate(
    candidate: Candidate,
    matrix: np.ndarray,
    config: ExperimentConfig,
) -> CandidateExecution:
    pilot_metrics, runtime = evaluate_candidate_pilot(candidate, matrix, config)
    prune, prune_reasons = should_prune(pilot_metrics, runtime_seconds=runtime)
    if prune:
        pilot_metrics["runtime_seconds"] = runtime
        return CandidateExecution(
            candidate=candidate,
            labels=np.full(len(matrix), -1, dtype=int),
            seed_label_runs=[],
            perturbation_label_runs=[],
            metrics=pilot_metrics,
            accepted=False,
            rejection_reasons=prune_reasons,
            runtime_seconds=runtime,
            search_stage="pilot_pruned",
        )

    return evaluate_candidate_full(candidate, matrix, config)


def evaluate_candidate_pilot(
    candidate: Candidate,
    matrix: np.ndarray,
    config: ExperimentConfig,
) -> tuple[dict[str, float], float]:
    start = time.perf_counter()
    pilot_indices = pilot_subset(
        matrix,
        sample_fraction=config.search.pilot_sample_fraction,
        min_rows=config.search.pilot_min_rows,
        seed=config.experiment.random_seed,
    )
    pilot_matrix = matrix[pilot_indices]
    pilot_metrics = _run_candidate(candidate, pilot_matrix, config, seeds=config.search.seeds_pilot)
    runtime = time.perf_counter() - start
    return pilot_metrics, runtime


def evaluate_candidate_full(
    candidate: Candidate,
    matrix: np.ndarray,
    config: ExperimentConfig,
) -> CandidateExecution:
    start = time.perf_counter()
    full_metrics, final_labels, seed_label_runs, perturbation_label_runs = (
        _run_candidate_with_perturbations(
            candidate,
            matrix,
            config,
        )
    )
    runtime = time.perf_counter() - start
    full_metrics["runtime_seconds"] = runtime
    decision = evaluate_acceptance(full_metrics, config)
    return CandidateExecution(
        candidate=candidate,
        labels=final_labels,
        seed_label_runs=seed_label_runs,
        perturbation_label_runs=perturbation_label_runs,
        metrics={**full_metrics, "final_weighted_score": decision.final_weighted_score},
        accepted=decision.accepted,
        rejection_reasons=decision.reasons,
        runtime_seconds=runtime,
        search_stage="full_evaluated",
    )


def evaluate_candidate_batch(
    candidates: list[Candidate],
    matrix: np.ndarray,
    config: ExperimentConfig,
) -> list[CandidateExecution]:
    executions: list[CandidateExecution] = []
    for candidate in candidates:
        executions.append(evaluate_candidate(candidate, matrix, config))
    return executions


def evaluate_candidate_batch_ray(
    candidates: list[Candidate],
    matrix: np.ndarray,
    config: ExperimentConfig,
) -> list[CandidateExecution]:
    if not candidates:
        return []
    try:
        import ray  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Ray requested but not installed. Use clustro[tracking].") from exc
    if not ray.is_initialized():
        raise RuntimeError("Ray requested but was not initialized before candidate evaluation.")

    matrix_ref = ray.put(matrix)
    config_ref = ray.put(config)
    futures = [
        _evaluate_candidate_remote.remote(candidate, matrix_ref, config_ref)
        for candidate in candidates
    ]
    # ray.get(list) preserves input order, keeping registries reproducible across
    # serial and Ray-backed execution for the same candidate list.
    return list(ray.get(futures))


def _ray_remote_evaluate_candidate(
    candidate: Candidate, matrix: np.ndarray, config: ExperimentConfig
) -> CandidateExecution:
    return evaluate_candidate(candidate, matrix, config)


try:
    import ray  # type: ignore

    _evaluate_candidate_remote = ray.remote(_ray_remote_evaluate_candidate)
except ImportError:
    _evaluate_candidate_remote = None


def executions_to_frame(executions: list[CandidateExecution]) -> pd.DataFrame:
    rows = []
    for execution in executions:
        row = {
            "candidate_id": execution.candidate.candidate_id,
            "family": execution.candidate.family,
            "continuous_transform": execution.candidate.preprocessing.get("continuous_transform"),
            "categorical_encoding": execution.candidate.preprocessing.get("categorical_encoding"),
            "representation_name": execution.candidate.representation["name"],
            "representation_params_json": json.dumps(
                execution.candidate.representation.get("params", {}), sort_keys=True
            ),
            "clustering_name": execution.candidate.clustering["name"],
            "clustering_params_json": json.dumps(
                execution.candidate.clustering.get("params", {}), sort_keys=True
            ),
            "accepted": execution.accepted,
            "hard_filter_passed": execution.accepted,
            "hard_rejection_reasons": ";".join(execution.rejection_reasons),
            "accepted_before_top_fraction": execution.accepted,
            "final_rejection_reasons": ";".join(execution.rejection_reasons),
            "search_stage": execution.search_stage,
            "rejection_reasons": ";".join(execution.rejection_reasons),
            **execution.metrics,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _run_candidate_with_perturbations(
    candidate: Candidate,
    matrix: np.ndarray,
    config: ExperimentConfig,
) -> tuple[dict[str, float], np.ndarray, list[np.ndarray], list[PerturbationLabelRun]]:
    metrics, label_runs = _collect_seed_runs(
        candidate, matrix, config.search.seeds_full, config=config
    )
    perturbation_runs = _collect_perturbations(candidate, matrix, config)
    stability_metrics = summarize_seed_stability(label_runs)
    perturbation_metrics = summarize_perturbation_stability(label_runs[0], perturbation_runs)
    silhouette_n_jobs = 1 if config.experiment.deterministic_mode == "strict" else None
    internal_metrics = compute_internal_metrics(
        matrix, label_runs[0], silhouette_n_jobs=silhouette_n_jobs
    )
    structure_metrics = structure_summary(label_runs[0])
    valid_labels = label_runs[0][label_runs[0] >= 0]
    cluster_sizes = np.bincount(valid_labels) if len(valid_labels) else np.array([0])
    min_cluster_fraction = (
        float(cluster_sizes.min() / len(label_runs[0])) if cluster_sizes.sum() else 0.0
    )
    combined = {
        **metrics,
        **internal_metrics,
        **stability_metrics,
        **perturbation_metrics,
        **structure_metrics,
        "cluster_balance": cluster_balance(label_runs[0]),
        "min_cluster_fraction": min_cluster_fraction,
        "parsimony_penalty": float(matrix.shape[1]) / max(matrix.shape[0], 1),
        "runtime_penalty": 0.0,
    }
    return combined, label_runs[0], label_runs, perturbation_runs


def _run_candidate(
    candidate: Candidate,
    matrix: np.ndarray,
    config: ExperimentConfig,
    *,
    seeds: list[int],
) -> dict[str, float]:
    metrics, label_runs = _collect_seed_runs(candidate, matrix, seeds, config=config)
    stability_metrics = summarize_seed_stability(label_runs)
    silhouette_n_jobs = 1 if config.experiment.deterministic_mode == "strict" else None
    internal_metrics = compute_internal_metrics(
        matrix, label_runs[0], silhouette_n_jobs=silhouette_n_jobs
    )
    return {**metrics, **internal_metrics, **stability_metrics, **structure_summary(label_runs[0])}


def _collect_seed_runs(
    candidate: Candidate,
    matrix: np.ndarray,
    seeds: list[int],
    *,
    config: ExperimentConfig,
) -> tuple[dict[str, float], list[np.ndarray]]:
    label_runs: list[np.ndarray] = []
    collected_metadata: list[dict[str, object]] = []
    for seed in seeds:
        representation_matrix = _fit_representation(candidate, matrix, seed, config=config)
        result = fit_predict_clusterer(
            candidate.clustering["name"],
            representation_matrix,
            candidate.clustering["params"],
            seed=seed,
            use_gpu_if_available=config.experiment.use_gpu_if_available,
            deterministic_mode=config.experiment.deterministic_mode,
        )
        label_runs.append(result.labels)
        collected_metadata.append(result.metadata)
    return {
        "seed_runs": float(len(seeds)),
        **_summarize_cluster_metadata(collected_metadata),
    }, label_runs


def _collect_perturbations(
    candidate: Candidate, matrix: np.ndarray, config: ExperimentConfig
) -> list[PerturbationLabelRun]:
    perturbations: list[PerturbationLabelRun] = []
    rng = np.random.default_rng(config.experiment.random_seed)
    for _ in range(config.search.perturbations_full):
        if config.search.perturbation_type == "bootstrap":
            indices = rng.choice(len(matrix), size=len(matrix), replace=True)
            kind = "bootstrap"
        else:
            indices = np.sort(
                rng.choice(len(matrix), size=max(2, int(len(matrix) * 0.8)), replace=False)
            )
            kind = "subsample"
        sampled = matrix[indices]
        representation = _fit_representation(
            candidate, sampled, config.experiment.random_seed, config=config
        )
        labels = fit_predict_clusterer(
            candidate.clustering["name"],
            representation,
            candidate.clustering["params"],
            seed=config.experiment.random_seed,
            use_gpu_if_available=config.experiment.use_gpu_if_available,
            deterministic_mode=config.experiment.deterministic_mode,
        ).labels
        perturbations.append(
            PerturbationLabelRun(indices=np.asarray(indices), labels=np.asarray(labels), kind=kind)
        )
    return perturbations


def _fit_representation(
    candidate: Candidate,
    matrix: np.ndarray,
    seed: int,
    *,
    config: ExperimentConfig,
) -> np.ndarray:
    name = candidate.representation["name"]
    params = dict(candidate.representation["params"])
    if name == "none":
        return IdentityRepresentation().fit_transform(matrix).matrix
    if name == "pca":
        return PcaRepresentation(random_state=seed, **params).fit_transform(matrix).matrix
    if name == "umap":
        return UmapRepresentation(random_state=seed, **params).fit_transform(matrix).matrix
    if name == "autoencoder":
        return (
            AutoencoderRepresentation(
                random_state=seed,
                use_gpu_if_available=config.experiment.use_gpu_if_available,
                deterministic_mode=config.experiment.deterministic_mode,
                **params,
            )
            .fit_transform(matrix)
            .matrix
        )
    raise ValueError(f"Unsupported representation method: {name}")


def _summarize_cluster_metadata(metadata_rows: list[dict[str, object]]) -> dict[str, float]:
    if not metadata_rows:
        return {}
    summary: dict[str, float] = {}
    keys = {key for row in metadata_rows for key in row}
    for key in keys:
        values = [row[key] for row in metadata_rows if key in row]
        numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
        if numeric_values:
            summary[key] = float(np.mean(numeric_values))
    return summary
