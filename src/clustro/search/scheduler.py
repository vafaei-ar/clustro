"""Candidate execution scheduler."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from clustro.clustering.wrappers import fit_predict_clusterer
from clustro.config.schema import ExperimentConfig
from clustro.evaluation.acceptance import evaluate_acceptance
from clustro.evaluation.metrics_internal import compute_internal_metrics
from clustro.evaluation.metrics_stability import (
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
    perturbation_label_runs: list[np.ndarray]
    metrics: dict[str, float]
    accepted: bool
    rejection_reasons: list[str]
    runtime_seconds: float


def evaluate_candidate_batch(
    candidates: list[Candidate],
    matrix: np.ndarray,
    config: ExperimentConfig,
) -> list[CandidateExecution]:
    executions: list[CandidateExecution] = []
    pilot_indices = pilot_subset(
        matrix,
        sample_fraction=config.search.pilot_sample_fraction,
        min_rows=config.search.pilot_min_rows,
        seed=config.experiment.random_seed,
    )
    pilot_matrix = matrix[pilot_indices]

    for candidate in candidates:
        start = time.perf_counter()
        pilot_metrics = _run_candidate(candidate, pilot_matrix, config, seeds=config.search.seeds_pilot)
        runtime = time.perf_counter() - start
        prune, prune_reasons = should_prune(pilot_metrics, runtime_seconds=runtime)
        if prune:
            pilot_metrics["runtime_seconds"] = runtime
            executions.append(
                CandidateExecution(
                    candidate=candidate,
                    labels=np.full(len(matrix), -1, dtype=int),
                    seed_label_runs=[],
                    perturbation_label_runs=[],
                    metrics=pilot_metrics,
                    accepted=False,
                    rejection_reasons=prune_reasons,
                    runtime_seconds=runtime,
                )
            )
            continue

        start = time.perf_counter()
        full_metrics, final_labels, seed_label_runs, perturbation_label_runs = _run_candidate_with_perturbations(
            candidate,
            matrix,
            config,
        )
        runtime = time.perf_counter() - start
        full_metrics["runtime_seconds"] = runtime
        decision = evaluate_acceptance(full_metrics, config)
        executions.append(
            CandidateExecution(
                candidate=candidate,
                labels=final_labels,
                seed_label_runs=seed_label_runs,
                perturbation_label_runs=perturbation_label_runs,
                metrics={**full_metrics, "final_weighted_score": decision.final_weighted_score},
                accepted=decision.accepted,
                rejection_reasons=decision.reasons,
                runtime_seconds=runtime,
            )
        )
    return executions


def executions_to_frame(executions: list[CandidateExecution]) -> pd.DataFrame:
    rows = []
    for execution in executions:
        row = {
            "candidate_id": execution.candidate.candidate_id,
            "family": execution.candidate.family,
            "representation_name": execution.candidate.representation["name"],
            "clustering_name": execution.candidate.clustering["name"],
            "accepted": execution.accepted,
            "rejection_reasons": ";".join(execution.rejection_reasons),
            **execution.metrics,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _run_candidate_with_perturbations(
    candidate: Candidate,
    matrix: np.ndarray,
    config: ExperimentConfig,
) -> tuple[dict[str, float], np.ndarray, list[np.ndarray], list[np.ndarray]]:
    metrics, label_runs = _collect_seed_runs(candidate, matrix, config.search.seeds_full, config=config)
    perturbation_runs = _collect_perturbations(candidate, matrix, config)
    stability_metrics = summarize_seed_stability(label_runs)
    perturbation_metrics = summarize_perturbation_stability(label_runs[0], perturbation_runs)
    internal_metrics = compute_internal_metrics(matrix, label_runs[0])
    structure_metrics = structure_summary(label_runs[0])
    valid_labels = label_runs[0][label_runs[0] >= 0]
    cluster_sizes = np.bincount(valid_labels) if len(valid_labels) else np.array([0])
    min_cluster_fraction = float(cluster_sizes.min() / len(label_runs[0])) if cluster_sizes.sum() else 0.0
    combined = {
        **metrics,
        **internal_metrics,
        **stability_metrics,
        **perturbation_metrics,
        **structure_metrics,
        "cluster_balance": cluster_balance(label_runs[0]),
        "min_cluster_fraction": min_cluster_fraction,
        "parsimony_penalty": -float(matrix.shape[1]) / max(matrix.shape[0], 1),
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
    internal_metrics = compute_internal_metrics(matrix, label_runs[0])
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
    return {"seed_runs": float(len(seeds)), **_summarize_cluster_metadata(collected_metadata)}, label_runs


def _collect_perturbations(candidate: Candidate, matrix: np.ndarray, config: ExperimentConfig) -> list[np.ndarray]:
    perturbations: list[np.ndarray] = []
    rng = np.random.default_rng(config.experiment.random_seed)
    for _ in range(config.search.perturbations_full):
        if config.search.perturbation_type == "bootstrap":
            indices = rng.choice(len(matrix), size=len(matrix), replace=True)
        else:
            indices = np.sort(rng.choice(len(matrix), size=max(2, int(len(matrix) * 0.8)), replace=False))
        sampled = matrix[indices]
        representation = _fit_representation(candidate, sampled, config.experiment.random_seed, config=config)
        labels = fit_predict_clusterer(
            candidate.clustering["name"],
            representation,
            candidate.clustering["params"],
            seed=config.experiment.random_seed,
            use_gpu_if_available=config.experiment.use_gpu_if_available,
            deterministic_mode=config.experiment.deterministic_mode,
        ).labels
        if len(indices) == len(matrix):
            perturbations.append(labels)
        else:
            full = np.full(len(matrix), -1, dtype=int)
            full[indices] = labels
            perturbations.append(full)
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
        return AutoencoderRepresentation(
            random_state=seed,
            use_gpu_if_available=config.experiment.use_gpu_if_available,
            deterministic_mode=config.experiment.deterministic_mode,
            **params,
        ).fit_transform(matrix).matrix
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
