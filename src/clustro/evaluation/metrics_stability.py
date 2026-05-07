"""Seed and perturbation stability metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass(slots=True)
class PerturbationLabelRun:
    indices: np.ndarray
    labels: np.ndarray
    kind: Literal["bootstrap", "subsample"]


def summarize_seed_stability(label_runs: list[np.ndarray]) -> dict[str, float]:
    if len(label_runs) < 2:
        return {"ari_seed": 1.0, "nmi_seed": 1.0}
    ari_values = []
    nmi_values = []
    for left, right in combinations(label_runs, 2):
        ari_values.append(adjusted_rand_score(left, right))
        nmi_values.append(normalized_mutual_info_score(left, right))
    return {
        "ari_seed": float(np.mean(ari_values)),
        "nmi_seed": float(np.mean(nmi_values)),
    }


def summarize_perturbation_stability(
    reference_labels: np.ndarray, perturbation_runs: list[PerturbationLabelRun]
) -> dict[str, float]:
    if not perturbation_runs:
        return {"mean_cluster_jaccard": 1.0}
    values: list[float] = []
    compared_rows: list[int] = []
    for run in perturbation_runs:
        reference_subset, perturbation_subset, indices = _prepare_perturbation_comparison(
            reference_labels, run
        )
        if len(indices) == 0:
            continue
        values.append(_aligned_mean_jaccard(reference_subset, perturbation_subset))
        compared_rows.append(len(indices))
    if not values:
        return {"mean_cluster_jaccard": 0.0, "perturbation_rows_compared_mean": 0.0}
    return {
        "mean_cluster_jaccard": float(np.mean(values)),
        "perturbation_rows_compared_mean": float(np.mean(compared_rows)),
    }


def _prepare_perturbation_comparison(
    reference_labels: np.ndarray,
    run: PerturbationLabelRun,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray(run.indices, dtype=int)
    labels = np.asarray(run.labels, dtype=int)
    if len(indices) != len(labels):
        raise ValueError("Perturbation indices and labels must have equal length.")
    if len(indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    if int(indices.min()) < 0 or int(indices.max()) >= len(reference_labels):
        raise ValueError("Perturbation indices must refer to original row positions.")

    mapped: dict[int, int] = {}
    if run.kind == "subsample":
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Subsample perturbation indices must be unique.")
        mapped = {int(index): int(label) for index, label in zip(indices, labels, strict=True)}
    else:
        for index, label in zip(indices, labels, strict=True):
            mapped.setdefault(int(index), int(label))

    unique_indices = np.asarray(sorted(mapped), dtype=int)
    perturbation_labels = np.asarray([mapped[int(index)] for index in unique_indices], dtype=int)
    return reference_labels[unique_indices], perturbation_labels, unique_indices


def _aligned_mean_jaccard(reference_labels: np.ndarray, other_labels: np.ndarray) -> float:
    ref_clusters = [cluster for cluster in np.unique(reference_labels) if cluster >= 0]
    other_clusters = [cluster for cluster in np.unique(other_labels) if cluster >= 0]
    if not ref_clusters or not other_clusters:
        return 0.0
    scores = []
    for ref_cluster in ref_clusters:
        ref_mask = reference_labels == ref_cluster
        best = 0.0
        for other_cluster in other_clusters:
            other_mask = other_labels == other_cluster
            intersection = np.logical_and(ref_mask, other_mask).sum()
            union = np.logical_or(ref_mask, other_mask).sum()
            if union:
                best = max(best, intersection / union)
        scores.append(best)
    return float(np.mean(scores))


def cluster_balance(labels: np.ndarray) -> float:
    counts = Counter(labels[labels >= 0].tolist())
    if not counts:
        return 0.0
    shares = np.array(list(counts.values()), dtype=float)
    shares = shares / shares.sum()
    return float(1.0 - np.std(shares))
