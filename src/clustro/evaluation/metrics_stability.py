"""Seed and perturbation stability metrics."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


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


def summarize_perturbation_stability(reference_labels: np.ndarray, perturbation_runs: list[np.ndarray]) -> dict[str, float]:
    if not perturbation_runs:
        return {"mean_cluster_jaccard": 1.0}
    values = [_aligned_mean_jaccard(reference_labels, labels) for labels in perturbation_runs]
    return {"mean_cluster_jaccard": float(np.mean(values))}


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
