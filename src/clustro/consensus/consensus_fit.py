"""Consensus clustering from co-association matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering

from clustro.consensus.coassociation import build_coassociation_matrix
from clustro.consensus.uncertainty import compute_uncertainty
from clustro.evaluation.metrics_stability import _aligned_mean_jaccard


@dataclass(slots=True)
class ConsensusResult:
    labels: np.ndarray
    coassociation: np.ndarray
    uncertainty: pd.DataFrame
    cluster_summary: pd.DataFrame
    bootstrap_stability: pd.DataFrame


def fit_consensus(
    label_runs: list[np.ndarray],
    weights: np.ndarray,
    row_ids: list[str],
    *,
    target_k: int,
    method: str = "hierarchical_on_coassociation",
    bootstrap_repeats: int = 0,
    random_seed: int = 0,
    coassociation: np.ndarray | None = None,
    ambiguous_top2_gap_threshold: float = 0.10,
    ambiguous_entropy_quantile: float = 0.90,
) -> ConsensusResult:
    coassociation = (
        coassociation
        if coassociation is not None
        else build_coassociation_matrix(label_runs, weights)
    )
    labels = cluster_from_coassociation(
        coassociation, target_k=target_k, method=method, random_seed=random_seed
    )
    uncertainty = compute_uncertainty(
        coassociation,
        labels,
        row_ids,
        ambiguous_top2_gap_threshold=ambiguous_top2_gap_threshold,
        ambiguous_entropy_quantile=ambiguous_entropy_quantile,
    )
    cluster_summary = summarize_consensus_clusters(coassociation, labels, uncertainty)
    bootstrap_stability = bootstrap_consensus_stability(
        label_runs,
        weights,
        row_ids,
        final_labels=labels,
        target_k=target_k,
        method=method,
        repeats=bootstrap_repeats,
        random_seed=random_seed,
    )
    return ConsensusResult(
        labels=labels,
        coassociation=coassociation,
        uncertainty=uncertainty,
        cluster_summary=cluster_summary,
        bootstrap_stability=bootstrap_stability,
    )


def cluster_from_coassociation(
    coassociation: np.ndarray,
    *,
    target_k: int,
    method: str,
    random_seed: int = 0,
) -> np.ndarray:
    if method == "spectral_on_coassociation":
        model = SpectralClustering(
            n_clusters=target_k,
            affinity="precomputed",
            random_state=random_seed,
            assign_labels="kmeans",
        )
        return np.asarray(model.fit_predict(coassociation), dtype=int)

    distance = 1.0 - coassociation
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    return fcluster(tree, t=target_k, criterion="maxclust") - 1


def summarize_consensus_clusters(
    coassociation: np.ndarray,
    labels: np.ndarray,
    uncertainty: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for cluster in sorted(int(value) for value in np.unique(labels) if value >= 0):
        members = np.where(labels == cluster)[0]
        within = coassociation[np.ix_(members, members)]
        non_diagonal = within[~np.eye(len(members), dtype=bool)]
        cluster_uncertainty = uncertainty.loc[uncertainty["consensus_label"] == cluster]
        rows.append(
            {
                "consensus_label": cluster,
                "cluster_size": int(len(members)),
                "mean_within_cluster_consensus": float(non_diagonal.mean())
                if non_diagonal.size
                else 1.0,
                "median_sample_entropy": float(cluster_uncertainty["entropy"].median())
                if not cluster_uncertainty.empty
                else 0.0,
                "mean_sample_entropy": float(cluster_uncertainty["entropy"].mean())
                if not cluster_uncertainty.empty
                else 0.0,
                "ambiguous_fraction": float(cluster_uncertainty["ambiguous"].mean())
                if not cluster_uncertainty.empty
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_consensus_stability(
    label_runs: list[np.ndarray],
    weights: np.ndarray,
    row_ids: list[str],
    *,
    final_labels: np.ndarray,
    target_k: int,
    method: str,
    repeats: int,
    random_seed: int,
) -> pd.DataFrame:
    clusters = sorted(int(value) for value in np.unique(final_labels) if value >= 0)
    if repeats <= 0 or not label_runs:
        return pd.DataFrame(
            {
                "consensus_label": clusters,
                "bootstrap_recovery_mean": [np.nan for _ in clusters],
                "bootstrap_recovery_std": [np.nan for _ in clusters],
                "bootstrap_repeats": [0 for _ in clusters],
            }
        )

    rng = np.random.default_rng(random_seed)
    scores: dict[int, list[float]] = {cluster: [] for cluster in clusters}
    for _ in range(repeats):
        indices = rng.choice(len(label_runs), size=len(label_runs), replace=True)
        sampled_runs = [label_runs[index] for index in indices]
        sampled_weights = weights[indices]
        total = sampled_weights.sum()
        if total > 0:
            sampled_weights = sampled_weights / total
        bootstrap = fit_consensus(
            sampled_runs,
            sampled_weights,
            row_ids,
            target_k=target_k,
            method=method,
            bootstrap_repeats=0,
            random_seed=random_seed,
        )
        for cluster in clusters:
            final_mask = np.where(final_labels == cluster, cluster, -1)
            scores[cluster].append(_aligned_mean_jaccard(final_mask, bootstrap.labels))

    return pd.DataFrame(
        [
            {
                "consensus_label": cluster,
                "bootstrap_recovery_mean": float(np.mean(values)) if values else 0.0,
                "bootstrap_recovery_std": float(np.std(values)) if values else 0.0,
                "bootstrap_repeats": int(len(values)),
            }
            for cluster, values in scores.items()
        ]
    )
