"""Weighted co-association matrix construction."""

from __future__ import annotations

import numpy as np


def build_coassociation_matrix(
    label_runs: list[np.ndarray],
    weights: np.ndarray,
    *,
    storage: str = "auto",
    max_dense_n: int = 10000,
) -> np.ndarray:
    if not label_runs:
        raise ValueError("Consensus requires at least one accepted run.")
    weights = np.asarray(weights, dtype=np.float64)
    n_samples = len(label_runs[0])
    if n_samples > max_dense_n and storage in {"auto", "sparse"}:
        raise RuntimeError(
            "Dense co-association matrix would exceed consensus.max_dense_n. "
            "Sparse/blockwise co-association is not implemented yet; increase max_dense_n "
            "only if this allocation is intentional."
        )
    matrix = np.zeros((n_samples, n_samples), dtype=np.float64)
    denominator = np.zeros((n_samples, n_samples), dtype=np.float64)

    for labels, weight in zip(label_runs, weights, strict=True):
        labels = np.asarray(labels, dtype=np.int64)
        valid_pair = (labels[:, None] >= 0) & (labels[None, :] >= 0)
        same_cluster = (labels[:, None] == labels[None, :]) & valid_pair
        w = np.float64(weight)
        matrix += w * same_cluster.astype(np.float64, copy=False)
        denominator += w * valid_pair.astype(np.float64, copy=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0)
    np.fill_diagonal(consensus, 1.0)
    return consensus
