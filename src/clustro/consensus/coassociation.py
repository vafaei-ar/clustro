"""Weighted co-association matrix construction."""

from __future__ import annotations

import numpy as np


def build_coassociation_matrix(label_runs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if not label_runs:
        raise ValueError("Consensus requires at least one accepted run.")
    n_samples = len(label_runs[0])
    matrix = np.zeros((n_samples, n_samples), dtype=float)
    denominator = np.zeros((n_samples, n_samples), dtype=float)

    for labels, weight in zip(label_runs, weights, strict=True):
        same_cluster = (
            (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0) & (labels[None, :] >= 0)
        )
        matrix += weight * same_cluster.astype(float)
        denominator += weight

    with np.errstate(divide="ignore", invalid="ignore"):
        consensus = np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0)
    np.fill_diagonal(consensus, 1.0)
    return consensus
