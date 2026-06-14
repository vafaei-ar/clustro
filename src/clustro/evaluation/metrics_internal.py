"""Internal clustering metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def compute_internal_metrics(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    silhouette_n_jobs: int | None = None,
) -> dict[str, float]:
    labels = np.asarray(labels)
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    valid_matrix = matrix[valid_mask]
    unique = set(valid_labels.tolist())
    # sklearn silhouette_score requires 1 < n_labels < n_samples; guard both edges.
    if len(unique) < 2 or len(unique) >= len(valid_matrix):
        return {
            "silhouette": -1.0,
            "davies_bouldin": float("inf"),
            "calinski_harabasz": 0.0,
        }
    silhouette_kw: dict[str, object] = {}
    if silhouette_n_jobs is not None:
        silhouette_kw["n_jobs"] = silhouette_n_jobs
    return {
        "silhouette": float(silhouette_score(valid_matrix, valid_labels, **silhouette_kw)),
        "davies_bouldin": float(davies_bouldin_score(valid_matrix, valid_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(valid_matrix, valid_labels)),
    }
