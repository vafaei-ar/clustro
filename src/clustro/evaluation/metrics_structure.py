"""Structural sanity metrics for clustering outputs."""

from __future__ import annotations

from collections import Counter

import numpy as np


def structure_summary(labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    valid = labels[labels >= 0]
    total = len(labels)
    if total == 0:
        return {
            "n_clusters": 0.0,
            "noise_fraction": 0.0,
            "dominant_cluster_fraction": 0.0,
            "tiny_cluster_fraction": 0.0,
        }
    counts = Counter(valid.tolist())
    dominant = max(counts.values()) / total if counts else 0.0
    tiny = sum(1 for count in counts.values() if count / total < 0.03)
    return {
        "n_clusters": float(len(counts)),
        "noise_fraction": float((labels < 0).mean()),
        "dominant_cluster_fraction": float(dominant),
        "tiny_cluster_fraction": float(tiny / max(len(counts), 1)),
    }
