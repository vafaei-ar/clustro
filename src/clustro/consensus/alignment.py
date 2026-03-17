"""Label alignment helpers."""

from __future__ import annotations

import numpy as np


def labels_to_membership(labels: np.ndarray) -> np.ndarray:
    clusters = [cluster for cluster in np.unique(labels) if cluster >= 0]
    membership = np.zeros((len(labels), len(clusters)), dtype=float)
    for index, cluster in enumerate(clusters):
        membership[:, index] = (labels == cluster).astype(float)
    return membership
