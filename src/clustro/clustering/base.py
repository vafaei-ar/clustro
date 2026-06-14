"""Clustering abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class ClusteringResult:
    labels: np.ndarray
    metadata: dict[str, object]
    # Latent matrix used directly for clustering (deep methods only).
    # None means the clustering was done on the input matrix itself.
    cluster_space_matrix: np.ndarray | None = field(default=None)
