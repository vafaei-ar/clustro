"""Clustering abstractions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ClusteringResult:
    labels: np.ndarray
    metadata: dict[str, object]
