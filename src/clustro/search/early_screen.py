"""Pilot-stage candidate screening."""

from __future__ import annotations

import numpy as np


def pilot_subset(matrix: np.ndarray, *, sample_fraction: float, min_rows: int, seed: int) -> np.ndarray:
    if len(matrix) <= min_rows:
        return np.arange(len(matrix))
    rng = np.random.default_rng(seed)
    n_rows = max(min_rows, int(len(matrix) * sample_fraction))
    return np.sort(rng.choice(len(matrix), size=n_rows, replace=False))
