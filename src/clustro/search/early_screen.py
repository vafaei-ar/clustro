"""Pilot-stage candidate screening."""

from __future__ import annotations

import numpy as np

from clustro.data.sampling import sample_without_replacement


def pilot_subset(
    matrix: np.ndarray, *, sample_fraction: float, min_rows: int, seed: int
) -> np.ndarray:
    return sample_without_replacement(
        len(matrix),
        sample_fraction=sample_fraction,
        min_rows=min_rows,
        seed=seed,
    )
