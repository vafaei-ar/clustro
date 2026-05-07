"""Sampling helpers for pilot and perturbation stages."""

from __future__ import annotations

import numpy as np


def sample_without_replacement(
    n_rows: int,
    *,
    sample_fraction: float,
    min_rows: int,
    seed: int,
) -> np.ndarray:
    if n_rows <= min_rows:
        return np.arange(n_rows)
    rng = np.random.default_rng(seed)
    sample_size = min(n_rows, max(min_rows, int(n_rows * sample_fraction)))
    return np.sort(rng.choice(n_rows, size=sample_size, replace=False))


def bootstrap_indices(n_rows: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(n_rows, size=n_rows, replace=True)


def subsample_indices(
    n_rows: int,
    *,
    sample_fraction: float = 0.8,
    min_rows: int = 2,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sample_size = min(n_rows, max(min_rows, int(n_rows * sample_fraction)))
    return np.sort(rng.choice(n_rows, size=sample_size, replace=False))
