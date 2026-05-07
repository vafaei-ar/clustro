"""Dataset splitting helpers for reproducible downstream validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class SplitIndices:
    train: np.ndarray
    test: np.ndarray


def stratified_train_test_indices(
    labels: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int,
) -> SplitIndices:
    indices = np.arange(len(labels))
    train, test = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return SplitIndices(train=np.sort(train), test=np.sort(test))


def train_test_indices(
    n_rows: int,
    *,
    test_size: float = 0.2,
    seed: int,
) -> SplitIndices:
    indices = np.arange(n_rows)
    train, test = train_test_split(indices, test_size=test_size, random_state=seed)
    return SplitIndices(train=np.sort(train), test=np.sort(test))
