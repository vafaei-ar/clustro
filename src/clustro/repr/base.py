"""Representation method abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(slots=True)
class RepresentationResult:
    matrix: np.ndarray
    metadata: dict[str, object]


class RepresentationMethod(Protocol):
    name: str

    def fit_transform(self, matrix: np.ndarray) -> RepresentationResult: ...
