"""Identity representation."""

from __future__ import annotations

import numpy as np

from clustro.repr.base import RepresentationResult


class IdentityRepresentation:
    name = "none"

    def fit_transform(self, matrix: np.ndarray) -> RepresentationResult:
        return RepresentationResult(matrix=matrix, metadata={"name": self.name})
