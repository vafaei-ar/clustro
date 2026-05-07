"""PCA representation wrapper."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from clustro.repr.base import RepresentationResult


class PcaRepresentation:
    name = "pca"

    def __init__(
        self, *, n_components: int, whiten: bool = False, random_state: int | None = None
    ) -> None:
        self.model = PCA(n_components=n_components, whiten=whiten, random_state=random_state)

    def fit_transform(self, matrix: np.ndarray) -> RepresentationResult:
        transformed = self.model.fit_transform(matrix)
        return RepresentationResult(
            matrix=transformed,
            metadata={
                "name": self.name,
                "explained_variance_ratio": self.model.explained_variance_ratio_.tolist(),
            },
        )
