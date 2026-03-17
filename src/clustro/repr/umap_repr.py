"""UMAP representation wrapper."""

from __future__ import annotations

import numpy as np
import umap

from clustro.repr.base import RepresentationResult


class UmapRepresentation:
    name = "umap"

    def __init__(
        self,
        *,
        n_components: int,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int | None = None,
    ) -> None:
        self.model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )

    def fit_transform(self, matrix: np.ndarray) -> RepresentationResult:
        transformed = self.model.fit_transform(matrix)
        return RepresentationResult(matrix=np.asarray(transformed), metadata={"name": self.name})
