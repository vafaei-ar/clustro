"""Classical clustering wrappers used in Milestone 1."""

from __future__ import annotations

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture

from clustro.clustering.base import ClusteringResult


def fit_predict_clusterer(name: str, matrix: np.ndarray, params: dict[str, object], *, seed: int) -> ClusteringResult:
    if name == "kmeans":
        model = KMeans(random_state=seed, n_init="auto", **params)
        labels = model.fit_predict(matrix)
        return ClusteringResult(labels=np.asarray(labels), metadata={"inertia": float(model.inertia_)})

    if name == "gmm":
        model = GaussianMixture(random_state=seed, **params)
        labels = model.fit_predict(matrix)
        return ClusteringResult(labels=np.asarray(labels), metadata={"bic": float(model.bic(matrix))})

    if name == "agglomerative":
        model = AgglomerativeClustering(**params)
        labels = model.fit_predict(matrix)
        return ClusteringResult(labels=np.asarray(labels), metadata={})

    if name == "hdbscan":
        model = HDBSCAN(**params)
        labels = model.fit_predict(matrix)
        return ClusteringResult(labels=np.asarray(labels), metadata={"probabilities": model.probabilities_.tolist()})

    raise ValueError(f"Unsupported clustering method: {name}")
