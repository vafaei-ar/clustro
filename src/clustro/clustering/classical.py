"""Classical clustering wrappers used in Milestone 1."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import (
    OPTICS,
    AgglomerativeClustering,
    Birch,
    KMeans,
    MiniBatchKMeans,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture

from clustro.clustering.base import ClusteringResult


def fit_predict_clusterer(
    name: str,
    matrix: np.ndarray,
    params: dict[str, object],
    *,
    seed: int,
    use_gpu_if_available: bool = False,
) -> ClusteringResult:
    if use_gpu_if_available:
        rapids_result = _fit_predict_rapids(name, matrix, params, seed=seed)
        if rapids_result is not None:
            return rapids_result

    if name == "kmeans":
        model = KMeans(random_state=seed, n_init="auto", **params)
        labels = model.fit_predict(matrix)
        return ClusteringResult(
            labels=np.asarray(labels),
            metadata={
                "inertia": float(model.inertia_),
                **_sklearn_metadata(name, use_gpu_if_available),
            },
        )

    if name == "minibatch_kmeans":
        model = MiniBatchKMeans(random_state=seed, n_init="auto", **params)
        labels = model.fit_predict(matrix)
        return ClusteringResult(
            labels=np.asarray(labels),
            metadata={
                "inertia": float(model.inertia_),
                **_sklearn_metadata(name, use_gpu_if_available),
            },
        )

    if name == "gmm":
        model = GaussianMixture(random_state=seed, **params)
        labels = model.fit_predict(matrix)
        return ClusteringResult(
            labels=np.asarray(labels),
            metadata={
                "bic": float(model.bic(matrix)),
                **_sklearn_metadata(name, use_gpu_if_available),
            },
        )

    if name == "agglomerative":
        model = AgglomerativeClustering(**params)
        labels = model.fit_predict(matrix)
        return ClusteringResult(
            labels=np.asarray(labels), metadata=_sklearn_metadata(name, use_gpu_if_available)
        )

    if name == "hdbscan":
        model = HDBSCAN(**params)
        labels = model.fit_predict(matrix)
        return ClusteringResult(
            labels=np.asarray(labels),
            metadata={
                "probabilities": model.probabilities_.tolist(),
                **_sklearn_metadata(name, use_gpu_if_available),
            },
        )

    if name == "spectral":
        model = SpectralClustering(random_state=seed, **params)
        labels = model.fit_predict(matrix)
        return ClusteringResult(
            labels=np.asarray(labels), metadata=_sklearn_metadata(name, use_gpu_if_available)
        )

    if name == "optics":
        model = OPTICS(**params)
        labels = model.fit_predict(matrix)
        return ClusteringResult(
            labels=np.asarray(labels),
            metadata={
                "reachability_mean": float(np.nanmean(model.reachability_)),
                "ordering_size": int(model.ordering_.shape[0]),
                **_sklearn_metadata(name, use_gpu_if_available),
            },
        )

    if name == "birch":
        model = Birch(**params)
        labels = model.fit_predict(matrix)
        return ClusteringResult(
            labels=np.asarray(labels),
            metadata={
                "subcluster_count": int(len(model.subcluster_centers_)),
                **_sklearn_metadata(name, use_gpu_if_available),
            },
        )

    raise ValueError(f"Unsupported clustering method: {name}")


def _fit_predict_rapids(
    name: str,
    matrix: np.ndarray,
    params: dict[str, object],
    *,
    seed: int,
) -> ClusteringResult | None:
    if name not in {"kmeans", "gmm"}:
        return None
    try:
        if name == "kmeans":
            cluster_module = import_module("cuml.cluster")
            model_cls = cluster_module.KMeans
            rapids_params = dict(params)
            rapids_params.setdefault("random_state", seed)
            model = model_cls(**rapids_params)
            labels = _to_numpy(model.fit_predict(matrix)).astype(int)
            inertia = getattr(model, "inertia_", None)
            metadata: dict[str, object] = {
                "accelerator": "rapids",
                "rapids_estimator": "cuml.cluster.KMeans",
            }
            if inertia is not None:
                metadata["inertia"] = float(_scalar(inertia))
            return ClusteringResult(labels=labels, metadata=metadata)

        mixture_module = import_module("cuml.mixture")
        model_cls = mixture_module.GaussianMixture
        rapids_params = dict(params)
        if "n_components" not in rapids_params and "n_clusters" in rapids_params:
            rapids_params["n_components"] = rapids_params.pop("n_clusters")
        rapids_params.setdefault("random_state", seed)
        model = model_cls(**rapids_params)
        labels = _to_numpy(model.fit_predict(matrix)).astype(int)
        return ClusteringResult(
            labels=labels,
            metadata={
                "accelerator": "rapids",
                "rapids_estimator": "cuml.mixture.GaussianMixture",
            },
        )
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
        return None


def _sklearn_metadata(name: str, use_gpu_if_available: bool) -> dict[str, object]:
    metadata: dict[str, object] = {"accelerator": "sklearn"}
    if use_gpu_if_available:
        if name in {"kmeans", "gmm"}:
            metadata["accelerator_fallback_reason"] = "rapids_unavailable_or_failed"
        else:
            metadata["accelerator_fallback_reason"] = "rapids_method_not_supported"
    return metadata


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy())
    if hasattr(value, "get"):
        return np.asarray(value.get())
    return np.asarray(value)


def _scalar(value: Any) -> float:
    array = _to_numpy(value)
    return float(array.reshape(-1)[0])
