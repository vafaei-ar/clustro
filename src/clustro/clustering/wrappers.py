"""Clustering dispatcher across classical and deep families."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from clustro.clustering.base import ClusteringResult
from clustro.clustering.classical import fit_predict_clusterer as fit_predict_classical_clusterer
from clustro.clustering.deep_dec import fit_predict_dec
from clustro.clustering.deep_vade import fit_predict_vade
from clustro.repr.ae_repr import train_autoencoder


def fit_predict_clusterer(
    name: str,
    matrix: np.ndarray,
    params: dict[str, object],
    *,
    seed: int,
    use_gpu_if_available: bool = False,
    deterministic_mode: str = "fast",
) -> ClusteringResult:
    if name in {
        "kmeans",
        "minibatch_kmeans",
        "gmm",
        "agglomerative",
        "hdbscan",
        "spectral",
        "optics",
        "birch",
    }:
        return fit_predict_classical_clusterer(
            name,
            matrix,
            params,
            seed=seed,
            use_gpu_if_available=use_gpu_if_available,
        )

    if name in {"ae_kmeans", "ae_gmm"}:
        latent = train_autoencoder(
            matrix,
            latent_dim=int(params.get("latent_dim", 10)),
            hidden_layers=_list_param(params.get("hidden_layers", [128, 64])),
            dropout=float(params.get("dropout", 0.0)),
            epochs=int(params.get("epochs", 100)),
            batch_size=int(params.get("batch_size", 256)),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            early_stopping_patience=int(params.get("early_stopping_patience", 10)),
            random_state=seed,
            use_gpu_if_available=use_gpu_if_available,
            deterministic_mode=deterministic_mode,
        )
        if name == "ae_kmeans":
            model = KMeans(
                n_clusters=int(params.get("n_clusters", 3)), random_state=seed, n_init="auto"
            )
            labels = model.fit_predict(latent.latent)
            metadata = {
                "inertia": float(model.inertia_),
                "reconstruction_loss": latent.reconstruction_loss,
            }
        else:
            model = GaussianMixture(
                n_components=int(params.get("n_clusters", params.get("n_components", 3))),
                covariance_type=str(params.get("covariance_type", "diag")),
                random_state=seed,
            )
            labels = model.fit_predict(latent.latent)
            metadata = {
                "bic": float(model.bic(latent.latent)),
                "reconstruction_loss": latent.reconstruction_loss,
                "average_confidence": float(model.predict_proba(latent.latent).max(axis=1).mean()),
            }
        metadata["latent_dim"] = latent.latent.shape[1]
        metadata["device"] = latent.metadata["device"]
        return ClusteringResult(labels=np.asarray(labels), metadata=metadata)

    if name == "dec":
        result = fit_predict_dec(
            matrix,
            params,
            seed=seed,
            use_gpu_if_available=use_gpu_if_available,
            deterministic_mode=deterministic_mode,
        )
        return ClusteringResult(
            labels=result.labels,
            metadata={
                "dec_loss": result.loss,
                "latent_dim": result.latent.shape[1],
                "reconstruction_loss": result.reconstruction_loss,
                "average_confidence": result.average_confidence,
                "assignment_entropy": result.assignment_entropy,
                "dec_iterations": result.iterations,
            },
        )

    if name == "vade":
        result = fit_predict_vade(
            matrix,
            params,
            seed=seed,
            use_gpu_if_available=use_gpu_if_available,
            deterministic_mode=deterministic_mode,
        )
        return ClusteringResult(
            labels=result.labels,
            metadata={
                "vade_loss": result.loss,
                "vade_bic": result.bic,
                "latent_dim": result.latent.shape[1],
                "average_confidence": result.average_confidence,
                "assignment_entropy": result.assignment_entropy,
                "probabilities": result.probabilities.tolist(),
            },
        )

    raise ValueError(f"Unsupported clustering method: {name}")


def _list_param(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, tuple):
        return [int(item) for item in value]
    return [int(value)]
