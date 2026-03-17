"""Compatibility checks for candidate combinations."""

from __future__ import annotations

from dataclasses import dataclass

from clustro.search.search_space import Candidate


@dataclass(slots=True)
class CompatibilityDecision:
    allowed: bool
    reasons: list[str]


def validate_candidate(candidate: Candidate, *, n_rows: int, n_features: int) -> CompatibilityDecision:
    reasons: list[str] = []
    representation_name = candidate.representation["name"]
    clustering_name = candidate.clustering["name"]
    clustering_params = candidate.clustering["params"]

    if clustering_name == "agglomerative" and clustering_params.get("linkage") == "ward":
        if clustering_params.get("metric", "euclidean") != "euclidean":
            reasons.append("ward_requires_euclidean")

    if clustering_name == "gmm" and clustering_params.get("covariance_type") == "full":
        if n_features > max(5, n_rows // 10):
            reasons.append("gmm_full_high_dimensionality")

    if clustering_name == "hdbscan" and representation_name == "umap" and n_rows < 25:
        reasons.append("hdbscan_umap_requires_more_rows")

    if representation_name == "autoencoder" and n_features < 2:
        reasons.append("autoencoder_requires_multiple_features")

    if clustering_name in {"ae_kmeans", "ae_gmm", "dec", "vade"} and n_features < 2:
        reasons.append("deep_clusterer_requires_multiple_features")

    if representation_name != "none" and clustering_name in {"ae_kmeans", "ae_gmm", "dec", "vade"}:
        reasons.append("deep_clusterer_requires_raw_processed_features")

    if clustering_name == "ae_gmm" and int(clustering_params.get("latent_dim", 10)) > max(2, n_rows // 5):
        reasons.append("ae_gmm_latent_dim_too_large")

    return CompatibilityDecision(allowed=not reasons, reasons=reasons)
