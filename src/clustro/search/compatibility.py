"""Compatibility checks for candidate combinations."""

from __future__ import annotations

from dataclasses import dataclass

from clustro.search.search_space import Candidate

# All clusterers that train their own internal autoencoder and therefore require
# the raw processed feature matrix (representation must be "none") and multiple input features.
DEEP_CLUSTERERS: frozenset[str] = frozenset(
    {
        "ae_kmeans",
        "ae_gmm",
        "ae_centroid_refinement",
        "vae_gmm",
        "dec",  # deprecated alias for ae_centroid_refinement
        "vade",  # deprecated alias for vae_gmm
    }
)

# Fixed-k methods that need an explicit number of clusters/components in params.
_FIXED_K_CLUSTERERS: frozenset[str] = frozenset(
    {
        "kmeans",
        "minibatch_kmeans",
        "gmm",
        "agglomerative",
        "spectral",
        "birch",
        "ae_kmeans",
        "ae_gmm",
        "ae_centroid_refinement",
        "vae_gmm",
    }
)


@dataclass(slots=True)
class CompatibilityDecision:
    allowed: bool
    reasons: list[str]


def validate_candidate(
    candidate: Candidate, *, n_rows: int, n_features: int
) -> CompatibilityDecision:
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

    if clustering_name == "spectral" and n_rows > 5000:
        reasons.append("spectral_impractical_for_dataset_size")

    if representation_name == "autoencoder" and n_features < 2:
        reasons.append("autoencoder_requires_multiple_features")

    if clustering_name in DEEP_CLUSTERERS and n_features < 2:
        reasons.append("deep_clusterer_requires_multiple_features")

    if representation_name != "none" and clustering_name in DEEP_CLUSTERERS:
        reasons.append("deep_clusterer_requires_raw_processed_features")

    if clustering_name == "ae_gmm" and int(clustering_params.get("latent_dim", 10)) > max(
        2, n_rows // 5
    ):
        reasons.append("ae_gmm_latent_dim_too_large")

    # Impossible-k guard: reject before training if the requested k is out of range.
    if clustering_name in _FIXED_K_CLUSTERERS:
        k_raw = clustering_params.get("n_clusters") or clustering_params.get("n_components")
        if k_raw is not None:
            k = int(k_raw)
            if k < 2:
                reasons.append("cluster_count_less_than_two")
            elif k >= n_rows:
                reasons.append("cluster_count_too_large_for_rows")

    return CompatibilityDecision(allowed=not reasons, reasons=reasons)
