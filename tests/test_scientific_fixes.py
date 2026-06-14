"""Tests covering all scientific fixes applied to clustro.

Each test corresponds to a specific issue identified in the scientific review.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from clustro.clustering.deep_dec import (
    AeCentroidRefinementArtifacts,
    DecArtifacts,
    fit_predict_ae_centroid_refinement,
    fit_predict_dec,
)
from clustro.clustering.deep_vade import (
    VadeArtifacts,
    VaeGmmArtifacts,
    fit_predict_vade,
    fit_predict_vae_gmm,
)
from clustro.clustering.wrappers import fit_predict_clusterer as _fit_predict_clusterer
from clustro.config.schema import ExperimentConfig
from clustro.consensus.uncertainty import compute_uncertainty
from clustro.data.schema import DatasetSchema
from clustro.evaluation.metric_utils import add_utility_columns
from clustro.evaluation.metrics_stability import (
    PerturbationLabelRun,
    _symmetric_mean_jaccard,
    cluster_balance,
    summarize_perturbation_stability,
)
from clustro.interpretation.permutation import (
    compute_full_fit_permutation_importance,
    compute_permutation_importance,
)
from clustro.interpretation.profiling import _cohens_h, build_pairwise_cluster_contrasts
from clustro.search.scheduler import (
    _run_candidate_with_perturbations,
    _summarize_seed_metrics,
)
from clustro.search.search_space import Candidate

try:
    import torch  # noqa: F401

    _torch_available = True
except ImportError:
    _torch_available = False


# ---------------------------------------------------------------------------
# 1. cluster_balance: normalized entropy, comparable across k
# ---------------------------------------------------------------------------


def test_cluster_balance_perfect_balance_returns_one() -> None:
    labels = np.array([0, 0, 1, 1])
    assert abs(cluster_balance(labels) - 1.0) < 1e-9


def test_cluster_balance_perfect_balance_k3() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert abs(cluster_balance(labels) - 1.0) < 1e-9


def test_cluster_balance_strong_imbalance_is_lower() -> None:
    balanced = np.array([0, 0, 0, 1, 1, 1])
    imbalanced = np.array([0, 0, 0, 0, 0, 1])
    assert cluster_balance(balanced) > cluster_balance(imbalanced)


def test_cluster_balance_comparable_across_k() -> None:
    # Perfectly balanced k=2 and k=6 should both return 1.0.
    k2 = np.array([0, 0, 1, 1])
    k6 = np.array([0, 1, 2, 3, 4, 5])
    assert abs(cluster_balance(k2) - 1.0) < 1e-9
    assert abs(cluster_balance(k6) - 1.0) < 1e-9


def test_cluster_balance_noise_labels_excluded() -> None:
    # -1 is noise and must not affect the balance calculation.
    with_noise = np.array([-1, 0, 0, 1, 1])
    without_noise = np.array([0, 0, 1, 1])
    assert abs(cluster_balance(with_noise) - cluster_balance(without_noise)) < 1e-9


def test_cluster_balance_all_noise_returns_zero() -> None:
    labels = np.array([-1, -1, -1])
    assert cluster_balance(labels) == 0.0


def test_cluster_balance_single_cluster_returns_one() -> None:
    labels = np.array([0, 0, 0])
    assert cluster_balance(labels) == 1.0


# ---------------------------------------------------------------------------
# 2. Symmetric Jaccard: penalises split and extra clusters
# ---------------------------------------------------------------------------


def test_symmetric_jaccard_perfect_match_returns_one() -> None:
    ref = np.array([0, 0, 1, 1])
    other = np.array([0, 0, 1, 1])
    assert abs(_symmetric_mean_jaccard(ref, other) - 1.0) < 1e-9


def test_symmetric_jaccard_split_cluster_penalised() -> None:
    # ref has 2 clusters; other splits one into two sub-clusters.
    ref = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    other = np.array([0, 0, 1, 1, 2, 2, 2, 2])  # k=3 vs k=2
    perfect = _symmetric_mean_jaccard(ref, ref)
    split = _symmetric_mean_jaccard(ref, other)
    assert split < perfect, "split cluster should reduce symmetric Jaccard"


def test_symmetric_jaccard_extra_clusters_penalised() -> None:
    ref = np.array([0, 0, 1, 1])
    # other has two extra irrelevant clusters for 2 noise-like points.
    other = np.array([0, 0, 1, 2])
    perfect = _symmetric_mean_jaccard(ref, ref)
    with_extra = _symmetric_mean_jaccard(ref, other)
    assert with_extra < perfect


def test_perturbation_stability_returns_symmetric_key() -> None:
    ref = np.array([0, 0, 1, 1])
    run = PerturbationLabelRun(
        indices=np.array([0, 1, 2, 3]),
        labels=np.array([0, 0, 1, 1]),
        kind="subsample",
    )
    result = summarize_perturbation_stability(ref, [run])
    assert "mean_cluster_jaccard" in result
    assert "mean_cluster_jaccard_symmetric" in result
    # Both keys must have the same value (symmetric is the new primary).
    assert result["mean_cluster_jaccard"] == result["mean_cluster_jaccard_symmetric"]


# ---------------------------------------------------------------------------
# 3. CH utility is candidate-intrinsic (no cross-candidate dependence)
# ---------------------------------------------------------------------------


def test_ch_utility_independent_of_other_candidates() -> None:
    frame_alone = pd.DataFrame(
        {"candidate_id": ["a"], "accepted": [True], "calinski_harabasz": [50.0]}
    )
    frame_with_peer = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "accepted": [True, True],
            "calinski_harabasz": [50.0, 500000.0],
        }
    )

    scored_alone = add_utility_columns(frame_alone, {"calinski_harabasz": 1.0})
    scored_with_peer = add_utility_columns(frame_with_peer, {"calinski_harabasz": 1.0})

    mask_a_alone = scored_alone["candidate_id"] == "a"
    mask_a_peer = scored_with_peer["candidate_id"] == "a"
    u_alone = scored_alone.loc[mask_a_alone, "utility_calinski_harabasz"].item()
    u_with_peer = scored_with_peer.loc[mask_a_peer, "utility_calinski_harabasz"].item()

    assert abs(u_alone - u_with_peer) < 1e-9, (
        f"CH utility changed from {u_alone} to {u_with_peer} "
        "just because another candidate was added"
    )


def test_higher_ch_gets_higher_utility() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["low", "high"],
            "accepted": [True, True],
            "calinski_harabasz": [10.0, 10000.0],
        }
    )
    scored = add_utility_columns(frame, {"calinski_harabasz": 1.0})
    u_low = scored.loc[scored["candidate_id"] == "low", "utility_calinski_harabasz"].item()
    u_high = scored.loc[scored["candidate_id"] == "high", "utility_calinski_harabasz"].item()
    assert u_high > u_low


# ---------------------------------------------------------------------------
# 4. Dual-space internal metrics in scheduler
# ---------------------------------------------------------------------------


def _minimal_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"name": "test", "output_dir": "out"},
            "data": {
                "path": "data.csv",
                "column_schema": {
                    "continuous": ["x"],
                    "binary": [],
                    "categorical": [],
                    "ordinal": [],
                },
            },
            "clustering": {
                "methods": [{"name": "kmeans", "params": {"n_clusters": [2]}}]
            },
        }
    )


def test_dual_space_metrics_present_when_repr_differs() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(40, 4))
    labels = np.array([0] * 20 + [1] * 20)
    # Simulate a 2D representation that is different from the original 4D matrix.
    repr_matrix = matrix[:, :2]

    summary = _summarize_seed_metrics(matrix, [labels], [repr_matrix], config=_minimal_config())

    assert "silhouette" in summary
    assert "silhouette_cluster_space" in summary
    assert "silhouette_original_space" in summary
    assert "davies_bouldin_cluster_space" in summary
    assert "davies_bouldin_original_space" in summary
    assert "calinski_harabasz_cluster_space" in summary
    assert "calinski_harabasz_original_space" in summary


def test_cluster_space_alias_equals_cluster_space_metric() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(20, 3))
    labels = np.array([0] * 10 + [1] * 10)
    repr_matrix = matrix[:, :2]

    summary = _summarize_seed_metrics(matrix, [labels], [repr_matrix], config=_minimal_config())

    assert abs(summary["silhouette"] - summary["silhouette_cluster_space"]) < 1e-9


def test_identity_repr_gives_same_cluster_and_original_space() -> None:
    rng = np.random.default_rng(2)
    matrix = rng.normal(size=(20, 3))
    labels = np.array([0] * 10 + [1] * 10)

    # When repr_matrices is None (identity), both spaces must be equal.
    summary = _summarize_seed_metrics(matrix, [labels], config=_minimal_config())

    assert abs(summary["silhouette_cluster_space"] - summary["silhouette_original_space"]) < 1e-9


# ---------------------------------------------------------------------------
# 5. parsimony_penalty is now cluster-complexity, not feature-dimensionality
# ---------------------------------------------------------------------------


def _make_candidate(n_clusters: int = 2) -> Candidate:
    return Candidate(
        candidate_id="c1",
        preprocessing={"continuous_transform": "standard", "categorical_encoding": "onehot"},
        representation={"name": "none", "params": {}},
        clustering={"name": "kmeans", "params": {"n_clusters": n_clusters}},
        family="kmeans",
    )


def _full_config(seeds: list[int] | None = None) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"name": "test", "output_dir": "out"},
            "data": {
                "path": "data.csv",
                "column_schema": {
                    "continuous": ["x", "y"],
                    "binary": [],
                    "categorical": [],
                    "ordinal": [],
                },
            },
            "search": {
                "seeds_full": seeds or [1, 2],
                "perturbations_full": 0,
                "stability_mode": "processed_matrix",
            },
            "clustering": {
                "methods": [{"name": "kmeans", "params": {"n_clusters": [2]}}]
            },
        }
    )


def test_parsimony_penalty_is_log_based_not_dimensionality() -> None:
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(60, 2))
    matrix[:30] += 3.0
    candidate = _make_candidate(n_clusters=2)
    config = _full_config()

    combined, _, _, _ = _run_candidate_with_perturbations(candidate, matrix, config)

    parsimony = combined["parsimony_penalty"]
    feature_dim = combined["feature_dimensionality_penalty"]

    # parsimony = log1p(k) / log1p(n) — for k=2, n=60, must be much less than 1
    expected_parsimony = math.log1p(2.0) / math.log1p(60.0)
    assert abs(parsimony - expected_parsimony) < 0.1, (
        f"parsimony_penalty={parsimony!r} but expected ~{expected_parsimony!r}"
    )

    # feature_dimensionality_penalty = n_features / n_samples = 2 / 60
    expected_fdp = 2.0 / 60.0
    assert abs(feature_dim - expected_fdp) < 1e-9


# ---------------------------------------------------------------------------
# 6. Uncertainty: renamed columns
# ---------------------------------------------------------------------------


def test_uncertainty_uses_consensus_support_columns() -> None:
    labels = np.array([0, 0, 1, 1])
    coassoc = np.array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=float
    )
    result = compute_uncertainty(coassoc, labels, ["a", "b", "c", "d"])

    assert "top1_consensus_support" in result.columns
    assert "top2_consensus_support" in result.columns
    assert "consensus_support_gap" in result.columns
    assert "consensus_support_0" in result.columns
    assert "consensus_support_1" in result.columns

    # Old names must NOT appear.
    assert "top1_membership" not in result.columns
    assert "top2_membership" not in result.columns
    assert "top2_gap" not in result.columns
    assert "membership_0" not in result.columns


# ---------------------------------------------------------------------------
# 7. Cohen's h for binary effect_size
# ---------------------------------------------------------------------------


def test_cohens_h_equal_proportions() -> None:
    assert abs(_cohens_h(0.5, 0.5)) < 1e-9


def test_cohens_h_strong_difference() -> None:
    h = _cohens_h(0.9, 0.1)
    assert h > 0.5, "Cohen's h should be large for a 90% vs 10% prevalence difference"


def test_cohens_h_boundary_values() -> None:
    # Should not raise for 0 or 1.
    assert math.isfinite(_cohens_h(0.0, 1.0))
    assert math.isfinite(_cohens_h(1.0, 0.0))


def test_binary_effect_size_is_cohens_h_not_raw_difference() -> None:
    frame = pd.DataFrame({"flag": [0, 0, 1, 1]})
    labels = pd.Series([0, 0, 1, 1])
    schema = DatasetSchema(continuous=[], binary=["flag"], categorical=[], ordinal=[])

    contrasts = build_pairwise_cluster_contrasts(frame, labels, schema)
    row = contrasts[contrasts["feature"] == "flag"].iloc[0]

    prevalence_diff = float(row["value"])
    effect_size = float(row["effect_size"])

    # cluster 0: flag=0 always; cluster 1: flag=1 always
    expected_h = _cohens_h(0.0, 1.0)
    assert abs(effect_size - expected_h) < 1e-6, (
        f"effect_size={effect_size:.4f} should be Cohen's h={expected_h:.4f}, "
        f"not raw diff={prevalence_diff:.4f}"
    )
    # The raw value column carries the prevalence difference.
    assert abs(prevalence_diff - (-1.0)) < 1e-9


# ---------------------------------------------------------------------------
# 8. Permutation importance: new function name, deprecation alias
# ---------------------------------------------------------------------------


def _fit_simple_rf() -> tuple[object, np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(40, 3))
    y = np.array([0] * 20 + [1] * 20)
    x[y == 1, 0] += 3.0
    rf = RandomForestClassifier(n_estimators=10, random_state=0)
    rf.fit(x, y)
    return rf, x, y, ["f0", "f1", "f2"]


def test_compute_full_fit_permutation_importance_works() -> None:
    rf, x, y, features = _fit_simple_rf()
    result = compute_full_fit_permutation_importance(rf, x, y, features, random_seed=0)
    assert not result.empty
    assert "feature" in result.columns
    assert "importance_mean" in result.columns


def test_deprecated_alias_emits_warning() -> None:
    rf, x, y, features = _fit_simple_rf()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_permutation_importance(rf, x, y, features, random_seed=0)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert not result.empty


# ---------------------------------------------------------------------------
# 9. Method name deprecation: dec → ae_centroid_refinement, vade → vae_gmm
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_dec_alias_emits_deprecation_warning() -> None:
    from clustro.clustering.wrappers import fit_predict_clusterer

    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(24, 4)).astype(np.float32)
    matrix[:12] += 2.0
    params = {
        "n_clusters": 2,
        "latent_dim": 2,
        "hidden_layers": [8],
        "pretrain_epochs": 2,
        "finetune_epochs": 2,
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_predict_clusterer("dec", matrix, params, seed=0)

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert any("ae_centroid_refinement" in str(w.message) for w in caught)


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_ae_centroid_refinement_no_warning() -> None:
    from clustro.clustering.wrappers import fit_predict_clusterer

    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(24, 4)).astype(np.float32)
    matrix[:12] += 2.0
    params = {
        "n_clusters": 2,
        "latent_dim": 2,
        "hidden_layers": [8],
        "pretrain_epochs": 2,
        "finetune_epochs": 2,
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_predict_clusterer("ae_centroid_refinement", matrix, params, seed=0)

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not dep_warnings
    assert result.labels is not None
    # New metadata key, not the old dec_loss key.
    assert "ae_centroid_refinement_loss" in result.metadata


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_vade_alias_emits_deprecation_warning() -> None:
    from clustro.clustering.wrappers import fit_predict_clusterer

    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(24, 4)).astype(np.float32)
    matrix[:12] += 2.0
    params = {"n_clusters": 2, "latent_dim": 2, "hidden_layers": [8], "epochs": 2}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_predict_clusterer("vade", matrix, params, seed=0)

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert any("vae_gmm" in str(w.message) for w in caught)


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_vae_gmm_no_warning_and_correct_metadata() -> None:
    from clustro.clustering.wrappers import fit_predict_clusterer

    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(24, 4)).astype(np.float32)
    matrix[:12] += 2.0
    params = {"n_clusters": 2, "latent_dim": 2, "hidden_layers": [8], "epochs": 2}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_predict_clusterer("vae_gmm", matrix, params, seed=0)

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not dep_warnings
    assert "vae_gmm_loss" in result.metadata
    assert "vae_gmm_bic" in result.metadata
    # Old keys must NOT appear.
    assert "vade_loss" not in result.metadata
    assert "vade_bic" not in result.metadata


# ---------------------------------------------------------------------------
# 10. Direct module-level renames in deep_dec.py and deep_vade.py
# ---------------------------------------------------------------------------
# These tests import from the implementation modules, not through wrappers.py.


def test_dec_artifacts_alias_points_to_ae_centroid_refinement_artifacts() -> None:
    assert DecArtifacts is AeCentroidRefinementArtifacts


def test_vade_artifacts_alias_points_to_vae_gmm_artifacts() -> None:
    assert VadeArtifacts is VaeGmmArtifacts


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_fit_predict_dec_module_emits_deprecation_warning() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(24, 4)).astype(np.float32)
    matrix[:12] += 2.0
    params = {
        "n_clusters": 2,
        "latent_dim": 2,
        "hidden_layers": [8],
        "pretrain_epochs": 2,
        "finetune_epochs": 2,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_predict_dec(
            matrix, params, seed=0, use_gpu_if_available=False, deterministic_mode="fast"
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert isinstance(result, AeCentroidRefinementArtifacts)


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_fit_predict_ae_centroid_refinement_module_no_warning() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(24, 4)).astype(np.float32)
    matrix[:12] += 2.0
    params = {
        "n_clusters": 2,
        "latent_dim": 2,
        "hidden_layers": [8],
        "pretrain_epochs": 2,
        "finetune_epochs": 2,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_predict_ae_centroid_refinement(
            matrix, params, seed=0, use_gpu_if_available=False, deterministic_mode="fast"
        )
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not dep
    assert isinstance(result, AeCentroidRefinementArtifacts)
    assert result.latent.shape[1] == 2


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_fit_predict_vade_module_emits_deprecation_warning() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(24, 4)).astype(np.float32)
    matrix[:12] += 2.0
    params = {"n_clusters": 2, "latent_dim": 2, "hidden_layers": [8], "epochs": 2}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_predict_vade(
            matrix, params, seed=0, use_gpu_if_available=False, deterministic_mode="fast"
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert isinstance(result, VaeGmmArtifacts)


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_fit_predict_vae_gmm_module_no_warning() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(24, 4)).astype(np.float32)
    matrix[:12] += 2.0
    params = {"n_clusters": 2, "latent_dim": 2, "hidden_layers": [8], "epochs": 2}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_predict_vae_gmm(
            matrix, params, seed=0, use_gpu_if_available=False, deterministic_mode="fast"
        )
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not dep
    assert isinstance(result, VaeGmmArtifacts)
    assert result.latent.shape[1] == 2


# ---------------------------------------------------------------------------
# 11. cluster_space_matrix propagation through ClusteringResult
# ---------------------------------------------------------------------------


def test_classical_clusterer_has_no_cluster_space_matrix() -> None:
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(30, 4))
    matrix[:15] += 3.0
    result = _fit_predict_clusterer("kmeans", matrix, {"n_clusters": 2}, seed=0)
    assert result.cluster_space_matrix is None


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_ae_kmeans_cluster_space_matrix_has_latent_shape() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(30, 4)).astype(np.float32)
    matrix[:15] += 3.0
    params = {"n_clusters": 2, "latent_dim": 2, "hidden_layers": [8], "epochs": 2}
    result = _fit_predict_clusterer("ae_kmeans", matrix, params, seed=0)
    assert result.cluster_space_matrix is not None
    assert result.cluster_space_matrix.shape == (30, 2), (
        f"Expected (30, 2) latent shape, got {result.cluster_space_matrix.shape}"
    )


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_ae_gmm_cluster_space_matrix_has_latent_shape() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(30, 4)).astype(np.float32)
    matrix[:15] += 3.0
    params = {"n_clusters": 2, "latent_dim": 2, "hidden_layers": [8], "epochs": 2}
    result = _fit_predict_clusterer("ae_gmm", matrix, params, seed=0)
    assert result.cluster_space_matrix is not None
    assert result.cluster_space_matrix.shape == (30, 2)


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_ae_centroid_refinement_cluster_space_matrix_has_latent_shape() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(30, 4)).astype(np.float32)
    matrix[:15] += 3.0
    params = {
        "n_clusters": 2,
        "latent_dim": 2,
        "hidden_layers": [8],
        "pretrain_epochs": 2,
        "finetune_epochs": 2,
    }
    result = _fit_predict_clusterer("ae_centroid_refinement", matrix, params, seed=0)
    assert result.cluster_space_matrix is not None
    assert result.cluster_space_matrix.shape == (30, 2)


@pytest.mark.skipif(not _torch_available, reason="torch not installed")
def test_vae_gmm_cluster_space_matrix_has_latent_shape() -> None:
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(30, 4)).astype(np.float32)
    matrix[:15] += 3.0
    params = {"n_clusters": 2, "latent_dim": 2, "hidden_layers": [8], "epochs": 2}
    result = _fit_predict_clusterer("vae_gmm", matrix, params, seed=0)
    assert result.cluster_space_matrix is not None
    assert result.cluster_space_matrix.shape == (30, 2)
