"""Tests for publication-readiness cleanup (Tasks 1–7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from clustro.config.schema import ExperimentConfig
from clustro.config.validators import _deep_merge, _load_defaults
from clustro.evaluation.metrics_internal import compute_internal_metrics
from clustro.interpretation.permutation import compute_cv_permutation_importance
from clustro.interpretation.surrogate import fit_surrogate_model
from clustro.search.compatibility import DEEP_CLUSTERERS, validate_candidate
from clustro.search.search_space import Candidate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    clustering_name: str,
    clustering_params: dict,
    representation_name: str = "none",
) -> Candidate:
    return Candidate(
        candidate_id="test",
        preprocessing={"continuous_transform": "standard", "categorical_encoding": "onehot"},
        representation={"name": representation_name, "params": {}},
        clustering={"name": clustering_name, "params": clustering_params},
        family=f"{representation_name}-{clustering_name}",
    )


# ---------------------------------------------------------------------------
# Task 1 — DEEP_CLUSTERERS set covers new names and deprecated aliases
# ---------------------------------------------------------------------------


def test_deep_clusterers_set_contains_new_names() -> None:
    assert "ae_centroid_refinement" in DEEP_CLUSTERERS
    assert "vae_gmm" in DEEP_CLUSTERERS


def test_deep_clusterers_set_contains_deprecated_aliases() -> None:
    assert "dec" in DEEP_CLUSTERERS
    assert "vade" in DEEP_CLUSTERERS


def test_deep_clusterers_set_contains_ae_kmeans_ae_gmm() -> None:
    assert "ae_kmeans" in DEEP_CLUSTERERS
    assert "ae_gmm" in DEEP_CLUSTERERS


@pytest.mark.parametrize("method", ["ae_centroid_refinement", "vae_gmm", "dec", "vade"])
def test_deep_clusterer_rejected_with_non_none_representation(method: str) -> None:
    candidate = _make_candidate(method, {"n_clusters": 3}, representation_name="pca")
    decision = validate_candidate(candidate, n_rows=100, n_features=10)
    assert not decision.allowed
    assert "deep_clusterer_requires_raw_processed_features" in decision.reasons


@pytest.mark.parametrize("method", ["ae_centroid_refinement", "vae_gmm"])
def test_deep_clusterer_accepted_with_none_representation(method: str) -> None:
    candidate = _make_candidate(method, {"n_clusters": 3}, representation_name="none")
    decision = validate_candidate(candidate, n_rows=100, n_features=10)
    assert "deep_clusterer_requires_raw_processed_features" not in decision.reasons


@pytest.mark.parametrize("method", ["ae_centroid_refinement", "vae_gmm"])
def test_deep_clusterer_rejected_with_single_feature(method: str) -> None:
    candidate = _make_candidate(method, {"n_clusters": 3}, representation_name="none")
    decision = validate_candidate(candidate, n_rows=100, n_features=1)
    assert not decision.allowed
    assert "deep_clusterer_requires_multiple_features" in decision.reasons


# ---------------------------------------------------------------------------
# Task 2 — Impossible-k validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["kmeans", "gmm", "agglomerative", "ae_centroid_refinement"])
def test_k_less_than_two_rejected(method: str) -> None:
    param_key = "n_components" if method == "gmm" else "n_clusters"
    candidate = _make_candidate(method, {param_key: 1})
    decision = validate_candidate(candidate, n_rows=50, n_features=5)
    assert not decision.allowed
    assert "cluster_count_less_than_two" in decision.reasons


@pytest.mark.parametrize("method", ["kmeans", "agglomerative"])
def test_k_equal_to_n_rows_rejected(method: str) -> None:
    candidate = _make_candidate(method, {"n_clusters": 20})
    decision = validate_candidate(candidate, n_rows=20, n_features=5)
    assert not decision.allowed
    assert "cluster_count_too_large_for_rows" in decision.reasons


@pytest.mark.parametrize("method", ["kmeans", "agglomerative"])
def test_k_greater_than_n_rows_rejected(method: str) -> None:
    candidate = _make_candidate(method, {"n_clusters": 25})
    decision = validate_candidate(candidate, n_rows=20, n_features=5)
    assert not decision.allowed
    assert "cluster_count_too_large_for_rows" in decision.reasons


def test_valid_k_accepted() -> None:
    candidate = _make_candidate("kmeans", {"n_clusters": 3})
    decision = validate_candidate(candidate, n_rows=50, n_features=5)
    assert "cluster_count_less_than_two" not in decision.reasons
    assert "cluster_count_too_large_for_rows" not in decision.reasons


def test_hdbscan_skips_k_check() -> None:
    # hdbscan has no fixed k — k check must not fire even if params are unusual.
    candidate = _make_candidate("hdbscan", {"min_cluster_size": 5})
    decision = validate_candidate(candidate, n_rows=50, n_features=5)
    assert "cluster_count_less_than_two" not in decision.reasons
    assert "cluster_count_too_large_for_rows" not in decision.reasons


def test_k_zero_n_clusters_rejected() -> None:
    # 0 is falsy — must not be silently ignored by the old `or`-based extraction.
    candidate = _make_candidate("kmeans", {"n_clusters": 0})
    decision = validate_candidate(candidate, n_rows=50, n_features=5)
    assert not decision.allowed
    assert "cluster_count_less_than_two" in decision.reasons


def test_k_zero_n_components_rejected() -> None:
    candidate = _make_candidate("gmm", {"n_components": 0})
    decision = validate_candidate(candidate, n_rows=50, n_features=5)
    assert not decision.allowed
    assert "cluster_count_less_than_two" in decision.reasons


@pytest.mark.parametrize("method", ["dec", "vade"])
def test_deprecated_alias_k_too_large_rejected(method: str) -> None:
    candidate = _make_candidate(method, {"n_clusters": 50}, representation_name="none")
    decision = validate_candidate(candidate, n_rows=50, n_features=5)
    assert "cluster_count_too_large_for_rows" in decision.reasons


# Task 2b — compute_internal_metrics safe when k >= n_samples
def test_internal_metrics_safe_when_k_equals_n_samples() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((4, 3))
    # 4 unique labels for 4 samples → k == n_samples → would crash silhouette
    labels = np.array([0, 1, 2, 3])
    result = compute_internal_metrics(matrix, labels)
    assert result["silhouette"] == -1.0
    assert result["davies_bouldin"] == float("inf")
    assert result["calinski_harabasz"] == 0.0


def test_internal_metrics_safe_when_k_greater_than_n_samples() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((3, 3))
    labels = np.array([0, 1, 2])
    result = compute_internal_metrics(matrix, labels)
    assert result["silhouette"] == -1.0


def test_internal_metrics_normal_case_returns_finite_values() -> None:
    rng = np.random.default_rng(0)
    matrix = np.vstack([rng.standard_normal((20, 3)), rng.standard_normal((20, 3)) + 5])
    labels = np.array([0] * 20 + [1] * 20)
    result = compute_internal_metrics(matrix, labels)
    assert result["silhouette"] > 0.0
    assert np.isfinite(result["davies_bouldin"])


# ---------------------------------------------------------------------------
# Task 3 — Example YAML files pass pydantic schema validation
# ---------------------------------------------------------------------------

EXAMPLES_CONFIG_DIR = Path(__file__).parent.parent / "examples" / "configs"


@pytest.mark.parametrize("yaml_file", list(EXAMPLES_CONFIG_DIR.glob("*.yaml")))
def test_example_yaml_validates_against_schema(yaml_file: Path) -> None:
    raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    merged = _deep_merge(_load_defaults(), raw)
    # Will raise pydantic.ValidationError if any unknown/invalid field is present.
    config = ExperimentConfig.model_validate(merged)
    assert config is not None


def test_example_yamls_have_no_export_format() -> None:
    for yaml_file in EXAMPLES_CONFIG_DIR.glob("*.yaml"):
        raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        reporting = raw.get("reporting", {})
        assert "export_format" not in reporting, (
            f"{yaml_file.name} still contains deprecated reporting.export_format"
        )


# ---------------------------------------------------------------------------
# Task 4 — Deep example scoring weights are non-negative and method-universal
# ---------------------------------------------------------------------------

_METHOD_SPECIFIC = {"average_confidence", "assignment_entropy", "reconstruction_loss"}


def test_deep_example_no_method_specific_weighted_score() -> None:
    yaml_path = EXAMPLES_CONFIG_DIR / "stroke_deep_example.yaml"
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    weighted = raw.get("evaluation", {}).get("acceptance", {}).get("weighted_score", {})
    forbidden = set(weighted.keys()) & _METHOD_SPECIFIC
    assert not forbidden, f"Method-specific metrics in weighted_score: {forbidden}"


def test_deep_example_davies_bouldin_weight_positive() -> None:
    yaml_path = EXAMPLES_CONFIG_DIR / "stroke_deep_example.yaml"
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    weighted = raw.get("evaluation", {}).get("acceptance", {}).get("weighted_score", {})
    db_weight = weighted.get("davies_bouldin", 0)
    assert db_weight > 0, f"davies_bouldin weight should be positive, got {db_weight}"


# ---------------------------------------------------------------------------
# Task 5 — consensus_support.parquet written by export_consensus_outputs
# ---------------------------------------------------------------------------


def test_export_consensus_outputs_writes_consensus_support(tmp_path: Path) -> None:
    from clustro.reporting.exports import export_consensus_outputs

    labels = pd.DataFrame({"consensus_label": [0, 1, 0, 1]})
    uncertainty = pd.DataFrame(
        {
            "consensus_label": [0, 1, 0, 1],
            "top1_consensus_support": [0.9, 0.8, 0.85, 0.7],
            "entropy": [0.1, 0.2, 0.15, 0.25],
            "ambiguous": [False, False, False, True],
            "consensus_support_gap": [0.5, 0.4, 0.45, 0.2],
        }
    )
    cluster_summary = pd.DataFrame({"consensus_label": [0, 1], "size": [2, 2]})
    bootstrap_stability = pd.DataFrame({"k": [2], "mean_ari": [0.9]})

    export_consensus_outputs(
        labels=labels,
        uncertainty=uncertainty,
        cluster_summary=cluster_summary,
        bootstrap_stability=bootstrap_stability,
        output_dir=tmp_path,
    )

    assert (tmp_path / "consensus_support.parquet").exists(), "Primary artifact missing"
    assert (tmp_path / "consensus_soft_membership.parquet").exists(), "Deprecated alias missing"


# ---------------------------------------------------------------------------
# Task 6 — CV robustness to tiny clusters
# ---------------------------------------------------------------------------


class _MinimalInterpretationConfig:
    cross_validation_folds = 5
    repeated_cv_repeats = 3
    surrogate_model = "random_forest"


def test_fit_surrogate_skips_cv_when_cluster_too_small() -> None:
    rng = np.random.default_rng(42)
    # 50 samples in class 0, 1 sample in class 1 → min_class_size=1 < 2
    matrix = rng.standard_normal((51, 5))
    labels = np.array([0] * 50 + [1])
    result = fit_surrogate_model(
        matrix,
        labels,
        [f"f{i}" for i in range(5)],
        _MinimalInterpretationConfig(),
        random_seed=0,
    )
    assert result.cv_metrics.empty
    assert result.warning is not None
    assert "CV skipped" in result.warning
    assert np.isnan(result.mean_metrics["macro_f1"])
    # Full-fit estimator must still be fitted so downstream permutation importance works.
    assert hasattr(result.estimator, "predict")


def test_fit_surrogate_reduces_folds_when_cluster_has_few_samples() -> None:
    rng = np.random.default_rng(42)
    # 50 in class 0, 3 in class 1 → min_class_size=3, effective_folds=3 (< config's 5)
    matrix = rng.standard_normal((53, 5))
    labels = np.array([0] * 50 + [1] * 3)
    result = fit_surrogate_model(
        matrix,
        labels,
        [f"f{i}" for i in range(5)],
        _MinimalInterpretationConfig(),
        random_seed=0,
    )
    assert not result.cv_metrics.empty
    assert result.warning is not None
    assert "reduced" in result.warning


def test_compute_cv_permutation_importance_safe_when_cluster_too_small() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((51, 4))
    labels = np.array([0] * 50 + [1])
    feature_names = ["a", "b", "c", "d"]
    result = compute_cv_permutation_importance(
        matrix,
        labels,
        feature_names,
        _MinimalInterpretationConfig(),
        random_seed=0,
    )
    assert list(result["feature"]) == feature_names or set(result["feature"]) == set(feature_names)
    assert (result["fold_count"] == 0).all()
    assert (result["importance_mean"] == 0.0).all()
