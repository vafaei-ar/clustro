"""Tests for code-quality cleanup fixes (Task 1-4 from cleanup pass)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from clustro.config.schema import ReportingConfig
from clustro.config.validators import load_experiment_config
from clustro.interpretation.profiling import (
    _safe_mode,
    build_cluster_profiles,
    build_pairwise_cluster_contrasts,
)
from clustro.reporting.exports import export_report_bundle

# ---------------------------------------------------------------------------
# Task 1 — Validator accepts new method names and still accepts old aliases
# ---------------------------------------------------------------------------


def _make_config_yaml(tmp_path: Path, method_name: str) -> Path:
    dataset = tmp_path / "data.csv"
    dataset.write_text("id,x,flag,group\n1,1.0,0,a\n2,2.0,1,b\n", encoding="utf-8")
    config = {
        "experiment": {"name": "test", "output_dir": str(tmp_path / "out")},
        "data": {
            "path": str(dataset),
            "id_column": "id",
            "column_schema": {
                "continuous": ["x"],
                "binary": ["flag"],
                "categorical": ["group"],
                "ordinal": [],
            },
        },
        "clustering": {"methods": [{"name": method_name, "params": {"n_clusters": [2]}}]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


@pytest.mark.parametrize(
    "method",
    [
        "ae_centroid_refinement",
        "vae_gmm",
        "dec",
        "vade",
        "kmeans",
        "ae_kmeans",
        "ae_gmm",
    ],
)
def test_validator_accepts_method(tmp_path: Path, method: str) -> None:
    config_path = _make_config_yaml(tmp_path, method)
    cfg = load_experiment_config(config_path)
    assert cfg.clustering.methods[0].name == method


def test_validator_rejects_unknown_method(tmp_path: Path) -> None:
    config_path = _make_config_yaml(tmp_path, "unknown_method_xyz")
    with pytest.raises(ValueError, match="Unsupported clustering method"):
        load_experiment_config(config_path)


# ---------------------------------------------------------------------------
# Task 2 — ReportingConfig: no export_format field, flags wire through
# ---------------------------------------------------------------------------


def test_reporting_config_has_no_export_format() -> None:
    cfg = ReportingConfig()
    assert not hasattr(cfg, "export_format")


def test_reporting_config_defaults() -> None:
    cfg = ReportingConfig()
    assert cfg.generate_figures is True
    assert cfg.generate_tables is True
    assert cfg.manuscript_bundle is True


def test_reporting_config_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        ReportingConfig.model_validate({"generate_figures": True, "export_format": ["csv"]})


def _make_candidate_registry() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["c1", "c2"],
            "family": ["kmeans", "kmeans"],
            "representation_name": ["none", "none"],
            "clustering_name": ["kmeans", "kmeans"],
            "silhouette": [0.4, 0.5],
            "ari_seed": [0.3, 0.4],
            "final_weighted_score": [0.6, 0.7],
            "accepted": [True, True],
            "davies_bouldin": [1.0, 0.9],
            "calinski_harabasz": [100.0, 120.0],
            "nmi_seed": [0.3, 0.4],
            "mean_cluster_jaccard": [0.5, 0.6],
            "cluster_balance": [0.8, 0.85],
            "search_stage": ["full_evaluated", "full_evaluated"],
            "final_rejection_reasons": ["", ""],
            "accepted_before_top_fraction": [True, True],
        }
    )


def test_generate_figures_false_skips_png(tmp_path: Path) -> None:
    registry = _make_candidate_registry()
    cfg = ReportingConfig(generate_figures=False, generate_tables=True, manuscript_bundle=False)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    with patch("clustro.reporting.exports.export_quality_vs_stability") as mock_fig:
        export_report_bundle(registry, tmp_path, reporting_config=cfg)
        mock_fig.assert_not_called()


def test_generate_tables_false_skips_optional_csv(tmp_path: Path) -> None:
    registry = _make_candidate_registry()
    cfg = ReportingConfig(generate_figures=False, generate_tables=False, manuscript_bundle=False)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    export_report_bundle(registry, tmp_path, reporting_config=cfg)

    # candidate_metrics.csv is always written regardless of generate_tables
    assert (report_dir / "candidate_metrics.csv").exists()
    # optional table skipped
    assert not (report_dir / "quality_vs_stability.csv").exists()
    assert not (report_dir / "accepted_candidate_heatmap.csv").exists()


def test_manuscript_bundle_false_skips_populate(tmp_path: Path) -> None:
    registry = _make_candidate_registry()
    cfg = ReportingConfig(generate_figures=False, generate_tables=False, manuscript_bundle=False)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    with patch("clustro.reporting.exports.populate_manuscript_bundle") as mock_bundle:
        export_report_bundle(registry, tmp_path, reporting_config=cfg)
        mock_bundle.assert_not_called()


def test_manuscript_bundle_true_calls_populate(tmp_path: Path) -> None:
    registry = _make_candidate_registry()
    cfg = ReportingConfig(generate_figures=False, generate_tables=False, manuscript_bundle=True)
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    with patch("clustro.reporting.exports.populate_manuscript_bundle") as mock_bundle:
        export_report_bundle(registry, tmp_path, reporting_config=cfg)
        mock_bundle.assert_called_once_with(tmp_path)


def test_default_reporting_config_used_when_none(tmp_path: Path) -> None:
    registry = _make_candidate_registry()
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    with patch("clustro.reporting.exports.populate_manuscript_bundle") as mock_bundle:
        with patch("clustro.reporting.exports.export_quality_vs_stability"):
            with patch("clustro.reporting.exports.export_search_flow_diagram"):
                with patch("clustro.reporting.exports.export_metric_heatmap"):
                    export_report_bundle(registry, tmp_path)
        mock_bundle.assert_called_once()


# ---------------------------------------------------------------------------
# Task 3 — _safe_mode helper handles NaN-only and empty series
# ---------------------------------------------------------------------------


def test_safe_mode_normal_values() -> None:
    s = pd.Series(["a", "b", "a", "c"])
    assert _safe_mode(s) == "a"


def test_safe_mode_all_nan() -> None:
    s = pd.Series([float("nan"), float("nan"), float("nan")])
    assert _safe_mode(s) == "unknown"


def test_safe_mode_empty_series() -> None:
    s = pd.Series([], dtype=object)
    assert _safe_mode(s) == "unknown"


def test_safe_mode_mixed_nan_and_values() -> None:
    s = pd.Series([None, "dog", "cat", "dog", None])
    assert _safe_mode(s) == "dog"


def test_safe_mode_custom_missing_label() -> None:
    s = pd.Series([float("nan")])
    assert _safe_mode(s, missing_label="N/A") == "N/A"


def test_build_cluster_profiles_all_nan_categorical() -> None:
    from clustro.data.schema import DatasetSchema

    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "flag": [0, 1, 0, 1],
            "group": [None, None, None, None],
        }
    )
    labels = pd.Series([0, 0, 1, 1])
    schema = DatasetSchema(
        continuous=["x"], binary=["flag"], categorical=["group"], ordinal=[]
    )
    profiles = build_cluster_profiles(frame, labels, schema)
    cat_rows = profiles[profiles["feature_type"] == "categorical"]
    assert not cat_rows.empty
    assert (cat_rows["value"] == "unknown").all()


def test_build_pairwise_contrasts_all_nan_categorical() -> None:
    from clustro.data.schema import DatasetSchema

    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "flag": [0, 1, 0, 1],
            "group": [None, None, None, None],
        }
    )
    labels = pd.Series([0, 0, 1, 1])
    schema = DatasetSchema(
        continuous=["x"], binary=["flag"], categorical=["group"], ordinal=[]
    )
    contrasts = build_pairwise_cluster_contrasts(frame, labels, schema)
    cat_rows = contrasts[contrasts["feature_type"] == "categorical"]
    assert not cat_rows.empty
    # Both modes are "unknown", so value should be "unknown vs unknown"
    assert (cat_rows["value"] == "unknown vs unknown").all()
