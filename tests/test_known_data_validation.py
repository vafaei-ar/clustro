"""Integration test: known-cluster validation on a deterministic toy dataset.

Checks that clustro recovers the expected 3-group structure from a fully
separable synthetic biomedical dataset and writes the expected publication
artifacts.  No torch, xgboost, SHAP, RAPIDS, Ray, or MLflow is required.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.metrics import adjusted_rand_score

from clustro import Experiment
from clustro.config.validators import load_experiment_config

# ---------------------------------------------------------------------------
# Paths to committed example files
# ---------------------------------------------------------------------------

REPO = Path(__file__).parent.parent
EXAMPLE_DATA = REPO / "examples" / "data" / "known_clusters.csv"
EXAMPLE_CONFIG = REPO / "examples" / "configs" / "known_clusters_example.yaml"

# ---------------------------------------------------------------------------
# Config loader: injects tmp_path for output dir so the repo tree stays clean
# ---------------------------------------------------------------------------


def _write_test_config(tmp_path: Path) -> Path:
    """Return a temp config YAML with absolute data and output paths."""
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    raw["data"]["path"] = str(EXAMPLE_DATA)
    raw["experiment"]["output_dir"] = str(tmp_path / "results")
    config_path = tmp_path / "known_clusters_test.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# Fixture: run the experiment once and share results across all test functions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def experiment_outputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("known_clusters")
    config_path = _write_test_config(tmp)
    exp = Experiment.from_yaml(config_path)
    exp.run()
    return {
        "root": exp.paths.root,
        "config_path": config_path,
    }


# ---------------------------------------------------------------------------
# Test 1 — Config validation
# ---------------------------------------------------------------------------


def test_example_config_loads_cleanly() -> None:
    """load_experiment_config must validate the committed YAML without errors."""
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    # Replace relative paths so the validator can resolve them.
    raw["data"]["path"] = str(EXAMPLE_DATA)
    raw["experiment"]["output_dir"] = "/tmp/clustro_schema_check"
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
        tmp_cfg = Path(fh.name)
    cfg = load_experiment_config(tmp_cfg)
    assert cfg.clustering.methods
    assert "true_cluster" in cfg.data.target_columns
    assert "outcome_90d" in cfg.data.target_columns


def test_target_columns_not_in_schema() -> None:
    """true_cluster and outcome_90d must not appear in column_schema."""
    raw = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    schema = raw["data"]["column_schema"]
    all_schema_cols = (
        schema.get("continuous", [])
        + schema.get("binary", [])
        + schema.get("categorical", [])
        + schema.get("ordinal", [])
    )
    assert "true_cluster" not in all_schema_cols
    assert "outcome_90d" not in all_schema_cols


# ---------------------------------------------------------------------------
# Test 2 — Pipeline completes and core output files exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel_path",
    [
        "candidate_registry.parquet",
        "accepted_candidates.parquet",
        "consensus_labels.csv",
        "consensus_uncertainty.csv",
        "consensus_support.parquet",
        "consensus_cluster_summary.csv",
        "interpretation/cluster_profiles.csv",
        "interpretation/pairwise_cluster_contrasts.csv",
        "reports/candidate_metrics.csv",
        "manuscript_bundle/methods/auto_generated_methods.md",
    ],
)
def test_output_file_exists(experiment_outputs, rel_path: str) -> None:
    path = experiment_outputs["root"] / rel_path
    assert path.exists(), f"Expected output missing: {rel_path}"


def test_deprecated_soft_membership_alias_also_written(experiment_outputs) -> None:
    """Backward-compat alias must exist alongside the primary artifact."""
    alias = experiment_outputs["root"] / "consensus_soft_membership.parquet"
    assert alias.exists(), "Deprecated alias consensus_soft_membership.parquet not written"


# ---------------------------------------------------------------------------
# Test 3 — At least one candidate accepted
# ---------------------------------------------------------------------------


def test_at_least_one_candidate_accepted(experiment_outputs) -> None:
    registry = pd.read_parquet(experiment_outputs["root"] / "candidate_registry.parquet")
    n_accepted = int(registry["accepted"].sum())
    assert n_accepted >= 1, f"No candidates accepted (registry has {len(registry)} rows)"


# ---------------------------------------------------------------------------
# Test 4 — Consensus recovers exactly 3 clusters
# ---------------------------------------------------------------------------


def test_consensus_produces_three_clusters(experiment_outputs) -> None:
    consensus = pd.read_csv(experiment_outputs["root"] / "consensus_labels.csv")
    n_clusters = int(consensus["consensus_label"].nunique())
    assert n_clusters == 3, f"Expected 3 consensus clusters, got {n_clusters}"


# ---------------------------------------------------------------------------
# Test 5 — ARI vs ground truth >= 0.90
# ---------------------------------------------------------------------------


def test_ari_against_known_labels(experiment_outputs) -> None:
    consensus = pd.read_csv(experiment_outputs["root"] / "consensus_labels.csv")
    ground_truth = pd.read_csv(EXAMPLE_DATA)
    ari = adjusted_rand_score(ground_truth["true_cluster"], consensus["consensus_label"])
    assert ari >= 0.90, f"ARI {ari:.4f} is below the expected 0.90 threshold"


# ---------------------------------------------------------------------------
# Test 6 — Feature names do not contain target/leakage columns
# ---------------------------------------------------------------------------


def test_leakage_columns_absent_from_preprocessing(experiment_outputs) -> None:
    """Confirm true_cluster and outcome_90d were never used as model features."""
    import json
    feature_space_path = (
        experiment_outputs["root"] / "interpretation" / "interpretation_feature_space.json"
    )
    if not feature_space_path.exists():
        pytest.skip("interpretation_feature_space.json not written — skipping leakage check")
    # The JSON stores the preprocessing config, not feature names directly.
    # Check the cluster_profiles instead, which lists schema columns.
    _ = json.loads(feature_space_path.read_text(encoding="utf-8"))  # ensure parseable
    profiles = pd.read_csv(
        experiment_outputs["root"] / "interpretation" / "cluster_profiles.csv"
    )
    profiled_features = set(profiles["feature"].tolist())
    assert "true_cluster" not in profiled_features
    assert "outcome_90d" not in profiled_features
