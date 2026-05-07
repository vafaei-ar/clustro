"""Manuscript bundle export helpers."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from clustro.utils.io import ensure_directory, read_yaml, write_json


def create_manuscript_bundle(root: Path) -> Path:
    bundle = ensure_directory(root / "manuscript_bundle")
    ensure_directory(bundle / "figures")
    ensure_directory(bundle / "tables")
    ensure_directory(bundle / "supplementary")
    ensure_directory(bundle / "methods")
    return bundle


def populate_manuscript_bundle(root: Path) -> Path:
    bundle = create_manuscript_bundle(root)

    figure_exports = [
        "quality_vs_stability.png",
        "quality_vs_stability.csv",
        "search_flow.csv",
        "search_flow_diagram.png",
        "accepted_candidate_heatmap.csv",
        "accepted_candidate_metric_heatmap.png",
        "method_family_acceptance_summary.csv",
        "consensus_matrix_plot_data.parquet",
        "coassociation_matrix_heatmap.png",
        "cluster_size_confidence.csv",
        "uncertainty_distribution_by_cluster.png",
        "feature_importance.csv",
        "feature_importance_top.png",
        "clinical_profile_heatmap.csv",
        "clinical_profile_heatmap.png",
        "final_embedding_plot_data.csv",
        "final_umap_embedding_plot_data.csv",
        "final_embedding_scatter.png",
    ]
    table_exports = [
        "method_family_summary.csv",
        "runtime_summary.csv",
        "consensus_cluster_summary.csv",
        "surrogate_cv_metrics.csv",
        "surrogate_confusion_matrix.csv",
        "cluster_profiles.csv",
        "pairwise_cluster_contrasts.csv",
        "correlation_groups.csv",
        "grouped_permutation_importance.csv",
    ]
    supplementary_exports = [
        root / "candidate_registry.parquet",
        root / "accepted_candidates.parquet",
        root / "rejected_candidates.parquet",
        root / "consensus_labels.csv",
        root / "consensus_uncertainty.csv",
        root / "consensus_soft_membership.parquet",
        root / "consensus_bootstrap_stability.csv",
        root / "interpretation" / "interpretation_feature_space.json",
        root / "consensus" / "coassociation_matrix.parquet",
        root / "experiment_manifest.json",
        root / "reports" / "search_flow.json",
    ]

    for name in figure_exports:
        _copy_if_exists(root / "reports" / name, bundle / "figures" / name)
    for name in table_exports:
        source = root / name
        if name in {
            "surrogate_cv_metrics.csv",
            "surrogate_confusion_matrix.csv",
            "cluster_profiles.csv",
            "pairwise_cluster_contrasts.csv",
            "correlation_groups.csv",
            "grouped_permutation_importance.csv",
        }:
            source = root / "interpretation" / name
        _copy_if_exists(source, bundle / "tables" / name)
    for source in supplementary_exports:
        _copy_if_exists(source, bundle / "supplementary" / source.name)

    config_snapshot = root / "state" / "config_snapshot.yaml"
    _copy_if_exists(config_snapshot, bundle / "methods" / "config_snapshot.yaml")
    write_json(bundle / "methods" / "software_versions.json", _software_versions())
    (bundle / "methods" / "auto_generated_methods.md").write_text(
        _methods_text(root),
        encoding="utf-8",
    )
    return bundle


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    ensure_directory(destination.parent)
    shutil.copy2(source, destination)


def _software_versions() -> dict[str, object]:
    packages = [
        "clustro",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "xgboost",
        "shap",
        "torch",
        "umap-learn",
        "hdbscan",
    ]
    installed: dict[str, str | None] = {}
    for package in packages:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = None
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": installed,
    }


def _methods_text(root: Path) -> str:
    config_snapshot = root / "state" / "config_snapshot.yaml"
    manifest_path = root / "experiment_manifest.json"
    config = read_yaml(config_snapshot) if config_snapshot.exists() else {}
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )

    experiment = config.get("experiment", {})
    data = config.get("data", {})
    search = config.get("search", {})
    preprocessing = config.get("preprocessing", {})
    representation = config.get("representation", {})
    clustering = config.get("clustering", {})
    interpretation = config.get("interpretation", {})
    consensus = config.get("consensus", {})

    representation_names = (
        ", ".join(method["name"] for method in representation.get("methods", [])) or "none"
    )
    clustering_names = (
        ", ".join(method["name"] for method in clustering.get("methods", [])) or "none"
    )
    id_columns = []
    if data.get("id_column"):
        id_columns.append(str(data["id_column"]))
    id_columns.extend(str(column) for column in data.get("id_columns", []))
    id_columns = list(dict.fromkeys(id_columns))
    column_schema = data.get("column_schema", {})
    accelerator = manifest.get("accelerator", {})
    perturbation_count = search.get("perturbations_full", "unknown")
    perturbation_type = search.get("perturbation_type", "unknown")
    consensus_weighting = consensus.get("run_weighting", {}).get("source", "unknown")

    return "\n".join(
        [
            "# Auto-Generated Methods Summary",
            "",
            "## Experiment",
            f"- Name: `{experiment.get('name', 'unknown')}`",
            f"- Experiment ID: `{manifest.get('experiment_id', 'unknown')}`",
            f"- Dataset path: `{manifest.get('dataset_path', data.get('path', 'unknown'))}`",
            f"- Output directory: `{manifest.get('output_dir', str(root))}`",
            f"- Random seed: `{experiment.get('random_seed', 'unknown')}`",
            "",
            "## Data and Schema",
            f"- ID columns preserved in outputs: `{', '.join(id_columns) or 'index'}`",
            f"- Continuous features: `{len(column_schema.get('continuous', []))}`",
            f"- Binary features: `{len(column_schema.get('binary', []))}`",
            f"- Categorical features: `{len(column_schema.get('categorical', []))}`",
            f"- Ordinal features: `{len(column_schema.get('ordinal', []))}`",
            "",
            "## Search Design",
            f"- Pilot seeds: `{search.get('seeds_pilot', [])}`",
            f"- Full seeds: `{search.get('seeds_full', [])}`",
            f"- Perturbations: `{perturbation_count}` via `{perturbation_type}`",
            f"- Continuous transforms: `{preprocessing.get('continuous_transforms', [])}`",
            f"- Categorical encodings: `{preprocessing.get('categorical_encoding', [])}`",
            f"- Representation methods: `{representation_names}`",
            f"- Clustering methods: `{clustering_names}`",
            "",
            "## Consensus and Interpretation",
            f"- Consensus weighting source: `{consensus_weighting}`",
            f"- Consensus method: `{consensus.get('consensus_method', 'unknown')}`",
            f"- Final k strategy: `{consensus.get('final_k_strategy', 'unknown')}`",
            f"- Surrogate model: `{interpretation.get('surrogate_model', 'unknown')}`",
            f"- Interpretation feature space: `{interpretation.get('feature_space', 'unknown')}`",
            "- Interpretation continuous transform: "
            f"`{interpretation.get('continuous_transform', 'unknown')}`",
            "- Interpretation categorical encoding: "
            f"`{interpretation.get('categorical_encoding', 'unknown')}`",
            f"- SHAP enabled: `{interpretation.get('use_shap', False)}`",
            "- Permutation importance enabled: "
            f"`{interpretation.get('use_permutation_importance', False)}`",
            "",
            "## Compute Environment",
            f"- Requested accelerator use: `{accelerator.get('requested', False)}`",
            f"- Active device: `{accelerator.get('device', 'unknown')}`",
            f"- Torch available: `{accelerator.get('torch_available', False)}`",
            f"- CUDA available: `{accelerator.get('cuda_available', False)}`",
            f"- RAPIDS available: `{accelerator.get('rapids_available', False)}`",
            "",
        ]
    )
