from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from clustro import Experiment


def test_end_to_end_smoke_run(tmp_path: Path) -> None:
    frame = _make_dataset()
    data_path = tmp_path / "toy.csv"
    frame.to_csv(data_path, index=False)

    config = {
        "experiment": {
            "name": "toy_run",
            "output_dir": str(tmp_path / "results" / "toy_run"),
            "random_seed": 7,
            "use_ray": False,
            "use_mlflow": False,
        },
        "data": {
            "path": str(data_path),
            "id_column": "patient_id",
            "column_schema": {
                "continuous": ["age", "bmi", "marker"],
                "binary": ["sex_male"],
                "categorical": ["site"],
                "ordinal": [],
            },
        },
        "search": {
            "pilot_sample_fraction": 0.5,
            "pilot_min_rows": 10,
            "seeds_pilot": [1, 2],
            "seeds_full": [1, 2],
            "perturbations_full": 2,
            "perturbation_type": "bootstrap",
            "optuna": {"enabled": False},
        },
        "preprocessing": {
            "continuous_transforms": ["standard"],
            "categorical_encoding": ["onehot"],
            "variance_threshold": {"enabled": False, "threshold": 0.0},
        },
        "representation": {"methods": [{"name": "none"}]},
        "clustering": {
            "methods": [
                {"name": "kmeans", "params": {"n_clusters": [2, 3]}},
                {"name": "agglomerative", "params": {"n_clusters": [2], "linkage": ["ward"]}},
            ]
        },
        "evaluation": {
            "internal_metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
            "structure_constraints": {
                "min_clusters": 2,
                "max_clusters": 5,
                "min_cluster_fraction": 0.05,
                "max_noise_fraction": 0.5,
                "dominant_cluster_cap": 0.95,
            },
            "acceptance": {
                "hard_thresholds": {
                    "silhouette_min": -1.0,
                    "ari_seed_min": -1.0,
                    "nmi_seed_min": -1.0,
                    "mean_cluster_jaccard_min": -1.0,
                },
                "weighted_score": {
                    "silhouette": 0.2,
                    "davies_bouldin": -0.05,
                    "calinski_harabasz": 0.05,
                    "ari_seed": 0.3,
                    "nmi_seed": 0.2,
                    "mean_cluster_jaccard": 0.2,
                    "cluster_balance": 0.1,
                },
                "accept_top_fraction_if_above": 1.0,
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    experiment = Experiment.from_yaml(config_path)
    experiment.run()

    output_dir = tmp_path / "results" / "toy_run"
    assert (output_dir / "experiment_manifest.json").exists()
    assert (output_dir / "candidate_registry.parquet").exists()
    assert (output_dir / "accepted_candidates.parquet").exists()
    assert (output_dir / "consensus_labels.csv").exists()
    assert (output_dir / "consensus_uncertainty.csv").exists()
    assert (output_dir / "consensus_bootstrap_stability.csv").exists()
    assert (output_dir / "reports" / "candidate_metrics.csv").exists()
    assert (output_dir / "reports" / "search_flow.csv").exists()
    assert (output_dir / "reports" / "search_flow_diagram.png").exists()
    assert (output_dir / "reports" / "cluster_size_confidence.csv").exists()
    assert (output_dir / "reports" / "uncertainty_distribution_by_cluster.png").exists()
    assert (output_dir / "reports" / "clinical_profile_heatmap.csv").exists()
    assert (output_dir / "reports" / "clinical_profile_heatmap.png").exists()
    assert (output_dir / "reports" / "final_embedding_plot_data.csv").exists()
    assert (output_dir / "reports" / "final_embedding_scatter.png").exists()
    assert (output_dir / "manuscript_bundle" / "methods" / "auto_generated_methods.md").exists()
    assert (output_dir / "manuscript_bundle" / "methods" / "software_versions.json").exists()
    assert (output_dir / "manuscript_bundle" / "methods" / "config_snapshot.yaml").exists()
    assert (output_dir / "manuscript_bundle" / "tables" / "method_family_summary.csv").exists()
    assert (output_dir / "manuscript_bundle" / "tables" / "surrogate_confusion_matrix.csv").exists()
    assert (output_dir / "manuscript_bundle" / "tables" / "pairwise_cluster_contrasts.csv").exists()
    assert (output_dir / "manuscript_bundle" / "figures" / "final_embedding_scatter.png").exists()
    assert (
        output_dir / "manuscript_bundle" / "supplementary" / "candidate_registry.parquet"
    ).exists()


def _make_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    first = pd.DataFrame(
        {
            "patient_id": [f"a{i}" for i in range(20)],
            "age": rng.normal(55, 3, 20),
            "bmi": rng.normal(28, 2, 20),
            "marker": rng.normal(1.5, 0.2, 20),
            "sex_male": rng.integers(0, 2, 20),
            "site": ["north"] * 20,
        }
    )
    second = pd.DataFrame(
        {
            "patient_id": [f"b{i}" for i in range(20)],
            "age": rng.normal(72, 4, 20),
            "bmi": rng.normal(34, 2, 20),
            "marker": rng.normal(3.0, 0.3, 20),
            "sex_male": rng.integers(0, 2, 20),
            "site": ["south"] * 20,
        }
    )
    return pd.concat([first, second], ignore_index=True)
