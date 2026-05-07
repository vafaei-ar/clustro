from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from clustro import Experiment


def test_multiple_id_columns_are_preserved_in_outputs(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "encounter_id": [f"e{i}" for i in range(12)],
            "patient_id": [f"p{i // 2}" for i in range(12)],
            "x": [1, 1.1, 1.2, 1.0, 4, 4.1, 4.2, 4.0, 7, 7.1, 7.2, 7.0],
            "y": [2, 2.1, 2.0, 2.2, 5, 5.1, 5.0, 5.2, 8, 8.1, 8.0, 8.2],
            "flag": [0, 1] * 6,
            "site": ["a"] * 6 + ["b"] * 6,
        }
    )
    data_path = tmp_path / "ids.csv"
    frame.to_csv(data_path, index=False)

    config = {
        "experiment": {
            "name": "multi_id",
            "output_dir": str(tmp_path / "results"),
            "random_seed": 1,
        },
        "data": {
            "path": str(data_path),
            "id_column": "encounter_id",
            "id_columns": ["patient_id"],
            "column_schema": {
                "continuous": ["x", "y"],
                "binary": ["flag"],
                "categorical": ["site"],
                "ordinal": [],
            },
        },
        "search": {
            "pilot_sample_fraction": 0.5,
            "pilot_min_rows": 6,
            "seeds_pilot": [1],
            "seeds_full": [1],
            "perturbations_full": 1,
            "perturbation_type": "bootstrap",
            "optuna": {"enabled": False},
        },
        "representation": {"methods": [{"name": "none"}]},
        "clustering": {"methods": [{"name": "kmeans", "params": {"n_clusters": [2]}}]},
        "interpretation": {
            "surrogate_model": "random_forest",
            "cross_validation_folds": 2,
            "repeated_cv_repeats": 1,
            "use_shap": False,
            "use_permutation_importance": False,
        },
        "evaluation": {
            "acceptance": {
                "hard_thresholds": {
                    "silhouette_min": -1.0,
                    "ari_seed_min": -1.0,
                    "nmi_seed_min": -1.0,
                    "mean_cluster_jaccard_min": -1.0,
                }
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    Experiment.from_yaml(config_path).run()

    labels = pd.read_csv(next((tmp_path / "results" / "candidates").glob("*/final_labels.csv")))
    consensus = pd.read_csv(tmp_path / "results" / "consensus_labels.csv")

    assert set(["row_id", "encounter_id", "patient_id", "label"]).issubset(labels.columns)
    assert set(["row_id", "encounter_id", "patient_id", "consensus_label"]).issubset(
        consensus.columns
    )
