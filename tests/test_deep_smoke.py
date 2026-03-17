from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

from clustro import Experiment


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_ae_kmeans_smoke_run(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(24)],
            "age": [50 + (i % 3) * 10 for i in range(24)],
            "bmi": [24 + (i % 4) for i in range(24)],
            "glucose": [90 + (i % 3) * 40 for i in range(24)],
            "sex_male": [i % 2 for i in range(24)],
            "site": ["north" if i < 12 else "south" for i in range(24)],
        }
    )
    data_path = tmp_path / "deep.csv"
    frame.to_csv(data_path, index=False)

    config = {
        "experiment": {
            "name": "deep_smoke",
            "output_dir": str(tmp_path / "results"),
            "random_seed": 5,
            "use_gpu_if_available": False,
        },
        "data": {
            "path": str(data_path),
            "id_column": "patient_id",
            "column_schema": {
                "continuous": ["age", "bmi", "glucose"],
                "binary": ["sex_male"],
                "categorical": ["site"],
                "ordinal": [],
            },
        },
        "search": {
            "pilot_sample_fraction": 0.5,
            "pilot_min_rows": 8,
            "seeds_pilot": [1],
            "seeds_full": [1],
            "perturbations_full": 1,
            "perturbation_type": "bootstrap",
            "optuna": {"enabled": False},
        },
        "representation": {"methods": [{"name": "none"}]},
        "clustering": {
            "methods": [
                {
                    "name": "ae_kmeans",
                    "params": {
                        "n_clusters": [2],
                        "latent_dim": [3],
                        "hidden_layers": [[8, 4]],
                        "epochs": 3,
                        "batch_size": 8,
                        "learning_rate": [0.001],
                        "early_stopping_patience": 2,
                    },
                }
            ]
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
        "interpretation": {
            "surrogate_model": "random_forest",
            "cross_validation_folds": 2,
            "repeated_cv_repeats": 1,
            "use_shap": False,
            "use_permutation_importance": True,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    Experiment.from_yaml(config_path).run()

    assert (tmp_path / "results" / "candidate_registry.parquet").exists()
    assert (tmp_path / "results" / "interpretation" / "surrogate_cv_metrics.csv").exists()
