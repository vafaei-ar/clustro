from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from clustro import Experiment


def test_status_reports_completed_stages(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(12)],
            "age": [50, 51, 52, 53, 70, 71, 72, 73, 60, 61, 62, 63],
            "bmi": [24, 25, 26, 27, 33, 34, 35, 36, 29, 30, 31, 32],
            "marker": [1.1, 1.0, 1.2, 1.3, 2.8, 2.9, 3.0, 3.1, 1.9, 2.0, 2.1, 2.2],
            "sex_male": [0, 1] * 6,
            "site": ["north"] * 6 + ["south"] * 6,
        }
    )
    data_path = tmp_path / "status.csv"
    frame.to_csv(data_path, index=False)

    config = {
        "experiment": {"name": "status_run", "output_dir": str(tmp_path / "results"), "random_seed": 9},
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

    experiment = Experiment.from_yaml(config_path)
    experiment.run()
    status = Experiment.from_output_dir(tmp_path / "results").status()

    assert status["run"] is not None
    assert status["consensus"] is not None
    assert status["interpretation"] is not None
    assert status["report"] is not None
