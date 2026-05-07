from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from clustro import Experiment
from clustro import experiment as experiment_module


def test_use_ray_routes_non_optuna_candidates_through_ray_batch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(16)],
            "age": [50, 51, 52, 53, 70, 71, 72, 73, 60, 61, 62, 63, 49, 48, 74, 75],
            "bmi": [24, 25, 26, 27, 33, 34, 35, 36, 29, 30, 31, 32, 23, 22, 37, 38],
            "marker": [
                1.1,
                1.0,
                1.2,
                1.3,
                2.8,
                2.9,
                3.0,
                3.1,
                1.9,
                2.0,
                2.1,
                2.2,
                0.9,
                0.8,
                3.2,
                3.3,
            ],
            "sex_male": [0, 1] * 8,
            "site": ["north"] * 8 + ["south"] * 8,
        }
    )
    data_path = tmp_path / "ray.csv"
    frame.to_csv(data_path, index=False)

    config = {
        "experiment": {
            "name": "ray_run",
            "output_dir": str(tmp_path / "results"),
            "random_seed": 7,
            "use_ray": True,
            "n_jobs": 2,
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
                {"name": "kmeans", "params": {"n_clusters": [2]}},
                {"name": "agglomerative", "params": {"n_clusters": [2], "linkage": ["ward"]}},
            ]
        },
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

    monkeypatch.setattr(
        experiment_module, "maybe_init_ray", lambda enabled, n_jobs=None: bool(enabled)
    )
    ray_batches: list[int] = []

    def fake_ray_batch(candidates, matrix, config):
        ray_batches.append(len(candidates))
        return [
            experiment_module.evaluate_candidate(candidate, matrix, config)
            for candidate in candidates
        ]

    monkeypatch.setattr(experiment_module, "evaluate_candidate_batch_ray", fake_ray_batch)

    Experiment.from_yaml(config_path).run()

    assert ray_batches == [2]
    manifest = pd.read_json(tmp_path / "results" / "experiment_manifest.json", typ="series")
    assert manifest["orchestration"]["ray_enabled"] is True
