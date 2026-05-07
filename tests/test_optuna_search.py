from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from clustro import Experiment
from clustro.config.schema import ExperimentConfig
from clustro.search.optuna_objective import suggest_candidate_for_family


def test_optuna_enabled_runs_family_limited_trials(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(20)],
            "age": [50, 51, 52, 53, 54, 55, 56, 57, 70, 71, 72, 73, 74, 75, 76, 77, 60, 61, 62, 63],
            "bmi": [24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 29, 30, 31, 32],
            "marker": [
                1.0,
                1.1,
                1.2,
                1.1,
                1.0,
                1.2,
                1.3,
                1.1,
                3.0,
                3.1,
                3.2,
                3.0,
                3.3,
                3.1,
                3.2,
                3.4,
                2.0,
                2.1,
                2.2,
                2.1,
            ],
            "sex_male": [0, 1] * 10,
            "site": ["north"] * 10 + ["south"] * 10,
        }
    )
    data_path = tmp_path / "optuna.csv"
    frame.to_csv(data_path, index=False)

    config = {
        "experiment": {
            "name": "optuna_run",
            "output_dir": str(tmp_path / "results"),
            "random_seed": 3,
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
            "optuna": {
                "enabled": True,
                "sampler": "TPESampler",
                "pruner": "MedianPruner",
                "n_trials_per_family": 1,
            },
        },
        "representation": {"methods": [{"name": "none"}]},
        "clustering": {
            "methods": [
                {"name": "kmeans", "params": {"n_clusters": [2, 3]}},
                {"name": "agglomerative", "params": {"n_clusters": [2, 3], "linkage": ["ward"]}},
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

    Experiment.from_yaml(config_path).run()

    registry = pd.read_parquet(tmp_path / "results" / "candidate_registry.parquet")
    assert len(registry) == 2
    assert set(registry["family"]) == {"kmeans", "agglomerative"}
    assert (tmp_path / "results" / "optuna" / "kmeans_trials.csv").exists()
    assert (tmp_path / "results" / "optuna" / "agglomerative_trials.csv").exists()

    trials = pd.read_csv(tmp_path / "results" / "optuna" / "kmeans_trials.csv")
    assert "trial_params_json" in trials.columns
    assert "kmeans_n_clusters" in trials.loc[0, "trial_params_json"]


def test_optuna_suggests_hyperparameters_and_candidate_ids_change() -> None:
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "optuna", "output_dir": "./results"},
            "data": {
                "path": "./dummy.csv",
                "column_schema": {
                    "continuous": ["x"],
                    "binary": [],
                    "categorical": ["site"],
                    "ordinal": [],
                },
            },
            "preprocessing": {
                "continuous_transforms": ["standard"],
                "categorical_encoding": ["onehot", "ordinal"],
            },
            "representation": {"methods": [{"name": "none"}]},
            "clustering": {"methods": [{"name": "kmeans", "params": {"n_clusters": [2, 3]}}]},
        }
    )

    first = suggest_candidate_for_family(
        _FakeTrial(choice_index=0), "kmeans", config, {"dataset": "toy"}
    )
    second = suggest_candidate_for_family(
        _FakeTrial(choice_index=-1), "kmeans", config, {"dataset": "toy"}
    )

    assert first.clustering["params"]["n_clusters"] == 2
    assert second.clustering["params"]["n_clusters"] == 3
    assert first.preprocessing["categorical_encoding"] == "onehot"
    assert second.preprocessing["categorical_encoding"] == "ordinal"
    assert first.candidate_id != second.candidate_id


class _FakeTrial:
    def __init__(self, *, choice_index: int) -> None:
        self.choice_index = choice_index
        self.params: dict[str, object] = {}

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        value = choices[self.choice_index]
        self.params[name] = value
        return value
