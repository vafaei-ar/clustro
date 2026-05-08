"""Regression: acceptance policy must not shrink eligibility across Experiment reruns."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from clustro import Experiment


def test_experiment_rerun_same_output_dir_preserves_top_fraction_counts(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(16)],
            "age": [50, 51, 52, 53, 70, 71, 72, 73, 60, 61, 62, 63, 49, 48, 74, 75],
            "bmi": [24, 25, 26, 27, 33, 34, 35, 36, 29, 30, 31, 32, 23, 22, 37, 38],
            "marker": [1.1, 1.0, 1.2, 1.3, 2.8, 2.9, 3.0, 3.1, 1.9, 2.0, 2.1, 2.2]
            + [0.9, 0.8, 3.2, 3.3],
            "sex_male": [0, 1] * 8,
            "site": ["north"] * 8 + ["south"] * 8,
        }
    )
    data_path = tmp_path / "rerun_accept.csv"
    frame.to_csv(data_path, index=False)

    results_dir = tmp_path / "results_accept_rerun"
    config_path = tmp_path / "cfg_accept_rerun.yaml"
    config = {
        "experiment": {
            "name": "accept_rerun",
            "output_dir": str(results_dir),
            "random_seed": 21,
            "deterministic_mode": "strict",
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
                {"name": "kmeans", "params": {"n_clusters": [2, 3, 4]}},
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
                },
                "accept_top_fraction_if_above": 0.55,
                "weighted_score": {
                    "silhouette": 0.25,
                    "davies_bouldin": 0.05,
                    "calinski_harabasz": 0.05,
                    "ari_seed": 0.3,
                    "nmi_seed": 0.2,
                    "mean_cluster_jaccard": 0.2,
                    "cluster_balance": 0.1,
                },
            }
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    Experiment.from_yaml(config_path).run()
    reg_a = pd.read_parquet(results_dir / "candidate_registry.parquet")
    flow_path = results_dir / "reports" / "search_flow.json"
    flow_a = json.loads(flow_path.read_text(encoding="utf-8"))

    Experiment.from_yaml(config_path).run()
    reg_b = pd.read_parquet(results_dir / "candidate_registry.parquet")
    flow_b = json.loads(flow_path.read_text(encoding="utf-8"))

    assert len(reg_a) == len(reg_b)

    full_stage_a = reg_a.loc[reg_a["search_stage"] == "full_evaluated"].reset_index(drop=True)
    full_stage_b = reg_b.loc[reg_b["search_stage"] == "full_evaluated"].reset_index(drop=True)

    assert len(full_stage_a) == len(full_stage_b)

    before_a = int(
        full_stage_a["accepted_before_top_fraction"].fillna(False).astype(bool).sum()
    )
    before_b = int(
        full_stage_b["accepted_before_top_fraction"].fillna(False).astype(bool).sum()
    )
    assert before_a == before_b
    accepted_final_a = int(full_stage_a["accepted"].fillna(False).astype(bool).sum())
    accepted_final_b = int(full_stage_b["accepted"].fillna(False).astype(bool).sum())
    assert accepted_final_a == accepted_final_b
    assert accepted_final_a <= int(
        full_stage_a["accepted_before_top_fraction"].fillna(False).astype(bool).sum()
    )

    assert flow_a == flow_b
