from __future__ import annotations

import pandas as pd

from clustro.config.schema import ExperimentConfig
from clustro.evaluation.acceptance import apply_acceptance_policy


def test_acceptance_policy_keeps_only_top_fraction_of_hard_pass_candidates() -> None:
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "acceptance", "output_dir": "./results"},
            "data": {
                "path": "./dummy.csv",
                "column_schema": {
                    "continuous": ["x"],
                    "binary": [],
                    "categorical": [],
                    "ordinal": [],
                },
            },
            "clustering": {"methods": [{"name": "kmeans"}]},
            "evaluation": {
                "acceptance": {
                    "accept_top_fraction_if_above": 0.5,
                    "weighted_score": {"silhouette": 1.0},
                }
            },
        }
    )
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "n_clusters": [3, 3, 3],
            "dominant_cluster_fraction": [0.34, 0.34, 0.34],
            "noise_fraction": [0.0, 0.0, 0.0],
            "min_cluster_fraction": [0.1, 0.1, 0.1],
            "silhouette": [0.9, 0.7, 0.1],
        }
    )

    filtered = apply_acceptance_policy(frame, config)

    assert filtered.loc[filtered["accepted"], "candidate_id"].tolist() == ["a", "b"]
    rejected = filtered.loc[~filtered["accepted"]].iloc[0]
    assert rejected["candidate_id"] == "c"
    assert rejected["hard_rejection_reasons"] == ""
    assert "outside_top_fraction_policy" in rejected["final_rejection_reasons"]
    assert rejected["rejection_reasons"] == rejected["final_rejection_reasons"]


def test_acceptance_policy_zero_fraction_keeps_no_candidates() -> None:
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "acceptance", "output_dir": "./results"},
            "data": {
                "path": "./dummy.csv",
                "column_schema": {
                    "continuous": ["x"],
                    "binary": [],
                    "categorical": [],
                    "ordinal": [],
                },
            },
            "clustering": {"methods": [{"name": "kmeans"}]},
            "evaluation": {
                "acceptance": {
                    "accept_top_fraction_if_above": 0.0,
                    "weighted_score": {"silhouette": 1.0},
                }
            },
        }
    )
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "n_clusters": [3, 3],
            "dominant_cluster_fraction": [0.34, 0.34],
            "noise_fraction": [0.0, 0.0],
            "min_cluster_fraction": [0.1, 0.1],
            "silhouette": [0.9, 0.7],
        }
    )

    filtered = apply_acceptance_policy(frame, config)

    assert not filtered["accepted"].any()
    assert filtered["hard_filter_passed"].all()
    assert filtered["final_rejection_reasons"].str.contains("outside_top_fraction_policy").all()
    assert filtered["rejection_reasons"].equals(filtered["final_rejection_reasons"])


def test_acceptance_policy_persisted_hard_gate_not_accepted_column() -> None:
    """Persisted hard pass still yields top-fraction selection despite accepted=False."""
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "acceptance", "output_dir": "./results"},
            "data": {
                "path": "./dummy.csv",
                "column_schema": {
                    "continuous": ["x"],
                    "binary": [],
                    "categorical": [],
                    "ordinal": [],
                },
            },
            "clustering": {"methods": [{"name": "kmeans"}]},
            "evaluation": {
                "acceptance": {
                    "accept_top_fraction_if_above": 0.5,
                    "weighted_score": {"silhouette": 1.0},
                }
            },
        }
    )
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "accepted": [False, False, False],
            "hard_filter_passed": [True, True, True],
            "hard_rejection_reasons": ["", "", ""],
            "n_clusters": [3, 3, 3],
            "dominant_cluster_fraction": [0.34, 0.34, 0.34],
            "noise_fraction": [0.0, 0.0, 0.0],
            "min_cluster_fraction": [0.1, 0.1, 0.1],
            "silhouette": [0.9, 0.7, 0.1],
        }
    )

    filtered = apply_acceptance_policy(frame, config)

    assert filtered.loc[filtered["accepted"], "candidate_id"].tolist() == ["a", "b"]
    rejected = filtered.loc[filtered["candidate_id"].eq("c")].iloc[0]
    assert rejected["hard_rejection_reasons"] == ""
    assert "outside_top_fraction_policy" in str(rejected["final_rejection_reasons"])
