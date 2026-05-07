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
            "accepted": [True, True, True],
            "rejection_reasons": ["", "", ""],
            "final_weighted_score": [0.9, 0.7, 0.1],
        }
    )

    filtered = apply_acceptance_policy(frame, config)

    assert filtered.loc[filtered["accepted"], "candidate_id"].tolist() == ["a", "b"]
    rejected = filtered.loc[~filtered["accepted"]].iloc[0]
    assert rejected["candidate_id"] == "c"
    assert "outside_top_fraction_policy" in rejected["rejection_reasons"]


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
            "accepted": [True, True],
            "rejection_reasons": ["", ""],
            "final_weighted_score": [0.9, 0.7],
        }
    )

    filtered = apply_acceptance_policy(frame, config)

    assert not filtered["accepted"].any()
    assert filtered["rejection_reasons"].str.contains("outside_top_fraction_policy").all()
