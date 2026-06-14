from __future__ import annotations

import pandas as pd
import pytest

from clustro.evaluation.metric_utils import (
    add_utility_columns,
    compute_utility_weighted_score,
    metric_to_utility,
)


def test_higher_davies_bouldin_has_lower_utility() -> None:
    assert metric_to_utility("davies_bouldin", 0.5) > metric_to_utility("davies_bouldin", 2.0)


def test_calinski_harabasz_utility_is_candidate_intrinsic() -> None:
    # CH utility must NOT depend on which other candidates are present.
    frame_two = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "accepted": [True, True],
            "calinski_harabasz": [10.0, 1_000_000.0],
        }
    )
    frame_one = pd.DataFrame(
        {
            "candidate_id": ["a"],
            "accepted": [True],
            "calinski_harabasz": [10.0],
        }
    )

    scored_two = add_utility_columns(frame_two, {"calinski_harabasz": 1.0})
    scored_one = add_utility_columns(frame_one, {"calinski_harabasz": 1.0})

    col = "utility_calinski_harabasz"
    utility_a_in_two = scored_two.loc[scored_two["candidate_id"] == "a", col].item()
    utility_a_alone = scored_one.loc[scored_one["candidate_id"] == "a", col].item()

    # Utility of candidate "a" must be the same whether "b" is present or not.
    assert abs(utility_a_in_two - utility_a_alone) < 1e-9
    # Higher CH gets higher utility.
    utility_b = scored_two.loc[scored_two["candidate_id"] == "b", col].item()
    assert utility_b > utility_a_in_two


def test_worse_davies_bouldin_scores_lower_when_other_metrics_equal() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["good", "bad"],
            "accepted": [True, True],
            "silhouette": [0.4, 0.4],
            "davies_bouldin": [0.5, 2.0],
        }
    )

    scored = add_utility_columns(frame, {"silhouette": 0.5, "davies_bouldin": 0.5})

    good = scored.loc[scored["candidate_id"] == "good", "final_weighted_score"].item()
    bad = scored.loc[scored["candidate_id"] == "bad", "final_weighted_score"].item()
    assert good > bad


def test_runtime_penalty_lowers_utility_score() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["fast", "slow"],
            "accepted": [True, True],
            "runtime_seconds": [1.0, 100.0],
        }
    )

    scored = add_utility_columns(frame, {"runtime": 1.0})

    fast = scored.loc[scored["candidate_id"] == "fast", "final_weighted_score"].item()
    slow = scored.loc[scored["candidate_id"] == "slow", "final_weighted_score"].item()
    assert fast > slow


def test_missing_weighted_metric_is_not_silently_zeroed() -> None:
    with pytest.raises(KeyError, match="required for weighted scoring is missing"):
        compute_utility_weighted_score({"silhouette": 0.5}, {"mean_cluster_jaccard": 1.0})

    frame = add_utility_columns(
        pd.DataFrame({"candidate_id": ["a"], "accepted": [True]}),
        {"mean_cluster_jaccard": 1.0},
    )
    assert pd.isna(frame.loc[0, "final_weighted_score"])
    assert frame.loc[0, "missing_score_metrics"] == "mean_cluster_jaccard"
