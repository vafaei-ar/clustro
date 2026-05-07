from __future__ import annotations

import numpy as np
import pandas as pd

from clustro.config.schema import InterpretationConfig
from clustro.data.schema import DatasetSchema
from clustro.interpretation.permutation import (
    build_correlation_groups,
    compute_grouped_permutation_importance,
    compute_permutation_importance,
)
from clustro.interpretation.profiling import build_pairwise_cluster_contrasts
from clustro.interpretation.surrogate import fit_surrogate_model


def test_surrogate_and_permutation_importance() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(60, 5))
    y = np.repeat([0, 1, 2], 20)
    x[y == 1, 0] += 2.5
    x[y == 2, 1] += 2.0
    feature_names = [f"f{i}" for i in range(x.shape[1])]

    result = fit_surrogate_model(
        x,
        y,
        feature_names,
        InterpretationConfig(
            surrogate_model="random_forest",
            cross_validation_folds=3,
            repeated_cv_repeats=1,
            use_shap=False,
            use_permutation_importance=True,
        ),
        random_seed=7,
    )

    assert set(["accuracy", "macro_f1", "balanced_accuracy"]).issubset(result.cv_metrics.columns)
    assert not result.confusion.empty
    importance = compute_permutation_importance(
        result.estimator, x, y, feature_names, random_seed=7
    )
    assert not importance.empty
    assert importance.iloc[0]["feature"] in feature_names
    groups = build_correlation_groups(x, feature_names, threshold=0.8)
    assert not groups.empty
    grouped_importance = compute_grouped_permutation_importance(
        result.estimator,
        x,
        y,
        feature_names,
        groups,
        random_seed=7,
    )
    assert not grouped_importance.empty
    assert set(["group_id", "features", "group_size"]).issubset(grouped_importance.columns)


def test_pairwise_cluster_contrasts_include_effect_sizes() -> None:
    frame = pd.DataFrame(
        {
            "age": [50.0, 52.0, 70.0, 72.0],
            "flag": [0, 1, 1, 1],
            "site": ["north", "north", "south", "south"],
        }
    )
    labels = pd.Series([0, 0, 1, 1])
    schema = DatasetSchema(continuous=["age"], binary=["flag"], categorical=["site"], ordinal=[])

    contrasts = build_pairwise_cluster_contrasts(frame, labels, schema)

    assert not contrasts.empty
    assert set(["cluster_left", "cluster_right", "feature", "contrast", "effect_size"]).issubset(
        contrasts.columns
    )
    assert "age" in set(contrasts["feature"])
