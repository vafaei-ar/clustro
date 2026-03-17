from __future__ import annotations

import numpy as np

from clustro.config.schema import InterpretationConfig
from clustro.interpretation.permutation import compute_permutation_importance
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
    importance = compute_permutation_importance(result.estimator, x, y, feature_names, random_seed=7)
    assert not importance.empty
    assert importance.iloc[0]["feature"] in feature_names
