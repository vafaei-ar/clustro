"""Permutation importance helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def compute_permutation_importance(
    estimator: object,
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    *,
    random_seed: int,
) -> pd.DataFrame:
    result = permutation_importance(
        estimator,
        matrix,
        labels,
        n_repeats=10,
        random_state=random_seed,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)
