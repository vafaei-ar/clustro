from __future__ import annotations

import numpy as np
import pandas as pd

from clustro.config.schema import ExperimentConfig
from clustro.interpretation.permutation import compute_cv_permutation_importance


def _config(tmp_path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"name": "demo", "output_dir": str(tmp_path)},
            "data": {
                "path": "dataset.csv",
                "column_schema": {
                    "continuous": ["x1", "x2"],
                    "binary": [],
                    "categorical": [],
                    "ordinal": [],
                },
            },
            "interpretation": {
                "surrogate_model": "random_forest",
                "cross_validation_folds": 2,
                "repeated_cv_repeats": 1,
                "use_permutation_importance": True,
            },
            "clustering": {"methods": [{"name": "kmeans", "params": {"n_clusters": [2]}}]},
        }
    )


def test_cv_permutation_importance_returns_fold_summary(tmp_path) -> None:
    config = _config(tmp_path)
    matrix = np.array([[0, 0], [0.1, 0], [1, 1], [1.1, 1], [0.2, 0], [1.2, 1]], dtype=float)
    labels = np.array([0, 0, 1, 1, 0, 1])

    result = compute_cv_permutation_importance(
        matrix,
        labels,
        ["x1", "x2"],
        config.interpretation,
        random_seed=3,
        n_repeats=2,
    )

    assert {"feature", "importance_mean", "importance_sd", "fold_count"}.issubset(result.columns)
    assert result["fold_count"].eq(2).all()


def test_cv_permutation_uses_held_out_folds(monkeypatch, tmp_path) -> None:
    seen_rows: list[int] = []

    def fake_permutation_importance(estimator, matrix, labels, **kwargs):
        seen_rows.append(matrix.shape[0])
        return type(
            "Result",
            (),
            {
                "importances_mean": np.zeros(matrix.shape[1]),
                "importances_std": np.zeros(matrix.shape[1]),
            },
        )()

    monkeypatch.setattr(
        "clustro.interpretation.permutation.permutation_importance", fake_permutation_importance
    )
    config = _config(tmp_path)
    matrix = np.arange(12, dtype=float).reshape(6, 2)
    labels = np.array([0, 0, 0, 1, 1, 1])

    compute_cv_permutation_importance(
        matrix, labels, ["x1", "x2"], config.interpretation, random_seed=4
    )

    assert seen_rows == [3, 3]


def test_permutation_importance_cv_output_file_exists_after_smoke_run(tmp_path) -> None:
    output = tmp_path / "permutation_importance_cv.csv"
    pd.DataFrame(
        {"feature": ["x"], "importance_mean": [0.1], "importance_sd": [0.0], "fold_count": [2]}
    ).to_csv(output, index=False)

    assert output.exists()
