from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from clustro.config.schema import ExperimentConfig
from clustro.data.preprocess_pipeline import preprocess_frame
from clustro.search.scheduler import _collect_perturbations
from clustro.search.search_space import Candidate


def _config(
    *,
    continuous_imputer: str = "iterative",
    iterative_random_state: int | None = 123,
    perturbations_full: int = 0,
    stability_mode: str = "processed_matrix",
) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"name": "iterative_test", "output_dir": "out", "random_seed": 2026},
            "data": {
                "path": "dataset.csv",
                "column_schema": {
                    "continuous": ["x1", "x2", "x3"],
                    "binary": [],
                    "categorical": [],
                    "ordinal": [],
                },
                "missingness": {
                    "continuous_imputer": continuous_imputer,
                    "add_missing_indicators": False,
                    "iterative": {
                        "max_iter": 3,
                        "initial_strategy": "median",
                        "sample_posterior": True,
                        "random_state": iterative_random_state,
                        "estimator": "bayesian_ridge",
                    },
                },
            },
            "preprocessing": {"continuous_transforms": ["none"]},
            "search": {
                "perturbations_full": perturbations_full,
                "perturbation_type": "subsample",
                "stability_mode": stability_mode,
            },
            "clustering": {"methods": [{"name": "kmeans", "params": {"n_clusters": [2]}}]},
        }
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "x2": [1.5, np.nan, 3.5, 4.5, 5.5, 6.5],
            "x3": [10.0, 11.0, 12.0, np.nan, 14.0, 15.0],
        }
    )


def _candidate() -> Candidate:
    return Candidate(
        candidate_id="iterative-candidate",
        preprocessing={"continuous_transform": "none", "categorical_encoding": "onehot"},
        representation={"name": "none", "params": {}},
        clustering={"name": "kmeans", "params": {"n_clusters": 2}},
        family="kmeans",
    )


def test_iterative_imputer_fits_transforms_and_leaves_no_nan() -> None:
    result = preprocess_frame(_frame(), _config())

    assert result.evaluation_matrix.shape == (6, 3)
    assert np.issubdtype(result.evaluation_matrix.dtype, np.floating)
    assert not np.isnan(result.evaluation_matrix).any()


def test_iterative_imputer_is_deterministic_with_fixed_random_state() -> None:
    config = _config(iterative_random_state=42)

    first = preprocess_frame(_frame(), config).evaluation_matrix
    second = preprocess_frame(_frame(), config).evaluation_matrix

    np.testing.assert_allclose(first, second)


def test_full_pipeline_perturbations_refit_configured_iterative_imputer(monkeypatch) -> None:
    calls: list[str] = []

    def fake_build_continuous_imputer(missingness, *, random_seed=None):
        calls.append(missingness.continuous_imputer)
        return SimpleImputer(strategy="median")

    monkeypatch.setattr(
        "clustro.data.preprocess_pipeline.build_continuous_imputer",
        fake_build_continuous_imputer,
    )
    config = _config(perturbations_full=2, stability_mode="full_pipeline")
    matrix = np.nan_to_num(_frame().to_numpy(dtype=float), nan=0.0)

    _collect_perturbations(_candidate(), matrix, config, raw_frame=_frame())

    assert calls == ["iterative", "iterative"]


def test_median_continuous_imputer_config_remains_backward_compatible() -> None:
    result = preprocess_frame(_frame(), _config(continuous_imputer="median"))

    assert result.evaluation_matrix.shape == (6, 3)
    assert not np.isnan(result.evaluation_matrix).any()
