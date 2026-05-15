from __future__ import annotations

import numpy as np
import pandas as pd

from clustro.config.schema import ExperimentConfig
from clustro.data.preprocess_pipeline import preprocess_frame


def _config(
    schema: dict[str, list[str]], *, add: bool = True, encoding: str = "onehot"
) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"name": "demo", "output_dir": "out"},
            "data": {
                "path": "dataset.csv",
                "column_schema": schema,
                "missingness": {"add_missing_indicators": add},
            },
            "preprocessing": {"categorical_encoding": [encoding]},
            "clustering": {"methods": [{"name": "kmeans", "params": {"n_clusters": [2]}}]},
        }
    )


def test_continuous_missingness_indicator_adds_feature() -> None:
    frame = pd.DataFrame({"albumin": [1.0, np.nan, 3.0], "x": [0.0, 1.0, 2.0]})
    schema = {"continuous": ["albumin", "x"], "binary": [], "categorical": [], "ordinal": []}

    with_indicators = preprocess_frame(frame, _config(schema, add=True))
    without_indicators = preprocess_frame(frame, _config(schema, add=False))

    assert "continuous__albumin__missing" in with_indicators.feature_names
    assert (
        with_indicators.evaluation_matrix.shape[1] > without_indicators.evaluation_matrix.shape[1]
    )


def test_categorical_missingness_indicator_with_onehot_is_numeric() -> None:
    frame = pd.DataFrame({"race": [1, np.nan, 2, 1]})
    schema = {"continuous": [], "binary": [], "categorical": ["race"], "ordinal": []}

    data = preprocess_frame(frame, _config(schema, add=True, encoding="onehot"))

    assert "categorical__race__missing" in data.feature_names
    assert np.issubdtype(data.evaluation_matrix.dtype, np.floating)


def test_binary_missingness_indicator_exists() -> None:
    frame = pd.DataFrame({"flag": [1, np.nan, 0, 1]})
    schema = {"continuous": [], "binary": ["flag"], "categorical": [], "ordinal": []}

    data = preprocess_frame(frame, _config(schema, add=True))

    assert "binary__flag__missing" in data.feature_names


def test_missingness_indicators_can_be_disabled() -> None:
    frame = pd.DataFrame({"albumin": [1.0, np.nan, 3.0]})
    schema = {"continuous": ["albumin"], "binary": [], "categorical": [], "ordinal": []}

    data = preprocess_frame(frame, _config(schema, add=False))

    assert not any(name.endswith("__missing") for name in data.feature_names)
