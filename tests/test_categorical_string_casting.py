"""Tests for casting categoricals to string before sklearn encoders."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from clustro.config.schema import ExperimentConfig
from clustro.data.encoders import (
    CategoricalStringCaster,
    RareCategoryCollapser,
    build_categorical_encoder,
)
from clustro.data.imputation import build_categorical_imputer
from clustro.data.preprocess_pipeline import preprocess_frame


def _base_config(
    *,
    column_schema: dict,
    categorical_encoding: list[str],
    rare_enabled: bool = True,
) -> ExperimentConfig:
    data_config = {
        "path": "./dummy.csv",
        "column_schema": column_schema,
    }
    if column_schema.get("ordinal"):
        data_config["ordinal_maps"] = {"code": [0.0, 1.0, 2.0, 99.0]}
    return ExperimentConfig.model_validate(
        {
            "experiment": {"name": "cast_test", "output_dir": "./results"},
            "data": data_config,
            "preprocessing": {
                "continuous_transforms": ["none"],
                "categorical_encoding": categorical_encoding,
                "rare_category_collapse": {
                    "enabled": rare_enabled,
                    "min_frequency": 2,
                    "replacement": "__RARE__",
                },
            },
            "clustering": {"methods": [{"name": "kmeans"}]},
        }
    )


def test_numeric_categorical_rare_collapse_onehot_no_type_error() -> None:
    """Mixed float and __RARE__ after collapse must not break OneHotEncoder."""
    frame = pd.DataFrame(
        {
            "code": [0.0, 0.0, 1.0, 2.0, 99.0],
        }
    )
    config = _base_config(
        column_schema={
            "continuous": [],
            "binary": [],
            "categorical": ["code"],
            "ordinal": [],
        },
        categorical_encoding=["onehot"],
    )
    result = preprocess_frame(frame, config, categorical_encoding="onehot")

    assert result.evaluation_matrix.shape == (5, 2)
    assert not np.isnan(result.evaluation_matrix).any()


def test_onehot_feature_names_valid_after_string_cast() -> None:
    frame = pd.DataFrame({"code": [0.0, 0.0, 1.0, 2.0, 99.0]})
    config = _base_config(
        column_schema={
            "continuous": [],
            "binary": [],
            "categorical": ["code"],
            "ordinal": [],
        },
        categorical_encoding=["onehot"],
    )
    result = preprocess_frame(frame, config, categorical_encoding="onehot")

    assert result.feature_names
    for name in result.feature_names:
        assert isinstance(name, str)
        assert name.strip()
        assert "\x00" not in name
    assert any("__RARE__" in n or "RARE" in n for n in result.feature_names)
    assert any(re.search(r"0\.0|code.*0", n) for n in result.feature_names)


@pytest.mark.parametrize("encoding", ["onehot", "ordinal"])
def test_numeric_categorical_rare_collapse_onehot_and_ordinal(encoding: str) -> None:
    frame = pd.DataFrame({"code": [0.0, 0.0, 1.0, 2.0, 99.0]})
    if encoding == "ordinal":
        column_schema = {
            "continuous": [],
            "binary": [],
            "categorical": [],
            "ordinal": ["code"],
        }
    else:
        column_schema = {
            "continuous": [],
            "binary": [],
            "categorical": ["code"],
            "ordinal": [],
        }
    config = _base_config(
        column_schema=column_schema,
        categorical_encoding=["onehot", "ordinal"],
    )
    result = preprocess_frame(frame, config, categorical_encoding=encoding)

    assert result.evaluation_matrix.shape[0] == 5
    assert not np.isnan(result.evaluation_matrix).any()


def test_binary_floats_not_through_categorical_pipeline() -> None:
    """Numeric 0/1 in binary must not require categorical string casting."""
    frame = pd.DataFrame(
        {
            "flag": [0.0, 1.0, 0.0, 1.0, 0.0],
            "code": [0.0, 0.0, 1.0, 2.0, 99.0],
        }
    )
    config = _base_config(
        column_schema={
            "continuous": [],
            "binary": ["flag"],
            "categorical": ["code"],
            "ordinal": [],
        },
        categorical_encoding=["onehot"],
    )
    result = preprocess_frame(frame, config, categorical_encoding="onehot")

    assert result.evaluation_matrix.shape == (5, 3)
    flag_col = next(i for i, n in enumerate(result.feature_names) if "flag" in n)
    assert set(np.unique(result.evaluation_matrix[:, flag_col])).issubset({0.0, 1.0})


def test_pipeline_cast_then_onehot_matches_preprocess_behavior() -> None:
    """Isolated pipeline: impute -> rare collapse -> string cast -> encoder."""
    x = np.asarray([[0.0], [0.0], [1.0], [2.0], [99.0]], dtype=object)
    pipe = Pipeline(
        [
            ("impute", build_categorical_imputer("most_frequent")),
            ("collapse", RareCategoryCollapser(min_frequency=2, replacement="__RARE__")),
            ("cast_strings", CategoricalStringCaster()),
            ("encode", build_categorical_encoder("onehot")),
        ]
    )
    out = pipe.fit_transform(x)
    assert out.shape == (5, 2)
    assert not np.isnan(out.astype(float)).any()
