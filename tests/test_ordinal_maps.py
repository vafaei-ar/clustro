from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from clustro.config.schema import ExperimentConfig
from clustro.data.preprocess_pipeline import preprocess_frame


def _config(levels: list[object] | None) -> ExperimentConfig:
    data = {
        "path": "dataset.csv",
        "column_schema": {"continuous": [], "binary": [], "categorical": [], "ordinal": ["grade"]},
    }
    if levels is not None:
        data["ordinal_maps"] = {"grade": levels}
    return ExperimentConfig.model_validate(
        {
            "experiment": {"name": "demo", "output_dir": "out"},
            "data": data,
            "clustering": {"methods": [{"name": "kmeans", "params": {"n_clusters": [2]}}]},
        }
    )


def test_numeric_ordinal_order_preserves_declared_map() -> None:
    data = preprocess_frame(pd.DataFrame({"grade": [0, 1, 2, 3]}), _config([0, 1, 2, 3]))

    assert data.evaluation_matrix[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_string_ordinal_order_preserves_declared_map() -> None:
    data = preprocess_frame(
        pd.DataFrame({"grade": ["mild", "moderate", "severe"]}),
        _config(["mild", "moderate", "severe"]),
    )

    assert data.evaluation_matrix[:, 0].tolist() == [0.0, 1.0, 2.0]


def test_missing_ordinal_map_fails_validation() -> None:
    with pytest.raises(ValueError, match="Ordinal column 'grade' requires an explicit level order"):
        _config(None)


def test_numeric_ordinal_does_not_use_lexical_string_order() -> None:
    data = preprocess_frame(pd.DataFrame({"grade": [1, 2, 10]}), _config([1, 2, 10]))

    assert data.evaluation_matrix[:, 0].tolist() == [0.0, 1.0, 2.0]
    assert np.all(np.diff(data.evaluation_matrix[:, 0]) > 0)
