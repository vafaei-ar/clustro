from __future__ import annotations

import pandas as pd

from clustro.config.schema import ExperimentConfig
from clustro.data.preprocess_pipeline import preprocess_frame


def test_rare_category_collapse_groups_infrequent_levels() -> None:
    frame = pd.DataFrame(
        {
            "id": [f"p{i}" for i in range(10)],
            "x": [float(i) for i in range(10)],
            "site": ["common"] * 7 + ["rare_a", "rare_b", "rare_c"],
        }
    )
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "rare", "output_dir": "./results"},
            "data": {
                "path": "./dummy.csv",
                "id_column": "id",
                "column_schema": {
                    "continuous": ["x"],
                    "binary": [],
                    "categorical": ["site"],
                    "ordinal": [],
                },
            },
            "preprocessing": {
                "continuous_transforms": ["none"],
                "categorical_encoding": ["onehot"],
                "rare_category_collapse": {
                    "enabled": True,
                    "min_frequency": 3,
                    "replacement": "__RARE__",
                },
            },
            "clustering": {"methods": [{"name": "kmeans"}]},
        }
    )

    result = preprocess_frame(frame, config)

    assert any("__RARE__" in name for name in result.feature_names)
    assert any("common" in name for name in result.feature_names)
    assert not any("rare_a" in name for name in result.feature_names)
    assert result.evaluation_matrix.shape[0] == len(frame)
