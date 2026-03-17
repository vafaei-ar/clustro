"""Preprocessing pipeline assembly for tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from clustro.config.schema import ExperimentConfig
from clustro.data.encoders import build_categorical_encoder
from clustro.data.imputation import build_categorical_imputer, build_continuous_imputer
from clustro.data.schema import DatasetSchema
from clustro.data.transformations import build_continuous_transform


@dataclass(slots=True)
class PreprocessedData:
    evaluation_matrix: np.ndarray
    feature_names: list[str]
    schema: DatasetSchema
    row_ids: list[str]
    preprocessor: Pipeline


def build_preprocessor(
    config: ExperimentConfig,
    schema: DatasetSchema,
    *,
    continuous_transform: str | None = None,
    categorical_encoding: str | None = None,
) -> Pipeline:
    missingness = config.data.missingness
    transform_name = continuous_transform or config.preprocessing.continuous_transforms[0]
    categorical_encoding_name = categorical_encoding or config.preprocessing.categorical_encoding[0]

    transformers = []

    if schema.continuous:
        transformers.append(
            (
                "continuous",
                Pipeline(
                    steps=[
                        ("impute", build_continuous_imputer(missingness.continuous_imputer)),
                        ("transform", build_continuous_transform(transform_name)),
                    ]
                ),
                schema.continuous,
            )
        )

    if schema.binary:
        transformers.append(
            (
                "binary",
                Pipeline(steps=[("impute", build_categorical_imputer("most_frequent"))]),
                schema.binary,
            )
        )

    if schema.categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", build_categorical_imputer(missingness.categorical_imputer)),
                        ("encode", build_categorical_encoder(categorical_encoding_name)),
                    ]
                ),
                schema.categorical,
            )
        )

    if schema.ordinal:
        transformers.append(
            (
                "ordinal",
                Pipeline(
                    steps=[
                        ("impute", build_categorical_imputer(missingness.categorical_imputer)),
                        ("encode", build_categorical_encoder("ordinal")),
                    ]
                ),
                schema.ordinal,
            )
        )

    steps: list[tuple[str, object]] = [("columns", ColumnTransformer(transformers=transformers))]
    if config.preprocessing.variance_threshold.enabled:
        steps.append(("variance", VarianceThreshold(threshold=config.preprocessing.variance_threshold.threshold)))
    return Pipeline(steps=steps)


def preprocess_frame(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    *,
    continuous_transform: str | None = None,
    categorical_encoding: str | None = None,
) -> PreprocessedData:
    schema = DatasetSchema.from_config(config.data.column_schema)
    schema.validate_against(frame)
    row_ids = _row_ids(frame, config)
    model = build_preprocessor(
        config,
        schema,
        continuous_transform=continuous_transform,
        categorical_encoding=categorical_encoding,
    )
    matrix = model.fit_transform(frame[schema.all_columns()])
    feature_names = _get_feature_names(model, schema)
    return PreprocessedData(
        evaluation_matrix=np.asarray(matrix, dtype=float),
        feature_names=feature_names,
        schema=schema,
        row_ids=row_ids,
        preprocessor=model,
    )


def _row_ids(frame: pd.DataFrame, config: ExperimentConfig) -> list[str]:
    if config.data.id_column is None:
        return [str(index) for index in frame.index]
    return frame[config.data.id_column].astype(str).tolist()
def _get_feature_names(model: Pipeline, schema: DatasetSchema) -> list[str]:
    transformer = model.named_steps["columns"]
    names = transformer.get_feature_names_out(schema.all_columns())
    return [str(name) for name in names]
