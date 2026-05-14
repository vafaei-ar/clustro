"""Categorical and ordinal encoding builders."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def build_categorical_encoder(name: str):
    if name == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if name == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError(f"Unsupported categorical encoding: {name}")


def build_explicit_ordinal_encoder(columns: list[str], ordinal_maps: dict[str, list[Any]]):
    categories = [list(ordinal_maps[column]) for column in columns]
    return OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )


class MissingIndicatorAppender(BaseEstimator, TransformerMixin):
    """Append fit-time missingness indicators after a block transformer.

    Indicators are added only for source columns with missing values observed at fit time,
    matching ``SimpleImputer(add_indicator=True)`` semantics while preserving explicit and
    readable feature names after arbitrary block-level transforms.
    """

    def __init__(self, transformer, block_name: str):
        self.transformer = transformer
        self.block_name = block_name

    def fit(self, X, y=None):  # noqa: N803
        frame = self._as_frame(X)
        self.input_features_ = [str(column) for column in frame.columns]
        missing_mask = frame.isna().to_numpy()
        self.indicator_indices_ = np.flatnonzero(missing_mask.any(axis=0)).astype(int)
        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(X, y)
        return self

    def transform(self, X):  # noqa: N803
        transformed = np.asarray(self.transformer_.transform(X), dtype=float)
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)
        if len(self.indicator_indices_) == 0:
            return transformed
        frame = self._as_frame(X)
        indicators = frame.isna().to_numpy(dtype=float)[:, self.indicator_indices_]
        return np.hstack([transformed, indicators])

    def fit_transform(self, X, y=None, **fit_params):  # noqa: N803
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = getattr(self, "input_features_", None)
        input_features = [str(feature) for feature in input_features]
        try:
            base_names = self.transformer_.get_feature_names_out(input_features)
        except AttributeError:
            base_names = np.asarray(input_features, dtype=object)
        indicator_names = [f"{input_features[index]}__missing" for index in self.indicator_indices_]
        return np.asarray([*map(str, base_names), *indicator_names], dtype=object)

    def _as_frame(self, X) -> pd.DataFrame:  # noqa: N803
        if isinstance(X, pd.DataFrame):
            return X
        columns = getattr(self, "input_features_", None)
        if columns is None:
            arr = np.asarray(X, dtype=object)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            columns = [f"x{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=columns)
        return pd.DataFrame(X, columns=columns)


class RareCategoryCollapser(BaseEstimator, TransformerMixin):
    """Collapse infrequent categorical levels to a shared replacement token."""

    def __init__(self, min_frequency: int | float = 0.01, replacement: str = "__RARE__"):
        self.min_frequency = min_frequency
        self.replacement = replacement

    def fit(self, X, y=None):  # noqa: N803
        values = np.asarray(X, dtype=object)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        threshold = self._threshold(values.shape[0])
        self.frequent_categories_: list[set[object]] = []
        for column_index in range(values.shape[1]):
            column = values[:, column_index]
            unique, counts = np.unique(column, return_counts=True)
            self.frequent_categories_.append(
                {
                    category
                    for category, count in zip(unique, counts, strict=True)
                    if count >= threshold
                }
            )
        return self

    def transform(self, X):  # noqa: N803
        values = np.asarray(X, dtype=object).copy()
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        for column_index, frequent in enumerate(self.frequent_categories_):
            mask = np.array([value not in frequent for value in values[:, column_index]])
            values[mask, column_index] = self.replacement
        return values

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(
                [f"x{i}" for i in range(len(self.frequent_categories_))], dtype=object
            )
        return np.asarray(input_features, dtype=object)

    def _threshold(self, n_rows: int) -> int:
        if isinstance(self.min_frequency, float) and self.min_frequency <= 1.0:
            return max(1, int(np.ceil(n_rows * self.min_frequency)))
        return max(1, int(self.min_frequency))


class CategoricalStringCaster(BaseEstimator, TransformerMixin):
    """Cast categorical cell values to str so sklearn encoders see a uniform dtype.

    After imputation and rare-category collapse, columns may mix numeric codes with
    string replacement tokens; OneHotEncoder and OrdinalEncoder reject mixed types.
    """

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        out = np.empty(arr.shape, dtype=object)
        for index in np.ndindex(arr.shape):
            value = arr[index]
            if pd.isna(value):
                out[index] = "__MISSING__"
            else:
                out[index] = str(value)
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            msg = "CategoricalStringCaster requires input_features for get_feature_names_out"
            raise ValueError(msg)
        return np.asarray(input_features, dtype=object)
