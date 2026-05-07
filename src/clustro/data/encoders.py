"""Categorical and ordinal encoding builders."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def build_categorical_encoder(name: str):
    if name == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if name == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError(f"Unsupported categorical encoding: {name}")


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
