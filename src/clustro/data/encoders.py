"""Categorical and ordinal encoding builders."""

from __future__ import annotations

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def build_categorical_encoder(name: str):
    if name == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if name == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    raise ValueError(f"Unsupported categorical encoding: {name}")
