"""Imputation builder helpers."""

from __future__ import annotations

from sklearn.impute import KNNImputer, SimpleImputer


def build_continuous_imputer(name: str):
    if name == "median":
        return SimpleImputer(strategy="median")
    if name == "knn":
        return KNNImputer()
    raise ValueError(f"Unsupported continuous imputer: {name}")


def build_categorical_imputer(name: str):
    if name == "most_frequent":
        return SimpleImputer(strategy="most_frequent")
    raise ValueError(f"Unsupported categorical imputer: {name}")
