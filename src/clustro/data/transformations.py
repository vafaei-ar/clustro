"""Continuous transformation builders."""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, RobustScaler, StandardScaler


def build_continuous_transform(name: str):
    if name == "none":
        return "passthrough"
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "power":
        return PowerTransformer()
    if name == "log1p_standard":
        return Pipeline(
            steps=[
                ("log1p", FunctionTransformer(func=_safe_log1p, validate=False)),
                ("scale", StandardScaler()),
            ]
        )
    raise ValueError(f"Unsupported continuous transform: {name}")


def _safe_log1p(values):
    if (values < 0).any():
        raise ValueError("log1p_standard requires non-negative continuous values.")
    import numpy as np

    return np.log1p(values)
