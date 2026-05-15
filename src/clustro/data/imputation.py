"""Imputation builder helpers."""

from __future__ import annotations

from typing import Any

from sklearn.impute import KNNImputer, SimpleImputer


def build_continuous_imputer(missingness: Any, *, random_seed: int | None = None):
    """Build the configured continuous imputer.

    ``missingness`` may be the full missingness config or, for backwards-compatible
    internal use, the historical imputer-name string.
    """
    if isinstance(missingness, str):
        name = missingness
        knn_config = None
        iterative_config = None
    else:
        name = missingness.continuous_imputer
        knn_config = missingness.knn
        iterative_config = missingness.iterative

    if name == "median":
        return SimpleImputer(strategy="median")
    if name == "knn":
        kwargs = {}
        if knn_config is not None:
            kwargs = {
                "n_neighbors": knn_config.n_neighbors,
                "weights": knn_config.weights,
            }
        return KNNImputer(**kwargs)
    if name == "iterative":
        if iterative_config is None:
            raise ValueError("Iterative imputer configuration is required.")
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        imputer_random_state = (
            iterative_config.random_state
            if iterative_config.random_state is not None
            else random_seed
        )
        return IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=iterative_config.max_iter,
            initial_strategy=iterative_config.initial_strategy,
            sample_posterior=iterative_config.sample_posterior,
            random_state=imputer_random_state,
        )
    raise ValueError(f"Unsupported continuous imputer: {name}")


def build_categorical_imputer(name: str):
    if name == "most_frequent":
        return SimpleImputer(strategy="most_frequent")
    raise ValueError(f"Unsupported categorical imputer: {name}")
