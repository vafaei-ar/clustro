"""SHAP utilities for fitted surrogate models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_shap_summary(
    estimator: object,
    matrix: np.ndarray,
    feature_names: list[str],
    *,
    max_rows: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP requested but not installed. Install clustro[deep].") from exc

    sample = matrix[: min(len(matrix), max_rows)]
    explainer = shap.Explainer(estimator, sample)
    explanation = explainer(sample)
    values = np.asarray(explanation.values)
    class_summary = pd.DataFrame()
    if values.ndim == 3:
        mean_abs = np.abs(values).mean(axis=(0, 2))
        flattened = np.abs(values).mean(axis=2)
        class_rows = []
        for class_index in range(values.shape[2]):
            class_mean = np.abs(values[:, :, class_index]).mean(axis=0)
            for feature_name, importance in zip(feature_names, class_mean, strict=True):
                class_rows.append(
                    {
                        "class_index": class_index,
                        "feature": feature_name,
                        "mean_abs_shap": float(importance),
                    }
                )
        class_summary = pd.DataFrame(class_rows).sort_values(
            ["class_index", "mean_abs_shap"],
            ascending=[True, False],
        )
    else:
        mean_abs = np.abs(values).mean(axis=0)
        flattened = values
    summary = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    detail = pd.DataFrame(flattened, columns=feature_names)
    return summary.reset_index(drop=True), detail, class_summary.reset_index(drop=True)
