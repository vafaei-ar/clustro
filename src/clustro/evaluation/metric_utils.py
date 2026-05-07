"""Utility transforms for comparable candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: Literal["higher", "lower"]
    transform: Literal["identity", "log1p", "inverse_log1p", "bounded"]
    default: float
    clip_min: float | None = None
    clip_max: float | None = None


METRIC_SPECS: dict[str, MetricSpec] = {
    "silhouette": MetricSpec("silhouette", "higher", "bounded", 0.0, -1.0, 1.0),
    "davies_bouldin": MetricSpec("davies_bouldin", "lower", "inverse_log1p", np.nan, 0.0, None),
    "calinski_harabasz": MetricSpec("calinski_harabasz", "higher", "log1p", np.nan, 0.0, None),
    "ari_seed": MetricSpec("ari_seed", "higher", "bounded", 0.0, -1.0, 1.0),
    "nmi_seed": MetricSpec("nmi_seed", "higher", "identity", 0.0, 0.0, 1.0),
    "mean_cluster_jaccard": MetricSpec("mean_cluster_jaccard", "higher", "identity", 0.0, 0.0, 1.0),
    "cluster_balance": MetricSpec("cluster_balance", "higher", "identity", 0.0, 0.0, 1.0),
    "average_confidence": MetricSpec("average_confidence", "higher", "identity", np.nan, 0.0, 1.0),
    "assignment_entropy": MetricSpec(
        "assignment_entropy", "lower", "inverse_log1p", np.nan, 0.0, None
    ),
    "reconstruction_loss": MetricSpec(
        "reconstruction_loss", "lower", "inverse_log1p", np.nan, 0.0, None
    ),
    "runtime": MetricSpec("runtime_seconds", "lower", "inverse_log1p", np.nan, 0.0, None),
    "runtime_seconds": MetricSpec("runtime_seconds", "lower", "inverse_log1p", np.nan, 0.0, None),
    "parsimony": MetricSpec("parsimony_penalty", "lower", "inverse_log1p", np.nan, 0.0, None),
    "parsimony_penalty": MetricSpec(
        "parsimony_penalty", "lower", "inverse_log1p", np.nan, 0.0, None
    ),
}


def metric_to_utility(metric_name: str, value: float) -> float:
    spec = _spec_for(metric_name)
    if pd.isna(value):
        return float("nan")
    clipped = float(value)
    if spec.clip_min is not None:
        clipped = max(clipped, spec.clip_min)
    if spec.clip_max is not None:
        clipped = min(clipped, spec.clip_max)
    if spec.transform == "bounded":
        if spec.clip_min is None or spec.clip_max is None:
            return clipped
        return (clipped - spec.clip_min) / (spec.clip_max - spec.clip_min)
    if spec.transform == "log1p":
        return float(np.log1p(max(clipped, 0.0)))
    if spec.transform == "inverse_log1p":
        return float(1.0 / (1.0 + np.log1p(max(clipped, 0.0))))
    return clipped


def compute_metric_utilities(
    metrics: dict[str, float], weights: dict[str, float]
) -> dict[str, float]:
    utilities: dict[str, float] = {}
    for metric_name in weights:
        raw_name = raw_metric_name(metric_name)
        if raw_name not in metrics:
            raise KeyError(f"Metric '{raw_name}' required for weighted scoring is missing.")
        utilities[f"utility_{utility_metric_name(metric_name)}"] = metric_to_utility(
            metric_name, float(metrics[raw_name])
        )
    return utilities


def compute_utility_weighted_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    utilities = compute_metric_utilities(metrics, weights)
    score = 0.0
    for metric_name, weight in weights.items():
        utility = utilities[f"utility_{utility_metric_name(metric_name)}"]
        if pd.isna(utility):
            raise KeyError(f"Metric '{raw_metric_name(metric_name)}' produced missing utility.")
        score += float(weight) * float(utility)
    return float(score)


def add_utility_columns(frame: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    result = frame.copy()
    if result.empty:
        return result
    for metric_name in weights:
        raw_name = raw_metric_name(metric_name)
        utility_name = f"utility_{utility_metric_name(metric_name)}"
        if raw_name not in result:
            result[utility_name] = np.nan
            continue
        result[utility_name] = result[raw_name].apply(
            lambda value, mn=metric_name: (
                metric_to_utility(mn, float(value)) if pd.notna(value) else np.nan
            )
        )
        if metric_name == "calinski_harabasz":
            utility_series = result[utility_name]
            valid_mask = utility_series.notna()
            if int(valid_mask.sum()) > 1:
                tmp = pd.DataFrame({"u": utility_series.loc[valid_mask]})
                if "candidate_id" in result.columns:
                    tmp["candidate_id"] = result.loc[valid_mask, "candidate_id"]
                    tmp = tmp.sort_values(["u", "candidate_id"], kind="mergesort")
                else:
                    tmp = tmp.sort_values("u", kind="mergesort")
                tmp["pct"] = tmp["u"].rank(pct=True, method="average")
                result.loc[tmp.index, utility_name] = tmp["pct"]
            elif int(valid_mask.sum()) == 1:
                result.loc[valid_mask, utility_name] = 1.0

    score = pd.Series(0.0, index=result.index)
    missing_required = pd.Series(False, index=result.index)
    for metric_name, weight in weights.items():
        utility_name = f"utility_{utility_metric_name(metric_name)}"
        missing_required |= result[utility_name].isna()
        score += float(weight) * result[utility_name].fillna(0.0)
    result["final_weighted_score"] = score
    if weights:
        result.loc[missing_required, "final_weighted_score"] = np.nan
        result["missing_score_metrics"] = _missing_score_metrics(result, weights)
    return result


def raw_metric_name(metric_name: str) -> str:
    return _spec_for(metric_name).name


def utility_metric_name(metric_name: str) -> str:
    if metric_name == "runtime_seconds":
        return "runtime"
    if metric_name == "parsimony_penalty":
        return "parsimony"
    return metric_name


def _spec_for(metric_name: str) -> MetricSpec:
    if metric_name not in METRIC_SPECS:
        return MetricSpec(metric_name, "higher", "identity", np.nan)
    return METRIC_SPECS[metric_name]


def _missing_score_metrics(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    values: list[str] = []
    for _, row in frame.iterrows():
        missing = [
            raw_metric_name(metric_name)
            for metric_name in weights
            if pd.isna(row.get(f"utility_{utility_metric_name(metric_name)}"))
        ]
        values.append(";".join(missing))
    return pd.Series(values, index=frame.index)
