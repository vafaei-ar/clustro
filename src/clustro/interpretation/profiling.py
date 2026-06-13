"""Cluster profiling summaries."""

from __future__ import annotations

import math

import pandas as pd

from clustro.data.schema import DatasetSchema


def _cohens_h(p1: float, p2: float) -> float:
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


def build_cluster_profiles(
    frame: pd.DataFrame, labels: pd.Series, schema: DatasetSchema
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["consensus_label"] = labels.to_numpy()
    rows: list[dict[str, object]] = []

    for cluster_label, cluster_frame in enriched.groupby("consensus_label"):
        for column in schema.continuous:
            rows.append(
                {
                    "cluster": int(cluster_label),
                    "feature": column,
                    "feature_type": "continuous",
                    "summary": "mean",
                    "value": float(cluster_frame[column].mean()),
                }
            )
        for column in schema.binary:
            rows.append(
                {
                    "cluster": int(cluster_label),
                    "feature": column,
                    "feature_type": "binary",
                    "summary": "prevalence",
                    "value": float(cluster_frame[column].mean()),
                }
            )
        for column in schema.categorical + schema.ordinal:
            top_value = cluster_frame[column].astype(str).mode().iloc[0]
            rows.append(
                {
                    "cluster": int(cluster_label),
                    "feature": column,
                    "feature_type": "categorical",
                    "summary": "mode",
                    "value": top_value,
                }
            )
    return pd.DataFrame(rows)


def build_pairwise_cluster_contrasts(
    frame: pd.DataFrame, labels: pd.Series, schema: DatasetSchema
) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["consensus_label"] = labels.to_numpy()
    cluster_ids = sorted(int(cluster) for cluster in enriched["consensus_label"].unique())
    rows: list[dict[str, object]] = []

    for index, left_cluster in enumerate(cluster_ids):
        left = enriched.loc[enriched["consensus_label"] == left_cluster]
        for right_cluster in cluster_ids[index + 1 :]:
            right = enriched.loc[enriched["consensus_label"] == right_cluster]

            for column in schema.continuous:
                left_values = left[column].astype(float)
                right_values = right[column].astype(float)
                rows.append(
                    {
                        "cluster_left": left_cluster,
                        "cluster_right": right_cluster,
                        "feature": column,
                        "feature_type": "continuous",
                        "contrast": "mean_difference",
                        "value": float(left_values.mean() - right_values.mean()),
                        "effect_size": _cohens_d(left_values, right_values),
                    }
                )

            for column in schema.binary:
                left_values = left[column].astype(float)
                right_values = right[column].astype(float)
                p1 = float(left_values.mean())
                p2 = float(right_values.mean())
                rows.append(
                    {
                        "cluster_left": left_cluster,
                        "cluster_right": right_cluster,
                        "feature": column,
                        "feature_type": "binary",
                        "contrast": "prevalence_difference",
                        "value": p1 - p2,
                        "effect_size": _cohens_h(p1, p2),
                    }
                )

            for column in schema.categorical + schema.ordinal:
                left_mode = left[column].astype(str).mode().iloc[0]
                right_mode = right[column].astype(str).mode().iloc[0]
                rows.append(
                    {
                        "cluster_left": left_cluster,
                        "cluster_right": right_cluster,
                        "feature": column,
                        "feature_type": "categorical",
                        "contrast": "mode_comparison",
                        "value": f"{left_mode} vs {right_mode}",
                        "effect_size": float(left_mode != right_mode),
                    }
                )

    return pd.DataFrame(rows)


def _cohens_d(left: pd.Series, right: pd.Series) -> float:
    left_std = float(left.std(ddof=1))
    right_std = float(right.std(ddof=1))
    pooled_variance = ((len(left) - 1) * (left_std**2) + (len(right) - 1) * (right_std**2)) / max(
        len(left) + len(right) - 2, 1
    )
    pooled_std = math.sqrt(max(pooled_variance, 0.0))
    if pooled_std == 0.0:
        return 0.0
    return float((left.mean() - right.mean()) / pooled_std)
