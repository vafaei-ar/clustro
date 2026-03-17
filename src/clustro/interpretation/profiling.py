"""Cluster profiling summaries."""

from __future__ import annotations

import pandas as pd

from clustro.data.schema import DatasetSchema


def build_cluster_profiles(frame: pd.DataFrame, labels: pd.Series, schema: DatasetSchema) -> pd.DataFrame:
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
