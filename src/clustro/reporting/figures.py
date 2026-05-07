"""Figure-ready exports for Milestone 1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def export_quality_vs_stability(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(frame["silhouette"], frame["ari_seed"])
    ax.set_xlabel("Silhouette")
    ax.set_ylabel("Seed ARI")
    ax.set_title("Quality vs Stability")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_search_flow_diagram(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(frame["stage"], frame["count"], color="#4C78A8")
    ax.set_ylabel("Candidate count")
    ax.set_title("Search Flow")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_metric_heatmap(frame: pd.DataFrame, path: Path) -> None:
    metric_columns = [
        column
        for column in [
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "ari_seed",
            "nmi_seed",
            "mean_cluster_jaccard",
            "cluster_balance",
            "final_weighted_score",
        ]
        if column in frame.columns
    ]
    if frame.empty or not metric_columns:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = frame[metric_columns].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(max(6, len(metric_columns)), max(3, len(frame) * 0.35)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(metric_columns)), metric_columns, rotation=45, ha="right")
    ax.set_yticks(
        range(len(frame)),
        frame.get("candidate_id", pd.Series(range(len(frame)))).astype(str).str[:8],
    )
    ax.set_title("Accepted Candidate Metrics")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_coassociation_heatmap(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(frame.to_numpy(dtype=float), cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title("Consensus Co-association")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Sample")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_embedding_scatter(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty or not {"x", "y", "consensus_label"}.issubset(frame.columns):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    scatter = ax.scatter(
        frame["x"], frame["y"], c=frame["consensus_label"], cmap="tab10", s=12, alpha=0.8
    )
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.set_title("Final Visualization Embedding")
    fig.colorbar(scatter, ax=ax, shrink=0.8, label="Consensus cluster")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_uncertainty_distribution(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty or not {"consensus_label", "entropy"}.issubset(frame.columns):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    clusters = sorted(frame["consensus_label"].unique())
    values = [
        frame.loc[frame["consensus_label"] == cluster, "entropy"].to_numpy() for cluster in clusters
    ]
    ax.boxplot(values, tick_labels=[str(cluster) for cluster in clusters])
    ax.set_xlabel("Consensus cluster")
    ax.set_ylabel("Entropy")
    ax.set_title("Uncertainty by Cluster")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_feature_importance_bar(frame: pd.DataFrame, path: Path, *, top_n: int = 20) -> None:
    if frame.empty or "feature" not in frame.columns:
        return
    value_column = "importance_mean" if "importance_mean" in frame.columns else "mean_abs_shap"
    if value_column not in frame.columns:
        return
    top = frame.head(top_n).iloc[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, max(4, len(top) * 0.25)))
    ax.barh(top["feature"].astype(str), top[value_column].astype(float))
    ax.set_xlabel(value_column)
    ax.set_title("Top Feature Importance")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def export_cluster_profile_heatmap(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty or not {"cluster", "feature", "value"}.issubset(frame.columns):
        return
    numeric = frame.copy()
    numeric["value"] = pd.to_numeric(numeric["value"], errors="coerce")
    numeric = numeric.dropna(subset=["value"])
    if numeric.empty:
        return
    pivot = numeric.pivot_table(index="feature", columns="cluster", values="value", aggfunc="mean")
    if pivot.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(max(5, pivot.shape[1] * 1.2), max(5, min(18, pivot.shape[0] * 0.18)))
    )
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)), [str(column) for column in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(index) for index in pivot.index], fontsize=6)
    ax.set_xlabel("Consensus cluster")
    ax.set_title("Clinical Profile Heatmap")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
