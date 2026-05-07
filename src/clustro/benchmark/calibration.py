"""Calibration utilities based on benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def calibrate_from_benchmark(root: Path) -> dict[str, object]:
    classical_metrics = pd.read_csv(
        root / "classical" / "results" / "reports" / "candidate_metrics.csv"
    )
    deep_metrics = pd.read_csv(root / "deep" / "results" / "reports" / "candidate_metrics.csv")

    accepted_deep = deep_metrics.loc[deep_metrics["accepted"]].copy()
    accepted_classical = classical_metrics.loc[classical_metrics["accepted"]].copy()
    recommendation = {
        "deep_runtime_ratio_vs_classical": float(
            accepted_deep["runtime_seconds"].mean()
            / max(accepted_classical["runtime_seconds"].mean(), 1e-9)
        )
        if not accepted_deep.empty and not accepted_classical.empty
        else None,
        "recommended_weighted_score": {
            "silhouette": 0.15,
            "davies_bouldin": -0.05,
            "calinski_harabasz": 0.05,
            "ari_seed": 0.25,
            "nmi_seed": 0.20,
            "mean_cluster_jaccard": 0.20,
            "cluster_balance": 0.10,
            "average_confidence": 0.10,
            "assignment_entropy": -0.08,
            "reconstruction_loss": -0.04,
            "runtime_penalty": -0.05,
            "parsimony_penalty": -0.03,
        },
        "recommended_deep_methods": _recommended_methods(accepted_deep),
        "recommended_latent_dims": [5, 8],
        "recommended_hidden_layers": [[64, 32], [128, 64]],
    }
    pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in recommendation["recommended_weighted_score"].items()
        ]
    ).to_csv(root / "calibrated_weighted_score.csv", index=False)
    pd.DataFrame({"method": recommendation["recommended_deep_methods"]}).to_csv(
        root / "recommended_deep_methods.csv",
        index=False,
    )
    return recommendation


def _recommended_methods(accepted_deep: pd.DataFrame) -> list[str]:
    if accepted_deep.empty:
        return ["ae_kmeans", "dec", "vade"]
    ranked = accepted_deep.sort_values("final_weighted_score", ascending=False)
    return ranked["family"].drop_duplicates().tolist()
