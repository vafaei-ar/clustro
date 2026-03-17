"""High-level export routines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clustro.reporting.figures import export_quality_vs_stability
from clustro.reporting.manuscript_bundle import create_manuscript_bundle
from clustro.reporting.tables import write_table
from clustro.utils.io import write_json


def export_experiment_tables(
    *,
    candidate_registry: pd.DataFrame,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    output_dir: Path,
) -> None:
    write_table(candidate_registry, output_dir / "candidate_registry.parquet")
    write_table(accepted, output_dir / "accepted_candidates.parquet")
    write_table(rejected, output_dir / "rejected_candidates.parquet")


def export_consensus_outputs(
    *,
    labels: pd.DataFrame,
    uncertainty: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    write_table(labels, output_dir / "consensus_labels.csv")
    write_table(uncertainty, output_dir / "consensus_uncertainty.csv")
    write_table(uncertainty, output_dir / "consensus_soft_membership.parquet")
    write_table(cluster_summary, output_dir / "consensus_cluster_summary.csv")


def export_report_bundle(candidate_registry: pd.DataFrame, output_dir: Path) -> None:
    report_dir = output_dir / "reports"
    write_table(candidate_registry, report_dir / "candidate_metrics.csv")
    plot_frame = candidate_registry.dropna(subset=["silhouette", "ari_seed"])
    if not plot_frame.empty:
        export_quality_vs_stability(plot_frame, report_dir / "quality_vs_stability.png")
    if not candidate_registry.empty:
        write_json(
            report_dir / "search_flow.json",
            {
                "candidate_count": int(len(candidate_registry)),
                "accepted_count": int(candidate_registry["accepted"].sum()),
                "rejected_count": int((~candidate_registry["accepted"]).sum()),
            },
        )
    create_manuscript_bundle(output_dir)
