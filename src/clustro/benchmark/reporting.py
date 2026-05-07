"""Benchmark reporting and comparison plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from clustro.reporting.tables import write_table
from clustro.utils.io import write_json


def export_benchmark_report(
    summary: pd.DataFrame, root: Path, calibration: dict[str, object]
) -> None:
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_table(summary, report_dir / "benchmark_summary.csv")
    write_json(report_dir / "calibration_recommendations.json", calibration)

    _bar_plot(summary, "accepted_count", "Accepted Candidates", report_dir / "accepted_count.png")
    _bar_plot(
        summary, "top_weighted_score", "Top Weighted Score", report_dir / "top_weighted_score.png"
    )
    _bar_plot(
        summary,
        "mean_family_runtime_seconds",
        "Mean Family Runtime (s)",
        report_dir / "mean_runtime_seconds.png",
    )


def _bar_plot(summary: pd.DataFrame, column: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary["benchmark_family"], summary[column])
    ax.set_title(title)
    ax.set_ylabel(column)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
