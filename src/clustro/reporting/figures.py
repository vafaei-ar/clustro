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
