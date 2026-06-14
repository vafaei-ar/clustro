"""Known-data validation example.

Runs clustro on a synthetic biomedical dataset with 3 known clusters and
reports the adjusted Rand index between consensus labels and the ground truth.

Usage:
    python examples/run_known_clusters.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from clustro import Experiment

CONFIG = Path(__file__).parent / "configs" / "known_clusters_example.yaml"
DATA = Path(__file__).parent / "data" / "known_clusters.csv"


def main() -> None:
    print(f"Running known-cluster validation on {DATA.name} ...")
    exp = Experiment.from_yaml(CONFIG)
    exp.run()

    output_root = exp.paths.root
    consensus = pd.read_csv(output_root / "consensus_labels.csv")
    ground_truth = pd.read_csv(DATA)

    ari = adjusted_rand_score(ground_truth["true_cluster"], consensus["consensus_label"])
    n_clusters = int(consensus["consensus_label"].nunique())

    print(f"  Consensus clusters found : {n_clusters}")
    print(f"  Adjusted Rand Index      : {ari:.4f}")
    if ari >= 0.90:
        print("  PASS — ARI >= 0.90")
    else:
        print(f"  WARN — ARI {ari:.4f} is below the expected 0.90 threshold")


if __name__ == "__main__":
    main()
