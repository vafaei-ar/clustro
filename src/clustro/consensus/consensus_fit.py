"""Consensus clustering from co-association matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from clustro.consensus.coassociation import build_coassociation_matrix
from clustro.consensus.uncertainty import compute_uncertainty


@dataclass(slots=True)
class ConsensusResult:
    labels: np.ndarray
    coassociation: np.ndarray
    uncertainty: pd.DataFrame


def fit_consensus(
    label_runs: list[np.ndarray],
    weights: np.ndarray,
    row_ids: list[str],
    *,
    target_k: int,
) -> ConsensusResult:
    coassociation = build_coassociation_matrix(label_runs, weights)
    distance = 1.0 - coassociation
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average")
    labels = fcluster(tree, t=target_k, criterion="maxclust") - 1
    uncertainty = compute_uncertainty(coassociation, labels, row_ids)
    return ConsensusResult(labels=labels, coassociation=coassociation, uncertainty=uncertainty)
