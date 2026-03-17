from __future__ import annotations

import numpy as np

from clustro.consensus.coassociation import build_coassociation_matrix
from clustro.consensus.consensus_fit import fit_consensus


def test_weighted_coassociation_and_consensus() -> None:
    labels_a = np.array([0, 0, 1, 1])
    labels_b = np.array([0, 0, 1, 1])
    weights = np.array([0.7, 0.3])

    matrix = build_coassociation_matrix([labels_a, labels_b], weights)
    assert matrix.shape == (4, 4)
    assert matrix[0, 1] == 1.0
    assert matrix[0, 2] == 0.0

    result = fit_consensus([labels_a, labels_b], weights, ["a", "b", "c", "d"], target_k=2)
    assert set(result.labels.tolist()) == {0, 1}
    assert {"row_id", "consensus_label", "entropy"}.issubset(result.uncertainty.columns)
