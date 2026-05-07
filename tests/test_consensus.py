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
    assert {"consensus_label", "cluster_size", "mean_within_cluster_consensus"}.issubset(
        result.cluster_summary.columns
    )
    assert {"consensus_label", "bootstrap_recovery_mean"}.issubset(
        result.bootstrap_stability.columns
    )


def test_spectral_consensus_and_bootstrap_outputs() -> None:
    labels_a = np.array([0, 0, 1, 1, 2, 2])
    labels_b = np.array([0, 0, 1, 1, 2, 2])
    labels_c = np.array([0, 0, 1, 1, 2, 2])
    weights = np.array([0.5, 0.3, 0.2])

    result = fit_consensus(
        [labels_a, labels_b, labels_c],
        weights,
        ["a", "b", "c", "d", "e", "f"],
        target_k=3,
        method="spectral_on_coassociation",
        bootstrap_repeats=3,
        random_seed=7,
    )

    assert set(result.labels.tolist()) == {0, 1, 2}
    assert len(result.cluster_summary) == 3
    assert len(result.bootstrap_stability) == 3
