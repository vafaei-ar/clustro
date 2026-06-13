from __future__ import annotations

import numpy as np

from clustro.consensus.uncertainty import compute_uncertainty


def test_perfect_deterministic_membership_is_not_ambiguous() -> None:
    labels = np.array([0, 0, 1, 1])
    coassociation = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=float)

    result = compute_uncertainty(coassociation, labels, ["a", "b", "c", "d"])

    assert np.allclose(result["entropy"], 0.0)
    assert (result["consensus_support_gap"] > 0.9).all()
    assert not result["ambiguous"].any()


def test_low_top2_gap_is_ambiguous() -> None:
    labels = np.array([0, 1])
    coassociation = np.array([[1.0, 0.95], [0.95, 1.0]])

    result = compute_uncertainty(
        coassociation, labels, ["a", "b"], ambiguous_top2_gap_threshold=0.1
    )

    assert result["ambiguous"].all()


def test_nonzero_high_entropy_rule_still_flags_rows() -> None:
    labels = np.array([0, 0, 1, 1])
    coassociation = np.array(
        [[1.0, 0.9, 0.1, 0.1], [0.9, 1.0, 0.1, 0.1], [0.1, 0.1, 1.0, 0.9], [0.5, 0.5, 0.5, 0.5]]
    )

    result = compute_uncertainty(
        coassociation,
        labels,
        ["a", "b", "c", "d"],
        ambiguous_top2_gap_threshold=0.0,
        ambiguous_entropy_quantile=0.75,
    )

    assert result.loc[3, "entropy"] > 0
    assert bool(result.loc[3, "ambiguous"])
