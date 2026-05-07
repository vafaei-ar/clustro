from __future__ import annotations

import numpy as np
import pytest

from clustro.evaluation.metrics_stability import (
    PerturbationLabelRun,
    _prepare_perturbation_comparison,
    summarize_perturbation_stability,
)


def test_bootstrap_stability_compares_unique_original_rows() -> None:
    reference = np.array([0, 0, 1, 1, 2, 2])
    run = PerturbationLabelRun(
        indices=np.array([0, 0, 1, 3, 3, 5]),
        labels=np.array([10, 10, 10, 11, 11, 12]),
        kind="bootstrap",
    )

    reference_subset, perturbation_subset, compared_indices = _prepare_perturbation_comparison(
        reference, run
    )

    assert compared_indices.tolist() == [0, 1, 3, 5]
    assert reference_subset.tolist() == [0, 0, 1, 2]
    assert perturbation_subset.tolist() == [10, 10, 11, 12]


def test_subsample_stability_compares_intersection_only() -> None:
    reference = np.array([0, 0, 1, 1, 2, 2])
    run = PerturbationLabelRun(
        indices=np.array([0, 2, 4]),
        labels=np.array([7, 8, 9]),
        kind="subsample",
    )

    reference_subset, perturbation_subset, compared_indices = _prepare_perturbation_comparison(
        reference, run
    )

    assert compared_indices.tolist() == [0, 2, 4]
    assert reference_subset.tolist() == [0, 1, 2]
    assert perturbation_subset.tolist() == [7, 8, 9]


def test_bootstrap_vector_is_not_treated_as_original_row_order() -> None:
    reference = np.array([0, 0, 1, 1, 2, 2])
    run = PerturbationLabelRun(
        indices=np.array([0, 0, 1, 3, 3, 5]),
        labels=np.array([0, 0, 0, 1, 1, 2]),
        kind="bootstrap",
    )

    metrics = summarize_perturbation_stability(reference, [run])

    assert metrics["perturbation_rows_compared_mean"] == 4.0


def test_subsample_duplicate_indices_are_rejected() -> None:
    reference = np.array([0, 0, 1, 1])
    run = PerturbationLabelRun(
        indices=np.array([0, 0, 2]),
        labels=np.array([1, 1, 2]),
        kind="subsample",
    )

    with pytest.raises(ValueError, match="Subsample perturbation indices must be unique"):
        summarize_perturbation_stability(reference, [run])
