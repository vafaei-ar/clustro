from __future__ import annotations

import numpy as np

from clustro.config.schema import ExperimentConfig
from clustro.search import scheduler
from clustro.search.scheduler import (
    _representative_seed_index,
    _summarize_seed_metrics,
    evaluate_candidate_full,
)
from clustro.search.search_space import Candidate


def _config(seeds: list[int]) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"name": "demo", "output_dir": "out"},
            "data": {
                "path": "dataset.csv",
                "column_schema": {
                    "continuous": ["x"],
                    "binary": [],
                    "categorical": [],
                    "ordinal": [],
                },
            },
            "search": {
                "seeds_full": seeds,
                "perturbations_full": 0,
                "stability_mode": "processed_matrix",
            },
            "evaluation": {"acceptance": {"hard_thresholds": {"silhouette_min": 0.0}}},
            "clustering": {"methods": [{"name": "kmeans", "params": {"n_clusters": [2]}}]},
        }
    )


def _candidate() -> Candidate:
    return Candidate(
        "c1",
        {"continuous_transform": "standard", "categorical_encoding": "onehot"},
        {"name": "none", "params": {}},
        {"name": "kmeans", "params": {"n_clusters": 2}},
        "test",
    )


def test_reordering_seeds_does_not_change_summary_metrics(monkeypatch) -> None:
    labels_by_seed = {
        1: np.array([0, 0, 1, 1]),
        2: np.array([0, 1, 0, 1]),
        3: np.array([0, 0, 1, 1]),
    }

    def fake_collect(candidate, matrix, seeds, *, config):
        runs = [labels_by_seed[seed] for seed in seeds]
        return {"seed_runs": float(len(seeds))}, runs, [matrix] * len(runs)

    monkeypatch.setattr(scheduler, "_collect_seed_runs", fake_collect)
    matrix = np.array([[0.0], [0.1], [1.0], [1.1]])

    a = evaluate_candidate_full(_candidate(), matrix, _config([1, 2, 3]))
    b = evaluate_candidate_full(_candidate(), matrix, _config([3, 2, 1]))

    assert a.metrics["silhouette"] == b.metrics["silhouette"]
    assert a.metrics["cluster_balance"] == b.metrics["cluster_balance"]


def test_representative_seed_is_not_necessarily_first() -> None:
    labels = [np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])]

    index, mean_ari = _representative_seed_index(labels)

    assert index == 1
    assert mean_ari > 0


def test_candidate_labels_correspond_to_representative_seed(monkeypatch) -> None:
    labels = [np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])]

    def fake_collect(candidate, matrix, seeds, *, config):
        return {"seed_runs": 3.0}, labels, [matrix] * len(labels)

    monkeypatch.setattr(scheduler, "_collect_seed_runs", fake_collect)
    execution = evaluate_candidate_full(
        _candidate(), np.array([[0.0], [0.1], [1.0], [1.1]]), _config([1, 2, 3])
    )

    assert execution.labels.tolist() == labels[1].tolist()
    assert execution.metrics["representative_seed"] == 2.0


def test_acceptance_uses_median_metrics_not_first_seed() -> None:
    matrix = np.array([[0.0], [0.1], [1.0], [1.1]])
    labels = [np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])]

    summary = _summarize_seed_metrics(matrix, labels, config=_config([1, 2, 3]))

    assert summary["silhouette"] == summary["silhouette_median"]
    assert summary["silhouette"] > 0
