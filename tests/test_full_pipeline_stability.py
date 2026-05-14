from __future__ import annotations

import numpy as np
import pandas as pd

from clustro.config.schema import ExperimentConfig
from clustro.evaluation.metrics_stability import (
    PerturbationLabelRun,
    summarize_perturbation_stability,
)
from clustro.search.scheduler import _collect_perturbations
from clustro.search.search_space import Candidate


def _candidate() -> Candidate:
    return Candidate(
        candidate_id="c1",
        preprocessing={"continuous_transform": "standard", "categorical_encoding": "onehot"},
        representation={"name": "none", "params": {}},
        clustering={"name": "kmeans", "params": {"n_clusters": 2}},
        family="kmeans",
    )


def _config(mode: str) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"name": "demo", "output_dir": "out", "random_seed": 7},
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
                "perturbations_full": 2,
                "perturbation_type": "subsample",
                "stability_mode": mode,
            },
            "clustering": {"methods": [{"name": "kmeans", "params": {"n_clusters": [2]}}]},
        }
    )


def test_full_pipeline_perturbation_refits_preprocessing(monkeypatch) -> None:
    calls: list[int] = []

    def fake_preprocess_frame(
        frame, config, *, continuous_transform=None, categorical_encoding=None
    ):
        calls.append(len(frame))
        return type("Preprocessed", (), {"evaluation_matrix": frame[["x"]].to_numpy(dtype=float)})()

    monkeypatch.setattr("clustro.search.scheduler.preprocess_frame", fake_preprocess_frame)
    matrix = np.arange(10, dtype=float).reshape(-1, 1)
    frame = pd.DataFrame({"x": np.arange(10, dtype=float)})

    _collect_perturbations(_candidate(), matrix, _config("full_pipeline"), raw_frame=frame)

    assert len(calls) == 2
    assert all(size == 8 for size in calls)


def test_processed_matrix_mode_does_not_refit_preprocessing(monkeypatch) -> None:
    def fail_preprocess_frame(*args, **kwargs):
        raise AssertionError("processed_matrix mode should not refit preprocessing")

    monkeypatch.setattr("clustro.search.scheduler.preprocess_frame", fail_preprocess_frame)
    matrix = np.arange(10, dtype=float).reshape(-1, 1)

    runs = _collect_perturbations(_candidate(), matrix, _config("processed_matrix"))

    assert len(runs) == 2


def test_perturbation_indices_preserve_original_row_identity() -> None:
    reference = np.array([0, 0, 1, 1, 0])
    run = PerturbationLabelRun(
        indices=np.array([4, 0, 2]), labels=np.array([0, 0, 1]), kind="subsample"
    )

    metrics = summarize_perturbation_stability(reference, [run])

    assert metrics["perturbation_rows_compared_mean"] == 3.0
