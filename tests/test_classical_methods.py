from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from clustro import Experiment
from clustro.clustering.classical import fit_predict_clusterer
from clustro.search.compatibility import validate_candidate
from clustro.search.search_space import Candidate


def test_new_classical_clusterers_fit_on_toy_matrix() -> None:
    matrix = _toy_matrix()
    cases = [
        ("minibatch_kmeans", {"n_clusters": 2, "batch_size": 6}),
        ("spectral", {"n_clusters": 2, "affinity": "nearest_neighbors", "n_neighbors": 4}),
        ("optics", {"min_samples": 2, "xi": 0.05}),
        ("birch", {"n_clusters": 2, "threshold": 0.5}),
    ]

    for name, params in cases:
        result = fit_predict_clusterer(name, matrix, params, seed=7)
        assert result.labels.shape == (matrix.shape[0],)
        assert len(np.unique(result.labels)) >= 1


def test_validate_config_accepts_new_classical_clusterer_names(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "x": [0.0, 0.1, 5.0, 5.1],
            "flag": [0, 1, 0, 1],
            "group": ["n", "n", "s", "s"],
        }
    ).to_csv(dataset_path, index=False)

    config = {
        "experiment": {"name": "classical_names", "output_dir": str(tmp_path / "results")},
        "data": {
            "path": str(dataset_path),
            "id_column": "id",
            "column_schema": {
                "continuous": ["x"],
                "binary": ["flag"],
                "categorical": ["group"],
                "ordinal": [],
            },
        },
        "representation": {"methods": [{"name": "none"}]},
        "clustering": {
            "methods": [
                {"name": "minibatch_kmeans", "params": {"n_clusters": [2]}},
                {
                    "name": "spectral",
                    "params": {
                        "n_clusters": [2],
                        "affinity": ["nearest_neighbors"],
                        "n_neighbors": [2],
                    },
                },
                {"name": "optics", "params": {"min_samples": [2], "xi": [0.05]}},
                {"name": "birch", "params": {"n_clusters": [2], "threshold": [0.5]}},
            ]
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    experiment = Experiment.from_yaml(config_path)

    assert [method.name for method in experiment.config.clustering.methods] == [
        "minibatch_kmeans",
        "spectral",
        "optics",
        "birch",
    ]


def test_spectral_is_rejected_for_large_datasets() -> None:
    candidate = Candidate(
        candidate_id="demo",
        preprocessing={"continuous_transform": "standard"},
        representation={"name": "none", "params": {}},
        clustering={"name": "spectral", "params": {"n_clusters": 4}},
        family="spectral",
    )

    decision = validate_candidate(candidate, n_rows=6000, n_features=20)

    assert not decision.allowed
    assert "spectral_impractical_for_dataset_size" in decision.reasons


def test_gpu_requested_falls_back_with_metadata_when_rapids_unavailable() -> None:
    result = fit_predict_clusterer(
        "agglomerative",
        _toy_matrix(),
        {"n_clusters": 2, "linkage": "ward"},
        seed=7,
        use_gpu_if_available=True,
    )

    assert result.metadata["accelerator"] == "sklearn"
    assert result.metadata["accelerator_fallback_reason"] == "rapids_method_not_supported"


def test_rapids_kmeans_path_can_be_used_when_available(monkeypatch) -> None:
    class FakeKMeans:
        def __init__(self, **params):
            self.params = params
            self.inertia_ = 1.5

        def fit_predict(self, matrix):
            return np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

    fake_cluster = types.ModuleType("cuml.cluster")
    fake_cluster.KMeans = FakeKMeans
    fake_cuml = types.ModuleType("cuml")
    fake_cuml.cluster = fake_cluster
    monkeypatch.setitem(sys.modules, "cuml", fake_cuml)
    monkeypatch.setitem(sys.modules, "cuml.cluster", fake_cluster)

    result = fit_predict_clusterer(
        "kmeans",
        _toy_matrix(),
        {"n_clusters": 2},
        seed=7,
        use_gpu_if_available=True,
    )

    assert result.metadata["accelerator"] == "rapids"
    assert result.metadata["rapids_estimator"] == "cuml.cluster.KMeans"
    assert result.metadata["inertia"] == 1.5


def _toy_matrix() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, -0.1],
            [0.15, 0.05],
            [5.0, 5.1],
            [5.2, 4.9],
            [4.9, 5.0],
            [5.1, 5.2],
        ],
        dtype=float,
    )
