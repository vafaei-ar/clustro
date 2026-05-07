from __future__ import annotations

import numpy as np

from clustro.data.sampling import bootstrap_indices, sample_without_replacement, subsample_indices
from clustro.data.splitting import stratified_train_test_indices, train_test_indices
from clustro.repr.cache import RepresentationCache


def test_sampling_helpers_are_reproducible() -> None:
    first = sample_without_replacement(20, sample_fraction=0.4, min_rows=5, seed=3)
    second = sample_without_replacement(20, sample_fraction=0.4, min_rows=5, seed=3)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 8

    boot = bootstrap_indices(10, seed=4)
    assert boot.shape == (10,)

    sub = subsample_indices(10, sample_fraction=0.5, seed=5)
    assert len(sub) == 5


def test_splitting_helpers_are_reproducible() -> None:
    split = train_test_indices(10, test_size=0.3, seed=7)
    assert len(split.train) == 7
    assert len(split.test) == 3

    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    stratified = stratified_train_test_indices(labels, test_size=0.25, seed=7)
    assert set(labels[stratified.test]) == {0, 1}


def test_representation_cache_round_trip(tmp_path) -> None:
    cache = RepresentationCache(tmp_path / "repr_cache")
    key = cache.key_for(
        method="pca",
        params={"n_components": 2},
        matrix_fingerprint={"rows": 3, "cols": 2},
        seed=11,
    )
    matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    metadata = {"name": "pca", "explained_variance_ratio": [0.9, 0.1]}

    cache.store(key, matrix, metadata)
    loaded_matrix, loaded_metadata = cache.load(key)

    assert cache.exists(key)
    np.testing.assert_allclose(loaded_matrix, matrix)
    assert loaded_metadata == metadata
