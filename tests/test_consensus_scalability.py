from __future__ import annotations

import numpy as np
import pytest

from clustro.consensus.coassociation import build_coassociation_matrix


def test_large_coassociation_raises_before_dense_allocation() -> None:
    labels = [np.zeros(6, dtype=int)]
    weights = np.array([1.0])

    with pytest.raises(RuntimeError, match="Sparse/blockwise co-association is not implemented"):
        build_coassociation_matrix(labels, weights, storage="auto", max_dense_n=5)
