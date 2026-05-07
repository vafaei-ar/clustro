from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from clustro.interpretation.shap_utils import compute_shap_summary


@pytest.mark.skipif(importlib.util.find_spec("shap") is None, reason="shap not installed")
def test_compute_shap_summary_returns_global_and_classwise_outputs() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(40, 4))
    y = np.repeat([0, 1], 20)
    x[y == 1, 0] += 2.0
    model = RandomForestClassifier(random_state=3, n_estimators=20)
    model.fit(x, y)

    summary, detail, classwise = compute_shap_summary(
        model,
        x,
        ["a", "b", "c", "d"],
        max_rows=20,
        random_seed=13,
        row_ids=[f"r{i}" for i in range(len(x))],
    )

    assert not summary.empty
    assert not detail.empty
    assert "feature" in summary.columns
    assert set(detail.columns) == {"sample_index", "row_id", "a", "b", "c", "d"}
    assert set(classwise.columns) >= {"class_index", "feature", "mean_abs_shap"}


@pytest.mark.skipif(importlib.util.find_spec("shap") is None, reason="shap not installed")
def test_compute_shap_summary_uses_deterministic_random_sample() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(40, 4))
    y = np.repeat([0, 1], 20)
    model = RandomForestClassifier(random_state=3, n_estimators=20)
    model.fit(x, y)

    _, first, _ = compute_shap_summary(model, x, ["a", "b", "c", "d"], max_rows=10, random_seed=13)
    _, second, _ = compute_shap_summary(model, x, ["a", "b", "c", "d"], max_rows=10, random_seed=13)

    assert first["sample_index"].tolist() == second["sample_index"].tolist()
    assert first["sample_index"].tolist() != list(range(10))
