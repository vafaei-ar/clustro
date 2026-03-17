from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from clustro.benchmark.runner import run_classical_vs_deep_benchmark


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_classical_vs_deep_benchmark(tmp_path: Path) -> None:
    summary = run_classical_vs_deep_benchmark(tmp_path / "benchmark")

    assert set(summary["benchmark_family"]) == {"classical", "deep"}
    assert (tmp_path / "benchmark" / "benchmark_summary.csv").exists()
