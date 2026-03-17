from __future__ import annotations

from pathlib import Path

from clustro.utils.paths import build_experiment_paths


def test_build_experiment_paths_creates_expected_directories(tmp_path: Path) -> None:
    paths = build_experiment_paths(tmp_path / "results" / "run1")

    assert paths.root.exists()
    assert paths.registry_dir.exists()
    assert paths.candidates_dir.exists()
    assert paths.consensus_dir.exists()
    assert paths.reports_dir.exists()
    assert paths.logs_dir.exists()
    assert paths.cache_dir.exists()
    assert paths.state_dir.exists()
