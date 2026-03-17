"""Path resolution and experiment directory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clustro.utils.io import ensure_directory


def resolve_config_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def resolve_from_config_dir(value: str | Path, *, config_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


@dataclass(slots=True)
class ExperimentPaths:
    root: Path
    registry_dir: Path
    candidates_dir: Path
    consensus_dir: Path
    reports_dir: Path
    logs_dir: Path
    cache_dir: Path
    state_dir: Path


def build_experiment_paths(output_dir: Path) -> ExperimentPaths:
    root = ensure_directory(output_dir)
    return ExperimentPaths(
        root=root,
        registry_dir=ensure_directory(root / "registry"),
        candidates_dir=ensure_directory(root / "candidates"),
        consensus_dir=ensure_directory(root / "consensus"),
        reports_dir=ensure_directory(root / "reports"),
        logs_dir=ensure_directory(root / "logs"),
        cache_dir=ensure_directory(root / "cache"),
        state_dir=ensure_directory(root / "state"),
    )
