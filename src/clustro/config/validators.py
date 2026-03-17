"""Config loading and validation helpers."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

from clustro.config.schema import ExperimentConfig
from clustro.utils.io import read_yaml
from clustro.utils.paths import resolve_config_path, resolve_from_config_dir


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = resolve_config_path(path)
    raw_config = _deep_merge(_load_defaults(), read_yaml(config_path))
    config = ExperimentConfig.model_validate(raw_config)
    resolved_data_path = resolve_from_config_dir(config.data.path, config_path=config_path)
    resolved_output_dir = resolve_from_config_dir(config.experiment.output_dir, config_path=config_path)
    if not resolved_data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {resolved_data_path}")
    _validate_search_space(config)
    return config.with_runtime_paths(
        config_path=config_path,
        resolved_data_path=resolved_data_path,
        resolved_output_dir=resolved_output_dir,
    )


def _validate_search_space(config: ExperimentConfig) -> None:
    if not config.clustering.methods:
        raise ValueError("At least one clustering method must be configured.")
    supported_clusterers = {
        "kmeans",
        "gmm",
        "agglomerative",
        "hdbscan",
        "ae_kmeans",
        "ae_gmm",
        "dec",
        "vade",
    }
    unsupported_clusterers = {method.name for method in config.clustering.methods}.difference(supported_clusterers)
    if unsupported_clusterers:
        raise ValueError(f"Unsupported clustering method(s): {sorted(unsupported_clusterers)}")
    encodings = set(config.preprocessing.categorical_encoding)
    if not encodings.issubset({"onehot", "ordinal"}):
        raise ValueError(f"Unsupported categorical encoding(s): {sorted(encodings)}")
    transforms = set(config.preprocessing.continuous_transforms)
    supported = {"none", "standard", "robust", "power", "log1p_standard"}
    unsupported = transforms.difference(supported)
    if unsupported:
        raise ValueError(f"Unsupported continuous transform(s): {sorted(unsupported)}")
    if "log1p_standard" in transforms and not config.data.column_schema.continuous:
        raise ValueError("log1p_standard requires at least one continuous column.")
    supported_representations = {"none", "pca", "umap", "autoencoder"}
    unsupported_representations = {method.name for method in config.representation.methods}.difference(
        supported_representations
    )
    if unsupported_representations:
        raise ValueError(f"Unsupported representation method(s): {sorted(unsupported_representations)}")


def _load_defaults() -> dict[str, Any]:
    defaults_path = files("clustro").joinpath("config/defaults.yaml")
    with defaults_path.open("r", encoding="utf-8") as handle:
        import yaml

        return yaml.safe_load(handle) or {}


def _deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
