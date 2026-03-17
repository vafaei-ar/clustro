"""Candidate search-space generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

from clustro.config.schema import ExperimentConfig, MethodConfig
from clustro.utils.hashing import stable_hash


@dataclass(slots=True)
class Candidate:
    candidate_id: str
    preprocessing: dict[str, Any]
    representation: dict[str, Any]
    clustering: dict[str, Any]
    family: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_candidates(config: ExperimentConfig, dataset_fingerprint: dict[str, Any]) -> list[Candidate]:
    candidates: list[Candidate] = []
    for transform in config.preprocessing.continuous_transforms:
        for representation_method in config.representation.methods:
            for clustering_method in config.clustering.methods:
                for repr_params in _expand_method_params(representation_method):
                    for cluster_params in _expand_method_params(clustering_method):
                        payload = {
                            "transform": transform,
                            "representation": {"name": representation_method.name, "params": repr_params},
                            "clustering": {"name": clustering_method.name, "params": cluster_params},
                            "dataset": dataset_fingerprint,
                        }
                        candidates.append(
                            Candidate(
                                candidate_id=stable_hash(payload),
                                preprocessing={"continuous_transform": transform},
                                representation={"name": representation_method.name, "params": repr_params},
                                clustering={"name": clustering_method.name, "params": cluster_params},
                                family=clustering_method.name,
                            )
                        )
    return candidates


def _expand_method_params(method: MethodConfig) -> list[dict[str, Any]]:
    if not method.params:
        return [{}]
    keys = list(method.params)
    value_grid = []
    for key in keys:
        raw = method.params[key]
        if isinstance(raw, list):
            value_grid.append(raw)
        else:
            value_grid.append([raw])
    return [dict(zip(keys, values, strict=True)) for values in product(*value_grid)]
