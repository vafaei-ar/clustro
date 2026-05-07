"""Small disk cache for reusable representation matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from clustro.utils.hashing import stable_hash
from clustro.utils.io import ensure_directory, write_json


@dataclass(slots=True)
class RepresentationCache:
    root: Path

    def __post_init__(self) -> None:
        ensure_directory(self.root)

    def key_for(
        self, *, method: str, params: dict[str, Any], matrix_fingerprint: dict[str, Any], seed: int
    ) -> str:
        return stable_hash(
            {
                "method": method,
                "params": params,
                "matrix": matrix_fingerprint,
                "seed": seed,
            }
        )

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def exists(self, key: str) -> bool:
        return self.path_for(key).exists()

    def load(self, key: str) -> tuple[np.ndarray, dict[str, Any]]:
        payload = np.load(self.path_for(key), allow_pickle=True)
        matrix = np.asarray(payload["matrix"])
        metadata = payload["metadata"].item()
        return matrix, dict(metadata)

    def store(self, key: str, matrix: np.ndarray, metadata: dict[str, Any]) -> Path:
        path = self.path_for(key)
        ensure_directory(path.parent)
        np.savez_compressed(path, matrix=matrix, metadata=np.asarray(metadata, dtype=object))
        write_json(self.root / f"{key}.json", {"metadata": metadata})
        return path
