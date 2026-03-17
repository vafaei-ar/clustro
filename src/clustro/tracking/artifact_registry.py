"""Filesystem-centric artifact registry for experiment outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import json

from clustro.utils.io import ensure_directory, write_json
from clustro.utils.paths import ExperimentPaths


@dataclass(slots=True)
class ArtifactRegistry:
    paths: ExperimentPaths

    def candidate_dir(self, candidate_id: str) -> Path:
        return ensure_directory(self.paths.candidates_dir / candidate_id)

    def candidate_file(self, candidate_id: str, name: str) -> Path:
        return self.candidate_dir(candidate_id) / name

    def manifest_path(self) -> Path:
        return self.paths.root / "experiment_manifest.json"

    def accepted_candidates_path(self) -> Path:
        return self.paths.root / "accepted_candidates.parquet"

    def rejected_candidates_path(self) -> Path:
        return self.paths.root / "rejected_candidates.parquet"

    def candidate_registry_path(self) -> Path:
        return self.paths.root / "candidate_registry.parquet"

    def runtime_summary_path(self) -> Path:
        return self.paths.root / "runtime_summary.csv"

    def method_family_summary_path(self) -> Path:
        return self.paths.root / "method_family_summary.csv"

    def consensus_file(self, name: str) -> Path:
        return self.paths.consensus_dir / name

    def report_file(self, name: str) -> Path:
        return self.paths.reports_dir / name

    def state_file(self, name: str) -> Path:
        return self.paths.state_dir / name

    def mark_stage(self, stage: str, payload: dict[str, object]) -> None:
        write_json(
            self.state_file(f"{stage}.json"),
            {
                "stage": stage,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **payload,
            },
        )

    def read_stage(self, stage: str) -> dict[str, object] | None:
        path = self.state_file(f"{stage}.json")
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
