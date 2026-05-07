"""Optional MLflow integration."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any


class MlflowLogger(AbstractContextManager["MlflowLogger"]):
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._mlflow = None

    def __enter__(self) -> MlflowLogger:
        if self.enabled:
            try:
                import mlflow  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "MLflow requested but not installed. Use clustro[tracking]."
                ) from exc
            self._mlflow = mlflow
        return self

    def __exit__(self, *args: object) -> None:
        if self._mlflow is not None:
            try:
                self._mlflow.end_run()
            except Exception:
                pass

    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> None:
        if self._mlflow is not None:
            self._mlflow.start_run(run_name=run_name, tags=tags or {})

    def log_params(self, params: dict[str, Any]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if self._mlflow is not None:
            self._mlflow.log_metrics(metrics)
