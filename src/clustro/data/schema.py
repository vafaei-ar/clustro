"""Dataset schema helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from clustro.config.schema import ColumnSchemaConfig


@dataclass(slots=True)
class DatasetSchema:
    continuous: list[str]
    binary: list[str]
    categorical: list[str]
    ordinal: list[str]

    @classmethod
    def from_config(cls, schema: ColumnSchemaConfig) -> "DatasetSchema":
        return cls(
            continuous=list(schema.continuous),
            binary=list(schema.binary),
            categorical=list(schema.categorical),
            ordinal=list(schema.ordinal),
        )

    def all_columns(self) -> list[str]:
        return self.continuous + self.binary + self.categorical + self.ordinal

    def validate_against(self, frame: pd.DataFrame) -> None:
        missing = sorted(set(self.all_columns()).difference(frame.columns))
        if missing:
            raise ValueError(f"Dataset is missing declared columns: {missing}")
