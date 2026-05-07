"""Synthetic benchmark dataset builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def build_benchmark_dataset(*, random_seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    blocks = []
    specs = [
        ("stable_a", 48, 54, 27, 108, "north", "private"),
        ("stable_b", 48, 66, 33, 168, "central", "public"),
        ("stable_c", 48, 77, 25, 132, "south", "public"),
    ]
    for prefix, n, age_mean, bmi_mean, glucose_mean, site, insurance in specs:
        block = pd.DataFrame(
            {
                "patient_id": [f"{prefix}_{i:03d}" for i in range(n)],
                "age": rng.normal(age_mean, 4.0, n).round(1),
                "bmi": rng.normal(bmi_mean, 2.0, n).round(2),
                "glucose": rng.normal(glucose_mean, 14.0, n).round(1),
                "sbp": rng.normal(120 + (age_mean - 50) * 0.5, 7.0, n).round(1),
                "dbp": rng.normal(78 + (bmi_mean - 25) * 0.3, 5.0, n).round(1),
                "marker": rng.normal(1.2 + (glucose_mean - 100) / 45.0, 0.3, n).round(3),
                "sex_male": rng.binomial(1, 0.5, n),
                "smoker": rng.binomial(1, 0.22 + 0.08 * (site == "central"), n),
                "hypertension": rng.binomial(1, 0.25 + 0.2 * (age_mean > 60), n),
                "race": rng.choice(
                    ["white", "black", "asian", "other"], size=n, p=[0.42, 0.25, 0.18, 0.15]
                ),
                "insurance_type": [insurance] * n,
                "site": [site] * n,
            }
        )
        blocks.append(block)
    frame = pd.concat(blocks, ignore_index=True)
    for column in ["bmi", "glucose", "race"]:
        indices = rng.choice(frame.index.to_numpy(), size=8, replace=False)
        frame.loc[indices, column] = np.nan
    return frame


def write_benchmark_inputs(root: Path, *, random_seed: int = 2026) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    dataset_path = root / "benchmark_dataset.csv"
    frame = build_benchmark_dataset(random_seed=random_seed)
    frame.to_csv(dataset_path, index=False)
    return dataset_path, root


def write_yaml_config(path: Path, config: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
