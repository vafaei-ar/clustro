"""Run weighting helpers for consensus."""

from __future__ import annotations

import numpy as np
import pandas as pd

from clustro.config.schema import ExperimentConfig


def compute_run_weights(frame: pd.DataFrame, config: ExperimentConfig) -> np.ndarray:
    source = config.consensus.run_weighting.source
    floor = config.consensus.run_weighting.floor
    values = frame[source].fillna(0.0).to_numpy(dtype=float)
    values = np.maximum(values, floor)
    if config.consensus.run_weighting.normalize:
        total = values.sum()
        if total > 0:
            values = values / total
    return values
