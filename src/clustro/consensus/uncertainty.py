"""Sample-level uncertainty estimates from co-association consensus."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_uncertainty(coassociation: np.ndarray, labels: np.ndarray, row_ids: list[str]) -> pd.DataFrame:
    clusters = [cluster for cluster in np.unique(labels) if cluster >= 0]
    memberships = np.zeros((len(labels), len(clusters)), dtype=float)
    for index, cluster in enumerate(clusters):
        members = np.where(labels == cluster)[0]
        if len(members) == 0:
            continue
        memberships[:, index] = coassociation[:, members].mean(axis=1)
    row_sums = memberships.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        memberships,
        row_sums,
        out=np.full_like(memberships, 1.0 / max(memberships.shape[1], 1)),
        where=row_sums > 0,
    )
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, None))).sum(axis=1)
    sorted_probs = np.sort(probabilities, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2] if probabilities.shape[1] >= 2 else np.ones(len(labels))
    frame = pd.DataFrame(
        {
            "row_id": row_ids,
            "consensus_label": labels,
            "entropy": entropy,
            "top2_gap": margin,
            "ambiguous": entropy > np.median(entropy) if len(entropy) else [],
        }
    )
    for index in range(probabilities.shape[1]):
        frame[f"membership_{index}"] = probabilities[:, index]
    return frame
