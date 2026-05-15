"""Sample-level uncertainty estimates from co-association consensus."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_uncertainty(
    coassociation: np.ndarray,
    labels: np.ndarray,
    row_ids: list[str],
    *,
    ambiguous_top2_gap_threshold: float = 0.10,
    ambiguous_entropy_quantile: float = 0.90,
) -> pd.DataFrame:
    clusters = sorted(int(cluster) for cluster in np.unique(labels) if cluster >= 0)
    memberships = np.zeros((len(labels), len(clusters)), dtype=np.float64)
    coassociation = np.asarray(coassociation, dtype=np.float64)
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
    max_entropy = np.log(max(probabilities.shape[1], 1))
    normalized_entropy = (
        entropy / max_entropy if max_entropy > 0 else np.zeros(len(labels), dtype=float)
    )
    sorted_probs = np.sort(probabilities, axis=1, kind="mergesort")
    top1 = sorted_probs[:, -1] if probabilities.shape[1] else np.ones(len(labels))
    top2 = sorted_probs[:, -2] if probabilities.shape[1] >= 2 else np.zeros(len(labels))
    margin = (
        sorted_probs[:, -1] - sorted_probs[:, -2]
        if probabilities.shape[1] >= 2
        else np.ones(len(labels))
    )
    entropy_cutoff = (
        float(np.quantile(entropy, ambiguous_entropy_quantile, method="linear"))
        if len(entropy)
        else np.inf
    )
    ambiguous_by_gap = margin < ambiguous_top2_gap_threshold
    ambiguous_by_entropy = (entropy > np.finfo(float).eps) & (entropy >= entropy_cutoff)
    ambiguous = ambiguous_by_gap | ambiguous_by_entropy
    frame = pd.DataFrame(
        {
            "row_id": row_ids,
            "consensus_label": labels,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "top1_membership": top1,
            "top2_membership": top2,
            "top2_gap": margin,
            "ambiguous": ambiguous,
        }
    )
    for index in range(probabilities.shape[1]):
        frame[f"membership_{index}"] = probabilities[:, index]
    return frame
