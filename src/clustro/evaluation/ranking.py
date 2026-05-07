"""Ranking helpers for accepted candidates."""

from __future__ import annotations

import pandas as pd

# Stable tie-breakers after final_weighted_score (descending). Candidate id is last and unique.
_RANK_TIE_BREAKERS: tuple[str, ...] = (
    "family",
    "continuous_transform",
    "categorical_encoding",
    "representation_name",
    "clustering_name",
    "candidate_id",
)


def rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(rank=pd.Series(dtype=int))
    sort_columns = ["final_weighted_score"]
    ascending = [False]
    for column in _RANK_TIE_BREAKERS:
        if column in frame.columns:
            sort_columns.append(column)
            ascending.append(True)
    ranked = frame.sort_values(
        sort_columns,
        ascending=ascending,
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked
