"""Ranking helpers for accepted candidates."""

from __future__ import annotations

import pandas as pd


def rank_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.assign(rank=pd.Series(dtype=int))
    ranked = (
        frame.sort_values("final_weighted_score", ascending=False).reset_index(drop=True).copy()
    )
    ranked["rank"] = ranked.index + 1
    return ranked
