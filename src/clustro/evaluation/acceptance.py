"""Acceptance and hard rejection logic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from clustro.config.schema import ExperimentConfig


@dataclass(slots=True)
class AcceptanceResult:
    accepted: bool
    reasons: list[str]
    final_weighted_score: float


def evaluate_acceptance(metrics: dict[str, float], config: ExperimentConfig) -> AcceptanceResult:
    thresholds = config.evaluation.acceptance.hard_thresholds
    structure = config.evaluation.structure_constraints
    reasons: list[str] = []

    n_clusters = int(metrics.get("n_clusters", 0))
    if n_clusters < structure.min_clusters or n_clusters > structure.max_clusters:
        reasons.append("cluster_count_out_of_range")
    if metrics.get("dominant_cluster_fraction", 0.0) > structure.dominant_cluster_cap:
        reasons.append("dominant_cluster_too_large")
    if metrics.get("noise_fraction", 0.0) > structure.max_noise_fraction:
        reasons.append("noise_fraction_too_large")
    if metrics.get("min_cluster_fraction", 1.0) < structure.min_cluster_fraction:
        reasons.append("cluster_too_small")

    for name, threshold in thresholds.items():
        metric_name = name.removesuffix("_min")
        if metrics.get(metric_name, float("-inf")) < threshold:
            reasons.append(f"{metric_name}_below_threshold")

    weighted_score = compute_weighted_score(metrics, config)
    return AcceptanceResult(
        accepted=not reasons, reasons=reasons, final_weighted_score=weighted_score
    )


def compute_weighted_score(metrics: dict[str, float], config: ExperimentConfig) -> float:
    weights = config.evaluation.acceptance.weighted_score
    score = 0.0
    for metric_name, weight in weights.items():
        score += weight * float(metrics.get(metric_name, 0.0))
    return score


def apply_acceptance_policy(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    if frame.empty:
        return frame

    result = frame.copy()
    hard_pass = result["accepted"].fillna(False).astype(bool)
    result.loc[~hard_pass, "accepted"] = False

    top_fraction = float(config.evaluation.acceptance.accept_top_fraction_if_above)
    eligible = result.loc[hard_pass].sort_values("final_weighted_score", ascending=False).copy()
    if eligible.empty:
        return result

    if top_fraction <= 0.0:
        accepted_ids: set[object] = set()
    elif top_fraction < 1.0:
        keep_count = max(1, math.ceil(len(eligible) * max(top_fraction, 0.0)))
        accepted_ids = set(eligible.head(keep_count)["candidate_id"].tolist())
    else:
        accepted_ids = set(eligible["candidate_id"].tolist())

    result["accepted"] = result["candidate_id"].isin(accepted_ids)
    dropped_mask = hard_pass & ~result["accepted"].astype(bool)
    if dropped_mask.any():
        existing = result.loc[dropped_mask, "rejection_reasons"].fillna("").astype(str)
        suffix = "outside_top_fraction_policy"
        result.loc[dropped_mask, "rejection_reasons"] = existing.apply(
            lambda value: suffix if not value else f"{value};{suffix}"
        )
    return result
