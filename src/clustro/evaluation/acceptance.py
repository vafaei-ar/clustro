"""Acceptance and hard rejection logic."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from clustro.config.schema import ExperimentConfig
from clustro.evaluation.metric_utils import (
    add_utility_columns,
    compute_utility_weighted_score,
    raw_metric_name,
)


@dataclass(slots=True)
class AcceptanceResult:
    accepted: bool
    reasons: list[str]
    final_weighted_score: float


def hard_gate_reasons(metrics: dict[str, float], config: ExperimentConfig) -> list[str]:
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
    return reasons


def hard_score_gate_reasons(metrics: dict[str, float], config: ExperimentConfig) -> list[str]:
    """Hard structural/threshold gates plus ability to compute weighted score."""
    reasons = list(hard_gate_reasons(metrics, config))
    try:
        compute_weighted_score(metrics, config)
    except KeyError as exc:
        reasons.append(str(exc).strip("'"))
    return reasons


def metrics_dict_from_registry_row(row: pd.Series, config: ExperimentConfig) -> dict[str, float]:
    metrics: dict[str, float] = {}
    structural = [
        "n_clusters",
        "dominant_cluster_fraction",
        "noise_fraction",
        "min_cluster_fraction",
    ]
    for name in structural:
        if name in row.index and pd.notna(row[name]):
            metrics[name] = float(row[name])

    for thresh_key in config.evaluation.acceptance.hard_thresholds:
        metric_name = thresh_key.removesuffix("_min")
        if metric_name in row.index and pd.notna(row[metric_name]):
            metrics[metric_name] = float(row[metric_name])

    for weighted_name in config.evaluation.acceptance.weighted_score:
        raw = raw_metric_name(weighted_name)
        if raw in row.index and pd.notna(row[raw]):
            metrics[raw] = float(row[raw])
    return metrics


def evaluate_acceptance(metrics: dict[str, float], config: ExperimentConfig) -> AcceptanceResult:
    reasons = hard_score_gate_reasons(metrics, config)
    weighted_score = float("nan")
    try:
        weighted_score = compute_weighted_score(metrics, config)
    except KeyError:
        pass
    return AcceptanceResult(
        accepted=not reasons, reasons=reasons, final_weighted_score=weighted_score
    )


def compute_weighted_score(metrics: dict[str, float], config: ExperimentConfig) -> float:
    return compute_utility_weighted_score(metrics, config.evaluation.acceptance.weighted_score)


def _persisted_hard_gate_columns(row: pd.Series, frame: pd.DataFrame) -> tuple[bool, str] | None:
    """Return (passed, hard_rejection_reasons) if row carries persisted hard-gate snapshot."""
    if "hard_filter_passed" not in frame.columns:
        return None
    value = row.get("hard_filter_passed")
    if pd.isna(value):
        return None
    hr_raw = row.get("hard_rejection_reasons")
    if hr_raw is None or (isinstance(hr_raw, float) and pd.isna(hr_raw)):
        hr_str = ""
    else:
        hr_str = str(hr_raw)
    return bool(value), hr_str


def apply_acceptance_policy(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    if frame.empty:
        return frame

    result = add_utility_columns(frame, config.evaluation.acceptance.weighted_score)

    hf_flags: list[bool] = []
    hr_strings: list[str] = []
    for _, row in result.iterrows():
        persisted = _persisted_hard_gate_columns(row, result)
        if persisted is not None:
            passed, hr_str = persisted
            hf_flags.append(passed)
            hr_strings.append(hr_str)
            continue
        metrics = metrics_dict_from_registry_row(row, config)
        rs = hard_score_gate_reasons(metrics, config)
        hf_flags.append(len(rs) == 0)
        hr_strings.append(";".join(rs))

    result["hard_filter_passed"] = hf_flags
    result["hard_rejection_reasons"] = hr_strings

    score_ok = result["final_weighted_score"].notna()
    hard_pass = pd.Series(hf_flags, index=result.index) & score_ok
    result["accepted_before_top_fraction"] = hard_pass

    top_fraction = float(config.evaluation.acceptance.accept_top_fraction_if_above)
    sort_cols = ["final_weighted_score"]
    ascending = [False]
    if "candidate_id" in result.columns:
        sort_cols.append("candidate_id")
        ascending.append(True)
    eligible = (
        result.loc[hard_pass]
        .sort_values(sort_cols, ascending=ascending, kind="mergesort")
        .copy()
    )

    result["final_rejection_reasons"] = result["hard_rejection_reasons"].fillna("").astype(str)

    if eligible.empty:
        result["accepted"] = False
        result["final_rejection_reasons"] = result["hard_rejection_reasons"].fillna("").astype(str)
        result["rejection_reasons"] = result["final_rejection_reasons"]
        return result

    if top_fraction <= 0.0:
        accepted_ids: set[object] = set()
    elif top_fraction < 1.0:
        keep_count = max(1, math.ceil(len(eligible) * max(top_fraction, 0.0)))
        accepted_ids = set(eligible.head(keep_count)["candidate_id"].tolist())
    else:
        accepted_ids = set(eligible["candidate_id"].tolist())

    result["accepted"] = result["candidate_id"].isin(accepted_ids)

    def _append_top_fraction_policy(reason_base: str) -> str:
        suffix = "outside_top_fraction_policy"
        base = str(reason_base or "")
        parts = [segment for segment in base.split(";") if segment]
        if suffix in parts:
            return base
        return suffix if not base else f"{base};{suffix}"

    dropped_mask = hard_pass & ~result["accepted"].astype(bool)
    result.loc[dropped_mask, "final_rejection_reasons"] = result.loc[
        dropped_mask, "hard_rejection_reasons"
    ].fillna("").map(_append_top_fraction_policy)

    result.loc[result["accepted"], "final_rejection_reasons"] = result.loc[
        result["accepted"], "hard_rejection_reasons"
    ].fillna("")

    result["rejection_reasons"] = result["final_rejection_reasons"]
    return result
