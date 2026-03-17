"""Acceptance and hard rejection logic."""

from __future__ import annotations

from dataclasses import dataclass

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
    return AcceptanceResult(accepted=not reasons, reasons=reasons, final_weighted_score=weighted_score)


def compute_weighted_score(metrics: dict[str, float], config: ExperimentConfig) -> float:
    weights = config.evaluation.acceptance.weighted_score
    score = 0.0
    for metric_name, weight in weights.items():
        score += weight * float(metrics.get(metric_name, 0.0))
    return score
