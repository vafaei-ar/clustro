"""Pruning heuristics for early candidate rejection."""

from __future__ import annotations


def should_prune(metrics: dict[str, float], *, runtime_seconds: float, runtime_cap_seconds: float = 300.0) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics.get("n_clusters", 0.0) <= 1:
        reasons.append("single_cluster")
    if metrics.get("tiny_cluster_fraction", 0.0) > 0.5:
        reasons.append("too_many_tiny_clusters")
    if metrics.get("silhouette", 0.0) < -0.05:
        reasons.append("silhouette_too_low")
    if metrics.get("ari_seed", 1.0) < 0.0:
        reasons.append("seed_stability_too_low")
    if metrics.get("noise_fraction", 0.0) > 0.5:
        reasons.append("noise_fraction_excessive")
    if runtime_seconds > runtime_cap_seconds:
        reasons.append("runtime_cap_exceeded")
    return (len(reasons) > 0, reasons)
