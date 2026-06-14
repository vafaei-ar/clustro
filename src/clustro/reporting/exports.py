"""High-level export routines."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clustro.config.schema import ReportingConfig
from clustro.reporting.figures import (
    export_cluster_profile_heatmap,
    export_coassociation_heatmap,
    export_embedding_scatter,
    export_feature_importance_bar,
    export_metric_heatmap,
    export_quality_vs_stability,
    export_search_flow_diagram,
    export_uncertainty_distribution,
)
from clustro.reporting.manuscript_bundle import populate_manuscript_bundle
from clustro.reporting.tables import write_table
from clustro.utils.io import write_json


def export_experiment_tables(
    *,
    candidate_registry: pd.DataFrame,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    output_dir: Path,
) -> None:
    write_table(candidate_registry, output_dir / "candidate_registry.parquet")
    write_table(accepted, output_dir / "accepted_candidates.parquet")
    write_table(rejected, output_dir / "rejected_candidates.parquet")


def export_consensus_outputs(
    *,
    labels: pd.DataFrame,
    uncertainty: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    bootstrap_stability: pd.DataFrame,
    output_dir: Path,
) -> None:
    write_table(labels, output_dir / "consensus_labels.csv")
    write_table(uncertainty, output_dir / "consensus_uncertainty.csv")
    # Primary name: columns carry consensus support scores, not calibrated probabilities.
    write_table(uncertainty, output_dir / "consensus_support.parquet")
    # Deprecated alias kept for one release; will be removed in the next major version.
    write_table(uncertainty, output_dir / "consensus_soft_membership.parquet")
    write_table(cluster_summary, output_dir / "consensus_cluster_summary.csv")
    write_table(bootstrap_stability, output_dir / "consensus_bootstrap_stability.csv")


def export_report_bundle(
    candidate_registry: pd.DataFrame,
    output_dir: Path,
    *,
    reporting_config: ReportingConfig | None = None,
) -> None:
    cfg = reporting_config if reporting_config is not None else ReportingConfig()
    report_dir = output_dir / "reports"
    # candidate_metrics.csv is the primary report entry point — always written.
    write_table(candidate_registry, report_dir / "candidate_metrics.csv")
    plot_frame = candidate_registry.dropna(subset=["silhouette", "ari_seed"])
    if not plot_frame.empty:
        if cfg.generate_figures:
            export_quality_vs_stability(plot_frame, report_dir / "quality_vs_stability.png")
        if cfg.generate_tables:
            write_table(
                plot_frame[
                    _existing_columns(
                        plot_frame,
                        [
                            "candidate_id",
                            "family",
                            "representation_name",
                            "clustering_name",
                            "silhouette",
                            "ari_seed",
                            "final_weighted_score",
                            "accepted",
                        ],
                    )
                ],
                report_dir / "quality_vs_stability.csv",
            )
    if not candidate_registry.empty:
        search_flow = _build_search_flow_frame(candidate_registry)
        if cfg.generate_tables:
            write_json(
                report_dir / "search_flow.json",
                dict(zip(search_flow["stage"], search_flow["count"], strict=True)),
            )
            write_table(search_flow, report_dir / "search_flow.csv")
        if cfg.generate_figures:
            export_search_flow_diagram(search_flow, report_dir / "search_flow_diagram.png")
        accepted = candidate_registry.loc[candidate_registry["accepted"]].copy()
        if not accepted.empty:
            heatmap_frame = accepted[
                _existing_columns(
                    accepted,
                    [
                        "candidate_id",
                        "family",
                        "representation_name",
                        "clustering_name",
                        "silhouette",
                        "davies_bouldin",
                        "calinski_harabasz",
                        "ari_seed",
                        "nmi_seed",
                        "mean_cluster_jaccard",
                        "cluster_balance",
                        "final_weighted_score",
                    ],
                )
            ]
            if cfg.generate_tables:
                write_table(heatmap_frame, report_dir / "accepted_candidate_heatmap.csv")
            if cfg.generate_figures:
                export_metric_heatmap(
                    heatmap_frame,
                    report_dir / "accepted_candidate_metric_heatmap.png",
                )
    if cfg.generate_tables:
        _copy_root_summary(
            output_dir / "method_family_summary.csv",
            report_dir / "method_family_acceptance_summary.csv",
        )
    _export_consensus_matrix_plot_data(output_dir, report_dir, cfg=cfg)
    _export_cluster_size_confidence(output_dir, report_dir, cfg=cfg)
    _export_feature_importance(output_dir, report_dir, cfg=cfg)
    _export_clinical_profile(output_dir, report_dir, cfg=cfg)
    _export_embedding_plot(output_dir, report_dir, cfg=cfg)
    if cfg.manuscript_bundle:
        populate_manuscript_bundle(output_dir)


def _build_search_flow_frame(candidate_registry: pd.DataFrame) -> pd.DataFrame:
    accepted = candidate_registry.get("accepted", pd.Series(False, index=candidate_registry.index))
    accepted = accepted.fillna(False).astype(bool)
    stage = candidate_registry.get("search_stage", pd.Series("", index=candidate_registry.index))
    reasons_col = (
        "final_rejection_reasons"
        if "final_rejection_reasons" in candidate_registry.columns
        else "rejection_reasons"
    )
    reasons = candidate_registry.get(
        reasons_col, pd.Series("", index=candidate_registry.index)
    ).fillna("")
    accepted_before = candidate_registry.get(
        "accepted_before_top_fraction", pd.Series(False, index=candidate_registry.index)
    )
    accepted_before = accepted_before.fillna(False).astype(bool)
    top_fraction_rejected = reasons.astype(str).str.contains("outside_top_fraction_policy")
    compatibility_rejected = stage.eq("compatibility_rejected")
    pilot_pruned = stage.eq("pilot_pruned") | reasons.astype(str).str.contains("optuna_pruned")
    full_evaluated = stage.eq("full_evaluated")
    hard_rejected = full_evaluated & ~accepted_before & ~top_fraction_rejected
    return pd.DataFrame(
        [
            {"stage": "generated_total", "count": int(len(candidate_registry))},
            {"stage": "compatibility_rejected", "count": int(compatibility_rejected.sum())},
            {"stage": "pilot_pruned", "count": int(pilot_pruned.sum())},
            {"stage": "full_evaluated", "count": int(full_evaluated.sum())},
            {"stage": "hard_rejected", "count": int(hard_rejected.sum())},
            {"stage": "accepted_before_top_fraction", "count": int(accepted_before.sum())},
            {"stage": "accepted_final", "count": int(accepted.sum())},
            {"stage": "top_fraction_rejected", "count": int(top_fraction_rejected.sum())},
            {"stage": "consensus_used", "count": int(accepted.sum())},
        ]
    )


def _export_consensus_matrix_plot_data(
    output_dir: Path, report_dir: Path, *, cfg: ReportingConfig
) -> None:
    source = output_dir / "consensus" / "coassociation_matrix.parquet"
    if not source.exists():
        return
    frame = pd.read_parquet(source)
    if cfg.generate_tables:
        write_table(frame, report_dir / "consensus_matrix_plot_data.parquet")
    if cfg.generate_figures:
        export_coassociation_heatmap(frame, report_dir / "coassociation_matrix_heatmap.png")


def _export_cluster_size_confidence(
    output_dir: Path, report_dir: Path, *, cfg: ReportingConfig
) -> None:
    cluster_summary_path = output_dir / "consensus_cluster_summary.csv"
    uncertainty_path = output_dir / "consensus_uncertainty.csv"
    if not cluster_summary_path.exists() or not uncertainty_path.exists():
        return
    cluster_summary = pd.read_csv(cluster_summary_path)
    uncertainty = pd.read_csv(uncertainty_path)
    summary = (
        uncertainty.groupby("consensus_label", as_index=False)
        .agg(
            mean_entropy=("entropy", "mean"),
            median_entropy=("entropy", "median"),
            mean_consensus_support_gap=("consensus_support_gap", "mean"),
            ambiguous_fraction=("ambiguous", "mean"),
        )
        .merge(cluster_summary, on="consensus_label", how="left")
    )
    if cfg.generate_tables:
        write_table(summary, report_dir / "cluster_size_confidence.csv")
    if cfg.generate_figures:
        export_uncertainty_distribution(
            uncertainty, report_dir / "uncertainty_distribution_by_cluster.png"
        )


def _export_feature_importance(
    output_dir: Path, report_dir: Path, *, cfg: ReportingConfig
) -> None:
    candidates = [
        output_dir / "interpretation" / "permutation_importance_cv.csv",
        output_dir / "interpretation" / "permutation_importance_full_fit_exploratory.csv",
        output_dir / "interpretation" / "shap_summary.csv",
    ]
    for source in candidates:
        if source.exists():
            frame = pd.read_csv(source)
            if cfg.generate_tables:
                write_table(frame, report_dir / "feature_importance.csv")
            if cfg.generate_figures:
                export_feature_importance_bar(frame, report_dir / "feature_importance_top.png")
            return


def _export_clinical_profile(
    output_dir: Path, report_dir: Path, *, cfg: ReportingConfig
) -> None:
    source = output_dir / "interpretation" / "cluster_profiles.csv"
    if not source.exists():
        return
    frame = pd.read_csv(source)
    if cfg.generate_tables:
        write_table(frame, report_dir / "clinical_profile_heatmap.csv")
    if cfg.generate_figures:
        export_cluster_profile_heatmap(frame, report_dir / "clinical_profile_heatmap.png")


def _export_embedding_plot(
    output_dir: Path, report_dir: Path, *, cfg: ReportingConfig
) -> None:
    source = output_dir / "reports" / "final_embedding_plot_data.csv"
    if not source.exists():
        return
    frame = pd.read_csv(source)
    if cfg.generate_figures:
        export_embedding_scatter(frame, report_dir / "final_embedding_scatter.png")


def _copy_root_summary(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    frame = pd.read_csv(source)
    write_table(frame, destination)


def _existing_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]
