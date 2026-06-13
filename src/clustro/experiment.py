"""Top-level experiment orchestration."""

from __future__ import annotations

import contextlib
import inspect
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from clustro.config.schema import ConsensusConfig, ExperimentConfig
from clustro.config.validators import load_experiment_config
from clustro.consensus.coassociation import build_coassociation_matrix
from clustro.consensus.consensus_fit import cluster_from_coassociation, fit_consensus
from clustro.consensus.weighting import compute_run_weights
from clustro.data.loaders import inspect_table, load_table
from clustro.data.preprocess_pipeline import preprocess_frame
from clustro.data.schema import DatasetSchema
from clustro.evaluation.acceptance import apply_acceptance_policy
from clustro.evaluation.ranking import rank_candidates
from clustro.interpretation.permutation import (
    build_correlation_groups,
    compute_cv_permutation_importance,
    compute_full_fit_permutation_importance,
    compute_grouped_permutation_importance,
)
from clustro.interpretation.profiling import (
    build_cluster_profiles,
    build_pairwise_cluster_contrasts,
)
from clustro.interpretation.shap_utils import compute_shap_summary
from clustro.interpretation.surrogate import fit_surrogate_model
from clustro.reporting.exports import (
    export_consensus_outputs,
    export_experiment_tables,
    export_report_bundle,
)
from clustro.reporting.tables import write_table
from clustro.search.compatibility import validate_candidate
from clustro.search.optuna_objective import optimize_family_candidates
from clustro.search.scheduler import (
    CandidateExecution,
    evaluate_candidate,
    evaluate_candidate_batch_ray,
    executions_to_frame,
)
from clustro.search.search_space import Candidate, generate_candidates
from clustro.tracking.artifact_registry import ArtifactRegistry
from clustro.tracking.mlflow_logger import MlflowLogger
from clustro.tracking.ray_monitor import maybe_init_ray
from clustro.utils.gpu import detect_gpu_status
from clustro.utils.hashing import dataframe_fingerprint, file_fingerprint, stable_hash
from clustro.utils.io import write_json, write_yaml
from clustro.utils.paths import ExperimentPaths, build_experiment_paths
from clustro.utils.random import set_global_seed


@contextlib.contextmanager
def _strict_numeric_context(deterministic_mode: str) -> Iterator[None]:
    if deterministic_mode != "strict":
        yield
        return
    try:
        import joblib
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1), joblib.parallel_backend("sequential", nested=True):
            yield
    except ImportError:
        yield


def _preprocessing_key(continuous_transform: object, categorical_encoding: object) -> str:
    return f"{continuous_transform}__{categorical_encoding}"


@dataclass(slots=True)
class Experiment:
    config: ExperimentConfig
    paths: ExperimentPaths
    registry: ArtifactRegistry

    @classmethod
    def from_yaml(cls, path: str | Path) -> Experiment:
        config = load_experiment_config(path)
        paths = build_experiment_paths(config.resolved_output_dir)
        registry = ArtifactRegistry(paths)
        return cls(config=config, paths=paths, registry=registry)

    @classmethod
    def from_output_dir(cls, path: str | Path) -> Experiment:
        root = Path(path).expanduser().resolve()
        config_snapshot = root / "state" / "config_snapshot.yaml"
        if not config_snapshot.exists():
            raise FileNotFoundError(f"No config snapshot found in {root}")
        return cls.from_yaml(config_snapshot)

    def validate(self) -> ExperimentConfig:
        return self.config

    def status(self) -> dict[str, object]:
        return {
            "run": self.registry.read_stage("run"),
            "consensus": self.registry.read_stage("consensus"),
            "interpretation": self.registry.read_stage("interpretation"),
            "report": self.registry.read_stage("report"),
        }

    def inspect_data(self) -> dict[str, object]:
        frame = load_table(self._data_path())
        return inspect_table(frame)

    def run(self) -> Experiment:
        with _strict_numeric_context(self.config.experiment.deterministic_mode):
            return self._run_impl()

    def _run_impl(self) -> Experiment:
        set_global_seed(self.config.experiment.random_seed)
        ray_enabled = maybe_init_ray(
            self.config.experiment.use_ray, n_jobs=self.config.experiment.n_jobs
        )
        gpu_status = detect_gpu_status(self.config.experiment.use_gpu_if_available)
        frame = load_table(self._data_path())
        preprocessing_cache: dict[str, Any] = {}
        invalid_transforms: dict[str, str] = {}
        for transform in self.config.preprocessing.continuous_transforms:
            for encoding in self.config.preprocessing.categorical_encoding:
                key = _preprocessing_key(transform, encoding)
                try:
                    preprocessing_cache[key] = preprocess_frame(
                        frame,
                        self.config,
                        continuous_transform=transform,
                        categorical_encoding=encoding,
                    )
                except ValueError as exc:
                    invalid_transforms[key] = str(exc)
        if not preprocessing_cache:
            raise RuntimeError(
                "No valid preprocessing transforms available for this dataset/config."
            )
        dataset_fingerprint = {
            "file": file_fingerprint(self._data_path()),
            "frame": dataframe_fingerprint(
                frame[
                    self.config.data.column_schema.continuous
                    + self.config.data.column_schema.binary
                    + self.config.data.column_schema.categorical
                    + self.config.data.column_schema.ordinal
                ]
            ),
        }
        experiment_id = stable_hash(
            {
                "experiment_name": self.config.experiment.name,
                "dataset": dataset_fingerprint,
                "config": self.config.model_dump(
                    mode="json",
                    exclude={"config_path", "resolved_data_path", "resolved_output_dir"},
                ),
            }
        )

        self._write_config_snapshot()
        self._write_manifest(
            {
                "experiment_id": experiment_id,
                "experiment_name": self.config.experiment.name,
                "dataset_path": str(self._data_path()),
                "output_dir": str(self.paths.root),
                "accelerator": {
                    "requested": gpu_status.requested,
                    "device": gpu_status.device,
                    "torch_available": gpu_status.torch_available,
                    "cuda_available": gpu_status.cuda_available,
                    "rapids_available": gpu_status.rapids_available,
                },
                "orchestration": {
                    "ray_enabled": ray_enabled,
                    "n_jobs": self.config.experiment.n_jobs,
                },
            }
        )

        candidates = generate_candidates(self.config, dataset_fingerprint)
        allowed: list[Candidate] = []
        rejected_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            transform_name = candidate.preprocessing["continuous_transform"]
            encoding_name = candidate.preprocessing["categorical_encoding"]
            preprocessing_key = _preprocessing_key(transform_name, encoding_name)
            if preprocessing_key in invalid_transforms:
                rejected_rows.append(
                    {
                        **self._candidate_registry_metadata(candidate),
                        "accepted": False,
                        "hard_filter_passed": False,
                        "hard_rejection_reasons": str(invalid_transforms[preprocessing_key]),
                        "accepted_before_top_fraction": False,
                        "final_rejection_reasons": str(invalid_transforms[preprocessing_key]),
                        "search_stage": "compatibility_rejected",
                        "rejection_reasons": invalid_transforms[preprocessing_key],
                    }
                )
                continue
            candidate_matrix = preprocessing_cache[preprocessing_key].evaluation_matrix
            decision = validate_candidate(
                candidate,
                n_rows=candidate_matrix.shape[0],
                n_features=candidate_matrix.shape[1],
            )
            if decision.allowed:
                allowed.append(candidate)
            else:
                rejected_rows.append(
                    {
                        **self._candidate_registry_metadata(candidate),
                        "accepted": False,
                        "hard_filter_passed": False,
                        "hard_rejection_reasons": ";".join(decision.reasons),
                        "accepted_before_top_fraction": False,
                        "final_rejection_reasons": ";".join(decision.reasons),
                        "search_stage": "compatibility_rejected",
                        "rejection_reasons": ";".join(decision.reasons),
                    }
                )

        completed_frame = self._read_completed_candidate_frame()
        completed_ids = (
            set(completed_frame["candidate_id"].tolist()) if not completed_frame.empty else set()
        )
        allowed_ids = {candidate.candidate_id for candidate in allowed}
        completed_ids &= allowed_ids
        if completed_ids:
            completed_frame = completed_frame.loc[
                completed_frame["candidate_id"].isin(completed_ids)
            ].reset_index(drop=True)
        self.registry.mark_stage(
            "run",
            {
                "completed": False,
                "experiment_id": experiment_id,
                "candidate_count": len(candidates),
                "allowed_count": len(allowed),
                "completed_candidate_count": len(completed_ids),
            },
        )

        with MlflowLogger(self.config.experiment.use_mlflow) as mlflow:
            mlflow.start_run(self.config.experiment.name, tags={"stage": "run"})
            mlflow.log_params({"candidate_count": len(candidates), "allowed_count": len(allowed)})
            executions: list[CandidateExecution] = []
            if self.config.search.optuna.enabled:
                optuna_candidates = [
                    candidate
                    for candidate in allowed
                    if candidate.candidate_id not in completed_ids
                ]
                families = sorted({candidate.family for candidate in optuna_candidates})
                for family in families:
                    family_candidates = [
                        candidate for candidate in optuna_candidates if candidate.family == family
                    ]
                    family_executions = optimize_family_candidates(
                        family=family,
                        candidates=family_candidates,
                        matrices=preprocessing_cache,
                        config=self.config,
                        study_name=f"{self.config.experiment.name}_{family}",
                        output_dir=self.paths.root,
                        dataset_fingerprint=dataset_fingerprint,
                        raw_frame=frame
                        if self.config.search.stability_mode == "full_pipeline"
                        else None,
                    )
                    for execution in family_executions:
                        executions.append(execution)
                        preprocessed = preprocessing_cache[
                            _preprocessing_key(
                                execution.candidate.preprocessing["continuous_transform"],
                                execution.candidate.preprocessing["categorical_encoding"],
                            )
                        ]
                        self._persist_candidate_outputs(execution, preprocessed.row_metadata)
                        completed_ids.add(execution.candidate.candidate_id)
                        self.registry.mark_stage(
                            "run",
                            {
                                "completed": False,
                                "experiment_id": experiment_id,
                                "candidate_count": len(candidates),
                                "allowed_count": len(allowed),
                                "completed_candidate_count": len(completed_ids),
                            },
                        )
            else:
                for preprocessing_key, preprocessed in preprocessing_cache.items():
                    transform_candidates = [
                        candidate
                        for candidate in allowed
                        if _preprocessing_key(
                            candidate.preprocessing["continuous_transform"],
                            candidate.preprocessing["categorical_encoding"],
                        )
                        == preprocessing_key
                        and candidate.candidate_id not in completed_ids
                    ]
                    if not transform_candidates:
                        continue
                    if ray_enabled:
                        ray_kwargs = (
                            {"raw_frame": frame}
                            if self.config.search.stability_mode == "full_pipeline"
                            and "raw_frame"
                            in inspect.signature(evaluate_candidate_batch_ray).parameters
                            else {}
                        )
                        transform_executions = evaluate_candidate_batch_ray(
                            transform_candidates,
                            preprocessed.evaluation_matrix,
                            self.config,
                            **ray_kwargs,
                        )
                        for execution in transform_executions:
                            executions.append(execution)
                            self._persist_candidate_outputs(execution, preprocessed.row_metadata)
                            completed_ids.add(execution.candidate.candidate_id)
                            self.registry.mark_stage(
                                "run",
                                {
                                    "completed": False,
                                    "experiment_id": experiment_id,
                                    "candidate_count": len(candidates),
                                    "allowed_count": len(allowed),
                                    "completed_candidate_count": len(completed_ids),
                                },
                            )
                    else:
                        for candidate in transform_candidates:
                            candidate_kwargs = (
                                {"raw_frame": frame}
                                if self.config.search.stability_mode == "full_pipeline"
                                and "raw_frame" in inspect.signature(evaluate_candidate).parameters
                                else {}
                            )
                            execution = evaluate_candidate(
                                candidate,
                                preprocessed.evaluation_matrix,
                                self.config,
                                **candidate_kwargs,
                            )
                            executions.append(execution)
                            self._persist_candidate_outputs(execution, preprocessed.row_metadata)
                            completed_ids.add(execution.candidate.candidate_id)
                            self.registry.mark_stage(
                                "run",
                                {
                                    "completed": False,
                                    "experiment_id": experiment_id,
                                    "candidate_count": len(candidates),
                                    "allowed_count": len(allowed),
                                    "completed_candidate_count": len(completed_ids),
                                },
                            )

        new_candidate_frame = executions_to_frame(executions)
        candidate_frame = pd.concat(
            [completed_frame, new_candidate_frame], ignore_index=True, sort=False
        )
        candidate_frame = self._normalize_candidate_frame(candidate_frame)
        candidate_frame = apply_acceptance_policy(candidate_frame, self.config)
        self._refresh_candidate_acceptance_summaries(candidate_frame)

        if rejected_rows:
            rejected_frame = pd.concat(
                [pd.DataFrame(rejected_rows), candidate_frame.loc[~candidate_frame["accepted"]]],
                ignore_index=True,
                sort=False,
            )
        else:
            rejected_frame = candidate_frame.loc[~candidate_frame["accepted"]].copy()
        accepted_frame = rank_candidates(candidate_frame.loc[candidate_frame["accepted"]].copy())
        registry_frame = pd.concat(
            [candidate_frame, pd.DataFrame(rejected_rows)], ignore_index=True, sort=False
        )

        export_experiment_tables(
            candidate_registry=registry_frame,
            accepted=accepted_frame,
            rejected=rejected_frame,
            output_dir=self.paths.root,
        )
        self._write_summaries(registry_frame)
        self.registry.mark_stage(
            "run",
            {
                "completed": True,
                "experiment_id": experiment_id,
                "candidate_count": len(candidates),
                "allowed_count": len(allowed),
                "completed_candidate_count": len(completed_ids),
            },
        )

        if not accepted_frame.empty:
            self.build_consensus()
            self.run_interpretation()
            self.report()
        return self

    def resume(self) -> Experiment:
        run_stage = self.registry.read_stage("run")
        if run_stage is None or not bool(run_stage.get("completed", False)):
            return self.run()
        consensus_stage = self.registry.read_stage("consensus")
        if consensus_stage is None:
            self.build_consensus()
        interpretation_stage = self.registry.read_stage("interpretation")
        if interpretation_stage is None:
            self.run_interpretation()
        report_stage = self.registry.read_stage("report")
        if report_stage is None:
            self.report()
        return self

    def build_consensus(self) -> Experiment:
        with _strict_numeric_context(self.config.experiment.deterministic_mode):
            return self._build_consensus_impl()

    def _build_consensus_impl(self) -> Experiment:
        accepted = self._read_frame(self.registry.accepted_candidates_path())
        if accepted.empty:
            raise RuntimeError("Cannot build consensus without accepted candidates.")
        # Deterministic row order for weighted sums / consensus (matches exported ranks).
        accepted = rank_candidates(accepted.copy())

        label_runs: list[np.ndarray] = []
        for candidate_id in accepted["candidate_id"]:
            label_frame = self._read_frame(
                self.registry.candidate_file(candidate_id, "final_labels.csv")
            )
            label_runs.append(label_frame["label"].to_numpy(dtype=int))

        weights = compute_run_weights(accepted, self.config)
        coassociation = build_coassociation_matrix(
            label_runs,
            weights,
            storage=self.config.consensus.coassociation_storage,
            max_dense_n=self.config.consensus.max_dense_n,
        )
        target_k = self._target_k(accepted, coassociation)
        base_label_frame = self._read_frame(
            self.registry.candidate_file(accepted.iloc[0]["candidate_id"], "final_labels.csv")
        )
        row_ids = base_label_frame["row_id"].astype(str).tolist()
        result = fit_consensus(
            label_runs,
            weights,
            row_ids,
            target_k=target_k,
            method=self.config.consensus.consensus_method,
            bootstrap_repeats=self.config.consensus.uncertainty.bootstrap_repeats,
            random_seed=self.config.experiment.random_seed,
            coassociation=coassociation,
            storage=self.config.consensus.coassociation_storage,
            max_dense_n=self.config.consensus.max_dense_n,
            ambiguous_top2_gap_threshold=(
                self.config.consensus.uncertainty.ambiguous_top2_gap_threshold
            ),
            ambiguous_entropy_quantile=(
                self.config.consensus.uncertainty.ambiguous_entropy_quantile
            ),
        )
        labels = base_label_frame.drop(columns=["label"]).copy()
        labels["consensus_label"] = result.labels
        uncertainty = result.uncertainty.merge(
            base_label_frame.drop(columns=["label"]),
            on="row_id",
            how="left",
        )
        export_consensus_outputs(
            labels=labels,
            uncertainty=uncertainty,
            cluster_summary=result.cluster_summary,
            bootstrap_stability=result.bootstrap_stability,
            output_dir=self.paths.root,
        )
        pd.DataFrame(result.coassociation).to_parquet(
            self.registry.consensus_file("coassociation_matrix.parquet"), index=False
        )
        self.registry.mark_stage("consensus", {"completed": True, "target_k": target_k})
        return self

    def report(self) -> Experiment:
        candidate_registry = self._read_frame(self.registry.candidate_registry_path())
        export_report_bundle(candidate_registry, self.paths.root)
        self.registry.mark_stage("report", {"completed": True})
        return self

    def run_interpretation(self) -> Experiment:
        consensus_path = self.paths.root / "consensus_labels.csv"
        if not consensus_path.exists():
            raise RuntimeError("Consensus labels are required before interpretation can run.")

        frame = load_table(self._data_path())
        feature_space = self._resolve_interpretation_feature_space()
        preprocessed = preprocess_frame(
            frame,
            self.config,
            continuous_transform=str(feature_space["continuous_transform"]),
            categorical_encoding=str(feature_space["categorical_encoding"]),
        )
        consensus = pd.read_csv(consensus_path)
        labels = consensus["consensus_label"].to_numpy(dtype=int)
        self._export_visualization_embeddings(preprocessed.evaluation_matrix, consensus)
        interpretation_dir = self.paths.root / "interpretation"
        write_json(interpretation_dir / "interpretation_feature_space.json", feature_space)

        result = fit_surrogate_model(
            preprocessed.evaluation_matrix,
            labels,
            preprocessed.feature_names,
            self.config.interpretation,
            random_seed=self.config.experiment.random_seed,
        )
        write_table(result.cv_metrics, interpretation_dir / "surrogate_cv_metrics.csv")
        write_table(result.confusion, interpretation_dir / "surrogate_confusion_matrix.csv")
        write_json(interpretation_dir / "surrogate_summary.json", result.mean_metrics)
        if result.warning is not None:
            write_json(
                interpretation_dir / "interpretation_warning.json", {"warning": result.warning}
            )

        correlation_groups = build_correlation_groups(
            preprocessed.evaluation_matrix,
            result.feature_names,
            threshold=self.config.interpretation.grouped_correlation_threshold,
        )
        write_table(correlation_groups, interpretation_dir / "correlation_groups.csv")

        if self.config.interpretation.use_permutation_importance:
            permutation = compute_full_fit_permutation_importance(
                result.estimator,
                preprocessed.evaluation_matrix,
                labels,
                result.feature_names,
                random_seed=self.config.experiment.random_seed,
            )
            permutation["importance_type"] = "full_fit_exploratory"
            write_table(
                permutation,
                interpretation_dir / "permutation_importance_full_fit_exploratory.csv",
            )
            permutation_cv = compute_cv_permutation_importance(
                preprocessed.evaluation_matrix,
                labels,
                result.feature_names,
                self.config.interpretation,
                random_seed=self.config.experiment.random_seed,
            )
            write_table(permutation_cv, interpretation_dir / "permutation_importance_cv.csv")
            write_table(
                permutation_cv.head(self.config.interpretation.top_n_features),
                interpretation_dir / "permutation_importance_top_features.csv",
            )
            grouped_permutation = compute_grouped_permutation_importance(
                result.estimator,
                preprocessed.evaluation_matrix,
                labels,
                result.feature_names,
                correlation_groups,
                random_seed=self.config.experiment.random_seed,
            )
            write_table(
                grouped_permutation, interpretation_dir / "grouped_permutation_importance.csv"
            )
            write_table(
                grouped_permutation.head(self.config.interpretation.top_n_features),
                interpretation_dir / "grouped_permutation_importance_top_features.csv",
            )

        if self.config.interpretation.use_shap:
            try:
                shap_summary, shap_values, shap_by_class = compute_shap_summary(
                    result.estimator,
                    preprocessed.evaluation_matrix,
                    result.feature_names,
                    random_seed=self.config.experiment.random_seed,
                    row_ids=preprocessed.row_ids,
                )
                write_table(shap_summary, interpretation_dir / "shap_summary.csv")
                write_table(shap_values, interpretation_dir / "shap_values.parquet")
                write_table(
                    shap_summary.head(self.config.interpretation.top_n_features),
                    interpretation_dir / "shap_summary_top_features.csv",
                )
                if not shap_by_class.empty:
                    write_table(shap_by_class, interpretation_dir / "shap_by_class.csv")
                    top_by_class = (
                        shap_by_class.groupby("class_index", group_keys=False)
                        .head(self.config.interpretation.top_n_features)
                        .reset_index(drop=True)
                    )
                    write_table(top_by_class, interpretation_dir / "shap_by_class_top_features.csv")
                write_json(interpretation_dir / "shap_status.json", {"status": "completed"})
            except RuntimeError as exc:
                write_json(
                    interpretation_dir / "shap_status.json",
                    {"status": "skipped", "reason": str(exc)},
                )

        profiles = build_cluster_profiles(
            frame[
                self.config.data.column_schema.continuous
                + self.config.data.column_schema.binary
                + self.config.data.column_schema.categorical
                + self.config.data.column_schema.ordinal
            ].copy(),
            consensus["consensus_label"],
            DatasetSchema.from_config(self.config.data.column_schema),
        )
        write_table(profiles, interpretation_dir / "cluster_profiles.csv")
        contrasts = build_pairwise_cluster_contrasts(
            frame[
                self.config.data.column_schema.continuous
                + self.config.data.column_schema.binary
                + self.config.data.column_schema.categorical
                + self.config.data.column_schema.ordinal
            ].copy(),
            consensus["consensus_label"],
            DatasetSchema.from_config(self.config.data.column_schema),
        )
        write_table(contrasts, interpretation_dir / "pairwise_cluster_contrasts.csv")
        self.registry.mark_stage("interpretation", {"completed": True})
        return self

    def export_manuscript_bundle(self) -> Experiment:
        return self.report()

    def _target_k(self, accepted: pd.DataFrame, coassociation: np.ndarray) -> int:
        counts = accepted["n_clusters"].round().astype(int)
        if counts.empty:
            raise RuntimeError("Cannot infer consensus cluster count from an empty accepted set.")
        if self.config.consensus.final_k_strategy == "data_driven":
            return self._select_data_driven_k(
                counts.to_numpy(dtype=int), coassociation, self.config.consensus
            )
        weighted = (
            accepted.groupby(counts)["final_weighted_score"].sum().sort_values(ascending=False)
        )
        return int(weighted.index[0])

    def _persist_candidate_outputs(
        self, execution: CandidateExecution, row_metadata: pd.DataFrame
    ) -> None:
        candidate_dir = self.registry.candidate_dir(execution.candidate.candidate_id)
        write_json(candidate_dir / "config_snapshot.json", execution.candidate.to_dict())
        write_json(candidate_dir / "metrics_summary.json", execution.metrics)
        write_json(
            self.registry.candidate_result_summary_file(execution.candidate.candidate_id),
            self._execution_row(execution),
        )
        final_labels = row_metadata.copy()
        final_labels["label"] = execution.labels
        write_table(final_labels, candidate_dir / "final_labels.csv")
        if execution.seed_label_runs:
            seed_frame = row_metadata.copy()
            for index, labels in enumerate(execution.seed_label_runs):
                seed_frame[f"seed_run_{index}"] = labels
            write_table(seed_frame, candidate_dir / "per_seed_labels.parquet")
        if execution.perturbation_label_runs:
            rows: list[pd.DataFrame] = []
            for index, run in enumerate(execution.perturbation_label_runs):
                labels = row_metadata.iloc[run.indices].reset_index(drop=True).copy()
                labels["perturbation_run"] = index
                labels["perturbation_kind"] = run.kind
                labels["original_row_index"] = run.indices
                labels["label"] = run.labels
                rows.append(labels)
            write_table(
                pd.concat(rows, ignore_index=True),
                candidate_dir / "per_perturbation_labels.parquet",
            )

    def _execution_row(self, execution: CandidateExecution) -> dict[str, object]:
        return {
            **self._candidate_registry_metadata(execution.candidate),
            "accepted": execution.accepted,
            "hard_filter_passed": execution.accepted,
            "hard_rejection_reasons": ";".join(execution.rejection_reasons),
            "accepted_before_top_fraction": execution.accepted,
            "final_rejection_reasons": ";".join(execution.rejection_reasons),
            "search_stage": execution.search_stage,
            "rejection_reasons": ";".join(execution.rejection_reasons),
            **execution.metrics,
        }

    def _resolve_interpretation_feature_space(self) -> dict[str, object]:
        mode = self.config.interpretation.feature_space
        if mode == "original_imputed_scaled":
            return {
                "feature_space": mode,
                "continuous_transform": self.config.interpretation.continuous_transform,
                "categorical_encoding": self.config.interpretation.categorical_encoding,
                "source_candidate_id": None,
                "rationale": (
                    "Predefined interpretation feature space independent of clustering search."
                ),
            }

        accepted = self._read_frame(self.registry.accepted_candidates_path())
        if accepted.empty:
            raise RuntimeError(
                "Accepted candidates are required for interpretation feature-space selection."
            )

        if mode == "best_candidate_preprocessing":
            ranked = rank_candidates(accepted.copy())
            row = ranked.iloc[0]
            return {
                "feature_space": mode,
                "continuous_transform": row["continuous_transform"],
                "categorical_encoding": row["categorical_encoding"],
                "source_candidate_id": row["candidate_id"],
                "rationale": "Using preprocessing from the top-ranked accepted candidate.",
            }

        grouped = (
            accepted.groupby(["continuous_transform", "categorical_encoding"], dropna=False)
            .size()
            .sort_values(ascending=False)
        )
        transform, encoding = grouped.index[0]
        return {
            "feature_space": mode,
            "continuous_transform": transform,
            "categorical_encoding": encoding,
            "source_candidate_id": None,
            "rationale": "Using the most frequent preprocessing among accepted candidates.",
        }

    def _candidate_registry_metadata(self, candidate: Candidate) -> dict[str, object]:
        return {
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "continuous_transform": candidate.preprocessing.get("continuous_transform"),
            "categorical_encoding": candidate.preprocessing.get("categorical_encoding"),
            "representation_name": candidate.representation["name"],
            "representation_params_json": json.dumps(
                candidate.representation.get("params", {}), sort_keys=True
            ),
            "clustering_name": candidate.clustering["name"],
            "clustering_params_json": json.dumps(
                candidate.clustering.get("params", {}), sort_keys=True
            ),
        }

    def _read_completed_candidate_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        if not self.paths.candidates_dir.exists():
            return pd.DataFrame()
        for candidate_dir in sorted(self.paths.candidates_dir.iterdir()):
            if not candidate_dir.is_dir():
                continue
            summary_path = self.registry.candidate_result_summary_file(candidate_dir.name)
            if not summary_path.exists():
                continue
            rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
        return pd.DataFrame(rows)

    def _normalize_candidate_frame(self, candidate_frame: pd.DataFrame) -> pd.DataFrame:
        if candidate_frame.empty:
            return pd.DataFrame(
                columns=[
                    "candidate_id",
                    "family",
                    "representation_name",
                    "clustering_name",
                    "accepted",
                    "rejection_reasons",
                ]
            )
        candidate_frame = (
            candidate_frame.drop_duplicates(subset=["candidate_id"], keep="last")
            .sort_values("candidate_id")
            .reset_index(drop=True)
        )
        return candidate_frame

    def _refresh_candidate_acceptance_summaries(self, candidate_frame: pd.DataFrame) -> None:
        if candidate_frame.empty:
            return
        for _, row in candidate_frame.iterrows():
            summary_path = self.registry.candidate_result_summary_file(str(row["candidate_id"]))
            if not summary_path.exists():
                continue
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            payload["accepted"] = bool(row["accepted"])
            payload["accepted_before_top_fraction"] = bool(row["accepted_before_top_fraction"])
            payload["hard_filter_passed"] = bool(row["hard_filter_passed"])
            payload["hard_rejection_reasons"] = str(row.get("hard_rejection_reasons") or "")
            payload["final_rejection_reasons"] = str(row.get("final_rejection_reasons") or "")
            payload["rejection_reasons"] = payload["final_rejection_reasons"]
            write_json(summary_path, payload)

    def _select_data_driven_k(
        self,
        counts: np.ndarray,
        coassociation: np.ndarray,
        consensus: ConsensusConfig,
    ) -> int:
        candidate_ks = sorted({int(count) for count in counts if int(count) >= 2})
        if len(candidate_ks) == 1:
            return candidate_ks[0]

        distance = 1.0 - coassociation
        best_k = candidate_ks[0]
        best_score = float("-inf")
        for candidate_k in candidate_ks:
            labels = cluster_from_coassociation(
                coassociation,
                target_k=candidate_k,
                method=consensus.consensus_method,
                random_seed=self.config.experiment.random_seed,
            )
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(distance, labels, metric="precomputed"))
            if score > best_score:
                best_score = score
                best_k = candidate_k
        return best_k

    def _export_visualization_embeddings(self, matrix: np.ndarray, consensus: pd.DataFrame) -> None:
        if len(consensus) < 3:
            return
        rng = np.random.default_rng(self.config.experiment.random_seed)
        max_rows = min(len(consensus), 2000)
        indices = (
            np.sort(rng.choice(len(consensus), size=max_rows, replace=False))
            if len(consensus) > max_rows
            else np.arange(len(consensus))
        )
        sample_matrix = matrix[indices]
        sample_labels = consensus.iloc[indices].reset_index(drop=True)

        perplexity = max(2, min(30, (len(sample_labels) - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=250,
            random_state=self.config.experiment.random_seed,
        )
        tsne_coords = tsne.fit_transform(sample_matrix)
        tsne_frame = sample_labels.copy()
        tsne_frame["x"] = tsne_coords[:, 0]
        tsne_frame["y"] = tsne_coords[:, 1]
        tsne_frame["embedding_method"] = "tsne"
        write_table(tsne_frame, self.paths.reports_dir / "final_embedding_plot_data.csv")

    def _write_summaries(self, registry_frame: pd.DataFrame) -> None:
        if registry_frame.empty:
            return
        runtime_summary = (
            registry_frame.groupby("family", as_index=False)["runtime_seconds"]
            .mean()
            .rename(columns={"runtime_seconds": "mean_runtime_seconds"})
        )
        method_summary = registry_frame.groupby("family", as_index=False).agg(
            accepted_count=("accepted", "sum"),
            candidate_count=("accepted", "count"),
        )
        runtime_summary.to_csv(self.registry.runtime_summary_path(), index=False)
        method_summary.to_csv(self.registry.method_family_summary_path(), index=False)

    def _write_manifest(self, payload: dict[str, object]) -> None:
        write_json(self.registry.manifest_path(), payload)

    def _write_config_snapshot(self) -> None:
        payload = self.config.model_dump(
            mode="json",
            exclude={"config_path", "resolved_data_path", "resolved_output_dir"},
        )
        payload["data"]["path"] = str(self._data_path())
        payload["experiment"]["output_dir"] = str(self.paths.root)
        write_yaml(
            self.registry.state_file("config_snapshot.yaml"),
            payload,
        )

    def _data_path(self) -> Path:
        if self.config.resolved_data_path is None:
            raise RuntimeError("Resolved data path missing from configuration.")
        return self.config.resolved_data_path

    def _read_frame(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported table format: {path}")
