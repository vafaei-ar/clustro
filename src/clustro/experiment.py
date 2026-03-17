"""Top-level experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clustro.config.schema import ExperimentConfig
from clustro.config.validators import load_experiment_config
from clustro.consensus.consensus_fit import fit_consensus
from clustro.consensus.weighting import compute_run_weights
from clustro.data.loaders import inspect_table, load_table
from clustro.data.preprocess_pipeline import preprocess_frame
from clustro.data.schema import DatasetSchema
from clustro.evaluation.ranking import rank_candidates
from clustro.interpretation.permutation import compute_permutation_importance
from clustro.interpretation.profiling import build_cluster_profiles
from clustro.interpretation.shap_utils import compute_shap_summary
from clustro.interpretation.surrogate import fit_surrogate_model
from clustro.reporting.exports import export_consensus_outputs, export_experiment_tables, export_report_bundle
from clustro.reporting.tables import write_table
from clustro.search.compatibility import validate_candidate
from clustro.search.scheduler import CandidateExecution, evaluate_candidate_batch, executions_to_frame
from clustro.search.search_space import Candidate, generate_candidates
from clustro.tracking.artifact_registry import ArtifactRegistry
from clustro.tracking.mlflow_logger import MlflowLogger
from clustro.tracking.ray_monitor import maybe_init_ray
from clustro.utils.hashing import dataframe_fingerprint, file_fingerprint, stable_hash
from clustro.utils.gpu import detect_gpu_status
from clustro.utils.io import write_json, write_yaml
from clustro.utils.paths import ExperimentPaths, build_experiment_paths
from clustro.utils.random import set_global_seed


@dataclass(slots=True)
class Experiment:
    config: ExperimentConfig
    paths: ExperimentPaths
    registry: ArtifactRegistry

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Experiment":
        config = load_experiment_config(path)
        paths = build_experiment_paths(config.resolved_output_dir)
        registry = ArtifactRegistry(paths)
        return cls(config=config, paths=paths, registry=registry)

    @classmethod
    def from_output_dir(cls, path: str | Path) -> "Experiment":
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

    def run(self) -> "Experiment":
        set_global_seed(self.config.experiment.random_seed)
        maybe_init_ray(self.config.experiment.use_ray)
        gpu_status = detect_gpu_status(self.config.experiment.use_gpu_if_available)
        frame = load_table(self._data_path())
        preprocessing_cache: dict[str, Any] = {}
        invalid_transforms: dict[str, str] = {}
        for transform in self.config.preprocessing.continuous_transforms:
            try:
                preprocessing_cache[transform] = preprocess_frame(
                    frame,
                    self.config,
                    continuous_transform=transform,
                )
            except ValueError as exc:
                invalid_transforms[transform] = str(exc)
        if not preprocessing_cache:
            raise RuntimeError("No valid preprocessing transforms available for this dataset/config.")
        reference_preprocessed = next(iter(preprocessing_cache.values()))
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
                "config": self.config.model_dump(mode="json", exclude={"config_path", "resolved_data_path", "resolved_output_dir"}),
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
            }
        )

        candidates = generate_candidates(self.config, dataset_fingerprint)
        allowed: list[Candidate] = []
        rejected_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            transform_name = candidate.preprocessing["continuous_transform"]
            if transform_name in invalid_transforms:
                rejected_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "family": candidate.family,
                        "representation_name": candidate.representation["name"],
                        "clustering_name": candidate.clustering["name"],
                        "accepted": False,
                        "rejection_reasons": invalid_transforms[transform_name],
                    }
                )
                continue
            candidate_matrix = preprocessing_cache[transform_name].evaluation_matrix
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
                        "candidate_id": candidate.candidate_id,
                        "family": candidate.family,
                        "representation_name": candidate.representation["name"],
                        "clustering_name": candidate.clustering["name"],
                        "accepted": False,
                        "rejection_reasons": ";".join(decision.reasons),
                    }
                )

        with MlflowLogger(self.config.experiment.use_mlflow) as mlflow:
            mlflow.start_run(self.config.experiment.name, tags={"stage": "run"})
            mlflow.log_params({"candidate_count": len(candidates), "allowed_count": len(allowed)})
            executions: list[CandidateExecution] = []
            for transform, preprocessed in preprocessing_cache.items():
                transform_candidates = [
                    candidate for candidate in allowed if candidate.preprocessing["continuous_transform"] == transform
                ]
                if not transform_candidates:
                    continue
                executions.extend(
                    evaluate_candidate_batch(transform_candidates, preprocessed.evaluation_matrix, self.config)
                )

        candidate_frame = executions_to_frame(executions)
        for execution in executions:
            self._persist_candidate_outputs(execution, reference_preprocessed.row_ids)

        if rejected_rows:
            rejected_frame = pd.concat([pd.DataFrame(rejected_rows), candidate_frame.loc[~candidate_frame["accepted"]]], ignore_index=True, sort=False)
        else:
            rejected_frame = candidate_frame.loc[~candidate_frame["accepted"]].copy()
        accepted_frame = rank_candidates(candidate_frame.loc[candidate_frame["accepted"]].copy())
        registry_frame = pd.concat([candidate_frame, pd.DataFrame(rejected_rows)], ignore_index=True, sort=False)

        export_experiment_tables(
            candidate_registry=registry_frame,
            accepted=accepted_frame,
            rejected=rejected_frame,
            output_dir=self.paths.root,
        )
        self._write_summaries(registry_frame)
        self.registry.mark_stage("run", {"completed": True, "experiment_id": experiment_id})

        if not accepted_frame.empty:
            self.build_consensus()
            self.run_interpretation()
            self.report()
        return self

    def resume(self) -> "Experiment":
        run_stage = self.registry.read_stage("run")
        if run_stage is None:
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

    def build_consensus(self) -> "Experiment":
        accepted = self._read_frame(self.registry.accepted_candidates_path())
        if accepted.empty:
            raise RuntimeError("Cannot build consensus without accepted candidates.")

        label_runs: list[np.ndarray] = []
        for candidate_id in accepted["candidate_id"]:
            label_frame = self._read_frame(self.registry.candidate_file(candidate_id, "final_labels.csv"))
            label_runs.append(label_frame["label"].to_numpy(dtype=int))

        weights = compute_run_weights(accepted, self.config)
        target_k = self._target_k(accepted)
        row_ids = self._read_frame(self.registry.candidate_file(accepted.iloc[0]["candidate_id"], "final_labels.csv"))["row_id"].astype(str).tolist()
        result = fit_consensus(label_runs, weights, row_ids, target_k=target_k)
        labels = pd.DataFrame({"row_id": row_ids, "consensus_label": result.labels})
        cluster_summary = (
            labels.groupby("consensus_label", as_index=False)
            .size()
            .rename(columns={"size": "cluster_size"})
        )
        export_consensus_outputs(
            labels=labels,
            uncertainty=result.uncertainty,
            cluster_summary=cluster_summary,
            output_dir=self.paths.root,
        )
        pd.DataFrame(result.coassociation).to_parquet(self.registry.consensus_file("coassociation_matrix.parquet"), index=False)
        self.registry.mark_stage("consensus", {"completed": True, "target_k": target_k})
        return self

    def report(self) -> "Experiment":
        candidate_registry = self._read_frame(self.registry.candidate_registry_path())
        export_report_bundle(candidate_registry, self.paths.root)
        self.registry.mark_stage("report", {"completed": True})
        return self

    def run_interpretation(self) -> "Experiment":
        consensus_path = self.paths.root / "consensus_labels.csv"
        if not consensus_path.exists():
            raise RuntimeError("Consensus labels are required before interpretation can run.")

        frame = load_table(self._data_path())
        preprocessed = preprocess_frame(
            frame,
            self.config,
            continuous_transform=self.config.preprocessing.continuous_transforms[0],
        )
        consensus = pd.read_csv(consensus_path)
        labels = consensus["consensus_label"].to_numpy(dtype=int)

        result = fit_surrogate_model(
            preprocessed.evaluation_matrix,
            labels,
            preprocessed.feature_names,
            self.config.interpretation,
            random_seed=self.config.experiment.random_seed,
        )
        interpretation_dir = self.paths.root / "interpretation"
        write_table(result.cv_metrics, interpretation_dir / "surrogate_cv_metrics.csv")
        write_json(interpretation_dir / "surrogate_summary.json", result.mean_metrics)
        if result.warning is not None:
            write_json(interpretation_dir / "interpretation_warning.json", {"warning": result.warning})

        if self.config.interpretation.use_permutation_importance:
            permutation = compute_permutation_importance(
                result.estimator,
                preprocessed.evaluation_matrix,
                labels,
                result.feature_names,
                random_seed=self.config.experiment.random_seed,
            )
            write_table(permutation, interpretation_dir / "permutation_importance.csv")
            write_table(
                permutation.head(self.config.interpretation.top_n_features),
                interpretation_dir / "permutation_importance_top_features.csv",
            )

        if self.config.interpretation.use_shap:
            try:
                shap_summary, shap_values, shap_by_class = compute_shap_summary(
                    result.estimator,
                    preprocessed.evaluation_matrix,
                    result.feature_names,
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
                write_json(interpretation_dir / "shap_status.json", {"status": "skipped", "reason": str(exc)})

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
        self.registry.mark_stage("interpretation", {"completed": True})
        return self

    def export_manuscript_bundle(self) -> "Experiment":
        return self.report()

    def _target_k(self, accepted: pd.DataFrame) -> int:
        counts = accepted["n_clusters"].round().astype(int)
        if counts.empty:
            raise RuntimeError("Cannot infer consensus cluster count from an empty accepted set.")
        weighted = accepted.groupby(counts)["final_weighted_score"].sum().sort_values(ascending=False)
        return int(weighted.index[0])

    def _persist_candidate_outputs(self, execution: CandidateExecution, row_ids: list[str]) -> None:
        candidate_dir = self.registry.candidate_dir(execution.candidate.candidate_id)
        write_json(candidate_dir / "config_snapshot.json", execution.candidate.to_dict())
        write_json(candidate_dir / "metrics_summary.json", execution.metrics)
        write_table(
            pd.DataFrame({"row_id": row_ids, "label": execution.labels}),
            candidate_dir / "final_labels.csv",
        )
        if execution.seed_label_runs:
            seed_frame = pd.DataFrame({"row_id": row_ids})
            for index, labels in enumerate(execution.seed_label_runs):
                seed_frame[f"seed_run_{index}"] = labels
            write_table(seed_frame, candidate_dir / "per_seed_labels.parquet")
        if execution.perturbation_label_runs:
            perturbation_frame = pd.DataFrame({"row_id": row_ids})
            for index, labels in enumerate(execution.perturbation_label_runs):
                perturbation_frame[f"perturbation_{index}"] = labels
            write_table(perturbation_frame, candidate_dir / "per_perturbation_labels.parquet")

    def _write_summaries(self, registry_frame: pd.DataFrame) -> None:
        if registry_frame.empty:
            return
        runtime_summary = (
            registry_frame.groupby("family", as_index=False)["runtime_seconds"].mean().rename(columns={"runtime_seconds": "mean_runtime_seconds"})
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
