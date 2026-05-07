"""Benchmark runner comparing classical and deep experiment bundles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from clustro import Experiment
from clustro.benchmark.calibration import calibrate_from_benchmark
from clustro.benchmark.reporting import export_benchmark_report
from clustro.benchmark.synthetic import write_benchmark_inputs, write_yaml_config


def run_classical_vs_deep_benchmark(root: Path, *, reuse_existing: bool = True) -> pd.DataFrame:
    dataset_path, root = write_benchmark_inputs(root)
    classical_dir = root / "classical"
    deep_dir = root / "deep"
    classical_config = classical_dir / "config.yaml"
    deep_config = deep_dir / "config.yaml"
    classical_dir.mkdir(parents=True, exist_ok=True)
    deep_dir.mkdir(parents=True, exist_ok=True)

    write_yaml_config(classical_config, _classical_config(dataset_path, classical_dir / "results"))
    write_yaml_config(deep_config, _deep_config(dataset_path, deep_dir / "results"))

    _run_or_resume(classical_config, classical_dir / "results", reuse_existing=reuse_existing)
    _run_or_resume(deep_config, deep_dir / "results", reuse_existing=reuse_existing)

    frame = pd.DataFrame(
        [
            _summarize_result("classical", classical_dir / "results"),
            _summarize_result("deep", deep_dir / "results"),
        ]
    )
    frame.to_csv(root / "benchmark_summary.csv", index=False)
    calibration = calibrate_from_benchmark(root)
    export_benchmark_report(frame, root, calibration)
    return frame


def _summarize_result(label: str, result_dir: Path) -> dict[str, object]:
    registry = pd.read_parquet(result_dir / "candidate_registry.parquet")
    accepted = pd.read_parquet(result_dir / "accepted_candidates.parquet")
    consensus = pd.read_csv(result_dir / "consensus_labels.csv")
    runtime = pd.read_csv(result_dir / "runtime_summary.csv")
    return {
        "benchmark_family": label,
        "candidate_count": int(len(registry)),
        "accepted_count": int(len(accepted)),
        "top_weighted_score": float(accepted["final_weighted_score"].max())
        if not accepted.empty
        else float("nan"),
        "consensus_clusters": int(consensus["consensus_label"].nunique()),
        "mean_family_runtime_seconds": float(runtime["mean_runtime_seconds"].mean())
        if not runtime.empty
        else 0.0,
    }


def _common_config(dataset_path: Path, output_dir: Path) -> dict[str, object]:
    return {
        "experiment": {
            "name": output_dir.parent.name,
            "output_dir": str(output_dir),
            "random_seed": 2026,
            "n_jobs": 1,
            "use_ray": False,
            "use_mlflow": False,
            "use_gpu_if_available": False,
            "deterministic_mode": "fast",
        },
        "data": {
            "path": str(dataset_path),
            "id_column": "patient_id",
            "column_schema": {
                "continuous": ["age", "bmi", "glucose", "sbp", "dbp", "marker"],
                "binary": ["sex_male", "smoker", "hypertension"],
                "categorical": ["race", "insurance_type", "site"],
                "ordinal": [],
            },
        },
        "search": {
            "pilot_sample_fraction": 0.4,
            "pilot_min_rows": 36,
            "seeds_pilot": [101],
            "seeds_full": [101, 102],
            "perturbations_full": 2,
            "perturbation_type": "bootstrap",
            "optuna": {"enabled": False},
        },
        "preprocessing": {
            "continuous_transforms": ["standard"],
            "categorical_encoding": ["onehot"],
            "variance_threshold": {"enabled": False, "threshold": 0.0},
        },
        "evaluation": {
            "internal_metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
            "structure_constraints": {
                "min_clusters": 2,
                "max_clusters": 5,
                "min_cluster_fraction": 0.03,
                "max_noise_fraction": 0.5,
                "dominant_cluster_cap": 0.95,
            },
            "acceptance": {
                "hard_thresholds": {
                    "silhouette_min": -1.0,
                    "ari_seed_min": -1.0,
                    "nmi_seed_min": -1.0,
                    "mean_cluster_jaccard_min": -1.0,
                },
                "weighted_score": {
                    "silhouette": 0.15,
                    "davies_bouldin": -0.05,
                    "calinski_harabasz": 0.05,
                    "ari_seed": 0.25,
                    "nmi_seed": 0.20,
                    "mean_cluster_jaccard": 0.20,
                    "cluster_balance": 0.10,
                    "average_confidence": 0.10,
                    "assignment_entropy": -0.08,
                    "reconstruction_loss": -0.04,
                    "runtime_penalty": -0.05,
                    "parsimony_penalty": -0.03,
                },
            },
        },
        "interpretation": {
            "surrogate_model": "random_forest",
            "cross_validation_folds": 3,
            "repeated_cv_repeats": 1,
            "use_shap": False,
            "use_permutation_importance": True,
            "top_n_features": 20,
            "grouped_correlation_threshold": 0.85,
        },
    }


def _classical_config(dataset_path: Path, output_dir: Path) -> dict[str, object]:
    config = _common_config(dataset_path, output_dir)
    config["representation"] = {
        "methods": [
            {"name": "none"},
            {"name": "pca", "params": {"n_components": [4], "whiten": [False]}},
        ],
    }
    config["clustering"] = {
        "methods": [
            {"name": "kmeans", "params": {"n_clusters": [2, 3, 4]}},
            {"name": "gmm", "params": {"n_components": [2, 3, 4], "covariance_type": ["diag"]}},
            {"name": "agglomerative", "params": {"n_clusters": [2, 3, 4], "linkage": ["ward"]}},
        ]
    }
    return config


def _deep_config(dataset_path: Path, output_dir: Path) -> dict[str, object]:
    config = _common_config(dataset_path, output_dir)
    config["representation"] = {"methods": [{"name": "none"}]}
    config["clustering"] = {
        "methods": [
            {
                "name": "ae_kmeans",
                "params": {
                    "n_clusters": [3],
                    "latent_dim": [5, 8],
                    "hidden_layers": [[64, 32]],
                    "epochs": 10,
                    "batch_size": 32,
                    "learning_rate": [0.001],
                    "early_stopping_patience": 4,
                },
            },
            {
                "name": "dec",
                "params": {
                    "n_clusters": [3],
                    "latent_dim": [5, 8],
                    "hidden_layers": [[64, 32]],
                    "pretrain_epochs": 8,
                    "finetune_epochs": 6,
                    "finetune_patience": 3,
                    "batch_size": 32,
                    "learning_rate": [0.001],
                },
            },
            {
                "name": "vade",
                "params": {
                    "n_clusters": [3],
                    "latent_dim": [5, 8],
                    "hidden_layers": [[64, 32]],
                    "epochs": 8,
                    "batch_size": 32,
                    "learning_rate": [0.001],
                },
            },
        ]
    }
    return config


def _run_or_resume(config_path: Path, result_dir: Path, *, reuse_existing: bool) -> None:
    if reuse_existing and (result_dir / "state" / "run.json").exists():
        Experiment.from_output_dir(result_dir).resume()
        return
    Experiment.from_yaml(config_path).run()
