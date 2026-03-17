"""Generate a synthetic dataset and run a full clustro smoke experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from clustro import Experiment


def main() -> None:
    root = Path(__file__).resolve().parent / "generated" / "synthetic_smoke"
    data_dir = root / "data"
    results_dir = root / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "synthetic_medical_clusters.csv"
    config_path = root / "config.yaml"

    frame = build_dataset()
    frame.to_csv(dataset_path, index=False)
    config = build_config(dataset_path=dataset_path, output_dir=results_dir)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    experiment = Experiment.from_yaml(config_path)
    experiment.run()

    accepted = pd.read_parquet(results_dir / "accepted_candidates.parquet")
    consensus = pd.read_csv(results_dir / "consensus_labels.csv")

    print(f"dataset: {dataset_path}")
    print(f"config: {config_path}")
    print(f"results: {results_dir}")
    print(f"accepted_candidates: {len(accepted)}")
    print(f"consensus_clusters: {consensus['consensus_label'].nunique()}")
    print("smoke run completed")


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    clusters = [
        {
            "prefix": "cardio",
            "n": 36,
            "age_mean": 58,
            "bmi_mean": 28,
            "glucose_mean": 112,
            "sbp_mean": 128,
            "dbp_mean": 79,
            "marker_mean": 1.4,
            "smoker_prob": 0.18,
            "htn_prob": 0.32,
            "sex_prob": 0.48,
            "site": "north",
            "insurance": "private",
        },
        {
            "prefix": "metabolic",
            "n": 36,
            "age_mean": 67,
            "bmi_mean": 34,
            "glucose_mean": 176,
            "sbp_mean": 146,
            "dbp_mean": 88,
            "marker_mean": 2.7,
            "smoker_prob": 0.36,
            "htn_prob": 0.72,
            "sex_prob": 0.55,
            "site": "central",
            "insurance": "public",
        },
        {
            "prefix": "frail",
            "n": 36,
            "age_mean": 78,
            "bmi_mean": 24,
            "glucose_mean": 138,
            "sbp_mean": 118,
            "dbp_mean": 71,
            "marker_mean": 3.5,
            "smoker_prob": 0.10,
            "htn_prob": 0.54,
            "sex_prob": 0.42,
            "site": "south",
            "insurance": "public",
        },
    ]

    frames = []
    for cluster in clusters:
        n = cluster["n"]
        frame = pd.DataFrame(
            {
                "patient_id": [f"{cluster['prefix']}_{index:03d}" for index in range(n)],
                "age": rng.normal(cluster["age_mean"], 3.0, n).round(1),
                "bmi": rng.normal(cluster["bmi_mean"], 2.2, n).round(2),
                "glucose": rng.normal(cluster["glucose_mean"], 12.0, n).round(1),
                "sbp": rng.normal(cluster["sbp_mean"], 8.0, n).round(1),
                "dbp": rng.normal(cluster["dbp_mean"], 5.0, n).round(1),
                "marker": rng.normal(cluster["marker_mean"], 0.25, n).round(3),
                "sex_male": rng.binomial(1, cluster["sex_prob"], n),
                "smoker": rng.binomial(1, cluster["smoker_prob"], n),
                "hypertension": rng.binomial(1, cluster["htn_prob"], n),
                "race": rng.choice(["white", "black", "asian", "other"], size=n, p=[0.45, 0.22, 0.18, 0.15]),
                "insurance_type": [cluster["insurance"]] * n,
                "site": [cluster["site"]] * n,
            }
        )
        frames.append(frame)

    full = pd.concat(frames, ignore_index=True)

    # Add a small amount of realistic missingness.
    missing_columns = ["bmi", "glucose", "race"]
    for column in missing_columns:
        indices = rng.choice(full.index.to_numpy(), size=6, replace=False)
        full.loc[indices, column] = np.nan

    return full


def build_config(*, dataset_path: Path, output_dir: Path) -> dict[str, object]:
    return {
        "experiment": {
            "name": "synthetic_smoke_run",
            "output_dir": str(output_dir),
            "random_seed": 2026,
            "n_jobs": 1,
            "use_ray": False,
            "use_mlflow": False,
            "use_gpu_if_available": False,
        },
        "data": {
            "path": str(dataset_path),
            "id_column": "patient_id",
            "target_columns": [],
            "column_schema": {
                "continuous": ["age", "bmi", "glucose", "sbp", "dbp", "marker"],
                "binary": ["sex_male", "smoker", "hypertension"],
                "categorical": ["race", "insurance_type", "site"],
                "ordinal": [],
            },
            "missingness": {
                "continuous_imputer": "median",
                "categorical_imputer": "most_frequent",
                "add_missing_indicators": True,
            },
        },
        "search": {
            "pilot_sample_fraction": 0.45,
            "pilot_min_rows": 24,
            "seeds_pilot": [101, 102],
            "seeds_full": [101, 102, 103],
            "perturbations_full": 3,
            "perturbation_type": "bootstrap",
            "optuna": {"enabled": False},
        },
        "preprocessing": {
            "continuous_transforms": ["standard", "robust"],
            "categorical_encoding": ["onehot"],
            "variance_threshold": {"enabled": False, "threshold": 0.0},
        },
        "representation": {
            "methods": [
                {"name": "none"},
                {"name": "pca", "params": {"n_components": [4, 6], "whiten": [False]}},
            ]
        },
        "clustering": {
            "methods": [
                {"name": "kmeans", "params": {"n_clusters": [2, 3, 4]}},
                {"name": "gmm", "params": {"n_components": [2, 3, 4], "covariance_type": ["diag"]}},
                {"name": "agglomerative", "params": {"n_clusters": [2, 3, 4], "linkage": ["ward", "average"]}},
            ]
        },
        "evaluation": {
            "internal_metrics": ["silhouette", "davies_bouldin", "calinski_harabasz"],
            "structure_constraints": {
                "min_clusters": 2,
                "max_clusters": 6,
                "min_cluster_fraction": 0.05,
                "max_noise_fraction": 0.40,
                "dominant_cluster_cap": 0.90,
            },
            "acceptance": {
                "hard_thresholds": {
                    "silhouette_min": -1.0,
                    "ari_seed_min": -1.0,
                    "nmi_seed_min": -1.0,
                    "mean_cluster_jaccard_min": -1.0,
                },
                "weighted_score": {
                    "silhouette": 0.20,
                    "davies_bouldin": -0.05,
                    "calinski_harabasz": 0.05,
                    "ari_seed": 0.25,
                    "nmi_seed": 0.20,
                    "mean_cluster_jaccard": 0.20,
                    "cluster_balance": 0.10,
                    "runtime_penalty": -0.03,
                    "parsimony_penalty": -0.02,
                },
                "accept_top_fraction_if_above": 1.0,
            },
        },
        "consensus": {
            "include_only_accepted": True,
            "run_weighting": {
                "source": "final_weighted_score",
                "normalize": True,
                "floor": 0.01,
            },
            "consensus_method": "hierarchical_on_coassociation",
            "final_k_strategy": "weighted_mode",
            "uncertainty": {"bootstrap_repeats": 10},
        },
        "reporting": {
            "generate_figures": True,
            "generate_tables": True,
            "export_format": ["csv", "parquet", "json"],
            "manuscript_bundle": True,
        },
    }


if __name__ == "__main__":
    main()
