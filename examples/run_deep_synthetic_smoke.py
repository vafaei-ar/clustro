"""Generate a small synthetic dataset and run Milestone 2 deep methods."""

from __future__ import annotations

from pathlib import Path

import yaml

from clustro import Experiment
from run_synthetic_smoke import build_dataset


def main() -> None:
    root = Path(__file__).resolve().parent / "generated" / "deep_synthetic_smoke"
    data_dir = root / "data"
    results_dir = root / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "synthetic_medical_clusters.csv"
    config_path = root / "config.yaml"
    build_dataset().to_csv(dataset_path, index=False)

    config = {
        "experiment": {
            "name": "deep_synthetic_smoke",
            "output_dir": str(results_dir),
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
            "target_columns": [],
            "column_schema": {
                "continuous": ["age", "bmi", "glucose", "sbp", "dbp", "marker"],
                "binary": ["sex_male", "smoker", "hypertension"],
                "categorical": ["race", "insurance_type", "site"],
                "ordinal": [],
            },
        },
        "search": {
            "pilot_sample_fraction": 0.5,
            "pilot_min_rows": 30,
            "seeds_pilot": [101],
            "seeds_full": [101],
            "perturbations_full": 1,
            "perturbation_type": "bootstrap",
            "optuna": {"enabled": False},
        },
        "preprocessing": {
            "continuous_transforms": ["standard"],
            "categorical_encoding": ["onehot"],
            "variance_threshold": {"enabled": False, "threshold": 0.0},
        },
        "representation": {
            "methods": [
                {
                    "name": "autoencoder",
                    "params": {
                        "latent_dim": [5],
                        "hidden_layers": [[32, 16]],
                        "dropout": [0.0],
                        "epochs": 10,
                        "batch_size": 32,
                        "learning_rate": [0.001],
                        "early_stopping_patience": 3,
                    },
                }
            ]
        },
        "clustering": {
            "methods": [
                {"name": "kmeans", "params": {"n_clusters": [3]}},
                {
                    "name": "ae_kmeans",
                    "params": {
                        "n_clusters": [3],
                        "latent_dim": [5],
                        "hidden_layers": [[32, 16]],
                        "epochs": 10,
                        "batch_size": 32,
                        "learning_rate": [0.001],
                        "early_stopping_patience": 3,
                    },
                },
                {
                    "name": "dec",
                    "params": {
                        "n_clusters": [3],
                        "latent_dim": [5],
                        "hidden_layers": [[32, 16]],
                        "pretrain_epochs": 8,
                        "finetune_epochs": 5,
                        "batch_size": 32,
                        "learning_rate": [0.001],
                    },
                },
                {
                    "name": "vade",
                    "params": {
                        "n_clusters": [3],
                        "latent_dim": [5],
                        "hidden_layers": [[32, 16]],
                        "epochs": 8,
                        "batch_size": 32,
                        "learning_rate": [0.001],
                    },
                },
            ]
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
                    "silhouette": 0.2,
                    "davies_bouldin": -0.05,
                    "calinski_harabasz": 0.05,
                    "ari_seed": 0.25,
                    "nmi_seed": 0.20,
                    "mean_cluster_jaccard": 0.20,
                    "cluster_balance": 0.10,
                },
                "accept_top_fraction_if_above": 1.0,
            },
        },
        "interpretation": {
            "surrogate_model": "random_forest",
            "cross_validation_folds": 3,
            "repeated_cv_repeats": 1,
            "use_shap": False,
            "use_permutation_importance": True,
            "top_n_features": 15,
            "grouped_correlation_threshold": 0.85,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    experiment = Experiment.from_yaml(config_path)
    experiment.run()
    print(f"deep smoke results: {results_dir}")


if __name__ == "__main__":
    main()
