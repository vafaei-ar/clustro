from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from clustro import Experiment


def test_strict_deterministic_mode_reproduces_classical_outputs(tmp_path: Path) -> None:
    frame = _make_deterministic_dataset()
    data_path = tmp_path / "deterministic.csv"
    frame.to_csv(data_path, index=False)

    first_dir = tmp_path / "run_a"
    second_dir = tmp_path / "run_b"
    _write_config(tmp_path / "config_a.yaml", data_path, first_dir)
    _write_config(tmp_path / "config_b.yaml", data_path, second_dir)

    Experiment.from_yaml(tmp_path / "config_a.yaml").run()
    Experiment.from_yaml(tmp_path / "config_b.yaml").run()

    first_registry = _stable_registry(pd.read_parquet(first_dir / "candidate_registry.parquet"))
    second_registry = _stable_registry(pd.read_parquet(second_dir / "candidate_registry.parquet"))
    # final_weighted_score aggregates many BLAS-heavy sklearn metrics; bitwise equality across
    # repeated processes is not guaranteed even under deterministic_mode=strict.
    score_col = "final_weighted_score"
    pd.testing.assert_frame_equal(
        first_registry.drop(columns=[score_col]),
        second_registry.drop(columns=[score_col]),
        check_exact=False,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        first_registry[score_col].to_numpy(dtype=np.float64),
        second_registry[score_col].to_numpy(dtype=np.float64),
        rtol=1e-3,
        atol=1e-10,
    )

    first_labels = pd.read_csv(first_dir / "consensus_labels.csv")
    second_labels = pd.read_csv(second_dir / "consensus_labels.csv")
    pd.testing.assert_series_equal(
        first_labels["consensus_label"],
        second_labels["consensus_label"],
        check_names=False,
    )

    first_uncertainty = pd.read_csv(first_dir / "consensus_uncertainty.csv").sort_values(
        "row_id"
    )
    second_uncertainty = pd.read_csv(second_dir / "consensus_uncertainty.csv").sort_values(
        "row_id"
    )
    # Coassociation → membership probabilities still pick up tiny float noise across processes.
    pd.testing.assert_frame_equal(
        first_uncertainty.drop(columns=["row_id"]).reset_index(drop=True),
        second_uncertainty.drop(columns=["row_id"]).reset_index(drop=True),
        check_exact=False,
        atol=1e-6,
        rtol=1e-3,
    )


def _write_config(path: Path, data_path: Path, output_dir: Path) -> None:
    config = {
        "experiment": {
            "name": path.stem,
            "output_dir": str(output_dir),
            "random_seed": 42,
            "use_ray": False,
            "use_mlflow": False,
            "use_gpu_if_available": False,
            "deterministic_mode": "strict",
        },
        "data": {
            "path": str(data_path),
            "id_column": "patient_id",
            "column_schema": {
                "continuous": ["age", "bmi", "marker"],
                "binary": ["sex_male"],
                "categorical": ["site"],
                "ordinal": [],
            },
        },
        "search": {
            "pilot_sample_fraction": 0.5,
            "pilot_min_rows": 10,
            "seeds_pilot": [1, 2],
            "seeds_full": [1, 2],
            "perturbations_full": 2,
            "perturbation_type": "bootstrap",
            "optuna": {"enabled": False},
        },
        "preprocessing": {
            "continuous_transforms": ["standard"],
            "categorical_encoding": ["onehot"],
            "variance_threshold": {"enabled": False, "threshold": 0.0},
        },
        "representation": {"methods": [{"name": "none"}]},
        "clustering": {
            "methods": [
                {"name": "kmeans", "params": {"n_clusters": [2, 3]}},
                {"name": "agglomerative", "params": {"n_clusters": [2], "linkage": ["ward"]}},
            ]
        },
        "evaluation": {
            "acceptance": {
                "hard_thresholds": {
                    "silhouette_min": -1.0,
                    "ari_seed_min": -1.0,
                    "nmi_seed_min": -1.0,
                    "mean_cluster_jaccard_min": -1.0,
                },
                "weighted_score": {
                    "silhouette": 0.2,
                    "davies_bouldin": 0.05,
                    "calinski_harabasz": 0.05,
                    "ari_seed": 0.3,
                    "nmi_seed": 0.2,
                    "mean_cluster_jaccard": 0.2,
                    "cluster_balance": 0.1,
                },
            },
        },
        "consensus": {"uncertainty": {"bootstrap_repeats": 2}},
        "interpretation": {
            "surrogate_model": "random_forest",
            "cross_validation_folds": 2,
            "repeated_cv_repeats": 1,
            "use_shap": False,
            "use_permutation_importance": False,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _stable_registry(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "family",
        "representation_name",
        "clustering_name",
        "accepted",
        "rejection_reasons",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "ari_seed",
        "nmi_seed",
        "mean_cluster_jaccard",
        "cluster_balance",
        "final_weighted_score",
        "n_clusters",
        "rank",
    ]
    available = [column for column in columns if column in frame.columns]
    return (
        frame[available]
        .sort_values(["family", "clustering_name", "n_clusters"])
        .reset_index(drop=True)
    )


def _make_deterministic_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(24)],
            "age": [
                50,
                51,
                52,
                53,
                54,
                55,
                70,
                71,
                72,
                73,
                74,
                75,
                60,
                61,
                62,
                63,
                64,
                65,
                48,
                49,
                76,
                77,
                66,
                67,
            ],
            "bmi": [
                24,
                25,
                26,
                27,
                28,
                29,
                33,
                34,
                35,
                36,
                37,
                38,
                29,
                30,
                31,
                32,
                33,
                34,
                23,
                24,
                39,
                40,
                35,
                36,
            ],
            "marker": [
                1.0,
                1.1,
                1.2,
                1.3,
                1.1,
                1.2,
                3.0,
                3.1,
                3.2,
                3.3,
                3.1,
                3.2,
                2.0,
                2.1,
                2.2,
                2.3,
                2.1,
                2.2,
                0.9,
                1.0,
                3.4,
                3.5,
                2.4,
                2.5,
            ],
            "sex_male": [0, 1] * 12,
            "site": ["north"] * 12 + ["south"] * 12,
        }
    )
