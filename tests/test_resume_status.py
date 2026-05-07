from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from clustro import Experiment
from clustro import experiment as experiment_module


def test_status_reports_completed_stages(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(12)],
            "age": [50, 51, 52, 53, 70, 71, 72, 73, 60, 61, 62, 63],
            "bmi": [24, 25, 26, 27, 33, 34, 35, 36, 29, 30, 31, 32],
            "marker": [1.1, 1.0, 1.2, 1.3, 2.8, 2.9, 3.0, 3.1, 1.9, 2.0, 2.1, 2.2],
            "sex_male": [0, 1] * 6,
            "site": ["north"] * 6 + ["south"] * 6,
        }
    )
    data_path = tmp_path / "status.csv"
    frame.to_csv(data_path, index=False)

    config = {
        "experiment": {
            "name": "status_run",
            "output_dir": str(tmp_path / "results"),
            "random_seed": 9,
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
            "pilot_min_rows": 6,
            "seeds_pilot": [1],
            "seeds_full": [1],
            "perturbations_full": 1,
            "perturbation_type": "bootstrap",
            "optuna": {"enabled": False},
        },
        "representation": {"methods": [{"name": "none"}]},
        "clustering": {"methods": [{"name": "kmeans", "params": {"n_clusters": [2]}}]},
        "interpretation": {
            "surrogate_model": "random_forest",
            "cross_validation_folds": 2,
            "repeated_cv_repeats": 1,
            "use_shap": False,
            "use_permutation_importance": False,
        },
        "evaluation": {
            "acceptance": {
                "hard_thresholds": {
                    "silhouette_min": -1.0,
                    "ari_seed_min": -1.0,
                    "nmi_seed_min": -1.0,
                    "mean_cluster_jaccard_min": -1.0,
                }
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    experiment = Experiment.from_yaml(config_path)
    experiment.run()
    status = Experiment.from_output_dir(tmp_path / "results").status()

    assert status["run"] is not None
    assert status["consensus"] is not None
    assert status["interpretation"] is not None
    assert status["report"] is not None


def test_resume_continues_from_completed_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(16)],
            "age": [50, 51, 52, 53, 70, 71, 72, 73, 60, 61, 62, 63, 49, 48, 74, 75],
            "bmi": [24, 25, 26, 27, 33, 34, 35, 36, 29, 30, 31, 32, 23, 22, 37, 38],
            "marker": [
                1.1,
                1.0,
                1.2,
                1.3,
                2.8,
                2.9,
                3.0,
                3.1,
                1.9,
                2.0,
                2.1,
                2.2,
                0.9,
                0.8,
                3.2,
                3.3,
            ],
            "sex_male": [0, 1] * 8,
            "site": ["north"] * 8 + ["south"] * 8,
        }
    )
    data_path = tmp_path / "resume.csv"
    frame.to_csv(data_path, index=False)

    config = {
        "experiment": {
            "name": "resume_run",
            "output_dir": str(tmp_path / "results"),
            "random_seed": 7,
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
            "pilot_min_rows": 8,
            "seeds_pilot": [1],
            "seeds_full": [1],
            "perturbations_full": 1,
            "perturbation_type": "bootstrap",
            "optuna": {"enabled": False},
        },
        "representation": {"methods": [{"name": "none"}]},
        "clustering": {
            "methods": [
                {"name": "kmeans", "params": {"n_clusters": [2]}},
                {"name": "agglomerative", "params": {"n_clusters": [2], "linkage": ["ward"]}},
            ]
        },
        "interpretation": {
            "surrogate_model": "random_forest",
            "cross_validation_folds": 2,
            "repeated_cv_repeats": 1,
            "use_shap": False,
            "use_permutation_importance": False,
        },
        "evaluation": {
            "acceptance": {
                "hard_thresholds": {
                    "silhouette_min": -1.0,
                    "ari_seed_min": -1.0,
                    "nmi_seed_min": -1.0,
                    "mean_cluster_jaccard_min": -1.0,
                }
            }
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    experiment = Experiment.from_yaml(config_path)
    original_evaluate_candidate = experiment_module.evaluate_candidate
    interrupted_calls = {"count": 0}

    def interrupt_after_first(candidate, matrix, config):
        interrupted_calls["count"] += 1
        if interrupted_calls["count"] > 1:
            raise RuntimeError("interrupt for resume test")
        return original_evaluate_candidate(candidate, matrix, config)

    monkeypatch.setattr(experiment_module, "evaluate_candidate", interrupt_after_first)

    with pytest.raises(RuntimeError, match="interrupt for resume test"):
        experiment.run()

    run_state = Experiment.from_output_dir(tmp_path / "results").status()["run"]
    assert run_state is not None
    assert run_state["completed"] is False
    assert run_state["completed_candidate_count"] == 1

    completed_summaries = list((tmp_path / "results" / "candidates").glob("*/result_summary.json"))
    assert len(completed_summaries) == 1
    completed_candidate_id = completed_summaries[0].parent.name

    resumed_calls: list[str] = []

    def collect_resumed_calls(candidate, matrix, config):
        resumed_calls.append(candidate.candidate_id)
        return original_evaluate_candidate(candidate, matrix, config)

    monkeypatch.setattr(experiment_module, "evaluate_candidate", collect_resumed_calls)

    Experiment.from_output_dir(tmp_path / "results").resume()

    assert len(resumed_calls) == 1
    assert completed_candidate_id not in resumed_calls
    final_status = Experiment.from_output_dir(tmp_path / "results").status()
    assert final_status["run"] is not None and final_status["run"]["completed"] is True
    assert (tmp_path / "results" / "candidate_registry.parquet").exists()
    assert (tmp_path / "results" / "consensus_labels.csv").exists()
