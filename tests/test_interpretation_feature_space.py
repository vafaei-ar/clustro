from __future__ import annotations

from pathlib import Path

import pandas as pd

from clustro.config.schema import ExperimentConfig
from clustro.experiment import Experiment
from clustro.tracking.artifact_registry import ArtifactRegistry
from clustro.utils.paths import build_experiment_paths


def test_original_interpretation_feature_space_uses_configured_transform() -> None:
    experiment = _experiment(
        {
            "feature_space": "original_imputed_scaled",
            "continuous_transform": "robust",
            "categorical_encoding": "ordinal",
        }
    )

    feature_space = experiment._resolve_interpretation_feature_space()

    assert feature_space["continuous_transform"] == "robust"
    assert feature_space["categorical_encoding"] == "ordinal"
    assert feature_space["source_candidate_id"] is None


def test_best_candidate_interpretation_feature_space_uses_top_ranked_candidate(
    tmp_path: Path,
) -> None:
    experiment = _experiment(
        {"feature_space": "best_candidate_preprocessing"},
        output_dir=tmp_path,
    )
    accepted = pd.DataFrame(
        {
            "candidate_id": ["low", "high"],
            "continuous_transform": ["standard", "robust"],
            "categorical_encoding": ["onehot", "ordinal"],
            "final_weighted_score": [0.1, 0.9],
        }
    )
    accepted.to_parquet(experiment.registry.accepted_candidates_path(), index=False)

    feature_space = experiment._resolve_interpretation_feature_space()

    assert feature_space["continuous_transform"] == "robust"
    assert feature_space["categorical_encoding"] == "ordinal"
    assert feature_space["source_candidate_id"] == "high"


def _experiment(
    interpretation: dict[str, object],
    *,
    output_dir: Path | None = None,
) -> Experiment:
    root = output_dir or Path("./results")
    config = ExperimentConfig.model_validate(
        {
            "experiment": {"name": "interpretation", "output_dir": str(root)},
            "data": {
                "path": "./dummy.csv",
                "column_schema": {
                    "continuous": ["x"],
                    "binary": [],
                    "categorical": ["site"],
                    "ordinal": [],
                },
            },
            "preprocessing": {
                "continuous_transforms": ["standard", "robust"],
                "categorical_encoding": ["onehot", "ordinal"],
            },
            "representation": {"methods": [{"name": "none"}]},
            "clustering": {"methods": [{"name": "kmeans", "params": {"n_clusters": [2]}}]},
            "interpretation": {
                "surrogate_model": "random_forest",
                "use_shap": False,
                "use_permutation_importance": False,
                **interpretation,
            },
        }
    )
    paths = build_experiment_paths(root)
    return Experiment(config=config, paths=paths, registry=ArtifactRegistry(paths))
