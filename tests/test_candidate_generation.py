from __future__ import annotations

import pandas as pd

from clustro.config.schema import ExperimentConfig
from clustro.data.preprocess_pipeline import preprocess_frame
from clustro.search.scheduler import CandidateExecution, executions_to_frame
from clustro.search.search_space import generate_candidates


def test_categorical_encoding_participates_in_candidate_graph() -> None:
    config = _config()
    candidates = generate_candidates(config, {"dataset": "toy"})

    assert len(candidates) == 2
    assert {candidate.preprocessing["categorical_encoding"] for candidate in candidates} == {
        "onehot",
        "ordinal",
    }
    assert len({candidate.candidate_id for candidate in candidates}) == 2


def test_preprocess_frame_uses_requested_categorical_encoding() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0], "site": ["a", "b", "a"]})
    config = _config()

    onehot = preprocess_frame(frame, config, categorical_encoding="onehot")
    ordinal = preprocess_frame(frame, config, categorical_encoding="ordinal")

    assert onehot.evaluation_matrix.shape[1] > ordinal.evaluation_matrix.shape[1]


def test_registry_includes_categorical_encoding_and_params_json() -> None:
    candidate = generate_candidates(_config(), {"dataset": "toy"})[0]
    execution = CandidateExecution(
        candidate=candidate,
        labels=pd.Series([0, 1]).to_numpy(),
        seed_label_runs=[],
        perturbation_label_runs=[],
        metrics={"n_clusters": 2.0, "runtime_seconds": 1.0},
        accepted=True,
        rejection_reasons=[],
        runtime_seconds=1.0,
    )

    frame = executions_to_frame([execution])

    assert frame.loc[0, "continuous_transform"] == "standard"
    assert frame.loc[0, "categorical_encoding"] in {"onehot", "ordinal"}
    assert frame.loc[0, "representation_params_json"] == "{}"
    assert frame.loc[0, "clustering_params_json"] == '{"n_clusters": 2}'


def _config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": {"name": "candidate", "output_dir": "./results"},
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
                "continuous_transforms": ["standard"],
                "categorical_encoding": ["onehot", "ordinal"],
            },
            "representation": {"methods": [{"name": "none"}]},
            "clustering": {"methods": [{"name": "kmeans", "params": {"n_clusters": [2]}}]},
        }
    )
