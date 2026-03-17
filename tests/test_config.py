from __future__ import annotations

from pathlib import Path

import yaml

from clustro import Experiment


def test_relative_paths_are_resolved_from_config_location(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    data_dir = project_dir / "data"
    data_dir.mkdir(parents=True)
    dataset_path = data_dir / "dataset.csv"
    dataset_path.write_text("id,x,flag,group\n1,1.0,0,a\n2,2.0,1,b\n", encoding="utf-8")

    config = {
        "experiment": {"name": "demo", "output_dir": "./results/demo"},
        "data": {
            "path": "./data/dataset.csv",
            "id_column": "id",
            "column_schema": {
                "continuous": ["x"],
                "binary": ["flag"],
                "categorical": ["group"],
                "ordinal": [],
            },
        },
        "clustering": {"methods": [{"name": "kmeans", "params": {"n_clusters": [2]}}]},
    }
    config_path = project_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    experiment = Experiment.from_yaml(config_path)

    assert experiment.config.resolved_data_path == dataset_path.resolve()
    assert experiment.paths.root == (project_dir / "results" / "demo").resolve()
