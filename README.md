# clustro

`clustro` is a stability-first clustering research package for tabular medical datasets.

The package is designed to be installed once and run from arbitrary dataset directories with
config-driven experiments, reproducible outputs, and paper-ready artifacts.

## Scientific Philosophy

- Favor robust, reproducible cluster structure over single-run peak scores.
- Evaluate clustering quality in the processed feature space by default.
- Reject unstable or degenerate candidates before ranking.
- Build consensus from accepted runs with weighted co-association, never by label averaging.

## Milestone Status

Current status:

- Milestone 1: under active validation.
- Classical clustering pipeline: implemented.
- Stability engine: corrected and tested for row-identity-aware perturbations.
- Consensus module: implemented for dense small-to-moderate datasets, with a guard against
  accidental large dense co-association allocation.
- Deep clustering: experimental and should be calibrated carefully for each dataset.
- GPU acceleration: optional/experimental and dependent on RAPIDS/cuML availability.
- Manuscript bundle: implemented but under validation.

This package is intended for research. Cluster results should be interpreted through
stability and sensitivity outputs, not a single best metric.

## Installation

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"
pip install -e ".[tracking]"
pip install -e ".[deep]"
pip install -e ".[gpu]"
```

## CLI Usage

```bash
clustro validate-config config.yaml
clustro inspect-data config.yaml
clustro run config.yaml
clustro resume /path/to/results/experiment_id
clustro status /path/to/results/experiment_id
clustro consensus /path/to/results/experiment_id
clustro interpret /path/to/results/experiment_id
clustro report /path/to/results/experiment_id
clustro export-paper-bundle /path/to/results/experiment_id
```

## Programmatic Usage

```python
from clustro import Experiment

exp = Experiment.from_yaml("config.yaml")
exp.run()
exp.build_consensus()
exp.run_interpretation()
exp.export_manuscript_bundle()
```

## Config Structure

See `examples/configs/stroke_example.yaml`, `examples/configs/sepsis_example.yaml`, and
`examples/configs/stroke_deep_example.yaml`.

The config defines:

- dataset path and explicit column schema
- output directory and experiment name
- preprocessing, representation, and clustering search spaces
- seed and perturbation settings
- acceptance thresholds and utility-weighted ranking
- explicit interpretation feature-space settings
- reporting options
- optional Optuna, Ray, GPU/RAPIDS, and rare-category preprocessing settings

## Running From Another Directory

```bash
cd ~/dev/clustro
pip install -e .

cd ~/projects/stroke_analysis
clustro run config.yaml
```

Relative paths in the config are resolved relative to the config file location, not the package
repository.

## Output Artifacts

Each experiment writes under the configured output directory, including:

- `experiment_manifest.json`
- `candidate_registry.parquet`
- `accepted_candidates.parquet`
- `rejected_candidates.parquet`
- consensus labels and uncertainty outputs
- consensus bootstrap stability and cluster-level consensus summaries
- `interpretation/interpretation_feature_space.json`
- expanded search-flow accounting in `reports/search_flow.csv` and `reports/search_flow.json`
- figure-ready CSV or Parquet files
- visualization plots, including search-flow, heatmaps, co-association matrix, t-SNE
  final embedding scatter, uncertainty distributions, and feature/profile summaries
- interpretation outputs such as surrogate CV metrics, permutation importance, and SHAP summaries
- a manuscript bundle directory

## Deep Smoke Examples

```bash
.venv/bin/python examples/run_synthetic_smoke.py
.venv/bin/python examples/run_deep_synthetic_smoke.py
```

## Benchmark Example

```bash
.venv/bin/python examples/run_benchmark_synthetic.py
```

This runs a synthetic comparison between a classical search space and a deep search space and
exports a `benchmark_summary.csv` bundle plus plots and calibration recommendations under
`examples/generated/benchmark_synthetic/report/`.

## Current Limitations

- Deep methods are implemented, but DEC/VaDE results should still be calibrated and reviewed
  carefully before publication-grade use on real studies.
- Representation cache utilities are available as a package module; broad automatic reuse is
  intentionally conservative and not enabled for every representation branch.
- Tracking integrations require optional dependencies when enabled.
- RAPIDS acceleration is opportunistic for compatible classical methods and falls back to
  scikit-learn when RAPIDS is unavailable or unsupported.
