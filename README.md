# clustro

`clustro` is a stability-first clustering research package for tabular medical datasets.

The package is designed to be installed once and run from arbitrary dataset directories with
config-driven experiments, reproducible outputs, and paper-ready artifacts.

## Scientific Philosophy

- Favor robust, reproducible cluster structure over single-run peak scores.
- Evaluate clustering quality in the actual clustering space by default, while exporting original processed-space metrics for clinical interpretability and fidelity checks.
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

- Deep methods (`ae_centroid_refinement`, `vae_gmm`, `ae_kmeans`, `ae_gmm`) are implemented but
  should be calibrated and reviewed carefully before publication-grade use on real studies.
- Representation cache utilities are available as a package module; broad automatic reuse is
  intentionally conservative and not enabled for every representation branch.
- Tracking integrations require optional dependencies when enabled.
- RAPIDS acceleration is opportunistic for compatible classical methods and falls back to
  scikit-learn when RAPIDS is unavailable or unsupported.

## Publication-Grade Validity Controls

Recent validity repairs focus on biomedical tabular clustering pitfalls:

- **Target-leakage validation.** `data.target_columns` are metadata only and must not appear in
  `data.column_schema.continuous`, `binary`, `categorical`, or `ordinal`. ID columns are likewise
  rejected from the modeling schema.
- **Missingness indicators.** When `data.missingness.add_missing_indicators: true`, preprocessing
  appends numeric indicator features for variables that were missing at fit time in continuous,
  binary, categorical, and ordinal blocks. Feature names use an explicit suffix such as
  `continuous__albumin__missing` or `categorical__race__missing`. Set the option to `false` to
  suppress these features.
- **Continuous imputation.** Median imputation remains the default baseline. KNN imputation and
  iterative Bayesian-ridge imputation are available for planned sensitivity analyses, but iterative
  imputation should not be treated as automatically superior; compare stability and interpretation
  outputs across imputation choices.
- **Explicit ordinal maps.** Every column listed in `data.column_schema.ordinal` must declare its
  clinical level order in `data.ordinal_maps`; automatic ordinal inference is not allowed. Numeric
  maps preserve numeric order exactly and unknown transform-time values encode to `-1`.

Example:

```yaml
data:
  column_schema:
    continuous: [age, albumin]
    binary: [sex]
    categorical: [race]
    ordinal: [premorbid_mrs, stroke_severity_group]
  ordinal_maps:
    premorbid_mrs: [0, 1, 2, 3, 4, 5, 6]
    stroke_severity_group: [mild, moderate, severe]
  missingness:
    continuous_imputer: median  # baseline; alternatives: knn, iterative
    add_missing_indicators: true
    iterative:
      max_iter: 10
      initial_strategy: median
      sample_posterior: false
      random_state: null  # falls back to experiment.random_seed when unset
      estimator: bayesian_ridge
```

### Perturbation Stability Modes

`search.stability_mode` controls how full-evaluation perturbations are run:

```yaml
search:
  stability_mode: full_pipeline  # or processed_matrix
```

- `full_pipeline` is the default publication-grade mode. Each bootstrap or subsample replicate
  starts from raw sampled rows, refits preprocessing, refits the representation, refits clustering,
  maps labels back to original row positions, and compares only rows shared with the full-data
  representative labels. Bootstrap duplicate row draws use the first occurrence for comparison.
- `processed_matrix` keeps the previous fast development behavior: perturbations sample the already
  preprocessed matrix, so imputation/scaling/encoding/variance-filter instability is not assessed.

### Multi-Seed Candidate Acceptance

Full candidate evaluation now summarizes all configured `search.seeds_full` runs. Internal metrics
and structure metrics are accepted/ranked using median summaries, with mean and SD columns retained
where useful. The saved candidate labels come from the representative seed run with the highest mean
ARI to the other seed runs, not necessarily the first seed.

### Interpretation Importance

Permutation importance is now exported in two forms:

- `interpretation/permutation_importance_cv.csv` contains fold-wise held-out permutation importance
  aggregated across repeated stratified CV folds and is preferred for manuscript interpretation.
- `interpretation/permutation_importance_full_fit_exploratory.csv` is a full-fit exploratory
  diagnostic only; it can be optimistic because importance is measured on the training data.

### Deep Clustering Methods

`clustro` provides experimental deep clustering methods:

- `ae_centroid_refinement` — trains an autoencoder, then refines cluster centres in latent space
  using a soft-assignment KL objective. The encoder is **frozen** during refinement, so this is
  **not** the full DEC algorithm (Xie et al. 2016). Use for exploratory benchmarking only.
- `vae_gmm` — trains a VAE with a unit-Gaussian KL term, then fits a GMM post-hoc on the latent
  means. The mixture prior is **never jointly learned**, so this is **not** the VaDE algorithm
  (Jiang et al. 2017). Use for exploratory benchmarking only.

Internal metrics for `ae_centroid_refinement`, `vae_gmm`, `ae_kmeans`, and `ae_gmm` are ranked
in the **cluster space** (the autoencoder latent space) by default, with the original processed-
space metrics also exported under the `_original_space` column suffix.

### Migration Notes

**Clustering method renames.** The method names `dec` and `vade` are deprecated and will be
removed in the next major release. Update your configs:

| Old name | New name                  |
|----------|---------------------------|
| `dec`    | `ae_centroid_refinement`  |
| `vade`   | `vae_gmm`                 |

The old names still work but emit a `DeprecationWarning` at runtime.

**`reporting.export_format` removed.** The field `reporting.export_format` (previously accepted in
`defaults.yaml` and configs but never read by any code) has been removed from `ReportingConfig`.
Remove it from any existing config files — pydantic will reject configs that still include it
because `ReportingConfig` uses `extra = "forbid"`.
