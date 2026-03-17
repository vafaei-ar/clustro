# Cursor Handoff for `clustro`

## 1. Objective

Build a Python package named `clustro` for **stability-first clustering of tabular medical datasets**.

This is **not** a generic AutoML package. It is a research framework for testing how robust clustering results are across preprocessing choices, representation learning choices, clustering families, random seeds, and perturbation resampling.

The package must be:

- installable with `pip install -e .` or `pip install .`
- importable from anywhere
- runnable on datasets stored in other directories
- able to write all outputs to a user-defined results directory
- suitable for producing artifacts needed for a scientific paper

The package should support:

- reproducible config-driven runs
- CPU and optional GPU execution
- early rejection of weak candidates
- live monitoring in a browser
- experiment tracking
- export of tables and figure-ready data
- consensus clustering across accepted results
- interpretation of final consensus clusters with supervised surrogate models and SHAP

---

## 2. Critical design requirements

### 2.1 Separation of package code and data/results
The codebase must be installable as a package and must **not** assume that datasets live inside the repo.

Required workflow:

1. Develop package in one directory
2. Install package
3. Run package from another project directory on arbitrary datasets
4. Write outputs to a dataset-specific results directory

Example desired usage:

```bash
# package development directory
cd ~/dev/clustro
pip install -e .

# separate dataset project directory
cd ~/projects/stroke_analysis
clustro run config.yaml

# outputs written to:
~/projects/stroke_analysis/results/stroke_run_01/
```

The package must accept absolute and relative paths in the config file.

### 2.2 Scientific guardrails
Do not build a cherry-picking engine. Build a **stability-first framework**.

Required scientific rules:

- use a curated search space, not every possible method
- evaluate robustness, not only internal clustering scores
- compute core evaluation metrics in the processed original feature space by default, not only in the DR space
- reject unstable or degenerate solutions before ranking
- do not average cluster labels directly across runs
- use a weighted co-association consensus matrix from accepted runs
- report uncertainty per sample
- treat post hoc feature importance as descriptive, not causal

### 2.3 Initial scope
Do **not** try to implement everything at once.

Implement in stages:

- **Milestone 1**: strong classical pipeline with stability, acceptance, consensus, reporting
- **Milestone 2**: deep extensions
- **Milestone 3**: polishing and benchmark utilities

---

## 3. Package architecture

Use a modern Python package layout.

```text
clustro/
  pyproject.toml
  README.md
  LICENSE
  .gitignore
  src/
    clustro/
      __init__.py
      cli.py
      config/
        schema.py
        validators.py
        defaults.yaml
      data/
        loaders.py
        schema.py
        splitting.py
        sampling.py
        imputation.py
        transformations.py
        encoders.py
        preprocess_pipeline.py
      repr/
        base.py
        none_repr.py
        pca_repr.py
        umap_repr.py
        ae_repr.py
        cache.py
      clustering/
        base.py
        classical.py
        deep_dec.py
        deep_vade.py
        wrappers.py
      search/
        search_space.py
        compatibility.py
        optuna_objective.py
        pruners.py
        scheduler.py
        early_screen.py
      evaluation/
        metrics_internal.py
        metrics_stability.py
        metrics_structure.py
        acceptance.py
        ranking.py
      consensus/
        coassociation.py
        weighting.py
        consensus_fit.py
        uncertainty.py
        alignment.py
      interpretation/
        surrogate.py
        shap_utils.py
        permutation.py
        profiling.py
      reporting/
        tables.py
        figures.py
        exports.py
        manuscript_bundle.py
      tracking/
        mlflow_logger.py
        ray_monitor.py
        artifact_registry.py
      utils/
        random.py
        hashing.py
        io.py
        parallel.py
        gpu.py
        paths.py
      tests/
  docs/
  examples/
    configs/
      stroke_example.yaml
      sepsis_example.yaml
```

Use `src/` layout.

---

## 4. Installation and packaging requirements

Use `pyproject.toml`.

Minimum requirements:

- Python 3.11+
- installable with `pip install -e .`
- CLI entry point called `clustro`
- clean extras:
  - `clustro[gpu]`
  - `clustro[dev]`
  - `clustro[deep]`

### 4.1 Example CLI entry points
Support these commands:

```bash
clustro validate-config config.yaml
clustro inspect-data config.yaml
clustro run config.yaml
clustro resume <experiment_id>
clustro consensus <experiment_id>
clustro interpret <experiment_id>
clustro report <experiment_id>
clustro export-paper-bundle <experiment_id>
```

### 4.2 Programmatic API
Expose:

```python
from clustro import Experiment

exp = Experiment.from_yaml("config.yaml")
exp.run()
exp.build_consensus()
exp.run_interpretation()
exp.export_manuscript_bundle()
```

---

## 5. Runtime model

The package must run on datasets in arbitrary directories.

### 5.1 Config-driven run
The config file should define:

- dataset path
- column schema
- output directory
- experiment name
- methods to include
- seeds
- evaluation thresholds
- interpretation settings

### 5.2 Output location
All outputs must go under a user-defined output directory, for example:

```yaml
experiment:
  name: "stroke_dataset_v1"
  output_dir: "./results/stroke_dataset_v1"
```

This output directory may be outside the package repo.

The package must create all subdirectories automatically.

### 5.3 Resume behavior
A run should checkpoint state. Resume should continue from incomplete stages instead of recomputing everything.

---

## 6. Data support

### 6.1 Supported tabular data for v1
Support:

- continuous numerical variables
- binary variables
- low-cardinality categorical variables
- ordinal variables only if explicitly declared

Do not silently infer ordinal structure.

### 6.2 Missingness
Support:

- median imputation for continuous
- most-frequent imputation for categorical
- optional KNN imputation for continuous
- optional missingness indicators

### 6.3 Column schema
Require explicit schema in config. Do not rely only on automatic dtype inference.

---

## 7. Preprocessing space

### 7.1 Continuous transforms
Support:

- none
- standard scaling
- robust scaling
- power transform
- log1p then scaling, only where valid

### 7.2 Categorical encoding
Support:

- one-hot encoding by default
- optional ordinal encoding only if explicitly requested

### 7.3 Feature filtering
Support:

- variance threshold
- optional rare-category collapsing

---

## 8. Representation methods

### Main search space
- none
- PCA
- UMAP
- autoencoder latent space

### Visualization only
- t-SNE 2D
- UMAP 2D

Important:
- Do **not** include t-SNE in the main clustering search space
- t-SNE is for visualization only

---

## 9. Clustering methods

### 9.1 Classical methods for Milestone 1
- KMeans
- MiniBatchKMeans
- GaussianMixture
- AgglomerativeClustering
- SpectralClustering
- HDBSCAN
- OPTICS
- Birch

### 9.2 Deep methods for Milestone 2
- AE + KMeans
- AE + GMM
- DEC
- VaDE

Do not implement additional deep methods until the framework is stable.

---

## 10. Compatibility rules

Implement an explicit compatibility engine that approves or blocks invalid combinations before execution.

Examples:

- `log1p_standard` only on valid positive-valued variables
- `ward` linkage only with Euclidean geometry
- `GMM(full)` disallowed if dimensionality is too high relative to sample size
- `spectral` disallowed for very large datasets if computationally impractical
- AE-based methods require dense inputs
- t-SNE disallowed in main search
- one-hot plus GMM may be blocked if dimensionality and sample size are unfavorable

Every rejected combination must store a reason.

---

## 11. Search design

### 11.1 Candidate definition
A candidate is one valid pipeline branch:

- preprocessing choice
- representation choice
- clustering choice
- hyperparameter setting

Each candidate must have a stable hash ID based on config plus dataset/schema fingerprint.

### 11.2 Search stages

#### Stage 0. Initialize
- validate config
- create experiment ID
- initialize MLflow
- initialize Ray if enabled
- detect GPU
- set all seeds

#### Stage 1. Build candidate graph
- generate valid combinations
- apply compatibility rules
- register candidates

#### Stage 2. Pilot screen
For each candidate:
- run on a sample subset
- use 2 seeds
- use 1 to 2 perturbation resamples
- compute quick metrics
- prune weak candidates early

#### Stage 3. Full evaluation
For surviving candidates:
- run on full data
- use full seed set, default 10
- use perturbation resampling, default 10
- compute full metrics
- accept or reject

#### Stage 4. Consensus
- use accepted runs only
- build weighted co-association matrix
- derive final consensus clustering
- compute uncertainty

#### Stage 5. Interpretation
- train surrogate classifier on consensus labels
- repeated cross-validation
- SHAP and permutation importance
- cluster profiling

#### Stage 6. Reporting
- export tables
- export figure-ready data
- export manuscript bundle

---

## 12. Search controller and pruning

Use Optuna for hyperparameter optimization and pruning.

### 12.1 Trial unit
One trial = one candidate pipeline with one hyperparameter setting.

### 12.2 Intermediate reporting
Each trial should report metrics after:
- pilot seed 1
- pilot seed 2
- perturbation 1
- perturbation 2
- later full-stage checkpoints

### 12.3 Pruning rules
Prune a candidate if during pilot:
- only one cluster is found
- too many tiny clusters are found
- silhouette is below threshold twice
- seed ARI is below threshold
- noise fraction is excessive
- runtime exceeds cap

### 12.4 Search organization
Use separate studies by family if useful, so cheap methods do not dominate the search budget.

---

## 13. Stability metrics

This is a core requirement.

### 13.1 Seed stability
For a fixed candidate:
- run multiple seeds
- compare pairwise partitions
- compute ARI and NMI across seeds
- summarize mean, median, SD

### 13.2 Perturbation stability
Support:
- bootstrap resampling
- subsampling without replacement

For each perturbation:
- refit candidate
- align replicate clusters to a reference partition
- compute cluster-wise Jaccard stability
- compute overall stability summaries

### 13.3 Local hyperparameter stability
Around the final chosen hyperparameters, perturb a few local values:
- `n_clusters ± 1` when relevant
- nearby UMAP neighbor counts
- nearby HDBSCAN min cluster sizes

If the structure collapses immediately, mark the candidate as fragile.

### 13.4 Observation-level stability
For each sample:
- proportion of runs where assignment is stable
- final consensus entropy
- alternative cluster support

---

## 14. Evaluation and acceptance

### 14.1 Internal metrics
Support:
- silhouette
- Davies-Bouldin
- Calinski-Harabasz

Compute in the processed evaluation space by default, not only in the low-dimensional representation.

### 14.2 Structural sanity checks
Support:
- minimum number of clusters
- maximum number of clusters
- minimum cluster size fraction
- maximum noise fraction
- dominant-cluster cap

### 14.3 Hard rejection
Reject if any holds:
- cluster count out of range
- one giant cluster dominates
- too many tiny clusters
- too much noise
- mean seed ARI below threshold
- mean bootstrap Jaccard below threshold

### 14.4 Weighted score
For accepted candidates, compute a weighted score from:
- internal quality
- seed stability
- perturbation stability
- cluster balance
- runtime penalty
- complexity penalty

### 14.5 Final accepted set
Accepted set should be configurable as:
- all candidates above fixed threshold
- top fraction among those passing hard filters
- or both

---

## 15. Consensus module

### 15.1 Input
Use accepted runs only.

Each accepted run provides:
- hard labels
- optional noise labels
- run-level metrics
- run weight

### 15.2 Weighted co-association
For each accepted run, define whether each pair of samples is co-clustered.

Build:

C_ij = sum_r w_r A^(r)_ij / sum_r w_r

Where:
- A^(r)_ij = 1 if i and j are together in run r
- w_r is the run weight

### 15.3 Final consensus clustering
Default:
- distance = `1 - C`
- hierarchical clustering on that distance
- choose final `k` based on stability and minimum cluster size

Alternative:
- spectral clustering on `C`

### 15.4 Sample-level uncertainty
For each sample:
- compute average affinity to each consensus cluster
- normalize into soft membership
- compute entropy
- compute gap between top two memberships

Outputs:
- final hard cluster label
- soft membership vector
- entropy
- ambiguous/stable flag

### 15.5 Cluster-level summaries
For each consensus cluster:
- size
- average within-cluster consensus
- bootstrap recovery
- median sample uncertainty

---

## 16. Deep learning module

Use PyTorch.

### 16.1 Autoencoder
Implement:
- encoder
- decoder
- latent extraction
- early stopping
- caching of latent embeddings

### 16.2 DEC
Implement standard DEC workflow:
- pretrain autoencoder
- initialize clusters in latent space
- optimize target distribution refinement
- log training curves

### 16.3 VaDE
Implement:
- VAE with Gaussian mixture prior
- soft assignments
- stored training metrics

### 16.4 Reproducibility
Control:
- Python seed
- NumPy seed
- PyTorch seed
- CUDA deterministic flags when strict mode requested

Expose:
- `deterministic_mode: strict | fast`

---

## 17. Parallelism and acceleration

### 17.1 Ray
Use Ray for parallel orchestration of candidate evaluations.

Each candidate or candidate batch can run as a Ray task or actor.

### 17.2 MLflow
Use MLflow for:
- parent experiment run
- child runs for candidates
- metric logging
- artifact logging
- config snapshots
- software version tracking

### 17.3 GPU
If RAPIDS cuML is available:
- accelerate compatible steps
- keep interface consistent
- log when GPU path is used

### 17.4 Caching
Cache:
- preprocessed matrices
- PCA fits
- UMAP embeddings
- autoencoder latent spaces
- reusable graphs if appropriate

---

## 18. Reporting requirements

This is mandatory. The package must produce everything needed for a paper.

### 18.1 Global experiment outputs
- `experiment_manifest.json`
- `candidate_registry.parquet`
- `accepted_candidates.parquet`
- `rejected_candidates.parquet`
- `method_family_summary.csv`
- `runtime_summary.csv`

### 18.2 Per-candidate outputs
For every candidate:
- config snapshot
- metric summary
- per-seed labels
- per-perturbation labels
- cluster size summary
- QC report
- optional plotting coordinates

### 18.3 Consensus outputs
- `consensus_labels.csv`
- `consensus_soft_membership.parquet`
- `consensus_uncertainty.csv`
- `coassociation_matrix.parquet` or sparse equivalent
- `consensus_cluster_summary.csv`
- `consensus_bootstrap_stability.csv`

### 18.4 Interpretation outputs
- surrogate CV metrics
- SHAP value tables
- SHAP summary tables
- permutation importance table
- cluster profile tables
- pairwise cluster contrast tables

### 18.5 Figure-ready data
Export raw data used for figures.

Required figure datasets:

1. Search flow data
2. Method family acceptance summary
3. Stability vs quality scatter data
4. Accepted-candidate heatmap data
5. Consensus matrix plot data
6. Final embedding plot data
7. Cluster size/confidence data
8. Feature importance data
9. Clinical profile heatmap data

### 18.6 Manuscript bundle
Create:

```text
results/<experiment_name>/
  manuscript_bundle/
    figures/
    tables/
    supplementary/
    methods/
```

Populate with:
- figure-ready CSV or Parquet files
- table-ready CSV files
- supplementary registries
- auto-generated methods text
- config snapshot
- software versions

---

## 19. Visualization outputs

Generate plots, but always also export underlying data.

Required plots:
- search flow diagram
- accepted candidate metric heatmap
- quality vs stability scatter
- co-association matrix heatmap
- final embedding scatter
- uncertainty distribution by cluster
- SHAP summary plots
- cluster profile plots

Separate plotting code from computation code.

---

## 20. Interpretation module

### 20.1 Surrogate classifier
Default:
- XGBoost classifier

Alternative:
- RandomForest classifier

### 20.2 Validation
Use repeated stratified CV:
- 5 folds
- 3 repeats

Store:
- accuracy
- macro F1
- balanced accuracy
- confusion matrix

If the surrogate cannot predict the consensus clusters meaningfully, emit a warning that interpretability is weak.

### 20.3 Explainability
Implement:
- SHAP global summaries
- one-vs-rest SHAP per cluster
- grouped permutation importance for correlated features
- cluster profile statistics with effect sizes

### 20.4 Correlation handling
Before permutation importance:
- identify highly correlated features
- either group them or use grouped permutation

---

## 21. Config example

Use this as a template.

```yaml
experiment:
  name: "stroke_dataset_v1"
  output_dir: "./results/stroke_dataset_v1"
  random_seed: 2026
  n_jobs: 16
  use_ray: true
  use_mlflow: true
  use_gpu_if_available: true

data:
  path: "./data/stroke.csv"
  id_column: "patient_id"
  target_columns: []
  column_schema:
    continuous: ["age", "bmi", "glucose", "sbp", "dbp"]
    binary: ["sex_male", "smoker", "hypertension"]
    categorical: ["race", "insurance_type"]
    ordinal: []
  missingness:
    continuous_imputer: "median"
    categorical_imputer: "most_frequent"
    add_missing_indicators: true

search:
  pilot_sample_fraction: 0.35
  pilot_min_rows: 1500
  seeds_pilot: [101, 102]
  seeds_full: [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
  perturbations_full: 10
  perturbation_type: "bootstrap"
  optuna:
    enabled: true
    sampler: "TPESampler"
    pruner: "MedianPruner"
    n_trials_per_family: 200

preprocessing:
  continuous_transforms:
    - "none"
    - "standard"
    - "robust"
    - "power"
    - "log1p_standard"
  categorical_encoding:
    - "onehot"
  variance_threshold:
    enabled: true
    threshold: 0.0

representation:
  methods:
    - name: "none"
    - name: "pca"
      params:
        n_components: [5, 10, 20, 50]
        whiten: [false, true]
    - name: "umap"
      params:
        n_components: [5, 10, 20]
        n_neighbors: [10, 20, 50]
        min_dist: [0.0, 0.1, 0.3]
        metric: ["euclidean", "manhattan"]
    - name: "autoencoder"
      params:
        latent_dim: [5, 10, 20]
        hidden_layers:
          - [128, 64]
          - [256, 128, 64]
        dropout: [0.0, 0.1]
        epochs: 100
        batch_size: 256
        learning_rate: [0.001, 0.0003]
        early_stopping_patience: 10

clustering:
  methods:
    - name: "kmeans"
      params:
        n_clusters: [2, 3, 4, 5, 6, 7, 8]
    - name: "gmm"
      params:
        n_components: [2, 3, 4, 5, 6, 7, 8]
        covariance_type: ["full", "diag"]
    - name: "agglomerative"
      params:
        n_clusters: [2, 3, 4, 5, 6, 7, 8]
        linkage: ["ward", "complete", "average"]
    - name: "spectral"
      params:
        n_clusters: [2, 3, 4, 5, 6]
        affinity: ["nearest_neighbors", "rbf"]
    - name: "hdbscan"
      params:
        min_cluster_size: [20, 50, 100]
        min_samples: [null, 5, 10, 20]
    - name: "optics"
      params:
        min_samples: [5, 10, 20]
        xi: [0.03, 0.05]
    - name: "birch"
      params:
        n_clusters: [2, 3, 4, 5, 6]
        threshold: [0.3, 0.5, 0.7]
    - name: "ae_kmeans"
    - name: "ae_gmm"
    - name: "dec"
      params:
        n_clusters: [2, 3, 4, 5, 6]
        pretrain_epochs: 100
        finetune_epochs: 100
    - name: "vade"
      params:
        n_clusters: [2, 3, 4, 5, 6]
        epochs: 150

evaluation:
  internal_metrics:
    - "silhouette"
    - "davies_bouldin"
    - "calinski_harabasz"
  stability_metrics:
    - "ari_seed"
    - "nmi_seed"
    - "jaccard_cluster_boot"
    - "coassociation_consistency"
  structure_constraints:
    min_clusters: 2
    max_clusters: 10
    min_cluster_fraction: 0.03
    max_noise_fraction: 0.35
  acceptance:
    hard_thresholds:
      silhouette_min: 0.02
      ari_seed_min: 0.40
      nmi_seed_min: 0.50
      mean_cluster_jaccard_min: 0.60
    weighted_score:
      silhouette: 0.10
      davies_bouldin: 0.05
      calinski_harabasz: 0.05
      ari_seed: 0.25
      nmi_seed: 0.15
      mean_cluster_jaccard: 0.25
      cluster_balance: 0.05
      parsimony_penalty: -0.05
      runtime_penalty: -0.05
    accept_top_fraction_if_above: 0.15

consensus:
  include_only_accepted: true
  run_weighting:
    source: "final_weighted_score"
    normalize: true
    floor: 0.01
  consensus_method: "hierarchical_on_coassociation"
  final_k_strategy: "data_driven"
  uncertainty:
    bootstrap_repeats: 50

interpretation:
  surrogate_model: "xgboost"
  cross_validation_folds: 5
  repeated_cv_repeats: 3
  use_shap: true
  use_permutation_importance: true
  top_n_features: 30
  grouped_correlation_threshold: 0.85

reporting:
  generate_figures: true
  generate_tables: true
  export_format: ["csv", "parquet", "json", "png", "pdf"]
  manuscript_bundle: true
```

---

## 22. Development milestones

### Milestone 1
Implement:
- installable package
- CLI
- config validation
- preprocessing
- PCA and UMAP
- KMeans, GMM, HDBSCAN, Agglomerative
- seed stability
- perturbation stability
- early pruning
- acceptance engine
- consensus module
- reporting exports
- MLflow and Ray integration

### Milestone 2
Implement:
- autoencoder
- AE + clusterer wrappers
- DEC
- VaDE
- GPU path
- interpretation module
- SHAP outputs

### Milestone 3
Implement:
- resume/checkpoint polish
- benchmark suite
- synthetic datasets
- enhanced manuscript bundle
- more robust testing

---

## 23. Testing requirements

### Unit tests
- config validation
- compatibility rules
- metric correctness
- consensus calculations
- artifact export paths

### Integration tests
- small synthetic dataset
- mixed-type dataset
- missing-data dataset
- resume mode
- Milestone 1 end-to-end smoke test

### Regression tests
- deterministic mode should reproduce results within defined tolerance

---

## 24. Required engineering standards

Require Cursor to:
- use typed Python where practical
- use dataclasses or pydantic-style validation where useful
- keep modules small and composable
- separate compute from plotting
- log all important failures
- avoid notebook-centric design
- document public interfaces
- write tests with pytest
- use stable hashing for artifact names
- never silently swallow numerical failures

---

## 25. What Cursor should not do

Do **not**:
- include t-SNE in main clustering search
- evaluate only in embedding space
- rank candidates using one metric only
- directly average labels across runs
- hard-code data paths inside repo
- assume datasets live inside the package
- write monolithic scripts instead of package modules
- start with deep clustering before the classical framework is stable

---

## 26. Deliverables Cursor must produce

Cursor should deliver:

1. Full installable package
2. `pyproject.toml`
3. CLI entry point
4. Example configs
5. README with installation and usage
6. Minimal synthetic smoke-test example
7. Unit and integration tests
8. Clear milestone progress if implementing incrementally

---

## 27. Exact prompt to paste into Cursor

Paste the text below into Cursor:

```text
Build a Python package named `clustro` using a `src/` layout and `pyproject.toml`.

Goal:
Create an installable research package for stability-first clustering of tabular medical datasets. The package must be installable once and runnable from any other directory on arbitrary datasets, with outputs written to a user-defined results directory.

Core scientific behavior:
- evaluate curated preprocessing -> representation -> clustering pipelines
- use repeated seeds and perturbation resampling
- prune weak candidates early
- accept only robust candidates
- build a weighted co-association consensus clustering from accepted runs
- estimate sample-level uncertainty
- interpret final consensus clusters with a supervised surrogate model and SHAP
- export all artifacts needed for a scientific paper

Critical package behavior:
- datasets will live outside the package repo
- output directory is defined by config
- package must support `pip install -e .`
- CLI command must be `clustro`
- code must be modular, typed where practical, and tested

Implement this architecture:
- config
- data
- repr
- clustering
- search
- evaluation
- consensus
- interpretation
- reporting
- tracking
- utils

Representation methods:
- none
- PCA
- UMAP
- autoencoder
- t-SNE only for visualization, not main clustering search

Clustering methods:
- KMeans
- MiniBatchKMeans
- GaussianMixture
- AgglomerativeClustering
- SpectralClustering
- HDBSCAN
- OPTICS
- Birch
- AE+KMeans
- AE+GMM
- DEC
- VaDE

Scientific constraints:
- evaluate core metrics in processed original feature space by default
- use seed stability and bootstrap/subsample stability
- reject candidates with hard thresholds before ranking
- consensus must use weighted co-association, not direct label averaging
- final outputs must include hard labels, soft membership, and uncertainty
- interpretation must use surrogate validation before feature importance claims

Engineering constraints:
- Python 3.11+
- pyproject.toml
- src layout
- CLI entry point
- MLflow for experiment tracking
- Ray for orchestration and dashboard
- Optuna for search and pruning
- PyTorch for deep methods
- optional RAPIDS acceleration if available
- pytest test suite
- resume/checkpoint support

Deliver Milestone 1 first:
- installable package
- classical methods only
- preprocessing
- PCA and UMAP
- seed stability
- perturbation stability
- pruning
- acceptance engine
- consensus module
- reporting bundle
- CLI
- tests
- README

Then Milestone 2:
- autoencoder
- AE+clusterers
- DEC
- VaDE
- GPU support
- interpretation with SHAP

Export artifacts:
- candidate registry
- accepted/rejected candidate tables
- per-candidate metrics
- per-seed labels
- per-perturbation labels
- consensus labels
- consensus soft membership
- consensus uncertainty
- co-association matrix
- cluster summary tables
- surrogate CV results
- SHAP summaries
- figure-ready CSV/Parquet files
- manuscript bundle with tables, figures data, supplementary data, methods text

Use the detailed spec in this markdown file as the source of truth.
Start by creating the full project skeleton and Milestone 1 implementation plan, then implement Milestone 1 cleanly.
```

---

## 28. Minimal usage examples to support

### Editable install
```bash
cd ~/dev/clustro
pip install -e .
```

### Run from a separate dataset directory
```bash
cd ~/projects/stroke_analysis
clustro validate-config config.yaml
clustro run config.yaml
```

### Use package programmatically from another directory
```python
from clustro import Experiment

exp = Experiment.from_yaml("/home/user/projects/stroke_analysis/config.yaml")
exp.run()
```

---

## 29. README requirements

The README must include:
- project purpose
- scientific philosophy
- installation
- CLI usage
- config structure
- running from another directory
- output artifact structure
- milestone status
- limitations

---

## 30. Final instruction to Cursor

Do not optimize for speed of first code delivery at the expense of architecture. A fast ugly implementation will fail when deep methods, GPU support, and paper-grade reporting are added later.

Build Milestone 1 correctly first.
