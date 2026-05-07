# clustro Implementation Status

This document reflects current package behavior relative to scientific-repair goals, reporting, and CI.

## Core pipeline (stable)

- Installable `src/` layout, `pyproject.toml`, and the `clustro` CLI.
- Config-driven runs with paths resolved relative to the config file.
- Tabular schema: continuous, binary, categorical, and ordinal features; missingness, scaling/transforms, encoding, variance filtering, optional rare-category collapsing.
- Representations: none, PCA, UMAP, autoencoder (with disk cache keyed by method, params, matrix fingerprint, and seed).
- Clustering: classical (KMeans, MiniBatchKMeans, GaussianMixture, Agglomerative, Spectral, HDBSCAN, OPTICS, Birch) and deep (AE+KMeans, AE+GMM, DEC, VaDE) with optional RAPIDS/cuML where supported.
- Search: compatibility checks, pilot screening, full evaluation, hard acceptance, top-fraction policy, ranking.
- Optuna: when `search.optuna.enabled` is true, objectives use `trial.suggest_categorical` (and related `suggest_*` calls) for preprocessing knobs—e.g. `continuous_transform`, `categorical_encoding`, `representation`—plus clustering hyperparameters; pruning semantics are preserved where wired.
- Ray: batch candidate evaluation when Optuna is off and Ray is on.
- Resume/checkpoints at stage and candidate granularity.

## Row-identity perturbation stability

- `perturbation_type` supports `bootstrap` and `subsample`.
- Stability (`mean_cluster_jaccard`, `perturbation_rows_compared_mean`) compares reference labels to perturbation labels on **aligned original row indices** (`PerturbationLabelRun.indices`), not label-only shuffles.
- Subsample runs require unique indices; bootstrap uses index→label mapping so duplicate draws collapse to a single label per original row before overlap with the reference.

## Utility metric scoring vs raw thresholds

- **Hard gates** in `evaluation.acceptance.hard_thresholds` apply to **raw** metric columns (`*_min` keys map to the corresponding metric).
- **Weighted ranking** converts metrics to **utilities** via `metric_to_utility` (direction, bounds, optional log transforms); `final_weighted_score` is the weighted sum. Calinski–Harabasz utilities can be percentile-ranked across candidates when multiple values exist.
- Registry outputs expose `utility_*` columns where applicable for audit trails.

## Categorical encoding in the candidate graph and caches

- The search Cartesian product includes each configured `categorical_encoding` together with continuous transforms so encoding is a first-class axis.
- With Optuna enabled, `categorical_encoding` is suggested and folded into candidate / graph keys (`_preprocessing_key`, edge deduplication).
- Representation cache keys use the preprocessed matrix fingerprint, which reflects the chosen encoding path.

## Co-association dense guard (`max_dense_n`)

- `consensus.max_dense_n` (default 10000) limits sample count when the code path would allocate a **dense** \(n \times n\) co-association matrix and `coassociation_storage` is `auto` or `sparse`: over the cap, construction raises a clear error (sparse/blockwise assembly is not implemented). Choosing `coassociation_storage: dense` skips this guard—use only when intentionally allocating the full dense matrix.

## Uncertainty columns and thresholds

- `consensus_uncertainty.csv` / soft membership carry entropy, top probability, top-2 gap, and an **`ambiguous`** flag driven by `consensus.uncertainty.ambiguous_top2_gap_threshold` and `ambiguous_entropy_quantile`; bootstrap stability tables accompany consensus exports.
- Cluster-level summaries feed report tables and manuscript artifacts (e.g. `cluster_size_confidence.csv`, uncertainty figures).

## Search flow reporting (expanded CSV + JSON)

- When the candidate registry is **non-empty**, `reports/search_flow.csv` and `reports/search_flow.json` are emitted together; stages include `generated_total`, `compatibility_rejected`, `pilot_pruned`, `full_evaluated`, `hard_rejected`, `accepted_before_top_fraction`, `accepted_final`, `top_fraction_rejected`, and `consensus_used` (counts align with `_build_search_flow_frame`).
- `search_flow_diagram.png` is produced in the same branch.

## Interpretation `feature_space` JSON export

- At interpretation start, `interpretation/interpretation_feature_space.json` records the resolved `feature_space` mode (`original_imputed_scaled`, `best_candidate_preprocessing`, `consensus_majority_preprocessing`), effective `continuous_transform` and `categorical_encoding`, optional `source_candidate_id`, and rationale text for methods reproducibility.

## Deterministic SHAP subsampling

- `compute_shap_summary` caps rows at `max_rows`, draws indices with `numpy.random.default_rng(random_seed)` without replacement, then **sorts** indices before building/explaining so SHAP inputs are reproducible for fixed seed and dataset order; optional `row_ids` are written into SHAP detail output.

## Manuscript bundle and supplementary

- Bundle population copies figures, tables, and supplementary artifacts including `interpretation_feature_space.json`, `search_flow.json` (when present under `reports/`), registries, consensus outputs, and co-association parquet when present.

## CI workflow

- `.github/workflows/ci.yml` runs on push and pull request: checkout, Python 3.11, editable install with dev extras, `pytest`, then `ruff check .`.

## Remaining limitations

- RAPIDS acceleration is limited to compatible classical paths and optional cuML availability.
- Ray applies to non-Optuna batches; Optuna studies remain sequential per design where ask/tell semantics matter.
- Very large \(n\) with `auto`/`sparse` storage still hits `max_dense_n` until sparse co-association exists, or use explicit `dense` with acceptable memory.
- Deep clustering remains dataset-sensitive in real cohorts.
