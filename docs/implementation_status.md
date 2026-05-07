# clustro Implementation Status

This document maps the original Cursor handoff goals to the current package implementation.

## Implemented

- Installable `src/` package with `pyproject.toml` and `clustro` CLI.
- Config-driven runs with relative paths resolved from the config file location.
- Explicit tabular schema support for continuous, binary, categorical, and ordinal features.
- Missingness handling, scaling/transforms, encoding, variance filtering, and optional rare-category collapsing.
- Main representation methods: none, PCA, UMAP, and autoencoder.
- Classical clusterers: KMeans, MiniBatchKMeans, GaussianMixture, Agglomerative, Spectral, HDBSCAN, OPTICS, and Birch.
- Deep clusterers: AE+KMeans, AE+GMM, DEC, and VaDE.
- Compatibility checks, pilot screening, full evaluation, hard acceptance, top-fraction accepted-set policy, and ranking.
- Optuna per-family candidate trials and pruning hooks.
- Ray-backed candidate batch execution for non-Optuna runs.
- Optional RAPIDS/cuML acceleration for compatible classical methods with safe scikit-learn fallback.
- Seed and perturbation stability metrics, consensus clustering, sample uncertainty, and bootstrap consensus stability.
- Configurable consensus method and final-k strategy.
- Supervised surrogate interpretation, confusion matrices, permutation/grouped permutation importance, SHAP summaries, cluster profiles, and pairwise cluster contrasts.
- Figure-ready report data and plots, including visualization-only t-SNE embedding outputs.
- Populated manuscript bundle with methods text, software versions, tables, figures, and supplementary artifacts.
- Stage and candidate-level resume/checkpoint behavior.
- Synthetic benchmark utilities and pytest coverage.

## Remaining Limitations

- RAPIDS acceleration is limited to compatible classical methods and depends on optional cuML availability.
- Ray orchestration is used for non-Optuna candidate batches; Optuna remains sequential to preserve study ask/tell semantics.
- Representation cache utilities are available but are not automatically applied to every representation path.
- Deep clustering methods should still be calibrated carefully on each real medical dataset.
