"""Permutation importance helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score


def compute_full_fit_permutation_importance(
    estimator: object,
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    *,
    random_seed: int,
) -> pd.DataFrame:
    """Permutation importance on the same data used to fit the final surrogate.

    This is an exploratory diagnostic only. For manuscript interpretation
    use compute_cv_permutation_importance instead, which evaluates on held-out
    folds and is not optimistic due to training-set overlap.
    """
    result = permutation_importance(
        estimator,
        matrix,
        labels,
        n_repeats=10,
        random_state=random_seed,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def compute_permutation_importance(
    estimator: object,
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    *,
    random_seed: int,
) -> pd.DataFrame:
    """Deprecated. Use compute_full_fit_permutation_importance for the exploratory
    full-fit version, or compute_cv_permutation_importance for manuscript results."""
    warnings.warn(
        "compute_permutation_importance is deprecated. "
        "Use compute_full_fit_permutation_importance (exploratory) or "
        "compute_cv_permutation_importance (primary, manuscript-quality). "
        "The full-fit version can be optimistic because importance is measured "
        "on the same data used to train the surrogate.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_full_fit_permutation_importance(
        estimator, matrix, labels, feature_names, random_seed=random_seed
    )


def build_correlation_groups(
    matrix: np.ndarray,
    feature_names: list[str],
    *,
    threshold: float,
) -> pd.DataFrame:
    if matrix.shape[1] == 0:
        return pd.DataFrame(columns=["group_id", "feature"])
    if matrix.shape[1] == 1:
        return pd.DataFrame([{"group_id": 0, "feature": feature_names[0]}])

    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    adjacency = np.abs(corr) >= threshold
    np.fill_diagonal(adjacency, True)

    seen: set[int] = set()
    groups: list[dict[str, object]] = []
    group_id = 0
    for start in range(len(feature_names)):
        if start in seen:
            continue
        stack = [start]
        members: list[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            members.append(current)
            neighbors = np.flatnonzero(adjacency[current]).tolist()
            stack.extend(neighbors)
        for member in sorted(members):
            groups.append({"group_id": group_id, "feature": feature_names[member]})
        group_id += 1
    return pd.DataFrame(groups)


def compute_grouped_permutation_importance(
    estimator: object,
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    group_frame: pd.DataFrame,
    *,
    random_seed: int,
    n_repeats: int = 10,
) -> pd.DataFrame:
    if group_frame.empty:
        return pd.DataFrame(
            columns=["group_id", "features", "group_size", "importance_mean", "importance_std"]
        )

    rng = np.random.default_rng(random_seed)
    baseline = float(accuracy_score(labels, estimator.predict(matrix)))
    rows: list[dict[str, object]] = []
    for group_id, subset in group_frame.groupby("group_id", sort=True):
        columns = [feature_names.index(feature) for feature in subset["feature"].tolist()]
        scores: list[float] = []
        for _ in range(n_repeats):
            shuffled = matrix.copy()
            order = rng.permutation(matrix.shape[0])
            shuffled[:, columns] = shuffled[order][:, columns]
            score = float(accuracy_score(labels, estimator.predict(shuffled)))
            scores.append(baseline - score)
        rows.append(
            {
                "group_id": int(group_id),
                "features": ";".join(subset["feature"].tolist()),
                "group_size": int(len(columns)),
                "importance_mean": float(np.mean(scores)),
                "importance_std": float(np.std(scores)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def compute_cv_permutation_importance(
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    config,
    *,
    random_seed: int,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Compute fold-wise held-out permutation importance for surrogate interpretation."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    from clustro.interpretation.surrogate import build_surrogate_estimator

    splitter = RepeatedStratifiedKFold(
        n_splits=config.cross_validation_folds,
        n_repeats=config.repeated_cv_repeats,
        random_state=random_seed,
    )
    per_feature: dict[str, list[float]] = {feature: [] for feature in feature_names}
    fold_count = 0
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(matrix, labels)):
        model = build_surrogate_estimator(
            config,
            random_seed=random_seed + fold_index,
            n_classes=int(np.unique(labels[train_idx]).size),
        )
        model.fit(matrix[train_idx], labels[train_idx])
        result = permutation_importance(
            model,
            matrix[test_idx],
            labels[test_idx],
            n_repeats=n_repeats,
            random_state=random_seed + fold_index,
            n_jobs=1,
        )
        for feature, value in zip(feature_names, result.importances_mean, strict=True):
            per_feature[feature].append(float(value))
        fold_count += 1

    rows = [
        {
            "feature": feature,
            "importance_mean": float(np.mean(values)) if values else 0.0,
            "importance_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "fold_count": fold_count,
        }
        for feature, values in per_feature.items()
    ]
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
