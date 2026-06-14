"""Surrogate models for consensus-cluster interpretation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from clustro.config.schema import InterpretationConfig


@dataclass(slots=True)
class SurrogateResult:
    estimator: object
    cv_metrics: pd.DataFrame
    confusion: pd.DataFrame
    mean_metrics: dict[str, float]
    feature_names: list[str]
    warning: str | None


def fit_surrogate_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    config: InterpretationConfig,
    *,
    random_seed: int,
) -> SurrogateResult:
    n_classes = int(np.unique(labels).size)
    estimator = build_surrogate_estimator(config, random_seed=random_seed, n_classes=n_classes)

    _, class_counts = np.unique(labels, return_counts=True)
    min_class_size = int(class_counts.min())
    effective_folds = min(config.cross_validation_folds, min_class_size)

    if effective_folds < 2:
        # Cannot do stratified k-fold: at least one cluster has fewer than 2 samples.
        estimator.fit(matrix, labels)
        classes = np.unique(labels)
        empty_conf = pd.DataFrame(
            np.zeros((len(classes), len(classes)), dtype=int),
            index=[f"actual_{c}" for c in classes],
            columns=[f"pred_{c}" for c in classes],
        ).reset_index(names="actual_label")
        return SurrogateResult(
            estimator=estimator,
            cv_metrics=pd.DataFrame(
                columns=["fold", "accuracy", "macro_f1", "balanced_accuracy"]
            ),
            confusion=empty_conf,
            mean_metrics={
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "balanced_accuracy": float("nan"),
            },
            feature_names=feature_names,
            warning=(
                f"CV skipped: smallest cluster has {min_class_size} sample(s), "
                "fewer than the 2 required for stratified k-fold. "
                "Feature importance from CV is unavailable."
            ),
        )

    splitter = RepeatedStratifiedKFold(
        n_splits=effective_folds,
        n_repeats=config.repeated_cv_repeats,
        random_state=random_seed,
    )
    rows = []
    truth: list[int] = []
    preds: list[int] = []
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(matrix, labels)):
        model = build_surrogate_estimator(
            config,
            random_seed=random_seed + fold_index,
            n_classes=int(np.unique(labels[train_idx]).size),
        )
        model.fit(matrix[train_idx], labels[train_idx])
        predictions = model.predict(matrix[test_idx])
        truth.extend(labels[test_idx].tolist())
        preds.extend(predictions.tolist())
        rows.append(
            {
                "fold": fold_index,
                "accuracy": float(accuracy_score(labels[test_idx], predictions)),
                "macro_f1": float(f1_score(labels[test_idx], predictions, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(labels[test_idx], predictions)),
            }
        )
    cv_metrics = pd.DataFrame(rows)
    estimator.fit(matrix, labels)
    classes = np.unique(labels)
    confusion = pd.DataFrame(
        confusion_matrix(truth, preds, labels=classes),
        index=[f"actual_{label}" for label in classes],
        columns=[f"pred_{label}" for label in classes],
    ).reset_index(names="actual_label")
    mean_metrics = {
        column: float(cv_metrics[column].mean())
        for column in ["accuracy", "macro_f1", "balanced_accuracy"]
    }
    warning: str | None = None
    if effective_folds < config.cross_validation_folds:
        warning = (
            f"CV folds reduced from {config.cross_validation_folds} to {effective_folds} "
            f"because the smallest cluster has only {min_class_size} sample(s)."
        )
    elif mean_metrics["macro_f1"] < 0.6 or mean_metrics["balanced_accuracy"] < 0.6:
        warning = "Surrogate performance is weak; feature importance should be treated cautiously."
    return SurrogateResult(
        estimator=estimator,
        cv_metrics=cv_metrics,
        confusion=confusion,
        mean_metrics=mean_metrics,
        feature_names=feature_names,
        warning=warning,
    )


def build_surrogate_estimator(config: InterpretationConfig, *, random_seed: int, n_classes: int):
    if config.surrogate_model == "xgboost":
        try:
            from xgboost import XGBClassifier

            objective = "binary:logistic" if n_classes <= 2 else "multi:softprob"
            kwargs = {
                "random_state": random_seed,
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "objective": objective,
                "eval_metric": "mlogloss",
            }
            if n_classes > 2:
                kwargs["num_class"] = n_classes
            return XGBClassifier(
                **kwargs,
            )
        except ImportError:
            pass
    return RandomForestClassifier(
        random_state=random_seed,
        n_estimators=300,
        class_weight="balanced",
    )
