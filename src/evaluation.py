"""
evaluation.py — Cross-validation helpers, metric computation, and MLflow logging.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    matthews_corrcoef,
    average_precision_score,
)
import mlflow

# Default CV strategy used throughout the project
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = ["roc_auc", "f1", "recall", "precision", "accuracy"]


def cv_evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv=None,
    scoring: list[str] | None = None,
    return_estimator: bool = False,
) -> dict[str, float]:
    """Run stratified k-fold CV and return mean ± std for each metric.

    Returns
    -------
    dict with keys like 'roc_auc_mean', 'roc_auc_std', 'f1_mean', etc.
    """
    cv = cv or CV
    scoring = scoring or SCORING

    results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_estimator=return_estimator,
        n_jobs=-1,
    )

    summary: dict[str, float] = {}
    for metric in scoring:
        key = f"test_{metric}"
        summary[f"{metric}_mean"] = float(np.mean(results[key]))
        summary[f"{metric}_std"] = float(np.std(results[key]))

    if return_estimator:
        summary["_estimators"] = results["estimator"]

    return summary


def log_mlflow_run(
    model_name: str,
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    experiment_name: str = "Heart-Disease-Kaggle",
) -> str:
    """Log a run to MLflow. Returns the run_id.

    Parameters
    ----------
    model_name : human-readable model label
    metrics    : output of cv_evaluate()
    params     : model hyperparameters to log
    tags       : extra tags (phase, feature_set, etc.)
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name) as run:
        if params:
            mlflow.log_params(params)
        for k, v in metrics.items():
            if not k.startswith("_"):
                mlflow.log_metric(k, v)
        if tags:
            mlflow.set_tags(tags)
        mlflow.set_tag("model_name", model_name)
        return run.info.run_id


def compute_metrics_from_proba(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute full metric suite from predicted probabilities."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Return threshold that maximises Youden's J statistic (sensitivity + specificity - 1)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])
