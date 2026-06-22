"""Evaluation metrics.

Per CLAUDE.md §9: regression must report MAE + RMSE + Pearson R + Spearman rho.
Classification must report ROC-AUC + PR-AUC + F1 + MCC. Reporting fewer is a
reviewer red flag.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def regression_metrics(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
) -> dict[str, float]:
    """Return MAE, RMSE, Pearson R, Spearman rho. CLAUDE.md §9."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if len(y_true) < 2:
        raise ValueError("need >= 2 samples for correlation metrics")

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "PearsonR": float(pearsonr(y_true, y_pred)[0]),
        "SpearmanRho": float(spearmanr(y_true, y_pred)[0]),
    }


def classification_metrics(
    y_true: NDArray[np.int_],
    y_score: NDArray[np.float64],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return ROC-AUC, PR-AUC, F1, MCC. CLAUDE.md §9.

    Args:
        y_true: Binary labels {0, 1}.
        y_score: Predicted probabilities for the positive class.
        threshold: Decision threshold for F1/MCC. Default 0.5; consider tuning
            on the validation set for BBBP-style imbalanced tasks.
    """
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_score.shape}")
    unique = np.unique(y_true)
    if not set(unique.tolist()).issubset({0, 1}):
        raise ValueError(f"y_true must be binary 0/1, got values {unique}")

    y_pred = (y_score >= threshold).astype(int)
    return {
        "ROC-AUC": float(roc_auc_score(y_true, y_score)),
        "PR-AUC": float(average_precision_score(y_true, y_score)),
        "F1": float(f1_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def metrics_for_task(
    task: str,
    y_true: NDArray[np.float64],
    y_pred_or_score: NDArray[np.float64],
) -> dict[str, float]:
    """Dispatch to regression / classification metrics by task name."""
    if task == "regression":
        return regression_metrics(y_true, y_pred_or_score)
    if task == "classification":
        return classification_metrics(y_true.astype(int), y_pred_or_score)
    raise ValueError(f"unknown task: {task!r}")
