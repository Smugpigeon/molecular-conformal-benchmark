"""Paired statistical tests for model A vs model B.

Per CLAUDE.md §16.4: any "model A outperforms model B" claim in paper must
be backed by:
  - paired permutation test (regression) with n_perm >= 10000
  - bootstrap (or DeLong) test for ROC-AUC comparison
  - Bonferroni correction when comparing >1 model pair simultaneously

This module exposes both a library API (for use in notebooks) and a CLI
that takes two CSVs of predictions and runs the appropriate test.

Saving predictions:
  Currently src/train.py only saves metrics. To use this module, models
  must also dump predictions to {run_dir}/val_predictions.csv with
  columns [smiles, y_true, y_pred]. See TODO at end of file.

Library use:
    from scripts.statistical_tests import paired_permutation_test_regression
    obs_a, obs_b, p = paired_permutation_test_regression(
        y_true, pred_a, pred_b, metric="rmse"
    )

CLI use:
    python scripts/statistical_tests.py compare \\
        --y-true val.csv:label \\
        --a runs/molformer_esol/val_predictions.csv:pred \\
        --b runs/chemfm_esol/val_predictions.csv:pred \\
        --metric rmse
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _metric_fn(name: str):
    """Lower-is-better unless `pearsonr_diff`/`auc` -> higher-is-better."""
    if name == "mae":
        return lambda y, p: mean_absolute_error(y, p), False
    if name == "rmse":
        return lambda y, p: float(np.sqrt(mean_squared_error(y, p))), False
    if name == "pearsonr_diff":
        return lambda y, p: float(pearsonr(y, p)[0]), True
    if name == "auc":
        return lambda y, p: float(roc_auc_score(y, p)), True
    raise ValueError(f"Unknown metric: {name}")


def paired_permutation_test_regression(
    y_true: NDArray[np.float64],
    pred_a: NDArray[np.float64],
    pred_b: NDArray[np.float64],
    metric: str = "rmse",
    n_perm: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Two-sided paired permutation test.

    Returns (metric_a, metric_b, p_value). p < 0.05 implies the difference
    between A and B is unlikely under H0 (same predictive ability).
    """
    fn, _higher_better = _metric_fn(metric)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if not (len(pred_a) == len(pred_b) == n):
        raise ValueError("y_true, pred_a, pred_b must have same length")

    obs_a = fn(y_true, pred_a)
    obs_b = fn(y_true, pred_b)
    obs_diff = abs(obs_a - obs_b)

    null_diffs = np.zeros(n_perm)
    for i in range(n_perm):
        # For each sample, randomly swap which model's prediction it gets.
        swap = rng.random(n) < 0.5
        perm_a = np.where(swap, pred_b, pred_a)
        perm_b = np.where(swap, pred_a, pred_b)
        null_diffs[i] = abs(fn(y_true, perm_a) - fn(y_true, perm_b))

    p = float((null_diffs >= obs_diff).mean())
    return obs_a, obs_b, p


def bootstrap_test_classification(
    y_true: NDArray[np.int_],
    score_a: NDArray[np.float64],
    score_b: NDArray[np.float64],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap-resampled paired AUC comparison.

    Approximation of DeLong's test. Returns (auc_a, auc_b, p_value).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    auc_a = roc_auc_score(y_true, score_a)
    auc_b = roc_auc_score(y_true, score_b)
    obs_diff = auc_a - auc_b

    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        d = roc_auc_score(y_true[idx], score_a[idx]) - roc_auc_score(y_true[idx], score_b[idx])
        boot_diffs.append(d)
    boot_diffs = np.array(boot_diffs)
    # Two-sided p
    p = 2.0 * min(
        float((boot_diffs >= obs_diff).mean()),
        float((boot_diffs <= obs_diff).mean()),
    )
    return auc_a, auc_b, p


def bonferroni_correct(p_values: list[float], alpha: float = 0.05) -> tuple[float, list[bool]]:
    """Return (corrected alpha, [significant?] per p)."""
    threshold = alpha / max(1, len(p_values))
    return threshold, [p < threshold for p in p_values]


# ---------------------------------------------------------------------------
# CLI: takes two CSVs of (y_true, pred) and runs the appropriate test.
# Predictions must be saved by training code first (TODO below).
# ---------------------------------------------------------------------------

def _load_column(spec: str) -> NDArray[np.float64]:
    """Spec format `path/to/file.csv:colname`."""
    path, _, col = spec.partition(":")
    df = pd.read_csv(path)
    if col not in df.columns:
        raise KeyError(f"{path} has no column {col!r}; available: {list(df.columns)}")
    return df[col].to_numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("compare", help="Compare two prediction CSVs.")
    p.add_argument("--y-true", required=True, help="path:column")
    p.add_argument("--a", required=True, help="path:column for model A predictions")
    p.add_argument("--b", required=True, help="path:column for model B predictions")
    p.add_argument("--metric", default="rmse",
                   choices=["mae", "rmse", "pearsonr_diff", "auc"])
    p.add_argument("--n-perm", type=int, default=10000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    y = _load_column(args.y_true)
    a = _load_column(args.a)
    b = _load_column(args.b)

    if args.metric == "auc":
        m_a, m_b, p = bootstrap_test_classification(y.astype(int), a, b)
    else:
        m_a, m_b, p = paired_permutation_test_regression(
            y, a, b, metric=args.metric, n_perm=args.n_perm
        )

    print(f"metric={args.metric}")
    print(f"  A: {m_a:.4f}")
    print(f"  B: {m_b:.4f}")
    print(f"  diff: {m_a - m_b:+.4f}")
    print(f"  p-value: {p:.4f}")
    if p < 0.05:
        print("  -> SIGNIFICANT at α=0.05 (uncorrected). Apply Bonferroni if multi-compare.")
    else:
        print("  -> NOT significant. CLAUDE.md §16.4: do not write 'outperforms'.")
    return 0


# ---------------------------------------------------------------------------
# TODO: extend src/train.py to also save val_predictions.csv with columns
#   [smiles, y_true, y_pred]
# This allows comparison without retraining. Minimal change in train.py:
#
#     val_df = split.val.copy()
#     val_df['y_pred'] = y_pred_val
#     val_df.to_csv(run_dir / 'val_predictions.csv', index=False)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.exit(main())
