"""Split conformal prediction for molecular regression.

Novelty thrust (项目调研汇总.md v2 §7.5.1): which molecular representation
gives the TIGHTEST, coverage-valid prediction intervals? Distribution-free
split conformal gives a finite-sample marginal coverage guarantee
(Vovk 2005; Lei et al. 2018), so a model that is over-confident still yields
VALID intervals -- the question is which backbone yields the NARROWEST ones.

Three variants:
  - split          : interval = pred ± q,  q = quantile of |y - pred| on cal
  - normalized      : interval = pred ± q·sigma(x), sigma = per-input uncertainty
                      (e.g., RF tree-std). Adapts width to local difficulty.
  - mondrian        : per-group quantile (e.g., by applicability-domain stratum)

Data discipline (CLAUDE.md §16.4): cal = validation split, eval = test split.
This is the ONE-TIME final test evaluation, not iterative tuning.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _conformal_quantile(scores: NDArray[np.float64], alpha: float) -> float:
    """Finite-sample-adjusted (1-alpha) quantile of nonconformity scores."""
    n = len(scores)
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def split_conformal(
    cal_y: NDArray[np.float64],
    cal_pred: NDArray[np.float64],
    test_pred: NDArray[np.float64],
    alpha: float = 0.1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Standard split conformal. Returns (lower, upper, q)."""
    scores = np.abs(cal_y - cal_pred)
    q = _conformal_quantile(scores, alpha)
    return test_pred - q, test_pred + q, q


def normalized_conformal(
    cal_y: NDArray[np.float64],
    cal_pred: NDArray[np.float64],
    cal_sigma: NDArray[np.float64],
    test_pred: NDArray[np.float64],
    test_sigma: NDArray[np.float64],
    alpha: float = 0.1,
    eps: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Locally-adaptive (normalized) split conformal.

    Nonconformity = |y - pred| / (sigma + eps). Intervals scale with the
    per-input uncertainty estimate sigma -> narrower where the model is
    confident, wider where it is not.
    """
    scores = np.abs(cal_y - cal_pred) / (cal_sigma + eps)
    q = _conformal_quantile(scores, alpha)
    half = q * (test_sigma + eps)
    return test_pred - half, test_pred + half, q


def mondrian_conformal(
    cal_y: NDArray[np.float64],
    cal_pred: NDArray[np.float64],
    cal_group: NDArray[np.int_],
    test_pred: NDArray[np.float64],
    test_group: NDArray[np.int_],
    alpha: float = 0.1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Group-conditional (Mondrian) conformal: separate quantile per group.

    Use e.g. applicability-domain strata as groups -> coverage holds WITHIN
    each stratum, not just marginally.
    """
    lower = np.empty_like(test_pred)
    upper = np.empty_like(test_pred)
    for g in np.unique(test_group):
        cal_mask = cal_group == g
        if cal_mask.sum() == 0:
            # No calibration data for this group -> fall back to global q.
            q = _conformal_quantile(np.abs(cal_y - cal_pred), alpha)
        else:
            q = _conformal_quantile(np.abs(cal_y[cal_mask] - cal_pred[cal_mask]), alpha)
        test_mask = test_group == g
        lower[test_mask] = test_pred[test_mask] - q
        upper[test_mask] = test_pred[test_mask] + q
    return lower, upper


def coverage(test_y, lower, upper) -> float:
    """Empirical coverage: fraction of test targets inside the interval."""
    return float(((test_y >= lower) & (test_y <= upper)).mean())


def mean_width(lower, upper) -> float:
    """Mean interval width (lower = tighter, at matched coverage)."""
    return float((upper - lower).mean())


def evaluate_conformal(
    cal_y, cal_pred, test_y, test_pred, alpha: float = 0.1
) -> dict[str, float]:
    """Run split conformal and report coverage + width at one alpha."""
    lo, hi, q = split_conformal(cal_y, cal_pred, test_pred, alpha)
    return {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "empirical_coverage": coverage(test_y, lo, hi),
        "mean_width": mean_width(lo, hi),
        "q": q,
    }
