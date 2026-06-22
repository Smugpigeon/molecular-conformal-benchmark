"""Conformalized Quantile Regression (CQR; Romano, Patterson, Candès, NeurIPS 2019).

Unlike split conformal (constant-width intervals), CQR conformalizes a
quantile regressor to give ADAPTIVE-width, coverage-valid intervals:

  1. Train a model to predict the lower/upper quantiles q_lo(x), q_hi(x).
  2. On the calibration set, conformity score E_i = max(q_lo(x_i) - y_i,
     y_i - q_hi(x_i))  (how far y falls outside the predicted band).
  3. Q = (1-alpha) finite-sample quantile of {E_i}.
  4. Test interval = [q_lo(x) - Q, q_hi(x) + Q].

This restores marginal coverage while keeping the band narrow where the model
is confident. We compare CQR vs split conformal per backbone.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cqr_calibrate(
    cal_y: NDArray[np.float64],
    cal_lo: NDArray[np.float64],
    cal_hi: NDArray[np.float64],
    alpha: float = 0.1,
) -> float:
    """Return the CQR conformity quantile Q from the calibration set."""
    e = np.maximum(cal_lo - cal_y, cal_y - cal_hi)
    n = len(e)
    if n == 0:
        return 0.0
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(e, level, method="higher"))


def cqr_intervals(
    test_lo: NDArray[np.float64], test_hi: NDArray[np.float64], q: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply the calibrated correction Q to test quantile predictions."""
    return test_lo - q, test_hi + q


def evaluate_cqr(
    cal_y, cal_lo, cal_hi, test_y, test_lo, test_hi, alpha: float = 0.1
) -> dict[str, float]:
    """Run CQR and report coverage + mean width."""
    q = cqr_calibrate(cal_y, cal_lo, cal_hi, alpha)
    lo, hi = cqr_intervals(test_lo, test_hi, q)
    cov = float(((test_y >= lo) & (test_y <= hi)).mean())
    width = float((hi - lo).mean())
    return {"coverage": cov, "width": width, "q": q,
            "alpha": alpha, "target_coverage": 1 - alpha}
