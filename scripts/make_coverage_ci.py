"""Coverage uncertainty table (reviewer request): for every (dataset, model) split-conformal
result, report k/n and a Wilson 95% CI on empirical coverage, so a 'below-nominal' point estimate
is read against its sampling error. Conformal's guarantee is marginal finite-sample under
exchangeability -- a single finite test split can land below 0.90 by chance, which the CI shows.

Reads results/final/conformal_full.csv; writes results/final/coverage_ci.csv.
Run: /opt/anaconda3/bin/python3 scripts/make_coverage_ci.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

FINAL = Path("results/final")

# Predefined-split test sizes (drug-like) + random-split test sizes (QM), and calibration (val)
# sizes, taken from the actual splits used by the conformal sweep.
N_TEST = {"esol": 227, "freesolv": 129, "lipophilicity": 840, "bace": 303, "qm7": 684, "qm8": 2178}
N_CAL = {"esol": 112, "freesolv": 64, "lipophilicity": 420, "bace": 151, "qm7": 683, "qm8": 2176}
ALPHA = 0.1


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main() -> None:
    d = pd.read_csv(FINAL / "conformal_full.csv")
    rows = []
    for _, r in d.iterrows():
        ds = r["dataset"]
        n = N_TEST.get(ds)
        if n is None:
            continue
        cov = float(r["empirical_coverage"])
        k = round(cov * n)
        lo, hi = wilson(k, n)
        ncal = N_CAL.get(ds, float("nan"))
        # finite-sample conformal order-statistic index k* = ceil((n_cal+1)(1-alpha))
        qidx = math.ceil((ncal + 1) * (1 - ALPHA)) if ncal == ncal else float("nan")
        rows.append({
            "dataset": ds, "model": r["model"], "n_test": n, "covered": k,
            "coverage": round(cov, 3), "ci95_lo": round(lo, 3), "ci95_hi": round(hi, 3),
            "cal_n": int(ncal) if ncal == ncal else None,
            "q_index": int(qidx) if qidx == qidx else None,
        })
    out = pd.DataFrame(rows)
    out.to_csv(FINAL / "coverage_ci.csv", index=False)
    n_below = ((out.coverage < 0.90) & (out.ci95_hi >= 0.90)).sum()
    n_sig_below = (out.ci95_hi < 0.90).sum()
    print(f"wrote {FINAL / 'coverage_ci.csv'}  ({len(out)} rows)")
    print(f"  point<0.90 but CI includes 0.90 (not sig. invalid): {n_below}")
    print(f"  CI entirely <0.90 (significantly below nominal):     {n_sig_below}")
    print(out[out.coverage < 0.90][["dataset", "model", "n_test", "covered", "coverage",
                                    "ci95_lo", "ci95_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
