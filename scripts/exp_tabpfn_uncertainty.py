"""A7  Does TabPFN's uncertainty flag the hard molecules in advance? + conformal calibration
(connects to the conformal-prediction thread). Server-side (GPU); writes per-test CSV, figures local.

Methods at nominal 90% (alpha=0.1), calibrated on validation, evaluated on test:
  * TabPFN native    [q05, q95] straight from TabPFN -- often miscalibrated
  * split-conformal  TabPFN mean +/- qhat,  qhat = conformal quantile of |val residual| (constant width)
  * CQR              conformalized TabPFN quantiles (variable width, Romano et al. 2019)
  * RF split-conformal  classic baseline (RF mean +/- qhat)
Per-test uncertainty = TabPFN interval width (q95-q05); we later test if it ranks the hard molecules.

Run on server: CUDA_VISIBLE_DEVICES=6 python scripts/exp_tabpfn_uncertainty.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_tabpfn import featurize as feat_desc  # ~200 descriptors  # noqa: E402
from scripts.tune_classic_ml import featurize as feat_morgan  # Morgan-2048  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402

SEED = 42
ALPHA = 0.1  # nominal 90% intervals
warnings.filterwarnings("ignore")
OUT = Path("results/exp"); OUT.mkdir(parents=True, exist_ok=True)


def _shim():
    import sklearn.utils.validation as skv
    if not hasattr(skv, "_is_pandas_df"):
        skv._is_pandas_df = lambda x: isinstance(x, pd.DataFrame)


def conf_quantile(scores, alpha):
    """Finite-sample split-conformal quantile."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(k, n)
    return float(np.sort(scores)[k - 1])


def cover_width(lo, hi, y):
    return float(np.mean((y >= lo) & (y <= hi))), float(np.mean(hi - lo))


def main():
    _shim()
    import torch
    from sklearn.ensemble import RandomForestRegressor
    from tabpfn import TabPFNRegressor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_all_seeds(SEED)

    df = pd.read_csv("data/processed/bace_clean.csv")
    sp = {k: df[df.set == k] for k in ["train", "validation", "test"]}
    yv = sp["validation"].label.to_numpy(float); yt = sp["test"].label.to_numpy(float)
    Xd = {k: feat_desc(sp[k].smiles.tolist(), "desc") for k in sp}
    ytr = sp["train"].label.to_numpy(float)

    reg = TabPFNRegressor(device=device, random_state=SEED, ignore_pretraining_limits=True)
    reg.fit(Xd["train"], ytr)

    def mq(X):
        mean = np.asarray(reg.predict(X), float)
        q = reg.predict(X, output_type="quantiles", quantiles=[0.05, 0.5, 0.95])
        return mean, np.asarray(q[0], float), np.asarray(q[2], float)

    mv, v05, v95 = mq(Xd["validation"])
    mt, t05, t95 = mq(Xd["test"])

    rows = {}
    # 1) native
    rows["TabPFN native"] = cover_width(t05, t95, yt)
    # 2) split-conformal on TabPFN mean
    qhat = conf_quantile(np.abs(yv - mv), ALPHA)
    rows["TabPFN split-conformal"] = cover_width(mt - qhat, mt + qhat, yt)
    # 3) CQR
    E = np.maximum(v05 - yv, yv - v95)
    qc = conf_quantile(E, ALPHA)
    rows["TabPFN CQR"] = cover_width(t05 - qc, t95 + qc, yt)
    # 4) RF split-conformal baseline
    Xm = {k: feat_morgan(sp[k].smiles.tolist(), "morgan") for k in sp}
    rf = RandomForestRegressor(n_estimators=500, max_features=0.3, random_state=SEED, n_jobs=-1).fit(Xm["train"], ytr)
    rf_v = rf.predict(Xm["validation"]); rf_t = rf.predict(Xm["test"])
    qrf = conf_quantile(np.abs(yv - rf_v), ALPHA)
    rows["RF split-conformal"] = cover_width(rf_t - qrf, rf_t + qrf, yt)

    summ = pd.DataFrame([{"method": k, "nominal": 1 - ALPHA, "coverage": round(c, 3),
                          "mean_width": round(w, 3)} for k, (c, w) in rows.items()])
    summ.to_csv(OUT / "conformal_summary.csv", index=False)
    print(summ.to_string(index=False))

    # per-test for the uncertainty-vs-error analysis (local figures)
    per = pd.DataFrame({"smiles": sp["test"].smiles.values, "y_true": yt, "tab_mean": mt,
                        "tab_q05": t05, "tab_q95": t95, "tab_width": t95 - t05,
                        "cqr_lo": t05 - qc, "cqr_hi": t95 + qc,
                        "abs_err": np.abs(mt - yt)})
    per.to_csv(OUT / "uncertainty_test.csv", index=False)
    from scipy.stats import spearmanr
    print(f"\nSpearman(TabPFN width, |error|) = {spearmanr(per.tab_width, per.abs_err)[0]:.3f}")
    print(f"wrote {OUT/'conformal_summary.csv'} + {OUT/'uncertainty_test.csv'}")


if __name__ == "__main__":
    main()
