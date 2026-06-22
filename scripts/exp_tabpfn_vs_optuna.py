"""Experiments E1 + E2 (server-side, data only -- figures are drawn locally with scienceplots).

E1  TabPFN vs Optuna budget curve: for RF/GBM/SVR run ONE 80-trial TPE study, then read off the
    best-of-first-B-trials at budgets B in {1,3,5,10,20,40,80}, refit, score test. TabPFN (0 tuning)
    is the reference. Answers "how many Optuna trials/seconds to match a single TabPFN forward pass".
E2  Data-efficiency learning curve: train sizes {50..1054}; fit TabPFN(desc) / RF / GBM and score
    test. Shows where TabPFN's small-sample edge is largest.

Outputs (server -> pulled to local):
  results/exp/budget_sweep.csv      model,budget,cv_mae,test_mae,test_r,wall_s
  results/exp/learning_curve.csv    model,train_size,test_mae,test_r

Run on server (GPU for TabPFN):
  CUDA_VISIBLE_DEVICES=6 python scripts/exp_tabpfn_vs_optuna.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_tabpfn import featurize as _feat_tab  # ~200 RDKit descList for "desc"  # noqa: E402
from scripts.tune_classic_ml import build, make_pipe, suggest  # noqa: E402
from scripts.tune_classic_ml import featurize as _feat_clf  # Morgan-2048 for "morgan"  # noqa: E402
from src.utils.metrics import regression_metrics  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402


def featurize(smiles, rep):
    """Dispatch so classic models keep Morgan-2048 and TabPFN keeps the ~200-D descriptor block
    (identical to the run_tabpfn / master-table featurization, so numbers stay comparable)."""
    if rep == "morgan":
        return _feat_clf(smiles, "morgan")
    if rep == "desc":
        return _feat_tab(smiles, "desc")
    raise ValueError(rep)

SEED = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
OUT = Path("results/exp"); OUT.mkdir(parents=True, exist_ok=True)
BUDGETS = [1, 3, 5, 10, 20, 40]  # 40 trials is well past where the Optuna curve flattens
SIZES = [50, 100, 200, 400, 700, 1054]


def _shim():
    import sklearn.utils.validation as skv
    if not hasattr(skv, "_is_pandas_df"):
        skv._is_pandas_df = lambda x: isinstance(x, pd.DataFrame)


def load():
    df = pd.read_csv("data/processed/bace_clean.csv")
    sp = {k: df[df.set == k] for k in ["train", "test"]}
    return sp


def e1_budget_sweep(sp, tab_test):
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    Xtr = {r: featurize(sp["train"].smiles.tolist(), r) for r in ["morgan"]}
    Xte = featurize(sp["test"].smiles.tolist(), "morgan")
    ytr = sp["train"].label.to_numpy(float); yte = sp["test"].label.to_numpy(float)
    rows = []
    for name in ["rf", "gbm", "svr"]:
        times = []
        t0 = time.perf_counter()

        def obj(trial):
            r = mean_absolute_error(ytr, cross_val_predict(make_pipe(name, suggest(trial, name)),
                                                           Xtr["morgan"], ytr, cv=cv, n_jobs=1))
            times.append(time.perf_counter() - t0)
            return r
        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
        study.optimize(obj, n_trials=max(BUDGETS))
        trials = study.trials
        for B in BUDGETS:
            sub = trials[:B]
            best = min(sub, key=lambda t: t.value)
            pipe = make_pipe(name, best.params).fit(Xtr["morgan"], ytr)
            mt = regression_metrics(yte, pipe.predict(Xte))
            rows.append({"model": name.upper(), "budget": B, "cv_mae": round(best.value, 4),
                         "test_mae": round(mt["MAE"], 4), "test_r": round(mt["PearsonR"], 4),
                         "wall_s": round(times[B - 1], 2)})
            print(f"  E1 {name:4s} B={B:3d}  cvMAE={best.value:.3f} testMAE={mt['MAE']:.3f} "
                  f"R={mt['PearsonR']:.3f} wall={times[B-1]:.1f}s")
    # TabPFN reference row (budget 0, ~its single-pass fit time)
    rows.append({"model": "TabPFN", "budget": 0, "cv_mae": None,
                 "test_mae": tab_test["MAE"], "test_r": tab_test["PearsonR"], "wall_s": tab_test["wall_s"]})
    pd.DataFrame(rows).to_csv(OUT / "budget_sweep.csv", index=False)
    print(f"wrote {OUT/'budget_sweep.csv'}")


def e2_learning_curve(sp, TabReg, device):
    Xm_tr = featurize(sp["train"].smiles.tolist(), "morgan")
    Xm_te = featurize(sp["test"].smiles.tolist(), "morgan")
    Xd_tr = featurize(sp["train"].smiles.tolist(), "desc")
    Xd_te = featurize(sp["test"].smiles.tolist(), "desc")
    ytr = sp["train"].label.to_numpy(float); yte = sp["test"].label.to_numpy(float)
    rng = np.random.default_rng(SEED)
    rows = []
    n = len(ytr)
    for sz in SIZES:
        idx = rng.choice(n, size=min(sz, n), replace=False)
        for name, fitfn in [
            ("RF", lambda i: build("rf", {"n_estimators": 500, "max_depth": "none",
                                          "max_features": "0.3", "min_samples_leaf": 1}).fit(Xm_tr[i], ytr[i])),
            ("GBM", lambda i: build("gbm", {"learning_rate": 0.1, "max_iter": 400, "max_depth": 4,
                                            "min_samples_leaf": 20, "l2_regularization": 1.0}).fit(Xm_tr[i], ytr[i])),
        ]:
            mdl = fitfn(idx)
            mt = regression_metrics(yte, mdl.predict(Xm_te))
            rows.append({"model": name, "train_size": sz, "test_mae": round(mt["MAE"], 4),
                         "test_r": round(mt["PearsonR"], 4)})
        # TabPFN
        reg = TabReg(device=device, random_state=SEED, ignore_pretraining_limits=True)
        reg.fit(Xd_tr[idx], ytr[idx])
        mt = regression_metrics(yte, np.asarray(reg.predict(Xd_te), float))
        rows.append({"model": "TabPFN", "train_size": sz, "test_mae": round(mt["MAE"], 4),
                     "test_r": round(mt["PearsonR"], 4)})
        print(f"  E2 size={sz:5d}  " + "  ".join(f"{r['model']}={r['test_mae']:.3f}"
                                                 for r in rows[-3:]))
    pd.DataFrame(rows).to_csv(OUT / "learning_curve.csv", index=False)
    print(f"wrote {OUT/'learning_curve.csv'}")


def main():
    _shim()
    import torch
    from tabpfn import TabPFNRegressor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_all_seeds(SEED)
    sp = load()

    # TabPFN reference for E1 (single forward pass on desc), timed
    Xd_tr = featurize(sp["train"].smiles.tolist(), "desc")
    Xd_te = featurize(sp["test"].smiles.tolist(), "desc")
    ytr = sp["train"].label.to_numpy(float); yte = sp["test"].label.to_numpy(float)
    t0 = time.perf_counter()
    reg = TabPFNRegressor(device=device, random_state=SEED, ignore_pretraining_limits=True)
    reg.fit(Xd_tr, ytr)
    pt = np.asarray(reg.predict(Xd_te), float)
    tab_wall = time.perf_counter() - t0
    tab = regression_metrics(yte, pt); tab["wall_s"] = round(tab_wall, 2)
    print(f"TabPFN(desc) ref: testMAE={tab['MAE']:.3f} R={tab['PearsonR']:.3f} wall={tab_wall:.1f}s")

    e1_budget_sweep(sp, tab)
    e2_learning_curve(sp, TabPFNRegressor, device)


if __name__ == "__main__":
    main()
