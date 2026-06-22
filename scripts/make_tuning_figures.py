"""Tuning figures F9-F12 for BACE_项目设计方案.md §5.4 (all scienceplots, English-internal text).

  F9  validation curve   -- sweep one hyperparameter, train vs 5-fold CV, show the over-fit sweet spot
  F10 learning curve     -- training-set size vs CV MAE, is BACE data-bottlenecked or model-bottlenecked?
  F11 hyperparam landscape-- Optuna trials as a CV-MAE heat-scatter (SVR C-gamma, GBM lr-depth)
  F12 default vs tuned   -- honest delta from tuning (often small) on the test set

Reads results/tuning/{summary.json,*_trials.csv}; refits a few light models for F9/F10.
Run: /opt/anaconda3/bin/python3 scripts/make_tuning_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import KFold, learning_curve, validation_curve  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.tune_classic_ml import featurize, make_pipe, regression_metrics  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402

SEED = 42
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": True})
FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
OUT = Path("results/tuning")
CMODELS = {"ridge": "Ridge", "svr": "SVR", "gbm": "GBM", "rf": "RF"}
CV = KFold(n_splits=5, shuffle=True, random_state=SEED)


def load_data():
    df = pd.read_csv("data/processed/bace_clean.csv")
    sp = {k: df[df.set == k] for k in ["train", "validation", "test"]}
    X = {k: featurize(v.smiles.tolist(), "morgan") for k, v in sp.items()}
    return {"Xtr": X["train"], "ytr": sp["train"].label.to_numpy(float),
            "Xte": X["test"], "yte": sp["test"].label.to_numpy(float)}


def fig9_validation_curve(d):
    """RF n_estimators and Ridge alpha: train vs CV MAE (negative MAE -> flip sign)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.7))
    # Ridge alpha (the textbook bias-variance knob)
    alphas = np.logspace(-2, 4, 12)
    tr, va = validation_curve(Pipeline([("s", StandardScaler()), ("m", Ridge())]),
                              d["Xtr"], d["ytr"], param_name="m__alpha", param_range=alphas,
                              cv=CV, scoring="neg_mean_absolute_error", n_jobs=-1)
    a.semilogx(alphas, -tr.mean(1), "o-", color="#4C72B0", label="train", ms=4)
    a.fill_between(alphas, -tr.mean(1) - tr.std(1), -tr.mean(1) + tr.std(1), color="#4C72B0", alpha=0.15)
    a.semilogx(alphas, -va.mean(1), "s-", color="#C44E52", label="5-fold CV", ms=4)
    a.fill_between(alphas, -va.mean(1) - va.std(1), -va.mean(1) + va.std(1), color="#C44E52", alpha=0.15)
    a.axvline(alphas[np.argmin(-va.mean(1))], ls="--", color="0.4", lw=1)
    a.set_xlabel(r"Ridge $\alpha$ (regularization)"); a.set_ylabel("MAE"); a.set_title("(a) Ridge")
    a.legend(fontsize=8)
    # RF n_estimators
    ne = [50, 100, 200, 300, 500, 700]
    tr, va = validation_curve(RandomForestRegressor(random_state=SEED, n_jobs=-1),
                              d["Xtr"], d["ytr"], param_name="n_estimators", param_range=ne,
                              cv=CV, scoring="neg_mean_absolute_error", n_jobs=1)
    b.plot(ne, -tr.mean(1), "o-", color="#4C72B0", label="train", ms=4)
    b.plot(ne, -va.mean(1), "s-", color="#C44E52", label="5-fold CV", ms=4)
    b.fill_between(ne, -va.mean(1) - va.std(1), -va.mean(1) + va.std(1), color="#C44E52", alpha=0.15)
    b.set_xlabel("RF n_estimators"); b.set_ylabel("MAE"); b.set_title("(b) Random Forest")
    b.legend(fontsize=8)
    fig.suptitle("F9  Validation curves: gap between train and CV reveals over-fitting", y=1.01, fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "fig_tune_validation_curve_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote fig_tune_validation_curve_j.png")


def fig10_learning_curve(d, best):
    """RF and GBM at their tuned settings: CV MAE vs training-set size."""
    fig, ax = plt.subplots(figsize=(5.4, 4))
    sizes = np.linspace(0.15, 1.0, 7)
    for name, col, mk in [("rf", "#55A868", "o"), ("gbm", "#8172B3", "s")]:
        est = make_pipe(name, best[name])
        ts, tr, va = learning_curve(est, d["Xtr"], d["ytr"], train_sizes=sizes, cv=CV,
                                    scoring="neg_mean_absolute_error", n_jobs=-1, random_state=SEED)
        ax.plot(ts, -va.mean(1), mk + "-", color=col, label=f"{CMODELS[name]} (CV)", ms=5)
        ax.fill_between(ts, -va.mean(1) - va.std(1), -va.mean(1) + va.std(1), color=col, alpha=0.15)
    ax.set_xlabel("training molecules"); ax.set_ylabel("5-fold CV MAE")
    ax.set_title("F10  Learning curves: still-declining slope = a data bottleneck", fontsize=10.5)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig_tune_learning_curve_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote fig_tune_learning_curve_j.png")


def fig11_landscape():
    """Optuna trials as CV-MAE heat-scatter: SVR (C, gamma) and GBM (lr, max_depth)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.4, 3.9))
    sv = pd.read_csv(OUT / "svr_morgan_trials.csv")
    sc = a.scatter(sv.params_C, sv.params_gamma, c=sv.value, cmap="viridis_r", s=44,
                   edgecolor="k", linewidth=0.3)
    bi = sv.value.idxmin()
    a.scatter(sv.params_C[bi], sv.params_gamma[bi], marker="*", s=320, color="#C44E52",
              edgecolor="k", linewidth=0.6, label="best", zorder=5)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("SVR  C"); a.set_ylabel(r"SVR  $\gamma$")
    a.set_title("(a) SVR landscape"); a.legend(fontsize=8, loc="lower left")
    fig.colorbar(sc, ax=a, label="CV MAE")
    gb = pd.read_csv(OUT / "gbm_morgan_trials.csv")
    sc2 = b.scatter(gb.params_learning_rate, gb.params_max_depth, c=gb.value, cmap="viridis_r",
                    s=44, edgecolor="k", linewidth=0.3)
    bi = gb.value.idxmin()
    b.scatter(gb.params_learning_rate[bi], gb.params_max_depth[bi], marker="*", s=320,
              color="#C44E52", edgecolor="k", linewidth=0.6, label="best", zorder=5)
    b.set_xscale("log"); b.set_xlabel("GBM learning_rate"); b.set_ylabel("GBM max_depth")
    b.set_title("(b) GBM landscape"); b.legend(fontsize=8, loc="upper right")
    fig.colorbar(sc2, ax=b, label="CV MAE")
    fig.suptitle("F11  Bayesian search landscape (each dot = one Optuna trial)", y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "fig_tune_landscape_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote fig_tune_landscape_j.png")


def fig12_default_vs_tuned(d, summ):
    """Honest delta: default sklearn settings vs the Optuna-tuned model on the test set."""
    defaults = {"ridge": Ridge(), "svr": SVR(kernel="rbf"),
                "gbm": HistGradientBoostingRegressor(random_state=SEED),
                "rf": RandomForestRegressor(random_state=SEED, n_jobs=-1)}
    rows = []
    for name in CMODELS:
        pipe = (Pipeline([("s", StandardScaler()), ("m", defaults[name])])
                if name in ("ridge", "svr") else defaults[name]).fit(d["Xtr"], d["ytr"])
        md = regression_metrics(d["yte"], pipe.predict(d["Xte"]))
        mt = next(r["test"] for r in summ if r["model"] == name)
        rows.append((CMODELS[name], md["MAE"], mt["MAE"], md["PearsonR"], mt["PearsonR"]))
    R = pd.DataFrame(rows, columns=["model", "def_mae", "tun_mae", "def_r", "tun_r"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    x = np.arange(len(R)); w = 0.38
    a.bar(x - w / 2, R.def_mae, w, label="default", color="#bbbbbb", edgecolor="k", linewidth=0.4)
    a.bar(x + w / 2, R.tun_mae, w, label="tuned", color="#4C72B0", edgecolor="k", linewidth=0.4)
    for i in x:
        dlt = R.def_mae[i] - R.tun_mae[i]
        a.annotate(f"{dlt:+.3f}", (i, max(R.def_mae[i], R.tun_mae[i]) + 0.005), ha="center", fontsize=7.5,
                   color="#1b5e20" if dlt > 0 else "#7a1f1f")
    a.set_xticks(x); a.set_xticklabels(R.model); a.set_ylabel("test MAE (lower better)")
    a.set_title("(a) MAE"); a.legend(fontsize=8)
    a.set_ylim(0, max(R.def_mae.max(), R.tun_mae.max()) * 1.18)
    b.bar(x - w / 2, R.def_r, w, label="default", color="#bbbbbb", edgecolor="k", linewidth=0.4)
    b.bar(x + w / 2, R.tun_r, w, label="tuned", color="#55A868", edgecolor="k", linewidth=0.4)
    for i in x:
        dlt = R.tun_r[i] - R.def_r[i]
        b.annotate(f"{dlt:+.3f}", (i, max(R.def_r[i], R.tun_r[i]) + 0.006), ha="center", fontsize=7.5,
                   color="#1b5e20" if dlt > 0 else "#7a1f1f")
    b.set_xticks(x); b.set_xticklabels(R.model); b.set_ylabel("test Pearson R (higher better)")
    b.set_title("(b) Pearson R"); b.legend(fontsize=8, loc="lower right"); b.set_ylim(0.6, 0.9)
    fig.suptitle("F12  Default vs tuned (test set): tuning helps, but honestly by how much", y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "fig_tune_default_vs_tuned_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote fig_tune_default_vs_tuned_j.png")
    print(R.round(3).to_string(index=False))


def main():
    set_all_seeds(SEED)
    d = load_data()
    summ = json.loads((OUT / "summary.json").read_text())
    summ = [r for r in summ if r["rep"] == "morgan"]
    best = {r["model"]: r["best_params"] for r in summ}
    fig9_validation_curve(d)
    fig10_learning_curve(d, best)
    fig11_landscape()
    fig12_default_vs_tuned(d, summ)


if __name__ == "__main__":
    main()
