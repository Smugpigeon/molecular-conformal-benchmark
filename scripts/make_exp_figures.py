"""Figures + analysis for the extra experiments E1-E5 (local, scienceplots).

E1  TabPFN vs Optuna budget curve        <- results/exp/budget_sweep.csv
E2  data-efficiency learning curve       <- results/exp/learning_curve.csv
E3  applicability-domain stratified MAE  <- saved test predictions + nearest-train Tanimoto
E4  cross-model residual correlation + simple ensemble
E5  activity-cliff stratified MAE        <- results/eda/activity_cliffs_top.csv

Everything reads tiny CSVs / runs RDKit on 1.5k molecules -- featherweight, no GPU.
Run: /opt/anaconda3/bin/python3 scripts/make_exp_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.metrics import regression_metrics  # noqa: E402

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": True})
FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
EXP = Path("results/exp"); TUN = Path("results/tuning"); BC = Path("results/bace_clean_final")
SEEDS = [42, 1337, 2024]

# label -> (family, list of test-prediction CSVs to average)
MODELS = [
    ("Ridge", "classic", [TUN / "ridge_morgan_test.csv"]),
    ("SVR", "classic", [TUN / "svr_morgan_test.csv"]),
    ("GBM", "classic", [TUN / "gbm_morgan_test.csv"]),
    ("RF", "classic", [TUN / "rf_morgan_test.csv"]),
    ("TabPFN", "tabpfn", [BC / "tabpfn_desc_test.csv"]),
    ("MolFormer", "deep", [BC / f"molformer_seed{s}_test.csv" for s in SEEDS]),
    ("ChemFM", "deep", [BC / f"chemfm_seed{s}_test.csv" for s in SEEDS]),
    ("Chemprop", "deep", [BC / f"chemprop_seed{s}_test.csv" for s in SEEDS]),
]
CMAP = {"classic": "#4C72B0", "tabpfn": "#C44E52", "deep": "#55A868"}


def _pred_col(df):
    return df["y_pred"] if "y_pred" in df.columns else df["y_score"]


def load_wide():
    """Merge every model's test predictions on smiles -> wide frame (deep = mean over seeds)."""
    base = None
    fam = {}
    for label, family, files in MODELS:
        files = [f for f in files if f.exists()]
        if not files:
            continue
        acc = None
        for f in files:
            d = pd.read_csv(f)[["smiles", "y_true"]].assign(pred=_pred_col(pd.read_csv(f)))
            acc = d if acc is None else acc.merge(d[["smiles", "pred"]], on="smiles",
                                                  suffixes=("", "_x")).assign(
                pred=lambda x: x[["pred", "pred_x"]].mean(axis=1)).drop(columns="pred_x")
        acc = acc.rename(columns={"pred": label})
        fam[label] = family
        base = acc[["smiles", "y_true", label]] if base is None else base.merge(
            acc[["smiles", label]], on="smiles")
    return base, fam


def fig_e1():
    p = EXP / "budget_sweep.csv"
    if not p.exists():
        print("skip E1 (no budget_sweep.csv)"); return
    d = pd.read_csv(p)
    tab = d[d.model == "TabPFN"].iloc[0]
    clf = d[d.model != "TabPFN"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.1))
    colors = {"RF": "#4C72B0", "GBM": "#DD8452", "SVR": "#8172B3"}
    for m, g in clf.groupby("model"):
        a.plot(g.budget, g.test_mae, "o-", color=colors.get(m, "k"), ms=4, label=m)
    a.axhline(tab.test_mae, ls="--", color="#C44E52", lw=1.6, label="TabPFN (0 tuning)")
    a.set_xscale("log"); a.set_xlabel("Optuna trials (log)"); a.set_ylabel("test MAE")
    a.set_title("(a) Optuna budget vs TabPFN"); a.legend(fontsize=8)
    # Pareto: accuracy vs tuning wall-clock
    for m, g in clf.groupby("model"):
        b.plot(g.wall_s, g.test_mae, "o-", color=colors.get(m, "k"), ms=4, label=m, alpha=0.85)
    b.scatter([tab.wall_s], [tab.test_mae], marker="*", s=320, color="#C44E52",
              edgecolor="k", linewidth=0.6, zorder=5, label="TabPFN")
    b.set_xlabel("tuning wall-clock (s)"); b.set_ylabel("test MAE")
    b.set_title("(b) accuracy vs tuning cost (top-left = best)"); b.legend(fontsize=8)
    fig.suptitle("E1  TabPFN (zero-tuning) vs Optuna-tuned classic ML on BACE", y=1.02, fontsize=12.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_exp_tabpfn_vs_optuna_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote fig_exp_tabpfn_vs_optuna_j.png")


def fig_e2():
    p = EXP / "learning_curve.csv"
    if not p.exists():
        print("skip E2 (no learning_curve.csv)"); return
    d = pd.read_csv(p)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.1))
    colors = {"TabPFN": "#C44E52", "RF": "#4C72B0", "GBM": "#DD8452"}
    for m, g in d.groupby("model"):
        g = g.sort_values("train_size")
        a.plot(g.train_size, g.test_mae, "o-", color=colors.get(m, "k"), ms=4, label=m)
        b.plot(g.train_size, g.test_r, "o-", color=colors.get(m, "k"), ms=4, label=m)
    for ax, yl, ttl in [(a, "test MAE", "(a) MAE vs train size"),
                        (b, "test Pearson R", "(b) Pearson vs train size")]:
        ax.set_xlabel("training molecules"); ax.set_ylabel(yl); ax.set_title(ttl); ax.legend(fontsize=8)
    fig.suptitle("E2  Data-efficiency: TabPFN's edge is largest when data is scarce", y=1.02, fontsize=12.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_exp_learning_curve_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote fig_exp_learning_curve_j.png")


def nearest_train_tanimoto(test_smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.DisableLog("rdApp.*")
    tr = pd.read_csv("data/processed/bace_clean.csv")
    tr = tr[tr.set == "train"].smiles.tolist()
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fp_tr = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr]
    out = []
    for s in test_smiles:
        fp = gen.GetFingerprint(Chem.MolFromSmiles(s))
        out.append(max(DataStructs.BulkTanimotoSimilarity(fp, fp_tr)))
    return np.asarray(out, dtype=float)


def fig_e3(wide, fam):
    sim = nearest_train_tanimoto(wide.smiles.tolist())
    # terciles -> low / med / high similarity to training chemistry
    q1, q2 = np.quantile(sim, [1 / 3, 2 / 3])
    band = np.where(sim < q1, "low", np.where(sim < q2, "med", "high"))
    labels = [m for m, _, _ in MODELS if m in wide.columns]
    bins = ["low", "med", "high"]
    M = np.zeros((len(labels), 3))
    for i, m in enumerate(labels):
        for j, bnd in enumerate(bins):
            sel = band == bnd
            M[i, j] = regression_metrics(wide.y_true[sel].to_numpy(float),
                                         wide[m][sel].to_numpy(float))["MAE"]
    counts = [int((band == b).sum()) for b in bins]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(labels)); w = 0.26
    for j, (bnd, c) in enumerate(zip(bins, ["#55A868", "#DD8452", "#C44E52"])):
        ax.bar(x + (j - 1) * w, M[:, j], w, color=c, edgecolor="k", linewidth=0.3,
               label=f"{bnd} sim (n={counts[j]})")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("test MAE"); ax.legend(fontsize=8, title="nearest-train Tanimoto")
    ax.set_title(f"E3  Applicability domain: MAE by similarity to training set "
                 f"(Tanimoto cuts {q1:.2f}/{q2:.2f})", fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_exp_ad_stratified_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"wrote fig_exp_ad_stratified_j.png (bin n={counts})")


def fig_e4(wide, fam):
    labels = [m for m, _, _ in MODELS if m in wide.columns]
    res = pd.DataFrame({m: wide[m] - wide.y_true for m in labels})
    C = res.corr(method="pearson")
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    im = a.imshow(C.values, cmap="RdBu_r", vmin=0, vmax=1)
    a.set_xticks(range(len(labels))); a.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    a.set_yticks(range(len(labels))); a.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            a.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                   color="white" if C.values[i, j] > 0.6 else "k")
    fig.colorbar(im, ax=a, fraction=0.046, label="residual Pearson")
    a.set_title("(a) cross-model residual correlation", fontsize=10)
    # ensemble vs best single
    yt = wide.y_true.to_numpy(float)
    best = min(labels, key=lambda m: regression_metrics(yt, wide[m].to_numpy(float))["MAE"])
    div = [m for m in ["TabPFN", "Chemprop", "RF"] if m in labels]
    combos = {f"best single\n({best})": wide[best],
              "mean: TabPFN\n+Chemprop+RF": wide[div].mean(axis=1),
              "mean: all models": wide[labels].mean(axis=1)}
    names = list(combos); maes = [regression_metrics(yt, v.to_numpy(float))["MAE"] for v in combos.values()]
    rs = [regression_metrics(yt, v.to_numpy(float))["PearsonR"] for v in combos.values()]
    xb = np.arange(len(names))
    bars = b.bar(xb, maes, color=["#C44E52", "#8172B3", "#4C72B0"], edgecolor="k", linewidth=0.4)
    for i, (mae, r) in enumerate(zip(maes, rs)):
        b.annotate(f"MAE {mae:.3f}\nR {r:.3f}", (i, mae), ha="center", va="bottom",
                   xytext=(0, 2), textcoords="offset points", fontsize=7.5)
    b.set_xticks(xb); b.set_xticklabels(names, fontsize=8); b.set_ylabel("test MAE")
    b.set_ylim(0, max(maes) * 1.25); b.set_title("(b) simple averaging ensemble", fontsize=10)
    fig.suptitle("E4  Do models make complementary errors? (low corr -> ensemble helps)",
                 y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "fig_exp_ensemble_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"wrote fig_exp_ensemble_j.png (best single={best})")


def fig_e5(wide, fam):
    p = Path("results/eda/activity_cliffs_top.csv")
    if not p.exists():
        print("skip E5 (no activity_cliffs_top.csv)"); return
    cl = pd.read_csv(p)
    cliff_smiles = set(cl.smiles_A) | set(cl.smiles_B)
    on = wide.smiles.isin(cliff_smiles).to_numpy()
    labels = [m for m, _, _ in MODELS if m in wide.columns]
    if on.sum() < 3:
        print(f"skip E5 (only {on.sum()} on-cliff test mols)"); return
    yt = wide.y_true.to_numpy(float)
    on_mae = [regression_metrics(yt[on], wide[m][on].to_numpy(float))["MAE"] for m in labels]
    off_mae = [regression_metrics(yt[~on], wide[m][~on].to_numpy(float))["MAE"] for m in labels]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, off_mae, w, color="#4C72B0", edgecolor="k", linewidth=0.3,
           label=f"off-cliff (n={int((~on).sum())})")
    ax.bar(x + w / 2, on_mae, w, color="#C44E52", edgecolor="k", linewidth=0.3,
           label=f"on-cliff (n={int(on.sum())})")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("test MAE"); ax.legend(fontsize=8.5)
    ax.set_title("E5  Activity-cliff stress test: every model errs more on cliff molecules",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_exp_cliff_stratified_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print(f"wrote fig_exp_cliff_stratified_j.png (on-cliff n={int(on.sum())})")


def main():
    fig_e1()
    fig_e2()
    wide, fam = load_wide()
    print(f"merged predictions: {wide.shape[0]} test molecules, models={list(wide.columns[2:])}")
    fig_e3(wide, fam)
    fig_e4(wide, fam)
    fig_e5(wide, fam)


if __name__ == "__main__":
    main()
