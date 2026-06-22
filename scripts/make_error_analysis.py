"""Prediction-error deep dive on the BACE test set, following standard QSAR error-analysis practice:

  A1 residual distribution + systematic bias (mean err, std, skew, kurtosis, %|err|>1 log)
  A2 regression-to-the-mean: signed error vs true pIC50 (under-predict potent, over-predict weak)
  A3 large-error molecules characterized vs easy ones (AD/Tanimoto, activity extremeness, cliff,
     size, undefined stereo, label conflict)  -- "what distinguishes mispredictions"
  A4 consensus-hard molecules (high mean |err| across all models) -> category decomposition + a
     RDKit structure grid of the worst offenders
  A5 two tuning regimes: default vs Optuna-tuned (classic 4) and zero-tuning TabPFN vs best Optuna
     -- does tuning fix specific molecules or just shrink overall? which categories flip?

Refs: Maggiora JCIM 2006 (outliers/cliffs); Cortes-Ciriano 2020 (kurtosis/cliffs limit QSAR);
Sahigara/Tropsha AD (JCIM 2009). Everything runs on saved predictions + RDKit on ~1.5k mols (light).

Run: /opt/anaconda3/bin/python3 scripts/make_error_analysis.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401
from scipy.stats import kurtosis, skew  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.metrics import regression_metrics  # noqa: E402

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": True})
warnings.filterwarnings("ignore")
FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
FINAL = Path("results/final"); FINAL.mkdir(parents=True, exist_ok=True)
TUN = Path("results/tuning"); BC = Path("results/bace_clean_final")
SEEDS = [42, 1337, 2024]
MODELS = [("Ridge", "classic", [TUN / "ridge_morgan_test.csv"]),
          ("SVR", "classic", [TUN / "svr_morgan_test.csv"]),
          ("GBM", "classic", [TUN / "gbm_morgan_test.csv"]),
          ("RF", "classic", [TUN / "rf_morgan_test.csv"]),
          ("TabPFN", "tabpfn", [BC / "tabpfn_desc_test.csv"]),
          ("MolFormer", "deep", [BC / f"molformer_seed{s}_test.csv" for s in SEEDS]),
          ("ChemFM", "deep", [BC / f"chemfm_seed{s}_test.csv" for s in SEEDS]),
          ("Chemprop", "deep", [BC / f"chemprop_seed{s}_test.csv" for s in SEEDS])]


def _pcol(df):
    return df["y_pred"] if "y_pred" in df.columns else df["y_score"]


def load_wide():
    """Merge each model's test predictions on smiles. Deep models = mean over the 3 seed files.
    Every file is sorted by smiles first so seed files align row-for-row before averaging."""
    base = None
    for label, _, files in MODELS:
        files = [f for f in files if f.exists()]
        if not files:
            continue
        mats = [pd.read_csv(f).sort_values("smiles").reset_index(drop=True) for f in files]
        pred = np.mean([_pcol(m).to_numpy(float) for m in mats], axis=0)
        acc = pd.DataFrame({"smiles": mats[0].smiles, "y_true": mats[0].y_true.to_numpy(float),
                            label: pred})
        base = acc if base is None else base.merge(acc[["smiles", label]], on="smiles")
    return base


def mol_features(smiles, y_true):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    clean = pd.read_csv("data/processed/bace_clean.csv")
    tr = clean[clean.set == "train"].smiles.tolist()
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fp_tr = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in tr]
    # label-conflict skeletons (InChIKey14 with >1 distinct label across the whole clean set)
    ik = {}
    for s, lab in zip(clean.smiles, clean.label):
        m = Chem.MolFromSmiles(s)
        if m:
            ik.setdefault(Chem.MolToInchiKey(m)[:14], set()).add(round(float(lab), 2))
    conf_ik = {k for k, v in ik.items() if len(v) > 1}
    # activity-cliff membership
    cl = pd.read_csv("results/eda/activity_cliffs_top.csv")
    cliff = set(cl.smiles_A) | set(cl.smiles_B)
    ymean = float(clean[clean.set == "train"].label.mean())
    rows = []
    for s, y in zip(smiles, y_true):
        m = Chem.MolFromSmiles(s)
        fp = gen.GetFingerprint(m)
        rows.append({
            "tanimoto_nn": max(DataStructs.BulkTanimotoSimilarity(fp, fp_tr)),
            "abs_dev_mean": abs(y - ymean),
            "MW": Descriptors.MolWt(m),
            "heavy": m.GetNumHeavyAtoms(),
            "undef_stereo": rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(m),
            "on_cliff": s in cliff,
            "label_conflict": (Chem.MolToInchiKey(m)[:14] in conf_ik),
            "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=m),
        })
    return pd.DataFrame(rows)


def a1_residual_distribution(wide, labels):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = []
    xs = np.linspace(-3, 3, 200)
    for lab in labels:
        e = (wide[lab] - wide.y_true).to_numpy(float)
        rows.append({"model": lab, "bias": e.mean(), "std": e.std(), "skew": skew(e),
                     "kurtosis": kurtosis(e), "pct_gt1": 100 * np.mean(np.abs(e) > 1.0)})
        from scipy.stats import gaussian_kde
        a.plot(xs, gaussian_kde(e)(xs), lw=1.3, label=lab)
    a.axvline(0, ls="--", color="0.5", lw=0.8)
    a.set_xlabel("signed error  (pred - true), log units"); a.set_ylabel("density")
    a.set_title("(a) residual distributions"); a.legend(fontsize=7, ncol=2)
    S = pd.DataFrame(rows)
    order = S.sort_values("kurtosis")
    b.barh(range(len(order)), order["kurtosis"], color="#8172B3", edgecolor="k", linewidth=0.4)
    b.set_yticks(range(len(order))); b.set_yticklabels(order["model"], fontsize=8)
    b.set_xlabel("excess kurtosis of residuals")
    b.set_title("(b) heavy tails (high kurtosis = cliff-driven outliers)", fontsize=9.5)
    fig.suptitle("A1  Residual distribution + systematic bias (BACE test, n=303)", y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "fig_err_residuals_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    S.round(3).to_csv(FINAL / "residual_stats.csv", index=False)
    print("wrote fig_err_residuals_j.png + residual_stats.csv")
    return S


def a2_regression_to_mean(wide, labels):
    show = [m for m in ["TabPFN", "RF", "MolFormer"] if m in labels]
    fig, axs = plt.subplots(1, len(show), figsize=(4.0 * len(show), 3.8), sharey=True)
    if len(show) == 1:
        axs = [axs]
    yt = wide.y_true.to_numpy(float)
    for ax, lab in zip(axs, show):
        e = (wide[lab] - wide.y_true).to_numpy(float)
        ax.scatter(yt, e, s=12, alpha=0.5, color="#4C72B0", edgecolor="none")
        sl, ic = np.polyfit(yt, e, 1)
        xx = np.array([yt.min(), yt.max()])
        ax.plot(xx, sl * xx + ic, color="#C44E52", lw=1.6, label=f"slope={sl:.2f}")
        ax.axhline(0, ls="--", color="0.5", lw=0.8)
        ax.set_xlabel(r"true pIC$_{50}$"); ax.set_title(lab, fontsize=10); ax.legend(fontsize=8)
    axs[0].set_ylabel("signed error (pred - true)")
    fig.suptitle("A2  Regression to the mean: negative slope = under-predict potent, "
                 "over-predict weak", y=1.03, fontsize=11.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_err_regression_to_mean_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote fig_err_regression_to_mean_j.png")


def a3_hard_vs_easy(wide, labels, feat):
    cons = np.mean([np.abs(wide[m] - wide.y_true) for m in labels], axis=0)
    hard = cons >= np.quantile(cons, 0.9)  # top decile by consensus |error|
    panels = [("tanimoto_nn", "nearest-train\nTanimoto (AD)"), ("abs_dev_mean", "|pIC50 - train mean|"),
              ("MW", "molecular weight"), ("undef_stereo", "# undefined\nstereocenters")]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, (col, ttl) in zip(axs, panels):
        data = [feat[col][~hard].to_numpy(float), feat[col][hard].to_numpy(float)]
        bp = ax.boxplot(data, labels=["easy", "hard"], patch_artist=True, widths=0.6, showfliers=False)
        for patch, c in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_title(ttl, fontsize=9)
    # categorical: cliff / label-conflict rates among hard vs easy
    fig.suptitle("A3  What distinguishes the top-decile-error molecules (hard) from the rest (easy)",
                 y=1.03, fontsize=11.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_err_hard_vs_easy_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # rates table
    rate = {"on_cliff": (feat.on_cliff[hard].mean(), feat.on_cliff[~hard].mean()),
            "label_conflict": (feat.label_conflict[hard].mean(), feat.label_conflict[~hard].mean())}
    print("wrote fig_err_hard_vs_easy_j.png")
    print(f"  hard on_cliff={rate['on_cliff'][0]:.0%} vs easy {rate['on_cliff'][1]:.0%}; "
          f"hard label_conflict={rate['label_conflict'][0]:.0%} vs easy {rate['label_conflict'][1]:.0%}")
    return cons, hard


def a4_consensus_hard(wide, labels, feat, cons):
    df = wide[["smiles", "y_true"]].copy()
    df["mean_abs_err"] = cons
    for c in ["tanimoto_nn", "abs_dev_mean", "on_cliff", "label_conflict", "undef_stereo"]:
        df[c] = feat[c].values
    top = df.sort_values("mean_abs_err", ascending=False).head(15).reset_index(drop=True)
    top.round(3).to_csv(FINAL / "consensus_hard_molecules.csv", index=False)
    # category decomposition of the top-30 hardest
    h30 = df.sort_values("mean_abs_err", ascending=False).head(30)
    cats = {"on cliff": h30.on_cliff.mean(), "label conflict": h30.label_conflict.mean(),
            "low AD (Tan<0.5)": (h30.tanimoto_nn < 0.5).mean(),
            "extreme activity\n(|dev|>1.5)": (h30.abs_dev_mean > 1.5).mean(),
            "undefined stereo": (h30.undef_stereo > 0).mean()}
    fig, ax = plt.subplots(figsize=(6.4, 4))
    k = list(cats); v = [cats[x] * 100 for x in k]
    ax.barh(range(len(k)), v, color="#C44E52", edgecolor="k", linewidth=0.4)
    ax.set_yticks(range(len(k))); ax.set_yticklabels(k, fontsize=9)
    ax.set_xlabel("% of the 30 hardest molecules"); ax.set_xlim(0, 100)
    for i, val in enumerate(v):
        ax.annotate(f"{val:.0f}%", (val, i), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8)
    ax.set_title("A4  Why are the consensus-hard molecules hard?\n(categories of the 30 worst, all models)",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_err_consensus_categories_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote consensus_hard_molecules.csv + fig_err_consensus_categories_j.png")
    # RDKit structure grid of top-12
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw
    RDLogger.DisableLog("rdApp.*")
    t12 = df.sort_values("mean_abs_err", ascending=False).head(12)
    mols = [Chem.MolFromSmiles(s) for s in t12.smiles]
    legends = [f"pIC50={r.y_true:.1f} err={r.mean_abs_err:.2f}"
               + (" CLIFF" if r.on_cliff else "") for _, r in t12.iterrows()]
    img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(280, 200), legends=legends)
    img.save(FIG / "fig_err_hard_structures.png")
    print("wrote fig_err_hard_structures.png")


def a5_two_tuning(wide, feat):
    """default vs Optuna-tuned (classic) and TabPFN vs best-Optuna, per-molecule."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from scripts.tune_classic_ml import featurize
    clean = pd.read_csv("data/processed/bace_clean.csv")
    tr = clean[clean.set == "train"]; te = clean[clean.set == "test"]
    Xtr = featurize(tr.smiles.tolist(), "morgan"); Xte = featurize(te.smiles.tolist(), "morgan")
    ytr = tr.label.to_numpy(float)
    order = te.smiles.tolist()
    defaults = {"Ridge": Pipeline([("s", StandardScaler()), ("m", Ridge())]),
                "SVR": Pipeline([("s", StandardScaler()), ("m", SVR())]),
                "GBM": HistGradientBoostingRegressor(random_state=42),
                "RF": RandomForestRegressor(random_state=42, n_jobs=4)}
    def_err = {}
    for name, mdl in defaults.items():
        p = mdl.fit(Xtr, ytr).predict(Xte)
        def_err[name] = pd.Series(np.abs(p - te.label.to_numpy(float)), index=order)
    w = wide.set_index("smiles")
    yt = w.y_true
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    # (a) default vs tuned abs-error per molecule, RF, colored by on-cliff
    on = feat.set_index(wide.smiles).on_cliff
    tuned_rf = (w["RF"] - yt).abs()
    de = def_err["RF"].reindex(w.index)
    col = np.where(on.reindex(w.index).fillna(False), "#C44E52", "#4C72B0")
    a.scatter(de, tuned_rf, s=16, c=col, alpha=0.6, edgecolor="none")
    lim = [0, float(max(de.max(), tuned_rf.max())) + 0.2]
    a.plot(lim, lim, ls="--", color="0.5", lw=1)
    a.set_xlabel("default RF |error|"); a.set_ylabel("Optuna-tuned RF |error|")
    a.set_title("(a) does tuning fix specific molecules?\n(red = activity cliff)", fontsize=9.5)
    a.set_xlim(lim); a.set_ylim(lim)
    # (b) TabPFN vs best-Optuna (RF) per-molecule: who wins, by category
    tab = (w["TabPFN"] - yt).abs(); rf = (w["RF"] - yt).abs()
    diff = (rf - tab)  # >0 => TabPFN better
    cats = {"on cliff": on.reindex(w.index).fillna(False).to_numpy(),
            "low AD": (feat.set_index(wide.smiles).tanimoto_nn.reindex(w.index) < 0.5).to_numpy(),
            "all": np.ones(len(w), bool)}
    names = list(cats); win = [100 * np.mean(diff.to_numpy()[cats[c]] > 0) for c in names]
    b.bar(range(len(names)), win, color=["#C44E52", "#DD8452", "#55A868"], edgecolor="k", linewidth=0.4)
    b.axhline(50, ls="--", color="0.5", lw=0.9)
    b.set_xticks(range(len(names))); b.set_xticklabels(names, fontsize=9)
    b.set_ylabel("% molecules where TabPFN beats tuned RF"); b.set_ylim(0, 100)
    for i, vv in enumerate(win):
        b.annotate(f"{vv:.0f}%", (i, vv), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    b.set_title("(b) TabPFN vs Optuna-tuned RF, by category", fontsize=9.5)
    fig.suptitle("A5  Two tuning regimes: default vs Optuna (a), zero-tuning TabPFN vs Optuna (b)",
                 y=1.02, fontsize=11.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_err_two_tuning_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # Jaccard of top-decile error sets default vs tuned (RF)
    def topset(s):
        return set(s.sort_values(ascending=False).head(int(0.1 * len(s))).index)
    j = len(topset(de) & topset(tuned_rf)) / len(topset(de) | topset(tuned_rf))
    print(f"wrote fig_err_two_tuning_j.png  (RF default-vs-tuned top-decile Jaccard={j:.2f})")


def main():
    wide = load_wide()
    labels = [m for m, _, _ in MODELS if m in wide.columns]
    print(f"loaded {len(wide)} test molecules, models={labels}")
    feat = mol_features(wide.smiles.tolist(), wide.y_true.to_numpy(float))
    a1_residual_distribution(wide, labels)
    a2_regression_to_mean(wide, labels)
    cons, hard = a3_hard_vs_easy(wide, labels, feat)
    a4_consensus_hard(wide, labels, feat, cons)
    a5_two_tuning(wide, feat)


if __name__ == "__main__":
    main()
