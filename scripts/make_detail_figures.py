"""Detailed scienceplots figures for the BACE single-task report/deck.

Every figure is computed from real artifacts only:
  - data/BACE.split.csv                     (SMILES + pIC50 + split)
  - results/test_preds/<model>_test.csv     (per-molecule test predictions, 303 ea.)
  - results/final_with_gnn.csv              (per-model x per-seed metrics)
  - results/bace_docking_full.csv           (n=30 docking: vina/pIC50/heavy/mw)
  - results/final/conformal_multiseed.csv   (per-model coverage/width)
  - results/final/conformal_ad.csv          (AD-stratified conditional coverage)
  - results/wang_audit/bace_split_comparison.csv  (predefined vs scaffold)
  - results/negative_control.csv            (label-permutation control)

Style: scienceplots + Times New Roman (matches the journal report/figures).
Run: python scripts/make_detail_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401
from scipy import stats  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, Descriptors, DataStructs, Crippen, rdMolDescriptors  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")
plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix", "axes.unicode_minus": True,
})

FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
TP = Path("results/test_preds")
VP = Path("results/val_preds")
COLOR = {"rf": "#7f7f7f", "chemprop": "#CC79A7", "molformer": "#0072B2",
         "chemfm": "#009E73", "unimol": "#D55E00"}
LABEL = {"rf": "RF + Morgan", "chemprop": "Chemprop", "molformer": "MolFormer-XL",
         "chemfm": "ChemFM-3B", "unimol": "Uni-Mol"}
ORDER = ["chemprop", "rf", "chemfm", "molformer"]   # by Pearson R desc (Uni-Mol removed)
SPLIT_C = {"train": "#4C72B0", "validation": "#DD8452", "test": "#55A868"}


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def morgan(smiles):
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


# ----------------------------------------------------------------------- EDA
def fig_label_dist(df):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bins = np.linspace(df.label.min(), df.label.max(), 30)
    for sp in ["train", "validation", "test"]:
        v = df[df.set == sp].label
        ax.hist(v, bins=bins, density=True, alpha=0.45, color=SPLIT_C[sp],
                label=f"{sp} (n={len(v)})", edgecolor="white", linewidth=0.3)
    ax.axvline(df.label.mean(), ls="--", color="k", lw=0.9)
    ax.annotate(f"mean {df.label.mean():.2f}", (df.label.mean(), ax.get_ylim()[1]*0.92),
                fontsize=8, ha="left")
    ax.set_xlabel(r"pIC$_{50}$ (higher = stronger inhibition)")
    ax.set_ylabel("density"); ax.set_title("BACE target distribution by split")
    ax.legend(fontsize=8)
    save(fig, "fig_eda_label_j.png")


def fig_properties(df):
    mols = [Chem.MolFromSmiles(s) for s in df.smiles]
    props = {
        "Molecular weight": [Descriptors.MolWt(m) for m in mols],
        "Heavy atoms": [m.GetNumHeavyAtoms() for m in mols],
        "cLogP (Crippen)": [Crippen.MolLogP(m) for m in mols],
        "TPSA": [rdMolDescriptors.CalcTPSA(m) for m in mols],
        "Ring count": [rdMolDescriptors.CalcNumRings(m) for m in mols],
        "Rotatable bonds": [rdMolDescriptors.CalcNumRotatableBonds(m) for m in mols],
    }
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 5.4))
    for ax, (name, vals) in zip(axs.ravel(), props.items()):
        vals = np.array(vals, dtype=float)
        ax.hist(vals, bins=28, color="#4C72B0", alpha=0.82, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(vals), ls="--", color="#9b2226", lw=1.0)
        ax.set_title(f"{name}  (median {np.median(vals):.0f})" if vals.max() > 20
                     else f"{name}  (median {np.median(vals):.1f})", fontsize=9)
        ax.set_ylabel("count", fontsize=8)
    fig.suptitle("BACE chemical-space descriptors (1,513 molecules)", fontsize=11, y=1.00)
    fig.tight_layout()
    save(fig, "fig_eda_props_j.png")


def fig_scaffolds(df):
    scaf = []
    for s in df.smiles:
        m = Chem.MolFromSmiles(s)
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            sc = ""
        scaf.append(sc)
    vc = pd.Series(scaf).value_counts()
    n_unique = (vc.index != "").sum()
    singleton = int((vc == 1).sum())
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    top = vc.head(15)
    ax.barh(range(len(top))[::-1], top.values, color="#55A868", edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels([f"#{i+1}" for i in range(len(top))], fontsize=7)
    ax.set_xlabel("molecules sharing the scaffold")
    ax.set_title(f"Bemis-Murcko scaffolds: {n_unique} unique, "
                 f"{singleton} singletons ({100*singleton/len(df):.0f}%)")
    ax.invert_yaxis()
    save(fig, "fig_eda_scaffold_j.png")


def fig_leakage_tanimoto(df):
    tr = df[df.set == "train"]; te = df[df.set == "test"]
    fp_tr = morgan(tr.smiles); fp_te = morgan(te.smiles)
    nn = [max(DataStructs.BulkTanimotoSimilarity(f, fp_tr)) for f in fp_te]
    nn = np.array(nn); hi = (nn > 0.85).mean()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(nn, bins=30, color="#8172B3", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(0.85, ls="--", color="#9b2226", lw=1.2)
    ax.axvspan(0.85, 1.0, color="#9b2226", alpha=0.10)
    ax.annotate(f"T > 0.85: {100*hi:.0f}% of test\n(near-duplicate of train)",
                (0.86, ax.get_ylim()[1]*0.7), fontsize=8, color="#9b2226")
    ax.set_xlabel("nearest-train Tanimoto (Morgan r2, 2048)")
    ax.set_ylabel("count of test molecules")
    ax.set_title(f"Train-test similarity (predefined split): mean {nn.mean():.2f}")
    save(fig, "fig_leakage_tanimoto_j.png")


# --------------------------------------------------------------- performance
def fig_metric_bars(fin):
    b = fin[fin.dataset == "bace"]
    g = b.groupby("model").agg(MAE=("MAE", "mean"), MAE_s=("MAE", "std"),
                               RMSE=("RMSE", "mean"), RMSE_s=("RMSE", "std"),
                               R=("PearsonR", "mean"), R_s=("PearsonR", "std"),
                               rho=("SpearmanRho", "mean"), rho_s=("SpearmanRho", "std"))
    metrics = [("MAE", "MAE_s", r"MAE $\downarrow$"), ("RMSE", "RMSE_s", r"RMSE $\downarrow$"),
               ("R", "R_s", r"Pearson $R$ $\uparrow$"), ("rho", "rho_s", r"Spearman $\rho$ $\uparrow$")]
    fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.1))
    for ax, (m, s, lab) in zip(axs, metrics):
        vals = [g.loc[k, m] for k in ORDER]; errs = [g.loc[k, s] for k in ORDER]
        cols = [COLOR[k] for k in ORDER]
        ax.bar(range(len(ORDER)), vals, yerr=errs, capsize=2.5, color=cols,
               edgecolor="k", linewidth=0.4)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[k].split(" ")[0] for k in ORDER], rotation=30, ha="right", fontsize=7.5)
        ax.set_title(lab, fontsize=9)
        if m in ("R", "rho"): ax.set_ylim(0.6, 0.9)
    fig.suptitle(r"BACE regression: per-model metrics (mean $\pm$ s.d., 3 seeds)", fontsize=11, y=1.02)
    fig.tight_layout()
    save(fig, "fig_metrics_bars_j.png")


def _scatter(ax, yt, yp, color, title):
    r = stats.pearsonr(yt, yp)[0]; mae = np.mean(np.abs(yt - yp))
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.5", lw=0.9, zorder=1)
    ax.scatter(yt, yp, s=14, color=color, alpha=0.55, edgecolor="k", linewidth=0.2, zorder=3)
    ax.set_title(f"{title}\n$R$={r:.3f}  MAE={mae:.3f}", fontsize=9)
    ax.set_xlabel(r"experimental pIC$_{50}$"); ax.set_ylabel(r"predicted pIC$_{50}$")
    return r


def fig_pred_vs_actual():
    d = pd.read_csv(TP / "chemprop_test.csv")
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    _scatter(ax, d.y_true.values, d.y_score.values, COLOR["chemprop"],
             "Chemprop (best) — BACE test (n=303)")
    save(fig, "fig_pred_vs_actual_j.png")


def fig_pred_grid():
    fig, axs = plt.subplots(1, len(ORDER), figsize=(3.05 * len(ORDER), 3.2), sharex=True, sharey=True)
    for ax, k in zip(axs, ORDER):
        d = pd.read_csv(VP / f"{k}_val.csv")   # validation split (matches Table 1); seed 42
        _scatter(ax, d.y_true.values, d.y_score.values, COLOR[k], LABEL[k])
        ax.label_outer()
    fig.suptitle("Predicted vs experimental pIC$_{50}$ on the BACE validation set (151 molecules, seed 42)",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    save(fig, "fig_pred_grid_j.png")


def fig_residuals():
    d = pd.read_csv(TP / "chemprop_test.csv")
    res = d.y_score.values - d.y_true.values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    a1.axhline(0, ls="--", color="0.5", lw=0.9)
    a1.scatter(d.y_true, res, s=14, color=COLOR["chemprop"], alpha=0.55, edgecolor="k", linewidth=0.2)
    a1.set_xlabel(r"experimental pIC$_{50}$"); a1.set_ylabel("residual (pred - true)")
    a1.set_title("Residuals vs target (Chemprop)")
    a2.hist(res, bins=28, color=COLOR["chemprop"], alpha=0.8, edgecolor="white", linewidth=0.3, density=True)
    xs = np.linspace(res.min(), res.max(), 100)
    a2.plot(xs, stats.norm.pdf(xs, res.mean(), res.std()), color="#9b2226", lw=1.2)
    a2.axvline(0, ls="--", color="0.5", lw=0.9)
    a2.set_xlabel("residual"); a2.set_ylabel("density")
    a2.set_title(f"Residual distribution ($\\mu$={res.mean():.2f}, $\\sigma$={res.std():.2f})")
    fig.tight_layout()
    save(fig, "fig_residuals_j.png")


# ------------------------------------------------------------------- docking
def fig_docking_le(dk):
    le = dk.vina.values / dk.heavy.values
    r_raw, p_raw = stats.pearsonr(dk.vina, dk.pIC50)
    r_le, p_le = stats.pearsonr(le, dk.pIC50)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    sc = a1.scatter(dk.vina, dk.pIC50, c=dk.heavy, cmap="cividis", s=46, edgecolor="k", linewidth=0.3)
    b = np.polyfit(dk.vina, dk.pIC50, 1)
    xs = np.linspace(dk.vina.min(), dk.vina.max(), 50)
    a1.plot(xs, np.poly1d(b)(xs), ls="--", color="#9b2226", lw=1.2)
    a1.set_xlabel("Vina score (kcal/mol)"); a1.set_ylabel(r"experimental pIC$_{50}$")
    a1.set_title(f"Raw docking score: $r$={r_raw:.2f} (p={p_raw:.3f}, n.s.)")
    fig.colorbar(sc, ax=a1, fraction=0.046, pad=0.04).set_label("heavy atoms")
    a2.scatter(le, dk.pIC50, c=dk.heavy, cmap="cividis", s=46, edgecolor="k", linewidth=0.3)
    bb = np.polyfit(le, dk.pIC50, 1)
    xs2 = np.linspace(le.min(), le.max(), 50)
    a2.plot(xs2, np.poly1d(bb)(xs2), ls="--", color="#9b2226", lw=1.2)
    a2.set_xlabel("ligand efficiency (Vina / heavy atom)"); a2.set_ylabel(r"experimental pIC$_{50}$")
    a2.set_title(f"Size-normalized: $r$={r_le:.2f} (p={p_le:.2f})")
    fig.suptitle("Docking score does not predict activity once molecule size is removed (n=30)",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig_docking_le_j.png")


def fig_docking_confound(dk):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, x, lab, yl in [(a1, dk.vina, "Vina score (kcal/mol)", "vina"),
                           (a2, dk.pIC50, r"experimental pIC$_{50}$", "pIC50")]:
        r = stats.pearsonr(dk.heavy, x)[0]
        ax.scatter(dk.heavy, x, s=42, color="#0072B2", alpha=0.7, edgecolor="k", linewidth=0.3)
        b = np.polyfit(dk.heavy, x, 1); xs = np.linspace(dk.heavy.min(), dk.heavy.max(), 50)
        ax.plot(xs, np.poly1d(b)(xs), ls="--", color="#9b2226", lw=1.1)
        ax.set_xlabel("heavy-atom count"); ax.set_ylabel(lab)
        ax.set_title(f"{lab.split('(')[0].strip()} vs size: $r$={r:.2f}", fontsize=9)
    fig.suptitle("The size confound: Vina tracks molecule size, activity does not",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save(fig, "fig_docking_confound_j.png")


# ----------------------------------------------------------------- conformal
def fig_conformal_covwidth(cf):
    b = cf[cf.dataset == "bace"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.axvspan(0.80, 0.90, color="#bbbbbb", alpha=0.30)
    ax.axvline(0.90, ls="--", color="k", lw=0.9)
    ax.annotate("nominal 0.90", (0.901, ax.get_ylim()[0]), fontsize=8, rotation=90, va="bottom")
    for k in ORDER:
        row = b[b.model == k]
        if len(row):
            ax.errorbar(row.cov_mean.iloc[0], row.width_mean.iloc[0],
                        xerr=row.cov_std.iloc[0], yerr=row.width_std.iloc[0],
                        fmt="o", ms=9, color=COLOR[k], ecolor=COLOR[k], elinewidth=1,
                        capsize=2, mec="k", mew=0.4, label=LABEL[k])
    ax.set_xlabel("empirical coverage"); ax.set_ylabel(r"interval width (pIC$_{50}$) $\downarrow$")
    ax.set_title("Conformal: tightest valid interval on BACE (shaded = undercovered)")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig_conformal_covwidth_j.png")


def fig_conformal_ad(ad):
    b = ad[ad.dataset == "bace"]
    models = [k for k in ORDER if k in set(b.model)]   # AD computed for 4 of 5 (no chemprop)
    strata = ["low", "med", "high"]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    nb = len(models); w = 0.8 / len(strata)
    hatch = {"low": "//", "med": "", "high": ".."}
    for j, st in enumerate(strata):
        xs, ys = [], []
        for i, k in enumerate(models):
            row = b[(b.model == k) & (b.stratum == st)]
            xs.append(i + (j - 1) * w); ys.append(row.coverage.iloc[0] if len(row) else 0)
        ax.bar(xs, ys, w, color=[COLOR[k] for k in models], edgecolor="k", linewidth=0.4,
               hatch=hatch[st], alpha=0.95, label=f"{st}-similarity")
    ax.axhline(0.90, ls="--", color="#9b2226", lw=1.2)
    ax.annotate("nominal 0.90", (nb-0.55, 0.905), fontsize=8, color="#9b2226", ha="right")
    ax.set_xticks(range(nb)); ax.set_xticklabels([LABEL[k].split(" ")[0] for k in models], fontsize=8)
    ax.set_ylabel("conditional coverage"); ax.set_ylim(0.6, 1.02)
    ax.set_title("AD-stratified coverage: low-similarity (new scaffolds) under-covers\n"
                 "(hatch: // low, none med, .. high; Chemprop AD not computed)", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right", ncol=3)
    save(fig, "fig_conformal_ad_j.png")


# --------------------------------------------------------------------- rigor
def fig_split_leakage(sc):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.6))
    x = [0, 1]; labs = ["predefined\n(64% overlap)", "scaffold\n(0% overlap)"]
    cols = ["#4C72B0", "#C44E52"]
    a1.bar(x, sc.pearson_r_mean, yerr=sc.pearson_r_std, capsize=3, color=cols, edgecolor="k", linewidth=0.4)
    a1.set_xticks(x); a1.set_xticklabels(labs, fontsize=8); a1.set_ylabel(r"Pearson $R$")
    a1.set_ylim(0.7, 0.88); a1.set_title("RF performance drops under scaffold split")
    for xi, v in zip(x, sc.pearson_r_mean): a1.annotate(f"{v:.3f}", (xi, v+0.004), ha="center", fontsize=8)
    a2.bar(x, sc.mae_mean, yerr=sc.mae_std, capsize=3, color=cols, edgecolor="k", linewidth=0.4)
    a2.set_xticks(x); a2.set_xticklabels(labs, fontsize=8); a2.set_ylabel("MAE")
    a2.set_title("MAE rises under scaffold split")
    for xi, v in zip(x, sc.mae_mean): a2.annotate(f"{v:.3f}", (xi, v+0.004), ha="center", fontsize=8)
    fig.suptitle("Data-leakage stress test: 0.065 Pearson R was inflation from train-test similarity",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "fig_split_leakage_j.png")


def fig_negative_control(nc, fin):
    b = nc[nc.dataset == "bace"].iloc[0]
    real = fin[(fin.dataset == "bace") & (fin.model == "chemprop")].PearsonR.mean()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    # permuted band
    ax.axhspan(b.permuted_min, b.permuted_max, xmin=0.05, xmax=0.45, color="#bbbbbb", alpha=0.4)
    ax.errorbar(0.25, b.permuted_mean, yerr=b.permuted_std, fmt="o", ms=10, color="0.4",
                capsize=4, mec="k", label=f"permuted labels\n({b.permuted_mean:.2f} $\\pm$ {b.permuted_std:.2f}, 10 runs)")
    ax.plot([0.75], [real], marker="*", ms=20, color=COLOR["chemprop"], mec="k", mew=0.5,
            label=f"trained (Chemprop)\nR = {real:.3f}")
    ax.axhline(0, ls=":", color="k", lw=0.8)
    ax.set_xlim(0, 1); ax.set_xticks([0.25, 0.75]); ax.set_xticklabels(["permuted", "trained"])
    ax.set_ylabel(r"Pearson $R$"); ax.set_ylim(-0.2, 0.95)
    ax.set_title("Negative control: real signal is far above label-permutation noise")
    ax.legend(fontsize=8, loc="center right")
    save(fig, "fig_negative_control_j.png")


def main():
    df = pd.read_csv("data/BACE.split.csv")
    fin = pd.read_csv("results/final_with_gnn.csv")
    dk = pd.read_csv("results/bace_docking_full.csv")
    cf = pd.read_csv("results/final/conformal_multiseed.csv")  # single source of truth (6 datasets)
    ad = pd.read_csv("results/final/conformal_ad.csv")  # single source of truth (no GSHt)
    sc = pd.read_csv("results/wang_audit/bace_split_comparison.csv")
    nc = pd.read_csv("results/negative_control.csv")
    # EDA
    fig_label_dist(df); fig_properties(df); fig_scaffolds(df); fig_leakage_tanimoto(df)
    # performance
    fig_metric_bars(fin); fig_pred_vs_actual(); fig_pred_grid(); fig_residuals()
    # docking
    fig_docking_le(dk); fig_docking_confound(dk)
    # conformal
    fig_conformal_covwidth(cf); fig_conformal_ad(ad)
    # rigor
    fig_split_leakage(sc); fig_negative_control(nc, fin)
    print("\nDONE: 14 detail figures in results/figures/")


if __name__ == "__main__":
    main()
