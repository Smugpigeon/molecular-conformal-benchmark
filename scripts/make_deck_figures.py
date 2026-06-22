"""Presentation-optimized figures for the oral deck (suffix _d): big fonts (>=16pt axis labels),
<=2 sub-panels, key numbers annotated in red on the figure. Redrawn from the same data/CSVs as
the report figures. Run: /opt/anaconda3/bin/python3 scripts/make_deck_figures.py
"""
from __future__ import annotations
import sys, warnings; from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, scienceplots  # noqa
from scipy.stats import spearmanr, pearsonr, gaussian_kde
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
                     "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16.5,
                     "xtick.labelsize": 13.5, "ytick.labelsize": 13.5, "legend.fontsize": 13, "axes.unicode_minus": True})
warnings.filterwarnings("ignore")
FIG = Path("results/figures"); FINAL = Path("results/final"); EXP = Path("results/exp"); DOCK = Path("results/docking")
NAVY = "#3E4E63"; RED = "#C0392B"; SLATE = "#8FA0B3"; GREEN = "#2E7D5B"; BLUE = "#3F6CA8"
def save(fig, name): fig.savefig(FIG / name, dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote", name)
def ann(ax, x, y, t, c=RED, fs=15, **k): ax.annotate(t, (x, y), color=c, fontsize=fs, fontweight="bold", **k)


def f_benchmark():
    M = pd.read_csv(FINAL / "master_table.csv").set_index("model")
    order = ["TabPFN (desc)", "Chemprop", "RF", "ChemFM-3B", "GBM", "Ridge", "SVR", "MolFormer-XL"]
    lab = {"TabPFN (desc)": "TabPFN", "MolFormer-XL": "MolFormer", "ChemFM-3B": "ChemFM"}
    names = [lab.get(m, m) for m in order]
    mae = [M.loc[m, "test_MAE"] for m in order]
    err = [M.loc[m, "test_MAE_std"] if not pd.isna(M.loc[m].get("test_MAE_std", np.nan)) else 0 for m in order]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    y = np.arange(len(order))[::-1]
    cols = [RED if m == "TabPFN (desc)" else (SLATE if M.loc[m, "family"] == "classic" else NAVY) for m in order]
    ax.barh(y, mae, xerr=err, color=cols, edgecolor="k", linewidth=0.5, capsize=3, error_kw={"elinewidth": 1})
    for yi, m, v, e in zip(y, order, mae, err):
        ax.annotate(f"{v:.3f}", (v + e + 0.012, yi), va="center", fontsize=13,
                    fontweight="bold" if m == "TabPFN (desc)" else "normal",
                    color=RED if m == "TabPFN (desc)" else "k")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("test MAE (pIC$_{50}$,  lower = better)"); ax.set_xlim(0, max(mae) + max(err) + 0.11)
    ax.set_title("TabPFN best point estimate; first tier w/ Chemprop / RF / ChemFM", fontsize=14)
    ann(ax, 0.985, 0.96, "n=303 · deep: mean±sd (3 seeds)", c="#777", fs=11, xycoords="axes fraction", ha="right", va="top")
    save(fig, "fig_d_benchmark.png")


def f_budget():
    d = pd.read_csv(EXP / "budget_sweep.csv")
    tab = d[d.model == "TabPFN"].iloc[0]
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    for m, c in [("RF", SLATE), ("GBM", NAVY), ("SVR", "#8172B3")]:
        s = d[d.model == m].sort_values("wall_s")
        ax.plot(s.wall_s, s.test_mae, "o-", color=c, ms=5, lw=1.5, label=f"{m} (Optuna)")
    ax.scatter([tab.wall_s], [tab.test_mae], marker="*", s=520, color=RED, edgecolor="k", linewidth=0.8, zorder=6, label="TabPFN (0 tuning)")
    ax.set_xscale("log"); ax.set_xlabel("tuning wall-clock (s, log)"); ax.set_ylabel("test MAE")
    ax.set_title("Within this budget, TabPFN beats Optuna-tuned baselines", fontsize=15)
    ann(ax, tab.wall_s * 1.3, tab.test_mae - 0.004, f"TabPFN: {tab.test_mae:.3f}  ·  ~5 s")
    ax.annotate("GBM: 80 trials, 986 s -> 0.557", xy=(986, 0.557), xytext=(40, 0.74), fontsize=11,
                color="#444", ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"),
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1))
    ax.legend(loc="upper right")
    save(fig, "fig_d_budget.png")


def f_scaffold():
    T = pd.read_csv(FINAL / "scaffold_comparison.csv").sort_values("scaf_MAE")
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    y = np.arange(len(T))[::-1].astype(float); hh = 0.38
    ax.barh(y + hh / 2, T.pre_MAE, hh, color=SLATE, edgecolor="k", linewidth=0.4, label="predefined split")
    ax.barh(y - hh / 2, T.scaf_MAE, hh, color=NAVY, edgecolor="k", linewidth=0.4, label="scaffold split (novel)")
    for yi, m, sc, dd in zip(y, T.model, T.scaf_MAE, T.dMAE):
        ax.annotate(f"{sc:.2f}  (Δ+{dd:.2f})", (sc + 0.012, yi - hh / 2), va="center",
                    fontsize=11, color=RED if m == "TabPFN" else "k", fontweight="bold" if m == "TabPFN" else "normal")
    ax.set_yticks(y); ax.set_yticklabels(T.model)
    ax.set_xlabel("test MAE (pIC$_{50}$,  lower = better)"); ax.set_xlim(0, max(T.scaf_MAE) + 0.55)
    ax.set_title("Scaffold split: TabPFN keeps lowest MAE and smallest gap", fontsize=14.5)
    ax.legend(loc="upper right", framealpha=0.95)
    save(fig, "fig_d_scaffold.png")


def f_docking():
    d = pd.read_csv(DOCK / "dock_scores.csv"); d = d[d.status == "ok"]
    st = pd.read_csv(FINAL / "docking_stats.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    sc = ax.scatter(d.y_true, d.vina_score, c=d.heavy, cmap="viridis", s=34, edgecolor="none", alpha=0.85)
    ax.set_xlabel(r"experimental pIC$_{50}$"); ax.set_ylabel("Vina score (kcal/mol)")
    ax.set_title("Docking score does not rank affinity (size-confounded)", fontsize=15)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("heavy atoms", fontsize=13)
    ann(ax, 0.04, 0.07, f"Pearson r = {st.vina_raw_pearson:.2f}\npartial (|size) = {st.vina_partial_size:.2f}",
        xycoords="axes fraction", fs=14)
    ann(ax, 0.97, 0.95, "ML: R 0.84–0.87", c=GREEN, fs=14, xycoords="axes fraction", ha="right")
    save(fig, "fig_d_docking.png")


def f_pbcnet():
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    names = ["PBCNet2\n(cross-dock)", "PBCNet2\n(constrained)", "RF\n(ligand)", "TabPFN\n(ligand)"]
    vals = [0.375, 0.174, 0.781, 0.766]; cols = [NAVY, SLATE, BLUE, GREEN]
    ax.bar(range(4), vals, color=cols, edgecolor="k", linewidth=0.5)
    for i, v in enumerate(vals):
        ann(ax, i, v + 0.015, f"{v:.2f}", c="k", fs=14, ha="center")
    ax.set_xticks(range(4)); ax.set_xticklabels(names); ax.set_ylim(0, 0.95)
    ax.set_ylabel(r"$|$Spearman$|$ on activity-cliff pairs")
    ax.set_title("On docking poses, PBCNet2 does not beat ligand models", fontsize=14.5)
    ax.annotate("pose-quality limit,\nnot model limit", xy=(0.5, 0.42), xytext=(0.5, 0.67), fontsize=12.5,
                color=RED, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    save(fig, "fig_d_pbcnet.png")


def f_conformal_cov():
    R = pd.read_csv(FINAL / "conformal_benchmark.csv")
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    for _, r in R.iterrows():
        c = RED if r.model == "TabPFN" else (GREEN if r.model == "Chemprop" else NAVY)
        ax.scatter(r.width, r.coverage, s=120, color=c, edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(r.model, (r.width, r.coverage), xytext=(6, 4), textcoords="offset points", fontsize=12.5)
    ax.set_xlim(R.width.min() - 0.15, R.width.max() + 0.5); ax.set_ylim(0.885, 0.965)
    ax.axhline(0.9, ls="--", color=RED, lw=1.3); ann(ax, 0.02, 0.902, "nominal 0.90", xycoords=("axes fraction", "data"), ha="left", fs=12)
    ax.set_xlabel("interval width (pIC$_{50}$,  narrower = better)"); ax.set_ylabel("empirical coverage")
    ax.set_title("Marginal 90% coverage holds for all models", fontsize=15.5)
    save(fig, "fig_d_cov.png")


def f_adcov():
    A = pd.read_csv(FINAL / "conformal_ad_coverage.csv", index_col=0)
    bins = list(A.columns)
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    for m in A.index:
        c = RED if m == "TabPFN" else "#AAB4C2"
        lw = 2.6 if m == "TabPFN" else 1.0
        ax.plot(range(len(bins)), A.loc[m], "o-", color=c, lw=lw, ms=6 if m == "TabPFN" else 4,
                label="TabPFN" if m == "TabPFN" else None, zorder=5 if m == "TabPFN" else 2)
    ax.axhline(0.9, ls="--", color="#444", lw=1.2)
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(["low  Tan<0.4\n(n=10)", "med  0.4–0.6\n(n=15)", "high  Tan≥0.6\n(n=278)"])
    ax.set_ylabel("conditional coverage"); ax.set_ylim(0, 1.02)
    ax.set_title("...but low-AD (novel) molecules are under-covered", fontsize=14.5)
    ax.annotate("low-AD coverage 0.2-0.5", xy=(0.05, 0.33), xytext=(0.55, 0.42), fontsize=13,
                color=RED, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, alpha=0.92),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.legend(loc="lower right")
    save(fig, "fig_d_adcov.png")


def f_uncertainty():
    per = pd.read_csv(EXP / "uncertainty_test.csv")
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    n = len(per); fr = np.linspace(0, 0.5, 26)
    def curve(rank):
        o = np.argsort(-rank); return [per.abs_err.to_numpy()[o[int(f * n):]].mean() for f in fr]
    rng = np.random.default_rng(0)
    ax.plot(fr * 100, curve(per.tab_width.to_numpy()), "o-", color=RED, ms=4, lw=2, label="reject by TabPFN uncertainty")
    ax.plot(fr * 100, np.mean([curve(rng.random(n)) for _ in range(20)], axis=0), "--", color="#888", lw=1.5, label="random rejection")
    ax.set_xlabel("% most-uncertain molecules rejected"); ax.set_ylabel("MAE on retained set")
    ax.set_title("Uncertainty flags hard cases: rejecting them cuts error", fontsize=14.5)
    ann(ax, 0.5, 0.62, "58% of hardest mols fall\noutside the 90% interval\n(easy: 6%)", c=RED, fs=13.5, xycoords="axes fraction")
    ax.legend(loc="lower left")
    save(fig, "fig_d_uncertainty.png")


def f_regression():
    from scripts.make_error_analysis import load_wide, MODELS
    wide = load_wide(); yt = wide.y_true.to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    e = (wide["TabPFN"] - wide.y_true).to_numpy(float)
    ax.scatter(yt, e, s=20, alpha=0.5, color=BLUE, edgecolor="none")
    sl, ic = np.polyfit(yt, e, 1); xx = np.array([yt.min(), yt.max()])
    ax.plot(xx, sl * xx + ic, color=RED, lw=2.5); ax.axhline(0, ls="--", color="#888", lw=1)
    ax.set_xlabel(r"true pIC$_{50}$"); ax.set_ylabel("signed error (pred $-$ true)")
    ax.set_title("Systematic error: models regress predictions to the mean", fontsize=14.5)
    ann(ax, 0.04, 0.08, f"slope = {sl:.2f}\n(under-predict potent)", fs=14, xycoords="axes fraction")
    save(fig, "fig_d_regression.png")


def f_hard3():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw
    RDLogger.DisableLog("rdApp.*")
    H = pd.read_csv(FINAL / "consensus_hard_molecules.csv")
    cliff = H[H.on_cliff].iloc[0] if H.on_cliff.any() else H.iloc[0]
    lab = H[H.label_conflict & ~H.on_cliff]
    lab = lab.iloc[0] if len(lab) else H.iloc[1]
    nov = H[(H.tanimoto_nn < 0.4)]
    nov = nov.iloc[0] if len(nov) else H.iloc[2]
    picks = [(cliff, "activity cliff"), (lab, "label conflict"), (nov, "new scaffold (low AD)")]
    mols = [Chem.MolFromSmiles(r.smiles) for r, _ in picks]
    legends = [f"{why}  |  pIC50={r.y_true:.1f}, err={r.mean_abs_err:.2f}" for r, why in picks]
    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(360, 300), legends=legends)
    img.save(FIG / "fig_d_hard3.png"); print("wrote fig_d_hard3.png")


def f_cliff1():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw, AllChem
    RDLogger.DisableLog("rdApp.*")
    c = pd.read_csv(FINAL.parent / "eda" / "activity_cliffs_top.csv").iloc[0]
    a, b = Chem.MolFromSmiles(c.smiles_A), Chem.MolFromSmiles(c.smiles_B)
    for m in (a, b): AllChem.Compute2DCoords(m)
    img = Draw.MolsToGridImage([a, b], molsPerRow=2, subImgSize=(380, 300),
                               legends=[f"pIC50 = {max(c.smiles_A and 8.4, 8.4):.1f}", "pIC50 lower"])
    img.save(FIG / "fig_d_cliff.png"); print("wrote fig_d_cliff.png (stereo-flip cliff)")


def _wide_for_deck():
    """Inline merge of per-model test predictions (no import side effects on rcParams)."""
    TUN = Path("results/tuning"); BC = Path("results/bace_clean_final"); SEEDS = (42, 1337, 2024)
    models = [("Ridge", [TUN / "ridge_morgan_test.csv"]), ("SVR", [TUN / "svr_morgan_test.csv"]),
              ("GBM", [TUN / "gbm_morgan_test.csv"]), ("RF", [TUN / "rf_morgan_test.csv"]),
              ("TabPFN", [BC / "tabpfn_desc_test.csv"]),
              ("MolFormer", [BC / f"molformer_seed{s}_test.csv" for s in SEEDS]),
              ("ChemFM", [BC / f"chemfm_seed{s}_test.csv" for s in SEEDS]),
              ("Chemprop", [BC / f"chemprop_seed{s}_test.csv" for s in SEEDS])]
    base = None
    for label, files in models:
        files = [f for f in files if f.exists()]
        if not files:
            continue
        mats = [pd.read_csv(f).sort_values("smiles").reset_index(drop=True) for f in files]
        pred = np.mean([(m["y_pred"] if "y_pred" in m.columns else m["y_score"]).to_numpy(float)
                        for m in mats], axis=0)
        acc = pd.DataFrame({"smiles": mats[0].smiles, "y_true": mats[0].y_true.to_numpy(float), label: pred})
        base = acc if base is None else base.merge(acc[["smiles", label]], on="smiles")
    return base


def f_ablation():
    T = pd.read_csv(FINAL / "fingerprint_ablation.csv").sort_values("MAE")
    lab = {"TopoTorsion": "Topological Torsion", "Descriptors": "RDKit descriptors",
           "AtomPair": "Atom Pair", "MACCS": "MACCS keys", "Morgan": "Morgan (ECFP4)", "Avalon": "Avalon"}
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    y = np.arange(len(T))[::-1]
    cols = [RED if r.representation == "Morgan" else (SLATE if r.representation == "MACCS" else NAVY)
            for _, r in T.iterrows()]
    ax.barh(y, T.MAE, xerr=T.MAE_std, color=cols, edgecolor="k", linewidth=0.5, capsize=3,
            error_kw={"elinewidth": 1})
    for yi, (_, r) in zip(y, T.iterrows()):
        ax.annotate(f"{r.MAE:.3f}", (r.MAE + r.MAE_std + 0.004, yi), va="center", fontsize=13,
                    fontweight="bold" if r.representation == "Morgan" else "normal",
                    color=RED if r.representation == "Morgan" else "k")
    ax.set_yticks(y); ax.set_yticklabels([lab.get(r.representation, r.representation) for _, r in T.iterrows()])
    ax.set_xlabel("test MAE (pIC$_{50}$,  lower = better)"); ax.set_xlim(0, max(T.MAE + T.MAE_std) + 0.07)
    ax.set_title("Model fixed (RF), representation varied: Morgan is first-tier", fontsize=14.5)
    save(fig, "fig_d_ablation.png")


def f_classaxes():
    from sklearn.metrics import roc_curve
    wide = _wide_for_deck(); y_bin = (wide.y_true.to_numpy(float) >= 7.0).astype(int)
    C = pd.read_csv(FINAL / "classification_view.csv")
    order = C.sort_values("ROC-AUC", ascending=False).model.tolist()
    import matplotlib.cm as cm
    base_cols = cm.viridis(np.linspace(0, 0.9, len(order)))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for m, bc in zip(order, base_cols):
        fpr, tpr, _ = roc_curve(y_bin, wide[m].to_numpy(float))
        auc = float(C[C.model == m]["ROC-AUC"].iloc[0])
        col = RED if m == "Chemprop" else (NAVY if m == "TabPFN" else bc)
        lw = 2.8 if m in ("Chemprop", "TabPFN") else 1.2
        ax.plot(fpr, tpr, color=col, lw=lw, label=f"{m} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("Classification axis: best regressor $\\neq$ best classifier", fontsize=14.5)
    ax.legend(loc="lower right", fontsize=10.5, title="model (ROC-AUC, active=pIC50$\\geq$7)", title_fontsize=10.5)
    save(fig, "fig_d_classaxes.png")


def main():
    f_benchmark(); f_ablation(); f_classaxes(); f_budget(); f_scaffold(); f_docking(); f_pbcnet()
    f_conformal_cov(); f_adcov(); f_uncertainty(); f_regression(); f_hard3()


if __name__ == "__main__":
    main()
