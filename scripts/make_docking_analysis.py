"""Phase-A docking analysis: Vina score vs experimental pIC50 vs ML predictions on the bace_clean
test set. Vina score is negative (more negative = better binding), so it should correlate
NEGATIVELY with pIC50. BACE has a known size confound (actives are large, Vina favours large
ligands), so we report raw AND size-controlled (partial) correlation, and compare against the
ligand-ML models. Honest takeaway expected: docking ranks affinity weakly (and size-confounded),
ML ranks it well -- but docking supplies the binding POSES that the PBCNet2 stage needs.

Run: /opt/anaconda3/bin/python3 scripts/make_docking_analysis.py
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
from scipy.stats import pearsonr, spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": True})
warnings.filterwarnings("ignore")
FIG = Path("results/figures"); FINAL = Path("results/final")


def partial_corr(x, y, z):
    rxy, rxz, ryz = pearsonr(x, y)[0], pearsonr(x, z)[0], pearsonr(y, z)[0]
    d = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return float((rxy - rxz * ryz) / d) if d > 0 else float("nan")


def main():
    d = pd.read_csv("results/docking/dock_scores.csv")
    d = d[d.vina_score.notna()].copy()
    # ML predictions on the same molecules
    rf = pd.read_csv("results/tuning/rf_morgan_test.csv").rename(columns={"y_pred": "rf"})
    tab = pd.read_csv("results/bace_clean_final/tabpfn_desc_test.csv").rename(columns={"y_pred": "tab"})
    d = d.merge(rf[["smiles", "rf"]], on="smiles", how="left").merge(tab[["smiles", "tab"]], on="smiles", how="left")
    n = len(d)

    r_raw = pearsonr(d.y_true, d.vina_score)[0]
    rho_raw = spearmanr(d.y_true, d.vina_score)[0]
    r_size_v = pearsonr(d.heavy, d.vina_score)[0]
    r_size_y = pearsonr(d.heavy, d.y_true)[0]
    r_part = partial_corr(d.y_true.to_numpy(float), d.vina_score.to_numpy(float), d.heavy.to_numpy(float))
    r_rf = pearsonr(d.dropna(subset=["rf"]).y_true, d.dropna(subset=["rf"]).rf)[0]
    r_tab = pearsonr(d.dropna(subset=["tab"]).y_true, d.dropna(subset=["tab"]).tab)[0]

    stats = {"n": n, "vina_raw_pearson": r_raw, "vina_raw_spearman": rho_raw,
             "vina_partial_size": r_part, "heavy_vs_vina": r_size_v, "heavy_vs_pIC50": r_size_y,
             "RF_pearson": r_rf, "TabPFN_pearson": r_tab}
    pd.DataFrame([stats]).round(3).to_csv(FINAL / "docking_stats.csv", index=False)
    for k, v in stats.items():
        print(f"  {k:22s} {v:.3f}" if isinstance(v, float) else f"  {k:22s} {v}")

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    sc = a.scatter(d.y_true, d.vina_score, c=d.heavy, cmap="viridis", s=34, edgecolor="k", linewidth=0.3)
    a.set_xlabel(r"experimental pIC$_{50}$"); a.set_ylabel("Vina score (kcal/mol, lower=better)")
    a.set_title(f"(a) docking vs activity (n={n})\nraw r={r_raw:.2f}, partial(|size) r={r_part:.2f}",
                fontsize=10)
    fig.colorbar(sc, ax=a, label="heavy atoms")
    # |correlation with pIC50|: docking (raw, size-controlled) vs ML
    names = ["Vina\n(raw)", "Vina\n(|size)", "RF\n(ML)", "TabPFN\n(ML)"]
    vals = [abs(r_raw), abs(r_part), abs(r_rf), abs(r_tab)]
    cols = ["#DD8452", "#C44E52", "#4C72B0", "#55A868"]
    b.bar(range(4), vals, color=cols, edgecolor="k", linewidth=0.4)
    for i, v in enumerate(vals):
        b.annotate(f"{v:.2f}", (i, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    b.set_xticks(range(4)); b.set_xticklabels(names, fontsize=8.5)
    b.set_ylabel(r"$|$Pearson$|$ with experimental pIC$_{50}$"); b.set_ylim(0, 1)
    b.set_title("(b) affinity ranking: docking vs ML", fontsize=10)
    fig.suptitle("Phase A  Structure docking into beta-secretase 4D8C -- score vs activity vs ML",
                 y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "fig_docking_analysis_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_docking_analysis_j.png + docking_stats.csv")


if __name__ == "__main__":
    main()
