"""Phase B payoff: on same-series pairs (esp. activity cliffs), can PBCNet2 (structure, pairwise)
rank the activity difference better than ligand models (RF/TabPFN), whose pair-delta is just the
difference of two absolute predictions? Ligand models famously fail on cliffs; PBCNet2 is built
for relative (lead-opt) ranking. We compare |Spearman(true dpIC50, predicted delta)| on cliff vs
series-control pairs, plus direction accuracy on cliffs.
"""
from __future__ import annotations
import sys, warnings; from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, scienceplots  # noqa: F401
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix"})
warnings.filterwarnings("ignore")
FIG = Path("results/figures"); FINAL = Path("results/final")

def main():
    P = pd.read_csv("results/docking/pbcnet_pred.csv")
    rf = pd.read_csv("results/tuning/rf_morgan_test.csv").set_index("smiles").y_pred.to_dict()
    tab = pd.read_csv("results/bace_clean_final/tabpfn_desc_test.csv").set_index("smiles").y_pred.to_dict()
    P["rf_delta"] = [rf.get(a, np.nan) - rf.get(b, np.nan) for a, b in zip(P.smiles1, P.smiles2)]
    P["tab_delta"] = [tab.get(a, np.nan) - tab.get(b, np.nan) for a, b in zip(P.smiles1, P.smiles2)]
    # orient PBCNet2 pre to the dpIC50 sign convention (it predicts ddG-like; flip if anti-correlated)
    s_all = spearmanr(P.dpic50, P.pre)[0]
    P["pbc_delta"] = P.pre * (1 if s_all >= 0 else -1)
    P = P.dropna(subset=["rf_delta", "tab_delta"]).reset_index(drop=True)

    def srow(sub):
        return {m: abs(spearmanr(sub.dpic50, sub[c])[0]) for m, c in
                [("PBCNet2", "pbc_delta"), ("RF", "rf_delta"), ("TabPFN", "tab_delta")]}
    groups = {"activity cliffs": P[P.is_cliff], "series controls": P[~P.is_cliff], "all pairs": P}
    tbl = pd.DataFrame({g: srow(s) for g, s in groups.items()}).T
    tbl["n"] = [len(s) for s in groups.values()]
    tbl.round(3).to_csv(FINAL / "pbcnet_vs_ligand.csv")
    print(tbl.round(3).to_string())

    # direction accuracy on cliffs (sign of predicted delta == sign of true)
    cl = P[P.is_cliff]
    dir_acc = {m: float(np.mean(np.sign(cl[c]) == np.sign(cl.dpic50))) for m, c in
               [("PBCNet2", "pbc_delta"), ("RF", "rf_delta"), ("TabPFN", "tab_delta")]}
    print("cliff direction accuracy:", {k: round(v, 3) for k, v in dir_acc.items()})

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
    gx = list(groups); models = ["PBCNet2", "RF", "TabPFN"]; cols = ["#C44E52", "#4C72B0", "#55A868"]
    w = 0.26; x = np.arange(len(gx))
    for k, m in enumerate(models):
        a.bar(x + (k - 1) * w, [tbl.loc[g, m] for g in gx], w, label=m, color=cols[k], edgecolor="k", linewidth=0.4)
    a.set_xticks(x); a.set_xticklabels([f"{g}\n(n={int(tbl.loc[g,'n'])})" for g in gx], fontsize=8.5)
    a.set_ylabel(r"$|$Spearman$|$ (true $\Delta$pIC$_{50}$ vs pred $\Delta$)"); a.legend(fontsize=8)
    a.set_title("(a) pairwise activity-difference ranking", fontsize=10)
    bx = np.arange(len(models))
    b.bar(bx, [dir_acc[m] for m in models], color=cols, edgecolor="k", linewidth=0.4)
    b.axhline(0.5, ls="--", color="0.5", lw=0.9, label="chance")
    for i, m in enumerate(models):
        b.annotate(f"{dir_acc[m]:.2f}", (i, dir_acc[m]), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
    b.set_xticks(bx); b.set_xticklabels(models, fontsize=9); b.set_ylim(0, 1)
    b.set_ylabel("direction accuracy on cliffs"); b.legend(fontsize=8)
    b.set_title(f"(b) get the cliff direction right? (n={len(cl)})", fontsize=10)
    fig.suptitle("Phase B  PBCNet2 (structure, pairwise) vs ligand models on BACE activity cliffs",
                 y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "fig_pbcnet_cliffs_j.png", dpi=300, bbox_inches="tight")
    print("wrote fig_pbcnet_cliffs_j.png + pbcnet_vs_ligand.csv")

if __name__ == "__main__":
    main()
