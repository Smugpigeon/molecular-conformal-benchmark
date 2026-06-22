"""A6  Per-scaffold error ranking: which Murcko scaffolds (chemical series) are hardest, and is
scaffold difficulty driven by how little the training set supports that scaffold (scaffold-level
applicability domain)? Consensus error = mean |pred-true| across all 8 models. Local, light.

Run: /opt/anaconda3/bin/python3 scripts/make_scaffold_error.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_error_analysis import load_wide, MODELS  # noqa: E402

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": True})
warnings.filterwarnings("ignore")
FIG = Path("results/figures"); FINAL = Path("results/final")


def main():
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")

    wide = load_wide()
    labels = [m for m, _, _ in MODELS if m in wide.columns]
    cons = np.mean([np.abs(wide[m] - wide.y_true) for m in labels], axis=0)
    clean = pd.read_csv("data/processed/bace_clean.csv")
    tr_scaf = (clean[clean.set == "train"].smiles
               .map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))))
    tr_count = tr_scaf.value_counts().to_dict()

    df = pd.DataFrame({"smiles": wide.smiles, "y_true": wide.y_true, "err": cons})
    df["scaffold"] = df.smiles.map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)))
    df["train_support"] = df.scaffold.map(lambda s: tr_count.get(s, 0))

    g = df.groupby("scaffold").agg(n_test=("err", "size"), mean_err=("err", "mean"),
                                   train_support=("train_support", "first")).reset_index()
    g = g[g.n_test >= 3].sort_values("mean_err", ascending=False)  # series with >=3 test mols
    g.round(3).to_csv(FINAL / "scaffold_error_ranking.csv", index=False)
    print(f"{len(g)} scaffolds with >=3 test molecules")

    # (1) ranking bar (top hardest + easiest) + (2) difficulty vs train support
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    top = pd.concat([g.head(8), g.tail(6)])
    y = np.arange(len(top))
    colors = ["#C44E52"] * 8 + ["#4C72B0"] * 6
    a.barh(y, top.mean_err[::-1], color=colors[::-1], edgecolor="k", linewidth=0.4)
    a.set_yticks(y); a.set_yticklabels([f"n={int(n)}, train={int(t)}"
                                        for n, t in zip(top.n_test[::-1], top.train_support[::-1])],
                                       fontsize=7)
    a.set_xlabel("mean consensus |error| (all 8 models)")
    a.set_title("(a) hardest (red) vs easiest (blue) scaffolds", fontsize=10)
    b.scatter(g.train_support, g.mean_err, s=np.clip(g.n_test * 8, 20, 200), alpha=0.6,
              color="#8172B3", edgecolor="k", linewidth=0.3)
    from scipy.stats import spearmanr
    rho = spearmanr(g.train_support, g.mean_err)[0]
    b.set_xlabel("# training molecules sharing the scaffold"); b.set_ylabel("mean consensus |error|")
    b.set_title(f"(b) under-supported scaffolds err more\n(Spearman rho={rho:.2f}, dot=#test)", fontsize=10)
    fig.suptitle("A6  Per-scaffold (chemical series) error ranking -- BACE test", y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "fig_err_scaffold_ranking_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_err_scaffold_ranking_j.png (train-support vs error Spearman={rho:.2f})")

    # RDKit grid of the 8 hardest scaffolds
    hard = g.head(8)
    mols = [Chem.MolFromSmiles(s) for s in hard.scaffold if s]
    legends = [f"err={e:.2f} n={int(n)} train={int(t)}"
               for e, n, t in zip(hard.mean_err, hard.n_test, hard.train_support)]
    img = Draw.MolsToGridImage([m for m in mols if m], molsPerRow=4, subImgSize=(260, 190),
                               legends=legends)
    img.save(FIG / "fig_err_scaffold_structures.png")
    print("wrote fig_err_scaffold_structures.png")
    print("\nTop-5 hardest series:")
    print(g.head(5)[["n_test", "train_support", "mean_err"]].to_string(index=False))


if __name__ == "__main__":
    main()
