"""BACE-1 docking pipeline (Part B) with size-matched sampling.

Docks BACE ligands into beta-secretase (PDB 4D8C, chain A) and compares the
Vina score against experimental pIC50.

Per 项目调研汇总.md v2 §5 + §16 honesty: the pilot showed actives were large
peptides and inactives tiny fragments, so raw correlation is confounded by
ligand SIZE (Vina favours bigger ligands). This script therefore:
  1. restricts to a drug-like heavy-atom window (default 20-40) to remove the
     peptide-vs-fragment confound,
  2. records MW + heavy-atom count,
  3. reports the raw correlation, the size confound, AND the partial
     correlation controlling for size.

Run (pilot):  python scripts/dock_bace.py --n 6
Run (main):   python scripts/dock_bace.py --n 30 --exhaustiveness 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr, spearmanr

from src.data.loaders import load_dataset
from src.structure.dock import dock_many
from src.utils.logging_setup import configure_logging
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


def _heavy(smi: str) -> float:
    m = Chem.MolFromSmiles(smi)
    return float(m.GetNumHeavyAtoms()) if m else float("nan")


def _mw(smi: str) -> float:
    m = Chem.MolFromSmiles(smi)
    return float(Descriptors.MolWt(m)) if m else float("nan")


def _partial_corr(x, y, z) -> float:
    """Partial correlation of x,y controlling for z."""
    rxy = pearsonr(x, y)[0]
    rxz = pearsonr(x, z)[0]
    ryz = pearsonr(y, z)[0]
    denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
    return float((rxy - rxz * ryz) / denom) if denom > 0 else float("nan")


def main() -> int:
    configure_logging("INFO")
    set_all_seeds(42)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receptor", default="structure/bace/bace_receptor.pdbqt")
    p.add_argument("--n", type=int, default=30, help="# ligands spanning pIC50 range")
    p.add_argument("--min-heavy", type=int, default=20)
    p.add_argument("--max-heavy", type=int, default=40)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--out", type=Path, default=Path("results/bace_docking_full.csv"))
    p.add_argument("--plot", type=Path, default=Path("results/figures/bace_docking_scatter.png"))
    args = p.parse_args()

    if not Path(args.receptor).exists():
        logger.error(f"Receptor not found: {args.receptor}")
        return 1

    split = load_dataset("bace")
    df = pd.concat([split.train, split.val, split.test]).reset_index(drop=True)
    df["heavy"] = df["smiles"].apply(_heavy)
    df["mw"] = df["smiles"].apply(_mw)

    # Size window -> drug-like (removes peptide giants + tiny fragments).
    win = df[(df["heavy"] >= args.min_heavy) & (df["heavy"] <= args.max_heavy)].copy()
    logger.info(
        f"Size window [{args.min_heavy},{args.max_heavy}] heavy atoms: "
        f"{len(win)}/{len(df)} molecules "
        f"(pIC50 {win['label'].min():.1f}-{win['label'].max():.1f})"
    )

    # Sample n spanning the pIC50 range (size already matched by the window).
    win = win.sort_values("label").reset_index(drop=True)
    idx = np.linspace(0, len(win) - 1, args.n).round().astype(int)
    sample = win.iloc[np.unique(idx)].reset_index(drop=True)

    scores = dock_many(
        sample["smiles"].tolist(), args.receptor, exhaustiveness=args.exhaustiveness
    )
    sample["vina"] = scores
    sample["pIC50"] = sample["label"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sample[["smiles", "pIC50", "heavy", "mw", "vina"]].to_csv(args.out, index=False)
    logger.info(f"Wrote {args.out}")

    valid = sample.dropna(subset=["vina"]).copy()
    if len(valid) < 3:
        logger.error("Too few successful docks for statistics.")
        return 1

    r_act = pearsonr(valid["pIC50"], valid["vina"])[0]
    rho_act = spearmanr(valid["pIC50"], valid["vina"])[0]
    r_size = pearsonr(valid["heavy"], valid["vina"])[0]
    r_pic_size = pearsonr(valid["pIC50"], valid["heavy"])[0]
    partial = _partial_corr(valid["pIC50"].values, valid["vina"].values, valid["heavy"].values)

    print("\n" + "=" * 56)
    print("BACE-1 DOCKING vs ACTIVITY (size-matched, §16 honest)")
    print("=" * 56)
    print(f"n (docked ok)                 = {len(valid)}")
    print(f"Pearson(pIC50, vina)          = {r_act:+.3f}   <- main result")
    print(f"Spearman(pIC50, vina)         = {rho_act:+.3f}")
    print(f"Pearson(heavy_atoms, vina)    = {r_size:+.3f}   <- size confound")
    print(f"Pearson(pIC50, heavy_atoms)   = {r_pic_size:+.3f}")
    print(f"PARTIAL r(pIC50,vina | size)  = {partial:+.3f}   <- size-controlled")
    print("(negative = stronger binders dock better; partial r isolates real signal)")

    # Scatter plot colored by ligand size.
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(6.2, 5))
    sc = ax.scatter(
        valid["pIC50"], valid["vina"], c=valid["heavy"],
        cmap="viridis", s=70, edgecolor="k", linewidth=0.5,
    )
    ax.set_xlabel("Experimental pIC50")
    ax.set_ylabel("Vina docking score (kcal/mol)")
    ax.set_title(
        f"BACE-1: docking vs activity (size-matched)\n"
        f"Pearson={r_act:.2f}, partial(|size)={partial:.2f}, n={len(valid)}"
    )
    fig.colorbar(sc, label="heavy atoms")
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=150)
    print(f"saved {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
