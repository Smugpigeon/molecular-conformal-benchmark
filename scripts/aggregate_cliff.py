"""Aggregate activity-cliff results across backbones (RF + neural).

Novelty thrust (项目调研汇总.md v2 §7.5.1): does any backbone -- especially
3D Uni-Mol -- handle activity cliffs better? Combines RF (results/cliff_rf.csv)
with neural per-dataset results (results/cliff_neural/*.csv) and compares the
cliff/non-cliff RMSE ratio. SemiMol (2026) skipped Uni-Mol/MolFormer; this
answers whether they beat the 2D baseline on cliffs.

Run: python scripts/aggregate_cliff.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.utils.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)
MLAB = {"rf": "RF+Morgan", "molformer": "MolFormer-XL", "unimol": "Uni-Mol", "chemfm": "ChemFM-3B"}


def main() -> int:
    configure_logging("INFO")
    frames = []
    rf = Path("results/cliff_rf.csv")
    if rf.exists():
        d = pd.read_csv(rf)
        d["model"] = "rf"
        frames.append(d)
    for p in Path("results/cliff_neural").glob("*.csv") if Path("results/cliff_neural").exists() else []:
        frames.append(pd.read_csv(p))
    if not frames:
        logger.error("no cliff results found")
        return 1
    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["cliff_ratio"])
    df.to_csv("results/cliff_all.csv", index=False)

    # Per-model summary across datasets (paired where possible).
    print("\n" + "=" * 64)
    print("ACTIVITY-CLIFF ROBUSTNESS BY BACKBONE (MoleculeACE)")
    print("=" * 64)
    print(f"{'model':<14}{'n_ds':>6}{'mean_cliff_rmse':>17}{'mean_noncliff':>15}{'mean_ratio':>12}")
    summary = []
    for m, g in df.groupby("model"):
        row = {
            "model": m, "n_ds": len(g),
            "cliff_rmse": g["rmse_cliff"].mean(),
            "noncliff_rmse": g["rmse_noncliff"].mean(),
            "ratio": g["cliff_ratio"].mean(),
            "frac_cliff_worse": (g["cliff_ratio"] > 1).mean(),
        }
        summary.append(row)
        print(f"{MLAB.get(m, m):<14}{len(g):>6}{row['cliff_rmse']:>17.3f}"
              f"{row['noncliff_rmse']:>15.3f}{row['ratio']:>12.3f}")

    sdf = pd.DataFrame(summary)
    print("\n(ratio > 1 = worse on cliffs. Lower ratio = more cliff-robust.)")
    best = sdf.loc[sdf["ratio"].idxmin(), "model"]
    print(f"Most cliff-robust backbone (lowest ratio): {MLAB.get(best, best)}")

    # Figure: cliff/non-cliff ratio by model (box across datasets).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = [m for m in ["rf", "molformer", "unimol", "chemfm"] if m in df["model"].unique()]
    data = [df[df.model == m]["cliff_ratio"].values for m in order]
    ax.boxplot(data, labels=[MLAB[m] for m in order], showmeans=True)
    ax.axhline(1.0, color="r", ls="--", lw=1, label="cliff = non-cliff")
    ax.set_ylabel("cliff / non-cliff RMSE ratio")
    ax.set_title("Activity-cliff robustness across MoleculeACE datasets\n(>1 = worse on cliffs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/figures/cliff_compare.png", dpi=150)
    print("saved results/cliff_all.csv + results/figures/cliff_compare.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
