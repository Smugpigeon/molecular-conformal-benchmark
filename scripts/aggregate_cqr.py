"""Aggregate CQR results + compare against split conformal (the key delta).

For the backbones present in both protocols (MolFormer, Chemprop; GBM~RF as the
classical pair), report split-conformal vs CQR interval width + coverage. CQR's
selling point: adaptive intervals that are tighter at matched (valid) coverage.

Run: python scripts/aggregate_cqr.py
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
# Paper-track CQR: the four predefined-split bioactivity/physicochemical datasets. GSHt is
# quarantined (not a paper dataset); QM7/QM8 use random splits where the AD/CQR contrast is
# near-degenerate, so they are excluded from this supplementary analysis.
REG = ["esol", "freesolv", "lipophilicity", "bace"]
# CQR model -> the split-conformal model it should be compared against.
# unimol is the headline pair: does adaptive CQR fix its "tightest but
# under-covers" split-conformal failure?
PAIR = {"gbm": "rf", "molformer": "molformer", "chemprop": "chemprop", "unimol": "unimol"}
MLAB = {"gbm": "GBM/RF", "molformer": "MolFormer", "chemprop": "Chemprop", "unimol": "Uni-Mol"}


def main() -> int:
    configure_logging("INFO")
    cqr_dir = Path("results/cqr")
    if not cqr_dir.exists() or not any(cqr_dir.glob("*.csv")):
        logger.error("no CQR results found"); return 1
    cqr = pd.concat([pd.read_csv(p) for p in cqr_dir.glob("*.csv")], ignore_index=True)
    g = cqr.groupby(["dataset", "model"]).agg(
        cqr_width=("width", "mean"), cqr_width_std=("width", "std"),
        cqr_cov=("coverage", "mean"), cqr_cov_std=("coverage", "std"),
        n=("seed", "count")).reset_index()
    g = g[g.dataset.isin(REG)].reset_index(drop=True)  # paper-track datasets only (no GSHt)
    Path("results/final").mkdir(parents=True, exist_ok=True)
    g.to_csv("results/final/cqr_summary.csv", index=False)  # single source of truth

    # Split conformal (multi-seed) for comparison.
    split = None
    sp = Path("results/final/conformal_multiseed.csv")  # single source of truth (6 datasets)
    if sp.exists():
        split = pd.read_csv(sp)

    print("\n" + "=" * 76)
    print("SPLIT CONFORMAL vs CQR @ 90% (width | coverage; mean over seeds)")
    print("=" * 76)
    print(f"{'dataset':<14}{'backbone':<11}{'split_w':>9}{'cqr_w':>9}{'split_cov':>11}{'cqr_cov':>9}")
    rows = []
    for d in REG:
        for cm, sm in PAIR.items():
            cr = g[(g.dataset == d) & (g.model == cm)]
            if len(cr) == 0:
                continue
            cr = cr.iloc[0]
            sw = scov = np.nan
            if split is not None:
                sr = split[(split.dataset == d) & (split.model == sm)]
                if len(sr):
                    sw, scov = sr.iloc[0]["width_mean"], sr.iloc[0]["cov_mean"]
            tighter = "tighter" if (not np.isnan(sw) and cr["cqr_width"] < sw) else ""
            print(f"{d:<14}{MLAB[cm]:<11}{sw:>9.2f}{cr['cqr_width']:>9.2f}"
                  f"{scov:>11.3f}{cr['cqr_cov']:>9.3f}  {tighter}")
            rows.append({"dataset": d, "backbone": cm, "split_w": sw, "cqr_w": cr["cqr_width"],
                         "split_cov": scov, "cqr_cov": cr["cqr_cov"]})

    cmp = pd.DataFrame(rows)
    cmp = cmp[cmp.dataset.isin(REG)].reset_index(drop=True)  # paper-track only (no GSHt)
    cmp.to_csv("results/final/split_vs_cqr.csv", index=False)  # single source of truth

    # Figure: split vs CQR width per backbone (one subplot per backbone).
    models = [m for m in PAIR if m in g["model"].unique()]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4), squeeze=False)
    for ax, m in zip(axes[0], models):
        sub = cmp[cmp.backbone == m]
        x = np.arange(len(sub)); w = 0.38
        ax.bar(x - w / 2, sub["split_w"], w, label="split", color="#888")
        ax.bar(x + w / 2, sub["cqr_w"], w, label="CQR", color="#1f77b4")
        ax.set_xticks(x); ax.set_xticklabels(sub["dataset"], rotation=30, fontsize=8)
        ax.set_title(MLAB[m]); ax.set_ylabel("interval width")
        ax.legend(fontsize=8)
    fig.suptitle("Split conformal vs CQR interval width")
    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/figures/split_vs_cqr.png", dpi=150)
    print("\nsaved results/final/cqr_summary.csv + results/final/split_vs_cqr.csv + figures/split_vs_cqr.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
