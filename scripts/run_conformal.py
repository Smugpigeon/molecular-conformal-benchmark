"""Conformal prediction benchmark — first version (RF backbone).

Novelty thrust (项目调研汇总.md v2 §7.5.1). For each regression dataset:
  train RF on train, calibrate on val, evaluate intervals on TEST (one-time
  final eval, §16.4). Reports empirical coverage (should hit target) and mean
  interval width (tighter = better at matched coverage), for both standard
  split conformal and normalized (RF tree-std) conformal.

This validates the conformal pipeline + gives the first real interval numbers.
Extending to MolFormer/Uni-Mol/ChemFM needs their cal+test predictions
(save_test_preds sweep) -- this script is structured to plug those in later.

Run: python scripts/run_conformal.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.featurizers import batch_morgan  # noqa: E402
from src.data.loaders import load_dataset  # noqa: E402
from src.eval.conformal import (  # noqa: E402
    coverage,
    evaluate_conformal,
    mean_width,
    normalized_conformal,
)
from src.utils.logging_setup import configure_logging  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402

logger = logging.getLogger(__name__)
REG = ["esol", "freesolv", "lipophilicity", "bace", "gsht"]
ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def _rf_predict_with_std(model, X):
    """Mean prediction + std across trees (per-input uncertainty for sigma)."""
    per_tree = np.stack([t.predict(X) for t in model.estimators_], axis=0)
    return per_tree.mean(0), per_tree.std(0)


def run_dataset(name: str, seed: int = 42) -> dict:
    set_all_seeds(seed)
    split = load_dataset(name)
    Xtr, itr = batch_morgan(split.train["smiles"].tolist())
    Xva, iva = batch_morgan(split.val["smiles"].tolist())
    Xte, ite = batch_morgan(split.test["smiles"].tolist())
    ytr = split.train["label"].to_numpy()[itr]
    yva = split.val["label"].to_numpy()[iva]
    yte = split.test["label"].to_numpy()[ite]

    model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=seed)
    model.fit(Xtr, ytr)
    cal_pred, cal_sigma = _rf_predict_with_std(model, Xva)
    test_pred, test_sigma = _rf_predict_with_std(model, Xte)

    # Standard split conformal at target 90%.
    std90 = evaluate_conformal(yva, cal_pred, yte, test_pred, alpha=0.1)
    # Normalized (locally-adaptive) conformal at 90%.
    lo, hi, _ = normalized_conformal(yva, cal_pred, cal_sigma, test_pred, test_sigma, alpha=0.1)
    norm_cov, norm_w = coverage(yte, lo, hi), mean_width(lo, hi)

    # Coverage calibration curve across alphas (standard split).
    curve = [evaluate_conformal(yva, cal_pred, yte, test_pred, alpha=a) for a in ALPHAS]

    logger.info(
        f"{name:14s} split: cov={std90['empirical_coverage']:.3f} "
        f"width={std90['mean_width']:.3f} | norm: cov={norm_cov:.3f} width={norm_w:.3f}"
    )
    return {
        "dataset": name, "model": "rf",
        "split_coverage": std90["empirical_coverage"], "split_width": std90["mean_width"],
        "norm_coverage": norm_cov, "norm_width": norm_w,
        "n_test": len(yte), "curve": curve,
    }


def main() -> int:
    configure_logging("INFO")
    rows = [run_dataset(d) for d in REG]

    # Save flat results.
    flat = pd.DataFrame([{k: r[k] for k in r if k != "curve"} for r in rows])
    Path("results").mkdir(exist_ok=True)
    flat.to_csv("results/conformal_rf.csv", index=False)

    # Figure: coverage calibration (target vs empirical) per dataset.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    targets = [1 - a for a in ALPHAS]
    ax1.plot([0.7, 0.95], [0.7, 0.95], "k--", label="ideal")
    for r in rows:
        emp = [c["empirical_coverage"] for c in r["curve"]]
        ax1.plot(targets, emp, marker="o", label=r["dataset"])
    ax1.set_xlabel("Target coverage (1-alpha)")
    ax1.set_ylabel("Empirical coverage (test)")
    ax1.set_title("Conformal coverage calibration (RF)")
    ax1.legend(fontsize=8)

    # Width @ 90% per dataset (split vs normalized).
    x = np.arange(len(rows))
    ax2.bar(x - 0.2, [r["split_width"] for r in rows], 0.4, label="split")
    ax2.bar(x + 0.2, [r["norm_width"] for r in rows], 0.4, label="normalized")
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["dataset"] for r in rows], rotation=15)
    ax2.set_ylabel("Mean interval width @ 90%")
    ax2.set_title("Interval width: split vs normalized (RF)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/figures/conformal_rf.png", dpi=150)

    print("\n" + "=" * 64)
    print("CONFORMAL PREDICTION (RF, target 90% coverage)")
    print("=" * 64)
    print(f"{'dataset':<14} {'split_cov':>10} {'split_w':>9} {'norm_cov':>9} {'norm_w':>8}")
    for r in rows:
        print(f"{r['dataset']:<14} {r['split_coverage']:>10.3f} {r['split_width']:>9.3f} "
              f"{r['norm_coverage']:>9.3f} {r['norm_width']:>8.3f}")
    print("\n(coverage should ~0.90; narrower width = more useful. "
          "normalized adapts width to local uncertainty)")
    print("saved results/conformal_rf.csv + results/figures/conformal_rf.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
