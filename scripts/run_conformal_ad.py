"""Applicability-domain-stratified conditional coverage (the moat).

Novelty thrust (项目调研汇总.md v2 §7.5.1 deliverable #4): split conformal
guarantees only MARGINAL coverage. We test CONDITIONAL coverage by stratifying
each test molecule by its max Tanimoto to the train set (applicability domain),
then reporting per-stratum coverage for each backbone. Hypothesis H3
(preregistered): the tightest backbone (Uni-Mol) under-covers specifically in
the low-similarity (OOD) stratum.

Reads val (cal) + test (eval) predictions from the ppi_bias eval sweep.
Run: python scripts/run_conformal_ad.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.loaders import load_dataset  # noqa: E402
from src.eval.conformal import split_conformal  # noqa: E402
from src.utils.logging_setup import configure_logging  # noqa: E402

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)
RUNS = Path("runs")
# Paper-track AD: the four predefined-split bioactivity/physicochemical datasets. GSHt is
# quarantined; QM7/QM8 use random splits where the Tanimoto applicability domain is near-degenerate.
REG = ["esol", "freesolv", "lipophilicity", "bace"]
MODELS = ["rf", "molformer", "chemfm", "unimol"]
MLAB = {"rf": "RF", "molformer": "MolFormer", "chemfm": "ChemFM", "unimol": "Uni-Mol"}
STRATA = ["low", "med", "high"]  # AD similarity terciles


def _fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(m)


def _ad_scores(test_smiles, train_fps) -> np.ndarray:
    """Max Tanimoto of each test molecule to the train set."""
    out = []
    for s in test_smiles:
        fp = _fp(s)
        if fp is None or not train_fps:
            out.append(np.nan)
        else:
            out.append(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
    return np.asarray(out)


def _latest_preds(dataset, model, seed=42):
    for d in reversed(sorted(RUNS.glob(f"*_{dataset}_{model}_seed{seed}"))):
        vp, tp = d / "val_predictions.csv", d / "test_predictions.csv"
        if vp.exists() and tp.exists():
            return pd.read_csv(vp), pd.read_csv(tp)
    return None, None


def main() -> int:
    configure_logging("INFO")
    rows = []
    for dataset in REG:
        split = load_dataset(dataset)
        train_fps = [fp for fp in (_fp(s) for s in split.train["smiles"]) if fp is not None]
        for model in MODELS:
            valdf, testdf = _latest_preds(dataset, model)
            if valdf is None:
                continue
            # Conformal quantile from calibration (val).
            lo, hi, q = split_conformal(
                valdf["y_true"].to_numpy(), valdf["y_score"].to_numpy(),
                testdf["y_score"].to_numpy(), alpha=0.1,
            )
            covered = (testdf["y_true"].to_numpy() >= lo) & (testdf["y_true"].to_numpy() <= hi)
            ad = _ad_scores(testdf["smiles"].tolist(), train_fps)
            # Tercile thresholds from this dataset's AD distribution.
            valid = ~np.isnan(ad)
            t1, t2 = np.nanquantile(ad, [1 / 3, 2 / 3])
            strat = np.full(len(ad), "high", dtype=object)
            strat[ad <= t2] = "med"
            strat[ad <= t1] = "low"
            for s in STRATA:
                mask = valid & (strat == s)
                if mask.sum() == 0:
                    continue
                rows.append({
                    "dataset": dataset, "model": model, "stratum": s,
                    "n": int(mask.sum()),
                    "coverage": float(covered[mask].mean()),
                    "mean_tanimoto": float(np.nanmean(ad[mask])),
                })

    df = pd.DataFrame(rows)
    Path("results/final").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/final/conformal_ad.csv", index=False)  # single source of truth (no GSHt)

    # Print per-model coverage by stratum (averaged over datasets).
    print("\n" + "=" * 60)
    print("AD-STRATIFIED CONDITIONAL COVERAGE (target 0.90)")
    print("=" * 60)
    print(f"{'model':<12}{'low-sim':>12}{'med-sim':>12}{'high-sim':>12}")
    for m in MODELS:
        cells = []
        for s in STRATA:
            sub = df[(df.model == m) & (df.stratum == s)]["coverage"]
            cells.append(f"{sub.mean():.3f}" if len(sub) else "--")
        print(f"{MLAB[m]:<12}{cells[0]:>12}{cells[1]:>12}{cells[2]:>12}")

    # Figure: coverage vs AD stratum per model.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = np.arange(len(STRATA))
    colors = {"rf": "#888", "molformer": "#1f77b4", "chemfm": "#d62728", "unimol": "#2ca02c"}
    for m in MODELS:
        ys = [df[(df.model == m) & (df.stratum == s)]["coverage"].mean() for s in STRATA]
        ax.plot(xs, ys, marker="o", label=MLAB[m], color=colors[m])
    ax.axhline(0.90, color="k", ls="--", lw=1, label="nominal 0.90")
    ax.set_xticks(xs); ax.set_xticklabels(["low\n(OOD)", "med", "high\n(similar)"])
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Conditional coverage by applicability domain\n(does under-coverage concentrate OOD?)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/figures/conformal_ad.png", dpi=150)
    print("saved results/final/conformal_ad.csv + results/figures/conformal_ad.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
