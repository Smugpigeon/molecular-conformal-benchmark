"""Non-conformal UQ baselines (GAP 3): show that ensemble-variance Gaussian intervals are
MISCALIBRATED, whereas split-conformal achieves valid coverage by construction -- i.e.,
conformal ADDS value, it is not a textbook method applied for show.

Baselines (no model surgery, feasible now):
  * RF tree-ensemble Gaussian: re-train RF, take per-tree test predictions -> mean + std,
    form a 90% Gaussian interval (mean +/- 1.645*std), measure empirical coverage + width.
    Run on all 6 regression datasets (CPU).
  * Deep-ensemble Gaussian (BACE only, where multi-seed preds exist): aggregate the per-seed
    deep-model test predictions -> mean + std -> 90% Gaussian interval -> coverage + width.

Compared side-by-side with the split-conformal coverage/width from results/final/conformal_full.csv.
(MC-Dropout + multi-dataset deep ensembles need GPU forward passes and are queued separately.)

Run on the server (drug env): python scripts/run_uq_baselines.py
Output: results/final/uq_baselines.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loaders import load_dataset  # noqa: E402

Z90 = 1.6448536269514722  # 90% two-sided Gaussian
REG = ["esol", "freesolv", "lipophilicity", "bace", "qm7", "qm8"]
FINAL = Path("results/final"); FINAL.mkdir(parents=True, exist_ok=True)
BC = Path("results/bace_clean_final")
DEEP_SEEDS = [42, 1337, 2024, 7, 2025]


def _morgan(smiles: list[str]) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator as rfg
    RDLogger.DisableLog("rdApp.*")
    gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
    return np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float64)


def _gauss(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> tuple[float, float]:
    half = Z90 * std
    cov = float((np.abs(y_true - mean) <= half).mean())
    width = float(2 * half.mean())
    return cov, width


def rf_ensemble_uq() -> list[dict]:
    from sklearn.ensemble import RandomForestRegressor
    rows = []
    for d in REG:
        try:
            sp = load_dataset(d)
        except Exception as e:
            print(f"  [skip rf {d}] {e}"); continue
        Xtr, ytr = _morgan(sp.train.smiles.tolist()), sp.train.label.to_numpy(np.float64)
        Xte, yte = _morgan(sp.test.smiles.tolist()), sp.test.label.to_numpy(np.float64)
        rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42).fit(Xtr, ytr)
        tree_preds = np.stack([t.predict(Xte) for t in rf.estimators_])  # (n_trees, n_test)
        cov, width = _gauss(yte, tree_preds.mean(0), tree_preds.std(0))
        rows.append({"dataset": d, "model": "rf", "uq_method": "tree-ensemble-Gaussian",
                     "coverage": round(cov, 3), "width": round(width, 3)})
        print(f"  RF {d:14s} gaussian cov={cov:.3f} width={width:.3f}")
    return rows


def deep_ensemble_uq_bace() -> list[dict]:
    rows = []
    for m in ["molformer", "chemfm", "chemprop"]:
        files = [BC / f"{m}_seed{s}_test.csv" for s in DEEP_SEEDS]
        files = [f for f in files if f.exists()]
        if len(files) < 2:
            continue
        mats = [pd.read_csv(f).sort_values("smiles").reset_index(drop=True) for f in files]
        col = "y_score" if "y_score" in mats[0].columns else "y_pred"
        preds = np.stack([mm[col].to_numpy(np.float64) for mm in mats])  # (n_seeds, n_test)
        yte = mats[0].y_true.to_numpy(np.float64)
        cov, width = _gauss(yte, preds.mean(0), preds.std(0))
        rows.append({"dataset": "bace_clean", "model": m, "uq_method": f"deep-ensemble-Gaussian({len(files)}seed)",
                     "coverage": round(cov, 3), "width": round(width, 3)})
        print(f"  deep-ens bace {m:10s} gaussian cov={cov:.3f} width={width:.3f} ({len(files)} seeds)")
    return rows


def main() -> None:
    print("[UQ baselines] ensemble-variance Gaussian intervals (target coverage 0.90)")
    rows = rf_ensemble_uq() + deep_ensemble_uq_bace()
    out = pd.DataFrame(rows)

    # attach split-conformal coverage/width for the same (dataset, model) for contrast
    cf = Path("results/final/conformal_full.csv")  # single source of truth (6 datasets)
    if cf.exists():
        c = pd.read_csv(cf)[["dataset", "model", "empirical_coverage", "mean_width"]].rename(
            columns={"empirical_coverage": "conformal_coverage", "mean_width": "conformal_width"})
        out = out.merge(c, on=["dataset", "model"], how="left")
    out.to_csv(FINAL / "uq_baselines.csv", index=False)
    print(f"\n  -> {FINAL / 'uq_baselines.csv'}")
    print("  KEY: Gaussian 'coverage' far from 0.90 = miscalibrated; conformal_coverage ~0.90 by construction.")


if __name__ == "__main__":
    main()
