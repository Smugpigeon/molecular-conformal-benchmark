"""Negative control for the bace_clean comparison (CLAUDE.md §16.8): shuffle the TRAIN labels,
refit, and check test Pearson collapses to ~0. If a model still predicts on shuffled labels, the
split/features leak or the model memorizes -- which would invalidate the headline numbers. Every
model shares the SAME bace_clean split, so RF + GBM + TabPFN collapsing certifies the split.

Run on the server (TabPFN needs a GPU for >1000 samples):
  CUDA_VISIBLE_DEVICES=6 python scripts/negative_control_bace.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.metrics import regression_metrics  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402

SEED = 42
warnings.filterwarnings("ignore")


def _sklearn_compat_shim() -> None:
    import sklearn.utils.validation as skv
    if not hasattr(skv, "_is_pandas_df"):
        def _is_pandas_df(x):
            try:
                import pandas as _pd
            except ImportError:
                return False
            return isinstance(x, _pd.DataFrame)
        skv._is_pandas_df = _is_pandas_df


def featurize(smiles, rep):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    RDLogger.DisableLog("rdApp.*")
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if rep == "morgan":
        gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        X = np.zeros((len(mols), 2048), dtype=np.float64)
        for i, m in enumerate(mols):
            X[i] = gen.GetFingerprint(m)
        return X
    names = [n for n, _ in Descriptors.descList]
    X = np.full((len(mols), len(names)), np.nan, dtype=np.float64)
    for i, m in enumerate(mols):
        for j, (_, fn) in enumerate(Descriptors.descList):
            try:
                X[i, j] = fn(m)
            except Exception:
                X[i, j] = np.nan
    X[~np.isfinite(X)] = np.nan
    return X


def main():
    _sklearn_compat_shim()
    import torch
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from tabpfn import TabPFNRegressor
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_all_seeds(SEED)
    df = pd.read_csv("data/processed/bace_clean.csv")
    sp = {k: df[df.set == k] for k in ["train", "test"]}
    ytr = sp["train"].label.to_numpy(float); yte = sp["test"].label.to_numpy(float)
    rng = np.random.default_rng(SEED)
    ytr_shuf = ytr.copy(); rng.shuffle(ytr_shuf)

    Xm_tr = featurize(sp["train"].smiles.tolist(), "morgan")
    Xm_te = featurize(sp["test"].smiles.tolist(), "morgan")
    Xd_tr = featurize(sp["train"].smiles.tolist(), "desc")
    Xd_te = featurize(sp["test"].smiles.tolist(), "desc")

    def rf():
        return RandomForestRegressor(n_estimators=500, max_features=0.3, random_state=SEED, n_jobs=-1)

    def gbm():
        return HistGradientBoostingRegressor(random_state=SEED)

    def tab():
        return TabPFNRegressor(device=device, random_state=SEED, ignore_pretraining_limits=True)

    rows = []
    print(f"{'model':16s} {'real R':>8s} {'shuffled R':>12s}  verdict")
    for name, mk, Xtr, Xte in [("RF (morgan)", rf, Xm_tr, Xm_te),
                               ("GBM (morgan)", gbm, Xm_tr, Xm_te),
                               ("TabPFN (desc)", tab, Xd_tr, Xd_te)]:
        r_real = regression_metrics(yte, mk().fit(Xtr, ytr).predict(Xte))["PearsonR"]
        r_shuf = regression_metrics(yte, mk().fit(Xtr, ytr_shuf).predict(Xte))["PearsonR"]
        ok = abs(r_shuf) < 0.15
        print(f"{name:16s} {r_real:8.3f} {r_shuf:12.3f}  {'PASS' if ok else 'FAIL leakage!'}")
        rows.append({"model": name, "real_R": round(r_real, 4),
                     "shuffled_R": round(r_shuf, 4), "pass": bool(ok)})
    out = Path("results/final/negative_control.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
