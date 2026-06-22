"""Run TabPFN on a dataset and write predictions in the runs/ layout that
run_conformal_full.py reads (val_predictions.csv + test_predictions.csv with columns
smiles,y_true,y_score), so TabPFN joins the multi-dataset conformal sweep uniformly.

Usage (server, drug env): python scripts/run_tabpfn_to_runs.py <dataset_name>
  reads data/processed/<dataset_name>.csv (columns smiles,label,set).
"""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")  # before tabpfn import

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_tabpfn import _sklearn_compat_shim, featurize  # noqa: E402
from src.data.loaders import load_dataset  # noqa: E402


def main() -> None:
    name = sys.argv[1]
    sp = load_dataset(name)  # resolves the real CSV path + smiles/label/set columns
    _sklearn_compat_shim()
    import torch
    from tabpfn import TabPFNRegressor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr = featurize(sp.train.smiles.tolist(), "desc")
    ytr = sp.train.label.to_numpy(np.float64)
    reg = TabPFNRegressor(device=device, random_state=42, ignore_pretraining_limits=True).fit(Xtr, ytr)

    out = Path(f"runs/manual_{name}_tabpfn_seed42")
    out.mkdir(parents=True, exist_ok=True)
    for split_df, fn in [(sp.val, "val_predictions.csv"), (sp.test, "test_predictions.csv")]:
        X = featurize(split_df.smiles.tolist(), "desc")
        pred = np.asarray(reg.predict(X), dtype=np.float64)
        pd.DataFrame({"smiles": split_df.smiles, "y_true": split_df.label.to_numpy(np.float64),
                      "y_score": pred}).to_csv(out / fn, index=False)
    print(f"tabpfn {name}: wrote {out} (device={device}, n_train={len(sp.train)})")


if __name__ == "__main__":
    main()
