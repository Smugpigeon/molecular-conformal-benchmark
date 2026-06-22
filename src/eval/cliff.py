"""Activity-cliff stratified evaluation (MoleculeACE benchmark).

Novelty thrust (项目调研汇总.md v2 §7.5.1): do molecular foundation models
(esp. 3D Uni-Mol) handle activity cliffs better than 2D models? SemiMol
(arXiv 2601.04507, 2026-01) explicitly does NOT evaluate Uni-Mol or MolFormer
on cliffs -- a 4-year gap since van Tilborg (JCIM 2022) opened it.

MoleculeACE CSVs (data/external/*.csv) columns:
  smiles, exp_mean [nM], y (standardized), cliff_mol (0/1), split, y [pXX]
We use the pActivity column "y [...]" as the regression target and stratify
test RMSE by cliff_mol.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)


def list_moleculeace(data_dir: str | Path = "data/external") -> list[str]:
    """Names of all MoleculeACE datasets (CHEMBL..._Ki etc.)."""
    return sorted(p.stem for p in Path(data_dir).glob("CHEMBL*.csv"))


def load_moleculeace(name: str, data_dir: str | Path = "data/external"):
    """Return (train_df, test_df, target_col). Target = pActivity column."""
    df = pd.read_csv(Path(data_dir) / f"{name}.csv")
    target_col = next(c for c in df.columns if c.startswith("y ["))
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return train, test, target_col


def _rmse(yt: NDArray, yp: NDArray) -> float:
    return float(np.sqrt(mean_squared_error(yt, yp))) if len(yt) else float("nan")


def cliff_stratified_rmse(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    cliff_mask: NDArray[np.bool_],
) -> dict[str, float]:
    """RMSE overall + on cliff vs non-cliff test molecules.

    The headline metric is rmse_cliff / rmse_noncliff: a ratio > 1 means the
    model is worse on activity cliffs (the known failure mode).
    """
    cliff = _rmse(y_true[cliff_mask], y_pred[cliff_mask])
    noncliff = _rmse(y_true[~cliff_mask], y_pred[~cliff_mask])
    return {
        "rmse_all": _rmse(y_true, y_pred),
        "rmse_cliff": cliff,
        "rmse_noncliff": noncliff,
        "cliff_ratio": cliff / noncliff if noncliff and not np.isnan(noncliff) else float("nan"),
        "n_cliff": int(cliff_mask.sum()),
        "n_noncliff": int((~cliff_mask).sum()),
    }
