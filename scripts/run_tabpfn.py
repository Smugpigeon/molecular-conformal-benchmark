"""TabPFN (v2) on BACE -- the zero-tuning tabular foundation model (Nature 2025,
s41586-024-08328-6), the modern alternative to Optuna/grid search for small molecular
regression. Multiple 2025-2026 cheminformatics papers report TabPFN's clearest gains exactly
here: small-sample REGRESSION. We run it with NO hyperparameter tuning (single forward pass)
on the SAME bace_clean split as the Optuna-tuned classic models, so the comparison is fair.

Two representations within TabPFN v2's practical feature budget (~500):
  * desc      -- the full RDKit descriptor set (~200 features; TabPFN ingests NaNs natively)
  * morgan512 -- ECFP/Morgan r2 folded to 512 bits (the folded-fingerprint setup in the papers)

Run: /opt/anaconda3/bin/python3 scripts/run_tabpfn.py
"""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")  # tabpfn 2.0.x + recent sklearn/scipy need this set pre-import

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.metrics import regression_metrics  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402

SEED = 42
DATA_CSV = os.environ.get("BACE_DATA", "data/processed/bace_clean.csv")
OUT = Path(os.environ.get("TUNE_OUT", "results/tuning")); OUT.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


def _sklearn_compat_shim() -> None:
    """Open tabpfn 2.0.x predates sklearn 1.8, which dropped the private helper
    `_is_pandas_df`. Re-provide it so the token-free build imports on a newer sklearn,
    instead of forcing the telemetry-gated tabpfn >=2.2 or downgrading a shared env."""
    import sklearn.utils.validation as skv
    if not hasattr(skv, "_is_pandas_df"):
        def _is_pandas_df(x):
            try:
                import pandas as _pd
            except ImportError:
                return False
            return isinstance(x, _pd.DataFrame)
        skv._is_pandas_df = _is_pandas_df


def featurize(smiles: list[str], rep: str) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors
    RDLogger.DisableLog("rdApp.*")
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if rep == "morgan512":
        gen = AllChem.GetMorganGenerator(radius=2, fpSize=512)
        X = np.zeros((len(mols), 512), dtype=np.float64)
        for i, m in enumerate(mols):
            X[i] = gen.GetFingerprint(m)
        return X
    if rep == "desc":
        # Trigger: rep == "desc"  Why: TabPFN papers feed the full RDKit descriptor block
        # Outcome: ~200-D continuous matrix; inf -> nan (TabPFN handles missing values natively)
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
    raise ValueError(rep)


def main() -> None:
    _sklearn_compat_shim()
    import torch
    from tabpfn import TabPFNRegressor

    # TabPFN refuses CPU with >1000 samples (slow); use GPU when present (server),
    # else CPU. ignore_pretraining_limits lets morgan512 (512>500 features) through.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_all_seeds(SEED)
    df = pd.read_csv(DATA_CSV)
    sp = {k: df[df.set == k] for k in ["train", "validation", "test"]}
    ytr = sp["train"].label.to_numpy(float)
    yva = sp["validation"].label.to_numpy(float)
    yte = sp["test"].label.to_numpy(float)

    results = []
    for rep in ["desc", "morgan512"]:
        Xtr = featurize(sp["train"].smiles.tolist(), rep)
        Xva = featurize(sp["validation"].smiles.tolist(), rep)
        Xte = featurize(sp["test"].smiles.tolist(), rep)
        reg = TabPFNRegressor(device=device, random_state=SEED, ignore_pretraining_limits=True)
        reg.fit(Xtr, ytr)  # "fit" = store context; no gradient training, no tuning
        pv = np.asarray(reg.predict(Xva), dtype=float)
        pt = np.asarray(reg.predict(Xte), dtype=float)
        mv = regression_metrics(yva, pv); mt = regression_metrics(yte, pt)
        pd.DataFrame({"smiles": sp["validation"].smiles, "y_true": yva, "y_pred": pv}).to_csv(
            OUT / f"tabpfn_{rep}_val.csv", index=False)
        pd.DataFrame({"smiles": sp["test"].smiles, "y_true": yte, "y_pred": pt}).to_csv(
            OUT / f"tabpfn_{rep}_test.csv", index=False)
        print(f"  [tabpfn|{rep:9s}] dim={Xtr.shape[1]:4d}  "
              f"val MAE={mv['MAE']:.3f} R={mv['PearsonR']:.3f}  "
              f"test MAE={mt['MAE']:.3f} R={mt['PearsonR']:.3f}")
        results.append({"model": "tabpfn", "rep": rep, "n_trials": 0, "cv_mae": None,
                        "best_params": {"tuning": "none (in-context)"}, "val": mv, "test": mt})

    path = OUT / "summary_tabpfn.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
