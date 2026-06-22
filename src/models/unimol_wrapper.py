"""Uni-Mol Tools wrapper for 3D molecular property prediction.

Per CLAUDE.md §4 / 项目调研汇总.md §4.1: Uni-Mol provides SE(3)-Transformer
on 3D conformers, especially useful for GSHt (covalent reactivity is a
geometric problem).

Per CLAUDE.md §8.8: Uni-Mol must run fp32, NOT fp16.

KNOWN LIMITATION: unimol-tools MolTrain has no public `seed` parameter; runs
are essentially deterministic regardless of our `cfg.seed`. Treat 3-seed
Uni-Mol results as n=1 effective measurement and document accordingly in
paper. Future work: patch unimol_tools.utils.set_seed if needed.

Pipeline:
  1. Pre-canonicalize SMILES via RDKit (drop molecules unimol can't parse)
  2. Write CSV in unimol-tools format (SMILES, TARGET)
  3. MolTrain.fit() (handles 3D conformer generation internally)
  4. MolPredict on val
  5. Compute OUR metrics (consistent with other backbones)
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from src.data.loaders import Split
from src.utils.metrics import metrics_for_task
from src.utils.rdkit_compat import patch_rdkit_pandas_compat

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

# rdkit PandasTools (pulled in by unimol_tools) is broken on pandas 2.2 +
# setuptools>=81; patch before any inline `import unimol_tools` below. See §11.
patch_rdkit_pandas_compat()


def _unimol_accepts(smi: str) -> bool:
    """True iff a SMILES passes BOTH RDKit canonicalization AND unimol-tools
    own check. Using their checker upfront guarantees same molecule count
    between csv-in and predictions-out (prevents `Prediction count mismatch`).
    """
    if not isinstance(smi, str) or not smi:
        return False
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return False
    # Use unimol's own validator; import inline because unimol_tools is heavy.
    try:
        from unimol_tools.data.datareader import MolDataReader  # noqa: PLC0415
        # check_smiles is an instance method; create a throwaway reader.
        reader = MolDataReader()
        try:
            reader.check_smiles(smi, is_train=False, smi_strict=True)
            return True
        except Exception:
            return False
    except ImportError:
        # If we can't import (test env), fall back to RDKit-only check.
        return True


def _canonicalize(smi: str) -> str | None:
    """Canonical SMILES via RDKit; None for unparseable."""
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _prepare_unimol_csv(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Write unimol-format CSV with strict pre-filtering.

    Trigger: unimol may silently drop SMILES at predict time, causing
             prediction-count mismatch with our y_true.
    Why:     filter aggressively *before* fit/predict so input row count ==
             output row count.
    Outcome: returns the cleaned DataFrame (with `label` column intact),
             so caller can use it as the source of truth for y_true.
    """
    df2 = df[["smiles", "label"]].copy()
    df2["smiles"] = df2["smiles"].apply(_canonicalize)
    n_before = len(df2)
    df2 = df2.dropna(subset=["smiles"]).reset_index(drop=True)
    # Also drop anything unimol's own validator rejects.
    mask = df2["smiles"].apply(_unimol_accepts)
    df2 = df2[mask].reset_index(drop=True)
    n_dropped = n_before - len(df2)
    if n_dropped:
        logger.warning(
            f"Dropped {n_dropped}/{n_before} SMILES "
            f"(failed RDKit or unimol validation) for {out_path.name}"
        )
    df2.rename(columns={"smiles": "SMILES", "label": "TARGET"}).to_csv(out_path, index=False)
    return df2


def train_unimol(
    cfg,
    split: Split,
    run_dir: Path,
) -> dict[str, float]:
    """Train Uni-Mol on a split, return val metrics."""
    from unimol_tools import MolPredict, MolTrain  # noqa: PLC0415

    save_dir = run_dir / "unimol_save"
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Prepare CSVs in unimol format ----
    train_csv = run_dir / "unimol_train.csv"
    val_csv = run_dir / "unimol_val.csv"
    _ = _prepare_unimol_csv(split.train, train_csv)
    val_clean = _prepare_unimol_csv(split.val, val_csv)  # source of truth for y_true
    logger.info(
        f"Wrote unimol CSVs: train={len(split.train)} (raw), "
        f"val={len(split.val)} -> {len(val_clean)} after strict filter"
    )

    # ---- Train ----
    # MolTrain runs its own CV internally; we'll just use train+val merged for
    # CV and rely on early stopping. For our purposes, we extract predictions
    # on the held-out val explicitly via MolPredict afterwards.
    # Note: unimol-tools data_type='molecule' assumes inference of 3D conformer.
    # unimol-tools metric names (verified against utils/metrics.py METRICS_REGISTER):
    #   regression     -> mae, mse, pearsonr, spearmanr, r2  (NB: no "rmse")
    #   classification -> auc, auroc, auprc, log_loss, acc, f1_score, mcc
    # Pick task-appropriate default; user can override via cfg.model.unimol_metric.
    unimol_metric = cfg.model.get("unimol_metric", None)
    if unimol_metric in (None, "default"):
        unimol_metric = "mse" if split.task == "regression" else "auc"

    train_kwargs = dict(
        task=split.task,
        data_type=cfg.model.data_type,
        epochs=cfg.model.epochs,
        batch_size=cfg.model.batch_size,
        learning_rate=cfg.model.lr,
        early_stopping=cfg.model.early_stopping,
        save_path=str(save_dir),
        kfold=1,  # disable CV; we use predetermined splits
        split="random",
        split_group_col=None,
        metrics=unimol_metric,
    )

    # Some unimol versions ignore `kfold=1`; pass remove_hs etc. only if available.
    logger.info(f"MolTrain kwargs: {train_kwargs}")
    clf = MolTrain(**train_kwargs)
    clf.fit(data=str(train_csv))
    logger.info("MolTrain.fit() finished")

    # ---- Predict on val ----
    predictor = MolPredict(load_model=str(save_dir))
    preds = predictor.predict(data=str(val_csv))
    preds = np.asarray(preds).flatten()
    if preds.shape[0] != len(val_clean):
        raise RuntimeError(
            f"Prediction count mismatch: got {preds.shape[0]}, expected {len(val_clean)}. "
            "Unimol dropped SMILES even after our strict pre-filter — investigate."
        )
    y_true = val_clean["label"].to_numpy()

    # Per-molecule val predictions (val_clean is the strictly-filtered val set).
    pd.DataFrame({
        "smiles": val_clean["smiles"].to_numpy(),
        "y_true": y_true,
        "y_score": preds,
    }).to_csv(run_dir / "val_predictions.csv", index=False)

    # §16.4 one-time test predictions for conformal/cliff studies.
    if cfg.get("save_test_preds", False) and len(split.test) > 0:
        test_csv = run_dir / "unimol_test.csv"
        test_clean = _prepare_unimol_csv(split.test, test_csv)
        test_preds = np.asarray(predictor.predict(data=str(test_csv))).flatten()
        if test_preds.shape[0] == len(test_clean):
            pd.DataFrame({
                "smiles": test_clean["smiles"].to_numpy(),
                "y_true": test_clean["label"].to_numpy(),
                "y_score": test_preds,
            }).to_csv(run_dir / "test_predictions.csv", index=False)

    return metrics_for_task(split.task, y_true, preds)
