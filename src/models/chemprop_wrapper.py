"""Chemprop v2 D-MPNN wrapper (assignment 策略2: SMILES -> 2D graph -> GNN).

Chemprop is a directed message-passing neural network operating on the 2D
molecular graph with atom (node) + bond (edge) features — exactly the
graph-network strategy the course offers as 策略2. This adds the
message-passing-GNN representation class to the backbone comparison.

Per CLAUDE.md §8: regression targets z-scaled on TRAIN only (Chemprop
convention, leak-safe); evaluation reports our own MAE/Pearson R (§9).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightning import pytorch as pl

from chemprop import data, featurizers, models, nn

from src.data.loaders import Split
from src.utils.metrics import metrics_for_task

logger = logging.getLogger(__name__)


def _make_dataset(df: pd.DataFrame, featurizer):
    """Build a MoleculeDataset, skipping RDKit-unparseable SMILES.

    Chemprop's make_mol raises (not returns None) on bad SMILES; we pre-filter
    and log the count (CLAUDE.md §7.5: no silent drops). Returns (dataset,
    kept_mask) so the caller aligns y_true/smiles to predictions.
    """
    from rdkit import Chem  # noqa: PLC0415

    dps, kept, n_fail = [], [], 0
    for s, y in zip(df["smiles"], df["label"]):
        if Chem.MolFromSmiles(s) is None:
            n_fail += 1
            kept.append(False)
            continue
        dps.append(data.MoleculeDatapoint.from_smi(s, np.array([float(y)], dtype=float)))
        kept.append(True)
    if n_fail:
        logger.warning(f"Chemprop: skipped {n_fail} RDKit-unparseable SMILES")
    return data.MoleculeDataset(dps, featurizer), np.array(kept, dtype=bool)


def train_chemprop(cfg, split: Split, run_dir) -> dict[str, float]:
    """Train Chemprop D-MPNN, return val metrics."""
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset, _ = _make_dataset(split.train, featurizer)
    val_dset, val_mask = _make_dataset(split.val, featurizer)

    # Regression: z-scale targets on train only (leak-safe), unscale at output.
    output_transform = None
    if split.task == "regression":
        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)
        output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

    nw = cfg.model.get("num_workers", 0)
    train_loader = data.build_dataloader(train_dset, batch_size=cfg.model.batch_size, num_workers=nw)
    val_loader = data.build_dataloader(val_dset, batch_size=cfg.model.batch_size, num_workers=nw, shuffle=False)

    # D-MPNN: bond message passing + mean aggregation + task head.
    mp = nn.BondMessagePassing(depth=cfg.model.depth, d_h=cfg.model.hidden_dim)
    agg = nn.MeanAggregation()
    if split.task == "regression":
        ffn = nn.RegressionFFN(
            input_dim=cfg.model.hidden_dim,
            hidden_dim=cfg.model.ffn_hidden_dim,
            n_layers=cfg.model.ffn_num_layers,
            output_transform=output_transform,
        )
    else:
        ffn = nn.BinaryClassificationFFN(
            input_dim=cfg.model.hidden_dim,
            hidden_dim=cfg.model.ffn_hidden_dim,
            n_layers=cfg.model.ffn_num_layers,
        )
    model = models.MPNN(mp, agg, ffn, batch_norm=True)

    trainer = pl.Trainer(
        max_epochs=cfg.model.epochs,
        accelerator="gpu" if cfg.model.get("use_gpu", True) else "cpu",
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        deterministic=False,  # chemprop scatter ops lack deterministic impl
    )
    logger.info(f"Training Chemprop D-MPNN ({split.task}) on {len(train_dset)} mols ...")
    trainer.fit(model, train_loader, val_loader)

    def _predict(loader, n):
        preds = trainer.predict(model, loader)
        arr = np.concatenate([p.numpy() for p in preds]).flatten()
        return arr[:n]

    y_val = split.val["label"].to_numpy()[val_mask]
    val_smiles = split.val["smiles"].to_numpy()[val_mask]
    y_pred_val = _predict(val_loader, len(val_dset))
    val_metrics = metrics_for_task(split.task, y_val, y_pred_val)
    logger.info(f"VAL ({split.name}): {val_metrics}")

    pd.DataFrame({
        "smiles": val_smiles[: len(y_pred_val)],
        "y_true": y_val, "y_score": y_pred_val,
    }).to_csv(run_dir / "val_predictions.csv", index=False)

    if cfg.get("save_test_preds", False) and len(split.test) > 0:
        test_dset, test_mask = _make_dataset(split.test, featurizer)
        test_loader = data.build_dataloader(test_dset, batch_size=cfg.model.batch_size, num_workers=nw, shuffle=False)
        y_pred_test = _predict(test_loader, len(test_dset))
        test_smiles = split.test["smiles"].to_numpy()[test_mask]
        test_y = split.test["label"].to_numpy()[test_mask]
        pd.DataFrame({
            "smiles": test_smiles[: len(y_pred_test)],
            "y_true": test_y[: len(y_pred_test)],
            "y_score": y_pred_test,
        }).to_csv(run_dir / "test_predictions.csv", index=False)

    return val_metrics
