"""CPU diagnostic: record Chemprop's per-epoch validation curve on BACE.

The production wrapper (src/models/chemprop_wrapper.py) builds the Lightning
Trainer with logger=False, so no per-epoch validation trajectory was ever saved
-- only the final-epoch model is used. This script CLOSES that verification gap
WITHOUT changing any production result: it replicates the exact production config
(D-MPNN depth=3 d_h=300, 50 epochs, NO early stopping, final-epoch model) but
attaches a CSVLogger so the validation-loss trajectory is recorded, then exports
a tidy per-epoch CSV for the convergence-verification figure.

Runs on CPU on purpose -- the shared GPUs are occupied, and this model is small
enough that CPU is fine. The final real-scale val RMSE is recomputed via the same
predict path as production to anchor the curve's scale and sanity-check that the
50-epoch budget lands on the plateau (not past it).

Usage: python scripts/verify_chemprop_convergence.py 42 1337 2024
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightning import pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from rdkit import Chem

from chemprop import data, featurizers, models, nn

from src.data.loaders import load_dataset
from src.utils.metrics import metrics_for_task
from src.utils.seed import set_all_seeds

OUT = Path("results/convergence")
OUT.mkdir(parents=True, exist_ok=True)


def make_dset(df, feat):
    dps, kept = [], []
    for s, y in zip(df["smiles"], df["label"]):
        if Chem.MolFromSmiles(s) is None:
            kept.append(False)
            continue
        dps.append(data.MoleculeDatapoint.from_smi(s, np.array([float(y)], dtype=float)))
        kept.append(True)
    return data.MoleculeDataset(dps, feat), np.array(kept, dtype=bool)


def run(seed: int) -> None:
    set_all_seeds(seed, deterministic=True)
    split = load_dataset("bace", data_root="data")
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    tr, _ = make_dset(split.train, feat)
    va, vmask = make_dset(split.val, feat)

    # z-scale targets on TRAIN only (leak-safe), unscale at output -- as production.
    scaler = tr.normalize_targets()
    va.normalize_targets(scaler)
    otf = nn.UnscaleTransform.from_standard_scaler(scaler)

    trl = data.build_dataloader(tr, batch_size=64, num_workers=0)
    val = data.build_dataloader(va, batch_size=64, num_workers=0, shuffle=False)

    mp = nn.BondMessagePassing(depth=3, d_h=300)
    ffn = nn.RegressionFFN(input_dim=300, hidden_dim=300, n_layers=2, output_transform=otf)
    model = models.MPNN(mp, nn.MeanAggregation(), ffn, batch_norm=True)

    logger = CSVLogger(save_dir=str(OUT), name=f"chemprop_seed{seed}")
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="cpu",
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=logger,
        deterministic=False,  # chemprop scatter ops lack a deterministic impl
        check_val_every_n_epoch=1,
    )

    t0 = time.time()
    trainer.fit(model, trl, val)
    dt = time.time() - t0

    # Final real-scale val metrics via the production predict path (anchor + sanity).
    preds = np.concatenate([p.numpy() for p in trainer.predict(model, val)]).flatten()
    yv = split.val["label"].to_numpy()[vmask]
    final = metrics_for_task("regression", yv, preds[: len(yv)])

    mcsv = Path(logger.log_dir) / "metrics.csv"
    m = pd.read_csv(mcsv)
    # Collapse the train-row / val-row interleave into one row per epoch.
    g = m.groupby("epoch").mean(numeric_only=True).reset_index()
    g.to_csv(OUT / f"chemprop_bace_seed{seed}.csv", index=False)

    print(f"seed{seed}: {dt:.0f}s | metrics.csv cols = {list(m.columns)}")
    print(f"  per-epoch rows: {len(g)} | final real val RMSE={final['RMSE']:.3f} "
          f"Pearson={final['PearsonR']:.3f} (production reports ~0.78 RMSE)")


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or [42]
    for s in seeds:
        run(s)
