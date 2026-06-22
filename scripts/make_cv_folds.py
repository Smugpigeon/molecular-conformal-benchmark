"""B: freeze the 50 PRIMARY scaffold-grouped CV folds (byte-identical to
perfold_significance.py) as per-fold CSVs, so the deep models (MolFormer / ChemFM /
Chemprop) can be CV-trained on the SAME folds and join the per-fold omnibus (k=8).

Each fold CSV (data/processed/cv/primary_r{R}_f{F}.csv) has columns smiles,label,set:
  * fold-test molecules        -> set=test     (identical to the CPU per-fold test block)
  * 90% of fold-train          -> set=train
  * 10% of fold-train (seeded) -> set=validation  (deep early stopping; HistGB carves its
                                                   own internally, RF/Ridge/SVR use full train)

Run (activate the `drug` env first): python scripts/make_cv_folds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.perfold_significance import (  # noqa: E402
    DATA, N_SPLITS_PRIMARY, PRIMARY_REPEAT_SEEDS, murcko_groups, primary_splits,
)

OUT = Path("data/processed/cv")
OUT.mkdir(parents=True, exist_ok=True)
RAW = Path("results/raw")
RAW.mkdir(parents=True, exist_ok=True)
VAL_FRAC = 0.10


def main() -> None:
    full = pd.read_csv(DATA)
    dev = full[full.set.isin(["train", "validation"])].reset_index(drop=True)
    sm = dev.smiles.to_numpy(dtype=object)
    y = dev.label.to_numpy(dtype=np.float64)
    groups = murcko_groups(sm)

    manifest, n = [], 0
    for rep_id, fold_id, tr, te in primary_splits(groups, N_SPLITS_PRIMARY, PRIMARY_REPEAT_SEEDS):
        # carve a seeded 10% validation slice out of the fold-train partition
        rng = np.random.default_rng(1000 * rep_id + fold_id)
        perm = rng.permutation(len(tr))
        nval = max(1, int(round(VAL_FRAC * len(tr))))
        val_idx, tr_core = tr[perm[:nval]], tr[perm[nval:]]

        rows = [(sm[i], y[i], "train") for i in tr_core]
        rows += [(sm[i], y[i], "validation") for i in val_idx]
        rows += [(sm[i], y[i], "test") for i in te]
        df = pd.DataFrame(rows, columns=["smiles", "label", "set"])
        df.to_csv(OUT / f"primary_r{rep_id}_f{fold_id}.csv", index=False)

        for i in te:
            manifest.append({"repeat": rep_id, "fold": fold_id, "smiles": sm[i], "role": "test"})
        n += 1
        print(f"  fold r{rep_id} f{fold_id}: train {len(tr_core)} / val {len(val_idx)} / test {len(te)}")

    pd.DataFrame(manifest).to_csv(RAW / "fold_manifest_primary.csv", index=False)
    print(f"wrote {n} fold CSVs to {OUT}/ + manifest to {RAW}/fold_manifest_primary.csv")


if __name__ == "__main__":
    main()
