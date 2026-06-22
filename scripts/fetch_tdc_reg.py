"""Fetch 3 public ADME regression datasets from TDC to expand the conformal
benchmark from 5 -> 8 datasets (addresses the 'too few datasets' critique).

Drug-like ADME regression, scaffold-split, saved to data/external_reg/<name>.csv
in this project's format (smiles, label, set). These complement the course's
5 regression tasks (solubility/hydration/lipophilicity/activity/reactivity)
with permeability / protein-binding / distribution.

Run: python scripts/fetch_tdc_reg.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)

# alias -> TDC name (all single-pred ADME regression)
DATASETS = {
    "caco2": "Caco2_Wang",      # permeability (~910)
    "ppbr": "PPBR_AZ",          # plasma protein binding rate (~1797)
    "vdss": "VDss_Lombardo",    # volume of distribution (~1130)
}


def main() -> int:
    configure_logging("INFO")
    from tdc.single_pred import ADME  # noqa: PLC0415

    out_dir = Path("data/external_reg")
    out_dir.mkdir(parents=True, exist_ok=True)
    for alias, tdc_name in DATASETS.items():
        d = ADME(name=tdc_name)
        sp = d.get_split(method="scaffold")  # dict: train / valid / test
        frames = []
        for tdc_key, set_name in [("train", "train"), ("valid", "validation"), ("test", "test")]:
            df = sp[tdc_key][["Drug", "Y"]].rename(columns={"Drug": "smiles", "Y": "label"})
            df["set"] = set_name
            frames.append(df)
        full = pd.concat(frames, ignore_index=True)
        out = out_dir / f"{alias}.csv"
        full.to_csv(out, index=False)
        logger.info(
            f"{alias} ({tdc_name}): n={len(full)} "
            f"(train {len(sp['train'])}, val {len(sp['valid'])}, test {len(sp['test'])}) "
            f"label [{full['label'].min():.2f}, {full['label'].max():.2f}] -> {out}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
