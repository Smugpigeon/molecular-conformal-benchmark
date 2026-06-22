"""SMILES canonicalization.

CLAUDE.md §7.4: canonicalize SMILES ONCE at preprocessing entry, write to
`data/processed/<name>_canonical.csv`. All downstream code reads canonical.
Never re-canonicalize in training loops.

Run: `python -m src.data.preprocess` (also `make canonicalize`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from src.data.loaders import DATA_REGISTRY
from src.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.warning")  # silence "WARNING" but keep error.


def canonicalize_smiles(smi: str) -> str | None:
    """Canonical SMILES via RDKit, or None if unparseable."""
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_dataset(csv_in: Path, csv_out: Path) -> dict[str, int]:
    """Canonicalize SMILES column, write to csv_out, return stats.

    CLAUDE.md §7.5: do not silently drop. Log every failed SMILES (first 5)
    and record n_failed in the returned stats dict.
    """
    df = pd.read_csv(csv_in)
    n_original = len(df)

    df["smiles_canonical"] = df["smiles"].apply(canonicalize_smiles)
    n_failed = int(df["smiles_canonical"].isna().sum())

    if n_failed > 0:
        logger.warning(f"{csv_in.name}: {n_failed}/{n_original} SMILES failed")
        failed = df.loc[df["smiles_canonical"].isna(), "smiles"].head(5).tolist()
        for s in failed:
            logger.warning(f"  Failed: {s!r}")

    df = df.dropna(subset=["smiles_canonical"]).copy()
    df["smiles"] = df["smiles_canonical"]
    df = df.drop(columns=["smiles_canonical"])

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)

    stats = {"original": n_original, "kept": len(df), "failed": n_failed}
    logger.info(f"Wrote {csv_out}: {stats}")
    return stats


def main() -> int:
    """Canonicalize all 6 datasets to data/processed/."""
    configure_logging("INFO")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-root", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    all_stats = {}
    for alias, (fname, _task, _metric) in DATA_REGISTRY.items():
        csv_in = args.data_root / fname
        if not csv_in.exists():
            logger.error(f"Missing: {csv_in}")
            continue
        csv_out = args.out_root / f"{alias}_canonical.csv"
        all_stats[alias] = canonicalize_dataset(csv_in, csv_out)

    logger.info("Summary:")
    for alias, stats in all_stats.items():
        logger.info(f"  {alias:15s} {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
