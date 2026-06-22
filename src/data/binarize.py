"""Convert regression datasets to clinically-thresholded binary classification.

Per 项目调研汇总.md v2 §2 (threshold verdict table). Only ESOL / BACE / BBBP
become classification; FreeSolv / Lipophilicity / GSHt stay regression (their
binary cutoffs have no defensible clinical meaning — documented in §2).

CLAUDE.md §7.1: data/ is read-only -> outputs go to data/processed/.
CLAUDE.md §7.2: never overwrite the original `set` split — it is preserved.
CLAUDE.md §7.4-7.5: canonicalize once, log dropped molecules (no silent drops).

Thresholds (all literature-backed, see §2):
- ESOL : soluble if logS > -4      (SwissADME qualitative scale)
- BACE : active  if pIC50 >= 7      (matches DeepChem official Class boundary 6.983)
- BBBP : keep original BBB+/- label, but clean (dedup + RDKit-validate + drop label conflicts)

Run: `python -m src.data.binarize`  (or `make binarize-cls`)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from src.utils.logging_setup import configure_logging

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.warning")


def _canonicalize(smi: str) -> str | None:
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _threshold_binarize(
    csv_in: Path, csv_out: Path, threshold: float, positive_when_greater: bool
) -> dict[str, int]:
    """Binarize a regression CSV by a threshold on the `label` column.

    positive_when_greater=True  -> label_bin = 1 if value > threshold (ESOL soluble)
    positive_when_greater=False -> label_bin = 1 if value >= threshold (BACE active)
    """
    df = pd.read_csv(csv_in)
    n0 = len(df)

    # Canonicalize + drop unparseable (no silent drop, §7.5).
    df["smiles_canon"] = df["smiles"].apply(_canonicalize)
    n_failed = int(df["smiles_canon"].isna().sum())
    if n_failed:
        for s in df.loc[df["smiles_canon"].isna(), "smiles"].head(5):
            logger.warning(f"  {csv_in.name}: failed SMILES {s!r}")
    df = df.dropna(subset=["smiles_canon"]).copy()

    df["label_orig"] = df["label"].astype(float)
    if positive_when_greater:
        df["label"] = (df["label_orig"] > threshold).astype(int)
    else:
        df["label"] = (df["label_orig"] >= threshold).astype(int)
    df["smiles"] = df["smiles_canon"]
    df = df[["smiles", "label", "label_orig", "set"]]

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)

    pos = int(df["label"].sum())
    stats = {
        "n_in": n0,
        "n_out": len(df),
        "n_failed": n_failed,
        "n_pos": pos,
        "pos_pct": round(100 * pos / max(1, len(df)), 1),
    }
    logger.info(f"{csv_in.name} -> {csv_out.name}: {stats}")
    return stats


def _clean_classification(csv_in: Path, csv_out: Path) -> dict[str, int]:
    """Clean an already-binary dataset (BBBP).

    Steps (per §2 + CLAUDE.md §7.5):
      1. canonicalize, drop RDKit failures
      2. drop exact canonical duplicates (keep first occurrence -> preserves its `set`)
      3. drop canonical SMILES with CONFLICTING labels across rows (ambiguous)
    """
    df = pd.read_csv(csv_in)
    n0 = len(df)

    df["smiles_canon"] = df["smiles"].apply(_canonicalize)
    n_failed = int(df["smiles_canon"].isna().sum())
    if n_failed:
        for s in df.loc[df["smiles_canon"].isna(), "smiles"].head(5):
            logger.warning(f"  {csv_in.name}: failed SMILES {s!r}")
    df = df.dropna(subset=["smiles_canon"]).copy()

    # Identify conflicting-label canonical SMILES.
    label_nunique = df.groupby("smiles_canon")["label"].nunique()
    conflict = set(label_nunique[label_nunique > 1].index)
    n_conflict_rows = int(df["smiles_canon"].isin(conflict).sum())
    df = df[~df["smiles_canon"].isin(conflict)].copy()

    # Drop exact duplicates (keep first -> preserves original set assignment).
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["smiles_canon"], keep="first").copy()
    n_dups = n_before_dedup - len(df)

    df["smiles"] = df["smiles_canon"]
    df = df[["smiles", "label", "set"]]
    df["label"] = df["label"].astype(int)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_out, index=False)

    pos = int(df["label"].sum())
    stats = {
        "n_in": n0,
        "n_out": len(df),
        "n_failed": n_failed,
        "n_conflict_rows_dropped": n_conflict_rows,
        "n_dups_dropped": n_dups,
        "n_pos": pos,
        "pos_pct": round(100 * pos / max(1, len(df)), 1),
    }
    logger.info(f"{csv_in.name} -> {csv_out.name}: {stats}")
    return stats


def main() -> int:
    configure_logging("INFO")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-root", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    out = args.out_root
    results = {}

    # ESOL: soluble if logS > -4
    results["esol_cls"] = _threshold_binarize(
        args.data_root / "ESOL.split.csv", out / "esol_cls.csv",
        threshold=-4.0, positive_when_greater=True,
    )
    # BACE: active if pIC50 >= 7 (= official DeepChem Class boundary)
    results["bace_cls"] = _threshold_binarize(
        args.data_root / "BACE.split.csv", out / "bace_cls.csv",
        threshold=7.0, positive_when_greater=False,
    )
    # BBBP: clean the original binary labels
    results["bbbp_cls"] = _clean_classification(
        args.data_root / "BBBP.split.csv", out / "bbbp_cls.csv",
    )

    logger.info("=" * 60)
    logger.info("BINARIZATION SUMMARY (项目调研汇总.md v2 §2)")
    logger.info("=" * 60)
    for name, st in results.items():
        logger.info(f"  {name:10s} n={st['n_out']:5d}  pos={st['pos_pct']}%")
    logger.info("FreeSolv / Lipophilicity / GSHt kept as REGRESSION (no binarize).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
