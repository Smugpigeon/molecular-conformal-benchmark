"""Check predefined train/val/test splits for leakage.

Per CLAUDE.md §16.3: every dataset must be audited for:
1. Exact canonical SMILES duplicates between train ↔ test
2. Tautomer-equivalent duplicates (InChI key first 14 chars)
3. Murcko scaffold overlap
4. High Tanimoto similarity pairs (> 0.85) — soft leak

Run: `python scripts/check_split_leakage.py`
Output: results/final/split_leakage.csv (paper-track 4 datasets by default) + console summary
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from src.data.loaders import DATA_REGISTRY, load_dataset
from src.utils.logging_setup import configure_logging

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

HIGH_SIM_THRESHOLD = 0.85


def _canonicalize(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return None if mol is None else Chem.MolToSmiles(mol, canonical=True)


def _inchi_first14(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)[:14]
    except Exception:
        return None


def _scaffold(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def _morgan_fp(smi: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprint(mol)


def audit_dataset(name: str, data_root: Path) -> dict:
    """Run full leakage audit on one dataset's train ↔ test split."""
    split = load_dataset(name, data_root=data_root)
    train_smi = split.train["smiles"].tolist()
    test_smi = split.test["smiles"].tolist()

    # ---- Canonical SMILES ----
    train_canon = {s for s in (_canonicalize(x) for x in train_smi) if s}
    test_canon = [_canonicalize(x) for x in test_smi]
    canonical_dups = [s for s in test_canon if s is not None and s in train_canon]

    # ---- InChI key (tautomer-insensitive) ----
    train_inchi = {s for s in (_inchi_first14(x) for x in train_smi) if s}
    test_inchi = [_inchi_first14(x) for x in test_smi]
    inchi_dups = [k for k in test_inchi if k is not None and k in train_inchi]

    # ---- Murcko scaffold ----
    train_scaf = {s for s in (_scaffold(x) for x in train_smi) if s}
    test_scaf = [_scaffold(x) for x in test_smi]
    scaf_overlap = [s for s in test_scaf if s is not None and s in train_scaf]

    # ---- Tanimoto nearest-neighbor ----
    train_fps = [fp for fp in (_morgan_fp(x) for x in train_smi) if fp is not None]
    test_fps = [fp for fp in (_morgan_fp(x) for x in test_smi) if fp is not None]
    max_sims: list[float] = []
    for tfp in test_fps:
        sims = DataStructs.BulkTanimotoSimilarity(tfp, train_fps)
        max_sims.append(max(sims) if sims else 0.0)
    max_sims_arr = np.asarray(max_sims) if max_sims else np.array([0.0])
    high_sim = int((max_sims_arr > HIGH_SIM_THRESHOLD).sum())

    return {
        "dataset": name,
        "n_train": len(train_smi),
        "n_test": len(test_smi),
        "canonical_dups": len(canonical_dups),
        "canonical_dups_pct": 100 * len(canonical_dups) / max(1, len(test_smi)),
        "inchi_tautomer_dups": len(inchi_dups),
        "inchi_tautomer_dups_pct": 100 * len(inchi_dups) / max(1, len(test_smi)),
        "scaffold_overlap": len(scaf_overlap),
        "scaffold_overlap_pct": 100 * len(scaf_overlap) / max(1, len(test_smi)),
        "high_tanimoto_count": high_sim,
        "high_tanimoto_pct": 100 * high_sim / max(1, len(test_smi)),
        "mean_nearest_tanimoto": float(max_sims_arr.mean()),
        "max_nearest_tanimoto": float(max_sims_arr.max()),
    }


def main() -> int:
    configure_logging("INFO")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    # Default = paper-track: the four predefined-split regression datasets, written to results/final/
    # (the SI source of truth). The full audit incl. bbbp/gsht/_cls is course-only; pass --datasets
    # and a separate --out for it.
    parser.add_argument("--out", type=Path, default=Path("results/final/split_leakage.csv"))
    parser.add_argument("--datasets", nargs="+",
                        default=["esol", "freesolv", "lipophilicity", "bace"])
    args = parser.parse_args()

    rows = []
    for name in args.datasets:
        logger.info(f"Auditing {name} ...")
        rows.append(audit_dataset(name, args.data_root))

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Console summary
    logger.info("=" * 70)
    logger.info("SPLIT LEAKAGE AUDIT (per CLAUDE.md §16.3)")
    logger.info("=" * 70)
    fmt = "{:<14} {:>5} {:>8} {:>8} {:>8} {:>10}"
    logger.info(fmt.format(
        "dataset", "test", "canDup", "tauDup", "scaf%", "Tan>0.85"
    ))
    for r in rows:
        logger.info(fmt.format(
            r["dataset"], r["n_test"],
            f'{r["canonical_dups"]}',
            f'{r["inchi_tautomer_dups"]}',
            f'{r["scaffold_overlap_pct"]:.1f}%',
            f'{r["high_tanimoto_pct"]:.1f}%',
        ))

    # Flag any hard leakage
    hard_leak = [r for r in rows if r["canonical_dups"] > 0 or r["inchi_tautomer_dups"] > 0]
    if hard_leak:
        logger.warning(
            f"\nHARD LEAKAGE in {len(hard_leak)} dataset(s) — "
            "predefined split has duplicate molecules. Paper MUST disclose."
        )
        for r in hard_leak:
            logger.warning(
                f"  {r['dataset']}: {r['canonical_dups']} canonical, "
                f"{r['inchi_tautomer_dups']} tautomer dups"
            )
    else:
        logger.info("\nNo hard leakage (canonical / tautomer duplicates).")

    logger.info(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
