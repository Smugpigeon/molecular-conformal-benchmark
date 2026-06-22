"""Prepare QM7 + QM8 (Tier-2 MoleculeNet regression) for the multi-dataset conformal
benchmark (PAPER track -- external data, outside the course red line by design).

  QM7: target u0_atom (atomization energy), single-target, ~7k molecules.
  QM8: 16 excited-state targets -> we keep E1-CC2 as the single representative so the
       conformal interval-width ranking stays single-output (MUBen reports QM8 multitask;
       a per-target sweep is a later refinement).

Canonicalize (once, RDKit) + drop parse failures + dedup canonical SMILES + random
80/10/10 split (seed 42). QM molecules are tiny (<=~8 heavy atoms) so Murcko scaffold
split degenerates; random is MoleculeNet's standard for QM and gives the EXCHANGEABLE
regime where split-conformal's marginal guarantee is expected to hold cleanly (low
label noise) -- the clean coverage-validity reference vs the scaffold-split drug datasets.

Source: MoleculeNet via deepchemdata S3 (/tmp/qm7.csv, /tmp/qm8.csv).
Run: /opt/anaconda3/bin/python3 scripts/prep_qm_datasets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


def prep(src: str, target: str, name: str) -> None:
    df = pd.read_csv(src)[["smiles", target]].dropna().rename(columns={target: "label"})
    seen: set[str] = set()
    can, lab, nfail = [], [], 0
    for s, l in zip(df.smiles, df.label):
        m = Chem.MolFromSmiles(s)
        if m is None:
            nfail += 1
            continue
        cs = Chem.MolToSmiles(m, canonical=True)
        if cs in seen:
            continue
        seen.add(cs)
        can.append(cs)
        lab.append(float(l))
    d = pd.DataFrame({"smiles": can, "label": np.array(lab, dtype=np.float64)})
    n = len(d)
    idx = np.random.default_rng(42).permutation(n)
    ntr, nva = int(0.8 * n), int(0.1 * n)
    sets = np.array(["train"] * n, dtype=object)
    sets[idx[ntr:ntr + nva]] = "validation"
    sets[idx[ntr + nva:]] = "test"
    d["set"] = sets
    d.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name}: {n} mols (parse-failed {nfail}); "
          f"split {(sets == 'train').sum()}/{(sets == 'validation').sum()}/{(sets == 'test').sum()}; "
          f"label {d.label.min():.3f}..{d.label.max():.3f} (target={target})")


if __name__ == "__main__":
    prep("/tmp/qm7.csv", "u0_atom", "qm7")
    prep("/tmp/qm8.csv", "E1-CC2", "qm8")
