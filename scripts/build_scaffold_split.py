"""Murcko scaffold split of bace_clean (DeepChem-style: largest scaffold sets -> train, rarest ->
test, so test = novel chemistry = harder). Same columns/format as bace_clean for drop-in reuse."""
from __future__ import annotations
import pandas as pd
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
df = pd.read_csv("data/processed/bace_clean.csv").reset_index(drop=True)
scaf = defaultdict(list)
for i, s in enumerate(df.smiles):
    sc = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
    scaf[sc].append(i)
# sort scaffold groups by size desc (largest -> train), deterministic tiebreak by scaffold string
groups = sorted(scaf.values(), key=lambda g: (-len(g), df.smiles[g[0]]))
n = len(df); ntr, nva = int(0.8 * n), int(0.1 * n)
split = [""] * n; ctr = 0
for g in groups:
    bucket = "train" if ctr < ntr else ("validation" if ctr < ntr + nva else "test")
    for i in g:
        split[i] = bucket
    ctr += len(g)
df["set"] = split
df.to_csv("data/processed/bace_scaffold.csv", index=False)
print("counts:", df.set.value_counts().to_dict())
# verify no scaffold spans splits
sc_split = defaultdict(set)
for i, s in enumerate(df.smiles):
    sc_split[MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))].add(df.set[i])
overlap = sum(1 for v in sc_split.values() if len(v) > 1)
print(f"scaffolds spanning >1 split: {overlap} (must be 0)")
print(f"unique scaffolds: {len(sc_split)}")
