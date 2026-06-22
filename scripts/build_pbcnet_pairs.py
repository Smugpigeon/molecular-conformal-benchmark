"""Select BACE same-series ligand pairs (both docked) for PBCNet2: Tanimoto>=0.5 = lead-opt-like;
label cliff if also |dpIC50|>=1.5. Output pairs + the unique docked-ligand indices needed."""
from __future__ import annotations
import sys; from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog("rdApp.*")
d = pd.read_csv("results/docking/dock_scores.csv")
d = d[d.status=="ok"].reset_index(drop=True)
gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in d.smiles]
rows=[]
for i in range(len(d)):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
    for off,t in enumerate(sims):
        j=i+1+off
        if t>=0.5:
            dp=float(d.y_true[i]-d.y_true[j])
            rows.append((int(d.idx[i]),int(d.idx[j]),d.smiles[i],d.smiles[j],round(t,3),round(dp,3),
                         bool(t>=0.5 and abs(dp)>=1.5)))
P=pd.DataFrame(rows,columns=["idx1","idx2","smiles1","smiles2","tanimoto","dpic50","is_cliff"])
P.to_csv("results/docking/pbcnet_pairs.csv",index=False)
need=sorted(set(P.idx1)|set(P.idx2))
Path("results/docking/needed_idx.txt").write_text(",".join(map(str,need)))
print(f"pairs={len(P)} cliffs={int(P.is_cliff.sum())} series_ctrl={int((~P.is_cliff).sum())} unique_ligands={len(need)}")
print(P.is_cliff.value_counts().to_dict())
