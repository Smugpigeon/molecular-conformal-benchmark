"""Fair retest: constrained relative poses for PBCNet2. For each activity-cliff pair, re-embed
lig2 with its shared scaffold (MCS with lig1) constrained to lig1's docked coordinates -> a
CONSISTENT relative pose (what a pairwise binding model needs), instead of two independent
cross-dock poses. Then PBCNet2 on these. Compares against the unconstrained run + ligand models.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np, pandas as pd, torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdFMCS
RDLogger.DisableLog("rdApp.*")
CODE=os.path.abspath("PBCNet2.0"); DATA=f"{CODE}/bace_data"; CDIR=f"{DATA}/constrained"
os.makedirs(CDIR, exist_ok=True)
sys.path.insert(0, CODE); sys.path.insert(0, f"{CODE}/model_code")
from Graph2pickle import graph_save  # noqa: E402
from Dataloader.dataloader import LeadOptDataset, collate_fn  # noqa: E402
from predict.predict import predict  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
SMOKE="--smoke" in sys.argv
POCKET=f"{DATA}/pocket.pdb"

def aligned_pose(ref_sdf, smiles2):
    ref=Chem.MolFromMolFile(ref_sdf)
    if ref is None: return None
    ref=Chem.RemoveHs(ref)
    q=Chem.MolFromSmiles(smiles2)
    if q is None: return None
    mcs=rdFMCS.FindMCS([ref,q],timeout=10,ringMatchesRingOnly=True,completeRingsOnly=True,
                       atomCompare=rdFMCS.AtomCompare.CompareElements)
    if mcs.numAtoms<6: return None
    patt=Chem.MolFromSmarts(mcs.smartsString)
    rm=ref.GetSubstructMatch(patt); qm=q.GetSubstructMatch(patt)
    if len(rm)<6 or len(qm)!=len(rm): return None
    conf=ref.GetConformer()
    cmap={qm[i]:conf.GetAtomPosition(rm[i]) for i in range(len(qm))}
    qh=Chem.AddHs(q)
    if AllChem.EmbedMolecule(qh,coordMap=cmap,randomSeed=42,useRandomCoords=False)!=0:
        return None
    try: AllChem.MMFFOptimizeMolecule(qh)
    except Exception: pass
    return qh

def main():
    P=pd.read_csv("results/docking/pbcnet_pairs.csv"); P=P[P.is_cliff].reset_index(drop=True)
    if SMOKE: P=P.head(3)
    print(f"cliff pairs: {len(P)}")
    rows=[]; n_ok=0
    for _,r in P.iterrows():
        i1,i2=int(r.idx1),int(r.idx2)
        ref_sdf=f"{DATA}/{i1}.sdf"
        if not os.path.exists(ref_sdf): continue
        q=aligned_pose(ref_sdf, r.smiles2)
        if q is None: continue
        qsdf=f"{CDIR}/{i1}_{i2}_lig2.sdf"
        w=Chem.SDWriter(qsdf); w.write(q); w.close()
        pk=f"{CDIR}/{i1}_{i2}_lig2.pkl"
        try: graph_save(qsdf, POCKET, pk)
        except Exception as e:
            print(f"  graph fail {i1}_{i2}: {type(e).__name__}"); continue
        if not os.path.exists(f"{DATA}/{i1}.pkl"): continue
        rows.append([f"{i1}.pkl", f"{i1}_{i2}_lig2.pkl", r.dpic50, r.dpic50, r.dpic50,
                     f"{DATA}/{i1}.pkl", pk, r.tanimoto, r.dpic50]); n_ok+=1
    print(f"constrained pairs built: {n_ok}")
    if SMOKE: print("smoke ok"); return
    pc=pd.DataFrame(rows,columns=["lig1","lig2","Label","Label1","Label2","dir_1","dir_2","tanimoto","dpic50"])
    pcf="results/docking/pbcnet_constrained_input.csv"; pc[["lig1","lig2","Label","Label1","Label2","dir_1","dir_2"]].to_csv(pcf,index=False)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    model=torch.load(f"{CODE}/PBCNet2.pth",map_location=torch.device(dev),weights_only=False); model.to(dev)
    dl=DataLoader(LeadOptDataset(pcf),collate_fn=collate_fn,batch_size=16,shuffle=False,drop_last=False)
    out=predict(model,dl,dev); pre=np.asarray(out[4],float)
    pc=pc.iloc[:len(pre)].copy(); pc["pre"]=pre
    pc.to_csv("results/docking/pbcnet_pred_constrained.csv",index=False)
    from scipy.stats import spearmanr
    print(f"CONSTRAINED PBCNet2: n={len(pc)} |Spearman(dpIC50,pre)|={abs(spearmanr(pc.dpic50,pc.pre)[0]):.3f}")
if __name__=="__main__": main()
