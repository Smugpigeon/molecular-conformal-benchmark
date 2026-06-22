"""Phase B: run PBCNet2 on BACE same-series/cliff pairs. Reuses Graph2pickle.graph_save +
model_code predict. Steps: poses(pdbqt)->SDF done in shell; here = union pocket + graph_save +
build predict.csv from pbcnet_pairs.csv + model predict. Outputs results/docking/pbcnet_pred.csv.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from scipy.spatial import distance_matrix
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

CODE = os.path.abspath("PBCNet2.0")
DATA = f"{CODE}/bace_data"
PROT = f"{DATA}/protein.pdb"
sys.path.insert(0, CODE); sys.path.insert(0, f"{CODE}/model_code")
from Graph2pickle import graph_save  # noqa: E402
from Dataloader.dataloader import LeadOptDataset, collate_fn  # noqa: E402
from predict.predict import predict  # noqa: E402

SMOKE = "--smoke" in sys.argv


def pocket_extract(sdf_paths, protein, out):
    mols = [Chem.MolFromMolFile(s) for s in sdf_paths]
    mols = [m for m in mols if m is not None]
    ligpos = np.concatenate([m.GetConformer().GetPositions() for m in mols])
    structure = PDBParser(QUIET=True).get_structure("p", protein)
    class Sel(Select):
        def accept_residue(self, res):
            rp = np.array([list(a.get_vector()) for a in res.get_atoms() if "H" not in a.get_id()])
            if rp.ndim < 2 or rp.shape[0] == 0:
                return 0
            return 1 if float(np.min(distance_matrix(rp, ligpos))) < 8.0 else 0
    io = PDBIO(); io.set_structure(structure); io.save(out, Sel())


def main():
    sdfs = sorted(glob.glob(f"{DATA}/*.sdf"), key=lambda p: int(os.path.basename(p)[:-4]))
    if SMOKE:
        sdfs = sdfs[:6]
    print(f"ligand SDFs: {len(sdfs)}")
    # union pocket (8A) once
    pocket = f"{DATA}/pocket.pdb"
    if not os.path.exists(pocket) or SMOKE:
        pocket_extract(sdfs, PROT, pocket)
        print(f"pocket written ({sum(1 for _ in open(pocket))} lines)")
    # graph_save each ligand -> pkl
    n_ok = 0
    for s in sdfs:
        pk = s.replace(".sdf", ".pkl")
        if not os.path.exists(pk):
            try:
                graph_save(s, pocket, pk); n_ok += 1
            except Exception as e:
                print(f"  graph fail {os.path.basename(s)}: {type(e).__name__}")
        else:
            n_ok += 1
    print(f"graphs ready: {n_ok}/{len(sdfs)}")
    if SMOKE:
        print("smoke ok"); return

    # build predict.csv from the pair list (Label = dpIC50)
    P = pd.read_csv("results/docking/pbcnet_pairs.csv")
    have = {int(os.path.basename(s)[:-4]) for s in glob.glob(f"{DATA}/*.pkl")}
    P = P[P.idx1.isin(have) & P.idx2.isin(have)].reset_index(drop=True)
    rows = []
    for _, r in P.iterrows():
        d1, d2 = f"{DATA}/{int(r.idx1)}.pkl", f"{DATA}/{int(r.idx2)}.pkl"
        rows.append([f"{int(r.idx1)}.pkl", f"{int(r.idx2)}.pkl", r.dpic50, r.dpic50, r.dpic50, d1, d2])
    pc = pd.DataFrame(rows, columns=["lig1", "lig2", "Label", "Label1", "Label2", "dir_1", "dir_2"])
    pcf = "results/docking/pbcnet_predict_input.csv"; pc.to_csv(pcf, index=False)
    print(f"predict.csv: {len(pc)} pairs")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(f"{CODE}/PBCNet2.pth", map_location=torch.device(dev), weights_only=False)
    model.to(dev)
    ds = LeadOptDataset(pcf)
    dl = DataLoader(ds, collate_fn=collate_fn, batch_size=16, shuffle=False, drop_last=False)
    out = predict(model, dl, dev)
    pre = out[4]  # valid_prediction (mae,rmse,mae_g,rmse_g,valid_prediction,...)
    P = P.iloc[:len(pre)].copy(); P["pre"] = np.asarray(pre, float)
    P.to_csv("results/docking/pbcnet_pred.csv", index=False)
    from scipy.stats import spearmanr
    print(f"PBCNet2 done: {len(P)} pairs | Spearman(dpIC50, pre)={spearmanr(P.dpic50, P.pre)[0]:.3f}")


if __name__ == "__main__":
    main()
