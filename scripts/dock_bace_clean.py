"""Parallel Vina docking of the bace_clean TEST set into 4D8C, reusing the project's validated
src/structure/dock.py (Meeko ligand prep + the May-30 receptor structure/bace/bace_receptor.pdbqt).
Each worker builds the receptor maps once, then docks its share. Records heavy-atom count so the
analysis can control for the well-known BACE size confound (Vina favours bigger ligands).

Outputs (server):
  results/docking/dock_scores.csv     idx, smiles, y_true (pIC50), vina_score, heavy, status
  results/docking/poses/<idx>.pdbqt   best pose (for the PBCNet2 stage)

Run on server (drug env):
  python scripts/dock_bace_clean.py --smoke
  python scripts/dock_bace_clean.py --workers 24
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.structure.dock import BOX_CENTER, BOX_SIZE, smiles_to_pdbqt  # noqa: E402

warnings.filterwarnings("ignore")
RECEPTOR = "structure/bace/bace_receptor.pdbqt"
OUT = Path("results/docking"); POSES = OUT / "poses"
_V = None


def _init():
    global _V
    from vina import Vina
    _V = Vina(sf_name="vina", cpu=1, seed=42, verbosity=0)  # 1 cpu/worker; parallelism is across ligands
    _V.set_receptor(RECEPTOR)
    _V.compute_vina_maps(center=list(BOX_CENTER), box_size=list(BOX_SIZE))


def _heavy(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    return int(m.GetNumHeavyAtoms()) if m else -1


def dock_one_p(arg):
    idx, smi, y = arg
    try:
        lig = smiles_to_pdbqt(smi)
        if lig is None:
            return (idx, smi, y, None, _heavy(smi), "prep_fail")
        _V.set_ligand_from_string(lig)
        _V.dock(exhaustiveness=8, n_poses=5)
        sc = float(_V.energies(n_poses=1)[0][0])
        POSES.mkdir(parents=True, exist_ok=True)
        _V.write_poses(str(POSES / f"{idx}.pdbqt"), n_poses=1, overwrite=True)
        return (idx, smi, y, sc, _heavy(smi), "ok")
    except Exception as e:
        return (idx, smi, y, None, _heavy(smi), f"err:{type(e).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); POSES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/processed/bace_clean.csv")
    te = df[df.set == "test"].reset_index(drop=True)
    jobs = [(i, s, float(y)) for i, (s, y) in enumerate(zip(te.smiles, te.label))]
    if args.smoke:
        jobs = jobs[:3]
    nw = 1 if args.smoke else args.workers
    print(f"docking {len(jobs)} ligands, box={BOX_CENTER}/{BOX_SIZE}, workers={nw}", flush=True)

    rows = []
    with mp.Pool(processes=nw, initializer=_init) as pool:
        for n, r in enumerate(pool.imap_unordered(dock_one_p, jobs), 1):
            rows.append(r)
            if n % 25 == 0 or args.smoke:
                ok = sum(1 for x in rows if x[3] is not None)
                print(f"  {n}/{len(jobs)} ({ok} ok); last score={r[3]} ({r[5]})", flush=True)
    out = pd.DataFrame(rows, columns=["idx", "smiles", "y_true", "vina_score", "heavy", "status"]).sort_values("idx")
    if not args.smoke:
        out.to_csv(OUT / "dock_scores.csv", index=False)
        print(f"wrote {OUT/'dock_scores.csv'} ({out.vina_score.notna().sum()}/{len(out)} docked)")


if __name__ == "__main__":
    main()
