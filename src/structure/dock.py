"""AutoDock Vina docking pipeline for BACE-1 (Part B).

Receptor prep is done ONCE offline (see project README / structure/bace/):
    mk_prepare_receptor.py --read_pdb rec_chainA.pdb -o bace_receptor -p \\
        --box_center 30.58 6.24 14.52 --box_size 22 22 26 --charge_model gasteiger

This module handles ligand prep (SMILES -> 3D -> PDBQT via RDKit + Meeko) and
docking (Vina python API). Per CLAUDE.md §7.5: failed molecules are logged,
not silently dropped — they get a NaN score.

Box (from co-crystal ligand BXD in 4D8C chain A):
    center = (30.58, 6.24, 14.52), size = (22, 22, 26)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

BOX_CENTER = (30.58, 6.24, 14.52)
BOX_SIZE = (22.0, 22.0, 26.0)


def smiles_to_pdbqt(smi: str, seed: int = 42) -> str | None:
    """SMILES -> 3D (ETKDGv3 + MMFF) -> PDBQT string. None on failure.

    Per CLAUDE.md §9: ETKDGv3 + MMFF94 is the project standard for 3D.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # Embedding failed; try a looser attempt.
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=seed) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass  # keep unoptimized geometry rather than fail

    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy  # noqa: PLC0415

        prep = MoleculePreparation()
        setups = prep.prepare(mol)
        pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
        if not ok:
            logger.warning(f"Meeko write failed for {smi!r}: {err}")
            return None
        return pdbqt
    except Exception as e:
        logger.warning(f"Ligand prep failed for {smi!r}: {e}")
        return None


def dock_one(
    smi: str,
    receptor_pdbqt: str | Path,
    exhaustiveness: int = 8,
    n_poses: int = 10,
    seed: int = 42,
) -> float:
    """Dock one SMILES, return best Vina affinity (kcal/mol). NaN on failure.

    Lower (more negative) = stronger predicted binding.
    """
    from vina import Vina  # noqa: PLC0415

    pdbqt = smiles_to_pdbqt(smi, seed=seed)
    if pdbqt is None:
        return float("nan")

    try:
        v = Vina(sf_name="vina", seed=seed, verbosity=0)
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_string(pdbqt)
        v.compute_vina_maps(center=list(BOX_CENTER), box_size=list(BOX_SIZE))
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        # energies(): rows of poses; col 0 = total affinity.
        best = float(v.energies(n_poses=1)[0][0])
        return best
    except Exception as e:
        logger.warning(f"Docking failed for {smi!r}: {e}")
        return float("nan")


def dock_many(
    smiles: list[str],
    receptor_pdbqt: str | Path,
    exhaustiveness: int = 8,
    seed: int = 42,
) -> list[float]:
    """Dock a list of SMILES, return list of best affinities (NaN on failure)."""
    scores: list[float] = []
    for i, smi in enumerate(smiles):
        score = dock_one(smi, receptor_pdbqt, exhaustiveness=exhaustiveness, seed=seed)
        scores.append(score)
        logger.info(f"[{i+1}/{len(smiles)}] {smi[:40]:40s} -> {score:.2f} kcal/mol")
    n_fail = int(np.isnan(scores).sum())
    if n_fail:
        logger.warning(f"{n_fail}/{len(smiles)} ligands failed to dock")
    return scores
