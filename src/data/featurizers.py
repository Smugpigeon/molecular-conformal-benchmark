"""Molecular featurizers.

CLAUDE.md §9: Morgan FP is fixed at radius=2, fpSize=2048 across the project.
Any other config must be named explicitly (e.g., morgan_r3_2048) so wandb
runs are not silently incomparable.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

logger = logging.getLogger(__name__)


def smiles_to_morgan(
    smi: str, radius: int = 2, n_bits: int = 2048
) -> NDArray[np.int8] | None:
    """Morgan fingerprint, defaults per CLAUDE.md §9.

    Returns:
        int8 array of shape (n_bits,) or None if SMILES is unparseable
        or yields an empty molecule.
    """
    # Trigger: empty / non-string input.
    # Why:     RDKit 2024.03 returns a 0-atom Mol for "" (not None), which
    #          would silently produce all-zero fingerprints and corrupt training.
    # Outcome: reject early so batch_morgan logs it as a failure.
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def batch_morgan(
    smiles: list[str],
    radius: int = 2,
    n_bits: int = 2048,
    drop_failed: bool = True,
) -> tuple[NDArray[np.int8], list[int]]:
    """Vectorized Morgan FP.

    CLAUDE.md §7.5: failed SMILES are logged, not silently dropped.

    Returns:
        (X, kept_indices) where X has shape (len(kept_indices), n_bits) and
        kept_indices preserves the original positions of successful parses.
        Use kept_indices to align y labels.
    """
    fps: list[NDArray[np.int8]] = []
    kept: list[int] = []
    failed_examples: list[str] = []
    for i, s in enumerate(smiles):
        fp = smiles_to_morgan(s, radius=radius, n_bits=n_bits)
        if fp is not None:
            fps.append(fp)
            kept.append(i)
        else:
            if len(failed_examples) < 5:
                failed_examples.append(s)

    n_failed = len(smiles) - len(kept)
    if n_failed > 0:
        logger.warning(
            f"Morgan FP: {n_failed}/{len(smiles)} SMILES failed to parse"
        )
        for s in failed_examples:
            logger.warning(f"  Failed: {s!r}")
        if not drop_failed:
            raise ValueError(
                f"{n_failed} SMILES failed and drop_failed=False"
            )

    if not fps:
        return np.zeros((0, n_bits), dtype=np.int8), kept
    return np.vstack(fps), kept
