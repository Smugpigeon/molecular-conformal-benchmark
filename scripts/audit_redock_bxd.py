"""Re-docking validation for the BACE-1 module (cognate / self-docking control).

RUN THIS ON THE SERVER ONLY (conda env `drug`, cwd ~/ppi_bias). It needs the
docking toolchain (Vina python API + Meeko) and network access to the PDB; the
project author's laptop has none of these, so it was NOT run there and contains
NO fabricated numbers.

WHAT IT DOES
------------
A redock (cognate self-docking) validation is the standard sanity check that the
docking setup can reproduce a known crystallographic binding mode before any
score is trusted. It:
  1. fetches PDB 4D8C (beta-secretase / BACE-1),
  2. extracts the co-crystal ligand BXD from chain A as the reference pose,
  3. prepares the receptor with Meeko using EXACTLY the box this project's
     module already uses (center 30.58, 6.24, 14.52; size 22 x 22 x 26 A),
  4. re-docks BXD with AutoDock Vina,
  5. computes RMSD(top docked pose, crystallographic BXD) with
     RDKit Chem.rdMolAlign.GetBestRMS (symmetry-corrected; obrms fallback).

Acceptance: redock RMSD < 2.0 A is the conventional "pose reproduced" threshold.
Report the actual value either way -- do NOT tune the box to pass.

PARAMETERS SOURCE (reused verbatim, not re-derived)
---------------------------------------------------
  - BOX_CENTER / BOX_SIZE and the Meeko/Vina receptor-prep recipe come from
        src/structure/dock.py        (BOX_CENTER, BOX_SIZE, smiles_to_pdbqt)
        scripts/dock_bace.py         (--receptor structure/bace/bace_receptor.pdbqt)
    and are corroborated in report/课程报告.md S3.4. The receptor is prepared
    ONCE offline exactly as documented in src/structure/dock.py lines 3-5:
        mk_prepare_receptor.py --read_pdb rec_chainA.pdb -o bace_receptor -p \
            --box_center 30.58 6.24 14.52 --box_size 22 22 26 --charge_model gasteiger

CATALYTIC DYAD / PROTONATION CAVEAT (read before trusting any score)
-------------------------------------------------------------------
BACE-1 is an aspartic protease whose catalytic dyad Asp32 / Asp228 sits in the
pocket. Its optimal activity is at acidic pH (~4.5), where one aspartate is
typically protonated and the other charged (mono-protonated dyad). The dyad
protonation state strongly affects H-bonding to the ligand and therefore the
Vina score and the docked pose. VERIFY the protonation of Asp32 / Asp228 in the
prepared receptor (e.g. with the pH ~4.5 model from a protonation tool such as
PDB2PQR/PROPKA or H++), and state in the report which dyad protonation was used.
This script does not alter protonation; it documents the requirement so the
receptor used here matches the one used for scripts/dock_bace.py.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import urllib.request
from pathlib import Path

# Reuse the project's box -- single source of truth, do not hardcode a second copy.
from src.structure.dock import BOX_CENTER, BOX_SIZE
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)

# BXD = the co-crystallised inhibitor in 4D8C (HET code BXD), per src/structure/dock.py.
LIGAND_RESNAME = "BXD"
PDB_ID = "4D8C"


def fetch_pdb(pdb_id: str, out_dir: Path) -> Path:
    """Download <pdb_id>.pdb from RCSB into out_dir (skip if present)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{pdb_id.lower()}.pdb"
    if dst.exists():
        logger.info(f"PDB already present: {dst}")
        return dst
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    logger.info(f"Downloading {url}")
    urllib.request.urlretrieve(url, dst)  # noqa: S310 (trusted RCSB host)
    return dst


def extract_ligand_pdb(pdb_path: Path, resname: str, chain: str, out_pdb: Path) -> int:
    """Write only HETATM lines matching `resname` on `chain` to out_pdb.

    Returns the number of atom lines written so the caller can fail fast on 0.
    """
    n = 0
    with pdb_path.open() as fh, out_pdb.open("w") as out:
        for line in fh:
            if line.startswith("HETATM") and line[17:20].strip() == resname:
                # PDB column 22 (0-based 21) is the chain id.
                if chain == "*" or line[21] == chain:
                    out.write(line)
                    n += 1
        out.write("END\n")
    return n


def ligand_to_rdkit(ref_pdb: Path, template_smiles: str | None):
    """Load the crystal ligand as an RDKit Mol.

    Crystallographic ligand records carry no bond orders, so if a reference
    SMILES is available we assign bond orders from it (recommended). Otherwise
    fall back to RDKit perception and let GetBestRMS handle heavy atoms only.
    """
    from rdkit import Chem  # noqa: PLC0415
    from rdkit.Chem import AllChem  # noqa: PLC0415

    mol = Chem.MolFromPDBFile(str(ref_pdb), removeHs=True, sanitize=False)
    if mol is None:
        raise RuntimeError(f"RDKit could not read crystal ligand from {ref_pdb}")
    if template_smiles:
        # Trigger: a reference SMILES for BXD was supplied (--ligand-smiles).
        # Why:     PDB ligand blocks lack bond orders; assigning from a template
        #          gives a chemically correct molecule for symmetry-aware RMSD.
        # Outcome: GetBestRMS compares like-for-like atoms instead of a guessed graph.
        ref = Chem.MolFromSmiles(template_smiles)
        if ref is None:
            raise ValueError(f"Bad --ligand-smiles: {template_smiles!r}")
        mol = AllChem.AssignBondOrdersFromTemplate(ref, mol)
    Chem.SanitizeMol(mol)
    return mol


def best_rms(ref_mol, probe_mol) -> float:
    """Symmetry-corrected RMSD via RDKit GetBestRMS (Angstrom)."""
    from rdkit.Chem import rdMolAlign  # noqa: PLC0415

    return float(rdMolAlign.GetBestRMS(probe_mol, ref_mol))


def obrms_fallback(ref_pdb: Path, probe_pdb: Path) -> float:
    """obrms RMSD fallback if the RDKit path cannot match the two graphs."""
    out = subprocess.run(  # noqa: S603
        ["obrms", str(ref_pdb), str(probe_pdb)],
        capture_output=True,
        text=True,
        check=True,
    )
    # obrms prints e.g. "RMSD <file>: 1.234"
    return float(out.stdout.strip().split()[-1])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_all_seeds(42)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--receptor",
        default="structure/bace/bace_receptor.pdbqt",
        help="SAME prepared receptor used by scripts/dock_bace.py (Meeko, gasteiger).",
    )
    p.add_argument("--chain", default="A", help="Chain to take BXD from (4D8C uses A).")
    p.add_argument(
        "--ligand-smiles",
        default=None,
        help="Reference SMILES for BXD (recommended) to assign bond orders for RMSD.",
    )
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--n-poses", type=int, default=10)
    p.add_argument("--work", type=Path, default=Path("results/wang_audit/redock"))
    args = p.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)

    if not Path(args.receptor).exists():
        logger.error(
            f"Receptor not found: {args.receptor}. Prepare it ONCE with Meeko exactly as "
            "documented in src/structure/dock.py (mk_prepare_receptor.py ... "
            "--box_center 30.58 6.24 14.52 --box_size 22 22 26 --charge_model gasteiger)."
        )
        return 1

    # 1-2. Fetch 4D8C and carve out the crystallographic BXD pose (reference).
    pdb_path = fetch_pdb(PDB_ID, args.work)
    ref_pdb = args.work / "bxd_crystal.pdb"
    n_atoms = extract_ligand_pdb(pdb_path, LIGAND_RESNAME, args.chain, ref_pdb)
    if n_atoms == 0:
        logger.error(
            f"No HETATM {LIGAND_RESNAME} found on chain {args.chain} in {pdb_path}. "
            "Check the HET code / chain (inspect the PDB header)."
        )
        return 1
    logger.info(f"Extracted crystal {LIGAND_RESNAME}: {n_atoms} atoms -> {ref_pdb}")

    ref_mol = ligand_to_rdkit(ref_pdb, args.ligand_smiles)

    # 3-4. Prepare BXD as a ligand and redock with Vina, reusing the project box.
    #      We start the ligand from its OWN crystal coordinates (cognate redock),
    #      preparing PDBQT via Meeko so the recipe matches src/structure/dock.py.
    from meeko import MoleculePreparation, PDBQTWriterLegacy  # noqa: PLC0415
    from rdkit import Chem  # noqa: PLC0415
    from vina import Vina  # noqa: PLC0415

    lig_for_dock = Chem.AddHs(ref_mol, addCoords=True)
    prep = MoleculePreparation()
    setups = prep.prepare(lig_for_dock)
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        logger.error(f"Meeko failed to write BXD PDBQT: {err}")
        return 1

    v = Vina(sf_name="vina", seed=42, verbosity=1)
    v.set_receptor(str(args.receptor))
    v.set_ligand_from_string(pdbqt)
    v.compute_vina_maps(center=list(BOX_CENTER), box_size=list(BOX_SIZE))
    v.dock(exhaustiveness=args.exhaustiveness, n_poses=args.n_poses)
    top_pdbqt = args.work / "bxd_redock_top.pdbqt"
    v.write_poses(str(top_pdbqt), n_poses=1, overwrite=True)
    top_score = float(v.energies(n_poses=1)[0][0])
    logger.info(f"Top redock Vina score = {top_score:+.3f} kcal/mol")

    # Convert the top docked pose to PDB for RDKit / obrms RMSD.
    top_pdb = args.work / "bxd_redock_top.pdb"
    subprocess.run(  # noqa: S603
        ["obabel", str(top_pdbqt), "-O", str(top_pdb)],
        capture_output=True,
        text=True,
        check=True,
    )

    # 5. RMSD: RDKit GetBestRMS first; obrms as fallback.
    try:
        probe = ligand_to_rdkit(top_pdb, args.ligand_smiles)
        rmsd = best_rms(ref_mol, probe)
        method = "RDKit GetBestRMS"
    except Exception as e:  # noqa: BLE001 (we want the obrms fallback, then re-raise on total failure)
        logger.warning(f"RDKit RMSD failed ({e}); trying obrms.")
        rmsd = obrms_fallback(ref_pdb, top_pdb)
        method = "obrms"

    verdict = "PASS (<2.0 A)" if rmsd < 2.0 else "FAIL (>=2.0 A)"
    print("\n" + "=" * 60)
    print(f"BACE-1 REDOCK VALIDATION (cognate self-docking, {PDB_ID}/{LIGAND_RESNAME})")
    print("=" * 60)
    print(f"box center                 = {BOX_CENTER}")
    print(f"box size                   = {BOX_SIZE}")
    print(f"top redock Vina score      = {top_score:+.3f} kcal/mol")
    print(f"redock RMSD ({method})     = {rmsd:.3f} A")
    print(f"verdict                    = {verdict}")
    print("NOTE: confirm Asp32/Asp228 dyad protonation (BACE optimal pH ~4.5);")
    print("      it strongly affects scoring and the reproduced pose.")

    # Persist a small machine-readable record for the report (no fabrication).
    rec = args.work / "redock_result.csv"
    with rec.open("w") as fh:
        fh.write("pdb,ligand,box_center,box_size,vina_score,rmsd_A,rmsd_method,verdict\n")
        fh.write(
            f"{PDB_ID},{LIGAND_RESNAME},"
            f"\"{BOX_CENTER}\",\"{BOX_SIZE}\","
            f"{top_score:.3f},{rmsd:.3f},{method},{verdict}\n"
        )
    logger.info(f"Wrote {rec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
