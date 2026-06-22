"""D: re-dock the native co-crystal ligand BXD into 4D8C and measure RMSD to the crystal pose.
Low RMSD validates the docking protocol (so the weak docking-vs-activity signal in Phase A is a
scoring-function limitation, not a docking failure)."""
from __future__ import annotations
import os, sys
sys.path.insert(0,".")
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
from src.structure.dock import smiles_to_pdbqt, BOX_CENTER, BOX_SIZE
ideal="docking/BXD_ideal.sdf"; crystal="docking/bxd_ref.pdb"
tmpl=Chem.MolFromMolFile(ideal)
smi=Chem.MolToSmiles(tmpl)
print("BXD SMILES:", smi)
# dock
from vina import Vina
lig=smiles_to_pdbqt(smi)
v=Vina(sf_name="vina",cpu=4,seed=42,verbosity=0)
v.set_receptor("structure/bace/bace_receptor.pdbqt")
v.compute_vina_maps(center=list(BOX_CENTER),box_size=list(BOX_SIZE))
v.set_ligand_from_string(lig); v.dock(exhaustiveness=16,n_poses=10)
v.write_poses("docking/bxd_redock.pdbqt",n_poses=1,overwrite=True)
print("redock score:",round(float(v.energies(n_poses=1)[0][0]),2),"kcal/mol")
# docked pose -> mol
_bin = os.path.dirname(sys.executable)  # current env's bin dir (Meeko's mk_export.py lives here)
os.system(f"{sys.executable} {_bin}/mk_export.py docking/bxd_redock.pdbqt -s docking/bxd_redock.sdf >/dev/null 2>&1")
docked=Chem.RemoveHs(Chem.MolFromMolFile("docking/bxd_redock.sdf"))
# crystal pose with correct bonds
cry=Chem.MolFromPDBFile(crystal,sanitize=False)
try: cry=AllChem.AssignBondOrdersFromTemplate(tmpl,cry)
except Exception as e: print("bond assign warn:",e)
cry=Chem.RemoveHs(cry)
rmsd=AllChem.GetBestRMS(docked,cry)
print(f"RE-DOCK RMSD to crystal = {rmsd:.2f} A   ({'GOOD (<2A)' if rmsd<2 else 'acceptable (<3A)' if rmsd<3 else 'poor'})")
