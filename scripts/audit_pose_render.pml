# audit_pose_render.pml — BACE-1 docked-pose binding-mode figure (Wang review, Point 6)
# =====================================================================================
# RUN ON THE SERVER ONLY. Needs PyMOL + the saved docked poses + the prepared receptor
# (these do not exist on the author's laptop). This script was NOT run locally and the
# report contains NO binding-mode description until a real render is produced here.
#
# MOLECULE SELECTION (from real data in results/bace_docking_full.csv, n=30 docked):
#   A) "potent, trustworthy"   -> row index 29: pIC50 = 9.19 (most potent), Vina = -8.92, LE = -0.297
#   B) "docking disagreement"  -> row index  3: pIC50 = 4.65 (near-weakest), Vina = -10.40 (near-strongest), LE = -0.372
#   Rationale: the pair makes the ligand-efficiency finding visual — Vina's *strongest*
#   binder (B) is experimentally one of the *weakest*, while the most potent ligand (A)
#   scores only mid-pack. Render both, contrast their contacts with the catalytic dyad.
#
# USAGE (run once per ligand; edit paths to the actual server files):
#   pymol -cq scripts/audit_pose_render.pml -- <receptor.pdb|pdbqt> <ligand_docked_pose.pdb|sdf|pdbqt> <out.png>
#
# CAVEAT: confirm the catalytic-dyad residue numbers in 4D8C. The canonical BACE-1
# catalytic aspartate dyad is Asp32 / Asp228; some constructs renumber — verify in the
# actual receptor and adjust `resi 32+228` below before trusting the H-bond picture.

python
from pymol import cmd
import sys

argv = sys.argv
recept, ligpose, outpng = argv[1], argv[2], argv[3]

cmd.load(recept, "receptor")
cmd.load(ligpose, "lig")

cmd.hide("everything")
cmd.bg_color("white")
cmd.show("cartoon", "receptor")
cmd.set("cartoon_transparency", 0.45, "receptor")
cmd.color("grey80", "receptor")

# catalytic aspartate dyad of BACE-1 (VERIFY numbering against 4D8C)
cmd.select("dyad", "receptor and resi 32+228 and resn ASP")
cmd.show("sticks", "dyad")
cmd.color("orange", "dyad and elem C")

# pocket residues within 4.5 A of the ligand (context)
cmd.select("pocket", "byres (receptor within 4.5 of lig)")
cmd.show("sticks", "pocket and not name C+N+O")
cmd.set("stick_transparency", 0.5, "pocket")

cmd.show("sticks", "lig")
cmd.color("cyan", "lig and elem C")
cmd.util.cnc("lig or dyad or pocket")

# polar contacts (H-bonds) ligand <-> dyad, and ligand <-> whole pocket
cmd.distance("hb_dyad", "lig", "dyad", mode=2)
cmd.color("red", "hb_dyad")
cmd.distance("hb_pocket", "lig", "pocket", mode=2)
cmd.color("yellow", "hb_pocket")

cmd.set("dash_width", 3)
cmd.zoom("lig", 6)
cmd.orient("lig")
cmd.set("ray_opaque_background", 0)
cmd.ray(1600, 1200)
cmd.png(outpng, dpi=300)
print("wrote", outpng)
python end
