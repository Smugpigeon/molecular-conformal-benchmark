"""EDA Step 1 (part 3): F3 activity-cliff structural pairs (the starred structural figure).

Renders real molecule pairs from results/eda/activity_cliffs_top.csv with RDKit, aligning
each pair on its maximum common substructure and highlighting the atoms that differ, so the
reader literally sees "same scaffold, one change (or a stereo inversion), activity jumps N log".
Complements F6 (the statistical cliff network) with concrete structures. RDKit for molecules.

Run: /opt/anaconda3/bin/python3 scripts/eda_bace_part3.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, rdFMCS  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402

RDLogger.DisableLog("rdApp.*")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix"})
FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
HL = (0.96, 0.55, 0.55)  # soft red for the differing atoms


def diff_atoms(a, b):
    """Atoms of a / b that are NOT in the maximum common substructure (the 'change')."""
    mcs = rdFMCS.FindMCS([a, b], timeout=10, matchValences=True,
                          ringMatchesRingOnly=True, completeRingsOnly=True)
    patt = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None
    if patt is None:
        return [], [], None
    ma = set(a.GetSubstructMatch(patt)); mb = set(b.GetSubstructMatch(patt))
    ha = [at.GetIdx() for at in a.GetAtoms() if at.GetIdx() not in ma]
    hb = [at.GetIdx() for at in b.GetAtoms() if at.GetIdx() not in mb]
    return ha, hb, patt


def render(mol, highlight, w=380, h=300):
    d = rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions(); o.bondLineWidth = 2; o.minFontSize = 13
    hl = {i: HL for i in highlight}
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=highlight,
                                       highlightAtomColors=hl)
    d.FinishDrawing()
    import io
    return Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB")


def main():
    df = pd.read_csv("results/eda/activity_cliffs_top.csv")
    # the cliff table only stores delta; recover each end's true pIC50 from the cleaned data
    lab = pd.read_csv("data/processed/bace_clean.csv").set_index("smiles").label.to_dict()
    df["label_A"] = df.smiles_A.map(lab); df["label_B"] = df.smiles_B.map(lab)
    df = df.dropna(subset=["label_A", "label_B"]).reset_index(drop=True)
    # one stereo-only cliff (Tanimoto 1.0 = identical 2D graph, stereo differs) + two group-swap
    stereo = df[df.Tanimoto >= 0.999].sort_values("delta_pIC50", ascending=False).iloc[0]
    swap = df[df.Tanimoto < 0.999].sort_values("delta_pIC50", ascending=False)
    rows = [("stereocenter inversion", stereo),
            ("single-group change", swap.iloc[0]),
            ("single-group change", swap.iloc[1])]

    fig, axs = plt.subplots(len(rows), 2, figsize=(8.6, 3.05 * len(rows)))
    for r, (kind, row) in enumerate(rows):
        a = Chem.MolFromSmiles(row.smiles_A); b = Chem.MolFromSmiles(row.smiles_B)
        AllChem.Compute2DCoords(a)
        ha, hb, patt = diff_atoms(a, b)
        # align b onto a through the shared core so the eye lands on the change
        if patt is not None:
            try:
                AllChem.GenerateDepictionMatching2DStructure(b, a, refPatt=patt)
            except Exception:
                AllChem.Compute2DCoords(b)
        else:
            AllChem.Compute2DCoords(b)
        # more active molecule on the left
        if row.label_A is not None and float(row.label_B) > float(row.label_A):
            a, b, ha, hb = b, a, hb, ha
            pa, pb = float(row.label_B), float(row.label_A)
        else:
            pa, pb = float(row.label_A), float(row.label_B)
        for ax, mol, hl, p, tag in [(axs[r, 0], a, ha, pa, "more active"),
                                    (axs[r, 1], b, hb, pb, "less active")]:
            ax.imshow(np.asarray(render(mol, hl))); ax.axis("off")
            ax.set_title(f"pIC$_{{50}}$ = {p:.2f}  ({tag})", fontsize=10.5,
                         color="#1b5e20" if tag == "more active" else "#7a1f1f")
        axs[r, 0].text(-0.07, 0.5, f"{kind}\n$\\Delta$pIC$_{{50}}$ = {row.delta_pIC50:.2f}\n"
                       f"Tanimoto = {row.Tanimoto:.2f}", transform=axs[r, 0].transAxes,
                       fontsize=10, va="center", ha="right", rotation=90,
                       color="#9b2226", fontweight="bold")
    fig.suptitle("BACE activity cliffs: near-identical structures, large activity jumps\n"
                 "(red = the atoms that differ; a one-atom or stereo change moves pIC$_{50}$ "
                 "by 2-3 log units)", fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0.03, 0, 1, 0.97))
    fig.savefig(FIG / "eda_cliff_pairs_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote eda_cliff_pairs_j.png")
    for kind, row in rows:
        print(f"  {kind:24s} dpIC50={row.delta_pIC50:.2f} Tan={row.Tanimoto:.2f}")


if __name__ == "__main__":
    main()
