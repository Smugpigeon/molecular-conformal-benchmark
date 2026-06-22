// Preview render of the NeurIPS-2026-format tables (Times New Roman + booktabs
// rules). The authentic artifact is nips_tables.tex (compile on Overleaf with the
// bundled neurips_2026.sty). All numbers are from the released result CSVs.

#set page(width: 19cm, height: auto, margin: (x: 1.3cm, top: 1.2cm, bottom: 1.2cm))
#set text(font: ("Times New Roman", "Noto Serif CJK SC"), size: 9.5pt)
#set par(justify: false, leading: 0.55em)

#align(center)[
  #text(size: 14pt, weight: "bold")[BACE-1 Inhibitor Activity Prediction:\
  Data Provenance, Integrity Audit, and Benchmark Tables]
  #v(3pt)
  #text(size: 10pt)[Yufan Tang #h(6pt)·#h(6pt) School of Pharmacy, Fudan University #h(6pt)·#h(6pt) 23307130372]
  #v(4pt)
  #text(size: 8.5pt, style: "italic", fill: rgb("#555"))[NeurIPS-2026-format preview · every value computed from released result files]
]
#v(8pt)

#let TR = 1pt
#let MR = 0.5pt
#let ms(m, s) = box[#m#text(size: 7pt)[\u{2009}±#s]]
#let bms(m, s) = box[*#m*#text(size: 7pt)[\u{2009}±#s]]
#let cap(n, body) = [#v(9pt) #text(weight: "bold")[Table #n.] #h(3pt) #text(size: 9pt)[#body] #v(4pt)]
#let grp(body) = table.cell(colspan: 3, inset: (top: 5pt, bottom: 2pt))[#text(style: "italic")[#body]]

// ===================== Table 1 =====================
#cap(1)[BACE-1 dataset card. Source is the course-provided MoleculeNet split; the
regression target is pIC#sub[50] (higher = stronger inhibition). RDKit
canonicalization (once at ingest) and de-duplication leave the count unchanged.]
#table(
  columns: (auto, 1fr), stroke: none, align: (left, left),
  inset: (x: 7pt, y: 3pt),
  table.hline(stroke: TR),
  [*Property*], [*Value*],
  table.hline(stroke: MR),
  [Task], [BACE-1 inhibition, regression (pIC#sub[50])],
  [Source], [BACE.split.csv (course-provided)],
  [Molecules (total)], [1,513],
  [Train / Validation / Test], [1,059 / 151 / 303 (predefined)],
  [pIC#sub[50] mean ± std], [6.52 ± 1.34],
  [pIC#sub[50] range], [\[2.54, 10.52\]],
  [Canonicalization failures], [0],
  [Canonical duplicate SMILES], [0],
  [Undefined stereochemistry], [1,078 / 1,513  (71%)],
  [Protein structure (docking)], [PDB 4D8C (public, RCSB; not in the CSV)],
  table.hline(stroke: TR),
)

// ===================== Table 2 =====================
#cap(2)[Data-integrity audit of the predefined split. Top: train/test leakage
diagnostics. Middle: a scaffold-disjoint re-split stress test (0% Murcko overlap)
quantifies inflation from train/test similarity. Bottom: a label-permutation
negative control; Pearson #emph[R] ≈ 0 confirms the signal is real, not memorized.]
#table(
  columns: (1fr, auto, auto), stroke: none, align: (left, center, center),
  inset: (x: 7pt, y: 3pt),
  table.hline(stroke: TR),
  [*Diagnostic / test*], [*Value*], [*Note*],
  table.hline(stroke: MR),
  grp[Train/test leakage (predefined split)],
  [Canonical duplicate SMILES (train ∩ test)], [0  (0.0%)], [clean],
  [Tautomer duplicates (InChIKey#sub[14])], [13  (4.3%)], [disclosed],
  [Murcko scaffold overlap (test in train)], [193  (63.7%)], [high],
  [Nearest-train Tanimoto > 0.85], [89  (29.4%)], [high],
  [Mean nearest-train Tanimoto], [0.775], [—],
  grp[Scaffold re-split stress test (RF + Morgan, 3 seeds)],
  [Pearson #emph[R]: predefined → scaffold], [0.843 → 0.778], [−0.065],
  [MAE: predefined → scaffold], [0.546 → 0.653], [+0.107],
  [Scaffold overlap: predefined → scaffold], [64.0% → 0%], [by design],
  grp[Negative control (label permutation, 10 repeats)],
  [Permuted Pearson #emph[R]], [−0.06 ± 0.04], [≈ 0, pass],
  table.hline(stroke: TR),
)

// ===================== Table 3 =====================
#cap(3)[Five molecular representations on BACE pIC#sub[50] regression (predefined
test split, mean ± std over 3 seeds). S1 = course strategy 1 (SMILES/fingerprint →
ML); S2 = strategy 2 (graph → GNN); 3D = beyond syllabus. Best per column in
*bold*. The graph network (Chemprop) wins on all four metrics.]
#table(
  columns: (1.7fr, 1fr, 1fr, 1fr, 1fr), stroke: none,
  align: (left, center, center, center, center),
  inset: (x: 6pt, y: 3.5pt),
  table.hline(stroke: TR),
  [*Representation*], [*MAE* ↓], [*RMSE* ↓], [*Pearson R* ↑], [*Spearman ρ* ↑],
  table.hline(stroke: MR),
  [RF + Morgan #super[S1]], ms("0.643","0.001"), ms("0.833","0.004"), ms("0.804","0.002"), ms("0.752","0.001"),
  [MolFormer-XL #super[S1]], ms("0.657","0.076"), ms("0.859","0.095"), ms("0.798","0.045"), ms("0.746","0.046"),
  [ChemFM-3B #super[S1]], ms("0.651","0.011"), ms("0.826","0.010"), ms("0.803","0.005"), ms("0.754","0.005"),
  [Chemprop (D-MPNN) #super[S2]], bms("0.604","0.017"), bms("0.778","0.016"), bms("0.835","0.007"), bms("0.784","0.006"),
  table.hline(stroke: TR),
)

// ===================== Table 4 =====================
#cap(4)[Split-conformal prediction intervals on BACE (target coverage 0.90, mean ±
std over 3 seeds; width in pIC#sub[50] units). The #emph[tightest] interval
(ChemFM) is not valid — it under-covers at 0.895 < 0.90 — whereas Chemprop gives
the tightest #emph[valid] interval (*bold*), ≈ ±1.33 pIC#sub[50].]
#table(
  columns: (1.5fr, 1.2fr, 1fr, 1.4fr), stroke: none,
  align: (left, center, center, left),
  inset: (x: 7pt, y: 3.5pt),
  table.hline(stroke: TR),
  [*Representation*], [*Coverage* (target 0.90)], [*Width* ↓], [*Status*],
  table.hline(stroke: MR),
  [RF + Morgan], ms("0.934","0.003"), ms("2.873","0.018"), [valid],
  [MolFormer-XL], ms("0.924","0.009"), ms("2.934","0.310"), [valid],
  [ChemFM-3B], ms("0.895","0.007"), ms("2.633","0.050"), [under-covers (tightest)],
  [Chemprop (D-MPNN)], ms("0.925","0.011"), bms("2.660","0.171"), [valid (tightest valid)],
  table.hline(stroke: TR),
)

// ===================== Table 5 =====================
#cap(5)[Per-model training adequacy on BACE (3 seeds). Every model was trained to a
validation plateau. RF has no epochs (budget = ensemble size). Chemprop uses the
final epoch in production; a diagnostic re-run confirms epoch 50 sits on the plateau
(final val loss within 0.005 of the best).]
#table(
  columns: (1.5fr, 1.1fr, 0.7fr, 1fr, 1.5fr), stroke: none,
  align: (left, left, center, center, left),
  inset: (x: 6pt, y: 3.5pt),
  table.hline(stroke: TR),
  [*Representation*], [*Selection rule*], [*Budget*], [*Selected epoch*], [*Convergence (val)*],
  table.hline(stroke: MR),
  [RF + Morgan #super[S1]], [ensemble size], [500 trees], [~50 trees#super[†]], [plateau by ~50 trees],
  [MolFormer-XL #super[S1]], [best-on-val], [30 ep], ms("24.0","3.7"), [converged; high variance],
  [ChemFM-3B #super[S1]], [best-on-val], [10 ep], ms("7.7","1.2"), [clean plateau],
  [Chemprop (D-MPNN) #super[S2]], [final epoch], [50 ep], [50 (best ~38)], [final ≈ best (Δ 0.005)],
  table.hline(stroke: TR),
)
#text(size: 7pt)[#super[†] first tree count within 1% of the 500-tree validation RMSE.]

#v(11pt)
#image("fig_convergence_j.png", width: 100%)
#v(2pt)
#text(size: 8.5pt)[*Figure 1.* #h(2pt) Training convergence verification on BACE
(4 representations, 3 seeds; mean ± s.d.). Solid = validation trajectory with 3-seed
band; faded = training loss (twin axis); ★ = selected epoch. (a) RF converges in tree
count — flat past ~50 trees; (b) MolFormer's wide band explains its largest seed
variance; (c) ChemFM (LoRA, 0.2% of parameters) plateaus by epoch 8; (d) Chemprop's
epoch-50 model is on the plateau (gap to best 0.005); (e) a normalized overlay
compares convergence speed across all four.]
