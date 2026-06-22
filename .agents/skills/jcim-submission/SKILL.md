---
name: jcim-submission
description: Shape a cheminformatics / molecular-ML manuscript to Journal of Chemical Information and Modeling (JCIM, ACS) submission standard. Use when the user wants to turn results into a JCIM paper, draft/restructure a JCIM Article, or check whether a manuscript meets JCIM + reproducibility bar. Encodes JCIM author requirements + the project's §16 anti-BS gate. Pair with journal-polish for prose. Not for non-ACS venues (Nature/Science/IEEE use different rules).
---

# JCIM-Submission

Turn rigorous cheminformatics results into a **JCIM Article draft** that meets ACS author
requirements AND the project's reproducibility/anti-BS gate. This skill defines the
*target*; `journal-polish` does the sentence-level prose. Run both.

## Honesty first (overrides everything)

A JCIM skill does **not** make weak science strong. It formats and frames genuine work.
- Never inflate tier or novelty to "look JCIM". A JCIM reviewer is a domain expert.
- Keep every honest hedge from `AGENTS.md §15–16`: non-significant results, leakage
  disclosure, AD under-coverage, the MUBen distinction. Strengthen, never delete.
- If the contribution is incremental, say so precisely and let rigor carry it.

## JCIM Article requirements (from ACS author guidelines, coden jcisd8)

- **Title**: ≤ 12 words, specific, no hype.
- **Abstract**: 3–4 sentences, concise — problem, what you did, key result, implication.
- **TOC graphic**: required (one clear figure that captures the contribution).
- **Sections** (standard original-research layout):
  Introduction → Methods (Computational/Experimental Section) → Results and Discussion →
  Conclusions. Article has **no hard word limit** (unlike Letters ~3500 w) — depth is fine;
  padding is not.
- **References**: ACS style, **complete including article titles**. Per `AGENTS.md §11`,
  never hand-type DOIs/PMIDs — derive + verify.
- **Language**: US English (ACS convention; do NOT use British spelling from Nature tools).
- **Data Availability Statement**: MANDATORY (ACS Research Data Policy Level 2) — state
  where data + code are publicly available.
- **Reproducibility**: software/servers must be testable by reviewers.

## Focus rule (what makes it an Article, not a report)

A JCIM Article reports **one clear contribution**. Restructure around the single novel
claim; demote supporting work to context, Methods, or Supporting Information.
- Lead the Introduction with the gap, not a textbook tour.
- The Abstract's key-result sentence must name the actual finding (a number), not a topic.

## §16 reproducibility gate (must pass before "submission-ready")

Check each; disclose what fails — do not hide it:
- [ ] **Split honesty (§16.3)**: leakage checked (canonical-SMILES dup, InChIKey-14,
      scaffold overlap, Tanimoto > 0.85). Disclose any > 0.
- [ ] **Comparison fairness (§16.2)**: every cited number tagged with split type + metric
      + n_rep. No "new SOTA" across non-comparable splits — say "in our split".
- [ ] **Significance (§16.4)**: no "outperforms" without a test; "comparable" if p > 0.05;
      Bonferroni for many comparisons.
- [ ] **Negative control (§16.8)**: label-shuffle sanity reported (R≈0 / AUC≈0.5).
- [ ] **Applicability domain (§16.6)**: metrics stratified by Tanimoto-to-train; report,
      don't hide, low-similarity degradation.
- [ ] **Preregistration (§16.9)**: hypotheses fixed before final test eval; failed ones
      reported as "not supported".
- [ ] **Noise floor (§16.5)**: report model error vs assay σ where known.

## Workflow

1. Identify the single contribution; pick the TOC-graphic figure.
2. Draft/restructure into the JCIM section layout from the existing material (don't invent
   results). Source of truth = build script, not the rendered file.
3. Apply `journal-polish` (US-English house style) to every section.
4. Run the §16 gate above; write the Data Availability Statement.
5. Output an **honest gap list** of what only the authors can finish: final author list +
   ORCIDs, real preregistration timestamp, public data/code DOI (HF/Zenodo), TOC graphic
   sign-off, co-author approval. Never fabricate these.
6. References: derive + verify (§11), never hand-type.
