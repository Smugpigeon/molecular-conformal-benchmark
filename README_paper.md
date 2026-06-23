# Conformal Uncertainty Quantification across Molecular Foundation Models

> Public-repo README for the JCIM manuscript. In a clean public checkout, use this file as `README.md`
> (the existing `README.md` is the course-project readme).

Code and data for *"Conformal Uncertainty Quantification across Molecular Foundation Models for
Small-Sample Regression"* (target: *J. Chem. Inf. Model.*).

A model-agnostic **split-conformal** benchmark of prediction-interval quality across modern
molecular foundation models (MolFormer-XL, ChemFM-3B, Uni-Mol, TabPFN) plus traditional QSAR and a
D-MPNN, on six small-sample regression datasets. We report interval **width** (efficiency) and
empirical **coverage** with Wilson 95% CIs (validity), show that native model uncertainty (MC-Dropout,
deep ensembles, Gaussian-NLL) is overconfident in the probes we evaluated while split conformal
restores near-nominal coverage (a finite-sample marginal guarantee), and that conditional coverage
degrades out of the applicability domain.

## Key results (reproducible)

| Finding | Where |
|---|---|
| Per-fold 8-model significance (TabPFN first; Friedman p≈8e-38) | `results/final/perfold_8model_*.csv`, `results/figures/cd_perfold_8model.png` |
| Tightest-is-not-valid conformal width/coverage (6 datasets) | `results/final/conformal_full.csv`, `results/figures/conformal_multiseed.png` |
| Native UQ overconfident vs conformal (MC-Dropout 0.28–0.62) | `results/final/{uq_extra,mc_dropout_deep,uq_baselines}.csv`, `results/figures/toc_graphic.png` |
| Random-vs-scaffold exchangeability | `results/final/gap4_dual_split.csv` |
| Cross-dataset Demšar CD | `results/figures/cd_crossdataset_{width,mae}.png` |

## Repository layout

```
src/        importable package (data loaders, models, conformal eval, metrics, seeds)
scripts/    pipeline + analysis scripts (one concern each)
configs/    Hydra configs (datasets, models, paths)
data/       public MoleculeNet CSVs (read-only); data/processed/ is rebuildable
results/    result tables (results/final, results/raw) + figures (results/figures)
paper/      achemso LaTeX: manuscript_jcim.tex, refs.bib, manuscript_jcim_SI.tex, si_tables.tex, overleaf/
```

## Quick start

```bash
mamba env create -f env.yml && conda activate drug      # or: pip install -e .

# 1) train predictors (deep models need a GPU); per-seed val+test predictions land in runs/
python -m src.train dataset=bace_clean model=tabpfn seed=42 save_test_preds=true   # example

# 2) conformal interval-quality benchmark (reads saved predictions)
python scripts/run_conformal_full.py            # 6-dataset width + coverage
python scripts/run_conformal_multiseed.py       # multi-seed error bars

# 3) non-conformal UQ baselines + exchangeability
python scripts/run_uq_extra.py bace             # Gaussian-NLL / MC-Dropout / temp scaling vs conformal
python scripts/mc_dropout_deep.py molformer bace 30
python scripts/build_dual_splits.py             # random vs scaffold conformal coverage

# 4) per-fold significance + figures + SI tables
python scripts/make_perfold_8model.py           # 8-model per-fold omnibus (Nadeau–Bengio, Holm)
python scripts/make_cd_crossdataset.py          # cross-dataset Demšar CD
python scripts/make_paper_figures.py            # TOC graphic + 8-model CD
python scripts/make_si.py                       # Supporting-Information tables
```

Seeds are fixed (`src/utils/seed.py`, {42, 1337, 2024}); deep models report mean ± s.d. over seeds.

## Data availability

- **ESOL, FreeSolv, Lipophilicity, BACE, QM7, QM8**: public MoleculeNet datasets. QM7/QM8 are
  fetched from the DeepChem/MoleculeNet S3 mirror and prepared by `scripts/prep_qm_datasets.py`
  (canonicalize + dedup + random 80/10/10 split).
- Cleaning is reproducible from `data/` (raw) to `data/processed/`.
- This repository is archived on Zenodo: **DOI [10.5281/zenodo.20800762](https://doi.org/10.5281/zenodo.20800762)**.
- Trained-model checkpoints are **not** required to reproduce the reported tables and figures: the
  scripts regenerate them from the released result CSVs in `results/final/`. Full retraining of the
  deep models from scratch requires GPU access.

## Reproducibility notes (honest)

- **Leakage** is audited and disclosed (canonical-SMILES duplicates, InChIKey-14, scaffold overlap,
  Tanimoto > 0.85); see SI.
- **Significance**: per-fold scaffold-grouped CV with the Nadeau–Bengio corrected resampled t-test
  (Holm). We write "comparable" when p > 0.05; the cross-dataset Demšar CD is reported as suggestive.
- **Negative control**: label permutation collapses correlation to |R| < 0.12.
- **Scope**: not every model is on every dataset for every analysis (TabPFN is out of regime on the
  large QM sets; Uni-Mol is not in the BACE per-fold omnibus) — stated in the paper.

## Release checklist

- [x] Author, ORCID, corresponding author, affiliation finalized.
- [x] Zenodo archive DOI minted (`10.5281/zenodo.20800762`) and cited in the paper.
- [x] License and data-redistribution terms set (MIT code; CC-BY-4.0 results; MoleculeNet upstream).
- [x] Public repository pushed (personal paths/keys and internal docs excluded).
- [ ] Verify reference volume/article numbers (`paper/refs.bib`) at proof (`chen2026` is ASAP).
- [ ] In the public checkout, use `README_paper.md` as `README.md`.

## License

Code: MIT (`LICENSE`). Included MoleculeNet CSVs follow their original licenses; derived result
tables are released CC-BY-4.0.
