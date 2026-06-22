"""B / M2+M3 fix: per-fold significance for the CPU-tier BACE models, replacing the
statistically-invalid per-MOLECULE Friedman in make_significance.py.

PRE-REGISTRATION (declared before any confirmatory run on real labels; CLAUDE.md §16.9):
  * Hypothesis: zero-tuning TabPFN (RDKit descriptors) is NOT worse than Optuna-tuned
    classic QSAR (Ridge/SVR/GBM/RF, Morgan-2048) in per-fold MAE under scaffold-grouped CV.
  * Replication unit: a held-out CV fold on the dev pool (train+val, n=1201). The 303
    predefined-test molecules are NEVER used as test blocks (that was the M3 error).
  * PRIMARY regime: scaffold-grouped CV (GroupKFold-10 x 5 repeats = 50 folds) -- matches
    the headline "robust under scaffold extrapolation" claim.
  * SECONDARY (exploratory): random RepeatedKFold(5 x 10 = 50) -- interpolation context.
  * Primary metric: MAE. Confirmatory family (the ONLY one): the 4 Nadeau-Bengio (2003)
    corrected resampled paired t-tests TabPFN-vs-{Ridge,SVR,GBM,RF} on the PRIMARY run,
    Holm-Bonferroni at FWER 0.05. Everything else is EXPLORATORY/DESCRIPTIVE:
    the omnibus Friedman, the Nemenyi CD diagram, the secondary run, the second-rho
    sensitivity, the permutation cross-check, the TabPFN-Morgan control arm.
  * M2 (repeatability symmetry): all CPU models draw variance from the SAME axis (the folds);
    Ridge/SVR/TabPFN are deterministic (fold refit = the replicate); RF/GBM average an inner
    3-seed loop per fold so their algorithmic randomness is folded in, not pinned to one seed.
  * Deep models (MolFormer/ChemFM/Chemprop) are GPU-bound -> held OUT of the omnibus; reported
    only in the descriptive fixed-test panel (build_fixed_test_panel). The frozen fold manifest
    makes their per-fold promotion a drop-in later.

Refs: Demsar 2006 (JMLR 7:1-30); Nadeau & Bengio 2003 (Mach Learn 52:239); Bouckaert & Frank
2004; Bengio & Grandvalet 2004 (no unbiased CV-variance estimator). The NB correction is the
literature-standard, *approximate* fix for the train-set overlap that makes a naive paired
t-test over CV folds anti-conservative.

Run (TabPFN uses GPU; classic use CPU; activate the `drug` env first):
  CUDA_VISIBLE_DEVICES=7 SCIPY_ARRAY_API=1 \
    python scripts/perfold_significance.py --smoke   # quick
  CUDA_VISIBLE_DEVICES=7 SCIPY_ARRAY_API=1 \
    python scripts/perfold_significance.py           # full
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")  # must precede tabpfn import

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, friedmanchisquare, rankdata, t as t_dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.metrics import regression_metrics  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402

warnings.filterwarnings("ignore")
DATA = os.environ.get("BACE_DATA", "data/processed/bace_clean.csv")
RAW = Path("results/raw"); RAW.mkdir(parents=True, exist_ok=True)
FINAL = Path("results/final"); FINAL.mkdir(parents=True, exist_ok=True)

INNER_SEEDS = (42, 1337, 2024)          # RF/GBM inner average (M2 symmetry)
PRIMARY_REPEAT_SEEDS = (42, 43, 44, 45, 46)
N_SPLITS_PRIMARY = 10
N_SPLITS_SECONDARY, N_REPEATS_SECONDARY = 5, 10
CONTROL = "TabPFN"                       # pre-registered control model
CLASSIC = ["Ridge", "SVR", "GBM", "RF"]  # the confirmatory family (vs TabPFN)

# Frozen Optuna hyperparameters (results/tuning/summary.json), declared before any test look.
HP = {
    "Ridge": {"alpha": 656.4817611753456},
    "SVR": {"C": 329.9557544633786, "gamma": 0.00022839379946875313, "epsilon": 0.23622748765680157},
    "GBM": {"learning_rate": 0.14887349914861522, "max_iter": 800, "max_depth": 6,
            "min_samples_leaf": 8, "l2_regularization": 8.238714922948002},
    "RF": {"n_estimators": 500, "max_depth": None, "max_features": 0.3, "min_samples_leaf": 1},
}


# --------------------------------------------------------------------------- features
def _mols(smiles):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    return [Chem.MolFromSmiles(s) for s in smiles]


def featurize(smiles, rep):
    from rdkit.Chem import AllChem, Descriptors
    mols = _mols(smiles)
    if rep == "morgan2048":
        gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
        return np.array([gen.GetFingerprint(m) for m in mols], dtype=np.float64)
    if rep == "morgan512":
        gen = AllChem.GetMorganGenerator(radius=2, fpSize=512)
        return np.array([gen.GetFingerprint(m) for m in mols], dtype=np.float64)
    if rep == "desc":
        names = [n for n, _ in Descriptors.descList]
        X = np.full((len(mols), len(names)), np.nan, dtype=np.float64)
        for i, m in enumerate(mols):
            for j, (_, fn) in enumerate(Descriptors.descList):
                try:
                    X[i, j] = fn(m)
                except Exception:
                    X[i, j] = np.nan
        X[~np.isfinite(X)] = np.nan
        return X
    raise ValueError(rep)


def murcko_groups(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
        except Exception:
            sc = ""
        # Trigger: acyclic molecule -> empty Murcko scaffold.
        # Why:     lumping all acyclic mols into one group would dump them in one fold.
        # Outcome: give each its own singleton group (its SMILES) so folds stay balanced.
        out.append(sc if sc else f"acyclic::{s}")
    return np.array(out, dtype=object)


# --------------------------------------------------------------------------- models
def make_model(name, seed=42):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    if name == "Ridge":
        return Pipeline([("s", StandardScaler()), ("m", Ridge(**HP["Ridge"]))])
    if name == "SVR":
        return Pipeline([("s", StandardScaler()), ("m", SVR(kernel="rbf", **HP["SVR"]))])
    if name == "GBM":
        return HistGradientBoostingRegressor(random_state=seed, early_stopping=True, **HP["GBM"])
    if name == "RF":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **HP["RF"])
    raise ValueError(name)


_TABPFN = {}


def tabpfn_predict(Xtr, ytr, Xte, rep):
    """Cached-import TabPFN regressor; GPU if available (server), else CPU."""
    if "reg" not in _TABPFN:
        import sklearn.utils.validation as skv
        if not hasattr(skv, "_is_pandas_df"):
            skv._is_pandas_df = lambda x: isinstance(x, pd.DataFrame)  # sklearn>=1.8 shim
        import torch
        from tabpfn import TabPFNRegressor
        _TABPFN["cls"] = TabPFNRegressor
        _TABPFN["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    reg = _TABPFN["cls"](device=_TABPFN["device"], random_state=42, ignore_pretraining_limits=True)
    reg.fit(Xtr, ytr)
    return np.asarray(reg.predict(Xte), dtype=float)


def fold_mae(name, feats, y, tr, te):
    """Return held-out-fold MAE for one model; RF/GBM averaged over INNER_SEEDS (M2)."""
    if name in ("TabPFN", "TabPFN-Morgan"):
        rep = "desc" if name == "TabPFN" else "morgan512"
        X = feats[rep]
        pred = tabpfn_predict(X[tr], y[tr], X[te], rep)
        return float(np.mean(np.abs(pred - y[te])))
    X = feats["morgan2048"]
    if name in ("RF", "GBM"):
        maes = []
        for s in INNER_SEEDS:
            mdl = make_model(name, seed=s).fit(X[tr], y[tr])
            maes.append(np.mean(np.abs(mdl.predict(X[te]) - y[te])))
        return float(np.mean(maes))
    mdl = make_model(name).fit(X[tr], y[tr])           # Ridge/SVR deterministic
    return float(np.mean(np.abs(mdl.predict(X[te]) - y[te])))


# --------------------------------------------------------------------------- CV splits
def primary_splits(groups, n_splits, repeat_seeds):
    from sklearn.model_selection import GroupKFold
    uniq = sorted(set(groups.tolist()))
    for rep_id, seed in enumerate(repeat_seeds):
        perm = np.random.default_rng(seed).permutation(len(uniq))
        gid_map = {g: int(perm[i]) for i, g in enumerate(uniq)}
        gids = np.array([gid_map[g] for g in groups])
        for fold_id, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(gids, groups=gids)):
            yield rep_id, fold_id, tr, te


def secondary_splits(n, n_splits, n_repeats, seed=42):
    from sklearn.model_selection import RepeatedKFold
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for k, (tr, te) in enumerate(rkf.split(np.arange(n))):
        yield k // n_splits, k % n_splits, tr, te


# --------------------------------------------------------------------------- run CV
def run_cv(regime, splits, feats, y, models, smiles=None, audit=False):
    rows, leak = [], []
    for rep_id, fold_id, tr, te in splits:
        for name in models:
            rows.append({"regime": regime, "repeat": rep_id, "fold": fold_id, "model": name,
                         "n_train": len(tr), "n_test": len(te), "MAE": fold_mae(name, feats, y, tr, te)})
        if audit and smiles is not None:
            g = murcko_groups(smiles)
            overlap = len(set(g[tr]) & set(g[te]))
            leak.append({"regime": regime, "repeat": rep_id, "fold": fold_id,
                         "scaffold_overlap": overlap, "n_test_scaffolds": len(set(g[te]))})
        print(f"  [{regime}] r{rep_id} f{fold_id} done", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(leak)


# --------------------------------------------------------------------------- statistics
def _pivot(df):
    """folds (rows) x model (cols) MAE matrix, paired by (repeat,fold)."""
    return df.pivot_table(index=["repeat", "fold"], columns="model", values="MAE")


def omnibus(M, models):
    cols = [M[m].to_numpy(float) for m in models]
    chi, p = friedmanchisquare(*cols)
    N, k = M.shape[0], len(models)
    denom = N * (k - 1) - chi
    F = ((N - 1) * chi) / denom if denom > 0 else np.inf
    p_id = 0.0 if not np.isfinite(F) else float(f_dist.sf(F, k - 1, (k - 1) * (N - 1)))
    ranks = np.apply_along_axis(rankdata, 1, M[models].to_numpy(float))  # rank within fold
    return {"friedman_chi2": float(chi), "friedman_p": float(p), "iman_davenport_F": float(F),
            "iman_davenport_p": p_id, "N_folds": int(N), "avg_rank": dict(zip(models, ranks.mean(0)))}


def nb_ttest(d, rho):
    """Nadeau-Bengio corrected resampled paired t-test on per-fold differences d."""
    N = len(d); var = float(np.var(d, ddof=1)); mean = float(np.mean(d))
    se = np.sqrt((1.0 / N + rho / (1.0 - rho)) * var)
    tval = mean / se if se > 0 else 0.0
    p = float(2 * t_dist.sf(abs(tval), df=N - 1))
    return mean, tval, p


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * pvals[idx]); adj[idx] = min(run, 1.0)
    return adj


def perm_p(d, n=10000, seed=0):
    rng = np.random.default_rng(seed); obs = abs(np.mean(d))
    cnt = sum(abs(np.mean(d * rng.choice([1, -1], size=len(d)))) >= obs for _ in range(n))
    return (cnt + 1) / (n + 1)


def confirmatory(M, rho_primary=1.0 / 9, rho_alt=0.25):
    rows = []
    for m in CLASSIC:
        d = (M[m] - M[CONTROL]).to_numpy(float)  # >0 => classic worse than TabPFN
        mean, tval, p = nb_ttest(d, rho_primary)
        _, _, p_alt = nb_ttest(d, rho_alt)
        rbc = float(np.mean(np.sign(d)))  # rank-biserial-style: +1 = always worse than TabPFN
        rows.append({"model": m, "dMAE_vs_TabPFN": round(mean, 4), "NB_t": round(tval, 3),
                     "NB_p_rho.111": round(p, 4), "NB_p_rho.25": round(p_alt, 4),
                     "perm_p": round(perm_p(d), 4), "frac_folds_worse": round((d > 0).mean(), 3),
                     "effect_sign": round(rbc, 3)})
    df = pd.DataFrame(rows)
    df["NB_holm_rho.111"] = np.round(holm(df["NB_p_rho.111"].to_numpy(float)), 4)
    df["NB_bonf_rho.111"] = np.round(np.minimum(df["NB_p_rho.111"] * len(CLASSIC), 1.0), 4)
    return df


def noise_floor(df_full):
    """In-dataset replicate spread (InChIKey14 multi-label) as a sigma proxy."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    ik = {}
    for s, lab in zip(df_full.smiles, df_full.label):
        m = Chem.MolFromSmiles(s)
        if m:
            ik.setdefault(Chem.MolToInchiKey(m)[:14], []).append(float(lab))
    spreads = [np.std(v, ddof=1) for v in ik.values() if len(v) > 1]
    return float(np.mean(spreads)) if spreads else float("nan"), len(spreads)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 repeat, classic+TabPFN only (pipeline check)")
    ap.add_argument("--negctrl", action="store_true", help="also run permuted-label negative control")
    args = ap.parse_args()
    set_all_seeds(42)

    full = pd.read_csv(DATA)
    dev = full[full.set.isin(["train", "validation"])].reset_index(drop=True)
    smiles = dev.smiles.tolist(); y = dev.label.to_numpy(float)
    print(f"dev pool n={len(dev)} (train+val); test held out for descriptive panel only")
    feats = {r: featurize(smiles, r) for r in ("morgan2048", "morgan512", "desc")}
    groups = murcko_groups(np.array(smiles, dtype=object))
    print(f"unique Murcko scaffolds in dev: {len(set(groups.tolist()))}")

    models = [*CLASSIC, CONTROL, "TabPFN-Morgan"]
    rep_seeds = PRIMARY_REPEAT_SEEDS[:1] if args.smoke else PRIMARY_REPEAT_SEEDS

    # PRIMARY (scaffold-grouped)
    prim = list(primary_splits(groups, N_SPLITS_PRIMARY, rep_seeds))
    dfp, leakp = run_cv("primary", prim, feats, y, models, smiles, audit=True)
    dfp.to_csv(RAW / "perfold_mae_primary.csv", index=False)
    leakp.to_csv(RAW / "perfold_leakage_primary.csv", index=False)
    Mp = _pivot(dfp)
    omni_p = omnibus(Mp, [*CLASSIC, CONTROL])
    conf = confirmatory(Mp)
    conf.to_csv(FINAL / "perfold_significance.csv", index=False)

    # SECONDARY (random) -- exploratory
    if not args.smoke:
        sec = list(secondary_splits(len(dev), N_SPLITS_SECONDARY, N_REPEATS_SECONDARY))
        dfs, leaks = run_cv("secondary", sec, feats, y, models, smiles, audit=True)
        dfs.to_csv(RAW / "perfold_mae_secondary.csv", index=False)
        leaks.to_csv(RAW / "perfold_leakage_secondary.csv", index=False)
        omni_s = omnibus(_pivot(dfs), [*CLASSIC, CONTROL])
    else:
        omni_s = None

    sigma, n_pairs = noise_floor(full)

    # negative control on PRIMARY (permuted dev labels): TabPFN + RF must collapse
    nc = {}
    if args.negctrl:
        yp = np.random.default_rng(0).permutation(y)
        dfn, _ = run_cv("negctrl", prim, feats, yp, ["RF", CONTROL])
        Mn = _pivot(dfn)
        base = float(np.mean(np.abs(yp - yp.mean())))  # MAE of predicting the mean
        nc = {"RF_mae": float(Mn["RF"].mean()), "TabPFN_mae": float(Mn[CONTROL].mean()),
              "mean_predictor_mae": base}

    # ---- report ----
    print("\n================ PRIMARY (scaffold-grouped CV) ================")
    print(f"omnibus (5 CPU models, N={omni_p['N_folds']} folds): Friedman chi2={omni_p['friedman_chi2']:.1f} "
          f"p={omni_p['friedman_p']:.2e} | Iman-Davenport p={omni_p['iman_davenport_p']:.2e}  [DESCRIPTIVE]")
    print("avg rank:", {k: round(v, 2) for k, v in omni_p["avg_rank"].items()})
    print("\nCONFIRMATORY  Nadeau-Bengio corrected t  (dMAE>0 => worse than TabPFN):")
    print(conf.to_string(index=False))
    print(f"\nper-fold mean MAE: " + ", ".join(f"{m}={Mp[m].mean():.3f}" for m in models))
    if omni_s:
        print(f"\nSECONDARY (random CV) omnibus [exploratory]: Iman-Davenport p={omni_s['iman_davenport_p']:.2e}")
    if sigma == sigma:
        print(f"\nlabel heterogeneity (InChIKey14 near-skeleton spread, n_pairs={n_pairs}): sigma={sigma:.3f} "
              f"-- UPPER bound incl. real stereo/tautomer SAR, NOT a clean assay floor (the cleaned set has no "
              f"exact replicates); for reference TabPFN per-fold MAE/this = {Mp[CONTROL].mean()/sigma:.2f}. "
              f"We do NOT claim to be at the assay noise floor.")
    else:
        print("\nlabel-heterogeneity sigma n/a")
    print(f"scaffold-fold leakage (primary): mean overlap={leakp.scaffold_overlap.mean():.2f} (target 0)")
    if nc:
        print(f"negative control (permuted labels): RF MAE={nc['RF_mae']:.3f} TabPFN MAE={nc['TabPFN_mae']:.3f} "
              f"vs mean-predictor {nc['mean_predictor_mae']:.3f} (should be ~equal)")

    pd.DataFrame([{**{"regime": "primary", **{f"avg_rank_{k}": v for k, v in omni_p["avg_rank"].items()}},
                   "friedman_p": omni_p["friedman_p"], "iman_davenport_p": omni_p["iman_davenport_p"],
                   "noise_sigma": sigma, "tabpfn_mae_over_sigma": Mp[CONTROL].mean() / sigma if sigma == sigma else None,
                   "scaffold_leak_mean": float(leakp.scaffold_overlap.mean()), **nc}]).to_csv(
        FINAL / "perfold_omnibus_summary.csv", index=False)
    print("\nwrote results/final/perfold_significance.csv + perfold_omnibus_summary.csv + results/raw/perfold_*.csv")


if __name__ == "__main__":
    main()
