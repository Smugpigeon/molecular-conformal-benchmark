"""GAP4: dual split (random vs scaffold) for the conformal exchangeability story.

Split-conformal's marginal coverage guarantee assumes exchangeability, which a RANDOM split
respects but a SCAFFOLD (out-of-domain) split breaks by design. This builds both 80/10/10 splits
per dataset (saved as results/derived_splits/<name>_{random,scaffold}.csv for the full multi-model
re-train) and runs a quick RF split-conformal coverage demo: coverage ~nominal under random,
degraded under scaffold -- the headline GAP4 evidence.

Run: /opt/anaconda3/bin/python3 scripts/build_dual_splits.py [dataset ...]
     (default: the 6 regression datasets; locally only those whose CSV exists will run)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loaders import load_dataset  # noqa: E402
from src.eval.conformal import split_conformal  # noqa: E402  (finite-sample quantile, method="higher")

# Derived splits are a RESULT artifact, not source data: keep them out of the read-only data/ tree.
OUT = Path("results/derived_splits"); OUT.mkdir(parents=True, exist_ok=True)
FINAL = Path("results/final"); FINAL.mkdir(parents=True, exist_ok=True)
ALPHA = 0.1


def _all_mols(name):
    sp = load_dataset(name)
    df = pd.concat([sp.train, sp.val, sp.test], ignore_index=True)[["smiles", "label"]]
    return df.reset_index(drop=True)


def _murcko(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out = []
    for i, s in enumerate(smiles):
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s))
        except Exception:
            sc = ""
        out.append(sc if sc else f"none{i}")
    return out


def _assign(n, idx_groups):
    """Greedy 80/10/10 by molecule count over groups (largest first -> test gets rarer scaffolds)."""
    tr, va, te = [], [], []
    for g in idx_groups:
        if len(tr) + len(g) <= 0.8 * n:
            tr += g
        elif len(va) + len(g) <= 0.1 * n:
            va += g
        else:
            te += g
    return tr, va, te


def make_splits(df):
    n = len(df)
    # random
    perm = np.random.default_rng(42).permutation(n)
    ntr, nva = int(0.8 * n), int(0.1 * n)
    rnd = np.array(["train"] * n, dtype=object)
    rnd[perm[ntr:ntr + nva]] = "validation"; rnd[perm[ntr + nva:]] = "test"
    # scaffold (groups largest-first)
    scafs = {}
    for i, sc in enumerate(_murcko(df.smiles.tolist())):
        scafs.setdefault(sc, []).append(i)
    groups = sorted(scafs.values(), key=len, reverse=True)
    tr, va, te = _assign(n, groups)
    scf = np.empty(n, dtype=object)
    scf[tr] = "train"; scf[va] = "validation"; scf[te] = "test"
    return rnd, scf, len(scafs)


def rf_conformal(df, setcol):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator as rfg
    from sklearn.ensemble import RandomForestRegressor
    RDLogger.DisableLog("rdApp.*")
    gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
    X = np.array([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in df.smiles], dtype=np.float64)
    y = df.label.to_numpy(np.float64)
    tr, va, te = setcol == "train", setcol == "validation", setcol == "test"
    rf = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42).fit(X[tr], y[tr])
    # Use the canonical finite-sample split conformal (method="higher") so coverage matches the
    # SI's finite-sample-correction claim and the main conformal_full results.
    pred_te = rf.predict(X[te])
    lo, hi, q = split_conformal(y[va], rf.predict(X[va]), pred_te, ALPHA)
    cov = float(((y[te] >= lo) & (y[te] <= hi)).mean())
    return cov, float(2 * q), int(te.sum())


def main():
    datasets = sys.argv[1:] or ["esol", "freesolv", "lipophilicity", "bace", "qm7", "qm8"]
    rows = []
    for name in datasets:
        try:
            df = _all_mols(name)
        except Exception as e:
            print(f"  [skip {name}] {e}"); continue
        rnd, scf, nscaf = make_splits(df)
        for tag, col in [("random", rnd), ("scaffold", scf)]:
            d2 = df.copy(); d2["set"] = col
            d2.to_csv(OUT / f"{name}_{tag}.csv", index=False)
        cov_r, w_r, nte_r = rf_conformal(df, rnd)
        cov_s, w_s, nte_s = rf_conformal(df, scf)
        rows.append({"dataset": name, "n": len(df), "n_scaffolds": nscaf,
                     "cov_random": round(cov_r, 3), "width_random": round(w_r, 3),
                     "cov_scaffold": round(cov_s, 3), "width_scaffold": round(w_s, 3),
                     "coverage_drop": round(cov_r - cov_s, 3)})
        print(f"  {name:14s} RF split-conformal | random cov={cov_r:.3f} | scaffold cov={cov_s:.3f} "
              f"| drop={cov_r - cov_s:+.3f}  (n_scaffolds={nscaf})")
    if rows:
        pd.DataFrame(rows).to_csv(FINAL / "gap4_dual_split.csv", index=False)
        print(f"\n  -> {FINAL / 'gap4_dual_split.csv'}  + results/derived_splits/<name>_{{random,scaffold}}.csv")
        print("  KEY: random coverage ~0.90 (exchangeable, valid); scaffold coverage drops (OOD breaks the guarantee).")


if __name__ == "__main__":
    main()
