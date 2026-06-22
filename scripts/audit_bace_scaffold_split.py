"""Wang-audit Point 3: BACE scaffold-split reality check.

A critical reviewer noted that the BACE regression Pearson R (~0.804) from the
project RF + Morgan baseline may be inflated by train/test similarity in the
*predefined* MoleculeNet split, which reportedly shares ~63.7% of test Murcko
scaffolds with the training set. This script QUANTIFIES how much the score
drops when the SAME molecules are re-partitioned by Bemis-Murcko scaffold so
that NO scaffold is shared between train and test (a clean leakage-free split).

What it does (everything on REAL data, no fabricated numbers):
  1. Load data/BACE.split.csv (READ-ONLY). Target = `label` (pIC50 regression),
     SMILES = `smiles`, predefined split = `set` (train/validation/test).
  2. PREDEFINED split: featurize with Morgan FP (radius=2, fpSize=2048, via
     AllChem.GetMorganGenerator), train RandomForestRegressor on `set==train`,
     evaluate on `set==test`. 3 seeds (42, 1337, 2024). Report Pearson R + MAE.
     (Sanity: should land near the project's reported R ~= 0.804.)
  3. SCAFFOLD split: group ALL molecules by MurckoScaffold.MurckoScaffoldSmiles,
     assign whole scaffold groups to train vs test (no scaffold shared), target
     test fraction ~= the predefined split's. Train the SAME RF config; evaluate
     on the held-out scaffold test set. 3 seeds. Report Pearson R + MAE.
  4. Compute test->train Murcko-scaffold overlap fraction for BOTH splits
     (predefined should be high, ~the reported 63.7%; scaffold should be ~0%).
  5. Write results/wang_audit/bace_split_comparison.csv.

Conventions (CLAUDE.md): Morgan radius=2 fpSize=2048; RF n_estimators=500,
max_depth=None (matches src/models/baseline_rf.py + configs/model/rf.yaml);
seeds 42/1337/2024; explicit numpy dtypes.

Run:
    python scripts/audit_bace_scaffold_split.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Silence RDKit's noisy C++ parse warnings; we handle failures explicitly below.
RDLogger.DisableLog("rdApp.*")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("bace_audit")

# --- Project-locked constants -------------------------------------------------
SEEDS: tuple[int, ...] = (42, 1337, 2024)
MORGAN_RADIUS: int = 2
MORGAN_NBITS: int = 2048
RF_N_ESTIMATORS: int = 500  # matches configs/model/rf.yaml
RF_MAX_DEPTH: int | None = None  # unlimited, matches configs/model/rf.yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "BACE.split.csv"
OUT_DIR = REPO_ROOT / "results" / "wang_audit"
OUT_CSV = OUT_DIR / "bace_split_comparison.csv"

SMILES_COL = "smiles"
TARGET_COL = "label"  # pIC50 regression target (NOT a binary Class label)
SPLIT_COL = "set"


# --- Featurization (mirrors src/data/featurizers.py exactly) ------------------
def smiles_to_morgan(
    smi: str, radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_NBITS
) -> NDArray[np.int8] | None:
    """Morgan fingerprint, defaults per CLAUDE.md S9. None if unparseable."""
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


def featurize_frame(
    df: pd.DataFrame,
) -> tuple[NDArray[np.int8], NDArray[np.float64], int]:
    """Featurize a dataframe; drop (and log) unparseable SMILES.

    Returns:
        (X int8 [n_kept, n_bits], y float64 [n_kept], n_failed).
    """
    fps: list[NDArray[np.int8]] = []
    ys: list[float] = []
    n_failed = 0
    for smi, y in zip(df[SMILES_COL].tolist(), df[TARGET_COL].tolist()):
        fp = smiles_to_morgan(smi)
        if fp is None:
            n_failed += 1
            logger.warning("Failed to featurize SMILES: %r", smi)
            continue
        fps.append(fp)
        ys.append(float(y))
    X = np.vstack(fps).astype(np.int8) if fps else np.zeros(
        (0, MORGAN_NBITS), dtype=np.int8
    )
    y = np.asarray(ys, dtype=np.float64)
    return X, y, n_failed


# --- Scaffold utilities -------------------------------------------------------
def murcko_scaffold(smi: str) -> str | None:
    """Bemis-Murcko scaffold SMILES, or None if SMILES is unparseable.

    An acyclic molecule yields an empty-string scaffold; we keep that as a
    distinct group key (it groups all acyclic molecules together).
    """
    if not isinstance(smi, str) or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # includeChirality=False: scaffold identity is a 2D ring-system concept;
    # BACE is 71% stereo-undefined (CLAUDE.md S11), so chirality-aware scaffolds
    # would fragment groups spuriously.
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_overlap_pct(
    test_smiles: list[str], train_smiles: list[str]
) -> float:
    """Fraction (%) of TEST molecules whose Murcko scaffold also appears in TRAIN.

    Molecule-level (not unique-scaffold-level): this is the leakage metric the
    reviewer cited (~63.7% of the test SET shares a scaffold with train).
    """
    train_scaffs: set[str] = set()
    for s in train_smiles:
        sc = murcko_scaffold(s)
        if sc is not None:
            train_scaffs.add(sc)
    n_total = 0
    n_overlap = 0
    for s in test_smiles:
        sc = murcko_scaffold(s)
        if sc is None:
            continue
        n_total += 1
        if sc in train_scaffs:
            n_overlap += 1
    if n_total == 0:
        return float("nan")
    return 100.0 * n_overlap / n_total


def make_scaffold_split(
    df: pd.DataFrame, test_frac: float, seed: int = 42
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Deterministic Bemis-Murcko scaffold split (no scaffold shared).

    Groups whole scaffold sets into test until the target test fraction is
    reached. To make the split clean and reproducible we order scaffold groups
    by descending size (the standard MoleculeNet/DeepChem scaffold-split
    convention: largest groups go to train first, smaller groups fill test),
    then place the smallest groups into the test bucket. This produces a fixed
    partition independent of `seed`; `seed` is retained in the signature so the
    caller can document that the SPLIT itself does not depend on the RF seed
    (only the RF model does).

    Returns:
        (train_mask, test_mask) boolean arrays aligned to df rows.
    """
    del seed  # split is deterministic; RF seed varies the model, not the split.
    scaffolds: list[str] = []
    for s in df[SMILES_COL].tolist():
        sc = murcko_scaffold(s)
        # Unparseable -> unique sentinel so it never silently merges groups.
        scaffolds.append(sc if sc is not None else f"__UNPARSEABLE__{len(scaffolds)}")

    # group row indices by scaffold
    groups: dict[str, list[int]] = {}
    for idx, sc in enumerate(scaffolds):
        groups.setdefault(sc, []).append(idx)

    # DeepChem convention: sort groups by size descending (ties broken by
    # scaffold string for determinism); fill train first, overflow to test.
    ordered = sorted(groups.values(), key=lambda g: (-len(g), scaffolds[g[0]]))

    n_total = len(df)
    n_test_target = int(round(test_frac * n_total))
    n_train_cap = n_total - n_test_target

    train_idx: list[int] = []
    test_idx: list[int] = []
    for g in ordered:
        if len(train_idx) + len(g) <= n_train_cap:
            train_idx.extend(g)
        else:
            test_idx.extend(g)

    train_mask = np.zeros(n_total, dtype=np.bool_)
    test_mask = np.zeros(n_total, dtype=np.bool_)
    train_mask[np.asarray(train_idx, dtype=np.int64)] = True
    test_mask[np.asarray(test_idx, dtype=np.int64)] = True
    return train_mask, test_mask


# --- RF train/eval ------------------------------------------------------------
def train_eval_rf(
    X_tr: NDArray[np.int8],
    y_tr: NDArray[np.float64],
    X_te: NDArray[np.int8],
    y_te: NDArray[np.float64],
    seed: int,
) -> tuple[float, float]:
    """Fit RF (project config) on train, return (Pearson R, MAE) on test."""
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te).astype(np.float64)
    r = float(pearsonr(pred, y_te)[0])
    mae = float(mean_absolute_error(y_te, pred))
    return r, mae


def run_split(
    name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> dict[str, float]:
    """Featurize train/test, run 3 seeds, return aggregated metrics + overlap."""
    X_tr, y_tr, fail_tr = featurize_frame(df_train)
    X_te, y_te, fail_te = featurize_frame(df_test)
    logger.info(
        "[%s] n_train=%d (failed=%d)  n_test=%d (failed=%d)",
        name,
        X_tr.shape[0],
        fail_tr,
        X_te.shape[0],
        fail_te,
    )

    rs: list[float] = []
    maes: list[float] = []
    for seed in SEEDS:
        r, mae = train_eval_rf(X_tr, y_tr, X_te, y_te, seed)
        logger.info("[%s] seed=%d  Pearson R=%.4f  MAE=%.4f", name, seed, r, mae)
        rs.append(r)
        maes.append(mae)

    overlap = scaffold_overlap_pct(
        df_test[SMILES_COL].tolist(), df_train[SMILES_COL].tolist()
    )
    logger.info("[%s] test->train Murcko-scaffold overlap = %.2f%%", name, overlap)

    return {
        "split_type": name,
        "pearson_r_mean": float(np.mean(rs)),
        "pearson_r_std": float(np.std(rs, ddof=0)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes, ddof=0)),
        "scaffold_overlap_pct": float(overlap),
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %s  shape=%s  columns=%s", DATA_PATH, df.shape, list(df.columns))
    if not {SMILES_COL, TARGET_COL, SPLIT_COL}.issubset(df.columns):
        raise ValueError(
            f"Expected columns {{{SMILES_COL},{TARGET_COL},{SPLIT_COL}}}, "
            f"got {list(df.columns)}"
        )
    logger.info("Split value counts:\n%s", df[SPLIT_COL].value_counts().to_string())
    logger.info(
        "Target `%s` stats: min=%.3f max=%.3f mean=%.3f n_unique=%d",
        TARGET_COL,
        df[TARGET_COL].min(),
        df[TARGET_COL].max(),
        df[TARGET_COL].mean(),
        df[TARGET_COL].nunique(),
    )

    # --- (1) PREDEFINED split: train on `train`, test on `test`.
    # Validation rows are excluded from training to match the project's
    # train/test usage (RF needs no early stopping; valid is held out, not
    # folded into train). This keeps the comparison apples-to-apples with the
    # reported baseline.
    df_pre_train = df[df[SPLIT_COL] == "train"].reset_index(drop=True)
    df_pre_test = df[df[SPLIT_COL] == "test"].reset_index(drop=True)
    predefined_test_frac = len(df_pre_test) / len(df)
    logger.info("Predefined test fraction = %.4f", predefined_test_frac)
    res_predefined = run_split("predefined", df_pre_train, df_pre_test)

    # --- (2) SCAFFOLD split on the SAME molecules (all rows pooled), matching
    # the predefined test fraction so n_test is comparable.
    train_mask, test_mask = make_scaffold_split(
        df, test_frac=predefined_test_frac, seed=42
    )
    df_sc_train = df[train_mask].reset_index(drop=True)
    df_sc_test = df[test_mask].reset_index(drop=True)
    logger.info(
        "Scaffold split: n_train=%d n_test=%d (test_frac=%.4f)",
        len(df_sc_train),
        len(df_sc_test),
        len(df_sc_test) / len(df),
    )
    res_scaffold = run_split("scaffold", df_sc_train, df_sc_test)

    out = pd.DataFrame([res_predefined, res_scaffold])
    out = out[
        [
            "split_type",
            "pearson_r_mean",
            "pearson_r_std",
            "mae_mean",
            "mae_std",
            "scaffold_overlap_pct",
            "n_train",
            "n_test",
        ]
    ]
    out.to_csv(OUT_CSV, index=False)
    logger.info("Wrote %s", OUT_CSV)
    print("\n===== BACE split comparison (RF + Morgan r2/2048, 3 seeds) =====")
    print(out.to_string(index=False))

    dr = res_predefined["pearson_r_mean"] - res_scaffold["pearson_r_mean"]
    print(
        f"\nPearson R drop (predefined -> scaffold): "
        f"{res_predefined['pearson_r_mean']:.3f} -> "
        f"{res_scaffold['pearson_r_mean']:.3f}  "
        f"(Delta = -{dr:.3f})"
    )


if __name__ == "__main__":
    main()
