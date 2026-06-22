"""Audit Point 4 — ligand efficiency (LE) on real BACE docking data.

A reviewer noted that the BACE docking analysis used a partial correlation to
"control for molecular size", whereas ligand efficiency (LE) is the
field-standard normalization. This script recomputes the
docking-vs-activity relationship on the EXISTING real docking results
(results/bace_docking_full.csv) using:

    - raw Pearson / Spearman of pIC50 vs Vina docking score,
    - partial correlation of pIC50 vs Vina controlling for heavy-atom count
      (the old framing, reproduced for comparison),
    - ligand efficiency LE = Vina / heavy (kcal/mol per heavy atom),
      Pearson / Spearman of pIC50 vs LE,
    - additionally Vina / MW (size normalization by molecular weight).

No docking is run here; this only post-processes existing real scores.

Every statistic, p-value and n is computed from the real CSV. Nothing is
estimated, fabricated, or hard-coded.

Output:
    results/wang_audit/bace_docking_ligand_efficiency.csv
        columns: metric, statistic, value, p_value, n
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Resolve project root from this file's location so the script is
# location-independent and reproducible from any working directory.
ROOT = Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "results" / "bace_docking_full.csv"
OUT_DIR = ROOT / "results" / "wang_audit"
OUT_CSV = OUT_DIR / "bace_docking_ligand_efficiency.csv"

REQUIRED_COLS = ("smiles", "pIC50", "heavy", "mw", "vina")


def partial_corr_xy_given_z(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[float, float, int]:
    """Pearson partial correlation of x and y controlling for z.

    Implemented via residualization: regress x on [1, z] and y on [1, z]
    by ordinary least squares, then correlate the residuals. The p-value
    uses the standard partial-correlation t-statistic with df = n - 3
    (two variables minus one controlled covariate).

    Args:
        x: First variable (1-D, length n).
        y: Second variable (1-D, length n).
        z: Controlled covariate (1-D, length n).

    Returns:
        (partial_r, p_value, n).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = x.shape[0]

    design = np.column_stack([np.ones(n, dtype=np.float64), z])
    # Residuals of x and y after removing the linear effect of z.
    beta_x, _, _, _ = np.linalg.lstsq(design, x, rcond=None)
    beta_y, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    res_x = x - design @ beta_x
    res_y = y - design @ beta_y

    partial_r, _ = stats.pearsonr(res_x, res_y)

    # Two-sided t-test for the partial correlation; df = n - 2 - k_control.
    df = n - 3
    denom = 1.0 - partial_r**2
    if df <= 0 or denom <= 0.0:
        p_value = float("nan")
    else:
        t_stat = partial_r * np.sqrt(df / denom)
        p_value = 2.0 * stats.t.sf(np.abs(t_stat), df)
    return float(partial_r), float(p_value), int(n)


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Input docking file not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing} in {IN_CSV}; "
            f"found columns {list(df.columns)}. Refusing to fabricate numbers."
        )

    # Fail-fast on any non-finite values rather than silently dropping rows.
    numeric_cols = ["pIC50", "heavy", "mw", "vina"]
    for col in numeric_cols:
        df[col] = df[col].astype(np.float64)
    if df[numeric_cols].isna().any().any():
        bad = df[numeric_cols].isna().sum().to_dict()
        raise ValueError(f"NaN values present in numeric columns: {bad}")

    n = int(len(df))
    pic50 = df["pIC50"].to_numpy(dtype=np.float64)
    vina = df["vina"].to_numpy(dtype=np.float64)
    heavy = df["heavy"].to_numpy(dtype=np.float64)
    mw = df["mw"].to_numpy(dtype=np.float64)

    # Ligand efficiency: docking score per heavy atom (kcal/mol per heavy atom).
    # More negative Vina = better binding, so LE keeps the same sign convention.
    le = vina / heavy
    # Size normalization by molecular weight (binding "efficiency" per Da).
    vina_per_mw = vina / mw

    rows: list[dict[str, object]] = []

    def add(metric: str, statistic: str, value: float, p_value: float) -> None:
        rows.append(
            {
                "metric": metric,
                "statistic": statistic,
                "value": round(float(value), 6),
                "p_value": round(float(p_value), 6),
                "n": n,
            }
        )

    # --- Raw docking vs activity ---
    r, p = stats.pearsonr(pic50, vina)
    add("pIC50_vs_vina_raw", "pearson", r, p)
    rs, ps = stats.spearmanr(pic50, vina)
    add("pIC50_vs_vina_raw", "spearman", rs, ps)

    # --- Partial correlation controlling for heavy-atom count (old framing) ---
    pr, pp, _ = partial_corr_xy_given_z(pic50, vina, heavy)
    add("pIC50_vs_vina_partial_ctrl_heavy", "pearson_partial", pr, pp)

    # --- Ligand efficiency LE = vina / heavy ---
    r_le, p_le = stats.pearsonr(pic50, le)
    add("pIC50_vs_LE_per_heavy", "pearson", r_le, p_le)
    rs_le, ps_le = stats.spearmanr(pic50, le)
    add("pIC50_vs_LE_per_heavy", "spearman", rs_le, ps_le)

    # --- Size normalization by MW (vina / mw) ---
    r_mw, p_mw = stats.pearsonr(pic50, vina_per_mw)
    add("pIC50_vs_vina_per_mw", "pearson", r_mw, p_mw)
    rs_mw, ps_mw = stats.spearmanr(pic50, vina_per_mw)
    add("pIC50_vs_vina_per_mw", "spearman", rs_mw, ps_mw)

    out_df = pd.DataFrame(rows, columns=["metric", "statistic", "value", "p_value", "n"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    # Console summary for the audit log.
    print(f"Loaded {IN_CSV} (n={n} real docked ligands)")
    print(out_df.to_string(index=False))
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
