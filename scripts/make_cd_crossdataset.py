"""Demsar (2006) cross-DATASET critical-difference diagram -- the statistically VALID
use of Friedman + Nemenyi (datasets as blocks, k models, N>=5 datasets). This is the
proper headline multi-model comparison for the paper, distinct from the single-dataset
per-fold CD (which is descriptive). Ranks models across the regression datasets by:
  (a) conformal interval WIDTH (tighter = better, at fixed valid coverage)
  (b) point-prediction MAE
Only complete blocks (models present on ALL datasets) enter each diagram.

Consumes results/final/conformal_full.csv (dataset, model, mean_width, empirical_coverage, test_mae).
Run after the conformal sweep. Output: results/figures/cd_crossdataset_{width,mae}.png +
results/final/cd_crossdataset_ranks.csv.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import friedmanchisquare, rankdata  # noqa: E402

FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
FINAL = Path("results/final"); FINAL.mkdir(parents=True, exist_ok=True)
Q05 = {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850}  # Nemenyi q_0.05
MLAB = {"rf": "RF+Morgan", "molformer": "MolFormer-XL", "chemfm": "ChemFM-3B",
        "unimol": "Uni-Mol", "tabpfn": "TabPFN"}


def _cd_diagram(mat: pd.DataFrame, title: str, fname: str) -> dict:
    """mat: rows=datasets, cols=models, values=metric (lower=better). Complete blocks only."""
    mat = mat.dropna(axis=1)  # keep only models present on every dataset
    models = list(mat.columns)
    N, k = mat.shape[0], len(models)
    if k < 2 or N < 3:
        print(f"  [skip {fname}] need >=2 models x >=3 datasets, have {k}x{N}")
        return {}
    ranks = np.apply_along_axis(rankdata, 1, mat.to_numpy(float))  # rank within dataset, 1=best
    avg = ranks.mean(0)
    chi, p = friedmanchisquare(*[mat[m].to_numpy(float) for m in models])
    CD = Q05.get(k, 2.85) * np.sqrt(k * (k + 1) / (6 * N))
    order = np.argsort(avg)

    fig, ax = plt.subplots(figsize=(9.0, 2.4 + 0.32 * k))
    ax.set_xlim(0.7, k + 0.3); ax.set_ylim(-0.6 * k - 1, 1.9); ax.axis("off")
    ax.plot([1, k], [0, 0], "k", lw=1)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.12], "k", lw=1); ax.text(r, 0.42, str(r), ha="center", fontsize=10)
    ax.plot([1, 1 + CD], [1.25, 1.25], "k", lw=2.5)
    ax.text(1 + CD / 2, 1.5, f"CD={CD:.2f}", ha="center", fontsize=10)
    for pos, i in enumerate(order):
        yy = -0.5 * (pos + 1)
        side = 0.7 if pos < k / 2 else k + 0.3
        ax.plot([avg[i], avg[i]], [0, yy], color="0.5", lw=0.8)
        ax.plot([avg[i], side], [yy, yy], color="0.5", lw=0.8)
        ax.text(side + (-0.05 if side < 1 else 0.05), yy,
                f"{MLAB.get(models[i], models[i])} ({avg[i]:.2f})",
                ha="right" if side < 1 else "left", va="center", fontsize=9.5)
    sa = avg[order]; ymark = 0.18; grp = 0; j = 0
    while j < k:
        m = j
        while m + 1 < k and sa[m + 1] - sa[j] <= CD:
            m += 1
        if m > j:
            ax.plot([sa[j] - 0.03, sa[m] + 0.03], [ymark + grp * 0.13, ymark + grp * 0.13], "r", lw=3)
            grp += 1
        j = m + 1 if m > j else j + 1
    ax.set_title(f"{title}\nFriedman p={p:.2e} over N={N} datasets; lower rank=better; "
                 f"red bar = not sig. different (Nemenyi CD={CD:.2f})", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {FIG / fname}  (N={N} datasets, k={k} models, Friedman p={p:.2e})")
    return {"metric": fname, "N_datasets": N, "k_models": k, "friedman_p": float(p), "CD": float(CD),
            **{f"avgrank_{models[i]}": float(avg[i]) for i in range(k)}}


def main() -> None:
    df = pd.read_csv("results/final/conformal_full.csv")  # single source of truth (6 datasets)
    rows = []
    for metric, title, fname in [
        ("mean_width", "Cross-dataset CD: conformal interval width (which model is tightest)",
         "cd_crossdataset_width.png"),
        ("test_mae", "Cross-dataset CD: point-prediction MAE", "cd_crossdataset_mae.png"),
    ]:
        if metric not in df.columns:
            print(f"  [skip] {metric} not in conformal_full.csv"); continue
        mat = df.pivot_table(index="dataset", columns="model", values=metric)
        r = _cd_diagram(mat, title, fname)
        if r:
            rows.append(r)
    if rows:
        pd.DataFrame(rows).to_csv(FINAL / "cd_crossdataset_ranks.csv", index=False)
        print(f"  -> {FINAL / 'cd_crossdataset_ranks.csv'}")


if __name__ == "__main__":
    main()
