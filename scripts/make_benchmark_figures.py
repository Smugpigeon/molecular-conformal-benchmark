"""Publication-quality MAIN-benchmark figures (scienceplots), matching the
conformal figure set's style. Renders from existing result CSVs (no recompute):

  1. fig_benchmark_regression -- MAE + Pearson R, 5 datasets x 5 backbones
  2. fig_bace_docking         -- Vina docking score vs experimental pIC50
                                 (honest: n=30, weak + non-significant trend)
  3. fig_activity_cliff       -- on- vs off-cliff RMSE; FMs penalised on cliffs
                                 (confirmatory, MoleculeACE 30 targets)

Run: python3 scripts/make_benchmark_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401
from scipy import stats  # noqa: E402

plt.style.use(["science", "no-latex"])
RES = Path("results")
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Identical palette to scripts/make_conformal_figures.py (Wong colorblind-safe).
MODEL_COLOR = {
    "rf": "#7f7f7f", "gbm": "#7f7f7f", "chemprop": "#CC79A7",
    "molformer": "#0072B2", "chemfm": "#009E73", "unimol": "#D55E00",
}
MODEL_LABEL = {
    "rf": "RF", "gbm": "GBM", "chemprop": "Chemprop",
    "molformer": "MolFormer", "chemfm": "ChemFM", "unimol": "Uni-Mol",
}
MODEL_ORDER = ["rf", "chemprop", "molformer", "chemfm", "unimol"]
DATA_LABEL = {"esol": "ESOL", "freesolv": "FreeSolv",
              "lipophilicity": "Lipo", "bace": "BACE", "gsht": "GSHt"}
DATA_ORDER = ["esol", "freesolv", "lipophilicity", "bace", "gsht"]


def _present(models) -> list[str]:
    s = set(models)
    return [m for m in MODEL_ORDER if m in s]


# ---------------------------------------------------------------- Fig 1
def fig_regression() -> None:
    """Grouped MAE (lower better) + Pearson R (higher better) across backbones."""
    df = pd.read_csv(RES / "final_with_gnn.csv")
    r = df[df.dataset.isin(DATA_ORDER)].dropna(subset=["MAE"])
    models = _present(r.model.unique())
    agg = r.groupby(["dataset", "model"]).agg(
        mae=("MAE", "mean"), mae_sd=("MAE", "std"),
        pr=("PearsonR", "mean"), pr_sd=("PearsonR", "std")).reset_index()

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True)
    n = len(models)
    w = 0.8 / n
    x = np.arange(len(DATA_ORDER))

    def grouped(ax, col, sd):
        for i, m in enumerate(models):
            ys, es = [], []
            for d in DATA_ORDER:
                row = agg[(agg.dataset == d) & (agg.model == m)]
                ys.append(float(row[col].iloc[0]) if len(row) else 0.0)
                es.append(float(row[sd].iloc[0]) if len(row) else 0.0)
            ax.bar(x + (i - (n - 1) / 2) * w, ys, w, yerr=es, capsize=1.5,
                   color=MODEL_COLOR[m], label=MODEL_LABEL[m],
                   edgecolor="k", linewidth=0.4)

    grouped(a1, "mae", "mae_sd")
    a1.set_ylabel("MAE $\\downarrow$")
    a1.set_title("Regression benchmark: predefined-split val performance (mean $\\pm$ s.d., 3 seeds)")
    grouped(a2, "pr", "pr_sd")
    a2.set_ylabel("Pearson $R$ $\\uparrow$")
    a2.set_ylim(0, 1.0)
    a2.set_xticks(x)
    a2.set_xticklabels([DATA_LABEL[d] for d in DATA_ORDER])
    a1.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.30))
    fig.savefig(FIG / "fig_benchmark_regression.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig_bace_docking() -> None:
    """Docking score vs experimental activity, honest about n and significance."""
    dk = pd.read_csv(RES / "bace_docking_full.csv")
    r, p = stats.pearsonr(dk.vina, dk.pIC50)
    sp = stats.spearmanr(dk.vina, dk.pIC50)[0]

    def partial(x, y, z):
        rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
        ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
        return stats.pearsonr(rx, ry)[0]

    pr = partial(dk.vina.values, dk.pIC50.values, dk.heavy.values)

    fig, ax = plt.subplots(figsize=(5.4, 4))
    sc = ax.scatter(dk.vina, dk.pIC50, c=dk.heavy, cmap="viridis", s=42,
                    edgecolor="k", linewidth=0.4, zorder=3)
    # OLS trend line
    b = np.polyfit(dk.vina, dk.pIC50, 1)
    xs = np.linspace(dk.vina.min(), dk.vina.max(), 50)
    ax.plot(xs, np.poly1d(b)(xs), ls="--", color="#D55E00", lw=1.2, zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("heavy-atom count (size confound)")
    ax.set_xlabel("AutoDock Vina score (kcal/mol, lower = stronger)")
    ax.set_ylabel("experimental pIC$_{50}$")
    sig = "n.s." if p >= 0.05 else f"p={p:.3f}"
    ax.set_title(f"BACE docking vs activity (n={len(dk)}): "
                 f"$r$={r:.2f} ({sig}); partial $r$={pr:.2f}")
    fig.savefig(FIG / "fig_bace_docking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def fig_activity_cliff() -> None:
    """On- vs off-cliff RMSE (left) + cliff-ratio distribution (right)."""
    cl = pd.read_csv(RES / "cliff_all.csv")
    models = _present(cl.model.unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.8))

    # (a) per-target off-cliff vs on-cliff RMSE; above diagonal = worse on cliff
    for m in models:
        g = cl[cl.model == m]
        a1.scatter(g.rmse_noncliff, g.rmse_cliff, s=20, color=MODEL_COLOR[m],
                   label=MODEL_LABEL[m], edgecolor="k", linewidth=0.25, alpha=0.85, zorder=3)
    hi = float(np.nanmax([cl.rmse_noncliff.max(), cl.rmse_cliff.max()])) * 1.05
    a1.plot([0, hi], [0, hi], ls=":", color="k", lw=0.8, zorder=1)
    a1.set_xlim(0, hi); a1.set_ylim(0, hi)
    a1.set_xlabel("RMSE off-cliff"); a1.set_ylabel("RMSE on-cliff")
    a1.set_title("Above diagonal = worse on cliffs")
    a1.legend(fontsize=7, loc="upper left")

    # (b) cliff-ratio (on/off) per backbone; >1 = cliff penalty
    data = [cl[cl.model == m].cliff_ratio.values for m in models]
    bp = a2.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color="k"))
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(MODEL_COLOR[m]); patch.set_alpha(0.75)
    # jittered points (index-based offset; no RNG so the figure is reproducible)
    for i, m in enumerate(models):
        y = cl[cl.model == m].cliff_ratio.values
        jit = (np.arange(len(y)) % 7 - 3) / 22.0
        a2.scatter(np.full(len(y), i + 1) + jit, y, s=8, color="k", alpha=0.35, zorder=3)
    a2.axhline(1.0, ls="--", color="#D55E00", lw=1.0)
    a2.set_xticks(range(1, len(models) + 1))
    a2.set_xticklabels([MODEL_LABEL[m] for m in models])
    a2.set_ylabel("cliff ratio  (RMSE on / off)")
    a2.set_title("$>$1 = penalised on activity cliffs")
    fig.suptitle("Activity-cliff stress test (MoleculeACE, 30 targets; confirmatory)",
                 y=1.02, fontsize=10)
    fig.savefig(FIG / "fig_activity_cliff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def fig_classification() -> None:
    """BBBP classification: 4 metrics x backbones (CLAUDE.md §9 requires all 4)."""
    df = pd.read_csv(RES / "final_with_gnn.csv")
    c = df[df.dataset == "bbbp"].dropna(subset=["ROC-AUC"])
    models = _present(c.model.unique())
    metrics = ["ROC-AUC", "PR-AUC", "F1", "MCC"]
    agg = c.groupby("model")[metrics].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    n = len(models)
    w = 0.8 / n
    x = np.arange(len(metrics))
    for i, m in enumerate(models):
        ys = [agg.loc[m, (mt, "mean")] for mt in metrics]
        es = [agg.loc[m, (mt, "std")] for mt in metrics]
        ax.bar(x + (i - (n - 1) / 2) * w, ys, w, yerr=es, capsize=1.5,
               color=MODEL_COLOR[m], label=MODEL_LABEL[m], edgecolor="k", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score $\\uparrow$")
    ax.set_ylim(0, 1.05)
    ax.set_title("BBBP classification: predefined-split val performance (mean $\\pm$ s.d., 3 seeds)")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(FIG / "fig_benchmark_classification.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5
def fig_bbbp_calibration() -> None:
    """Reliability diagram before/after temperature scaling (data from CSV)."""
    p = RES / "bbbp_calibration.csv"
    if not p.exists():
        print("  (skip calibration: results/bbbp_calibration.csv not present yet)")
        return
    d = pd.read_csv(p)
    meta = pd.read_csv(RES / "bbbp_calibration_meta.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=0.9, label="perfect calibration")
    nmax = d.n.max()
    # Marker area ∝ bin count: the over-confident cluster (n=93 at conf≈1.0)
    # dominates; tiny n=1-2 bins are noise and stay small. No connecting lines
    # (they would imply a function through the noise bins).
    for kind, color, mk in [("before", "#D55E00", "o"), ("after", "#0072B2", "s")]:
        sub = d[d.kind == kind]
        ax.scatter(sub.conf, sub.acc, s=20 + sub.n / nmax * 420, marker=mk,
                   color=color, alpha=0.7, edgecolor="k", linewidth=0.4,
                   label=f"{kind} T (ECE={meta[f'ece_{kind}']:.3f})", zorder=3)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0.5, 1.02); ax.set_ylim(0.45, 1.04)
    drop = 100 * (meta["ece_before"] - meta["ece_after"]) / max(meta["ece_before"], 1e-9)
    ax.set_title(f"BBBP reliability (T={meta['T']:.1f}; ECE $\\downarrow${drop:.0f}%; "
                 f"marker $\\propto$ bin count)")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(FIG / "fig_bbbp_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    fig_regression()
    fig_classification()
    fig_bace_docking()
    fig_activity_cliff()
    fig_bbbp_calibration()
    made = sorted(p.name for p in FIG.glob("fig_benchmark_*.png")) + \
        ["fig_bace_docking.png", "fig_activity_cliff.png", "fig_bbbp_calibration.png"]
    made = [m for m in made if (FIG / m).exists()]
    print("wrote", len(made), "benchmark figures to", FIG)
    for m in made:
        print("  ", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
