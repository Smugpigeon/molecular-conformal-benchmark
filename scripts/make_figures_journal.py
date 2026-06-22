"""Journal-unified figures for the BACE deck: scienceplots style + Times New Roman
(matches the journal-formatted report's Latin font). Renders from existing result
CSVs (no recompute, no fabricated numbers). Outputs *_j.png so the report's
originals are left untouched.

  fig_benchmark_regression_j  -- MAE + Pearson R, 5 datasets x 5 backbones
  fig_bace_docking_j          -- Vina score vs pIC50 (n=30, weak/n.s.)
  fig_tradeoff_j              -- conformal: interval width vs coverage, faceted
Run: python3 scripts/make_figures_journal.py
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401
from scipy import stats  # noqa: E402

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",     # Times-matched math glyphs
    "axes.unicode_minus": True,
})
RES = Path("results"); FIG = RES / "figures"; FIG.mkdir(parents=True, exist_ok=True)

MODEL_COLOR = {"rf": "#7f7f7f", "chemprop": "#CC79A7", "molformer": "#0072B2",
               "chemfm": "#009E73", "unimol": "#D55E00"}
MODEL_LABEL = {"rf": "RF", "chemprop": "Chemprop", "molformer": "MolFormer",
               "chemfm": "ChemFM", "unimol": "Uni-Mol"}
MODEL_ORDER = ["rf", "chemprop", "molformer", "chemfm", "unimol"]
DATA_LABEL = {"esol": "ESOL", "freesolv": "FreeSolv", "lipophilicity": "Lipo",
              "bace": "BACE", "gsht": "GSHt"}
DATA_ORDER = ["esol", "freesolv", "lipophilicity", "bace", "gsht"]


def present(models):
    s = set(models)
    return [m for m in MODEL_ORDER if m in s]


def fig_regression():
    df = pd.read_csv(RES / "final_with_gnn.csv")
    r = df[df.dataset.isin(DATA_ORDER)].dropna(subset=["MAE"])
    models = present(r.model.unique())
    agg = r.groupby(["dataset", "model"]).agg(
        mae=("MAE", "mean"), mae_sd=("MAE", "std"),
        pr=("PearsonR", "mean"), pr_sd=("PearsonR", "std")).reset_index()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True)
    n = len(models); w = 0.8 / n; x = np.arange(len(DATA_ORDER))

    def grouped(ax, col, sd):
        for i, m in enumerate(models):
            ys, es = [], []
            for d in DATA_ORDER:
                row = agg[(agg.dataset == d) & (agg.model == m)]
                ys.append(float(row[col].iloc[0]) if len(row) else 0.0)
                es.append(float(row[sd].iloc[0]) if len(row) else 0.0)
            ax.bar(x + (i - (n - 1) / 2) * w, ys, w, yerr=es, capsize=1.5,
                   color=MODEL_COLOR[m], label=MODEL_LABEL[m], edgecolor="k", linewidth=0.4)

    grouped(a1, "mae", "mae_sd"); a1.set_ylabel(r"MAE $\downarrow$")
    a1.set_title(r"Regression benchmark (mean $\pm$ s.d., 3 seeds)")
    grouped(a2, "pr", "pr_sd"); a2.set_ylabel(r"Pearson $R$ $\uparrow$"); a2.set_ylim(0, 1.0)
    a2.set_xticks(x); a2.set_xticklabels([DATA_LABEL[d] for d in DATA_ORDER])
    a1.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.30))
    fig.savefig(FIG / "fig_benchmark_regression_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_docking():
    dk = pd.read_csv(RES / "bace_docking_full.csv")
    r, p = stats.pearsonr(dk.vina, dk.pIC50)

    def partial(x, y, z):
        rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
        ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
        return stats.pearsonr(rx, ry)[0]

    pr = partial(dk.vina.values, dk.pIC50.values, dk.heavy.values)
    fig, ax = plt.subplots(figsize=(5.4, 4))
    sc = ax.scatter(dk.vina, dk.pIC50, c=dk.heavy, cmap="cividis", s=42,
                    edgecolor="k", linewidth=0.4, zorder=3)
    b = np.polyfit(dk.vina, dk.pIC50, 1)
    xs = np.linspace(dk.vina.min(), dk.vina.max(), 50)
    ax.plot(xs, np.poly1d(b)(xs), ls="--", color="#9b2226", lw=1.2, zorder=2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("heavy-atom count (size confound)")
    ax.set_xlabel("AutoDock Vina score (kcal/mol, lower = stronger)")
    ax.set_ylabel(r"experimental pIC$_{50}$")
    sig = "n.s." if p >= 0.05 else f"p={p:.3f}"
    ax.set_title(f"BACE docking vs activity (n={len(dk)}): $r$={r:.2f} ({sig}); partial $r$={pr:.2f}")
    fig.savefig(FIG / "fig_bace_docking_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff():
    cf = pd.read_csv("results/final/conformal_multiseed.csv")  # single source of truth (6 datasets)
    models = present(cf.model.unique())
    fig, axs = plt.subplots(1, len(DATA_ORDER), figsize=(11, 2.7), sharey=True)
    for ax, d in zip(axs, DATA_ORDER):
        g = cf[cf.dataset == d]
        ax.axvspan(0.70, 0.90, color="#bbbbbb", alpha=0.30, zorder=0)   # undercovered
        ax.axvline(0.90, ls="--", color="k", lw=0.8, zorder=1)
        for m in models:
            row = g[g.model == m]
            if len(row):
                ax.scatter(float(row.cov_mean.iloc[0]), float(row.width_mean.iloc[0]),
                           s=34, color=MODEL_COLOR[m], edgecolor="k", linewidth=0.4,
                           label=MODEL_LABEL[m], zorder=3)
        ax.set_title(DATA_LABEL[d], fontsize=9)
        ax.set_xlabel("coverage"); ax.set_xlim(0.78, 1.02)
    axs[0].set_ylabel("interval width")
    axs[-1].legend(fontsize=6, loc="upper right", framealpha=0.9)
    fig.suptitle(r"Tightest $\neq$ valid: interval width vs empirical coverage "
                 r"(shaded = undercovered; nominal 0.90)", y=1.06, fontsize=10)
    fig.savefig(FIG / "fig_tradeoff_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_regression(); fig_docking(); fig_tradeoff()
    print("wrote: fig_benchmark_regression_j.png, fig_bace_docking_j.png, fig_tradeoff_j.png")
