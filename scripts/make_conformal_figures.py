"""Publication-quality conformal-benchmark figures (scienceplots).

Renders the cross-foundation-model conformal regression story from the
existing result CSVs (no recomputation). Maps to the four novelty
deliverables in CLAUDE.md §1 plus the Uni-Mol CQR finding:

  1. fig_tradeoff_width_coverage  -- "tightest != valid" (width vs coverage)
  2. fig_width_leaderboard        -- tightest VALID interval per dataset
  3. fig_coverage_heatmap         -- backbone x dataset coverage overview
  4. fig_ad_conditional_coverage  -- under-coverage concentrates on novel chemistry
  5. fig_split_vs_cqr_scatter     -- does adaptive CQR restore Uni-Mol coverage?

Run: python3 scripts/make_conformal_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401  (registers the 'science' style)

plt.style.use(["science", "no-latex"])

RES = Path("results")
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TARGET = 0.90

# Wong colorblind-safe palette; Uni-Mol = vermilion protagonist.
MODEL_COLOR = {
    "rf": "#7f7f7f", "gbm": "#7f7f7f", "chemprop": "#CC79A7",
    "molformer": "#0072B2", "chemfm": "#009E73", "unimol": "#D55E00",
}
MODEL_LABEL = {
    "rf": "RF", "gbm": "GBM", "chemprop": "Chemprop",
    "molformer": "MolFormer", "chemfm": "ChemFM", "unimol": "Uni-Mol",
}
# rf and gbm share gray but never co-occur in one figure (rf = split-conformal
# classical baseline; gbm = CQR classical baseline).
MODEL_ORDER = ["rf", "gbm", "chemprop", "molformer", "chemfm", "unimol"]
# Paper-track AD/CQR figures: the four predefined-split bioactivity/physicochemical datasets.
# GSHt is quarantined; QM7/QM8 use random splits (AD/CQR contrast near-degenerate).
DATA_LABEL = {"esol": "ESOL", "freesolv": "FreeSolv",
              "lipophilicity": "Lipo", "bace": "BACE"}
DATA_ORDER = ["esol", "freesolv", "lipophilicity", "bace"]


def _order_models(models) -> list[str]:
    present = set(models)
    return [m for m in MODEL_ORDER if m in present]


# ---------------------------------------------------------------- Fig 1
def fig_tradeoff() -> None:
    """Per-dataset width vs coverage; the headline 'tightest != valid'."""
    df = pd.read_csv("results/final/conformal_multiseed.csv")  # single source of truth (6 datasets)
    models = _order_models(df["model"].unique())
    fig, axes = plt.subplots(1, len(DATA_ORDER), figsize=(11, 2.5), sharey=True)
    for ax, d in zip(axes, DATA_ORDER):
        sub = df[df.dataset == d]
        for m in models:
            r = sub[sub.model == m]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.errorbar(r.width_mean, r.cov_mean, xerr=r.width_std, yerr=r.cov_std,
                        fmt="o", ms=5, color=MODEL_COLOR[m], capsize=2,
                        label=MODEL_LABEL[m], zorder=3)
        ax.axhline(TARGET, ls="--", lw=0.8, color="k", zorder=1)
        ax.axhspan(0.70, TARGET, color="#D55E00", alpha=0.06, zorder=0)
        ax.set_title(DATA_LABEL[d])
        ax.set_xlabel("interval width")
        ax.margins(x=0.25)
    axes[0].set_ylabel("coverage")
    axes[0].set_ylim(0.78, 1.005)
    axes[-1].legend(fontsize=6, loc="lower right", framealpha=0.9)
    fig.suptitle("Tightest $\\neq$ valid: width vs coverage (shaded = under-covers 90% target)",
                 y=1.02, fontsize=10)
    fig.savefig(FIG / "fig_tradeoff_width_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def fig_width_leaderboard() -> None:
    """Grouped width bars; hatch = invalid coverage (width not comparable)."""
    df = pd.read_csv("results/final/conformal_multiseed.csv")  # single source of truth (6 datasets)
    models = _order_models(df["model"].unique())
    fig, ax = plt.subplots(figsize=(8, 3))
    n = len(models)
    w = 0.8 / n
    x = np.arange(len(DATA_ORDER))
    for i, m in enumerate(models):
        ws, es, hatch = [], [], []
        for d in DATA_ORDER:
            r = df[(df.dataset == d) & (df.model == m)]
            if len(r):
                r = r.iloc[0]
                ws.append(r.width_mean); es.append(r.width_std)
                hatch.append("///" if r.cov_mean < TARGET - 0.015 else "")
            else:
                ws.append(0); es.append(0); hatch.append("")
        bars = ax.bar(x + (i - (n - 1) / 2) * w, ws, w, yerr=es, capsize=1.5,
                      color=MODEL_COLOR[m], label=MODEL_LABEL[m],
                      edgecolor="k", linewidth=0.4)
        for b, h in zip(bars, hatch):
            if h:
                b.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels([DATA_LABEL[d] for d in DATA_ORDER])
    ax.set_ylabel("interval width (90% target)")
    ax.set_title("Interval-width leaderboard (hatched = coverage $<$ target, invalid)")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.savefig(FIG / "fig_width_leaderboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3
def fig_coverage_heatmap() -> None:
    """Backbone x dataset coverage; diverging colormap centered on target."""
    df = pd.read_csv("results/final/conformal_multiseed.csv")  # single source of truth (6 datasets)
    models = _order_models(df["model"].unique())
    M = np.full((len(DATA_ORDER), len(models)), np.nan)
    for i, d in enumerate(DATA_ORDER):
        for j, m in enumerate(models):
            r = df[(df.dataset == d) & (df.model == m)]
            if len(r):
                M[i, j] = r.iloc[0].cov_mean
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    im = ax.imshow(M, cmap="RdBu", vmin=0.80, vmax=1.00, aspect="auto")
    # center the diverging map on the 0.90 target
    im.set_clim(TARGET - 0.10, TARGET + 0.10)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in models], rotation=30, ha="right")
    ax.set_yticks(range(len(DATA_ORDER)))
    ax.set_yticklabels([DATA_LABEL[d] for d in DATA_ORDER])
    for i in range(len(DATA_ORDER)):
        for j in range(len(models)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="k")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("coverage")
    ax.set_title(f"Empirical coverage (target = {TARGET:.2f}; red = under-covers)")
    fig.savefig(FIG / "fig_coverage_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4
def fig_ad_conditional() -> None:
    """Coverage vs applicability-domain stratum; the §16.6 moat."""
    df = pd.read_csv("results/final/conformal_ad.csv")  # single source of truth (no GSHt)
    order = ["low", "med", "high"]
    models = _order_models(df["model"].unique())
    # mean coverage over datasets per (model, stratum) + std band
    g = df.groupby(["model", "stratum"]).agg(
        cov=("coverage", "mean"), cov_sd=("coverage", "std"),
        tani=("mean_tanimoto", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xs = np.arange(len(order))
    for m in models:
        ys = [g[(g.model == m) & (g.stratum == s)]["cov"].values for s in order]
        sd = [g[(g.model == m) & (g.stratum == s)]["cov_sd"].values for s in order]
        ys = [float(y[0]) if len(y) else np.nan for y in ys]
        sd = [float(s[0]) if len(s) and not np.isnan(s[0]) else 0.0 for s in sd]
        ax.errorbar(xs, ys, yerr=sd, fmt="o-", ms=5, lw=1.3, capsize=2,
                    color=MODEL_COLOR[m], label=MODEL_LABEL[m])
    ax.axhline(TARGET, ls="--", lw=0.8, color="k")
    ax.axhspan(0.5, TARGET, color="#D55E00", alpha=0.06)
    # tanimoto annotation per stratum
    tani = [g[g.stratum == s]["tani"].mean() for s in order]
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}\n(T$\\approx${t:.2f})" for s, t in zip(order, tani)])
    ax.set_xlabel("applicability-domain stratum (Tanimoto to nearest train)")
    ax.set_ylabel("coverage")
    ax.set_title("Marginal coverage hides it: novel chemistry is under-covered")
    ax.legend(fontsize=7, loc="lower right")
    fig.savefig(FIG / "fig_ad_conditional_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 5
def fig_split_vs_cqr() -> None:
    """Does adaptive CQR restore coverage / tighten width? Uni-Mol highlighted."""
    df = pd.read_csv("results/final/split_vs_cqr.csv").dropna(subset=["split_cov", "cqr_cov"])  # no GSHt
    models = _order_models(df["backbone"].unique())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6))

    # (a) coverage: split -> CQR, want points crossing up to the target line
    for m in models:
        sub = df[df.backbone == m]
        a1.scatter(sub.split_cov, sub.cqr_cov, s=28, color=MODEL_COLOR[m],
                   label=MODEL_LABEL[m], edgecolor="k", linewidth=0.3, zorder=3)
    lim = [0.80, 1.01]
    a1.plot(lim, lim, ls=":", color="k", lw=0.8, zorder=1)
    a1.axhline(TARGET, ls="--", lw=0.7, color="#D55E00")
    a1.axvline(TARGET, ls="--", lw=0.7, color="#D55E00")
    a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("split-conformal coverage"); a1.set_ylabel("CQR coverage")
    a1.set_title("Coverage: above diagonal = CQR improves")

    # (b) width: split -> CQR, below diagonal = CQR tighter
    for m in models:
        sub = df[df.backbone == m]
        a2.scatter(sub.split_w, sub.cqr_w, s=28, color=MODEL_COLOR[m],
                   label=MODEL_LABEL[m], edgecolor="k", linewidth=0.3, zorder=3)
    mx = float(np.nanmax([df.split_w.max(), df.cqr_w.max()])) * 1.05
    a2.plot([0, mx], [0, mx], ls=":", color="k", lw=0.8, zorder=1)
    a2.set_xlim(0, mx); a2.set_ylim(0, mx)
    a2.set_xlabel("split-conformal width"); a2.set_ylabel("CQR width")
    a2.set_title("Width: below diagonal = CQR tighter")
    a2.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    fig.suptitle("Split conformal $\\rightarrow$ CQR (Uni-Mol: under-coverage restored on ESOL)",
                 y=1.01, fontsize=10)
    fig.savefig(FIG / "fig_split_vs_cqr_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    fig_tradeoff()
    fig_width_leaderboard()
    fig_coverage_heatmap()
    fig_ad_conditional()
    fig_split_vs_cqr()
    made = sorted(p.name for p in FIG.glob("fig_*.png"))
    print("wrote", len(made), "figures to", FIG)
    for m in made:
        print("  ", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
