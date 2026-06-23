"""Publication figures for the JCIM manuscript:
  (1) TOC graphic -- the contribution in one panel: native model confidence (MC-Dropout, deep
      ensembles) is overconfident (coverage << nominal), split conformal restores validity.
  (2) 8-model per-fold critical-difference diagram (BACE, scaffold-grouped repeated CV).
Reads results/final/{uq_baselines,mc_dropout_deep}.csv + results/raw/perfold_8model_mae.csv.

Run: /opt/anaconda3/bin/python3 scripts/make_paper_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401
from scipy.stats import friedmanchisquare, rankdata  # noqa: E402

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": True})
FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
RED, BLUE, GREY = "#C0392B", "#2C5AA0", "#999999"


def toc_overconfidence():
    mc = pd.read_csv("results/final/mc_dropout_deep.csv")
    ub = pd.read_csv("results/final/uq_baselines.csv")
    de = ub[ub.uq_method.str.startswith("deep-ensemble")].copy()
    rows = []  # (label, native_cov, conformal_cov_or_None)
    for _, r in mc.iterrows():
        rows.append((f"MolFormer MC-Dropout · {r.dataset}", r.coverage_mcdropout, r.coverage_conformal))
    for _, r in de.iterrows():
        rows.append((f"{r.model.capitalize()} deep-ensemble · bace", r.coverage, None))
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    y = np.arange(len(rows))
    ax.axvspan(0.90, 1.0, color="#2E8B57", alpha=0.08)
    ax.axvline(0.90, color=GREY, ls="--", lw=1.2)
    ax.text(0.90, len(rows) - 0.2, "nominal 0.90", fontsize=8, color="#444", ha="center", va="bottom")
    conf_labeled = False
    for i, (lab, nat, conf) in enumerate(rows):
        if conf is not None:
            ax.plot([nat, conf], [i, i], color=GREY, lw=1.2, zorder=1)
            ax.scatter(conf, i, s=70, color=BLUE, zorder=3, label=None if conf_labeled else "split conformal")
            conf_labeled = True
        ax.scatter(nat, i, s=70, color=RED, marker="v", zorder=3,
                   label="native (MC-Dropout / deep ensemble)" if i == 0 else None)
        ax.text(nat - 0.015, i, f"{nat:.2f}", ha="right", va="center", fontsize=7.5, color=RED)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0.18, 1.0); ax.set_ylim(-0.6, len(rows) + 0.25)
    ax.set_xlabel("Empirical coverage at nominal 0.90  (higher = more honest)")
    ax.set_title("Native model confidence is overconfident;\nsplit conformal restores valid coverage", fontsize=11)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, framealpha=0.95)
    fig.tight_layout(); fig.savefig(FIG / "toc_graphic.png", dpi=300, bbox_inches="tight", pad_inches=0.2); plt.close(fig)
    print(f"  -> {FIG / 'toc_graphic.png'}")


def cd_8model():
    Q05 = {8: 3.031}
    df = pd.read_csv("results/raw/perfold_8model_mae.csv")
    M = df.pivot_table(index=["repeat", "fold"], columns="model", values="MAE").dropna()
    order_models = ["Ridge", "SVR", "GBM", "RF", "TabPFN", "MolFormer", "ChemFM", "Chemprop"]
    models = [m for m in order_models if m in M.columns]
    N, k = M.shape[0], len(models)
    ranks = np.apply_along_axis(rankdata, 1, M[models].to_numpy(float))
    avg = ranks.mean(0)
    chi, p = friedmanchisquare(*[M[m].to_numpy(float) for m in models])
    CD = Q05.get(k, 3.031) * np.sqrt(k * (k + 1) / (6 * N))
    o = np.argsort(avg)
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.set_xlim(0.7, k + 0.3); ax.set_ylim(-0.6 * k - 1, 1.9); ax.axis("off")
    ax.plot([1, k], [0, 0], "k", lw=1)
    for r in range(1, k + 1):
        ax.plot([r, r], [0, 0.12], "k", lw=1); ax.text(r, 0.42, str(r), ha="center", fontsize=10)
    ax.plot([1, 1 + CD], [1.25, 1.25], "k", lw=2.5); ax.text(1 + CD / 2, 1.5, f"CD={CD:.2f}", ha="center", fontsize=10)
    for pos, i in enumerate(o):
        yy = -0.5 * (pos + 1); side = 0.7 if pos < k / 2 else k + 0.3
        ax.plot([avg[i], avg[i]], [0, yy], color="0.5", lw=0.8)
        ax.plot([avg[i], side], [yy, yy], color="0.5", lw=0.8)
        star = " (control)" if models[i] == "TabPFN" else ""
        ax.text(side + (-0.05 if side < 1 else 0.05), yy, f"{models[i]}{star} ({avg[i]:.2f})",
                ha="right" if side < 1 else "left", va="center", fontsize=9)
    sa = avg[o]; grp = 0; j = 0
    while j < k:
        m = j
        while m + 1 < k and sa[m + 1] - sa[j] <= CD:
            m += 1
        if m > j:
            ax.plot([sa[j] - 0.03, sa[m] + 0.03], [0.18 + grp * 0.13, 0.18 + grp * 0.13], "r", lw=3); grp += 1
        j = m + 1 if m > j else j + 1
    ax.set_title(f"Per-fold critical-difference (BACE-1, N={N} scaffold-grouped CV folds); "
                 f"lower rank = better; Friedman p={p:.1e}", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "cd_perfold_8model.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {FIG / 'cd_perfold_8model.png'}  (N={N}, k={k}, Friedman p={p:.2e})")


def cov_vs_width():
    """Tightest-is-not-valid: coverage vs width (normalized within dataset), each point a model."""
    d = pd.read_csv("results/final/conformal_full.csv")
    d["wrel"] = d.groupby("dataset")["mean_width"].transform(lambda x: x / x.min())
    COL = {"rf": "#888", "molformer": "#1f77b4", "chemfm": "#d62728", "unimol": "#2ca02c", "tabpfn": "#9467bd"}
    LAB = {"rf": "RF+Morgan", "molformer": "MolFormer", "chemfm": "ChemFM", "unimol": "Uni-Mol", "tabpfn": "TabPFN"}
    ymin = float(d.empirical_coverage.min())
    lo = min(0.80, ymin - 0.02)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axhspan(lo, 0.90, color=RED, alpha=0.07)
    ax.axhline(0.90, color=GREY, ls="--", lw=1.2)
    for m, c in COL.items():
        s = d[d.model == m]
        if len(s):
            ax.scatter(s.wrel, s.empirical_coverage, c=c, s=62, label=LAB[m],
                       edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_ylim(lo, 1.0)
    ax.set_xlabel("Relative interval width (1 = tightest model in that dataset; lower = tighter)")
    ax.set_ylabel("Empirical coverage (nominal 0.90)")
    ax.set_title("Tightest is not valid: conformal coverage vs width\n(each point one dataset $\\times$ model)", fontsize=11)
    ax.text(ax.get_xlim()[1] * 0.98, 0.898, "under-covers (invalid)", ha="right", va="top", fontsize=8.5, color=RED)
    ax.text(ax.get_xlim()[0] + 0.02, 0.992, "tightest models cluster here → some under-cover",
            ha="left", va="top", fontsize=7.5, color="#555")
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "cov_vs_width.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {FIG / 'cov_vs_width.png'}")


def ad_conditional():
    """Conditional coverage by applicability-domain stratum (BACE-1, 8 models); TabPFN holds."""
    d = pd.read_csv("results/final/conformal_ad_coverage.csv", index_col=0)
    x = np.arange(d.shape[1])
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.axhline(0.90, color=GREY, ls="--", lw=1.2)
    for m in d.index:
        bold = (m == "TabPFN")
        ax.plot(x, d.loc[m].to_numpy(float), marker="o", lw=3 if bold else 1.3,
                color="#9467bd" if bold else None, label=m, zorder=4 if bold else 2,
                alpha=1.0 if bold else 0.65)
    ax.set_xticks(x); ax.set_xticklabels([c.split(" ")[0] for c in d.columns])
    ax.set_xlabel("Applicability domain (Tanimoto-to-nearest-train stratum)")
    ax.set_ylabel("Conditional coverage (nominal 0.90)")
    ax.set_title("Coverage degrades out of domain; TabPFN holds (BACE-1)", fontsize=11)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "ad_conditional.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {FIG / 'ad_conditional.png'}  (low-AD stratum is small, n~10 on BACE)")


def dual_split():
    """Random vs scaffold conformal coverage per dataset (exchangeability)."""
    d = pd.read_csv("results/final/gap4_dual_split.csv").iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    y = np.arange(len(d))
    ax.axvline(0.90, color=GREY, ls="--", lw=1.2)
    for i, r in d.iterrows():
        ax.plot([r.cov_scaffold, r.cov_random], [i, i], color=GREY, lw=1.2, zorder=1)
        ax.scatter(r.cov_random, i, s=72, color=BLUE, zorder=3, label="random (exchangeable)" if i == 0 else None)
        ax.scatter(r.cov_scaffold, i, s=72, color=RED, marker="s", zorder=3, label="scaffold (OOD)" if i == 0 else None)
    ax.set_yticks(y); ax.set_yticklabels(d.dataset)
    ax.set_xlabel("Conformal coverage (nominal 0.90)")
    ax.set_title("Exchangeability: scaffold split degrades coverage on drug data", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "dual_split.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {FIG / 'dual_split.png'}")


if __name__ == "__main__":
    toc_overconfidence()
    cd_8model()
    cov_vs_width()
    ad_conditional()
    dual_split()
    print("done.")
