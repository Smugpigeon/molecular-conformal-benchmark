"""A7 figures (local): does TabPFN uncertainty flag hard molecules, and how do conformal
intervals compare? Reads results/exp/{uncertainty_test,conformal_summary}.csv.

Run: /opt/anaconda3/bin/python3 scripts/make_uncertainty_figures.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scienceplots  # noqa: E402,F401
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_error_analysis import MODELS, load_wide  # noqa: E402

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": True})
warnings.filterwarnings("ignore")
FIG = Path("results/figures"); EXP = Path("results/exp")


def main():
    per = pd.read_csv(EXP / "uncertainty_test.csv")
    summ = pd.read_csv(EXP / "conformal_summary.csv")
    # consensus-hard label (top decile mean |err| across all models)
    wide = load_wide()
    labels = [m for m, _, _ in MODELS if m in wide.columns]
    cons = pd.Series(np.mean([np.abs(wide[m] - wide.y_true) for m in labels], axis=0).ravel(),
                     index=wide.smiles)
    per = per.merge(cons.rename("cons_err").reset_index().rename(columns={"index": "smiles"}),
                    on="smiles", how="left")
    hard = (per.cons_err >= per.cons_err.quantile(0.9)).to_numpy()

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.4))
    a, b, c, d = axs.ravel()

    # (a) uncertainty (interval width) vs |error|
    rho = spearmanr(per.tab_width, per.abs_err)[0]
    a.scatter(per.tab_width[~hard], per.abs_err[~hard], s=14, alpha=0.5, color="#4C72B0", label="easy")
    a.scatter(per.tab_width[hard], per.abs_err[hard], s=22, alpha=0.8, color="#C44E52", label="consensus-hard")
    a.set_xlabel("TabPFN uncertainty (q95 - q05 width)"); a.set_ylabel("TabPFN |error|")
    a.set_title(f"(a) uncertainty tracks error (Spearman={rho:.2f})", fontsize=10); a.legend(fontsize=8)

    # (b) error-rejection curve: reject the most-uncertain x%, MAE of the rest
    n = len(per)
    fracs = np.linspace(0, 0.5, 26)
    def curve(rank_by):
        order = np.argsort(-rank_by)  # most "risky" first
        out = []
        for fr in fracs:
            keep = order[int(fr * n):]
            out.append(per.abs_err.to_numpy()[keep].mean())
        return out
    b.plot(fracs * 100, curve(per.tab_width.to_numpy()), "o-", ms=3, color="#C44E52", label="reject by TabPFN uncertainty")
    b.plot(fracs * 100, curve(per.abs_err.to_numpy()), "-", color="#55A868", label="oracle (reject by true error)")
    rng = np.random.default_rng(0)
    rand = np.mean([curve(rng.random(n)) for _ in range(20)], axis=0)
    b.plot(fracs * 100, rand, "--", color="0.5", label="random rejection")
    b.set_xlabel("% most-uncertain molecules rejected"); b.set_ylabel("MAE on the retained set")
    b.set_title("(b) selective prediction: uncertainty-guided rejection", fontsize=10); b.legend(fontsize=8)

    # (c) conformal coverage vs nominal
    x = np.arange(len(summ))
    cols = ["#C44E52", "#DD8452", "#8172B3", "#4C72B0"][:len(summ)]
    c.bar(x, summ.coverage, color=cols, edgecolor="k", linewidth=0.4)
    c.axhline(0.9, ls="--", color="k", lw=1, label="nominal 0.90")
    for i, v in enumerate(summ.coverage):
        c.annotate(f"{v:.3f}", (i, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5)
    c.set_xticks(x); c.set_xticklabels(summ.method, rotation=20, ha="right", fontsize=7.5)
    c.set_ylim(0.8, 1.0); c.set_ylabel("empirical coverage"); c.set_title("(c) conformal coverage", fontsize=10)
    c.legend(fontsize=8)

    # (d) conformal interval width (efficiency)
    d.bar(x, summ.mean_width, color=cols, edgecolor="k", linewidth=0.4)
    for i, v in enumerate(summ.mean_width):
        d.annotate(f"{v:.2f}", (i, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5)
    d.set_xticks(x); d.set_xticklabels(summ.method, rotation=20, ha="right", fontsize=7.5)
    d.set_ylabel("mean interval width (log units)")
    d.set_title("(d) interval width -- narrower = more useful", fontsize=10)

    fig.suptitle("A7  TabPFN uncertainty flags hard molecules + conformal calibration (nominal 90%)",
                 y=1.0, fontsize=12.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_err_uncertainty_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # hard-detection: are consensus-hard molecules more often OUTSIDE the native interval?
    out_native = ((per.y_true < per.tab_q05) | (per.y_true > per.tab_q95)).to_numpy()
    print("wrote fig_err_uncertainty_j.png")
    print(f"  Spearman(uncertainty, |error|) = {rho:.3f}")
    print(f"  consensus-hard outside native 90% interval: {out_native[hard].mean():.0%}  "
          f"vs easy: {out_native[~hard].mean():.0%}")


if __name__ == "__main__":
    main()
