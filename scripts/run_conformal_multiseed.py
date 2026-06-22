"""Multi-seed conformal benchmark: width + coverage with cross-seed error bars.

Per project §8 (3 seeds mandatory) + the sufficiency bar for a credible
conformal benchmark: split-conformal interval width depends on the single
calibration draw, so a single-seed width leaderboard can be a coin-flip. This
aggregates split conformal over seeds {42, 1337, 2024} and reports mean ± std
of width and empirical coverage per (dataset, backbone).

Run: python scripts/run_conformal_multiseed.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.eval.conformal import split_conformal, coverage, mean_width  # noqa: E402
from src.utils.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)
RUNS = Path("runs")
# GAP6: width/coverage error bars over seeds. 6 clean regression datasets (gsht quarantined).
REG = ["esol", "freesolv", "lipophilicity", "bace", "qm7", "qm8"]
MODELS = ["rf", "chemprop", "molformer", "chemfm", "unimol", "tabpfn"]
MLAB = {"rf": "RF", "chemprop": "Chemprop", "molformer": "MolFormer", "chemfm": "ChemFM",
        "unimol": "Uni-Mol", "tabpfn": "TabPFN"}
COLORS = {"rf": "#888", "chemprop": "#9467bd", "molformer": "#1f77b4", "chemfm": "#d62728",
          "unimol": "#2ca02c", "tabpfn": "#9467bd"}
SEEDS = [42, 1337, 2024]
ALPHA = 0.1


def _preds(dataset, model, seed):
    for d in reversed(sorted(RUNS.glob(f"*_{dataset}_{model}_seed{seed}"))):
        vp, tp = d / "val_predictions.csv", d / "test_predictions.csv"
        if vp.exists() and tp.exists():
            return pd.read_csv(vp), pd.read_csv(tp)
    return None, None


def main() -> int:
    configure_logging("INFO")
    rows = []
    for dataset in REG:
        for model in MODELS:
            ws, cs, ns = [], [], 0
            for seed in SEEDS:
                valdf, testdf = _preds(dataset, model, seed)
                if valdf is None:
                    continue
                lo, hi, _ = split_conformal(
                    valdf["y_true"].to_numpy(), valdf["y_score"].to_numpy(),
                    testdf["y_score"].to_numpy(), ALPHA)
                ws.append(mean_width(lo, hi))
                cs.append(coverage(testdf["y_true"].to_numpy(), lo, hi))
                ns += 1
            if ns == 0:
                continue
            rows.append({
                "dataset": dataset, "model": model, "n_seeds": ns,
                "width_mean": np.mean(ws), "width_std": np.std(ws),
                "cov_mean": np.mean(cs), "cov_std": np.std(cs),
            })
            logger.info(f"{dataset:14s} {model:10s} n={ns} "
                        f"width={np.mean(ws):.2f}±{np.std(ws):.2f} "
                        f"cov={np.mean(cs):.3f}±{np.std(cs):.3f}")

    df = pd.DataFrame(rows)
    Path("results/final").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/final/conformal_multiseed.csv", index=False)

    # Table.
    print("\n" + "=" * 84)
    print("MULTI-SEED CONFORMAL @ 90% (width mean±std | coverage mean±std; n seeds)")
    print("=" * 84)
    look = {(r["dataset"], r["model"]): r for r in rows}
    print(f"{'dataset':<14}" + "".join(f"{MLAB[m]:>14}" for m in MODELS))
    for d in REG:
        line = f"{d:<14}"
        for m in MODELS:
            r = look.get((d, m))
            line += f"{r['width_mean']:>7.2f}±{r['width_std']:.2f}" if r else f"{'--':>14}"
        print(line)
    print("\nCoverage (mean±std), * = under-covers (mean+std < 0.90):")
    for d in REG:
        line = f"{d:<14}"
        for m in MODELS:
            r = look.get((d, m))
            if r:
                flag = "*" if (r["cov_mean"] + r["cov_std"]) < 0.90 else " "
                line += f"{r['cov_mean']:>6.3f}{flag}      "
            else:
                line += f"{'--':>14}"
        print(line)

    # Figure: width with error bars.
    fig, ax = plt.subplots(figsize=(9, 4.5)); x = np.arange(len(REG)); w = 0.16
    for i, m in enumerate(MODELS):
        means = [look.get((d, m), {}).get("width_mean", np.nan) for d in REG]
        errs = [look.get((d, m), {}).get("width_std", 0) for d in REG]
        ax.bar(x + i * w, means, w, yerr=errs, capsize=2, label=MLAB[m], color=COLORS[m])
    ax.set_xticks(x + 2 * w); ax.set_xticklabels(REG, rotation=10)
    ax.set_ylabel("90% conformal interval width (mean±std, 3 seeds)")
    ax.set_title("Multi-seed conformal interval width across 5 backbones")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/figures/conformal_multiseed.png", dpi=150)
    print("\nsaved results/final/conformal_multiseed.csv + results/figures/conformal_multiseed.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
