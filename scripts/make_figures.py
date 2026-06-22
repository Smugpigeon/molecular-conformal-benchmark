"""Generate report/PPT figures from aggregated results + saved predictions.

Produces (under results/figures/):
  1. bench_cls_roc_auc.png   - grouped bars: ROC-AUC, 4 backbones x 3 cls tasks
  2. bench_reg_rmse.png      - grouped bars: RMSE, 4 backbones x 6 reg tasks
  3. roc_curves_<dataset>.png- ROC overlay of 4 backbones per cls dataset
  4. confusion_<dataset>.png - confusion matrix of best backbone per cls dataset

ROC/confusion need val_predictions.csv (saved by train.py after the
predictions patch). Run scripts/sweep_predictions.sh first.

Run: python scripts/make_figures.py
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

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: E402

from src.utils.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)

FIG = Path("results/figures")
RUNS = Path("runs")
MODELS = ["rf", "molformer", "chemfm", "unimol"]
MODEL_LABEL = {"rf": "RF+Morgan", "molformer": "MolFormer-XL",
               "chemfm": "ChemFM-3B", "unimol": "Uni-Mol"}
CLS = ["esol_cls", "bace_cls", "bbbp_cls"]
REG = ["esol", "freesolv", "lipophilicity", "bace", "bbbp", "gsht"]
COLORS = {"rf": "#888", "molformer": "#1f77b4", "chemfm": "#d62728", "unimol": "#2ca02c"}


def _latest_pred(dataset: str, model: str, seed: int = 42) -> Path | None:
    cands = sorted(RUNS.glob(f"*_{dataset}_{model}_seed{seed}"))
    for d in reversed(cands):
        p = d / "val_predictions.csv"
        if p.exists():
            return p
    return None


def _mean_std(df: pd.DataFrame, dataset: str, model: str, metric: str) -> tuple[float, float]:
    """Mean and std of a metric over seeds for one (dataset, model)."""
    sub = df[(df.dataset == dataset) & (df.model == model)][metric].dropna()
    if len(sub) == 0:
        return float("nan"), 0.0
    return float(sub.mean()), float(sub.std(ddof=0))


def _grouped_bars(df, datasets, metric, ylabel, title, out, ylim=None) -> None:
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(datasets)), 4.5))
    width = 0.2
    x = np.arange(len(datasets))
    for i, m in enumerate(MODELS):
        means = [_mean_std(df, d, m, metric)[0] for d in datasets]
        errs = [_mean_std(df, d, m, metric)[1] for d in datasets]
        ax.bar(x + i * width, means, width, yerr=errs, capsize=3,
               label=MODEL_LABEL[m], color=COLORS[m])
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(datasets, rotation=15)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"saved {out.name}")


def bar_charts(results_csv: Path) -> None:
    # final_all.csv is per-run (one row per seed, raw float metrics).
    df = pd.read_csv(results_csv)
    _grouped_bars(df, CLS, "ROC-AUC", "ROC-AUC (higher better)",
                  "Classification: ROC-AUC by backbone",
                  FIG / "bench_cls_roc_auc.png", ylim=(0.8, 1.0))
    _grouped_bars(df, REG, "RMSE", "RMSE (lower better)",
                  "Regression: RMSE by backbone",
                  FIG / "bench_reg_rmse.png")


def roc_and_confusion() -> None:
    for dataset in CLS:
        # ---- ROC overlay ----
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
        best = (None, -1.0, None)  # (model, auc, df)
        for m in MODELS:
            p = _latest_pred(dataset, m)
            if p is None:
                logger.warning(f"no predictions for {dataset}/{m}")
                continue
            d = pd.read_csv(p)
            auc = roc_auc_score(d.y_true, d.y_score)
            fpr, tpr, _ = roc_curve(d.y_true, d.y_score)
            ax.plot(fpr, tpr, color=COLORS[m], label=f"{MODEL_LABEL[m]} ({auc:.3f})")
            if auc > best[1]:
                best = (m, auc, d)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"ROC — {dataset}")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(FIG / f"roc_{dataset}.png", dpi=150)
        plt.close(fig)
        logger.info(f"saved roc_{dataset}.png")

        # ---- Confusion matrix of best backbone ----
        if best[2] is None:
            continue
        m, auc, d = best
        cm = confusion_matrix(d.y_true, (d.y_score >= 0.5).astype(int))
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=14)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["neg", "pos"]); ax.set_yticklabels(["neg", "pos"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion — {dataset}\nbest: {MODEL_LABEL[m]} (AUC {auc:.3f})")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(FIG / f"confusion_{dataset}.png", dpi=150)
        plt.close(fig)
        logger.info(f"saved confusion_{dataset}.png")


def main() -> int:
    configure_logging("INFO")
    FIG.mkdir(parents=True, exist_ok=True)
    results_csv = Path("results/final_all.csv")
    if results_csv.exists():
        bar_charts(results_csv)
    else:
        logger.warning(f"{results_csv} missing; skip bar charts")
    roc_and_confusion()
    logger.info("Figures done -> results/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
