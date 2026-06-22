"""Full conformal benchmark across 4 backbones (reads saved predictions).

Novelty thrust (项目调研汇总.md v2 §7.5.1): which molecular representation
gives the tightest, coverage-valid split-conformal intervals? Reads the
val (calibration) + test (evaluation) predictions saved by the ppi_bias eval
sweep (save_test_preds=true). Split conformal is used for ALL backbones so
the interval-width comparison is fair (no model-specific uncertainty needed).

Run: python scripts/run_conformal_full.py
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

from src.eval.conformal import evaluate_conformal  # noqa: E402
from src.utils.logging_setup import configure_logging  # noqa: E402

logger = logging.getLogger(__name__)
RUNS = Path("runs")
# Paper-track conformal sweep: 6 clean regression datasets (gsht quarantined from the
# headline per the §16 scaffold-leakage + authorship-conflict caveats; QM7/QM8 added as
# the low-noise / large-n regime). TabPFN added as the small-sample tabular FM.
REG = ["esol", "freesolv", "lipophilicity", "bace", "qm7", "qm8"]
MODELS = ["rf", "molformer", "chemfm", "unimol", "tabpfn"]
MLAB = {"rf": "RF+Morgan", "molformer": "MolFormer-XL", "chemfm": "ChemFM-3B",
        "unimol": "Uni-Mol", "tabpfn": "TabPFN"}
COLORS = {"rf": "#888", "molformer": "#1f77b4", "chemfm": "#d62728",
          "unimol": "#2ca02c", "tabpfn": "#9467bd"}


def _latest_preds(dataset: str, model: str, seed: int = 42):
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
            valdf, testdf = _latest_preds(dataset, model)
            if valdf is None:
                logger.warning(f"missing preds: {dataset}/{model}")
                continue
            res = evaluate_conformal(
                valdf["y_true"].to_numpy(), valdf["y_score"].to_numpy(),
                testdf["y_true"].to_numpy(), testdf["y_score"].to_numpy(),
                alpha=0.1,
            )
            res["test_mae"] = float(np.mean(np.abs(
                testdf["y_true"].to_numpy() - testdf["y_score"].to_numpy())))
            res.update({"dataset": dataset, "model": model})
            rows.append(res)
            logger.info(
                f"{dataset:14s} {model:10s} cov={res['empirical_coverage']:.3f} "
                f"width={res['mean_width']:.3f}"
            )

    df = pd.DataFrame(rows)
    # Single source of truth: results/final/ (the paper reads here; avoid a stale root duplicate).
    Path("results/final").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/final/conformal_full.csv", index=False)

    # Width comparison table (only datasets where all backbones present).
    print("\n" + "=" * 72)
    print("CONFORMAL INTERVAL WIDTH @ 90% (lower = tighter; coverage in parens)")
    print("=" * 72)
    print(f"{'dataset':<14}" + "".join(f"{MLAB[m]:>16}" for m in MODELS))
    width_lookup = {(r["dataset"], r["model"]): r for r in rows}
    for d in REG:
        line = f"{d:<14}"
        for m in MODELS:
            r = width_lookup.get((d, m))
            line += f"{r['mean_width']:>10.2f}({r['empirical_coverage']:.2f})" if r else f"{'--':>16}"
        print(line)

    # Figure: grouped bars of interval width by backbone.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(REG)); w = 0.2
    for i, m in enumerate(MODELS):
        widths = [width_lookup.get((d, m), {}).get("mean_width", np.nan) for d in REG]
        ax.bar(x + i * w, widths, w, label=MLAB[m], color=COLORS[m])
    ax.set_xticks(x + 1.5 * w); ax.set_xticklabels(REG, rotation=15)
    ax.set_ylabel("90% conformal interval width (tighter = better)")
    ax.set_title("Which backbone gives the tightest valid prediction intervals?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/figures/conformal_full.png", dpi=150)

    # Coverage validity check: all should be >= ~0.88 (target 0.90, finite-sample).
    bad = df[df["empirical_coverage"] < 0.85]
    if len(bad):
        print("\nWARNING: under-coverage (>5% below target) in:")
        for _, r in bad.iterrows():
            print(f"  {r['dataset']}/{r['model']}: {r['empirical_coverage']:.3f}")
    else:
        print("\nAll backbones achieve valid coverage (>=0.85 at target 0.90).")
    print("saved results/final/conformal_full.csv + results/figures/conformal_full.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
