"""Consolidate ALL models on the shared bace_clean split into one comparison:
classic ML (Ridge/SVR/GBM/RF, Optuna-tuned, local), TabPFN (zero-tuning, GPU), and the three
fine-tuned deep models (MolFormer / ChemFM / Chemprop, 3 seeds each, server).

Everything is scored from per-molecule prediction CSVs with the SAME metric code (MAE / RMSE /
Pearson / Spearman, CLAUDE.md §9). Deep models report mean +/- std over seeds 42/1337/2024.

Inputs (all local after the pull step):
  results/tuning/{ridge,svr,gbm,rf}_morgan_{val,test}.csv     classic (single, Optuna-tuned)
  results/bace_clean_final/tabpfn_{desc,morgan512}_{val,test}.csv
  results/bace_clean_final/{molformer,chemfm,chemprop}_seed{42,1337,2024}_{val,test}.csv

Outputs:
  results/final/master_table.csv   one row per model (val + test, 4 metrics, +/- std for deep)
  results/figures/fig_all_models_{mae,pearson}_j.png   headline comparison
  results/figures/fig_deep_overfit_j.png               val-vs-test gap (early-stopping check)

Run: /opt/anaconda3/bin/python3 scripts/build_final_comparison.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.metrics import regression_metrics  # noqa: E402

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix", "axes.unicode_minus": True})
FIG = Path("results/figures"); FIG.mkdir(parents=True, exist_ok=True)
FINAL = Path("results/final"); FINAL.mkdir(parents=True, exist_ok=True)
TUN = Path("results/tuning"); BC = Path("results/bace_clean_final")
SEEDS = [42, 1337, 2024]


def score(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = pd.read_csv(path)
    pred = d["y_pred"] if "y_pred" in d.columns else d["y_score"]
    return regression_metrics(d["y_true"].to_numpy(float), pred.to_numpy(float))


def agg(metric_dicts: list[dict]) -> dict:
    """mean +/- std across seeds for each metric."""
    keys = metric_dicts[0]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metric_dicts], dtype=float)
        out[k] = float(vals.mean()); out[k + "_std"] = float(vals.std(ddof=0))
    return out


def main() -> None:
    rows = []
    # ---- classic (single Optuna-tuned run, local) ----
    for m, label in [("ridge", "Ridge"), ("svr", "SVR"), ("gbm", "GBM"), ("rf", "RF")]:
        v = score(TUN / f"{m}_morgan_val.csv"); t = score(TUN / f"{m}_morgan_test.csv")
        if t:
            rows.append({"model": label, "family": "classic", "rep": "Morgan-2048",
                         "n_seed": 1, "val": v, "test": t})
    # ---- TabPFN (zero-tuning) ----
    for rep, label in [("desc", "TabPFN (desc)"), ("morgan512", "TabPFN (FP512)")]:
        v = score(BC / f"tabpfn_{rep}_val.csv"); t = score(BC / f"tabpfn_{rep}_test.csv")
        if t:
            rows.append({"model": label, "family": "tabpfn", "rep": rep,
                         "n_seed": 1, "val": v, "test": t})
    # ---- deep (3 seeds) ----
    for m, label in [("molformer", "MolFormer-XL"), ("chemfm", "ChemFM-3B"),
                     ("chemprop", "Chemprop")]:
        vs = [score(BC / f"{m}_seed{s}_val.csv") for s in SEEDS]
        ts = [score(BC / f"{m}_seed{s}_test.csv") for s in SEEDS]
        vs = [x for x in vs if x]; ts = [x for x in ts if x]
        if ts:
            rows.append({"model": label, "family": "deep", "rep": "SMILES/graph",
                         "n_seed": len(ts), "val": agg(vs), "test": agg(ts)})

    # ---- master table ----
    flat = []
    for r in rows:
        row = {"model": r["model"], "family": r["family"], "rep": r["rep"], "n_seed": r["n_seed"]}
        for sp in ("val", "test"):
            for k in ("MAE", "RMSE", "PearsonR", "SpearmanRho"):
                row[f"{sp}_{k}"] = round(r[sp][k], 4)
                if k + "_std" in r[sp]:
                    row[f"{sp}_{k}_std"] = round(r[sp][k + "_std"], 4)
        flat.append(row)
    T = pd.DataFrame(flat)
    T.to_csv(FINAL / "master_table.csv", index=False)
    show = ["model", "family", "test_MAE", "test_RMSE", "test_PearsonR", "test_SpearmanRho"]
    print(T[show].to_string(index=False))
    print(f"\nwrote {FINAL/'master_table.csv'}")

    # ---- headline figures: test MAE + test Pearson across all models ----
    order = T.sort_values("test_MAE").reset_index(drop=True)
    cmap = {"classic": "#4C72B0", "tabpfn": "#C44E52", "deep": "#55A868"}
    for metric, better, fname, ttl in [
        ("test_MAE", "lower", "fig_all_models_mae_j.png", "Test MAE (lower = better)"),
        ("test_PearsonR", "higher", "fig_all_models_pearson_j.png", "Test Pearson R (higher = better)")]:
        od = T.sort_values(metric, ascending=(better == "lower")).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        x = np.arange(len(od)); colors = [cmap[f] for f in od.family]
        err = od[metric + "_std"].fillna(0) if metric + "_std" in od.columns else None
        ax.bar(x, od[metric], color=colors, edgecolor="k", linewidth=0.4,
               yerr=err, capsize=3, error_kw={"elinewidth": 0.8})
        for i, v in enumerate(od[metric]):
            ax.annotate(f"{v:.3f}", (i, v), ha="center",
                        va="bottom" if better == "higher" else "top",
                        xytext=(0, 3 if better == "higher" else -12), textcoords="offset points",
                        fontsize=7.5)
        ax.set_xticks(x); ax.set_xticklabels(od.model, rotation=28, ha="right", fontsize=8.5)
        ax.set_ylabel(metric.replace("test_", "test "))
        handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[k]) for k in cmap]
        ax.legend(handles, ["classic ML (tuned)", "TabPFN (no tuning)", "deep FT (3 seeds)"],
                  fontsize=8, loc="best")
        ax.set_title(f"BACE all-model comparison -- {ttl}  (shared bace_clean split, n_test=303)",
                     fontsize=10.5)
        fig.tight_layout(); fig.savefig(FIG / fname, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {fname}")

    # ---- overfitting check: val vs test MAE (best-on-val early stopping should keep them close) ----
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for _, r in T.iterrows():
        ax.scatter(r.val_MAE, r.test_MAE, s=70, color=cmap[r.family], edgecolor="k",
                   linewidth=0.5, zorder=3)
        ax.annotate(r.model, (r.val_MAE, r.test_MAE), fontsize=7,
                    xytext=(4, 3), textcoords="offset points")
    lo = float(min(T.val_MAE.min(), T.test_MAE.min())) - 0.05
    hi = float(max(T.val_MAE.max(), T.test_MAE.max())) + 0.05
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.5", lw=1, label="val = test (no over-fit)")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("validation MAE"); ax.set_ylabel("test MAE")
    ax.set_title("Val vs test MAE -- points on the line = no over-fitting", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig_deep_overfit_j.png", dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote fig_deep_overfit_j.png")


if __name__ == "__main__":
    main()
