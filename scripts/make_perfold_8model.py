"""B finalization: collect the 450 deep-CV fold predictions and merge them with the 5 CPU-tier
models into ONE 8-model per-fold omnibus (the Option-B unified significance). Deep fold-MAE =
mean over the 3 inner seeds per (model, repeat, fold). Reuses perfold_significance's Friedman /
Iman-Davenport / Nadeau-Bengio / Holm machinery.

NOTE on inference scope (Codex review CRITICAL 2): the pre-registration (perfold_significance.py)
fixes the CONFIRMATORY family as TabPFN vs the four classic models (Ridge/SVR/GBM/RF). The three
deep models (MolFormer/ChemFM/Chemprop) were pre-registered as held out, so their rows in this
8-model NB table are EXPLORATORY/descriptive, not confirmatory -- report them as such.

Run after the deep CV finishes (activate the `drug` env first):
  python scripts/make_perfold_8model.py
Output: results/final/perfold_8model_significance.csv + perfold_8model_omnibus.csv + raw matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.perfold_significance import (  # noqa: E402
    CONTROL, holm, nb_ttest, omnibus, perm_p,
)

RAW = Path("results/raw"); FINAL = Path("results/final")
DEEP = {"molformer": "MolFormer", "chemfm": "ChemFM", "chemprop": "Chemprop"}
SEEDS = [42, 1337, 2024]
CPU_MODELS = ["Ridge", "SVR", "GBM", "RF", "TabPFN"]
ALL = CPU_MODELS + list(DEEP.values())


def collect_deep() -> pd.DataFrame:
    rows = []
    for m, lab in DEEP.items():
        for R in range(5):
            for F in range(10):
                maes = []
                for s in SEEDS:
                    p = Path(f"runs/deepcv/{m}_primary_r{R}_f{F}_seed{s}/test_predictions.csv")
                    if p.exists():
                        df = pd.read_csv(p)
                        c = "y_score" if "y_score" in df.columns else "y_pred"
                        maes.append(float(np.abs(df.y_true - df[c]).mean()))
                if maes:
                    rows.append({"repeat": R, "fold": F, "model": lab,
                                 "MAE": float(np.mean(maes)), "n_seeds": len(maes)})
    return pd.DataFrame(rows)


def main():
    cpu = pd.read_csv(RAW / "perfold_mae_primary.csv")
    cpu = cpu[cpu.model.isin(CPU_MODELS)][["repeat", "fold", "model", "MAE"]]
    deep = collect_deep()
    print(f"deep fold rows collected: {len(deep)} (expect 150 = 3 models x 50 folds)")
    long = pd.concat([cpu, deep[["repeat", "fold", "model", "MAE"]]], ignore_index=True)
    M = long.pivot_table(index=["repeat", "fold"], columns="model", values="MAE")
    M = M.dropna()  # keep folds where all 8 models present
    models = [m for m in ALL if m in M.columns]
    print(f"8-model matrix: N={M.shape[0]} folds x k={len(models)} models ({models})")
    long.to_csv(RAW / "perfold_8model_mae.csv", index=False)

    omni = omnibus(M, models)
    rows = []
    for m in [x for x in models if x != CONTROL]:
        d = (M[m] - M[CONTROL]).to_numpy(float)
        mean, tval, p = nb_ttest(d, 1.0 / 9)
        _, _, p_alt = nb_ttest(d, 0.25)
        rows.append({"model": m, "dMAE_vs_TabPFN": round(mean, 4), "NB_t": round(tval, 3),
                     "NB_p_rho.111": round(p, 4), "NB_p_rho.25": round(p_alt, 4),
                     "perm_p": round(perm_p(d), 4), "frac_folds_worse": round((d > 0).mean(), 3)})
    conf = pd.DataFrame(rows)
    # Holm-correct BOTH rho assumptions (the manuscript reports both as Holm-corrected).
    conf["NB_holm_rho.111"] = np.round(holm(conf["NB_p_rho.111"].to_numpy(float)), 4)
    conf["NB_holm_rho.25"] = np.round(holm(conf["NB_p_rho.25"].to_numpy(float)), 4)
    conf.to_csv(FINAL / "perfold_8model_significance.csv", index=False)
    pd.DataFrame([{"N_folds": omni["N_folds"], "friedman_p": omni["friedman_p"],
                   "iman_davenport_p": omni["iman_davenport_p"],
                   **{f"avgrank_{k}": round(v, 2) for k, v in omni["avg_rank"].items()}}]).to_csv(
        FINAL / "perfold_8model_omnibus.csv", index=False)

    print(f"\nomnibus (8 models, N={omni['N_folds']} folds): Friedman p={omni['friedman_p']:.2e} | "
          f"Iman-Davenport p={omni['iman_davenport_p']:.2e}")
    print("avg rank:", {k: round(v, 2) for k, v in sorted(omni["avg_rank"].items(), key=lambda kv: kv[1])})
    print("\nNB-t vs TabPFN (Holm) -- CONFIRMATORY for Ridge/SVR/GBM/RF; EXPLORATORY for the deep models:")
    print(conf.sort_values("dMAE_vs_TabPFN").to_string(index=False))
    print(f"\nper-fold mean MAE: " + ", ".join(f"{m}={M[m].mean():.3f}" for m in models))
    print("wrote results/final/perfold_8model_significance.csv + perfold_8model_omnibus.csv")


if __name__ == "__main__":
    main()
