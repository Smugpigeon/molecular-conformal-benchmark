"""Single-task vs multi-task on the physicochemical trio (ESOL/FreeSolv/Lipo).

Per 项目调研汇总.md v2 §4.1. Runs 3 seeds, reports per-task val RMSE for both
schemes and the transfer delta (negative delta = multi-task helps). Per
CLAUDE.md §16.1 negative transfer is reported, not hidden.

Run: python scripts/run_mtl.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.mtl import PHYSCHEM_TRIO, run_mtl_experiment
from src.utils.logging_setup import configure_logging
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)
SEEDS = [42, 1337, 2024]


def main() -> int:
    configure_logging("INFO")
    out = Path("results/mtl_trio.csv")

    # Collect RMSE per (scheme, task, seed).
    rows = []
    for seed in SEEDS:
        set_all_seeds(seed)
        res = run_mtl_experiment()
        for scheme, per_task in res.items():
            for task, metrics in per_task.items():
                rows.append({
                    "seed": seed, "scheme": scheme, "task": task,
                    "RMSE": metrics["RMSE"], "PearsonR": metrics["PearsonR"],
                })
        logger.info(f"seed {seed} done")

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # Aggregate: mean RMSE per (scheme, task).
    agg = df.groupby(["task", "scheme"])["RMSE"].agg(["mean", "std"]).reset_index()
    pivot = agg.pivot(index="task", columns="scheme", values="mean")

    print("\n" + "=" * 64)
    print("MULTI-TASK vs SINGLE-TASK — physicochemical trio (val RMSE, n=3 seeds)")
    print("=" * 64)
    print(f"{'task':<14} {'single':>10} {'multi':>10} {'delta':>10}  transfer")
    for task in PHYSCHEM_TRIO:
        st = pivot.loc[task, "single_task"]
        mt = pivot.loc[task, "multi_task"]
        delta = mt - st  # negative = multi-task lower RMSE = helps
        verdict = "POSITIVE (multi helps)" if delta < -0.005 else (
            "NEGATIVE (multi hurts)" if delta > 0.005 else "neutral")
        print(f"{task:<14} {st:>10.3f} {mt:>10.3f} {delta:>+10.3f}  {verdict}")

    print("\nCLAUDE.md §16.1: negative transfer reported honestly, not hidden.")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
