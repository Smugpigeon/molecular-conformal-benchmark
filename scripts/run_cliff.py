"""Activity-cliff stratified benchmark — first version (RF backbone).

Novelty thrust (项目调研汇总.md v2 §7.5.1). For each of the 30 MoleculeACE
datasets: train RF on train, predict test, report RMSE on cliff vs non-cliff
test molecules. Aggregates the cliff/non-cliff RMSE ratio across datasets.

Extending to MolFormer/Uni-Mol/ChemFM (the actual novelty -- SemiMol skipped
them) reuses this stratification; this RF version validates the pipeline.

Run: python scripts/run_cliff.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.featurizers import batch_morgan  # noqa: E402
from src.eval.cliff import cliff_stratified_rmse, list_moleculeace, load_moleculeace  # noqa: E402
from src.utils.logging_setup import configure_logging  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402

logger = logging.getLogger(__name__)


def run_dataset(name: str, seed: int = 42) -> dict | None:
    set_all_seeds(seed)
    train, test, target = load_moleculeace(name)
    Xtr, itr = batch_morgan(train["smiles"].tolist())
    Xte, ite = batch_morgan(test["smiles"].tolist())
    ytr = train[target].to_numpy()[itr]
    yte = test[target].to_numpy()[ite]
    cliff = test["cliff_mol"].to_numpy()[ite].astype(bool)
    if cliff.sum() < 3 or (~cliff).sum() < 3:
        return None  # too few in a stratum for stable RMSE

    model = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=seed)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    res = cliff_stratified_rmse(yte, pred, cliff)
    res["dataset"] = name
    res["model"] = "rf"
    return res


def main() -> int:
    configure_logging("INFO")
    datasets = list_moleculeace()
    logger.info(f"Found {len(datasets)} MoleculeACE datasets")

    rows = []
    for d in datasets:
        try:
            r = run_dataset(d)
            if r:
                rows.append(r)
                logger.info(
                    f"{d:22s} cliff={r['rmse_cliff']:.3f} "
                    f"noncliff={r['rmse_noncliff']:.3f} ratio={r['cliff_ratio']:.2f}"
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"{d}: {e}")

    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/cliff_rf.csv", index=False)

    ratio = df["cliff_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    print("\n" + "=" * 60)
    print("ACTIVITY-CLIFF STRATIFIED RMSE (RF, 30 MoleculeACE datasets)")
    print("=" * 60)
    print(f"datasets evaluated      = {len(df)}")
    print(f"mean RMSE on cliffs     = {df['rmse_cliff'].mean():.3f}")
    print(f"mean RMSE on non-cliffs = {df['rmse_noncliff'].mean():.3f}")
    print(f"mean cliff/non ratio    = {ratio.mean():.2f}  (>1 = worse on cliffs)")
    print(f"datasets where cliff worse = {(ratio > 1).sum()}/{len(ratio)}")

    # Figure: cliff vs non-cliff RMSE scatter across datasets.
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["rmse_noncliff"], df["rmse_cliff"], s=40, alpha=0.7, edgecolor="k")
    lim = [0, max(df["rmse_cliff"].max(), df["rmse_noncliff"].max()) * 1.05]
    ax.plot(lim, lim, "k--", label="equal")
    ax.set_xlabel("RMSE on non-cliff test molecules")
    ax.set_ylabel("RMSE on activity-cliff test molecules")
    ax.set_title(f"RF: activity cliffs are harder\n"
                 f"({(ratio > 1).sum()}/{len(ratio)} datasets above diagonal)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend()
    fig.tight_layout()
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("results/figures/cliff_rf.png", dpi=150)
    print("saved results/cliff_rf.csv + results/figures/cliff_rf.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
