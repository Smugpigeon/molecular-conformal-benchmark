"""Permuted-label negative control.

Per CLAUDE.md §16.8: every reported metric must be compared against a
permuted-label baseline. If a model trained on shuffled labels achieves
non-trivial performance, the dataset has leak (features carry the label
indirectly) — and any "real" model result is suspect.

Sanity expectations:
- regression: shuffled-label Pearson R ≈ 0 (tolerance |R| < 0.15)
- classification: shuffled-label ROC-AUC ≈ 0.5 (tolerance |AUC - 0.5| < 0.05)

Run: `python scripts/negative_control.py [--n-repeats 10]`
Output: results/negative_control.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.data.featurizers import batch_morgan
from src.data.loaders import DATA_REGISTRY, load_dataset
from src.utils.logging_setup import configure_logging
from src.utils.metrics import metrics_for_task
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


def run_negative_control(
    name: str,
    data_root: Path,
    n_repeats: int = 10,
    n_estimators: int = 200,
    seed_base: int = 42,
) -> dict:
    """Train RF + Morgan on shuffled labels n_repeats times."""
    split = load_dataset(name, data_root=data_root)
    X_train, idx_train = batch_morgan(split.train["smiles"].tolist())
    X_val, idx_val = batch_morgan(split.val["smiles"].tolist())
    y_train = split.train["label"].to_numpy()[idx_train]
    y_val = split.val["label"].to_numpy()[idx_val]

    main_metric = "PearsonR" if split.task == "regression" else "ROC-AUC"
    results: list[float] = []

    for r in range(n_repeats):
        set_all_seeds(seed_base + r)
        # Shuffle training labels — features unchanged.
        y_train_shuf = np.random.permutation(y_train)

        if split.task == "regression":
            model = RandomForestRegressor(
                n_estimators=n_estimators, n_jobs=-1, random_state=seed_base + r
            )
            model.fit(X_train, y_train_shuf)
            pred = model.predict(X_val)
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators, n_jobs=-1,
                random_state=seed_base + r, class_weight="balanced",
            )
            model.fit(X_train, y_train_shuf.astype(int))
            pred = model.predict_proba(X_val)[:, 1]

        m = metrics_for_task(split.task, y_val, pred)
        results.append(m[main_metric])

    arr = np.asarray(results)
    return {
        "dataset": name,
        "task": split.task,
        "main_metric": main_metric,
        "permuted_mean": float(arr.mean()),
        "permuted_std": float(arr.std()),
        "permuted_min": float(arr.min()),
        "permuted_max": float(arr.max()),
        "n_repeats": n_repeats,
    }


def sanity_check(row: dict) -> tuple[bool, str]:
    """Return (passed, message). Trips if shuffled labels still predict."""
    if row["task"] == "regression":
        if abs(row["permuted_mean"]) > 0.15:
            return False, (
                f"|PearsonR|={abs(row['permuted_mean']):.3f} > 0.15 — "
                "features may leak label info"
            )
        return True, f"OK: PearsonR={row['permuted_mean']:+.3f} ≈ 0"
    # classification
    if abs(row["permuted_mean"] - 0.5) > 0.05:
        return False, (
            f"|ROC-AUC - 0.5|={abs(row['permuted_mean'] - 0.5):.3f} > 0.05 — "
            "possible class imbalance / label leak"
        )
    return True, f"OK: ROC-AUC={row['permuted_mean']:.3f} ≈ 0.5"


def main() -> int:
    configure_logging("INFO")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/negative_control.csv"))
    args = parser.parse_args()

    rows = []
    for name in DATA_REGISTRY:
        logger.info(f"Negative control on {name} ({args.n_repeats} permutations) ...")
        result = run_negative_control(name, args.data_root, args.n_repeats)
        rows.append(result)
        logger.info(
            f"  permuted {result['main_metric']}: "
            f"{result['permuted_mean']:+.3f} ± {result['permuted_std']:.3f} "
            f"(range [{result['permuted_min']:+.3f}, {result['permuted_max']:+.3f}])"
        )

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Sanity
    logger.info("=" * 70)
    logger.info("NEGATIVE CONTROL SANITY CHECK (per CLAUDE.md §16.8)")
    logger.info("=" * 70)
    n_failed = 0
    for r in rows:
        passed, msg = sanity_check(r)
        status = "PASS" if passed else "FAIL"
        logger.warning if not passed else logger.info
        getattr(logger, "info" if passed else "warning")(
            f"  [{status}] {r['dataset']:<14}: {msg}"
        )
        if not passed:
            n_failed += 1

    if n_failed:
        logger.warning(
            f"\n{n_failed}/{len(rows)} datasets FAILED negative control. "
            "Investigate before publishing any result on those datasets."
        )
    else:
        logger.info(
            "\nAll datasets pass — shuffled labels produce ≈ random predictions, "
            "so non-shuffled model gains are genuine signal."
        )
    logger.info(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
