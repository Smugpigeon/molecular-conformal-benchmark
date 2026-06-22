"""Test-set evaluation entry point.

Per CLAUDE.md §8.4: the test set is only viewed ONCE before paper submission.
This script enforces it by requiring `--final` for test evaluation.

Usage:
    python -m src.evaluate --run-id <hydra_run_dir>           # val
    python -m src.evaluate --run-id <run> --split test --final  # paper-final
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.data.featurizers import batch_morgan
from src.data.loaders import load_dataset
from src.utils.logging_setup import configure_logging
from src.utils.metrics import metrics_for_task
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


def main() -> int:
    configure_logging("INFO")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        required=True,
        type=Path,
        help="Path to run dir produced by src.train (contains model.pkl + config.yaml)",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Split to evaluate. Test requires --final.",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help=(
            "Required to evaluate test set. CLAUDE.md §8.4: test set is only "
            "viewed once before paper submission."
        ),
    )
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        logger.error(
            "Refusing to evaluate test set without --final.\n"
            "CLAUDE.md §8.4: test set is only viewed once before paper. "
            "If you really mean it, re-run with --final and commit a paper-ready note."
        )
        return 2

    run_dir = args.run_id if args.run_id.exists() else Path("runs") / args.run_id
    if not run_dir.exists():
        logger.error(f"Run dir not found: {run_dir}")
        return 1

    cfg = OmegaConf.load(run_dir / "config.yaml")
    set_all_seeds(cfg.seed)

    artifact = pickle.loads((run_dir / "model.pkl").read_bytes())
    model = artifact["model"]
    logger.info(f"Loaded model trained with sklearn=={artifact['sklearn_version']}")

    split = load_dataset(cfg.dataset.name, data_root=cfg.paths.data)
    eval_df = getattr(split, args.split)
    X, idx = batch_morgan(eval_df["smiles"].tolist())
    y = eval_df["label"].to_numpy()[idx]

    if split.task == "regression":
        y_pred = model.predict(X)
    else:
        y_pred = model.predict_proba(X)[:, 1]
    metrics = metrics_for_task(split.task, y, y_pred)
    logger.info(f"{args.split.upper()} metrics ({split.name}): {metrics}")

    out = run_dir / f"{args.split}_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
