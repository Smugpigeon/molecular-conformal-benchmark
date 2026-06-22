"""Training entry point.

Hydra-driven. Per CLAUDE.md §5:
    python -m src.train dataset=gsht model=rf
    python -m src.train dataset=bbbp model=rf seed=1337

CLAUDE.md §8.1: set_all_seeds() is the first line of main().
CLAUDE.md §8.4: this script reports validation metrics only. Use
`src/evaluate.py` (with --final flag) to touch the test set.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.data.featurizers import batch_morgan
from src.data.loaders import load_dataset
from src.models.baseline_rf import build_rf
from src.utils.logging_setup import configure_logging
from src.utils.metrics import metrics_for_task
from src.utils.seed import set_all_seeds

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # CLAUDE.md §8.1: seeds first.
    set_all_seeds(cfg.seed, deterministic=cfg.get("deterministic", True))

    # Hydra owns the run directory; mirror its working dir as our run dir.
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    configure_logging(cfg.get("log_level", "INFO"), log_dir=run_dir)

    logger.info("=" * 60)
    logger.info("CONFIG:\n" + OmegaConf.to_yaml(cfg))
    logger.info(f"Run dir: {run_dir}")

    # ---- 1. Load data ----
    split = load_dataset(cfg.dataset.name, data_root=cfg.paths.data)
    logger.info(f"Split: {split.summary()}")

    # ---- 2. Featurize ----
    # Currently RF + Morgan only. Foundation models will branch here later.
    X_train, idx_train = batch_morgan(split.train["smiles"].tolist())
    X_val, idx_val = batch_morgan(split.val["smiles"].tolist())
    y_train = split.train["label"].to_numpy()[idx_train]
    y_val = split.val["label"].to_numpy()[idx_val]

    # ---- 3. Train + eval (dispatched by model.name) ----
    if cfg.model.name == "rf":
        # Featurize + train + eval (sklearn path)
        model = build_rf(
            task=split.task,
            n_estimators=cfg.model.n_estimators,
            random_state=cfg.seed,
        )
        logger.info(f"Training {type(model).__name__} on {len(X_train)} samples...")
        model.fit(X_train, y_train)

        if split.task == "regression":
            y_pred_val = model.predict(X_val)
        else:
            y_pred_val = model.predict_proba(X_val)[:, 1]
        val_metrics = metrics_for_task(split.task, y_val, y_pred_val)

        # Save per-molecule val predictions (enables ROC/confusion figures +
        # §16.4 statistical tests). idx_val aligns predictions to kept rows.
        val_smiles = split.val["smiles"].to_numpy()[idx_val]
        pd.DataFrame(
            {"smiles": val_smiles, "y_true": y_val, "y_score": y_pred_val}
        ).to_csv(run_dir / "val_predictions.csv", index=False)

        # §16.4 one-time test evaluation (conformal/cliff need held-out test preds).
        if cfg.get("save_test_preds", False) and len(split.test) > 0:
            X_test, idx_test = batch_morgan(split.test["smiles"].tolist())
            y_test = split.test["label"].to_numpy()[idx_test]
            if split.task == "regression":
                y_pred_test = model.predict(X_test)
            else:
                y_pred_test = model.predict_proba(X_test)[:, 1]
            test_smiles = split.test["smiles"].to_numpy()[idx_test]
            pd.DataFrame(
                {"smiles": test_smiles, "y_true": y_test, "y_score": y_pred_test}
            ).to_csv(run_dir / "test_predictions.csv", index=False)

        import sklearn  # noqa: PLC0415

        artifact = {
            "model": model,
            "sklearn_version": sklearn.__version__,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "kept_train_idx": idx_train,
        }
        (run_dir / "model.pkl").write_bytes(pickle.dumps(artifact))

    elif cfg.model.name == "molformer":
        # GPU path. Featurization happens inside (tokenizer).
        from src.models.molformer_wrapper import train_molformer  # noqa: PLC0415
        val_metrics = train_molformer(cfg, split, run_dir)

    elif cfg.model.name == "chemfm":
        from src.models.chemfm_wrapper import train_chemfm  # noqa: PLC0415
        val_metrics = train_chemfm(cfg, split, run_dir)

    elif cfg.model.name == "unimol":
        from src.models.unimol_wrapper import train_unimol  # noqa: PLC0415
        val_metrics = train_unimol(cfg, split, run_dir)

    elif cfg.model.name == "chemprop":
        from src.models.chemprop_wrapper import train_chemprop  # noqa: PLC0415
        val_metrics = train_chemprop(cfg, split, run_dir)

    else:
        raise NotImplementedError(
            f"Model {cfg.model.name!r} not implemented yet. "
            "See configs/model/ for planned wrappers."
        )

    logger.info(f"VAL metrics ({cfg.dataset.name}): {val_metrics}")
    (run_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2))
    OmegaConf.save(cfg, run_dir / "config.yaml")
    logger.info(f"Saved artifacts to {run_dir}")


if __name__ == "__main__":
    main()
