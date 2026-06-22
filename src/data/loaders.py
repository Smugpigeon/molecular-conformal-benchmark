"""Dataset loaders.

Per CLAUDE.md §7.2: prefer the predefined train/val/test split provided in
each CSV's `set` column. Scaffold-split variants live under
`data/processed/<name>_scaffold.csv` and must NEVER overwrite the original.

CLAUDE.md §7.1: `data/` is read-only. Any cleaning goes to `data/processed/`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# Registry of datasets shipped with this project.
# Format: alias -> (filename relative to data_root, task type, main metric name)
DATA_REGISTRY: dict[str, tuple[str, str, str]] = {
    # --- Original (regression + 1 classification), teacher-provided ---
    "esol":          ("ESOL.split.csv",          "regression",     "RMSE"),
    "freesolv":      ("FreeSol.split.csv",       "regression",     "RMSE"),
    "lipophilicity": ("Lipophilicity.split.csv", "regression",     "RMSE"),
    "bace":          ("BACE.split.csv",          "regression",     "RMSE"),
    "bbbp":          ("BBBP.split.csv",          "classification", "ROC-AUC"),
    "gsht":          ("GSHt-419.split.csv",      "regression",     "PearsonR"),
    # --- v2 binary-classification variants (built by src.data.binarize) ---
    # 项目调研汇总.md v2 §2: only these 3 have defensible binary thresholds.
    "esol_cls":      ("processed/esol_cls.csv",  "classification", "ROC-AUC"),
    "bace_cls":      ("processed/bace_cls.csv",  "classification", "ROC-AUC"),
    "bbbp_cls":      ("processed/bbbp_cls.csv",  "classification", "ROC-AUC"),
}


@dataclass
class Split:
    """Train/val/test split with task metadata."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    task: str           # "regression" or "classification"
    name: str           # dataset alias (lowercase)
    metric_main: str    # main metric used for model selection

    def summary(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "task": self.task,
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "metric": self.metric_main,
        }


def load_dataset(name: str, data_root: Path | str = "data") -> Split:
    """Load a MoleculeNet dataset with the predetermined split.

    Args:
        name: One of esol, freesolv, lipophilicity, bace, bbbp, gsht
            (case-insensitive).
        data_root: Path to directory containing the CSVs. CLAUDE.md §7 mandates
            this is read-only.

    Returns:
        Split dataclass.

    Raises:
        KeyError: if `name` is not in DATA_REGISTRY.
        FileNotFoundError: if the CSV is missing.
    """
    key = name.lower()
    if key not in DATA_REGISTRY:
        raise KeyError(
            f"Unknown dataset {name!r}. Known: {sorted(DATA_REGISTRY)}"
        )
    filename, task, metric_main = DATA_REGISTRY[key]
    csv_path = Path(data_root) / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {csv_path}. "
            f"CLAUDE.md §7.1: data/ must contain the raw CSVs."
        )

    df = pd.read_csv(csv_path)

    required_cols = {"smiles", "label", "set"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    expected_sets = {"train", "validation", "test"}
    actual_sets = set(df["set"].unique())
    if not actual_sets.issubset(expected_sets):
        unknown = actual_sets - expected_sets
        logger.warning(f"{key}: unexpected set values: {unknown}")

    train = df[df["set"] == "train"].reset_index(drop=True)
    val   = df[df["set"] == "validation"].reset_index(drop=True)
    test  = df[df["set"] == "test"].reset_index(drop=True)

    split = Split(
        train=train,
        val=val,
        test=test,
        task=task,
        name=key,
        metric_main=metric_main,
    )
    logger.info(
        f"Loaded {key}: "
        f"train={len(train)}, val={len(val)}, test={len(test)}, task={task}"
    )
    return split
