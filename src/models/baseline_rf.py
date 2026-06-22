"""Random Forest + Morgan FP baseline.

Per CLAUDE.md §15: simplicity first. RF on Morgan FP is the floor every
foundation model must beat. CLAUDE.md §11: BBBP is 76% positive so
class_weight='balanced' is mandatory for classification.
"""

from __future__ import annotations

from typing import Union

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

RFModel = Union[RandomForestRegressor, RandomForestClassifier]


def build_rf(
    task: str,
    n_estimators: int = 500,
    max_depth: int | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RFModel:
    """Random Forest factory.

    Args:
        task: 'regression' or 'classification'.
        n_estimators: Number of trees. 500 is a strong default; 1000 for
            small datasets like GSHt.
        max_depth: Tree depth cap. None = unlimited (default).
        random_state: Seed.
        n_jobs: Parallel cores. -1 uses all.

    Returns:
        sklearn estimator.

    Raises:
        ValueError: if task is neither 'regression' nor 'classification'.
    """
    if task == "regression":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if task == "classification":
        # Trigger: classification head needed.
        # Why:     CLAUDE.md §11 — BBBP 76% positive, must rebalance.
        # Outcome: minority class gets higher weight in node splits.
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced",
        )
    raise ValueError(f"task must be 'regression' or 'classification', got {task!r}")
