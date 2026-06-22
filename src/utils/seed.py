"""Centralized seed management.

Per CLAUDE.md §8.1: must be called as the FIRST line of every entry point
(train.py, evaluate.py, scripts/*). Covers Python random, NumPy, PyTorch
(CPU + CUDA), cudnn determinism, and required env vars.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42, deterministic: bool = True) -> None:
    """Set all RNGs and deterministic flags.

    Args:
        seed: Random seed. Default 42 (also use 1337 / 2024 for the 3-seed runs
            required by CLAUDE.md §8.2).
        deterministic: If True (default), set cudnn.deterministic and disable
            cudnn.benchmark. Slower but reproducible. CLAUDE.md §8.1 mandates
            True for paper-grade experiments; only flip False during initial
            sweep exploration.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic CUDA matmul on PyTorch >= 1.11.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(torch, "use_deterministic_algorithms"):
                # warn_only because some ops (e.g., scatter) have no det impl.
                torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        # torch optional for pure-sklearn baselines.
        logger.debug("torch not installed; skipping torch seed setup")

    logger.info(f"Seeds set: seed={seed}, deterministic={deterministic}")
