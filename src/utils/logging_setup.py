"""Structured logging.

Call `configure_logging(level)` once at the start of every entry point.
Logs go to stderr with timestamp + module + level. JSONL log file under
runs/<run_id>/log.jsonl if `log_dir` provided.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure root logger.

    Args:
        level: Logging level name (DEBUG/INFO/WARNING/ERROR).
        log_dir: If provided, also write logs to log_dir/log.txt.
    """
    level_int = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "log.txt"))

    logging.basicConfig(
        level=level_int,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Quiet noisy third-party loggers.
    for noisy in ["urllib3", "matplotlib", "PIL", "transformers.tokenization_utils"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
