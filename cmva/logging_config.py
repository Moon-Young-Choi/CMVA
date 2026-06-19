"""Logging setup for CMVA."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(path: str | Path = "logs/cmva.log") -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
