from __future__ import annotations

import logging

try:
    from rich.logging import RichHandler

    _HANDLER = RichHandler(rich_tracebacks=True, show_path=False)
except ImportError:  # pragma: no cover - rich is a soft dependency
    _HANDLER = logging.StreamHandler()


def get_logger(name: str = "automl_agent", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(_HANDLER)
        logger.setLevel(level)
        logger.propagate = False
    return logger
