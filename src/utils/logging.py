from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Optional, TypeVar

from src.core.progress import TqdmProgressReporter

T = TypeVar("T")

# Default log format used in get_logger
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> logging.Logger:
    """Get a configured logger.

    This helper returns a named logger with a StreamHandler attached.
    Handlers are only added once per logger name to avoid duplicated logs.

    Args:
        name: Logger name.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        log_format: Format string for the log messages.

    Returns:
        A configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Only configure the logger if it has no handlers yet
    if not logger.handlers:
        logger.setLevel(level)

        handler = logging.StreamHandler()
        handler.setLevel(level)

        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


def get_progress_bar(
    iterable: Iterable[T],
    desc: Optional[str] = None,
    **tqdm_kwargs,
) -> Iterable[T]:
    """Wrap an iterable with tqdm progress bar.

    Args:
        iterable: Any iterable to iterate over.
        desc: Optional description shown on the left of the progress bar.
        **tqdm_kwargs: Extra keyword arguments forwarded to `tqdm`.

    Returns:
        An iterable wrapped by `tqdm`.
    """
    return TqdmProgressReporter().wrap(iterable, desc=desc, **tqdm_kwargs)
