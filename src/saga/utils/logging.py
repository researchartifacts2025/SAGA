"""Logging helpers.

Rich-formatted logging with sane defaults; quiet mode for CI; verbose mode for
investigations. Uses the standard ``logging`` module so consumer code stays
framework-neutral.
"""

from __future__ import annotations

import logging
import os

from rich.logging import RichHandler


_LOG_FORMAT = "%(message)s"
_DATE_FORMAT = "[%X]"

_INITIALIZED = False


def setup_logging(level: str | int = "INFO", rich: bool | None = None) -> None:
    """Configure root logging once per process.

    ``level`` may be a string ("DEBUG", "INFO", "WARNING", "ERROR") or an int.
    Passing ``rich=False`` disables the colored handler (useful in CI logs that
    re-render ANSI as raw escape codes).
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    if rich is None:
        rich = os.environ.get("SAGA_PLAIN_LOG", "") == ""

    handlers: list[logging.Handler]
    if rich:
        handlers = [RichHandler(rich_tracebacks=True, show_time=True, show_path=False)]
    else:
        handlers = [logging.StreamHandler()]

    logging.basicConfig(
        level=_normalize_level(level),
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
    )
    _INITIALIZED = True


def _normalize_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    return getattr(logging, level.upper(), logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger. Lazily initializes logging at INFO."""
    if not _INITIALIZED:
        setup_logging()
    return logging.getLogger(name)
