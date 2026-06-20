"""Logging configuration — structlog + RotatingFileHandler (§12.7).

The codebase uses structlog throughout but never configures it.
This module provides a single ``configure_logging`` call that
sets up structured JSON logging to a file (with rotation) and
routes stdlib logging through structlog so httpx warnings etc.
are captured.

The key-redaction processor ensures sensitive fields (key, token,
authorization, password) are never written to disk.
"""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog

# Guard against duplicate handler registrations.
_LOGGING_CONFIGURED: bool = False

_KEY_PATTERN = re.compile(r"(?i)(key|token|authorization|password)")


def _redact_keys(
    logger: logging.Logger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: dict,
) -> dict:
    """Redact any event field whose name matches key/token/auth/password."""
    return {
        k: "***" if _KEY_PATTERN.search(k) else v
        for k, v in event_dict.items()
    }


def configure_logging(
    brain_root: Path,
    *,
    level: str = "INFO",
) -> Path:
    """Configure structlog + stdlib logging for the Second Brain daemon.

    Creates a log directory at ``<brain_root>/.brain/logs/`` and writes
    to ``second-brain.log`` with 5 MB x 5 rotation.  Idempotent — a second
    call is a no-op (the file handler is only added once).

    Args:
        brain_root: The brain root directory (``cfg.brain_root``).
        level: Log level string (default ``"INFO"``).

    Returns:
        The resolved path to the log file.
    """
    global _LOGGING_CONFIGURED  # noqa: PLW0603

    log_dir = brain_root / ".brain" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "second-brain.log"

    if _LOGGING_CONFIGURED:
        return log_path

    # -- File handler with rotation ---------------------------------------
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    # -- structlog configuration ------------------------------------------
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.stdlib.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _redact_keys,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # -- Route stdlib logging through structlog ---------------------------
    logging.basicConfig(
        level=level,
        format="%(message)s",
        force=True,
        handlers=[file_handler],
    )

    _LOGGING_CONFIGURED = True
    return log_path
