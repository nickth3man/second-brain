"""Logging configuration — structlog + RotatingFileHandler (§12.7).

The codebase uses structlog throughout but never configures it.
This module provides a single ``configure_logging`` call that
sets up structured JSON logging to a file (with rotation) and
routes stdlib logging through structlog so httpx warnings etc.
are captured.

When ``console=True`` (default for CLI commands), a human-readable
``ConsoleRenderer`` is also added to stderr so every log event streams
to the terminal in real time alongside the file log.

The key-redaction processor ensures sensitive fields (key, token,
authorization, password) are never written to disk (and never printed
to the console).
"""

from __future__ import annotations

import logging
import re
import sys
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
    console: bool = True,
) -> Path:
    """Configure structlog + stdlib logging for the Second Brain daemon.

    Creates a log directory at ``<brain_root>/.brain/logs/`` and writes
    to ``second-brain.log`` with 5 MB x 5 rotation.  Idempotent — a second
    call is a no-op (the file handler is only added once).

    Args:
        brain_root: The brain root directory (``cfg.brain_root``).
        level: Log level string (default ``"INFO"``).
        console: When ``True`` (default), also attach a ``ConsoleRenderer``
            handler to ``stderr`` so log events stream to the terminal in
            real time.  Set ``False`` in the daemon server where stderr is
            not watched.

    Returns:
        The resolved path to the log file.
    """
    global _LOGGING_CONFIGURED  # noqa: PLW0603

    log_dir = brain_root / ".brain" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "second-brain.log"

    if _LOGGING_CONFIGURED:
        return log_path

    # Shared pre-chain processors (run before the per-handler renderer).
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.stdlib.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _redact_keys,
    ]

    # -- File handler (JSON, rotation) ------------------------------------
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    handlers: list[logging.Handler] = [file_handler]

    # -- Optional console handler (human-readable, stderr) ----------------
    if console:
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=False),
            ],
            foreign_pre_chain=shared_processors,
        )
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

    # -- structlog configuration ------------------------------------------
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # -- Route stdlib logging through structlog ---------------------------
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for h in handlers:
        root_logger.addHandler(h)

    _LOGGING_CONFIGURED = True
    return log_path
