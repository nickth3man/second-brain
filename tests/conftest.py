"""Shared pytest fixtures.

Auto-resets structlog global configuration after every test so that
``second_brain.log.configure_logging`` (which mutates structlog + stdlib
logging globally and is called by the observability/pipeline tests) cannot
leak its processor chain, wrapper class, or file handler into unrelated
tests. Without this, a single ``configure_logging`` call poisons the whole
suite (e.g. broken ``StackInfoRenderer`` chain raised TypeErrors in phase
4/5/6 tests that only log incidentally).
"""

from __future__ import annotations

import pytest
import structlog

import second_brain.log as sb_log


@pytest.fixture(autouse=True)
def _reset_structlog_each_test() -> None:
    """Restore structlog defaults + clear the one-shot config guard per test."""
    yield
    structlog.reset_defaults()
    sb_log._LOGGING_CONFIGURED = False
