"""Extension routing and temp-file detection (§6, §12.3)."""

from __future__ import annotations

import re

from second_brain.config import Config, ext_to_stage

TEMP_FILE_PATTERNS: list[str] = [
    r"^~\$",
    r"\.goutputstream",
    r"\.crdownload$",
    r"\.part$",
    r"\.swp$",
    r"\.tmp$",
    r"^~.*~$",
    r"\.bak$",
]

_TEMP_RE = re.compile("|".join(TEMP_FILE_PATTERNS))


def is_temp_file(name: str) -> bool:
    """Return ``True`` if *name* matches any known temp-file pattern.

    Used by the watcher's stable-file gate (§12.3) to ignore in-progress
    writes by editors and browsers.
    """
    return bool(_TEMP_RE.search(name))


def route(ext: str, cfg: Config) -> str:
    """Return the pipeline stage name for a file extension.

    Delegates to :func:`ext_to_stage`; exists as a thin wrapper so the
    pipeline only imports from ``daemon.router`` rather than ``config``.
    """
    return ext_to_stage(ext, cfg.types)
