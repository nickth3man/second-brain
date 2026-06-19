"""Text passthrough parser (§6).

Reads the file as UTF-8 and returns the raw content as-is.  Used for ``text``,
``code``, and ``structured`` pipeline stages (no transformation needed — the
extract step adds the AI summary).
"""

from __future__ import annotations

from pathlib import Path


def parse_text(path: Path) -> str:
    """Read *path* as UTF-8 and return the content verbatim.

    Args:
        path: Path to a text/code/structured file.

    Returns:
        The raw file content as a string.
    """
    return path.read_text(encoding="utf-8", errors="replace")
