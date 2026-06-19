"""Web-page parser (§6).

Uses ``readability-lxml`` to extract the readable content from an HTML file,
then strips remaining markup to produce plain text.
"""

from __future__ import annotations

import re
from pathlib import Path


async def parse_web(path: Path) -> str:
    """Extract readable text from an HTML file.

    Reads the file, runs it through ``readability.Document``, and removes
    remaining HTML tags via a simple regex.

    Args:
        path: Path to an HTML file.

    Returns:
        The extracted readable text (title + body).
    """
    try:
        from readability import Document

        html = path.read_text(encoding="utf-8", errors="replace")
        doc = Document(html)
        summary = doc.summary()
        # Strip remaining HTML tags
        text = re.sub(r"<[^>]+>", "", summary)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        title = doc.title()
        if title:
            return f"# {title}\n\n{text}"
        return text
    except Exception as e:
        return f"[HTML parse failed: {e}]"
