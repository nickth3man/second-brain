"""EPUB parser (§6).

EPUB files are ZIP archives containing XHTML/HTML content documents.
This parser iterates the ZIP entries, strips markup from each content page,
and concatenates the text.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path


async def parse_ebook(path: Path) -> str:
    """Extract text from an EPUB file.

    Args:
        path: Path to an ``.epub`` file.

    Returns:
        Concatenated text from all XHTML/HTML content documents.
    """
    try:
        texts: list[str] = []
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith((".xhtml", ".html", ".htm")):
                    html = z.read(name).decode("utf-8", errors="replace")
                    text = re.sub(r"<[^>]+>", "", html)
                    text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        texts.append(text)
        return "\n\n".join(texts)
    except Exception as e:
        return f"[EPUB parse failed: {e}]"
