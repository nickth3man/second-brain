"""Office document parser (§6).

Uses ``mammoth`` to convert ``.docx`` / ``.doc`` / ``.odt`` / ``.rtf`` files to
markdown.  ``.xlsx`` / ``.xls`` / ``.pptx`` / ``.ppt`` return a deferred note;
their structured data support lands when there is demand.
"""

from __future__ import annotations

from pathlib import Path


async def parse_office(path: Path) -> str:
    """Convert an office document to markdown.

    ``.docx`` / ``.doc`` / ``.odt`` / ``.rtf`` are converted via ``mammoth``.
    Spreadsheets and presentations return a placeholder note explaining the gap.

    Args:
        path: Path to the office file.

    Returns:
        Markdown text of the document content, or a deferred-support note.
    """
    suffix = path.suffix.lower()

    if suffix in {".docx", ".doc", ".odt", ".rtf"}:
        try:
            import mammoth

            with open(path, "rb") as f:
                result = mammoth.convert_to_markdown(f)
            return result.value
        except Exception as e:
            return f"Office parse failed: {e}"

    if suffix in {".xlsx", ".xls"}:
        return f"[{suffix} not yet parsed — Phase 3 office xlsx/pptx deferred]"

    if suffix in {".pptx", ".ppt"}:
        return f"[{suffix} not yet parsed — Phase 3 office xlsx/pptx deferred]"

    # Fallback for any unknown sub-extension within the office stage
    return f"[{suffix} not yet parsed — Phase 3 office xlsx/pptx deferred]"
