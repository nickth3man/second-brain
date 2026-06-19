"""Parse dispatch — select a parser by pipeline stage (§5, §6).

Phase 3 Wave 1 implements text/code/structured (raw passthrough), office
(mammoth), web (readability-lxml), and ebook (zipfile).  Wave 2 adds
pdf (PyMuPDF), vision (OpenRouter), audio (ffmpeg + OpenRouter STT), and
video (ffmpeg keyframe + STT).
"""

from __future__ import annotations

from pathlib import Path

from second_brain.config import Config
from second_brain.openrouter_client import OpenRouterClient


async def parse_to_markdown(
    path: Path,
    stage: str,
    cfg: Config,
    client: OpenRouterClient,
) -> str:
    """Parse *path* according to its pipeline *stage* and return markdown.

    Args:
        path: Path to the raw inbox file.
        stage: Pipeline stage name from :func:`route`.
        cfg: App config (used by Wave-2 parsers for DPI, model names, etc.).
        client: OpenRouter client (used by Wave-2 parsers for vision/STT).

    Returns:
        The parsed markdown body.

    Raises:
        NotImplementedError: if *stage* is a Wave-2 stage (pdf, vision, audio,
            video).
        ValueError: if *stage* is unknown.
    """
    if stage in {"text", "code", "structured"}:
        from second_brain.parse.text import parse_text

        return parse_text(path)

    if stage == "office":
        from second_brain.parse.office import parse_office

        return await parse_office(path)

    if stage == "web":
        from second_brain.parse.web import parse_web

        return await parse_web(path)

    if stage == "ebook":
        from second_brain.parse.ebook import parse_ebook

        return await parse_ebook(path)

    if stage in {"pdf", "vision", "audio", "video"}:
        raise NotImplementedError(f"{stage} parsing arrives in Phase 3 Wave 2")

    raise ValueError(f"Unknown pipeline stage: {stage}")
