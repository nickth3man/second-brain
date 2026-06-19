"""PDF page renderer + vision-OCR parser (§6, §12.3, §12.7).

Uses PyMuPDF (``fitz``) to render each page to a PNG image, then sends the
image to an OpenRouter vision model for OCR.  Per-page progress is checkpointed
to ``.brain/cache/<sha>.progress.json`` for resumability (§12.3).  The
PyMuPDF dependency is isolated in ``_render_page`` (§12.7) — swapping to
pypdfium2 requires changing only that function (and the ``fitz.open`` call
in ``parse_pdf``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz

from second_brain.config import Config
from second_brain.openrouter_client import OpenRouterClient


def _progress_path(cfg: Config, sha: str) -> Path:
    """Return the path to the per-page progress checkpoint file.

    Args:
        cfg: App configuration (used for ``brain_root``).
        sha: First 16 hex chars of the file SHA-256.

    Returns:
        Path to ``.brain/cache/{sha}.progress.json``.
    """
    return cfg.brain_root / ".brain" / "cache" / f"{sha}.progress.json"


def _render_page(page: fitz.Page, cfg: Config) -> bytes:
    """Render a PyMuPDF page to PNG (or JPEG) image bytes.

    This function ISOLATES the PyMuPDF dependency per §12.7.  To swap
    renderers (e.g. to pypdfium2), only this function needs to change.

    Args:
        page: A PyMuPDF ``Page`` object.
        cfg: App configuration (``pdf_dpi``, ``pdf_image_format``,
            ``pdf_alpha``).

    Returns:
        Encoded image bytes (PNG by default, JPEG if configured).
    """
    mat = fitz.Matrix(cfg.ingestion.pdf_dpi / 72, cfg.ingestion.pdf_dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=cfg.ingestion.pdf_alpha)
    fmt = cfg.ingestion.pdf_image_format
    if fmt == "jpeg":
        return pix.tobytes("jpeg")
    return pix.tobytes("png")


async def parse_pdf(path: Path, cfg: Config, client: OpenRouterClient) -> str:
    """Render a PDF page-by-page and OCR each page via a vision model.

    Each page is rendered to an image using ``_render_page``, then sent to
    ``client.vision_describe`` for OCR.  Completed page indices are
    checkpointed after every page so partial progress is not lost on
    interruption (§12.3).  If a single page OCR fails, an error note is
    inserted and the parser continues with the remaining pages.

    Args:
        path: Path to the PDF file.
        cfg: App configuration.
        client: OpenRouter client for vision API calls.

    Returns:
        Concatenated OCR text for all newly-processed pages, separated by
        double newlines.  (Pages already marked done in the progress file
        are skipped; the caller is responsible for combining partial runs.)
    """
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    progress_path = _progress_path(cfg, sha)

    done: set[int] = set()
    if progress_path.is_file():
        done = set(json.loads(progress_path.read_text()))

    doc = fitz.open(str(path))
    n = doc.page_count
    pages_text: list[str] = []

    for i in range(n):
        if i in done:
            continue
        try:
            img_bytes = _render_page(doc[i], cfg)
            text = await client.vision_describe(
                cfg.models.vision,
                [img_bytes],
                (
                    f"OCR this PDF page (page {i + 1} of {n}). "
                    "Return the text content faithfully, preserving headings "
                    "and structure. If the page has no text, say '[blank page]'."
                ),
            )
            pages_text.append(text)
        except Exception as e:
            pages_text.append(f"\n[page {i + 1} OCR failed: {e}]\n")
        # Checkpoint progress after every page (§12.3)
        done.add(i)
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps(sorted(done)))

    doc.close()
    return "\n\n".join(pages_text)
