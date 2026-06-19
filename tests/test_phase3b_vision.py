"""Phase 3 Wave 2a tests — PDF (PyMuPDF + vision OCR) and image parsers (§6).

Tests use real PyMuPDF to create tiny in-memory PDFs and a ``FakeVisionClient``
that returns canned strings.  No network or real API calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# -- stubs --------------------------------------------------------------------


@dataclass
class _FakeIngestion:
    pdf_dpi: int = 200
    pdf_image_format: str = "png"
    pdf_alpha: bool = False
    vision_max_edge_px: int = 2048


@dataclass
class _FakeModels:
    vision: str = "test-vision-model"


@dataclass
class _FakeCfg:
    brain_root: Path
    ingestion: _FakeIngestion = field(default_factory=_FakeIngestion)
    models: _FakeModels = field(default_factory=_FakeModels)


class FakeVisionClient:
    """Records calls and returns canned strings derived from the prompt.

    For PDF prompts it extracts the page number (e.g. "page 2 of 3") and
    returns a string including it.  For other prompts it returns a default
    description.
    """

    def __init__(self) -> None:
        self.call_count: int = 0
        self.calls: list[tuple[str, list[bytes], str, str]] = []

    async def vision_describe(
        self,
        model: str,
        images: list[bytes],
        prompt: str,
        *,
        mime: str = "image/png",
    ) -> str:
        self.call_count += 1
        self.calls.append((model, images, prompt, mime))
        m = re.search(r"page (\d+) of (\d+)", prompt)
        if m:
            page_num = int(m.group(1))
            return f"[OCR canned text for page {page_num}]"
        return "[Canned image description]"


# -- helpers ------------------------------------------------------------------


def _make_pdf(path: Path, texts: list[str]) -> None:
    """Create a tiny PDF with one page per *text*, each containing that text."""
    import fitz

    doc = fitz.open()
    for t in texts:
        p = doc.new_page()
        p.insert_text((50, 72), t)
    doc.save(str(path))
    doc.close()


def _progress_path(cfg: _FakeCfg, sha: str) -> Path:
    """Replicate the production ``_progress_path`` for test assertions."""
    return cfg.brain_root / ".brain" / "cache" / f"{sha}.progress.json"


# -- TestParsePdf -------------------------------------------------------------


class TestParsePdf:
    """parse_pdf renders pages and OCRs them via the vision client."""

    async def test_round_trip(self, tmp_path: Path) -> None:
        """Fresh 2-page PDF — both pages OCR'd, progress file written."""
        from second_brain.parse.pdf import parse_pdf

        pdf = tmp_path / "test.pdf"
        _make_pdf(pdf, ["Hello PDF page one", "Hello PDF page two"])

        cfg = _FakeCfg(brain_root=tmp_path)
        client = FakeVisionClient()
        result = await parse_pdf(pdf, cfg, client)

        assert "[OCR canned text for page 1]" in result
        assert "[OCR canned text for page 2]" in result
        assert client.call_count == 2

        # Progress file written with both indices
        sha = hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]
        progress = _progress_path(cfg, sha)
        assert progress.exists()
        assert json.loads(progress.read_text()) == [0, 1]

    async def test_resume_skips_done_pages(self, tmp_path: Path) -> None:
        """Pre-existing progress for page 0 — page 0 not re-OCR'd."""
        from second_brain.parse.pdf import parse_pdf

        pdf = tmp_path / "test.pdf"
        _make_pdf(pdf, ["Page A", "Page B"])

        cfg = _FakeCfg(brain_root=tmp_path)

        # Write progress marking page 0 as done
        sha = hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]
        progress = _progress_path(cfg, sha)
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text("[0]")

        client = FakeVisionClient()
        await parse_pdf(pdf, cfg, client)

        assert client.call_count == 1  # only page 1 was OCR'd
        # Progress now includes both pages
        assert json.loads(progress.read_text()) == [0, 1]

    async def test_page_failure_tolerance(self, tmp_path: Path) -> None:
        """A failing page does not abort the whole document."""
        from second_brain.parse.pdf import parse_pdf

        pdf = tmp_path / "test.pdf"
        _make_pdf(pdf, ["Page zero", "Page one"])

        cfg = _FakeCfg(brain_root=tmp_path)
        client = FakeVisionClient()

        # Inject a failure on the second vision_describe call
        original_describe = client.vision_describe

        async def failing_describe(
            model: str,
            images: list[bytes],
            prompt: str,
            *,
            mime: str = "image/png",
        ) -> str:
            if client.call_count == 1:  # second call (page index 1)
                msg = "mock vision error"
                raise RuntimeError(msg)
            return await original_describe(model, images, prompt, mime=mime)

        client.vision_describe = failing_describe  # type: ignore[assignment]

        result = await parse_pdf(pdf, cfg, client)

        assert "[OCR canned text for page 1]" in result
        assert "[page 2 OCR failed: mock vision error]" in result


# -- TestParseImage -----------------------------------------------------------


class TestParseImage:
    """parse_image sends image bytes to the vision client."""

    async def test_png_image(self, tmp_path: Path) -> None:
        from second_brain.parse.image import parse_image

        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        cfg = _FakeCfg(brain_root=tmp_path)
        client = FakeVisionClient()
        result = await parse_image(img, cfg, client)

        assert result == "[Canned image description]"
        assert client.call_count == 1

    async def test_jpeg_image(self, tmp_path: Path) -> None:
        from second_brain.parse.image import parse_image

        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        cfg = _FakeCfg(brain_root=tmp_path)
        client = FakeVisionClient()
        result = await parse_image(img, cfg, client)

        assert result == "[Canned image description]"
        # Verify MIME was correctly inferred
        assert client.calls[0][3] == "image/jpeg"

    async def test_failure_fallback(self, tmp_path: Path) -> None:
        from second_brain.parse.image import parse_image

        img = tmp_path / "photo.png"
        img.write_bytes(b"not an image")

        cfg = _FakeCfg(brain_root=tmp_path)
        client = FakeVisionClient()

        # Make vision_describe raise
        async def failing(*args: object, **kwargs: object) -> str:
            msg = "bad image data"
            raise RuntimeError(msg)

        client.vision_describe = failing  # type: ignore[assignment]

        result = await parse_image(img, cfg, client)
        assert "[image parse failed: bad image data]" in result
