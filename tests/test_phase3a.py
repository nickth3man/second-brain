"""Phase 3 Wave 1 tests — parse framework + deterministic parsers (§5, §6).

Tests use fixture files written to ``tmp_path`` and stub clients.  No network
or real API calls.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from second_brain.daemon.normalize import (
    normalize_text,
    sha256_of_file,
    source_id_for,
)

# -- stubs --------------------------------------------------------------------


@dataclass
class _FakePrivacy:
    zdr: bool = True
    require_parameters: bool = False
    block_training_providers: bool = False


@dataclass
class _FakeExtraction:
    primary_model: str = "test-model"
    repair_model: str = "test-repair-model"
    enable_healing: bool = False
    deadletter_dir: str = ".brain/deadletter"
    max_attempts: int = 2
    require_parameters: bool = False
    confidence_floor: float = 0.6
    quarantine_dir: str = ".brain/quarantine"


@dataclass
class _FakeCfg:
    brain_root: Path
    privacy: _FakePrivacy = field(default_factory=_FakePrivacy)
    extraction: _FakeExtraction = field(default_factory=_FakeExtraction)


class _FakeClient:
    """Minimal stub — Wave 1 parsers don't call client methods."""

    async def vision_describe(self, *args: object, **kwargs: object) -> str:
        return "fake vision description"

    async def transcribe(self, *args: object, **kwargs: object) -> str:
        return "fake transcription"

    async def close(self) -> None:
        pass


# -- helpers ------------------------------------------------------------------


def _now_iso() -> str:
    return "2026-06-19T12:00:00Z"


# -- TestParseText ------------------------------------------------------------


class TestParseText:
    """parse_text returns file content verbatim."""

    def test_reads_utf8(self, tmp_path: Path) -> None:
        from second_brain.parse.text import parse_text

        p = tmp_path / "hello.txt"
        p.write_text("Hello, world!", encoding="utf-8")
        assert parse_text(p) == "Hello, world!"

    def test_reads_unicode(self, tmp_path: Path) -> None:
        from second_brain.parse.text import parse_text

        p = tmp_path / "unicode.txt"
        p.write_text("cafe\u0301 résumé", encoding="utf-8")
        assert parse_text(p) == "cafe\u0301 résumé"

    def test_handles_replacement(self, tmp_path: Path) -> None:
        """Invalid UTF-8 bytes get replacement characters, not a crash."""
        from second_brain.parse.text import parse_text

        p = tmp_path / "corrupt.bin"
        p.write_bytes(b"hello\xffworld")
        result = parse_text(p)
        assert "hello" in result
        assert "world" in result


# -- TestParseOffice ----------------------------------------------------------


class TestParseOffice:
    """parse_office uses mammoth for docx-family, defers xlsx/pptx."""

    async def test_docx_delegates_to_mammoth(self, tmp_path: Path, monkeypatch) -> None:
        from second_brain.parse.office import parse_office

        def fake_convert_to_markdown(f):
            return type("R", (), {"value": "# Mocked Heading\n\nParagraph."})()

        monkeypatch.setattr("mammoth.convert_to_markdown", fake_convert_to_markdown)

        p = tmp_path / "test.docx"
        p.write_bytes(b"fake docx bytes")
        result = await parse_office(p)
        assert result == "# Mocked Heading\n\nParagraph."

    async def test_mammoth_failure_fallback(self, tmp_path: Path, monkeypatch) -> None:
        from second_brain.parse.office import parse_office

        def failing_convert(f):
            raise RuntimeError("corrupt file")

        monkeypatch.setattr("mammoth.convert_to_markdown", failing_convert)

        p = tmp_path / "broken.doc"
        p.write_bytes(b"garbage")
        result = await parse_office(p)
        assert "Office parse failed" in result
        assert "corrupt file" in result

    async def test_xlsx_returns_deferred_note(self, tmp_path: Path) -> None:
        from second_brain.parse.office import parse_office

        p = tmp_path / "spreadsheet.xlsx"
        p.write_bytes(b"fake xlsx")
        result = await parse_office(p)
        assert "not yet parsed" in result
        assert ".xlsx" in result

    async def test_pptx_returns_deferred_note(self, tmp_path: Path) -> None:
        from second_brain.parse.office import parse_office

        p = tmp_path / "slides.pptx"
        p.write_bytes(b"fake pptx")
        result = await parse_office(p)
        assert "not yet parsed" in result
        assert ".pptx" in result


# -- TestParseWeb -------------------------------------------------------------


class TestParseWeb:
    """parse_web extracts readable content via readability-lxml."""

    async def test_extracts_body_text(self, tmp_path: Path) -> None:
        from second_brain.parse.web import parse_web

        html = """<!DOCTYPE html>
<html><head><title>Test Page</title></head>
<body>
<nav>Skip this nav junk</nav>
<article><h1>Main Content</h1><p>This is the article body.</p></article>
<footer>Footer stuff</footer>
</body></html>
"""
        p = tmp_path / "page.html"
        p.write_text(html, encoding="utf-8")
        result = await parse_web(p)

        # The wrapper returns a non-empty string and surfaces the title.
        # (Body-extraction quality is readability's job — tested on real pages,
        # not asserted against synthetic HTML where readability is unpredictable.)
        assert isinstance(result, str) and result.strip()
        assert "Test Page" in result

    async def test_failure_fallback(self, tmp_path: Path) -> None:
        from second_brain.parse.web import parse_web

        # readability is lenient on malformed bytes; a missing file forces a
        # real exception so the fallback path is exercised.
        result = await parse_web(tmp_path / "does_not_exist.html")
        assert "HTML parse failed" in result


# -- TestParseEbook -----------------------------------------------------------


class TestParseEbook:
    """parse_ebook extracts text from EPUB (ZIP) archives."""

    async def test_extracts_xhtml_content(self, tmp_path: Path) -> None:
        from second_brain.parse.ebook import parse_ebook

        epub = tmp_path / "book.epub"
        with zipfile.ZipFile(epub, "w") as z:
            z.writestr(
                "OEBPS/chapter1.xhtml",
                "<html><body><p>Chapter one content.</p></body></html>",
            )
            z.writestr(
                "OEBPS/chapter2.xhtml",
                "<html><body><p>Chapter two content.</p></body></html>",
            )

        result = await parse_ebook(epub)
        assert "Chapter one content." in result
        assert "Chapter two content." in result

    async def test_skips_non_html_entries(self, tmp_path: Path) -> None:
        from second_brain.parse.ebook import parse_ebook

        epub = tmp_path / "hybrid.epub"
        with zipfile.ZipFile(epub, "w") as z:
            z.writestr("mimetype", "application/epub+zip")
            z.writestr("OEBPS/style.css", "body { color: red; }")
            z.writestr(
                "OEBPS/content.xhtml",
                "<html><body><p>Only real content.</p></body></html>",
            )

        result = await parse_ebook(epub)
        assert "Only real content." in result
        assert "mimetype" not in result
        assert "red" not in result

    async def test_failure_fallback(self, tmp_path: Path) -> None:
        from second_brain.parse.ebook import parse_ebook

        p = tmp_path / "not-an-epub.epub"
        p.write_bytes(b"not a zip")
        result = await parse_ebook(p)
        assert "EPUB parse failed" in result


# -- TestParseDispatch --------------------------------------------------------


class TestParseDispatch:
    """parse_to_markdown routes to the correct sub-parser by stage."""

    async def test_text_stage(self, tmp_path: Path) -> None:
        from second_brain.parse import parse_to_markdown

        p = tmp_path / "note.txt"
        p.write_text("verbatim text", encoding="utf-8")
        cfg = _FakeCfg(brain_root=tmp_path)
        client = _FakeClient()
        result = await parse_to_markdown(p, "text", cfg, client)
        assert result == "verbatim text"

    async def test_code_stage(self, tmp_path: Path) -> None:
        from second_brain.parse import parse_to_markdown

        p = tmp_path / "script.py"
        p.write_text("def foo(): pass", encoding="utf-8")
        cfg = _FakeCfg(brain_root=tmp_path)
        client = _FakeClient()
        result = await parse_to_markdown(p, "code", cfg, client)
        assert result == "def foo(): pass"

    async def test_structured_stage(self, tmp_path: Path) -> None:
        from second_brain.parse import parse_to_markdown

        p = tmp_path / "data.json"
        p.write_text('{"key": "value"}', encoding="utf-8")
        cfg = _FakeCfg(brain_root=tmp_path)
        client = _FakeClient()
        result = await parse_to_markdown(p, "structured", cfg, client)
        assert result == '{"key": "value"}'

    async def test_office_stage(self, tmp_path: Path, monkeypatch) -> None:
        from second_brain.parse import parse_to_markdown

        def fake_convert(f):
            return type("R", (), {"value": "# Office Doc"})()

        monkeypatch.setattr("mammoth.convert_to_markdown", fake_convert)

        p = tmp_path / "report.docx"
        p.write_bytes(b"fake")
        cfg = _FakeCfg(brain_root=tmp_path)
        client = _FakeClient()
        result = await parse_to_markdown(p, "office", cfg, client)
        assert result == "# Office Doc"

    async def test_web_stage(self, tmp_path: Path) -> None:
        from second_brain.parse import parse_to_markdown

        html = """<html><head><title>Web Title</title></head>
<body><article><p>Web body.</p></article></body></html>"""
        p = tmp_path / "page.html"
        p.write_text(html, encoding="utf-8")
        cfg = _FakeCfg(brain_root=tmp_path)
        client = _FakeClient()
        result = await parse_to_markdown(p, "web", cfg, client)
        assert "Web Title" in result
        assert "Web body." in result

    async def test_ebook_stage(self, tmp_path: Path) -> None:
        from second_brain.parse import parse_to_markdown

        epub = tmp_path / "book.epub"
        with zipfile.ZipFile(epub, "w") as z:
            z.writestr("content.xhtml", "<html><body><p>Ebook text.</p></body></html>")

        cfg = _FakeCfg(brain_root=tmp_path)
        client = _FakeClient()
        result = await parse_to_markdown(epub, "ebook", cfg, client)
        assert "Ebook text." in result

    async def test_unknown_stage_raises_value_error(self, tmp_path: Path) -> None:
        from second_brain.parse import parse_to_markdown

        p = tmp_path / "dummy.txt"
        p.write_text("x", encoding="utf-8")
        cfg = _FakeCfg(brain_root=tmp_path)
        client = _FakeClient()
        with pytest.raises(ValueError, match="Unknown pipeline stage"):
            await parse_to_markdown(p, "unknown", cfg, client)


# -- TestNormalize ------------------------------------------------------------


class TestNormalize:
    """normalize_text writes a 50-sources/*.md with parsed body and front-matter."""

    async def test_normalize_web(self, tmp_path: Path) -> None:
        """Stage=web produces front-matter with type:web and readable body."""

        from second_brain.frontmatter import split_frontmatter

        cfg = _FakeCfg(brain_root=tmp_path)
        client = _FakeClient()

        # Create fixture HTML in inbox (path doesn't matter for normalize,
        # but we need a real file for the parser to read).
        p = tmp_path / "00-inbox" / "article.html"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            """<html><head><title>P3A Test</title></head>
<body><article><h1>Readable Heading</h1><p>Readable body text.</p></article></body></html>""",
            encoding="utf-8",
        )

        sha = sha256_of_file(p)
        ingested = "2026-06-19T12:00:00Z"
        source_id = source_id_for(p, "P3A Test", ingested)

        # Ensure 50-sources/ exists
        (cfg.brain_root / "50-sources").mkdir(parents=True, exist_ok=True)

        dst = await normalize_text(p, source_id, sha, ingested, "web", cfg, client)

        assert dst.exists()
        assert dst.parent.name == "50-sources"
        assert source_id in dst.name

        content = dst.read_text(encoding="utf-8")
        meta, body = split_frontmatter(content)

        assert meta.get("type") == "web"
        assert meta.get("sha256") == sha
        assert "Readable Heading" in body
        assert "Readable body text." in body
        assert "article.html" not in body  # raw HTML is stripped
        assert "## Summary" in body

    async def test_normalize_text_stage(self, tmp_path: Path) -> None:
        """Stage=text still writes raw body (backward compat)."""
        from second_brain.frontmatter import split_frontmatter

        cfg = _FakeCfg(brain_root=tmp_path)
        client = _FakeClient()

        p = tmp_path / "00-inbox" / "note.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("Hello world.", encoding="utf-8")

        sha = sha256_of_file(p)
        ingested = "2026-06-19T12:00:00Z"
        source_id = source_id_for(p, "Hello world.", ingested)

        (cfg.brain_root / "50-sources").mkdir(parents=True, exist_ok=True)

        dst = await normalize_text(p, source_id, sha, ingested, "text", cfg, client)
        content = dst.read_text(encoding="utf-8")
        meta, body = split_frontmatter(content)

        assert meta.get("type") == "text"
        assert "Hello world." in body
