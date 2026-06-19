"""Phase 5A — wiki render engine tests.

Covers the markdown → HTML pipeline: PageIndex construction, wikilink
resolution (4 link states per §10), infobox rendering per §4.6, and the
end-to-end ``render_topic_page`` pipeline per §12.4.

No network, no server.  Every test runs against ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from second_brain.frontmatter import dump_frontmatter
from second_brain.state import BrainStateStore
from second_brain.web.index_model import PageEntry, PageIndex
from second_brain.web.infobox import render_infobox
from second_brain.web.render import _strip_computed_sections, render_topic_page
from second_brain.web.wikilink import render_markdown

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeCfg:
    """Minimal config stub — satisfies BrainStateStore's ``cfg.brain_root``."""

    def __init__(self, brain_root: Path) -> None:
        self.brain_root = brain_root


# ---------------------------------------------------------------------------
# PageIndex
# ---------------------------------------------------------------------------


class TestPageIndexFromStore:
    """PageIndex.from_store builds entries and backlinks correctly."""

    def test_from_store_and_backlinks(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.ensure_topic("topic-a", "Topic A")
        store.ensure_topic("topic-b", "Topic B")
        store.record_link("topic-a", "topic-b")

        index = PageIndex.from_store(store)

        assert "topic-a" in index.entries
        assert "topic-b" in index.entries
        assert index.entries["topic-a"].links_to == ["topic-b"]
        assert index.entries["topic-b"].linked_from == ["topic-a"]
        # Backlinks: topic-b has topic-a linking to it
        assert index.backlinks("topic-b") == ["topic-a"]
        # topic-a has nothing linking to it
        assert index.backlinks("topic-a") == []

    def test_empty_store(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        index = PageIndex.from_store(store)
        assert index.entries == {}

    def test_type_is_string(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.ensure_topic("test", "Test")
        index = PageIndex.from_store(store)
        assert index.entries["test"].type == "concept"


class TestPageIndexResolve:
    """Wikilink target resolution (title, alias, slug, missing)."""

    def _make_index(self) -> PageIndex:
        return PageIndex(
            {
                "rag-and-vector-search": PageEntry(
                    slug="rag-and-vector-search",
                    title="RAG & Vector Search",
                    type="concept",
                    aliases=["RAG", "Retrieval-Augmented Generation"],
                ),
                "llm-fundamentals": PageEntry(
                    slug="llm-fundamentals",
                    title="LLM Fundamentals",
                    type="concept",
                ),
            }
        )

    def test_resolve_by_slug(self) -> None:
        index = self._make_index()
        assert index.resolve("rag-and-vector-search") == "rag-and-vector-search"

    def test_resolve_by_title(self) -> None:
        index = self._make_index()
        # Title match is case-insensitive
        assert index.resolve("llm fundamentals") == "llm-fundamentals"
        assert index.resolve("LLM Fundamentals") == "llm-fundamentals"
        assert index.resolve("RAG & Vector Search") == "rag-and-vector-search"

    def test_resolve_by_alias(self) -> None:
        index = self._make_index()
        assert index.resolve("RAG") == "rag-and-vector-search"
        assert index.resolve("Retrieval-Augmented Generation") == "rag-and-vector-search"
        # Alias match is case-insensitive
        assert index.resolve("rag") == "rag-and-vector-search"

    def test_resolve_missing(self) -> None:
        index = self._make_index()
        assert index.resolve("Nonexistent") is None
        assert index.resolve("Totally Unknown Topic") is None

    def test_slug_for_target(self) -> None:
        index = self._make_index()
        assert index.slug_for_target("Some Topic") == "some-topic"
        assert index.slug_for_target("Foo (Bar)") == "foo-bar"

    def test_exists(self) -> None:
        index = self._make_index()
        assert index.exists("rag-and-vector-search") is True
        assert index.exists("missing-slug") is False


# ---------------------------------------------------------------------------
# Wikilink rendering  (§10)
# ---------------------------------------------------------------------------


class TestWikilinkRendering:
    """Four link states: exists, missing, self, piped — plus external."""

    def _make_index(self) -> PageIndex:
        return PageIndex(
            {
                "exists": PageEntry(slug="exists", title="Exists", type="concept"),
                "self": PageEntry(slug="self", title="Self", type="concept"),
                "capital-case": PageEntry(
                    slug="capital-case",
                    title="Capital Case",
                    type="concept",
                ),
            }
        )

    def test_exists_blue_link(self) -> None:
        """[[Exists]] → blue `<a>` with proper href and class."""
        index = self._make_index()
        html = render_markdown("before [[Exists]] after", "current", index)
        assert 'href="/topic/exists"' in html
        assert 'class="wikilink"' in html
        assert "wikilink--missing" not in html
        assert "wikilink--self" not in html
        # Display is the page name
        assert "Exists</a>" in html

    def test_missing_red_link(self) -> None:
        """[[Missing]] → red link with ``?action=create``."""
        index = self._make_index()
        html = render_markdown("[[Missing]]", "current", index)
        assert "?action=create" in html
        assert 'class="wikilink wikilink--missing"' in html
        assert "(page does not exist)" in html

    def test_self_is_span(self) -> None:
        """[[Self]] on the Self page → ``<span class="wikilink--self">``."""
        index = self._make_index()
        html = render_markdown("[[Self]]", "self", index)
        assert 'class="wikilink wikilink--self"' in html
        assert "<a" not in html  # No link tag

    def test_piped_display(self) -> None:
        """[[Exists|custom display]] → custom text in the link."""
        index = self._make_index()
        html = render_markdown("[[Exists|custom]]", "current", index)
        assert "custom</a>" in html
        assert 'title="Exists"' in html

    def test_title_case_target_resolves(self) -> None:
        """[[Capital Case]] resolves by title match (case-insensitive)."""
        index = self._make_index()
        html = render_markdown("[[Capital Case]]", "current", index)
        assert 'href="/topic/capital-case"' in html
        assert 'class="wikilink"' in html

    def test_external_link_has_extlink_class(self) -> None:
        """External markdown links get ``target=_blank`` and ``extlink``."""
        index = self._make_index()
        html = render_markdown("[example](https://example.com)", "current", index)
        assert 'target="_blank"' in html
        assert 'rel="noopener"' in html
        assert 'class="extlink"' in html

    def test_section_link(self) -> None:
        """[[Exists#section]] → href includes #section."""
        index = self._make_index()
        html = render_markdown("[[Exists#details]]", "current", index)
        assert 'href="/topic/exists#details"' in html
        assert "details</a>" in html or "Exists#details</a>" in html

    def test_section_with_pipe(self) -> None:
        """[[Exists#section|show]] → custom display + #section in href."""
        index = self._make_index()
        html = render_markdown("[[Exists#details|show]]", "current", index)
        assert 'href="/topic/exists#details"' in html
        assert "show#details</a>" in html or "show</a>" not in html

    def test_wikilink_inside_paragraph(self) -> None:
        """Wikilink inside a paragraph does not break surrounding text."""
        index = self._make_index()
        html = render_markdown("Hello [[Exists]] world.", "current", index)
        assert html.startswith("<p>") or "<p>" in html
        assert "Hello " in html
        assert " world." in html


# ---------------------------------------------------------------------------
# Infobox  (§4.6)
# ---------------------------------------------------------------------------


class TestInfobox:
    """Typed infobox rendering — concept, note, unknown types."""

    def test_concept_infobox(self) -> None:
        meta = {"title": "RAG", "created": "2026-01-15", "confidence": 0.85}
        html = render_infobox(meta, "concept", 3)
        assert html is not None
        assert '<aside class="infobox">' in html
        assert "<table>" in html
        assert "Source count" in html
        assert "3" in html  # source count value
        assert "2026-01-15" in html  # First seen = created
        assert "0.85" in html  # Confidence
        assert "</aside>" in html

    def test_note_returns_none(self) -> None:
        assert render_infobox({}, "note", 0) is None

    def test_unknown_type_returns_none(self) -> None:
        assert render_infobox({}, "unknown-type", 0) is None

    def test_person_infobox_aliases(self) -> None:
        meta = {"aliases": ["John Doe", "JD"], "role": "Engineer"}
        html = render_infobox(meta, "person", 2)
        assert html is not None
        assert "John Doe, JD" in html
        assert "Engineer" in html
        assert "2" in html  # Sources

    def test_work_infobox(self) -> None:
        meta = {"author": "Jane", "kind": "book", "year": 2024, "tldr": "Great book"}
        html = render_infobox(meta, "work", 1)
        assert html is not None
        assert "Jane" in html
        assert "book" in html
        assert "2024" in html
        assert "Great book" in html

    def test_missing_fields_render_dash(self) -> None:
        html = render_infobox({}, "concept", 0)
        assert html is not None
        assert "—" in html  # Missing fields use em-dash

    def test_all_types_have_infobox_except_note(self) -> None:
        for t in ("person", "work", "project", "tool", "place", "event"):
            html = render_infobox({}, t, 0)
            assert html is not None, f"type '{t}' should have infobox"
        assert render_infobox({}, "note", 0) is None


# ---------------------------------------------------------------------------
# Section stripping
# ---------------------------------------------------------------------------


class TestStripComputedSections:
    """``## See also`` is removed before rendering."""

    def test_strips_see_also(self) -> None:
        body = "## Synthesis\n\ncontent\n\n## See also\n\n[[Other]]\n\n## Sources\n\nsource"
        result = _strip_computed_sections(body)
        assert "## See also" not in result
        assert "[[Other]]" not in result
        assert "## Synthesis" in result
        assert "## Sources" in result

    def test_no_see_also_is_noop(self) -> None:
        body = "## Synthesis\n\ncontent\n\n## Sources\n\nsource"
        result = _strip_computed_sections(body)
        assert result == body

    def test_empty_body(self) -> None:
        assert _strip_computed_sections("") == ""
        assert _strip_computed_sections("\n\n") == "\n\n"


# ---------------------------------------------------------------------------
# Full pipeline  (§12.4)
# ---------------------------------------------------------------------------


class TestRenderTopicPage:
    """End-to-end ``render_topic_page`` with real files and store."""

    def test_render_topic_page(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.ensure_topic("test-topic", "Test Topic")
        store.ensure_topic("other", "Other Topic")

        # Create the wiki page on disk
        wiki_dir = tmp_path / "90-wiki"
        wiki_dir.mkdir()
        page = wiki_dir / "test-topic.md"
        meta = {
            "title": "Test",
            "type": "concept",
            "created": "2026-06-01",
            "confidence": 0.9,
            "sources": ["src1", "src2"],
        }
        body = "\n## Synthesis\n\nHello [[Other Topic]] world.\n\n## See also\n\nstale content\n"
        page.write_text(dump_frontmatter(meta, body))

        # Register a backlink: "other" links to "test-topic"
        store.record_link("other", "test-topic")

        result = render_topic_page("test-topic", store)

        assert result.slug == "test-topic"
        assert result.title == "Test"
        assert result.infobox is not None
        assert "2" in result.infobox  # source_count

        # Body should have wikilink resolved, no stale See also
        assert 'href="/topic/other"' in result.html_body
        assert "## See also" not in result.html_body

        # Breadcrumbs
        assert result.breadcrumbs == [("Home", "/"), ("Test", "/topic/test-topic")]

        # See also (computed backlinks)
        assert len(result.see_also) == 1
        assert result.see_also[0] == ("other", "Other Topic")

    def test_missing_page_file(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        with pytest.raises(FileNotFoundError):
            render_topic_page("nonexistent", store)

    def test_stale_see_also_stripped_from_html(self, tmp_path: Path) -> None:
        """Ensure authored ``## See also`` is not present in html_body."""
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.ensure_topic("test", "Test")

        wiki_dir = tmp_path / "90-wiki"
        wiki_dir.mkdir()
        page = wiki_dir / "test.md"
        page.write_text(
            "---\ntitle: Test\ntype: concept\n---\n\n"
            "## Synthesis\n\nbody\n\n"
            "## See also\n\n[[Old]]\n"
        )

        result = render_topic_page("test", store)
        assert "## See also" not in result.html_body
        # No backlinks -> empty see_also
        assert result.see_also == []

    def test_note_type_no_infobox(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.ensure_topic("note-page", "Note Page")

        wiki_dir = tmp_path / "90-wiki"
        wiki_dir.mkdir()
        page = wiki_dir / "note-page.md"
        meta = {"title": "Note", "type": "note"}
        page.write_text(dump_frontmatter(meta, "\n## Synthesis\n\njust a note\n"))

        result = render_topic_page("note-page", store)
        assert result.infobox is None

    def test_meta_preserved(self, tmp_path: Path) -> None:
        """meta dict is passed through unchanged."""
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.ensure_topic("meta-test", "Meta Test")

        wiki_dir = tmp_path / "90-wiki"
        wiki_dir.mkdir()
        page = wiki_dir / "meta-test.md"
        page.write_text(
            "---\ntitle: Meta Test\ntype: concept\ncustom_field: hello\n"
            "---\n\n## Synthesis\n\nbody\n"
        )

        result = render_topic_page("meta-test", store)
        assert result.meta["custom_field"] == "hello"
        assert result.meta["title"] == "Meta Test"
