"""Topic page render pipeline — §12.4.

Reads a ``90-wiki/<slug>.md`` file, parses front-matter, renders wikilinks
via mistune, builds infobox/breadcrumbs/backlinks, and returns a structured
``RenderedPage`` for the Jinja2 template (Phase 5B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from second_brain.frontmatter import split_frontmatter
from second_brain.state import BrainStateStore
from second_brain.web.index_model import PageIndex
from second_brain.web.infobox import render_infobox
from second_brain.web.wikilink import render_markdown


@dataclass
class RenderedPage:
    """Fully rendered topic page ready for template injection."""

    slug: str
    title: str
    html_body: str
    infobox: str | None
    breadcrumbs: list[tuple[str, str]]
    see_also: list[tuple[str, str]]
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_computed_sections(body: str) -> str:
    """Remove the authored ``## See also`` block (we recompute it)."""
    lines = body.splitlines(keepends=True)
    result: list[str] = []
    in_see_also = False
    for line in lines:
        if line.startswith("## ") and "see also" in line.lower():
            in_see_also = True
            continue
        if in_see_also:
            # Next heading of the same level ends the section.
            if line.startswith("## "):
                in_see_also = False
                result.append(line)
            continue
        result.append(line)
    return "".join(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_topic_page(slug: str, store: BrainStateStore) -> RenderedPage:
    """Render a wiki topic page from disk + store into a ``RenderedPage``.

    Steps (§12.4):
    1. Read ``90-wiki/<slug>.md``; split front-matter.
    2. Build ``PageIndex`` from the store's topic graph.
    3. Strip any stale ``## See also`` section.
    4. Render body (wikilinks + markdown) to HTML.
    5. Build infobox from type + front-matter.
    6. Build breadcrumbs.
    7. Build ``see_also`` from backlinks.

    Args:
        slug: The topic slug (e.g. ``"rag-and-vector-search"``).
        store: A ``BrainStateStore`` with topic state.

    Returns:
        A ``RenderedPage`` ready for Jinja2 templating.

    Raises:
        FileNotFoundError: if the wiki page file does not exist.
    """
    page_path: Path = store.cfg.brain_root / "90-wiki" / f"{slug}.md"
    text = page_path.read_text(encoding="utf-8")
    meta, body = split_frontmatter(text)

    index = PageIndex.from_store(store)
    body = _strip_computed_sections(body)

    page_type = meta.get("type", "concept")
    sources = meta.get("sources") or []
    source_count = len(sources)

    html_body = render_markdown(body, slug, index)
    infobox = render_infobox(meta, page_type, source_count)

    title = meta.get("title", slug)
    breadcrumbs = [("Home", "/"), (title, f"/topic/{slug}")]

    see_also = [(s, index.entries[s].title) for s in index.backlinks(slug) if s in index.entries]

    return RenderedPage(
        slug=slug,
        title=title,
        html_body=html_body,
        infobox=infobox,
        breadcrumbs=breadcrumbs,
        see_also=see_also,
        meta=meta,
    )
