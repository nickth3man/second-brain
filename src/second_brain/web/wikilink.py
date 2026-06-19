"""Wikilink renderer — mistune 3 inline plugin + §10 rendering matrix.

Uses the proper mistune 3 plugin API: registers an inline rule for
``[[wikilinks]]`` and a renderer method that consults the ``PageIndex``
for existence / self-link detection.

See ARCHITECTURE.md §10 (rendering matrix) and §12.4 (pipeline).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mistune import HTMLRenderer, Markdown, create_markdown, safe_entity

from second_brain.slug import slugify

if TYPE_CHECKING:
    from second_brain.web.index_model import PageIndex

# ---------------------------------------------------------------------------
# Regex — captures target, optional section, optional display
# ---------------------------------------------------------------------------
#: Matches ``[[target]]``, ``[[target#section]]``, ``[[target|display]]``,
#: and ``[[target#section|display]]``.
#: Uses named groups (``wl_`` prefix) to avoid group-number conflicts in
#: mistune's combined inline regex (``compile_sc`` joins all rules via ``|``).
WIKILINK_RE = (
    r"\[\["
    r"(?P<wl_target>[^\]|#]+)"
    r"(?:#(?P<wl_section>[^\]|]+))?"
    r"(?:\|(?P<wl_display>[^\]]+))?"
    r"\]\]"
)

# ---------------------------------------------------------------------------
# Custom HTML renderer (external links)
# ---------------------------------------------------------------------------


class _ExtLinkRenderer(HTMLRenderer):
    """HTML renderer that marks external links with ``target=_blank`` etc.

    ``NAME`` stays ``"html"`` so plugins that conditionally register on
    ``md.renderer.NAME == "html"`` still apply.
    """

    NAME = "html"

    def link(self, text: str, url: str, title: str | None = None) -> str:
        if url and ("://" in url or url.startswith("//")):
            s = '<a href="' + self.safe_url(url) + '"'
            s += ' target="_blank" rel="noopener" class="extlink"'
            if title:
                s += ' title="' + safe_entity(title) + '"'
            return s + ">" + text + "</a>"
        return super().link(text, url, title)


# ---------------------------------------------------------------------------
# Plugin factory — closure captures index + current_slug
# ---------------------------------------------------------------------------


def _make_wikilink_plugin(index: PageIndex, current_slug: str):
    """Create a mistune plugin that resolves ``[[wikilinks]]`` per §10.

    Closure captures *index* and *current_slug* so the renderer can decide
    the link state (exists / missing / self).
    """

    def parse_wikilink(inline, m, state):
        target = m.group("wl_target").strip()
        section = m.group("wl_section") or ""
        display = m.group("wl_display") or target
        state.append_token(
            {
                "type": "wikilink",
                "raw": target,
                "attrs": {
                    "target": target,
                    "section": section,
                    "display": display,
                },
            }
        )
        return m.end()

    def render_wikilink(renderer, text: str, target: str, section: str, display: str) -> str:
        slug = slugify(target)
        resolved = index.resolve(target)
        display_escaped = safe_entity(display)
        href_section = ""

        if section:
            display_escaped = safe_entity(f"{display}#{section}")
            href_section = f"#{section}"

        # Self-link — span, not clickable (§10).
        if slug == current_slug:
            return f'<span class="wikilink wikilink--self">{display_escaped}</span>'

        # Exists — blue link (§10).
        if resolved is not None:
            return (
                f'<a href="/topic/{resolved}{href_section}"'
                f' title="{safe_entity(target)}"'
                f' class="wikilink">'
                f"{display_escaped}"
                f"</a>"
            )

        # Missing — red link with create action (§10).
        return (
            f'<a href="/topic/{slug}{href_section}?action=create"'
            f' title="{safe_entity(target)} (page does not exist)"'
            f' class="wikilink wikilink--missing">'
            f"{display_escaped}"
            f"</a>"
        )

    def plugin(md: Markdown) -> None:
        md.inline.register(
            "wikilink",
            WIKILINK_RE,
            parse_wikilink,
            before="link",
        )
        if md.renderer and md.renderer.NAME == "html":
            md.renderer.register("wikilink", render_wikilink)

    return plugin


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_markdown(text: str, current_slug: str, index: PageIndex) -> str:
    """Render *text* (markdown + wikilinks) to HTML.

    Args:
        text: Markdown body that may contain ``[[wikilinks]]``.
        current_slug: Slug of the page being rendered (for self-link
            detection).
        index: A ``PageIndex`` used to resolve existence.

    Returns:
        Full HTML string with wikilinks rendered per §10.
    """
    plugin = _make_wikilink_plugin(index, current_slug)
    md = create_markdown(
        renderer=_ExtLinkRenderer(),
        plugins=[plugin],
    )
    return cast(str, md(text))
