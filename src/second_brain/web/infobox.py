"""Typed infobox renderer — §4.6 page-type schemas.

Each page ``type`` selects an ordered list of display fields.  The renderer
fills values from the front-matter dict (``meta``) and a source count, then
emits an ``<aside class="infobox">`` HTML snippet.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Infobox field definitions  (§4.6 starter types)
# ---------------------------------------------------------------------------
INFOBOX_FIELDS: dict[str, list[str]] = {
    "concept": [
        "Key idea",
        "Source count",
        "First seen",
        "Updated",
        "Confidence",
        "Top sources",
    ],
    "person": [
        "Role",
        "Affiliation",
        "Aliases",
        "First mentioned",
        "Sources",
    ],
    "work": [
        "Author",
        "Kind",
        "Year",
        "Link",
        "TL;DR",
        "Sources",
    ],
    "project": [
        "Status",
        "Started",
        "Stack",
        "Related topics",
        "Sources",
    ],
    "tool": [
        "Category",
        "Vendor/URL",
        "License",
        "First used",
        "Sources",
    ],
    "place": [
        "Location",
        "Aliases",
        "First mentioned",
        "Sources",
    ],
    "event": [
        "Date",
        "Aliases",
        "Sources",
    ],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _escape(text: str) -> str:
    """Minimal HTML-escape for safe text-content insertion."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _value_for(field: str, meta: dict, source_count: int) -> str:
    """Return the display value for *field* given *meta* and *source_count*.

    Computed fields (Source count, First seen, …) are calculated directly;
    all other fields are looked up by normalising the field name to a
    meta key (lowercased, spaces → underscores).  Missing values render
    as ``—``.
    """
    computed = {
        "Source count": str(source_count),
        "Sources": str(source_count),
        "First seen": str(meta.get("created", "—")),
        "Updated": str(meta.get("updated", "—")),
        "Confidence": f"{meta.get('confidence', 0.0):.2f}",
        "Top sources": str(source_count),
        "TL;DR": str(meta.get("tldr", "—")),
    }
    if field in computed:
        return computed[field]

    # Generic lookup by normalised key
    key = field.lower().replace(" ", "_")
    val = meta.get(key, "—")
    if val is None:
        return "—"
    if isinstance(val, list):
        return ", ".join(str(v) for v in val) if val else "—"
    sval = str(val)
    return sval if sval.strip() else "—"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_infobox(meta: dict, page_type: str, source_count: int) -> str | None:
    """Render an HTML infobox ``<aside>`` for the given page *type*.

    Args:
        meta: Front-matter dict (contains ``title``, ``created``,
            ``updated``, ``confidence``, ``aliases``, …).
        page_type: The page type string (e.g. ``"concept"``, ``"note"``).
        source_count: Number of sources linked to this topic.

    Returns:
        An HTML string, or ``None`` when the page type has no infobox
        (``note``, unknown types).
    """
    fields = INFOBOX_FIELDS.get(page_type)
    if fields is None:
        return None

    rows = "\n".join(
        f"      <tr><th>{_escape(f)}</th><td>{_escape(_value_for(f, meta, source_count))}</td></tr>"
        for f in fields
    )

    return f'<aside class="infobox">\n  <table>\n{rows}\n  </table>\n</aside>'
