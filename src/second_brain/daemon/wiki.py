"""Wiki page writer/merger — ``90-wiki/<slug>.md`` per §4.2.

Two entry points:

* :func:`write_new_topic` — creates a fresh page.
* :func:`merge_into_topic` — appends to an existing page (mechanical merge;
  Phase 4 will add AI-compaction).
"""

from __future__ import annotations

from pathlib import Path

from second_brain.atomicio import write_atomic
from second_brain.config import Config
from second_brain.frontmatter import dump_frontmatter, split_frontmatter
from second_brain.models import LinkDecision


def _date(iso: str) -> str:
    """Extract ``YYYY-MM-DD`` from an ISO 8601 string."""
    return iso[:10]


def _topic_path(cfg: Config, slug: str) -> Path:
    return cfg.brain_root / "90-wiki" / f"{slug}.md"


def write_new_topic(
    cfg: Config,
    store,
    slug: str,
    title: str,
    decision: LinkDecision,
    source_id: str,
    ingested_iso: str,
) -> None:
    """Create a fresh wiki page for a brand-new topic (``action == NEW``).

    Front-matter is initialised per §4.2.  The ``## Sources`` entry uses a
    one-liner derived from *decision.merged_section* (the first non-empty
    line) as its summary.
    """
    dt = _date(ingested_iso)
    # Derive a one-line summary from the merged section
    tldr = _derive_tldr(decision.merged_section)

    meta = {
        "title": title,
        "slug": slug,
        "type": "concept",
        "tags": [],
        "aliases": [],
        "created": dt,
        "updated": dt,
        "source_count": 1,
        "confidence": decision.confidence,
        "related": [],
    }

    body = (
        f"# {title}\n\n"
        f"## Synthesis\n{decision.merged_section}\n\n"
        f"## Sources\n"
        f"- **[{dt}]** {title}\n"
        f"  -> [source](../50-sources/{source_id}.md)\n"
        f"  > {tldr}\n\n"
        f"## Open questions\n- \n\n"
        f"## Related\n"
    )

    write_atomic(_topic_path(cfg, slug), dump_frontmatter(meta, body))


def merge_into_topic(
    cfg: Config,
    store,
    slug: str,
    decision: LinkDecision,
    source_id: str,
    ingested_iso: str,
    tldr: str,
) -> None:
    """Append a new source's contribution to an existing wiki page.

    Splices a ``### From {source_id}`` sub-heading and merged section into
    ``## Synthesis``, and adds a new entry to ``## Sources``.  Front-matter
    counters are bumped atomically.
    """
    path = _topic_path(cfg, slug)
    text = path.read_text(encoding="utf-8")
    meta, body = split_frontmatter(text)

    dt = _date(ingested_iso)
    existing_title = meta.get("title", slug)

    # Content to splice in
    new_synthesis = f"\n\n### From {source_id}\n{decision.merged_section}"
    new_source_entry = (
        f"- **[{dt}]** {existing_title}\n"
        f"  -> [source](../50-sources/{source_id}.md)\n"
        f"  > {tldr}\n"
    )

    # Splice the new synthesis block *before* ## Sources
    sources_marker = "\n## Sources\n"
    sources_pos = body.find(sources_marker)
    if sources_pos >= 0:
        before = body[:sources_pos]
        after = body[sources_pos:]
        body = before + new_synthesis + after

        # Now insert the source entry right after the ## Sources heading
        new_sources_pos = body.find(sources_marker)
        after_heading = new_sources_pos + len(sources_marker)
        body = body[:after_heading] + new_source_entry + body[after_heading:]
    else:
        # No Sources section yet — append everything
        body += (
            new_synthesis
            + f"\n\n## Sources\n{new_source_entry}"
            + "\n\n## Open questions\n- \n\n## Related\n"
        )

    # Update front-matter
    meta["updated"] = dt
    meta["source_count"] = int(meta.get("source_count", 0)) + 1
    meta["confidence"] = max(
        float(meta.get("confidence", 0.0)), decision.confidence
    )

    write_atomic(path, dump_frontmatter(meta, body))


def _derive_tldr(merged_section: str) -> str:
    """Extract a short one-liner from the merged section."""
    for line in merged_section.strip().split("\n"):
        line = line.strip()
        if line:
            return line[:120]
    return "(summary pending)"
