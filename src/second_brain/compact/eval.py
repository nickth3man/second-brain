"""Health-check evaluation — L0 structural + L1 heuristics (§11, §12.5).

Runs every ingest (code-only, no API).  Produces a dict consumed by the
health panel (Phase 5) and ``INDEX.md`` health section.

References
----------
- ARCHITECTURE.md §11 (anti-graveyard: decay vectors, near-dup, confidence,
  health report)
- ARCHITECTURE.md §12.5 item 4a (eval MVP slice: L0 + L1, free/code-only,
  every ingest)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def run_health_check(cfg: object, store: object) -> dict:  # noqa: ARG001
    """Scan ``store.state`` + wiki files and return a health report dict.

    Fields returned
    ---------------
    source_count, topic_count
        Straight counts.
    orphans : dict
        ``{"sources": [...], "topics": [...]}`` — sources with empty
        ``topics`` AND topic slugs with empty ``sources``.
    broken_links : list[tuple[str, str]]
        ``(from_slug, target_slug)`` pairs where a ``links_to`` entry
        points to a slug not in ``store.state.topics``.
    near_duplicates : list
        Left empty for L0; the dedup module handles semantic near-dup.
    empty_extractions : list[str]
        Source IDs at stage ``FAILED`` or with empty topics.
    stale_topics : list[str]
        Slugs whose ``updated`` field is older than 90 days.
    avg_confidence : float
        Mean of all topic ``confidence`` values (0.0 if none).
    schema_violations : list[dict]
        Best-effort missing-required-field checks.
    """
    topics = store.state.topics
    sources = store.state.sources

    source_count = len(sources)
    topic_count = len(topics)

    # -- orphans ---------------------------------------------------------
    orphan_sources = [sid for sid, src in sources.items() if not src.topics]
    orphan_topics = [slug for slug, t in topics.items() if not t.sources]
    orphans: dict[str, list[str]] = {
        "sources": orphan_sources,
        "topics": orphan_topics,
    }

    # -- broken links ----------------------------------------------------
    broken_links: list[tuple[str, str]] = []
    for slug, t in topics.items():
        for target in t.links_to:
            if target not in topics:
                broken_links.append((slug, target))

    # -- near duplicates (L0: empty list) --------------------------------
    near_duplicates: list[tuple[str, str, float]] = []

    # -- empty extractions -----------------------------------------------
    empty_extractions: list[str] = [
        sid
        for sid, src in sources.items()
        if src.stage == "failed" or not src.topics
    ]

    # -- stale topics (>90 days since updated) ---------------------------
    now = datetime.now(UTC)
    stale_topics: list[str] = []
    for slug, t in topics.items():
        try:
            updated_str = t.updated
            if updated_str.endswith("Z"):
                updated_str = updated_str[:-1] + "+00:00"
            updated_dt = datetime.fromisoformat(updated_str)
            if (now - updated_dt) > timedelta(days=90):
                stale_topics.append(slug)
        except (ValueError, TypeError, AttributeError):
            stale_topics.append(slug)  # unparseable date counts as stale

    # -- average confidence ----------------------------------------------
    avg_confidence = (
        sum(t.confidence for t in topics.values()) / len(topics) if topics else 0.0
    )

    # -- schema violations (best-effort) ---------------------------------
    schema_violations: list[dict] = []
    for slug, t in topics.items():
        if not t.title:
            schema_violations.append(
                {"entity": slug, "field": "title", "issue": "empty"}
            )
    for sid, src in sources.items():
        if not src.sha256:
            schema_violations.append(
                {"entity": sid, "field": "sha256", "issue": "empty"}
            )

    return {
        "source_count": source_count,
        "topic_count": topic_count,
        "orphans": orphans,
        "broken_links": broken_links,
        "near_duplicates": near_duplicates,
        "empty_extractions": empty_extractions,
        "stale_topics": stale_topics,
        "avg_confidence": avg_confidence,
        "schema_violations": schema_violations,
    }


def render_health_markdown(report: dict) -> str:
    """Render the health report as a ``## Brain Health`` markdown section.

    Suitable for appending to ``INDEX.md``.
    """
    o = report["orphans"]
    n_orphan_sources = len(o["sources"])
    n_orphan_topics = len(o["topics"])
    total_orphans = n_orphan_sources + n_orphan_topics

    lines = [
        "## Brain Health\n",
        "\n",
        f"- **Sources**: {report['source_count']}\n",
        f"- **Topics**: {report['topic_count']}\n",
        f"- **Orphans**: {total_orphans}"
        f" ({n_orphan_sources} sources, {n_orphan_topics} topics)\n",
        f"- **Broken links**: {len(report['broken_links'])}\n",
        f"- **Empty extractions**: {len(report['empty_extractions'])}\n",
        f"- **Stale topics** (>90d): {len(report['stale_topics'])}\n",
        f"- **Avg confidence**: {report['avg_confidence']:.3f}\n",
        f"- **Schema violations**: {len(report['schema_violations'])}\n",
    ]
    return "".join(lines)
