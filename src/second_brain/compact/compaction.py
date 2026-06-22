"""Scheduled self-improvement pass — compaction/merge (§8).

Performs:
1. Topic pair discovery via centroid cosine similarity.
2. Merge of similar topics (slug_b -> slug_a).
3. Rewrite of merged topic's ``## Synthesis`` via LLM.
4. Changelog audit trail.

**Compaction cadence:** daily OR every 25 sources (caller decides when
to invoke).  **Conservative:** merge at >=0.85, NEVER delete, rewrite
stale synthesis, refresh open-questions/related, log every change.

References
----------
- ARCHITECTURE.md §8 (compaction: cadence, merge threshold, no-delete
  rule, changelog)
"""

from __future__ import annotations

from second_brain.atomicio import write_atomic
from second_brain.compact.dedup import cosine
from second_brain.frontmatter import dump_frontmatter, split_frontmatter
from second_brain.state import now_iso

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_synthesis(text: str) -> str:
    """Extract the ``## Synthesis`` section body from a wiki page text.

    Returns everything between the ``## Synthesis`` heading and the next
    ``## `` heading (or end of body).  Returns empty string if not found.
    """
    _, body = split_frontmatter(text)
    lines = body.split("\n")
    in_synthesis = False
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## Synthesis"):
            in_synthesis = True
            continue
        if in_synthesis and stripped.startswith("## "):
            break
        if in_synthesis:
            collected.append(line)
    return "\n".join(collected).strip()


def _topic_page_path(cfg: object, slug: str) -> object:
    """Return the path to a topic's wiki page."""
    return cfg.brain_root / "90-wiki" / f"{slug}.md"


# ---------------------------------------------------------------------------
# Topic pair discovery
# ---------------------------------------------------------------------------


def _topic_pairs_by_similarity(
    vec_store: object,
    store: object,
    threshold: float,
) -> list[tuple[str, str, float]]:
    """Find topic pairs whose centroid cosine >= *threshold*.

    Recomputes each topic's centroid, then compares all pairs via
    :func:`cosine`.  Returns pairs sorted by similarity descending.
    O(n^2) over topics — fine for MVP.
    """
    topics = store.state.topics
    if len(topics) < 2:
        return []

    centroids: dict[str, list[float]] = {}
    for slug in topics:
        centroid = vec_store.recompute_centroid(slug)
        if centroid is not None:
            centroids[slug] = centroid

    slugs = list(centroids.keys())
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(slugs)):
        for j in range(i + 1, len(slugs)):
            a, b = slugs[i], slugs[j]
            sim = cosine(centroids[a], centroids[b])
            if sim >= threshold:
                pairs.append((a, b, sim))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# ---------------------------------------------------------------------------
# Synthesis rewrite
# ---------------------------------------------------------------------------


async def _rewrite_synthesis(
    client: object,
    cfg: object,
    slug_a: str,
    slug_b: str,
    store: object,  # noqa: ARG001
) -> str:
    """Rewrite the merged topic's ``## Synthesis`` by calling the LLM.

    Reads both wiki pages' Synthesis sections and prompts the model to
    produce a single unified section.
    """
    path_a = _topic_page_path(cfg, slug_a)
    path_b = _topic_page_path(cfg, slug_b)

    synthesis_a = _extract_synthesis(
        path_a.read_text(encoding="utf-8") if path_a.exists() else ""
    )
    synthesis_b = _extract_synthesis(
        path_b.read_text(encoding="utf-8") if path_b.exists() else ""
    )

    prompt = (
        "You are merging two wiki topic pages. Produce a single unified "
        "## Synthesis section (markdown) that integrates both, removing "
        "redundancy.\n\n"
        f"Pages: {synthesis_a} / {synthesis_b}"
    )
    _, clean_content = await client.chat_completion_clean(
        cfg.models.text,
        [{"role": "user", "content": prompt}],
    )
    return clean_content


# ---------------------------------------------------------------------------
# Main compaction entry point
# ---------------------------------------------------------------------------


async def run_compaction(
    cfg: object,
    store: object,
    vec_store: object,
    client: object,
    *,
    merge_threshold: float = 0.85,
) -> dict:
    """Run one compaction pass over the brain.

    1. Find topic centroid pairs >= *merge_threshold*.
    2. For each pair (highest similarity first): merge slug_b INTO slug_a
       (a = more sources, or lexicographic tiebreak).
    3. Rewrite a's ``## Synthesis`` via LLM.
    4. Log each merge to the changelog.
    5. ``store.save()``.

    **Never deletes** wiki pages or state entries (§8).  The merged-from
    page gets a redirect note.

    Returns:
        A summary dict with ``merges`` (int), ``pairs`` (list), and
        ``merged_into`` (dict[b->a]).
    """
    pairs = _topic_pairs_by_similarity(vec_store, store, merge_threshold)
    if not pairs:
        from second_brain.compact.eval import write_topic_source_cosine_metric

        write_topic_source_cosine_metric(cfg, store, vec_store)
        return {"merges": 0, "pairs": [], "merged_into": {}}

    merges = 0
    merged_pairs: list[tuple[str, str, float]] = []
    merged_into: dict[str, str] = {}
    merged_this_pass: set[str] = set()

    for slug_a, slug_b, sim in pairs:
        if slug_a in merged_this_pass or slug_b in merged_this_pass:
            continue

        topic_a = store.state.topics[slug_a]
        topic_b = store.state.topics[slug_b]

        # a = topic with more sources; lexicographic tiebreak
        if len(topic_b.sources) > len(topic_a.sources) or (
            len(topic_b.sources) == len(topic_a.sources) and slug_b < slug_a
        ):
            slug_a, slug_b = slug_b, slug_a
            topic_a, topic_b = topic_b, topic_a

        # Capture original source counts BEFORE merging (needed for weighted confidence).
        a_count = len(topic_a.sources)
        b_count = len(topic_b.sources)

        # -- merge sources (union) ----------------------------------
        for src in topic_b.sources:
            if src not in topic_a.sources:
                topic_a.sources.append(src)

        # -- merge links_to (union, avoid self-link) -----------------
        for link in topic_b.links_to:
            if link not in topic_a.links_to and link != slug_a:
                topic_a.links_to.append(link)

        # -- merge linked_from (union, avoid self-link) --------------
        for link in topic_b.linked_from:
            if link not in topic_a.linked_from and link not in (slug_a, slug_b):
                topic_a.linked_from.append(link)

        # -- redirect links that pointed to/from slug_b --------------
        for link_slug in topic_b.linked_from:
            if link_slug in store.state.topics:
                other = store.state.topics[link_slug]
                if slug_b in other.links_to:
                    other.links_to.remove(slug_b)
                    if slug_a not in other.links_to:
                        other.links_to.append(slug_a)

        for link_slug in topic_b.links_to:
            if link_slug in store.state.topics:
                other = store.state.topics[link_slug]
                if slug_b in other.linked_from:
                    other.linked_from.remove(slug_b)
                    if slug_a not in other.linked_from:
                        other.linked_from.append(slug_a)

        # -- merge tags & aliases (union) ----------------------------
        for tag in topic_b.tags:
            if tag not in topic_a.tags:
                topic_a.tags.append(tag)
        for alias in topic_b.aliases:
            if alias not in topic_a.aliases:
                topic_a.aliases.append(alias)

        # -- confidence (weighted by source count) -------------------
        total_sources = a_count + b_count
        if total_sources > 0:
            topic_a.confidence = (
                topic_a.confidence * a_count
                + topic_b.confidence * b_count
            ) / total_sources

        # -- update timestamps ---------------------------------------
        now = now_iso()
        topic_a.updated = now
        topic_b.updated = now

        # -- move vec_store memberships and recompute centroid -------
        for src in topic_b.sources:
            vec_store.add_topic_member(slug_a, src)
        vec_store.recompute_centroid(slug_a)

        # -- write redirect note in slug_b's wiki page ---------------
        b_path = _topic_page_path(cfg, slug_b)
        if b_path.exists():
            meta, body = split_frontmatter(b_path.read_text(encoding="utf-8"))
            body = f"> Merged into [[{slug_a}]]\n\n{body}"
            meta["updated"] = now[:10]
            write_atomic(b_path, dump_frontmatter(meta, body))

        # -- rewrite a's synthesis via LLM ---------------------------
        new_synthesis = await _rewrite_synthesis(client, cfg, slug_a, slug_b, store)
        a_path = _topic_page_path(cfg, slug_a)
        if a_path.exists():
            a_meta, a_body = split_frontmatter(
                a_path.read_text(encoding="utf-8")
            )

            # Replace the ## Synthesis section body.
            lines = a_body.split("\n")
            new_lines: list[str] = []
            in_synthesis = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("## Synthesis"):
                    in_synthesis = True
                    new_lines.append(line)
                    new_lines.append(new_synthesis)
                    continue
                if in_synthesis and stripped.startswith("## "):
                    in_synthesis = False
                    new_lines.append(line)
                    continue
                if not in_synthesis:
                    new_lines.append(line)

            a_meta["updated"] = now[:10]
            a_meta["source_count"] = len(topic_a.sources)
            a_meta["confidence"] = topic_a.confidence

            write_atomic(a_path, dump_frontmatter(a_meta, "\n".join(new_lines)))

        # -- changelog -----------------------------------------------
        store.append_changelog({
            "kind": "compact",
            "action": "merge",
            "from": slug_b,
            "into": slug_a,
            "similarity": sim,
        })
        store.append_changelog({
            "kind": "compact",
            "action": "rewrite_synthesis",
            "topic": slug_a,
        })

        merged_this_pass.add(slug_a)
        merged_this_pass.add(slug_b)
        merged_pairs.append((slug_a, slug_b, sim))
        merged_into[slug_b] = slug_a
        merges += 1

    store.save()
    from second_brain.compact.eval import write_topic_source_cosine_metric

    write_topic_source_cosine_metric(cfg, store, vec_store)
    return {
        "merges": merges,
        "pairs": merged_pairs,
        "merged_into": merged_into,
    }
