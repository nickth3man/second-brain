"""Rebuild engine — ``brain rebuild --from-sources|--from-inbox`` (§12.6).

Two modes:

* ``--from-sources``: fast path. Re-runs linking/embedding from existing
  ``50-sources/*.md`` without calling the extraction LLM. Preserves each
  topic's existing ``## Synthesis`` verbatim and mechanically reassembles
  ``## Sources``.
* ``--from-inbox``: deep path. Archives derived state, then replays the full
  ingestion pipeline from ``00-inbox/``.

Both modes take an implicit snapshot to ``.brain/snapshots/pre-rebuild-<ts>/``
before destructive work and restore from it if anything fails after the first
write. ``--dry-run`` reports counters and never writes.
"""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel

from second_brain.atomicio import write_atomic, write_json_atomic
from second_brain.config import Config
from second_brain.daemon.index import DebouncedIndex
from second_brain.daemon.linker import EmbeddingLinker
from second_brain.daemon.pipeline import ingest_file
from second_brain.frontmatter import dump_frontmatter, split_frontmatter
from second_brain.models import BrainState, PageType, SourceState, TopicState
from second_brain.openrouter_client import OpenRouterClient
from second_brain.slug import slugify
from second_brain.state import BrainStateStore, now_iso
from second_brain.vectors.embed import Embedder
from second_brain.vectors.store import VectorStore, chunk_text

log = structlog.get_logger(__name__)

SNAPSHOT_DIR_PREFIX = "pre-rebuild-"


class RebuildPlan(BaseModel):
    """Summary of a rebuild operation, returned to the CLI for display."""

    mode: str
    dry_run: bool
    sources_seen: int
    sources_skipped: int
    topics_before: int
    topics_after: int
    snapshot_dir: Path | None = None


# -- snapshot / rollback helpers ---------------------------------------------


def _replace_tree(src: Path, dst: Path) -> None:
    """Replace *dst* tree with a copy of *src* (removes *dst* first if it exists)."""
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists() and any(src.iterdir()):
        shutil.copytree(src, dst)


def _snapshot_rebuild_state(cfg: Config, ts: str) -> Path:
    """Copy recoverable state into ``.brain/snapshots/pre-rebuild-<ts>/``.

    Captures ``state.json``, ``changelog.jsonl``, ``embeddings.db`` (if
    present), and the entire ``90-wiki/`` and ``50-sources/`` trees. Dirs are
    created as needed.
    """
    snapshot_dir = cfg.brain_root / ".brain" / "snapshots" / f"{SNAPSHOT_DIR_PREFIX}{ts}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    brain_dir = cfg.brain_root / ".brain"
    src_state = brain_dir / "state.json"
    if src_state.exists():
        shutil.copy2(src_state, snapshot_dir / "state.json")

    src_changelog = brain_dir / "changelog.jsonl"
    if src_changelog.exists():
        shutil.copy2(src_changelog, snapshot_dir / "changelog.jsonl")

    src_embeddings = brain_dir / "embeddings.db"
    if src_embeddings.exists():
        shutil.copy2(src_embeddings, snapshot_dir / "embeddings.db")

    _replace_tree(cfg.brain_root / "90-wiki", snapshot_dir / "90-wiki")
    _replace_tree(cfg.brain_root / "50-sources", snapshot_dir / "50-sources")

    log.info("rebuild.snapshot_taken", snapshot_dir=str(snapshot_dir))
    return snapshot_dir


def _rotate_pre_rebuild_snapshots(cfg: Config, keep: int = 3) -> None:
    """Prune ``.brain/snapshots/pre-rebuild-*`` directories, keeping the newest *keep*.

    ISO timestamps sort lexically, so sorting by name is equivalent to sorting
    by time.
    """
    snapshots_root = cfg.brain_root / ".brain" / "snapshots"
    if not snapshots_root.exists():
        return

    dirs = sorted(
        [d for d in snapshots_root.iterdir() if d.is_dir() and d.name.startswith(SNAPSHOT_DIR_PREFIX)],
        key=lambda d: d.name,
    )
    to_remove = dirs[:-keep] if keep > 0 and len(dirs) > keep else (dirs if keep == 0 else [])
    for old in to_remove:
        try:
            shutil.rmtree(old)
            log.info("rebuild.snapshot_pruned", snapshot_dir=str(old))
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            log.warning("rebuild.snapshot_prune_failed", snapshot_dir=str(old), error=str(exc))


def _restore_from_snapshot(cfg: Config, snapshot_dir: Path) -> None:
    """Best-effort rollback to a pre-rebuild snapshot.

    Replaces the current ``90-wiki/`` and ``50-sources/`` trees with the
    snapshot and copies back ``state.json``, ``changelog.jsonl``, and
    ``embeddings.db``. Errors are logged but not raised — this is called
    during exception handling and must not mask the original failure.
    """
    log.error("rebuild.restore_start", snapshot_dir=str(snapshot_dir))

    _replace_tree(snapshot_dir / "90-wiki", cfg.brain_root / "90-wiki")
    _replace_tree(snapshot_dir / "50-sources", cfg.brain_root / "50-sources")

    brain_dir = cfg.brain_root / ".brain"
    for name in ("state.json", "changelog.jsonl", "embeddings.db"):
        src = snapshot_dir / name
        dst = brain_dir / name
        try:
            if src.exists():
                shutil.copy2(src, dst)
        except Exception as exc:
            log.error("rebuild.restore_file_failed", file=name, error=str(exc))


# -- section extraction helpers ----------------------------------------------


def _extract_section(body: str, heading: str) -> str:
    """Return the content under *heading* (e.g. ``## Synthesis``) up to the next ``## ``.

    Returns the empty string if the section is missing. The returned text
    preserves internal newlines but strips leading/trailing blank lines.
    """
    marker = f"\n{heading}\n"
    pos = body.find(marker)
    if pos < 0:
        # Also try at start of body (no leading newline in marker)
        if body.startswith(f"{heading}\n"):
            pos = 0
            start = len(f"{heading}\n")
        else:
            return ""
    else:
        start = pos + len(marker)
    next_heading = body.find("\n## ", start)
    if next_heading < 0:
        raw = body[start:]
    else:
        raw = body[start:next_heading]
    return raw.strip("\n")


def _derive_title_from_body(body: str) -> str:
    """Pull the first ``# `` heading from a source body as its title."""
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


def _extract_summary(body: str) -> str:
    """Return the contents of the source's ``## Summary`` section."""
    return _extract_section(body, "## Summary")


def _humanize_slug(slug: str) -> str:
    """Create a fallback human title from a slug: hyphen -> space, title case."""
    return slug.replace("-", " ").replace("_", " ").strip().title()


# -- from-sources rebuild ----------------------------------------------------


def _read_source_info(path: Path) -> tuple[dict, str, str] | None:
    """Parse a 50-sources/*.md file and return (meta, body, source_id) or None."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        meta, body = split_frontmatter(text)
    except ValueError:
        return None
    return meta, body, path.stem


async def _embed_sources(
    source_paths: list[Path],
    embedder: Embedder,
) -> dict[str, list[tuple[str, list[float]]]]:
    """Chunk and embed every valid source body.

    Returns a mapping ``source_id -> [(chunk_text, vector), ...]``.
    """
    source_chunks: dict[str, list[tuple[str, list[float]]]] = {}
    for path in source_paths:
        info = _read_source_info(path)
        if info is None:
            continue
        meta, body, source_id = info
        chunks = chunk_text(body)
        if chunks:
            vectors = await embedder.embed_texts(chunks)
            source_chunks[source_id] = list(zip(chunks, vectors, strict=False))
        else:
            source_chunks[source_id] = []
    return source_chunks


def _build_fresh_state_from_sources(
    cfg: Config,
    store: BrainStateStore,
    valid_sources: list[tuple[Path, dict, str, list[str]]],
) -> tuple[BrainState, dict[str, list[str]]]:
    """Build a fresh ``BrainState`` purely from source front-matter + existing wiki.

    *valid_sources* is a list of ``(path, meta, body, topics)`` for sources that
    are not skipped. Returns ``(fresh_state, source_id_to_topics)``.
    """
    fresh_state = BrainState()
    source_id_to_topics: dict[str, list[str]] = {}

    for path, meta, body, topics in valid_sources:
        source_id = path.stem
        source_id_to_topics[source_id] = topics

        for slug in topics:
            if slug not in fresh_state.topics:
                # Try to read existing wiki page front-matter for title/type/etc.
                wiki_path = cfg.brain_root / "90-wiki" / f"{slug}.md"
                existing_meta: dict = {}
                if wiki_path.exists():
                    try:
                        existing_meta, _ = split_frontmatter(wiki_path.read_text(encoding="utf-8"))
                    except ValueError:
                        existing_meta = {}

                title = existing_meta.get("title") or _humanize_slug(slug)
                page_type_str = existing_meta.get("type", "concept")
                try:
                    page_type = PageType(page_type_str.upper())
                except (KeyError, ValueError):
                    page_type = PageType.CONCEPT

                tags = existing_meta.get("tags", []) or []
                aliases = existing_meta.get("aliases", []) or []

                old_topic = store.state.topics.get(slug)
                confidence = old_topic.confidence if old_topic else 0.0
                created = existing_meta.get("created") or now_iso()[:10]
                updated = existing_meta.get("updated") or now_iso()[:10]

                fresh_state.topics[slug] = TopicState(
                    title=title,
                    type=page_type,
                    tags=list(tags),
                    aliases=list(aliases),
                    sources=[],
                    links_to=list(old_topic.links_to) if old_topic else [],
                    linked_from=[],
                    confidence=confidence,
                    created=created,
                    updated=updated,
                )

            topic = fresh_state.topics[slug]
            if source_id not in topic.sources:
                topic.sources.append(source_id)

    # Recompute linked_from from links_to.
    for slug, topic in fresh_state.topics.items():
        for target in topic.links_to:
            if target in fresh_state.topics and slug not in fresh_state.topics[target].linked_from:
                fresh_state.topics[target].linked_from.append(slug)

    # Build source registry from existing state (preserves sha256, raw, etc.).
    for path, meta, body, topics in valid_sources:
        source_id = path.stem
        old = store.state.sources.get(source_id)
        raw = meta.get("source") or (old.raw if old else "")
        sha256 = meta.get("sha256") or (old.sha256 if old else "")
        src_type = meta.get("type") or (old.type if old else "text")
        ingested = meta.get("ingested") or (old.ingested if old else now_iso())
        tokens = meta.get("tokens") or (old.tokens if old else 0)
        fresh_state.sources[source_id] = SourceState(
            sha256=sha256,
            topics=list(topics),
            raw=raw,
            embedding_model=old.embedding_model if old else None,
            stage=old.stage if old else "done",
            tokens=tokens,
            type=src_type,
            ingested=ingested,
        )

    return fresh_state, source_id_to_topics


def _write_wiki_from_state(
    cfg: Config,
    state: BrainState,
    valid_sources: list[tuple[Path, dict, str, list[str]]],
    source_chunks: dict[str, list[tuple[str, list[float]]]],
    *,
    synthesis_by_slug: dict[str, str] | None = None,
    open_questions_by_slug: dict[str, str] | None = None,
) -> None:
    """Write fresh ``90-wiki/*.md`` pages from the rebuilt state.

    Synthesis is preserved verbatim from the pre-rebuild wiki page. ``## Sources``
    is reassembled from each source's ``## Summary``. ``## Related`` is rebuilt
    from ``links_to``.

    When *synthesis_by_slug* and *open_questions_by_slug* are provided (as
    pre-captured dicts) they are used instead of reading the live wiki page,
    which may have been rmtree'd before the call.
    """
    # Build quick lookups.
    source_meta: dict[str, dict] = {}
    source_body: dict[str, str] = {}
    for path, meta, body, _ in valid_sources:
        source_id = path.stem
        source_meta[source_id] = meta
        source_body[source_id] = body

    for slug, topic in state.topics.items():
        if synthesis_by_slug is not None:
            synthesis = synthesis_by_slug.get(slug, "")
        else:
            old_wiki_path = cfg.brain_root / "90-wiki" / f"{slug}.md"
            old_body = ""
            if old_wiki_path.exists():
                try:
                    _, old_body = split_frontmatter(old_wiki_path.read_text(encoding="utf-8"))
                except ValueError:
                    old_body = ""
            synthesis = _extract_section(old_body, "## Synthesis")

        if open_questions_by_slug is not None:
            open_questions = open_questions_by_slug.get(slug, "")
        else:
            old_wiki_path = cfg.brain_root / "90-wiki" / f"{slug}.md"
            old_body = ""
            if old_wiki_path.exists():
                try:
                    _, old_body = split_frontmatter(old_wiki_path.read_text(encoding="utf-8"))
                except ValueError:
                    old_body = ""
            open_questions = _extract_section(old_body, "## Open questions")

        # Assemble Sources section.
        sources_lines: list[str] = []
        for source_id in sorted(topic.sources):
            meta = source_meta.get(source_id, {})
            body = source_body.get(source_id, "")
            ingested = meta.get("ingested", "")
            dt = ingested[:10] if ingested else ""
            title = _derive_title_from_body(body)
            raw = meta.get("source", "")
            summary = _extract_summary(body)
            raw_part = f" · [raw](../{raw})" if raw else ""
            sources_lines.append(
                f"- **[{dt}]** {title}\n"
                f"  -> [source](../50-sources/{source_id}.md)"
                f"{raw_part}\n"
                f"  > {summary}\n"
            )

        related_lines = [f"- [[{target}]]" for target in sorted(topic.links_to)]
        sources_block = "".join(sources_lines)
        related_block = "\n".join(related_lines) if related_lines else "- "

        page_body = (
            f"# {topic.title}\n\n"
            f"## Synthesis\n{synthesis}\n\n"
            f"## Sources\n{sources_block}\n\n"
            f"## Open questions\n{open_questions or '- '}\n\n"
            f"## Related\n{related_block}\n"
        )

        meta = {
            "title": topic.title,
            "slug": slug,
            "type": topic.type.value.lower(),
            "tags": topic.tags,
            "aliases": topic.aliases,
            "created": topic.created,
            "updated": topic.updated,
            "source_count": len(topic.sources),
            "confidence": round(topic.confidence, 4),
            "related": topic.links_to,
        }

        write_atomic(
            cfg.brain_root / "90-wiki" / f"{slug}.md",
            dump_frontmatter(meta, page_body),
        )


async def rebuild_from_sources(
    cfg: Config,
    client: OpenRouterClient,
    dry_run: bool = False,
) -> RebuildPlan:
    """Fast rebuild from existing ``50-sources/*.md`` files.

    Re-chunks and re-embeds source bodies, rebuilds topics, state, wiki pages,
    and the vector store. Preserves existing ``## Synthesis`` sections verbatim.
    """
    store = BrainStateStore.load(cfg)
    topics_before = len(store.state.topics)

    sources_dir = cfg.brain_root / "50-sources"
    source_paths = sorted(sources_dir.glob("*.md")) if sources_dir.exists() else []

    # Pass 1: classify sources (valid vs skipped).
    valid_sources: list[tuple[Path, dict, str, list[str]]] = []
    sources_skipped = 0
    for path in source_paths:
        info = _read_source_info(path)
        if info is None:
            sources_skipped += 1
            continue
        meta, body, source_id = info
        topics = meta.get("topics", []) or []

        old = store.state.sources.get(source_id)
        stage = old.stage if old else ""
        if not topics or stage == "failed":
            sources_skipped += 1
            continue

        valid_sources.append((path, meta, body, topics))

    sources_seen = len(source_paths)

    if dry_run:
        topics_after = len({slug for _, _, _, topics in valid_sources for slug in topics})
        return RebuildPlan(
            mode="from-sources",
            dry_run=True,
            sources_seen=sources_seen,
            sources_skipped=sources_skipped,
            topics_before=topics_before,
            topics_after=topics_after,
        )

    # Pass 2: embed valid source bodies.
    embedder = Embedder(client, cfg)
    await embedder.ensure_dim()
    source_chunks = await _embed_sources([p for p, _, _, _ in valid_sources], embedder)

    # Pass 3: build fresh state.
    fresh_state, source_id_to_topics = _build_fresh_state_from_sources(cfg, store, valid_sources)
    topics_after = len(fresh_state.topics)

    # Snapshot + rotate before any destructive write.
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    snapshot_dir = _snapshot_rebuild_state(cfg, ts)
    _rotate_pre_rebuild_snapshots(cfg, keep=3)

    # Pre-capture synthesis/open_questions from live wiki pages before
    # the destructive phase (SHOULD fix: stale-page removal via rmtree).
    synthesis_by_slug: dict[str, str] = {}
    open_questions_by_slug: dict[str, str] = {}
    for slug in fresh_state.topics:
        wiki_path = cfg.brain_root / "90-wiki" / f"{slug}.md"
        if wiki_path.exists():
            try:
                _, body = split_frontmatter(wiki_path.read_text(encoding="utf-8"))
            except ValueError:
                body = ""
            synthesis_by_slug[slug] = _extract_section(body, "## Synthesis")
            open_questions_by_slug[slug] = _extract_section(body, "## Open questions")
        else:
            synthesis_by_slug[slug] = ""
            open_questions_by_slug[slug] = ""

    rebuild_db = cfg.brain_root / ".brain" / "embeddings.rebuild.db"
    vec_store: VectorStore | None = None
    destructive_started = False

    try:
        # -- destructive phase begins ---------------------------------------
        destructive_started = True
        # Remove old wiki pages so dropped topics leave no orphans.
        wiki_dir = cfg.brain_root / "90-wiki"
        if wiki_dir.exists():
            shutil.rmtree(wiki_dir)
        wiki_dir.mkdir(parents=True, exist_ok=True)
        # Write fresh wiki pages from pre-captured synthesis.
        _write_wiki_from_state(
            cfg, fresh_state, valid_sources, source_chunks,
            synthesis_by_slug=synthesis_by_slug,
            open_questions_by_slug=open_questions_by_slug,
        )

        # Write fresh state.json.
        fresh_store = BrainStateStore(cfg)
        fresh_store.state = fresh_state
        fresh_store.save()
        store = fresh_store

        # Build fresh vector store.
        if rebuild_db.exists():
            rebuild_db.unlink()
        vec_store = VectorStore(rebuild_db, cfg.models.embedding, dim=embedder.dim)

        for source_id, chunks in source_chunks.items():
            topics = source_id_to_topics.get(source_id, [])
            if not topics or not chunks:
                continue
            # Assign chunks to the first topic (mirrors pipeline.py), but
            # register membership for every assigned topic.
            vec_store.upsert_source_chunks(source_id, topics[0], chunks)
            for slug in topics:
                vec_store.add_topic_member(slug, source_id)

        for slug in fresh_state.topics:
            vec_store.recompute_centroid(slug)

        vec_store.close()
        vec_store = None

        # Atomic swap embeddings.db.
        target_db = cfg.brain_root / ".brain" / "embeddings.db"
        os.replace(rebuild_db, target_db)

        # Regenerate INDEX.md.
        index = DebouncedIndex(cfg, store)
        await index.flush_now()

        # Append changelog entry.
        store.append_changelog(
            {
                "kind": "rebuild",
                "mode": "from-sources",
                "dry_run": False,
                "sources_seen": sources_seen,
                "sources_skipped": sources_skipped,
                "topics_before": topics_before,
                "topics_after": topics_after,
            }
        )

        return RebuildPlan(
            mode="from-sources",
            dry_run=False,
            sources_seen=sources_seen,
            sources_skipped=sources_skipped,
            topics_before=topics_before,
            topics_after=topics_after,
            snapshot_dir=snapshot_dir,
        )

    except Exception:
        if destructive_started:
            _restore_from_snapshot(cfg, snapshot_dir)
        raise
    finally:
        if vec_store is not None:
            vec_store.close()


# -- from-inbox rebuild ------------------------------------------------------


async def rebuild_from_inbox(
    cfg: Config,
    client: OpenRouterClient,
    dry_run: bool = False,
) -> RebuildPlan:
    """Deep rebuild from ``00-inbox/``.

    Archives derived state, removes ``50-sources/``, ``90-wiki/``,
    ``.brain/state.json``, and ``.brain/embeddings.db``, then replays the full
    ingestion pipeline for every inbox file sequentially.
    """
    inbox_dir = cfg.brain_root / "00-inbox"
    inbox_files = sorted(inbox_dir.iterdir()) if inbox_dir.exists() else []
    # Only consider regular files, ignore directories/symlinks for simplicity.
    inbox_files = [p for p in inbox_files if p.is_file()]

    sources_seen = len(inbox_files)
    sources_skipped = 0

    if dry_run:
        return RebuildPlan(
            mode="from-inbox",
            dry_run=True,
            sources_seen=sources_seen,
            sources_skipped=sources_skipped,
            topics_before=0,
            topics_after=0,
        )

    # Guard: empty inbox would delete everything for nothing.
    if sources_seen == 0:
        raise RuntimeError("00-inbox/ is empty; nothing to rebuild")

    # Snapshot + rotate before destructive work.
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    snapshot_dir = _snapshot_rebuild_state(cfg, ts)
    _rotate_pre_rebuild_snapshots(cfg, keep=3)

    store = BrainStateStore.load(cfg)
    topics_before = len(store.state.topics)

    rebuild_db = cfg.brain_root / ".brain" / "embeddings.rebuild.db"
    vec_store: VectorStore | None = None
    destructive_started = False

    try:
        # -- destructive phase begins ---------------------------------------
        destructive_started = True
        sources_dir = cfg.brain_root / "50-sources"
        wiki_dir = cfg.brain_root / "90-wiki"
        brain_dir = cfg.brain_root / ".brain"

        if sources_dir.exists():
            shutil.rmtree(sources_dir)
        sources_dir.mkdir(parents=True, exist_ok=True)
        if wiki_dir.exists():
            shutil.rmtree(wiki_dir)
        wiki_dir.mkdir(parents=True, exist_ok=True)
        for p in (brain_dir / "state.json", brain_dir / "embeddings.db"):
            if p.exists():
                p.unlink()

        # Fresh stores.
        store = BrainStateStore.load(cfg)  # empty because we deleted state.json
        embedder = Embedder(client, cfg)
        await embedder.ensure_dim()
        if rebuild_db.exists():
            rebuild_db.unlink()
        vec_store = VectorStore(rebuild_db, cfg.models.embedding, dim=embedder.dim)
        linker = EmbeddingLinker(embedder, vec_store, cfg.ingestion.merge_threshold)
        index = DebouncedIndex(cfg, store)

        for path in inbox_files:
            await ingest_file(
                path,
                cfg,
                store,
                client,
                linker,
                index,
                embedder=embedder,
                vec_store=vec_store,
            )

        await index.flush_now()

        # Atomic swap embeddings.db.
        target_db = brain_dir / "embeddings.db"
        vec_store.close()
        vec_store = None
        os.replace(rebuild_db, target_db)

        # Save final state.
        store.save()

        topics_after = len(store.state.topics)

        store.append_changelog(
            {
                "kind": "rebuild",
                "mode": "from-inbox",
                "dry_run": False,
                "sources_seen": sources_seen,
                "sources_skipped": sources_skipped,
                "topics_before": topics_before,
                "topics_after": topics_after,
            }
        )

        return RebuildPlan(
            mode="from-inbox",
            dry_run=False,
            sources_seen=sources_seen,
            sources_skipped=sources_skipped,
            topics_before=topics_before,
            topics_after=topics_after,
            snapshot_dir=snapshot_dir,
        )

    except Exception:
        if destructive_started:
            _restore_from_snapshot(cfg, snapshot_dir)
        raise
    finally:
        if vec_store is not None:
            vec_store.close()
