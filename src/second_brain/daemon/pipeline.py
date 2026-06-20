"""Per-file ingestion pipeline (§5 stages 1–6) and daemon runner (§12.3).

Pipeline stages per file **executed synchronously** (no per-file concurrency):

1. Hash & dedup      — sha256 skip if already seen
2. Normalise         — write ``50-sources/<id>.md`` with front-matter
3. Extract           — LLM -> structured :class:`LibrarianOutput`
4. Link              — slug-match or embedding-match -> merge or spawn
5. Wiki update       — write/merge ``90-wiki/<slug>.md``
6. Index update      — mark INDEX.md dirty (debounced flush)

Any exception (except :class:`CreditExhaustedError`) transitions the source
to ``FAILED``, copies the raw file to the deadletter directory, and is
swallowed so one bad file does not crash the daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import shutil
import time
from pathlib import Path

import structlog

from second_brain.config import Config
from second_brain.daemon.extract import ExtractionError, extract
from second_brain.daemon.index import DebouncedIndex
from second_brain.daemon.linker import EmbeddingLinker, LinkContext, Linker
from second_brain.daemon.normalize import (
    normalize_text,
    sha256_of_file,
    source_id_for,
)
from second_brain.daemon.router import route
from second_brain.daemon.watcher import InboxWatcher
from second_brain.daemon.wiki import merge_into_topic, write_new_topic
from second_brain.models import IngestStage, PageType, SourceState, TopicAction
from second_brain.openrouter_client import (
    CreditExhaustedError,
    OpenRouterClient,
)
from second_brain.slug import slugify
from second_brain.state import BrainStateStore, now_iso
from second_brain.vectors.embed import Embedder
from second_brain.vectors.store import VectorStore

log = structlog.get_logger(__name__)

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")

PER_FILE_TIMEOUT_S = 600


# -- helpers ------------------------------------------------------------------


def _fail_file_safe(
    path: Path,
    cfg: Config,
    msg: str,
    log_exc: bool = True,
    store: BrainStateStore | None = None,
    source_id: str | None = None,
    sha: str | None = None,
    exc: Exception | None = None,
) -> None:
    """Transition a file to FAILED, deadletter-copy, log, and save state.

    This helper MUST never raise.  All operations are wrapped in try/except.
    """
    try:
        if source_id and store:
            # Ensure the source is registered before transitioning.
            if source_id not in store.state.sources:
                try:
                    rel = path.resolve().relative_to(cfg.brain_root.resolve()).as_posix()
                except Exception:
                    rel = str(path)
                store.record_source(
                    source_id,
                    SourceState(
                        sha256=sha or "",
                        raw=rel,
                        type="unknown",
                        stage=IngestStage.FAILED,
                        error=msg,
                    ),
                )
            store.transition(source_id, IngestStage.FAILED, error=msg)
    except Exception:
        pass

    with contextlib.suppress(Exception):
        _deadletter(path, cfg)

    with contextlib.suppress(Exception):
        log.error(
            "pipeline.file.failed",
            source_id=source_id,
            sha=sha,
            stage="failed",
            error=msg,
            error_type=type(exc).__name__ if exc else "Unknown",
        )

    try:
        if store is not None:
            store.save()
    except Exception:
        log.error("state.save_failed")


# -- per-file pipeline --------------------------------------------------------


async def ingest_file(
    path: Path,
    cfg: Config,
    store: BrainStateStore,
    client: OpenRouterClient,
    linker: Linker,
    index: DebouncedIndex,
    *,
    embedder: Embedder | None = None,
    vec_store: VectorStore | None = None,
) -> IngestStage:
    """Run the full ingestion pipeline (stages 1–6) on a single file.

    Args:
        embedder: Optional embedding client for Phase 2 chunk embedding.
        vec_store: Optional vector store for Phase 2 chunk + centroid writes.

    Returns:
        The final pipeline stage (``DONE`` or ``FAILED``).
    """
    # -- Stage 1: Hash & dedup -----------------------------------------
    sha = sha256_of_file(path)

    # Exact-sha256 dedup (§11)
    if any(s.sha256 == sha for s in store.state.sources.values()):
        store.append_changelog(
            {"kind": "ingest", "action": "dedup_skip", "sha": sha}
        )
        return IngestStage.DONE

    ingested = now_iso()
    # Compute source_id here so the try block below has it.
    # We read a small preview from the file for source_id_for.
    try:
        preview = path.read_text(encoding="utf-8", errors="replace")[:2000]
    except Exception:
        preview = ""
    source_id = source_id_for(path, preview, ingested)

    t_total = time.perf_counter()

    try:
        stage = route(path.suffix, cfg)

        try:
            rel = path.resolve().relative_to(cfg.brain_root.resolve()).as_posix()
        except ValueError:
            rel = str(path)

        store.record_source(
            source_id,
            SourceState(
                sha256=sha,
                raw=rel,
                type=stage,
                ingested=ingested,
                stage=IngestStage.HASHING,
            ),
        )
        store.transition(source_id, IngestStage.NORMALIZED)
        log.info("pipeline.stage.start", source_id=source_id, sha=sha, stage="HASHING")

        # -- Stage 2: Normalise ---------------------------------------
        t_stage = time.perf_counter()
        try:
            _, body = await normalize_text(
                path, source_id, sha, ingested, stage, cfg, client
            )
        except ValueError as e:
            _fail_file_safe(
                path, cfg, msg=str(e), store=store,
                source_id=source_id, sha=sha, exc=e,
            )
            return IngestStage.FAILED
        log.info(
            "pipeline.stage.end",
            source_id=source_id, sha=sha, stage="NORMALIZED",
            latency_ms=round((time.perf_counter() - t_stage) * 1000, 1),
        )

        # Check for partial transcription sentinel.
        partial_sentinel = "<!-- sb:partial"
        if partial_sentinel in body:
            store.state.sources[source_id].partial = True
            # Strip the sentinel from body before extract/embed.
            idx = body.index(partial_sentinel)
            close_idx = body.find("-->", idx)
            if close_idx != -1:
                sentinel = body[idx : close_idx + 3]
                body = body.replace(sentinel, "").rstrip()
                # Re-read and strip the sentinel from the saved source file.
                src_path = cfg.brain_root / "50-sources" / f"{source_id}.md"
                if src_path.exists():
                    src_text = src_path.read_text(encoding="utf-8")
                    src_text = src_text.replace(sentinel, "").rstrip()
                    src_path.write_text(src_text, encoding="utf-8")
            log.info(
                "pipeline.partial",
                source_id=source_id, sha=sha,
            )

        # -- Stage 3: Extract -----------------------------------------
        t_stage = time.perf_counter()
        store.transition(source_id, IngestStage.EXTRACTED)
        log.info("pipeline.stage.start", source_id=source_id, sha=sha, stage="EXTRACTED")
        try:
            out = await extract(
                client, cfg, body, store.all_topic_titles(),
                source_type=stage,
            )
        except CreditExhaustedError:
            raise
        except (ExtractionError, Exception) as e:
            _fail_file_safe(
                path, cfg, msg=str(e), store=store,
                source_id=source_id, sha=sha, exc=e,
            )
            return IngestStage.FAILED
        log.info(
            "pipeline.stage.end",
            source_id=source_id, sha=sha, stage="EXTRACTED",
            latency_ms=round((time.perf_counter() - t_stage) * 1000, 1),
        )

        # -- Phase 2 embedding step (before linking) ------------------
        source_chunks: list[tuple[str, list[float]]] = []
        if embedder is not None and vec_store is not None:
            from second_brain.vectors.store import chunk_text

            t_stage = time.perf_counter()
            log.info(
                "pipeline.stage.start",
                source_id=source_id, sha=sha, stage="EMBEDDING",
            )
            ctexts = chunk_text(body)
            cvecs = await embedder.embed_texts(ctexts)
            source_chunks = list(zip(ctexts, cvecs, strict=False))
            log.info(
                "pipeline.stage.end",
                source_id=source_id, sha=sha, stage="EMBEDDING",
                n_chunks=len(source_chunks),
                latency_ms=round((time.perf_counter() - t_stage) * 1000, 1),
            )

        # -- Stage 4: Link --------------------------------------------
        t_stage = time.perf_counter()
        ctx = LinkContext(
            brain_store=store,
            embedder=embedder,
            vec_store=vec_store,
            source_id=source_id,
            source_chunks=source_chunks,
        )
        decisions = await linker.link(out.topics, ctx)
        store.transition(source_id, IngestStage.LINKED)
        log.info(
            "pipeline.stage.end",
            source_id=source_id, sha=sha, stage="LINKED",
            latency_ms=round((time.perf_counter() - t_stage) * 1000, 1),
        )

        # Upsert the source's chunks before the per-decision loop
        if vec_store is not None and source_chunks and decisions:
            vec_store.upsert_source_chunks(
                source_id, decisions[0].target_slug, source_chunks
            )

        # -- Stage 5: Wiki update -------------------------------------
        t_stage = time.perf_counter()
        store.transition(source_id, IngestStage.WIKI_MERGED)
        log.info(
            "pipeline.stage.start",
            source_id=source_id, sha=sha, stage="WIKI_MERGED",
        )
        for decision in decisions:
            store.ensure_topic(
                decision.target_slug,
                decision.name,
                PageType.CONCEPT,
                decision.confidence,
            )
            store.add_source_to_topic(decision.target_slug, source_id)
            if vec_store is not None:
                vec_store.add_topic_member(decision.target_slug, source_id)

            if decision.action == TopicAction.NEW:
                write_new_topic(
                    cfg,
                    store,
                    decision.target_slug,
                    decision.name,
                    decision,
                    source_id,
                    ingested,
                )
            else:
                merge_into_topic(
                    cfg,
                    store,
                    decision.target_slug,
                    decision,
                    source_id,
                    ingested,
                    out.tldr,
                )

            # Detect [[wikilinks]] in merged_section -> record graph edges
            for m in _WIKILINK_RE.findall(decision.merged_section):
                linked_slug = slugify(m.strip())
                if linked_slug in store.state.topics:
                    store.record_link(decision.target_slug, linked_slug)

        # Recompute centroids for all affected topics
        if vec_store is not None:
            for d in decisions:
                vec_store.recompute_centroid(d.target_slug)

        log.info(
            "pipeline.stage.end",
            source_id=source_id, sha=sha, stage="WIKI_MERGED",
            latency_ms=round((time.perf_counter() - t_stage) * 1000, 1),
        )

        # -- Stage 6: Index -------------------------------------------
        index.mark_dirty()
        store.transition(source_id, IngestStage.INDEXED)
        store.transition(source_id, IngestStage.DONE)
        store.save()
        store.append_changelog(
            {
                "kind": "ingest",
                "action": "done",
                "source": source_id,
                "topics": [d.target_slug for d in decisions],
            }
        )

        total_latency_ms = round((time.perf_counter() - t_total) * 1000, 1)
        log.info(
            "pipeline.file.done",
            source_id=source_id,
            sha=sha,
            topics_count=len(decisions),
            total_latency_ms=total_latency_ms,
        )

        return IngestStage.DONE

    except CreditExhaustedError:
        raise  # Stop the daemon (§12.3)

    except Exception as e:
        # Catch-all: one bad file must not crash the daemon
        _fail_file_safe(
            path, cfg, msg=str(e), store=store,
            source_id=source_id, sha=sha, exc=e,
        )
        return IngestStage.FAILED


def _deadletter(path: Path, cfg: Config) -> None:
    """Copy the raw file to the deadletter directory."""
    dead_dir = Path(cfg.extraction.deadletter_dir)
    if not dead_dir.is_absolute():
        dead_dir = cfg.brain_root / dead_dir
    dead_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dead_dir / path.name)


# -- daemon runner ------------------------------------------------------------


async def run_daemon(cfg: Config) -> None:
    """Start the file-watcher daemon.

    Builds all dependencies, starts ``watchdog``, and processes files from
    the async queue one at a time.  Stops on ``KeyboardInterrupt`` or
    ``CreditExhaustedError``.
    """
    # Validate key early so the daemon fails fast
    _ = OpenRouterClient(cfg)  # resolves API key on first access

    store = BrainStateStore.load(cfg)
    index = DebouncedIndex(cfg, store)
    queue: asyncio.Queue[Path] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    watcher = InboxWatcher(cfg.brain_root / "00-inbox", loop, queue)
    observer = watcher.start()

    log.info("daemon.start", inbox=str(cfg.brain_root / "00-inbox"))

    client: OpenRouterClient | None = None
    vec_store: VectorStore | None = None
    try:
        client = OpenRouterClient(cfg)
        embedder = Embedder(client, cfg)
        dim = await embedder.ensure_dim()
        vec_store = VectorStore(
            cfg.brain_root / ".brain/embeddings.db",
            cfg.models.embedding,
            dim=dim,
        )
        linker: Linker = EmbeddingLinker(
            embedder, vec_store, cfg.ingestion.merge_threshold
        )

        while True:
            path = await queue.get()
            log.info("daemon.ingest", path=str(path))
            try:
                stage = await asyncio.wait_for(
                    ingest_file(
                        path, cfg, store, client, linker, index,
                        embedder=embedder, vec_store=vec_store,
                    ),
                    timeout=PER_FILE_TIMEOUT_S,
                )
                log.info("daemon.done", path=str(path), stage=str(stage))
            except CreditExhaustedError:
                log.error("daemon.credit_exhausted")
                raise
            except TimeoutError:
                msg = f"file processing exceeded {PER_FILE_TIMEOUT_S}s timeout"
                log.error("daemon.timeout", path=str(path))
                _fail_file_safe(path, cfg, msg=msg, store=store)
            except Exception as e:
                log.error("daemon.ingest_failed", path=str(path), error=str(e))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("daemon.shutdown")
    finally:
        watcher.stop(observer)
        await index.flush_now()
        if vec_store is not None:
            vec_store.close()
        if client is not None:
            await client.close()
