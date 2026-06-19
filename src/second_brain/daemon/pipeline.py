"""Per-file ingestion pipeline (§5 stages 1–6) and daemon runner (§12.3).

Pipeline stages per file **executed synchronously** (no per-file concurrency):

1. Hash & dedup      — sha256 skip if already seen
2. Normalise         — write ``50-sources/<id>.md`` with front-matter
3. Extract           — LLM → structured :class:`LibrarianOutput`
4. Link              — slug-match → merge or spawn
5. Wiki update       — write/merge ``90-wiki/<slug>.md``
6. Index update      — mark INDEX.md dirty (debounced flush)

Any exception (except :class:`CreditExhaustedError`) transitions the source
to ``FAILED``, copies the raw file to the deadletter directory, and is
swallowed so one bad file does not crash the daemon.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path

import structlog

from second_brain.config import Config
from second_brain.daemon.extract import ExtractionError, extract
from second_brain.daemon.index import DebouncedIndex
from second_brain.daemon.linker import Linker, SlugLinker
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

log = structlog.get_logger(__name__)

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)")


# ── per-file pipeline ────────────────────────────────────────────────────────


async def ingest_file(
    path: Path,
    cfg: Config,
    store: BrainStateStore,
    client: OpenRouterClient,
    linker: Linker,
    index: DebouncedIndex,
) -> IngestStage:
    """Run the full ingestion pipeline (stages 1–6) on a single file.

    Returns:
        The final pipeline stage (``DONE`` or ``FAILED``).
    """
    # ── Stage 1: Hash & dedup ─────────────────────────────────────────
    body = path.read_text(encoding="utf-8", errors="replace")
    sha = sha256_of_file(path)

    # Exact-sha256 dedup (§11)
    if any(s.sha256 == sha for s in store.state.sources.values()):
        store.append_changelog(
            {"kind": "ingest", "action": "dedup_skip", "sha": sha}
        )
        return IngestStage.DONE

    ingested = now_iso()
    source_id = source_id_for(path, body, ingested)

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

        # ── Stage 2: Normalise ───────────────────────────────────────
        try:
            await normalize_text(
                path, source_id, sha, ingested, stage, cfg
            )
        except ValueError as e:
            store.transition(source_id, IngestStage.FAILED, error=str(e))
            _deadletter(path, cfg)
            store.save()
            store.append_changelog(
                {
                    "kind": "ingest",
                    "action": "failed",
                    "source": source_id,
                    "error": str(e),
                }
            )
            return IngestStage.FAILED

        # ── Stage 3: Extract ─────────────────────────────────────────
        store.transition(source_id, IngestStage.EXTRACTED)
        try:
            out = await extract(client, cfg, body, store.all_topic_titles())
        except CreditExhaustedError:
            raise
        except (ExtractionError, Exception) as e:
            store.transition(source_id, IngestStage.FAILED, error=str(e))
            _deadletter(path, cfg)
            store.save()
            store.append_changelog(
                {
                    "kind": "ingest",
                    "action": "failed",
                    "source": source_id,
                    "error": str(e),
                }
            )
            return IngestStage.FAILED

        # ── Stage 4: Link ────────────────────────────────────────────
        store.transition(source_id, IngestStage.LINKED)
        decisions = linker.link(out.topics, store)

        # ── Stage 5: Wiki update ─────────────────────────────────────
        store.transition(source_id, IngestStage.WIKI_MERGED)
        for decision in decisions:
            store.ensure_topic(
                decision.target_slug,
                decision.name,
                PageType.CONCEPT,
                decision.confidence,
            )
            store.add_source_to_topic(decision.target_slug, source_id)

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

            # Detect [[wikilinks]] in merged_section → record graph edges
            for m in _WIKILINK_RE.findall(decision.merged_section):
                linked_slug = slugify(m.strip())
                if linked_slug in store.state.topics:
                    store.record_link(decision.target_slug, linked_slug)

        # ── Stage 6: Index ───────────────────────────────────────────
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

        return IngestStage.DONE

    except CreditExhaustedError:
        raise  # Stop the daemon (§12.3)

    except Exception as e:
        # Catch-all: one bad file must not crash the daemon
        store.transition(source_id, IngestStage.FAILED, error=str(e))
        _deadletter(path, cfg)
        store.save()
        store.append_changelog(
            {
                "kind": "ingest",
                "action": "failed",
                "source": source_id,
                "error": str(e),
            }
        )
        return IngestStage.FAILED


def _deadletter(path: Path, cfg: Config) -> None:
    """Copy the raw file to the deadletter directory."""
    dead_dir = Path(cfg.extraction.deadletter_dir)
    if not dead_dir.is_absolute():
        dead_dir = cfg.brain_root / dead_dir
    dead_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dead_dir / path.name)


# ── daemon runner ────────────────────────────────────────────────────────────


async def run_daemon(cfg: Config) -> None:
    """Start the file-watcher daemon.

    Builds all dependencies, starts ``watchdog``, and processes files from
    the async queue one at a time.  Stops on ``KeyboardInterrupt`` or
    ``CreditExhaustedError``.
    """
    # Validate key early so the daemon fails fast
    _ = OpenRouterClient(cfg)  # resolves API key on first access

    store = BrainStateStore.load(cfg)
    linker: Linker = SlugLinker()
    index = DebouncedIndex(cfg, store)
    queue: asyncio.Queue[Path] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    watcher = InboxWatcher(cfg.brain_root / "00-inbox", loop, queue)
    observer = watcher.start()

    log.info("daemon.start", inbox=str(cfg.brain_root / "00-inbox"))

    client: OpenRouterClient | None = None
    try:
        client = OpenRouterClient(cfg)

        while True:
            path = await queue.get()
            log.info("daemon.ingest", path=str(path))
            try:
                stage = await ingest_file(
                    path, cfg, store, client, linker, index
                )
                log.info("daemon.done", path=str(path), stage=str(stage))
            except CreditExhaustedError:
                log.error("daemon.credit_exhausted")
                raise
            except Exception as e:
                log.error("daemon.ingest_failed", path=str(path), error=str(e))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("daemon.shutdown")
    finally:
        watcher.stop(observer)
        await index.flush_now()
        if client is not None:
            await client.close()
