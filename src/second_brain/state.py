"""Brain state persistence — load (backup recovery), save (atomic+backup),
state machine transitions, and append-only changelog.

See §12.3 (state machine + atomic writes + recovery) and §12.6
(backups/migration) in ARCHITECTURE.md.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from pydantic import ValidationError

from second_brain.atomicio import rolling_backup, write_json_atomic
from second_brain.frontmatter import split_frontmatter
from second_brain.models import BrainState, IngestStage, PageType, SourceState, TopicState

log = structlog.get_logger(__name__)

# -- constants ----------------------------------------------------------------

STATE_FILENAME = ".brain/state.json"
CHANGELOG_FILENAME = ".brain/changelog.jsonl"
SCHEMA_VERSION = 1


# -- helpers ------------------------------------------------------------------


def now_iso() -> str:
    """Return UTC now in ISO 8601 with ``Z`` suffix (e.g. ``"2026-06-19T14:23:01Z"``)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_wikilinks(body: str) -> list[str]:
    """Return slug-like targets from simple [[wikilinks]] in markdown body."""
    import re

    targets: list[str] = []
    for raw in re.findall(r"\[\[([^\]]+)\]\]", body):
        page = raw.split("|", 1)[0].split("#", 1)[0].strip()
        slug = page.lower().replace("_", "-").replace(" ", "-")
        slug = re.sub(r"[^a-z0-9-]+", "", slug).strip("-")
        if slug:
            targets.append(slug)
    return targets


# -- BrainStateStore ----------------------------------------------------------


class BrainStateStore:
    """Manages `.brain/state.json` with atomic writes, rolling backups, and
    an append-only changelog.

    **Thread-safety:** not guaranteed — the daemon owns all writes.  The web UI
    should open the file read-only.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.path: Path = cfg.brain_root / STATE_FILENAME
        self.changelog_path: Path = cfg.brain_root / CHANGELOG_FILENAME
        self.state = BrainState()

    # -- load / save ------------------------------------------------------

    @classmethod
    def load(cls, cfg: Any) -> BrainStateStore:
        """Create a store and attempt to load state from disk.

        Recovery chain (§12.6): primary -> ``.bak`` -> ``.bak-1`` -> ``.bak-2``
        -> fresh (empty) state.  Logs which fallback was used via structlog.

        The parent directory of the state file is created if it does not exist.
        """
        self = cls(cfg)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        candidates: list[Path] = [
            self.path,
            self.path.parent / f"{self.path.name}.bak",
            self.path.parent / f"{self.path.name}.bak-1",
            self.path.parent / f"{self.path.name}.bak-2",
        ]

        loaded = False
        for candidate in candidates:
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                self.state = BrainState.model_validate(data)
                loaded = True
                if candidate != self.path:
                    log.info("state_recovered_from_backup", fallback=candidate.name)
                break
            except (OSError, json.JSONDecodeError, ValidationError):
                continue

        if not loaded:
            log.warning("state_starting_fresh", path=str(self.path))

        return self

    def save(self) -> None:
        """Atomically serialise state as JSON, then mirror to a rolling backup.

        Ordered **write-then-backup** (§12.6): ``write_json_atomic`` already
        makes the write all-or-nothing (temp + fsync + os.replace), so the
        backup's job is post-write redundancy — it holds the last *good*
        written state, recoverable if the primary is later corrupted on disk.
        Backup-first would leave no ``.bak`` after the very first save.
        """
        write_json_atomic(self.path, self.state.model_dump(mode="json"))
        rolling_backup(self.path, depth=3)

    # -- source registry -------------------------------------------------

    def has_source(self, sha256: str) -> bool:
        """Return ``True`` if a source with the given content hash exists."""
        return sha256 in self.state.sources

    def get_source(self, source_id: str) -> SourceState | None:
        """Look up a source by its id (filename stem), or return ``None``."""
        return self.state.sources.get(source_id)

    def record_source(self, source_id: str, st: SourceState) -> None:
        """Register (or overwrite) a source entry.  Caller must ``save()``."""
        self.state.sources[source_id] = st

    # -- state machine (§12.3) -------------------------------------------

    def transition(
        self,
        source_id: str,
        stage: IngestStage,
        error: str | None = None,
    ) -> None:
        """Move a source to the next pipeline stage.

        If *stage* is ``FAILED``, the *error* message is recorded.
        Idempotent — safe to call repeatedly.
        """
        src = self.state.sources.get(source_id)
        if src is None:
            return
        src.stage = stage
        if stage == IngestStage.FAILED:
            src.error = error
        self.state.updated = now_iso()

    # -- topic management ------------------------------------------------

    def ensure_topic(
        self,
        slug: str,
        title: str,
        page_type: PageType = PageType.CONCEPT,
        confidence: float = 0.0,
    ) -> TopicState:
        """Return the topic for *slug*, creating it if absent.

        **Idempotent:** if the topic already exists it is returned unchanged
        (the caller still needs to ``save()`` for other changes).
        """
        existing = self.state.topics.get(slug)
        if existing is not None:
            return existing

        now = now_iso()
        topic = TopicState(
            title=title,
            type=page_type,
            confidence=confidence,
            created=now,
            updated=now,
        )
        self.state.topics[slug] = topic
        return topic

    def add_source_to_topic(self, slug: str, source_id: str) -> bool:
        """Append *source_id* to the topic's source list (if absent).

        Returns:
            ``True`` if the list was modified, ``False`` if already present
            or the topic does not exist.
        """
        topic = self.state.topics.get(slug)
        if topic is None:
            return False
        if source_id in topic.sources:
            return False
        topic.sources.append(source_id)
        topic.updated = now_iso()
        self.state.updated = now_iso()
        return True

    def record_link(self, from_slug: str, to_slug: str) -> None:
        """Record a bidirectional topic link.

        Adds *to_slug* to *from_slug*'s ``links_to`` and *from_slug* to
        *to_slug*'s ``linked_from`` (both deduped).  Silently ignored if
        either topic slug does not exist.
        """
        from_topic = self.state.topics.get(from_slug)
        to_topic = self.state.topics.get(to_slug)
        if from_topic is None or to_topic is None:
            return

        changed = False
        if to_slug not in from_topic.links_to:
            from_topic.links_to.append(to_slug)
            changed = True
        if from_slug not in to_topic.linked_from:
            to_topic.linked_from.append(from_slug)
            changed = True

        if changed:
            from_topic.updated = now_iso()
            to_topic.updated = now_iso()
            self.state.updated = now_iso()

    def all_topic_titles(self) -> dict[str, str]:
        """Return ``{slug: title}`` for the extract prompt context."""
        return {slug: t.title for slug, t in self.state.topics.items()}

    # -- changelog -------------------------------------------------------

    def append_changelog(self, entry: dict) -> None:
        """Append a JSON line to the changelog.

        Adds ``ts`` (ISO 8601 with ``Z``) if not present in *entry*.  The
        file is opened, written, flushed, and fsynced on every call.
        """
        if "ts" not in entry:
            entry["ts"] = now_iso()
        self.changelog_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with open(self.changelog_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def reconcile_filesystem(cfg: Any, store: BrainStateStore) -> bool:
    """Reconcile ``state.json`` with derived source/wiki files (§12.3).

    The pass is intentionally conservative: it restores missing state entries
    from valid front-matter in ``50-sources`` and ``90-wiki``, and removes
    state entries whose derived files no longer exist. Raw inbox files are
    never changed.

    Returns:
        ``True`` when state changed and was saved.
    """
    changed = False
    root = cfg.brain_root

    source_dir = root / "50-sources"
    source_files = {
        path.stem: path
        for path in source_dir.glob("*.md")
        if path.is_file() and path.name != ".gitkeep"
    } if source_dir.exists() else {}

    for source_id in list(store.state.sources):
        if source_id not in source_files:
            del store.state.sources[source_id]
            changed = True

    for source_id, path in source_files.items():
        if source_id in store.state.sources:
            continue
        try:
            meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        store.state.sources[source_id] = SourceState(
            sha256=str(meta.get("sha256", "")),
            topics=list(meta.get("topics", []) or []),
            raw=str(meta.get("source", "")),
            embedding_model=meta.get("embedding_model"),
            stage=IngestStage.DONE,
            tokens=int(meta.get("tokens", 0) or 0),
            type=str(meta.get("type", "text")),
            ingested=str(meta.get("ingested", "")),
        )
        changed = True

    wiki_dir = root / "90-wiki"
    wiki_files = {
        path.stem: path
        for path in wiki_dir.glob("*.md")
        if path.is_file() and path.name != ".gitkeep"
    } if wiki_dir.exists() else {}

    for slug in list(store.state.topics):
        if slug not in wiki_files:
            del store.state.topics[slug]
            changed = True

    for slug, path in wiki_files.items():
        if slug in store.state.topics:
            continue
        try:
            meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        raw_type = str(meta.get("type", "concept")).lower()
        try:
            page_type = PageType(raw_type)
        except ValueError:
            page_type = PageType.CONCEPT
        store.state.topics[slug] = TopicState(
            title=str(meta.get("title", slug)),
            type=page_type,
            tags=list(meta.get("tags", []) or []),
            aliases=list(meta.get("aliases", []) or []),
            sources=[
                source_id
                for source_id, src in store.state.sources.items()
                if slug in src.topics
            ],
            links_to=sorted(set(list(meta.get("related", []) or []) + _parse_wikilinks(body))),
            linked_from=[],
            confidence=float(meta.get("confidence", 0.0) or 0.0),
            created=str(meta.get("created", "")),
            updated=str(meta.get("updated", "")),
        )
        changed = True

    for _slug, topic in store.state.topics.items():
        topic.sources = [sid for sid in topic.sources if sid in store.state.sources]
        refreshed_links = [target for target in topic.links_to if target in store.state.topics]
        if refreshed_links != topic.links_to:
            topic.links_to = refreshed_links
            changed = True
        topic.linked_from = []

    for source_id, src in store.state.sources.items():
        src.topics = [slug for slug in src.topics if slug in store.state.topics]
        for slug in src.topics:
            topic = store.state.topics.get(slug)
            if topic is not None and source_id not in topic.sources:
                topic.sources.append(source_id)
                changed = True

    for slug, topic in store.state.topics.items():
        for target in topic.links_to:
            linked = store.state.topics.get(target)
            if linked is not None and slug not in linked.linked_from:
                linked.linked_from.append(slug)
                changed = True

    if changed:
        store.state.updated = now_iso()
        store.save()
        log.info(
            "state_reconciled",
            sources=len(store.state.sources),
            topics=len(store.state.topics),
        )
    return changed
