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
from second_brain.models import BrainState, IngestStage, PageType, SourceState, TopicState

log = structlog.get_logger(__name__)

# ── constants ────────────────────────────────────────────────────────────────

STATE_FILENAME = ".brain/state.json"
CHANGELOG_FILENAME = ".brain/changelog.jsonl"
SCHEMA_VERSION = 1


# ── helpers ──────────────────────────────────────────────────────────────────


def now_iso() -> str:
    """Return UTC now in ISO 8601 with ``Z`` suffix (e.g. ``"2026-06-19T14:23:01Z"``)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── BrainStateStore ──────────────────────────────────────────────────────────


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

    # ── load / save ──────────────────────────────────────────────────────

    @classmethod
    def load(cls, cfg: Any) -> BrainStateStore:
        """Create a store and attempt to load state from disk.

        Recovery chain (§12.6): primary → ``.bak`` → ``.bak-1`` → ``.bak-2``
        → fresh (empty) state.  Logs which fallback was used via structlog.

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

    # ── source registry ─────────────────────────────────────────────────

    def has_source(self, sha256: str) -> bool:
        """Return ``True`` if a source with the given content hash exists."""
        return sha256 in self.state.sources

    def get_source(self, source_id: str) -> SourceState | None:
        """Look up a source by its id (filename stem), or return ``None``."""
        return self.state.sources.get(source_id)

    def record_source(self, source_id: str, st: SourceState) -> None:
        """Register (or overwrite) a source entry.  Caller must ``save()``."""
        self.state.sources[source_id] = st

    # ── state machine (§12.3) ───────────────────────────────────────────

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

    # ── topic management ────────────────────────────────────────────────

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

    # ── changelog ───────────────────────────────────────────────────────

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
