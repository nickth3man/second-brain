"""Debounced INDEX.md writer (§4.3, §8).

The :class:`DebouncedIndex` regenerates ``INDEX.md`` (the brain's front door)
with a configurable debounce delay.  Each call to :meth:`mark_dirty` resets
the timer, so rapid ingestion batches produce a single flush 30 s after the
last file settles.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from second_brain.atomicio import write_atomic
from second_brain.config import Config
from second_brain.state import BrainStateStore


class DebouncedIndex:
    """Regenerate ``INDEX.md`` with a debounce delay after the last dirty mark."""

    def __init__(
        self,
        cfg: Config,
        store: BrainStateStore,
        delay: float = 30.0,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.delay = delay
        self._timer_handle: asyncio.TimerHandle | None = None
        self._loop = asyncio.get_event_loop()

    def mark_dirty(self) -> None:
        """(Re)schedule a flush *delay* seconds from now.

        Safe to call multiple times — each call cancels the previous timer
        so rapid batches produce a single flush.
        """
        if self._timer_handle is not None:
            self._timer_handle.cancel()
        self._timer_handle = self._loop.call_later(self.delay, self._flush_sync)

    def _flush_sync(self) -> None:
        """Bridge from ``call_later`` (sync) to the async flush."""
        asyncio.ensure_future(self._flush())

    async def _flush(self) -> None:
        state = self.store.state
        n_sources = len(state.sources)
        n_topics = len(state.topics)
        now_str = datetime.now(UTC).strftime("%Y-%m-%d")

        lines: list[str] = [
            "# Second Brain\n",
            f"**{n_sources} sources · {n_topics} topics · last updated {now_str}**\n",
        ]

        # ── Recent sources (up to 10, newest first) ──────────────────
        if state.sources:
            lines.append("\n## Recent\n")
            sorted_sources = sorted(
                state.sources.values(),
                key=lambda s: s.ingested,
                reverse=True,
            )[:10]
            for src in sorted_sources:
                topic_info = ""
                if src.topics:
                    topic_info = f" [→ {src.topics[0]}]"
                lines.append(
                    f"- {_date(src.ingested)} → {src.raw}{topic_info}\n"
                )

        # ── Topics (alphabetical) ─────────────────────────────────────
        if state.topics:
            lines.append("\n## Topics\n")
            sorted_items = sorted(
                state.topics.items(),
                key=lambda item: item[1].title.lower(),
            )
            for slug, topic in sorted_items:
                lines.append(
                    f"- [{topic.title}](90-wiki/{slug}.md)"
                    f" — {len(topic.sources)} sources\n"
                )

        content = "".join(lines)
        write_atomic(self.cfg.brain_root / "INDEX.md", content)

    async def flush_now(self) -> None:
        """Immediate flush, cancelling any pending timer.

        Used by tests and during daemon shutdown.
        """
        if self._timer_handle is not None:
            self._timer_handle.cancel()
            self._timer_handle = None
        await self._flush()


def _date(iso: str) -> str:
    """Extract ``YYYY-MM-DD`` from an ISO 8601 string."""
    return iso[:10]
