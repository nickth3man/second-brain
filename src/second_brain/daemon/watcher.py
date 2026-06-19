"""Watchdog-based file watcher for ``00-inbox/`` with stable-file gate (§12.3).

The :class:`InboxWatcher` monitors the inbox directory via ``watchdog``,
ignores temp-file patterns, polls for file stability (size + mtime unchanged
across two polls), and emits stable file paths onto an :class:`asyncio.Queue`
for the pipeline.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from second_brain.daemon.router import is_temp_file


class InboxWatcher(FileSystemEventHandler):
    """Watchdog handler that emits stable file paths onto an async queue.

    Args:
        inbox: Directory to watch (typically ``cfg.brain_root / "00-inbox"``).
        loop: The running asyncio event loop.
        queue: Queue on which stable file paths are pushed.
        settle_seconds: Time (s) between stability polls.
    """

    def __init__(
        self,
        inbox: Path,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[Path],
        settle_seconds: float = 0.5,
    ) -> None:
        super().__init__()
        self.inbox = inbox.resolve()
        self.loop = loop
        self.queue = queue
        self.settle = settle_seconds

    # ── event handlers (§12.3: on_created *and* on_moved are first-class) ──

    def _to_path(self, raw: bytes | str) -> Path:
        """Normalise a watchdog event path (``bytes | str``) to a ``Path``."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return Path(raw).resolve()

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        path = self._to_path(event.src_path)
        if is_temp_file(path.name):
            return
        self._schedule_stability(path)

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        dest = self._to_path(event.dest_path)
        if is_temp_file(dest.name):
            return
        if self.inbox in dest.parents:
            self._schedule_stability(dest)

    # ── stability polling ─────────────────────────────────────────────

    def _schedule_stability(self, path: Path) -> None:
        """Run the blocking stability poll in a thread-pool executor."""
        self.loop.run_in_executor(None, self._poll_stable, path)

    def _poll_stable(self, path: Path) -> None:
        """Block until the file is stable (exists, size > 0, unchanged).

        Polls twice, *settle_seconds* apart, checking that size and mtime
        are identical between polls.  On stable, pushes the path onto the
        queue via ``call_soon_threadsafe``.
        """
        time.sleep(self.settle)
        try:
            s1 = path.stat()
        except OSError:
            return
        if s1.st_size == 0:
            return

        time.sleep(self.settle)
        try:
            s2 = path.stat()
        except OSError:
            return
        if s2.st_size == s1.st_size and s2.st_mtime == s1.st_mtime:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, path)

    # ── lifecycle ─────────────────────────────────────────────────────

    def start(self) -> Observer:
        """Create, schedule on the inbox directory, and start the observer.

        Returns:
            The started ``Observer`` instance (caller must stop it).
        """
        observer = Observer()
        observer.schedule(self, str(self.inbox), recursive=True)
        observer.start()
        return observer

    @staticmethod
    def stop(observer: Observer) -> None:
        """Stop the observer and wait for its thread to join."""
        observer.stop()
        observer.join()
