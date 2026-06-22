"""Watchdog-based file watcher for ``00-inbox/`` with stable-file gate (§12.3).

The :class:`InboxWatcher` monitors the inbox directory via ``watchdog``,
ignores temp-file patterns, polls for file stability (size + mtime unchanged
across two polls), and emits stable file paths onto an :class:`asyncio.Queue`
for the pipeline.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import structlog
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from second_brain.daemon.router import is_temp_file

log = structlog.get_logger(__name__)


def _can_open_exclusively(path: Path) -> bool:
    """Return True if *path* can be opened without sharing (§12.3).

    On Windows this uses ``CreateFileW`` with ``dwShareMode=0`` and
    ``dwDesiredAccess=0`` (``GENERIC_READ`` is used for the access mask so
    we can read the file once opened), which fails with
    ``ERROR_SHARING_VIOLATION`` if another process holds the file open for
    writing.  On success we immediately close the handle and return True.

    On non-Windows platforms we fall back to a best-effort ``open(path, 'rb')``
    check; POSIX ``open(2)`` does not enforce the same kind of mandatory
    share-mode lock, so this is only an approximate check and is documented
    as such.
    """
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        CreateFileW = kernel32.CreateFileW
        CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        CreateFileW.restype = wintypes.HANDLE

        INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
        GENERIC_READ = 0x80000000
        OPEN_EXISTING = 3
        FILE_SHARE_NONE = 0
        FILE_ATTRIBUTE_NORMAL = 0x80

        handle = CreateFileW(
            str(path),
            GENERIC_READ,
            FILE_SHARE_NONE,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            None,
        )
        if handle == INVALID_HANDLE_VALUE:
            return False
        kernel32.CloseHandle(handle)
        return True

    # Non-Windows: best-effort read-open.  POSIX does not provide a reliable
    # exclusive-open check, so this is a weaker guarantee.
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


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

    # -- event handlers (§12.3: on_created *and* on_moved are first-class) --

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

    # -- stability polling ---------------------------------------------

    def _schedule_stability(self, path: Path) -> None:
        """Run the blocking stability poll in a thread-pool executor."""
        self.loop.run_in_executor(None, self._poll_stable, path)

    def _poll_stable(self, path: Path) -> None:
        """Block until the file is stable (§12.3 stable-file gate).

        A file is considered stable when **all** of the following hold:

        1. ``stat()`` succeeds (file exists).
        2. ``st_size > 0`` (non-empty).
        3. ``st_size`` and ``st_mtime`` are unchanged across two polls
           separated by ``settle_seconds``.
        4. The file can be opened exclusively — i.e. no other process
           currently holds it open for writing.  On Windows this is a real
           share-mode check via ``CreateFileW(..., dwShareMode=0)``; on other
           platforms it is a best-effort ``open(path, 'rb')`` check.

        On stable, pushes the path onto the queue via
        ``call_soon_threadsafe``.  If the exclusivity check fails the file is
        presumed to still be being written and the poll is aborted (the
        caller may reschedule).
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
        if s2.st_size != s1.st_size or s2.st_mtime != s1.st_mtime:
            return

        if not _can_open_exclusively(path):
            log.debug("file_still_locked_skip_poll", path=str(path))
            return

        self.loop.call_soon_threadsafe(self.queue.put_nowait, path)

    # -- lifecycle -----------------------------------------------------

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
