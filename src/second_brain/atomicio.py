"""Atomic file writes and rolling backups (§12.6).

Provides write_atomic (same-dir tmp + fsync + os.replace), write_json_atomic,
and rolling_backup (3-deep rotation with shutil.copy2).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def write_atomic(path: Path, data: str | bytes, *, text_mode: bool = True) -> None:
    """Atomically write *data* to *path* using a same-directory temp file.

    Writes to ``path.tmp.<pid>``, flushes, fsyncs, then ``os.replace`` over it.
    Cleans up the temp file on exception.

    Args:
        path: Target file path (parent dir is created if missing).
        data: Content to write (str or bytes).
        text_mode: If True and *data* is str, write as text (utf-8).
                   If False and *data* is str, encode to utf-8 then write binary.
                   Ignored for bytes input.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.name}.tmp.{os.getpid()}"
    try:
        if isinstance(data, bytes):
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        elif text_mode:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        else:
            with open(tmp, "wb") as f:
                f.write(data.encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, obj: Any) -> None:
    """Serialize *obj* as JSON and write atomically to *path*."""
    data = json.dumps(obj, indent=2, ensure_ascii=False)
    write_atomic(path, data, text_mode=True)


def rolling_backup(path: Path, depth: int = 3) -> None:
    """Rotate backups for *path*, keeping at most *depth* copies.

    If *path* exists::

        path                -> path.bak
        path.bak            -> path.bak-1
        path.bak-1          -> path.bak-2
        ...
        path.bak-(depth-2)  -> path.bak-(depth-1)
        path.bak-(depth-1)  (removed)

    Uses shutil.copy2 for the primary backup (preserves mtime) and
    shutil.move for rotation.

    Win32 note: os.replace may copy instead of rename, so backups are
    mandatory (§12.6). This function ensures at least *depth* historical
    copies exist at all times.
    """
    if not path.exists():
        return

    parent = path.parent
    stem = path.name

    # Remove the oldest backup slot
    (parent / f"{stem}.bak-{depth - 1}").unlink(missing_ok=True)

    # Rotate existing backups upward
    for i in range(depth - 2, -1, -1):
        src = parent / (f"{stem}.bak" if i == 0 else f"{stem}.bak-{i}")
        dst = parent / f"{stem}.bak-{i + 1}"
        if src.exists():
            shutil.move(str(src), str(dst))

    # Copy the current file to the .bak slot
    shutil.copy2(path, parent / f"{stem}.bak")
