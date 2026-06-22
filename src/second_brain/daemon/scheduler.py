"""Compaction scheduler — triggers compaction daily OR every N new sources (§8).

Two triggers (§8 Track 5-2a):
  - Daily: 24h elapsed since last compaction.
  - Threshold: N new sources since last compaction (default 25).

After compaction: git commit versioned files (§12.8) + daily snapshot (§12.6).

Design notes
------------
* Never raises out of :func:`maybe_run_compaction` — a scheduler failure must
  not take down the daemon watcher loop. The single ``run_compaction`` call
  is wrapped in try/except; the post-compaction side-effects (snapshot +
  git commit) each swallow their own errors.
* Git commit is scoped to *versioned* project files only (§12.8 amended) —
  derived data (``50-sources/``, ``90-wiki/``, ``INDEX.md``, ``state.json``,
  ``changelog.jsonl``) is gitignored and must never be staged.
"""

from __future__ import annotations

import contextlib
import datetime
import shutil
import subprocess

import structlog

from second_brain.config import Config
from second_brain.state import BrainStateStore, now_iso

log = structlog.get_logger(__name__)

DAILY_INTERVAL_S = 24 * 60 * 60  # 24 hours
DEFAULT_SOURCE_THRESHOLD = 25

# Files / dirs that are checked into git and may be auto-committed after a
# compaction pass (§12.8 amended).  Anything not listed here (derived data)
# is gitignored and must never be staged.
_VERSIONED_PATHS: tuple[str, ...] = (
    "ARCHITECTURE.md",
    "README.md",
    "src/",
    "tests/",
    "config.example.toml",
    "pyproject.toml",
    "uv.lock",
)


async def maybe_run_compaction(
    cfg: Config,
    store: BrainStateStore,
    vec_store,
    client,
    *,
    source_threshold: int = DEFAULT_SOURCE_THRESHOLD,
) -> bool:
    """Check if compaction should run, and run it if so.

    Returns:
        ``True`` if compaction ran, ``False`` otherwise. Never raises —
        any failure is logged and reported as ``False``.
    """
    from second_brain.compact.compaction import run_compaction

    should_run = False
    reason = ""

    # Trigger 1: source count threshold
    if store.state.sources_since_compaction >= source_threshold:
        should_run = True
        reason = (
            f"threshold_reached ({store.state.sources_since_compaction} "
            f">= {source_threshold})"
        )

    # Trigger 2: daily interval (only checked if threshold not hit)
    if not should_run and store.state.last_compaction_ts:
        try:
            last = datetime.datetime.fromisoformat(
                store.state.last_compaction_ts.replace("Z", "+00:00")
            )
            elapsed = (
                datetime.datetime.now(datetime.UTC) - last
            ).total_seconds()
        except ValueError:
            log.warning(
                "compaction.bad_timestamp",
                ts=store.state.last_compaction_ts,
            )
            elapsed = 0.0
        if elapsed >= DAILY_INTERVAL_S:
            should_run = True
            reason = f"daily_interval ({elapsed / 3600:.1f}h elapsed)"
    elif not should_run and not store.state.last_compaction_ts:
        # First-ever run — compact if there are any sources at all
        if len(store.state.sources) > 0:
            should_run = True
            reason = "first_run"

    if not should_run:
        return False

    log.info("compaction.scheduler.triggered", reason=reason)

    try:
        summary = await run_compaction(
            cfg,
            store,
            vec_store,
            client,
            merge_threshold=cfg.compaction.merge_threshold,
        )
    except Exception as exc:  # never bubble up to the watcher loop
        log.error("compaction.scheduler.failed", error=str(exc))
        return False

    # Reset counter + timestamp on a successful compaction
    store.state.sources_since_compaction = 0
    store.state.last_compaction_ts = now_iso()
    store.save()

    # Post-compaction side-effects — each swallows its own errors so a
    # snapshot or git failure cannot unwind the compaction that already ran.
    post_compaction(cfg, store, summary)

    log.info(
        "compaction.scheduler.done",
        merges=summary.get("merges", 0),
        reason=reason,
    )
    return True


def post_compaction(cfg: Config, store: BrainStateStore, summary: dict) -> None:
    """Post-compaction side effects: snapshot + git commit (§12.6, §12.8).

    Factored out so both the scheduler (``maybe_run_compaction``) and the
    daemon's ``POST /compact`` endpoint run identical side-effects after a
    successful compaction. Each sub-step swallows its own errors so neither
    a snapshot failure nor a git failure can unwind the compaction that
    already ran.
    """
    _write_snapshot(cfg, store)
    _git_commit_on_compaction(cfg, summary)


def _write_snapshot(cfg: Config, store: BrainStateStore) -> None:  # noqa: ARG001
    """Write a daily snapshot of ``state.json`` to ``.brain/snapshots/`` (§12.6).

    Rotates to keep only the last 30 snapshots (lexicographic order on the
    timestamped filename == chronological order).
    """
    src = cfg.brain_root / ".brain" / "state.json"
    snap_dir = cfg.brain_root / ".brain" / "snapshots"
    try:
        snap_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        dst = snap_dir / f"state-{ts}.json"
        # Collision guard: if two compactions land in the same second, append
        # a numeric suffix. The format ``state-{ts}-{i}.json`` still matches
        # the ``state-*.json`` glob and sorts after the base name.
        if dst.exists():
            i = 1
            while (snap_dir / f"state-{ts}-{i}.json").exists():
                i += 1
            dst = snap_dir / f"state-{ts}-{i}.json"
        if src.exists():
            shutil.copy2(src, dst)

        # Rotate: keep only the last 30 snapshots.
        snaps = sorted(snap_dir.glob("state-*.json"))
        if len(snaps) > 30:
            for old in snaps[:-30]:
                with contextlib.suppress(Exception):
                    old.unlink(missing_ok=True)
    except Exception as exc:  # passive — never block the scheduler
        log.warning("compaction.snapshot_failed", error=str(exc))


def _git_commit_on_compaction(cfg: Config, summary: dict) -> None:
    """Commit changed versioned project files after compaction (§12.8).

    Scoped to the entries in :data:`_VERSIONED_PATHS` only. No-op if nothing
    is staged (derived data is gitignored). Graceful fallback if git is not
    available or this is not a git repo.
    """
    brain_root = cfg.brain_root

    # Check git availability + repo-ness. Any failure => silent no-op.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(brain_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or result.stdout.strip() != "true":
            return  # not a git repo
    except (FileNotFoundError, OSError):
        return  # git not installed

    # Stage versioned files only (§12.8 amended). Derived data (sources,
    # wiki, INDEX.md, state.json, changelog.jsonl) is gitignored and must
    # never be staged here.
    try:
        subprocess.run(
            ["git", "add", "--", *_VERSIONED_PATHS],
            cwd=str(brain_root),
            capture_output=True,
            check=False,
        )

        # Exit code 0 from ``git diff --cached --quiet`` means *nothing* is
        # staged — no-op rather than creating an empty commit.
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(brain_root),
            capture_output=True,
            check=False,
        )
        if diff.returncode == 0:
            return  # nothing staged

        merges = summary.get("merges", 0)
        msg = (
            f"chore(compaction): auto-commit post-compaction ({merges} merges)"
        )
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(brain_root),
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        log.warning("git.commit_failed", error=str(exc))
