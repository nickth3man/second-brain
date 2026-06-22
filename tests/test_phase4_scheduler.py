"""Phase 4 tests — compaction scheduler (§8 Track 5-2a).

No network, no API.  Mocks out ``run_compaction`` so no real compaction
runs; uses the same ``_FakeCfg`` pattern as ``test_phase4.py``.

Covers:
- Threshold trigger (sources_since_compaction >= 25)
- Below-threshold no-op
- Daily-interval trigger (>24h since last compaction)
- First-run trigger (no prior compaction, sources exist)
- ``_write_snapshot`` file creation + rotation
- ``_git_commit_on_compaction`` no-op when nothing staged

References
----------
- ARCHITECTURE.md §8 (compaction cadence)
- ARCHITECTURE.md §12.6 (daily snapshots)
- ARCHITECTURE.md §12.8 amended (git commit on compaction)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

from second_brain.config import TypesCfg
from second_brain.daemon import scheduler as scheduler_mod
from second_brain.daemon.scheduler import (
    DEFAULT_SOURCE_THRESHOLD,
    _git_commit_on_compaction,
    _write_snapshot,
    maybe_run_compaction,
)
from second_brain.models import SourceState
from second_brain.state import BrainStateStore

DIM = 8


# ---------------------------------------------------------------------------
# Stubs (same pattern as test_phase4)
# ---------------------------------------------------------------------------


@dataclass
class _FakeCompaction:
    merge_threshold: float = 0.85


@dataclass
class _FakeCfg:
    brain_root: Path
    types: TypesCfg
    compaction: _FakeCompaction = field(default_factory=_FakeCompaction)


def _make_cfg(tmp_path: Path) -> _FakeCfg:
    return _FakeCfg(
        brain_root=tmp_path,
        types=TypesCfg(
            text=["md", "txt", "markdown"],
            code=["py", "js", "ts"],
            structured=["json", "yaml", "toml"],
            vision=[],
            pdf=[],
            office=[],
            web=[],
            ebook=[],
            audio=[],
            video=[],
        ),
    )


def _ensure_dirs(cfg: _FakeCfg) -> None:
    (cfg.brain_root / "00-inbox").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / "50-sources").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / "90-wiki").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / ".brain").mkdir(parents=True, exist_ok=True)


class _FakeVecStore:
    """Stand-in for VectorStore — never used because run_compaction is mocked."""

    pass


class _FakeClient:
    """Stand-in for OpenRouterClient — never used because run_compaction is mocked."""

    pass


def _iso(dt: datetime) -> str:
    """Format a datetime as the same ISO 8601 ``now_iso`` produces."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


async def _fake_run_compaction_ok(*args, **kwargs):  # noqa: ARG001
    """Mock replacement that returns a plausible summary dict."""
    return {"merges": 2, "pairs": [], "merged_into": {}}


# ---------------------------------------------------------------------------
# Trigger tests
# ---------------------------------------------------------------------------


class TestSchedulerTriggers:
    """maybe_run_compaction trigger logic (run_compaction is mocked)."""

    @pytest.mark.asyncio
    async def test_maybe_run_compaction_threshold(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)
        # Set a recent compaction ts so the daily trigger doesn't also fire.
        store.state.last_compaction_ts = _iso(datetime.now(UTC))
        store.state.sources_since_compaction = DEFAULT_SOURCE_THRESHOLD
        # Need at least one source for first_run edge cases (not used here
        # because threshold fires first, but defensive).
        store.state.sources["src-1"] = SourceState(sha256="x", raw="r")

        with mock.patch(
            "second_brain.compact.compaction.run_compaction",
            new=_fake_run_compaction_ok,
        ):
            ran = await maybe_run_compaction(
                cfg, store, _FakeVecStore(), _FakeClient()
            )

        assert ran is True
        # Counter reset, timestamp updated.
        assert store.state.sources_since_compaction == 0
        assert store.state.last_compaction_ts != ""

    @pytest.mark.asyncio
    async def test_maybe_run_compaction_below_threshold(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)
        # Recent compaction + small counter -> no trigger.
        store.state.last_compaction_ts = _iso(datetime.now(UTC))
        store.state.sources_since_compaction = 5

        with mock.patch(
            "second_brain.compact.compaction.run_compaction",
            new_callable=mock.AsyncMock,
            return_value={"merges": 0, "pairs": [], "merged_into": {}},
        ) as m:
            ran = await maybe_run_compaction(
                cfg, store, _FakeVecStore(), _FakeClient()
            )

        assert ran is False
        m.assert_not_called()
        # Counter unchanged.
        assert store.state.sources_since_compaction == 5

    @pytest.mark.asyncio
    async def test_maybe_run_compaction_daily(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)
        # Last compaction >24h ago.
        old = datetime.now(UTC) - timedelta(hours=25)
        store.state.last_compaction_ts = _iso(old)
        store.state.sources_since_compaction = 0

        with mock.patch(
            "second_brain.compact.compaction.run_compaction",
            new=_fake_run_compaction_ok,
        ):
            ran = await maybe_run_compaction(
                cfg, store, _FakeVecStore(), _FakeClient()
            )

        assert ran is True
        assert store.state.sources_since_compaction == 0  # reset
        # Timestamp was refreshed.
        assert store.state.last_compaction_ts != _iso(old)

    @pytest.mark.asyncio
    async def test_maybe_run_compaction_first_run(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)
        # First-ever run: no prior compaction, but sources exist.
        store.state.last_compaction_ts = ""
        store.state.sources_since_compaction = 0
        store.state.sources["src-1"] = SourceState(sha256="x", raw="r")

        with mock.patch(
            "second_brain.compact.compaction.run_compaction",
            new=_fake_run_compaction_ok,
        ):
            ran = await maybe_run_compaction(
                cfg, store, _FakeVecStore(), _FakeClient()
            )

        assert ran is True
        assert store.state.last_compaction_ts != ""

    @pytest.mark.asyncio
    async def test_first_run_skipped_when_no_sources(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)
        store.state.last_compaction_ts = ""
        store.state.sources_since_compaction = 0
        # No sources at all -> nothing to compact.

        with mock.patch(
            "second_brain.compact.compaction.run_compaction",
            new_callable=mock.AsyncMock,
            return_value={"merges": 0, "pairs": [], "merged_into": {}},
        ) as m:
            ran = await maybe_run_compaction(
                cfg, store, _FakeVecStore(), _FakeClient()
            )

        assert ran is False
        m.assert_not_called()

    @pytest.mark.asyncio
    async def test_compaction_failure_does_not_reset_counter(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)
        store.state.last_compaction_ts = _iso(datetime.now(UTC))
        store.state.sources_since_compaction = DEFAULT_SOURCE_THRESHOLD

        async def _boom(*a, **k):  # noqa: ARG001
            raise RuntimeError("boom")

        with mock.patch(
            "second_brain.compact.compaction.run_compaction", new=_boom
        ):
            ran = await maybe_run_compaction(
                cfg, store, _FakeVecStore(), _FakeClient()
            )

        # Failure surfaces as False, counter is NOT reset, scheduler survives.
        assert ran is False
        assert store.state.sources_since_compaction == DEFAULT_SOURCE_THRESHOLD


# ---------------------------------------------------------------------------
# Periodic-task cancellation tests
# ---------------------------------------------------------------------------


class TestPeriodicTaskCancellation:
    """The daemon's periodic compaction task must be cleanly cancellable.

    On shutdown, ``run_daemon`` cancels the periodic task and awaits it
    (§12.3). If the task swallowed ``CancelledError`` or hung, shutdown
    would block forever. These tests verify the cancellation path works
    end-to-end with the same loop shape used in ``run_daemon``.
    """

    @pytest.mark.asyncio
    async def test_periodic_compaction_task_is_cancellable(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)
        # Recent ts + zero counter so maybe_run_compaction is a no-op if it
        # ever runs — what we're testing here is cancellation, not compaction.
        store.state.last_compaction_ts = _iso(datetime.now(UTC))
        store.state.sources_since_compaction = 0

        # Mirror the _periodic_compaction_check loop from run_daemon exactly.
        async def _periodic_compaction_check() -> None:
            while True:
                await asyncio.sleep(3600)  # 1 hour
                try:
                    await maybe_run_compaction(
                        cfg, store, _FakeVecStore(), _FakeClient()
                    )
                except Exception as exc:  # noqa: BLE001
                    # Daemon loop swallows per-iteration errors but NOT
                    # CancelledError — that must propagate.
                    _ = exc

        periodic_task = asyncio.create_task(_periodic_compaction_check())
        # Let the task start and reach the asyncio.sleep(3600) await point.
        await asyncio.sleep(0)

        periodic_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await periodic_task

        assert periodic_task.cancelled()

    @pytest.mark.asyncio
    async def test_maybe_run_compaction_cancellable_mid_await(
        self, tmp_path: Path
    ) -> None:
        """maybe_run_compaction itself must be cancellable mid-await.

        If run_compaction is in-flight when shutdown fires, the wrapping
        task's CancelledError must propagate rather than be swallowed by
        the scheduler's broad except.
        """
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)
        store.state.last_compaction_ts = _iso(datetime.now(UTC))
        store.state.sources_since_compaction = DEFAULT_SOURCE_THRESHOLD

        started = asyncio.Event()

        async def _slow_compaction(*args, **kwargs):  # noqa: ARG001
            started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                raise
            return {"merges": 0, "pairs": [], "merged_into": {}}

        with mock.patch(
            "second_brain.compact.compaction.run_compaction",
            new=_slow_compaction,
        ):
            task = asyncio.create_task(
                maybe_run_compaction(
                    cfg, store, _FakeVecStore(), _FakeClient()
                )
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


class TestSnapshot:
    """_write_snapshot creates + rotates snapshot files."""

    def test_write_snapshot_creates_file(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)

        # Seed a state.json we can copy from.
        state_path = cfg.brain_root / ".brain" / "state.json"
        state_path.write_text(
            json.dumps({"schema_version": 1, "topics": {}}), encoding="utf-8"
        )

        _write_snapshot(cfg, store)

        snap_dir = cfg.brain_root / ".brain" / "snapshots"
        snaps = list(snap_dir.glob("state-*.json"))
        assert len(snaps) == 1
        assert json.loads(snaps[0].read_text(encoding="utf-8"))[
            "schema_version"
        ] == 1

    def test_write_snapshot_rotates_to_30(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)

        state_path = cfg.brain_root / ".brain" / "state.json"
        state_path.write_text("{}", encoding="utf-8")

        snap_dir = cfg.brain_root / ".brain" / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        # Pre-seed 35 old snapshots with lexicographically sortable names.
        for i in range(35):
            (snap_dir / f"state-2024010{i:02d}T000000Z.json").write_text(
                "{}", encoding="utf-8"
            )

        _write_snapshot(cfg, store)

        snaps = list(snap_dir.glob("state-*.json"))
        # Rotation keeps max 30.
        assert len(snaps) == 30

    def test_write_snapshot_missing_state_is_silent(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)
        # Don't write state.json — should not raise.
        _write_snapshot(cfg, store)
        # Snapshot dir exists but no snapshot was taken (no source file).
        snap_dir = cfg.brain_root / ".brain" / "snapshots"
        assert snap_dir.exists()
        assert list(snap_dir.glob("state-*.json")) == []


# ---------------------------------------------------------------------------
# Git commit tests
# ---------------------------------------------------------------------------


class TestGitCommit:
    """_git_commit_on_compaction — graceful fallback + no-op cases."""

    def test_git_commit_noop_when_nothing_staged(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)

        # Subprocess.run is mocked so we never touch the real git repo.
        def fake_run(cmd, *args, **kwargs):  # noqa: ARG001
            class R:
                # Pretend it IS a git repo, but `git diff --cached --quiet`
                # returns 0 (nothing staged).
                def __init__(self, c: list[str]) -> None:
                    self._cmd = c

                @property
                def returncode(self) -> int:
                    if "rev-parse" in self._cmd:
                        return 0
                    if "diff" in self._cmd:
                        return 0  # nothing staged
                    return 0

                @property
                def stdout(self) -> str:
                    if "rev-parse" in self._cmd:
                        return "true\n"
                    return ""

            return R(cmd)

        commits: list[list[str]] = []

        def fake_run_recording(cmd, *args, **kwargs):  # noqa: ARG001
            class R:
                def __init__(self, c: list[str]) -> None:
                    self._cmd = c

                @property
                def returncode(self) -> int:
                    if "rev-parse" in self._cmd:
                        return 0
                    if "diff" in self._cmd:
                        return 0  # nothing staged
                    return 0

                @property
                def stdout(self) -> str:
                    return "true\n" if "rev-parse" in self._cmd else ""

            if "commit" in cmd:
                commits.append(cmd)
            return R(cmd)

        with mock.patch(
            "second_brain.daemon.scheduler.subprocess.run",
            new=fake_run_recording,
        ):
            _git_commit_on_compaction(cfg, {"merges": 1})

        # No commit should have been issued because nothing was staged.
        assert commits == []

    def test_git_commit_noop_when_not_a_repo(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)

        calls: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):  # noqa: ARG001
            calls.append(cmd)

            class R:
                returncode = 1  # not a git repo
                stdout = ""

            return R()

        with mock.patch(
            "second_brain.daemon.scheduler.subprocess.run", new=fake_run
        ):
            _git_commit_on_compaction(cfg, {"merges": 1})

        # Only the rev-parse probe was issued; no add/commit attempted.
        assert any("rev-parse" in c for c in calls)
        assert not any("add" in c or "commit" in c for c in calls)

    def test_git_commit_commits_when_staged(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)

        commits: list[list[str]] = []

        def fake_run(cmd, *args, **kwargs):  # noqa: ARG001
            class R:
                def __init__(self, c: list[str]) -> None:
                    self._cmd = c

                @property
                def returncode(self) -> int:
                    if "rev-parse" in self._cmd:
                        return 0
                    if "diff" in self._cmd:
                        return 1  # something IS staged
                    return 0

                @property
                def stdout(self) -> str:
                    return "true\n" if "rev-parse" in self._cmd else ""

            if "commit" in cmd:
                commits.append(cmd)
            return R(cmd)

        with mock.patch(
            "second_brain.daemon.scheduler.subprocess.run", new=fake_run
        ):
            _git_commit_on_compaction(cfg, {"merges": 3})

        # Exactly one commit was issued, with the expected message format.
        assert len(commits) == 1
        assert "3 merges" in commits[0][commits[0].index("-m") + 1]

    def test_git_commit_never_raises_on_oserror(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)

        def fake_run(cmd, *args, **kwargs):  # noqa: ARG001
            raise FileNotFoundError("git not installed")

        # Should swallow the FileNotFoundError and not raise.
        with mock.patch(
            "second_brain.daemon.scheduler.subprocess.run", new=fake_run
        ):
            _git_commit_on_compaction(cfg, {"merges": 1})


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_threshold_constant_matches_spec() -> None:
    """§8 Track 5-2a: threshold is 25 new sources."""
    assert DEFAULT_SOURCE_THRESHOLD == 25


def test_daily_interval_constant() -> None:
    """24h == 86400s."""
    assert scheduler_mod.DAILY_INTERVAL_S == 24 * 60 * 60


def test_versioned_paths_exclude_derived_data() -> None:
    """§12.8 amended: derived data must NEVER be in the staging list."""
    forbidden = (
        "50-sources",
        "90-wiki",
        "INDEX.md",
        "state.json",
        "changelog.jsonl",
    )
    for path in scheduler_mod._VERSIONED_PATHS:
        for bad in forbidden:
            assert bad not in path, (
                f"{bad!r} must never appear in _VERSIONED_PATHS (got {path!r})"
            )
