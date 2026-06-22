"""Phase 5 — brain rebuild tests (§12.6).

Covers ``rebuild_from_sources`` (fast path), ``rebuild_from_inbox`` (deep path),
dry-run mode, snapshot/restore, daemon-refusal, idempotency, and preservation of
non-derived files.

No network, no API.  Uses ``_FakeCfg``, ``FakeClient``, ``FakeEmbedder`` stubs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import second_brain.rebuild as sb_rebuild
from second_brain.atomicio import write_atomic
from second_brain.frontmatter import dump_frontmatter
from second_brain.models import IngestStage, SourceState
from second_brain.rebuild import rebuild_from_inbox, rebuild_from_sources
from second_brain.state import BrainStateStore

DIM = 8


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _FakeDaemon:
    http_host: str = "127.0.0.1"
    http_port: int = 8001


@dataclass
class _FakeModels:
    embedding: str = "test-embed"


@dataclass
class _FakeIngestion:
    merge_threshold: float = 0.85


@dataclass
class _FakeCfg:
    brain_root: Path
    daemon: _FakeDaemon = field(default_factory=_FakeDaemon)
    models: _FakeModels = field(default_factory=_FakeModels)
    ingestion: _FakeIngestion = field(default_factory=_FakeIngestion)


class FakeEmbedder:
    """Deterministic embedder — no API calls."""

    def __init__(self, client=None, cfg=None) -> None:
        self.dim = DIM
        self.client = client
        self.cfg = cfg

    async def ensure_dim(self) -> int:
        return self.dim

    async def embed_one(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.1] * self.dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
        return [[0.1] * self.dim for _ in texts]

    async def embed_query(self, query: str) -> list[float]:  # noqa: ARG002
        return [0.1] * self.dim


class FakeClient:
    """Fake OpenRouter client (only close() is needed when Embedder is patched)."""

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path: Path) -> _FakeCfg:
    return _FakeCfg(brain_root=tmp_path)


def _seed_sources_and_wiki(cfg: _FakeCfg) -> None:
    """Create 2 source files, 2 wiki pages, and a pre-seeded BrainStateStore."""
    (cfg.brain_root / "50-sources").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / "90-wiki").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / ".brain").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / "00-inbox").mkdir(parents=True, exist_ok=True)

    # Source A
    src_a = (
        "---\n"
        "topics: [topic-a]\n"
        "---\n"
        "# Source A Title\n"
        "## Summary\n"
        "This is the summary for source A.\n"
    )
    (cfg.brain_root / "50-sources" / "src-a.md").write_text(src_a, encoding="utf-8")

    # Source B
    src_b = (
        "---\n"
        "topics: [topic-b]\n"
        "---\n"
        "# Source B Title\n"
        "## Summary\n"
        "This is the summary for source B.\n"
    )
    (cfg.brain_root / "50-sources" / "src-b.md").write_text(src_b, encoding="utf-8")

    # Pre-seed state
    store = BrainStateStore.load(cfg)
    store.ensure_topic("topic-a", "Topic A")
    store.ensure_topic("topic-b", "Topic B")
    store.state.sources["src-a"] = SourceState(
        sha256="aaa",
        raw="src-a.md",
        topics=["topic-a"],
        stage=IngestStage.DONE,
    )
    store.state.sources["src-b"] = SourceState(
        sha256="bbb",
        raw="src-b.md",
        topics=["topic-b"],
        stage=IngestStage.DONE,
    )
    store.save()

    # Write wiki pages with Synthesis
    wiki_a_meta = {
        "title": "Topic A",
        "slug": "topic-a",
        "type": "concept",
        "created": "2026-06-01",
        "confidence": 0.8,
    }
    wiki_a_body = (
        "# Topic A\n\n"
        "## Synthesis\n\n"
        "Existing synthesis for topic A.\n\n"
        "## Sources\n\n"
        "## Open questions\n"
        "-\n\n"
        "## Related\n"
        "-\n"
    )
    write_atomic(
        cfg.brain_root / "90-wiki" / "topic-a.md",
        dump_frontmatter(wiki_a_meta, wiki_a_body),
    )

    wiki_b_meta = {
        "title": "Topic B",
        "slug": "topic-b",
        "type": "concept",
        "created": "2026-06-01",
        "confidence": 0.7,
    }
    wiki_b_body = (
        "# Topic B\n\n"
        "## Synthesis\n\n"
        "Existing synthesis for topic B.\n\n"
        "## Sources\n\n"
        "## Open questions\n"
        "-\n\n"
        "## Related\n"
        "-\n"
    )
    write_atomic(
        cfg.brain_root / "90-wiki" / "topic-b.md",
        dump_frontmatter(wiki_b_meta, wiki_b_body),
    )


# -- fake ingest_file for rebuild_from_inbox tests --------------------------


async def _fake_ingest_file(
    path: Path,
    cfg: _FakeCfg,
    store: BrainStateStore,
    client: object,  # noqa: ARG001
    linker: object,  # noqa: ARG001
    index: object,  # noqa: ARG001
    *,
    embedder: object = None,  # noqa: ARG002
    vec_store: object = None,  # noqa: ARG002
    progress: object = None,  # noqa: ARG002
) -> IngestStage:
    """Fake ingest_file that writes one source + wiki page directly."""
    stem = path.stem
    slug = f"topic-{stem}"

    # Write source file
    src_text = (
        "---\n"
        f"topics: [{slug}]\n"
        "---\n"
        f"# {stem.title()} Title\n"
        "## Summary\n"
        f"Summary of {stem}.\n"
    )
    (cfg.brain_root / "50-sources" / f"{stem}.md").write_text(src_text, encoding="utf-8")

    # Write wiki page
    meta = {"title": stem.title(), "slug": slug, "type": "concept"}
    body = (
        f"# {stem.title()}\n\n"
        "## Synthesis\n\n"
        f"Synthesis for {stem}.\n\n"
        "## Sources\n\n"
        "## Open questions\n"
        "-\n\n"
        "## Related\n"
        "-\n"
    )
    write_atomic(
        cfg.brain_root / "90-wiki" / f"{slug}.md",
        dump_frontmatter(meta, body),
    )

    # Register in state
    store.ensure_topic(slug, stem.title())
    store.state.sources[stem] = SourceState(
        sha256="fff",
        raw=f"{stem}.md",
        topics=[slug],
        stage=IngestStage.DONE,
    )

    return IngestStage.DONE


# ---------------------------------------------------------------------------
# Test: rebuild_from_sources — dry run
# ---------------------------------------------------------------------------


class TestRebuildFromSources:
    """rebuild_from_sources — dry run, full run, idempotency."""

    @pytest.mark.asyncio
    async def test_rebuild_from_sources_dry_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dry run reports counters and never writes."""
        monkeypatch.setattr("second_brain.rebuild.Embedder", FakeEmbedder)
        cfg = _make_cfg(tmp_path)
        _seed_sources_and_wiki(cfg)

        # Record pre-rebuild mtimes
        wiki_a = cfg.brain_root / "90-wiki" / "topic-a.md"
        wiki_b = cfg.brain_root / "90-wiki" / "topic-b.md"
        state_json = cfg.brain_root / ".brain" / "state.json"
        pre_mtimes = {
            "wiki-a": os.path.getmtime(wiki_a),
            "wiki-b": os.path.getmtime(wiki_b),
            "state": os.path.getmtime(state_json),
        }
        pre_wiki_a = wiki_a.read_text(encoding="utf-8")
        pre_state = state_json.read_text(encoding="utf-8")

        client = FakeClient()
        plan = await rebuild_from_sources(cfg, client, dry_run=True)

        assert plan.mode == "from-sources"
        assert plan.dry_run is True
        assert plan.sources_seen == 2
        assert plan.sources_skipped == 0
        assert plan.topics_before == 2
        assert plan.topics_after == 2
        assert plan.snapshot_dir is None

        # Verify no file mutations
        assert os.path.getmtime(wiki_a) == pre_mtimes["wiki-a"]
        assert os.path.getmtime(wiki_b) == pre_mtimes["wiki-b"]
        assert os.path.getmtime(state_json) == pre_mtimes["state"]
        assert wiki_a.read_text(encoding="utf-8") == pre_wiki_a
        assert state_json.read_text(encoding="utf-8") == pre_state

        # No snapshot dir created
        snapshots_root = cfg.brain_root / ".brain" / "snapshots"
        assert not snapshots_root.exists()

    # ------------------------------------------------------------------
    # Test: rebuild_from_sources — full run
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rebuild_from_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full run: topics recreated, synthesis preserved, sources assembled."""
        monkeypatch.setattr("second_brain.rebuild.Embedder", FakeEmbedder)
        cfg = _make_cfg(tmp_path)
        _seed_sources_and_wiki(cfg)

        client = FakeClient()
        plan = await rebuild_from_sources(cfg, client, dry_run=False)

        assert plan.mode == "from-sources"
        assert plan.dry_run is False
        assert plan.sources_seen == 2
        assert plan.sources_skipped == 0
        assert plan.topics_before == 2
        assert plan.topics_after == 2
        assert plan.snapshot_dir is not None
        assert plan.snapshot_dir.exists()

        # 1. Topics recreated in fresh state.json
        store = BrainStateStore.load(cfg)
        assert "topic-a" in store.state.topics
        assert "topic-b" in store.state.topics
        assert "src-a" in store.state.sources
        assert "src-b" in store.state.sources

        # 2. Synthesis preserved verbatim
        topic_a_text = (cfg.brain_root / "90-wiki" / "topic-a.md").read_text(
            encoding="utf-8"
        )
        assert "Existing synthesis for topic A." in topic_a_text
        topic_b_text = (cfg.brain_root / "90-wiki" / "topic-b.md").read_text(
            encoding="utf-8"
        )
        assert "Existing synthesis for topic B." in topic_b_text

        # 3. Sources section contains source title + summary text
        assert "Source A Title" in topic_a_text
        assert "This is the summary for source A." in topic_a_text
        assert "Source B Title" in topic_b_text
        assert "This is the summary for source B." in topic_b_text

        # 4. embeddings.db exists
        assert (cfg.brain_root / ".brain" / "embeddings.db").exists()

        # 5. Snapshot dir exists with expected subdir structure
        assert (plan.snapshot_dir / "90-wiki").exists()
        assert (plan.snapshot_dir / "state.json").exists()

        # 6. Changelog has a kind: "rebuild" entry
        changelog_path = cfg.brain_root / ".brain" / "changelog.jsonl"
        changelog_lines = changelog_path.read_text(encoding="utf-8").strip().split("\n")
        rebuild_entries = [
            json.loads(ln)
            for ln in changelog_lines
            if json.loads(ln).get("kind") == "rebuild"
        ]
        assert len(rebuild_entries) == 1
        assert rebuild_entries[0]["mode"] == "from-sources"

    # ------------------------------------------------------------------
    # Test: rebuild_from_inbox
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rebuild_from_inbox(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deep rebuild: inbox files trigger full re-ingestion."""
        monkeypatch.setattr("second_brain.rebuild.Embedder", FakeEmbedder)
        monkeypatch.setattr("second_brain.rebuild.ingest_file", _fake_ingest_file)
        cfg = _make_cfg(tmp_path)

        # Create inbox file
        (cfg.brain_root / "00-inbox").mkdir(parents=True, exist_ok=True)
        inbox_file = cfg.brain_root / "00-inbox" / "my-note.txt"
        inbox_file.write_text("original inbox content", encoding="utf-8")

        # Pre-seed some old state (will be wiped)
        (cfg.brain_root / "50-sources").mkdir(parents=True, exist_ok=True)
        (cfg.brain_root / "90-wiki").mkdir(parents=True, exist_ok=True)
        (cfg.brain_root / ".brain").mkdir(parents=True, exist_ok=True)
        store = BrainStateStore.load(cfg)
        store.ensure_topic("old-topic", "Old Topic")
        store.save()
        (cfg.brain_root / "50-sources" / "old-source.md").write_text(
            "old", encoding="utf-8"
        )
        (cfg.brain_root / "90-wiki" / "old-topic.md").write_text(
            "old wiki", encoding="utf-8"
        )

        client = FakeClient()
        plan = await rebuild_from_inbox(cfg, client, dry_run=False)

        assert plan.mode == "from-inbox"
        assert plan.dry_run is False
        assert plan.sources_seen == 1
        assert plan.sources_skipped == 0

        # 00-inbox untouched
        assert inbox_file.exists()
        assert inbox_file.read_text(encoding="utf-8") == "original inbox content"

        # 50-sources regenerated (new file exists, old file gone)
        assert (cfg.brain_root / "50-sources" / "my-note.md").exists()
        assert not (cfg.brain_root / "50-sources" / "old-source.md").exists()

        # 90-wiki regenerated
        assert (cfg.brain_root / "90-wiki" / "topic-my-note.md").exists()
        assert not (cfg.brain_root / "90-wiki" / "old-topic.md").exists()

    # ------------------------------------------------------------------
    # Test: preserves inbox / changelog
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rebuild_preserves_inbox_config_changelog(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-derived files are untouched; changelog is appended, not truncated."""
        monkeypatch.setattr("second_brain.rebuild.Embedder", FakeEmbedder)
        cfg = _make_cfg(tmp_path)
        _seed_sources_and_wiki(cfg)

        # Seed an inbox file
        inbox_file = cfg.brain_root / "00-inbox" / "test.txt"
        inbox_file.write_text("inbox content", encoding="utf-8")

        # Seed pre-existing changelog entries
        store = BrainStateStore.load(cfg)
        store.append_changelog({"kind": "test", "msg": "pre-existing entry 1"})
        store.append_changelog({"kind": "test", "msg": "pre-existing entry 2"})

        changelog_path = cfg.brain_root / ".brain" / "changelog.jsonl"
        pre_lines = changelog_path.read_text(encoding="utf-8").strip().split("\n")
        pre_entries = [json.loads(ln) for ln in pre_lines]

        # Record inbox and config status
        pre_inbox = inbox_file.read_text(encoding="utf-8")
        config_toml = cfg.brain_root / "config.toml"
        had_config = config_toml.exists()

        client = FakeClient()
        await rebuild_from_sources(cfg, client, dry_run=False)

        # Inbox untouched
        assert inbox_file.exists()
        assert inbox_file.read_text(encoding="utf-8") == pre_inbox

        # Config.toml unchanged (or never existed)
        assert config_toml.exists() == had_config

        # Pre-existing changelog lines still present (first N lines match)
        post_lines = changelog_path.read_text(encoding="utf-8").strip().split("\n")
        post_entries = [json.loads(ln) for ln in post_lines]
        assert len(post_entries) > len(pre_entries)
        for i, pre_entry in enumerate(pre_entries):
            # Compare entries without 'ts' (timestamps differ)
            pre_no_ts = {k: v for k, v in pre_entry.items() if k != "ts"}
            post_no_ts = {k: v for k, v in post_entries[i].items() if k != "ts"}
            assert pre_no_ts == post_no_ts, (
                f"Pre-existing changelog entry {i} was modified"
            )

    # ------------------------------------------------------------------
    # Test: snapshot created and restores on failure
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rebuild_snapshot_created_and_restores(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Snapshot/restore: a failure after the first write reverts wiki + state."""
        monkeypatch.setattr("second_brain.rebuild.Embedder", FakeEmbedder)
        cfg = _make_cfg(tmp_path)
        _seed_sources_and_wiki(cfg)

        # Record pre-rebuild content
        pre_wiki_a = (cfg.brain_root / "90-wiki" / "topic-a.md").read_text(
            encoding="utf-8"
        )
        pre_state = (cfg.brain_root / ".brain" / "state.json").read_text(
            encoding="utf-8"
        )

        # Monkeypatch BrainStateStore.save to fail after the first destructive write
        def _failing_save(self) -> None:  # noqa: ARG001
            raise RuntimeError("boom")

        monkeypatch.setattr(BrainStateStore, "save", _failing_save)

        client = FakeClient()
        with pytest.raises(RuntimeError, match="boom"):
            await rebuild_from_sources(cfg, client, dry_run=False)

        # Verify wiki and state are restored to pre-rebuild content
        post_wiki_a = (cfg.brain_root / "90-wiki" / "topic-a.md").read_text(
            encoding="utf-8"
        )
        post_state = (cfg.brain_root / ".brain" / "state.json").read_text(
            encoding="utf-8"
        )
        assert post_wiki_a == pre_wiki_a
        assert post_state == pre_state

    # ------------------------------------------------------------------
    # Test: _extract_section body-start edge case (SHOULD fix 5)
    # ------------------------------------------------------------------

    def test_extract_section_body_start(self) -> None:
        """Body starting with heading returns full content (no off-by-one)."""
        from second_brain.rebuild import _extract_section

        body = "## Synthesis\ncontent here\n\n## Next\nstuff"
        assert _extract_section(body, "## Synthesis") == "content here"

    # ------------------------------------------------------------------
    # Test: rebuild_from_inbox snapshot/restore (MUST fix 3)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rebuild_from_inbox_snapshot_restores(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inbox rebuild: snapshot restores 50-sources, 90-wiki, state on failure."""
        monkeypatch.setattr("second_brain.rebuild.Embedder", FakeEmbedder)
        cfg = _make_cfg(tmp_path)
        _seed_sources_and_wiki(cfg)

        # Record pre-rebuild content
        pre_source_a = (cfg.brain_root / "50-sources" / "src-a.md").read_text(
            encoding="utf-8"
        )
        pre_wiki_a = (cfg.brain_root / "90-wiki" / "topic-a.md").read_text(
            encoding="utf-8"
        )
        pre_state = (cfg.brain_root / ".brain" / "state.json").read_text(
            encoding="utf-8"
        )

        # Seed 2 inbox files
        inbox_dir = cfg.brain_root / "00-inbox"
        (inbox_dir / "note1.txt").write_text("note1", encoding="utf-8")
        (inbox_dir / "note2.txt").write_text("note2", encoding="utf-8")

        # Fail on the first ingest_file call
        call_count = [0]

        async def _failing_ingest_first(
            path, cfg_, store, client, linker, index, **kwargs
        ):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("boom during inbox replay")
            return await _fake_ingest_file(
                path, cfg_, store, client, linker, index, **kwargs
            )

        monkeypatch.setattr(
            "second_brain.rebuild.ingest_file", _failing_ingest_first
        )

        client = FakeClient()
        with pytest.raises(RuntimeError, match="boom during inbox replay"):
            await rebuild_from_inbox(cfg, client, dry_run=False)

        # 50-sources restored
        post_source_a = (cfg.brain_root / "50-sources" / "src-a.md").read_text(
            encoding="utf-8"
        )
        assert post_source_a == pre_source_a

        # 90-wiki restored
        post_wiki_a = (cfg.brain_root / "90-wiki" / "topic-a.md").read_text(
            encoding="utf-8"
        )
        assert post_wiki_a == pre_wiki_a

        # state.json restored
        post_state = (cfg.brain_root / ".brain" / "state.json").read_text(
            encoding="utf-8"
        )
        assert post_state == pre_state

    # ------------------------------------------------------------------
    # Test: partial wiki write failure + rollback (SHOULD fix 6)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rebuild_from_sources_partial_wiki_write_restores(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure mid-write still restores the full wiki + state."""
        monkeypatch.setattr("second_brain.rebuild.Embedder", FakeEmbedder)
        cfg = _make_cfg(tmp_path)
        _seed_sources_and_wiki(cfg)

        pre_wiki_a = (cfg.brain_root / "90-wiki" / "topic-a.md").read_text(
            encoding="utf-8"
        )
        pre_wiki_b = (cfg.brain_root / "90-wiki" / "topic-b.md").read_text(
            encoding="utf-8"
        )
        pre_state = (cfg.brain_root / ".brain" / "state.json").read_text(
            encoding="utf-8"
        )

        # Fail on the second write_atomic call (first wiki page written, second raises).
        write_count = [0]
        orig_write = sb_rebuild.write_atomic

        def _failing_write(path, data, **kwargs):
            write_count[0] += 1
            if write_count[0] >= 2:
                raise RuntimeError("boom after first page")
            orig_write(path, data, **kwargs)

        monkeypatch.setattr(
            "second_brain.rebuild.write_atomic", _failing_write
        )

        client = FakeClient()
        with pytest.raises(RuntimeError, match="boom after first page"):
            await rebuild_from_sources(cfg, client, dry_run=False)

        # Both pages and state restored
        post_wiki_a = (cfg.brain_root / "90-wiki" / "topic-a.md").read_text(
            encoding="utf-8"
        )
        post_wiki_b = (cfg.brain_root / "90-wiki" / "topic-b.md").read_text(
            encoding="utf-8"
        )
        post_state = (cfg.brain_root / ".brain" / "state.json").read_text(
            encoding="utf-8"
        )
        assert post_wiki_a == pre_wiki_a
        assert post_wiki_b == pre_wiki_b
        assert post_state == pre_state

    # ------------------------------------------------------------------
    # Test: dropped-topic page removal (SHOULD fix 7)
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rebuild_from_sources_removes_dropped_topic_pages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stale wiki pages for dropped topics are removed after rebuild."""
        monkeypatch.setattr("second_brain.rebuild.Embedder", FakeEmbedder)
        cfg = _make_cfg(tmp_path)
        _seed_sources_and_wiki(cfg)

        # Add a third source with FAILED stage and a wiki page
        src_c = (
            "---\n"
            "topics: [topic-c]\n"
            "---\n"
            "# Source C Title\n"
            "## Summary\n"
            "Summary for source C.\n"
        )
        (cfg.brain_root / "50-sources" / "src-c.md").write_text(
            src_c, encoding="utf-8"
        )

        store = BrainStateStore.load(cfg)
        store.ensure_topic("topic-c", "Topic C")
        store.state.sources["src-c"] = SourceState(
            sha256="ccc",
            raw="src-c.md",
            topics=["topic-c"],
            stage=IngestStage.FAILED,
        )
        store.save()

        wiki_c_meta = {
            "title": "Topic C",
            "slug": "topic-c",
            "type": "concept",
        }
        wiki_c_body = (
            "# Topic C\n\n"
            "## Synthesis\n\n"
            "Synthesis for C.\n\n"
            "## Sources\n\n"
            "## Open questions\n"
            "-\n\n"
            "## Related\n"
            "-\n"
        )
        write_atomic(
            cfg.brain_root / "90-wiki" / "topic-c.md",
            dump_frontmatter(wiki_c_meta, wiki_c_body),
        )

        client = FakeClient()
        plan = await rebuild_from_sources(cfg, client, dry_run=False)

        # Dropped topic's page should be removed
        assert not (cfg.brain_root / "90-wiki" / "topic-c.md").exists()

        # Other pages still exist
        assert (cfg.brain_root / "90-wiki" / "topic-a.md").exists()
        assert (cfg.brain_root / "90-wiki" / "topic-b.md").exists()

        # Plan counters reflect the dropped topic
        assert plan.topics_before == 3
        assert plan.topics_after == 2

    # ------------------------------------------------------------------
    # Test: daemon refusal
    # ------------------------------------------------------------------

    def test_rebuild_refuses_when_daemon_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI refuses when the daemon health endpoint is reachable."""
        from typer.testing import CliRunner

        from second_brain.cli import app

        cfg = _make_cfg(tmp_path)

        # Point load_config to our fake cfg
        monkeypatch.setattr("second_brain.cli.load_config", lambda: cfg)

        # Mock httpx.AsyncClient so the daemon health check succeeds
        with patch("httpx.AsyncClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_cls.return_value = mock_instance
            mock_instance.__aenter__.return_value = mock_instance

            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_instance.get.return_value = mock_response

            runner = CliRunner()
            result = runner.invoke(app, ["rebuild", "--from-sources"])

        assert result.exit_code != 0, (
            f"Expected non-zero exit code, got {result.exit_code}. "
            f"Output: {result.output}"
        )
        assert "Daemon is running on port" in result.output

    # ------------------------------------------------------------------
    # Test: idempotency
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rebuild_idempotent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two rebuilds from the same seed produce identical output."""
        monkeypatch.setattr("second_brain.rebuild.Embedder", FakeEmbedder)
        cfg = _make_cfg(tmp_path)
        _seed_sources_and_wiki(cfg)

        client_a = FakeClient()
        plan1 = await rebuild_from_sources(cfg, client_a, dry_run=False)
        state1 = (cfg.brain_root / ".brain" / "state.json").read_text(
            encoding="utf-8"
        )
        wiki1 = (cfg.brain_root / "90-wiki" / "topic-a.md").read_text(
            encoding="utf-8"
        )

        client_b = FakeClient()
        plan2 = await rebuild_from_sources(cfg, client_b, dry_run=False)
        state2 = (cfg.brain_root / ".brain" / "state.json").read_text(
            encoding="utf-8"
        )
        wiki2 = (cfg.brain_root / "90-wiki" / "topic-a.md").read_text(
            encoding="utf-8"
        )

        # Topic counts match
        assert plan1.topics_after == plan2.topics_after
        assert plan1.sources_seen == plan2.sources_seen

        # File content identical
        assert state1 == state2, "state.json differs between runs"
        assert wiki1 == wiki2, "wiki page content differs between runs"
