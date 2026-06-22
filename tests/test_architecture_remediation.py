"""Regression tests for architecture-audit remediations."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from second_brain.daemon.normalize import normalize_text
from second_brain.frontmatter import dump_frontmatter, split_frontmatter
from second_brain.models import SourceState
from second_brain.state import BrainStateStore, reconcile_filesystem


class FakeClient:
    async def close(self) -> None:
        pass


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        brain_root=tmp_path,
        types=SimpleNamespace(
            text=["txt", "md"],
            code=[],
            structured=[],
            vision=[],
            pdf=[],
            office=[],
            web=[],
            ebook=[],
            audio=[],
            video=[],
        ),
        ingestion=SimpleNamespace(max_audio_minutes=60),
        models=SimpleNamespace(stt="stt"),
    )


async def test_source_frontmatter_has_schema_version(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    inbox = tmp_path / "00-inbox" / "note.txt"
    inbox.parent.mkdir(parents=True)
    inbox.write_text("hello", encoding="utf-8")

    path, _body = await normalize_text(
        inbox,
        "note",
        "abc",
        "2026-06-22T00:00:00Z",
        "text",
        cfg,
        FakeClient(),
    )

    meta, _ = split_frontmatter(path.read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1


def test_reconcile_restores_state_from_filesystem(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    source_dir = tmp_path / "50-sources"
    wiki_dir = tmp_path / "90-wiki"
    source_dir.mkdir(parents=True)
    wiki_dir.mkdir(parents=True)

    source_text = dump_frontmatter(
        {
            "schema_version": 1,
            "source": "00-inbox/note.txt",
            "type": "text",
            "ingested": "2026-06-22T00:00:00Z",
            "sha256": "abc",
            "tokens": 3,
            "topics": ["topic-a"],
        },
        "# Note\n",
    )
    (source_dir / "note.md").write_text(source_text, encoding="utf-8")

    wiki_text = dump_frontmatter(
        {
            "schema_version": 1,
            "title": "Topic A",
            "slug": "topic-a",
            "type": "concept",
            "created": "2026-06-22",
            "updated": "2026-06-22",
            "confidence": 0.8,
            "related": [],
        },
        "# Topic A\n",
    )
    (wiki_dir / "topic-a.md").write_text(wiki_text, encoding="utf-8")

    store = BrainStateStore.load(cfg)
    changed = reconcile_filesystem(cfg, store)

    assert changed is True
    assert "note" in store.state.sources
    assert "topic-a" in store.state.topics
    assert store.state.topics["topic-a"].sources == ["note"]


def test_reconcile_removes_missing_derived_records(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    store = BrainStateStore.load(cfg)
    store.state.sources["missing"] = SourceState(sha256="abc", raw="00-inbox/missing.txt")
    store.ensure_topic("missing-topic", "Missing Topic")
    store.save()

    changed = reconcile_filesystem(cfg, store)

    assert changed is True
    assert store.state.sources == {}
    assert store.state.topics == {}


def test_cli_query_fallback_does_not_open_vector_store() -> None:
    import second_brain.cli as cli

    source = inspect.getsource(cli.search)
    assert "VectorStore(" not in source
    assert "Semantic search unavailable" in source


def test_pyproject_declares_literal_chat_stack() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"pydantic-ai"' in text
    assert '"fastapi>=0.135"' in text
