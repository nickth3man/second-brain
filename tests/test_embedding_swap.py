"""Blue/green embedding swap tests (§12.6)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from second_brain.frontmatter import dump_frontmatter
from second_brain.models import SourceState
from second_brain.state import BrainStateStore
from second_brain.vectors.swap import recover_embedding_swap, swap_embeddings


class FakeClient:
    async def embedding(self, model: str, input: str | list[str]) -> list[float]:  # noqa: ARG002
        return [1.0, 0.0, 0.0]


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        brain_root=tmp_path,
        models=SimpleNamespace(embedding="old-model"),
    )


def _seed_source(tmp_path: Path, store: BrainStateStore) -> None:
    source_dir = tmp_path / "50-sources"
    source_dir.mkdir(parents=True)
    (source_dir / "note.md").write_text(
        dump_frontmatter(
            {
                "source": "00-inbox/note.txt",
                "type": "text",
                "ingested": "2026-06-22T00:00:00Z",
                "sha256": "abc",
                "tokens": 3,
                "topics": ["topic-a"],
            },
            "# Note\nbody",
        ),
        encoding="utf-8",
    )
    store.state.sources["note"] = SourceState(
        sha256="abc",
        raw="00-inbox/note.txt",
        topics=["topic-a"],
        embedding_model="old-model",
    )
    store.ensure_topic("topic-a", "Topic A")
    store.state.embedding_model = "old-model"
    store.state.embedding_dim = 3
    store.save()


async def test_embedding_swap_success_updates_state(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    store = BrainStateStore.load(cfg)
    _seed_source(tmp_path, store)

    result = await swap_embeddings(
        cfg,
        store,
        FakeClient(),
        new_model="new-model",
        dim=3,
    )

    assert result.status == "swapped"
    assert (tmp_path / ".brain" / "embeddings.db").exists()
    assert not (tmp_path / ".brain" / "embeddings.new").exists()
    assert store.state.embedding_model == "new-model"
    assert store.state.sources["note"].embedding_model == "new-model"
    assert store.state.embedding_swap_in_progress is None


async def test_embedding_swap_rolls_back_when_quality_worse(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    store = BrainStateStore.load(cfg)
    _seed_source(tmp_path, store)
    active = tmp_path / ".brain" / "embeddings.db"
    active.write_text("old", encoding="utf-8")

    async def score(path: Path) -> float:
        return 1.0 if path == active else 0.50

    result = await swap_embeddings(
        cfg,
        store,
        FakeClient(),
        new_model="bad-model",
        dim=3,
        score_fn=score,
    )

    assert result.status == "rolled_back"
    assert active.read_text(encoding="utf-8") == "old"
    assert store.state.embedding_model == "old-model"
    assert store.state.embedding_swap_in_progress is None


def test_embedding_swap_recovery_restores_rollback_db(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    store = BrainStateStore.load(cfg)
    rollback = tmp_path / ".brain" / "embeddings.rollback"
    rollback.mkdir(parents=True)
    (rollback / "embeddings.db").write_text("old", encoding="utf-8")
    (tmp_path / ".brain" / "embeddings.new").mkdir()
    store.state.embedding_swap_in_progress = {"stage": "swapping"}
    store.save()

    changed = recover_embedding_swap(cfg, store)

    assert changed is True
    assert (tmp_path / ".brain" / "embeddings.db").read_text(encoding="utf-8") == "old"
    assert not (tmp_path / ".brain" / "embeddings.new").exists()
    assert store.state.embedding_swap_in_progress is None
