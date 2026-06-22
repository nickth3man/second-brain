"""Phase 2 follow-up tests — ``find_near_duplicates_for_source`` unit test.

Verifies the two-tier source-embedding strategy:

1. ``vec_store.source_centroid(sid)`` is preferred when non-None.
2. ``embedder.embed_one(text)`` is the fallback for legacy sources with
   no chunks in the store.

No network, no API.  Uses real sqlite-vec with dim=8 + deterministic
fake embedder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from second_brain.compact.dedup import (
    _SOURCE_EMBEDDING_CACHE,
    find_near_duplicates_for_source,
)
from second_brain.config import TypesCfg
from second_brain.models import SourceState
from second_brain.state import BrainStateStore
from second_brain.vectors.store import VectorStore

DIM = 8
DEDUP_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Stubs (same pattern as test_phase2b / test_phase4)
# ---------------------------------------------------------------------------


@dataclass
class _FakeExtraction:
    primary_model: str = "test-model"
    repair_model: str = "test-repair-model"
    enable_healing: bool = False
    deadletter_dir: str = ".brain/deadletter"
    max_attempts: int = 2
    require_parameters: bool = False
    confidence_floor: float = 0.6
    quarantine_dir: str = ".brain/quarantine"


@dataclass
class _FakeModels:
    text: str = "test-model"
    vision: str = "test-vision"
    embedding: str = "test-embed"
    stt: str = "test-stt"
    chat: str = "test-chat"
    judge: str = "test-judge"


@dataclass
class _FakeIngestion:
    merge_threshold: float = 0.7
    pdf_dpi: int = 200
    pdf_image_format: str = "png"
    pdf_alpha: bool = False
    vision_max_images_per_request: int = 8
    vision_max_edge_px: int = 2048
    max_audio_minutes: int = 120


@dataclass
class _FakeCfg:
    brain_root: Path
    types: TypesCfg
    extraction: _FakeExtraction = field(default_factory=_FakeExtraction)
    models: _FakeModels = field(default_factory=_FakeModels)
    ingestion: _FakeIngestion = field(default_factory=_FakeIngestion)


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
    (cfg.brain_root / "50-sources").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / ".brain").mkdir(parents=True, exist_ok=True)


class TrackingEmbedder:
    """Deterministic embedder that records every ``embed_one`` call.

    Used to assert that the preferred tier (``source_centroid``) skips
    re-embedding when it returns a non-None vector.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.embed_calls: list[str] = []
        self._vectors: dict[str, list[float]] = {}

    def set_vector(self, text: str, vec: list[float]) -> None:
        self._vectors[text] = vec

    async def embed_one(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        if text in self._vectors:
            return self._vectors[text]
        # Default deterministic vector far from anything interesting.
        return [0.0] * self.dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_one(t) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        return await self.embed_one(query)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFindNearDuplicatesForSource:
    """Unit tests for the two-tier ``find_near_duplicates_for_source``."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> None:
        """Reset the module-level embedding cache between tests."""
        _SOURCE_EMBEDDING_CACHE.clear()
        yield
        _SOURCE_EMBEDDING_CACHE.clear()

    @pytest.mark.asyncio
    async def test_prefers_centroid_and_skips_embedder(
        self, tmp_path: Path
    ) -> None:
        """``source_centroid`` hit -> ``embedder.embed_one`` NOT called."""
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)

        # Two existing sources — both have chunks in the vec store, so
        # their centroids are available and the embedder is never used.
        store.state.sources["src-a"] = SourceState(
            sha256="aaa", raw="src-a.md", topics=[]
        )
        store.state.sources["src-b"] = SourceState(
            sha256="bbb", raw="src-b.md", topics=[]
        )
        # Write the fallback source files anyway — they should NOT be read
        # because the centroid tier succeeds.
        (cfg.brain_root / "50-sources" / "src-a.md").write_text(
            "fallback A content", encoding="utf-8"
        )
        (cfg.brain_root / "50-sources" / "src-b.md").write_text(
            "fallback B content", encoding="utf-8"
        )

        embedder = TrackingEmbedder(DIM)

        db_path = tmp_path / "vecs.db"
        vec_store = VectorStore(db_path, model="test", dim=DIM)
        try:
            # src-a centroid is identical to the new embedding (sim == 1.0)
            near_vec = [1.0] + [0.0] * (DIM - 1)
            vec_store.upsert_source_chunks(
                "src-a", "topic-a", [("a text", list(near_vec))]
            )
            # src-b centroid is orthogonal (sim == 0.0) -> filtered out
            far_vec = [0.0, 1.0] + [0.0] * (DIM - 2)
            vec_store.upsert_source_chunks(
                "src-b", "topic-b", [("b text", list(far_vec))]
            )

            hits = await find_near_duplicates_for_source(
                cfg, store, embedder, vec_store, "src-new",
                list(near_vec), threshold=DEDUP_THRESHOLD,
            )

            # src-a is a hit, src-b is filtered out (sim 0.0 < 0.95)
            hit_ids = [sid for sid, _ in hits]
            assert "src-a" in hit_ids
            assert "src-b" not in hit_ids

            # Critical: embedder was never called because centroids hit.
            assert embedder.embed_calls == []
        finally:
            vec_store.close()

    @pytest.mark.asyncio
    async def test_falls_back_to_embedder_for_legacy_sources(
        self, tmp_path: Path
    ) -> None:
        """``source_centroid`` returns None -> embedder is used."""
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)

        # src-legacy has NO chunks in the vec store (so source_centroid
        # returns None) but DOES have a 50-sources file.
        store.state.sources["src-legacy"] = SourceState(
            sha256="abc", raw="src-legacy.md", topics=[]
        )
        legacy_body = "Legacy content for embedding"
        legacy_path = cfg.brain_root / "50-sources" / "src-legacy.md"
        legacy_path.write_text(legacy_body, encoding="utf-8")

        embedder = TrackingEmbedder(DIM)
        # Configure the fallback vector to match the new embedding.
        near_vec = [1.0] + [0.0] * (DIM - 1)
        embedder.set_vector(legacy_body, list(near_vec))

        db_path = tmp_path / "vecs-empty.db"
        vec_store = VectorStore(db_path, model="test", dim=DIM)
        try:
            hits = await find_near_duplicates_for_source(
                cfg, store, embedder, vec_store, "src-new",
                list(near_vec), threshold=DEDUP_THRESHOLD,
            )

            assert [sid for sid, _ in hits] == ["src-legacy"]
            # Confirm the fallback actually fired.
            assert legacy_body in embedder.embed_calls
        finally:
            vec_store.close()

    @pytest.mark.asyncio
    async def test_results_sorted_descending(self, tmp_path: Path) -> None:
        """Multiple hits are returned sorted by similarity descending."""
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)

        for sid in ("src-hi", "src-mid", "src-low"):
            store.state.sources[sid] = SourceState(
                sha256=sid, raw=f"{sid}.md", topics=[]
            )

        embedder = TrackingEmbedder(DIM)

        db_path = tmp_path / "vecs-sorted.db"
        vec_store = VectorStore(db_path, model="test", dim=DIM)
        try:
            new_vec = [1.0] + [0.0] * (DIM - 1)
            # Three centroids of decreasing similarity to new_vec:
            # cosine(1,0,..) vs (cos θ, sin θ, 0,..) is just cos θ.
            vec_store.upsert_source_chunks(
                "src-hi", "t-hi",
                [("hi", [1.0] + [0.0] * (DIM - 1))],  # sim 1.0
            )
            vec_store.upsert_source_chunks(
                "src-mid", "t-mid",
                [("mid", [0.99, 0.141] + [0.0] * (DIM - 2))],  # sim ~0.99
            )
            vec_store.upsert_source_chunks(
                "src-low", "t-low",
                [("low", [0.96, 0.28] + [0.0] * (DIM - 2))],  # sim ~0.96
            )

            hits = await find_near_duplicates_for_source(
                cfg, store, embedder, vec_store, "src-new",
                list(new_vec), threshold=DEDUP_THRESHOLD,
            )

            assert [sid for sid, _ in hits] == ["src-hi", "src-mid", "src-low"]
            sims = [sim for _, sim in hits]
            assert sims == sorted(sims, reverse=True)
        finally:
            vec_store.close()

    @pytest.mark.asyncio
    async def test_threshold_filtering(self, tmp_path: Path) -> None:
        """Similarity below the threshold is excluded."""
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)

        store.state.sources["src-below"] = SourceState(
            sha256="x", raw="src-below.md", topics=[]
        )

        embedder = TrackingEmbedder(DIM)
        db_path = tmp_path / "vecs-thresh.db"
        vec_store = VectorStore(db_path, model="test", dim=DIM)
        try:
            new_vec = [1.0] + [0.0] * (DIM - 1)
            # cos(45°) ~= 0.707 — well below 0.95 threshold
            vec_store.upsert_source_chunks(
                "src-below", "t",
                [("x", [0.7071, 0.7071] + [0.0] * (DIM - 2))],
            )

            hits = await find_near_duplicates_for_source(
                cfg, store, embedder, vec_store, "src-new",
                list(new_vec), threshold=DEDUP_THRESHOLD,
            )
            assert hits == []
        finally:
            vec_store.close()
