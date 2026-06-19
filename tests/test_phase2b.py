"""Phase 2B tests — retrieval, EmbeddingLinker, pipeline wiring, no network.

All vector tests use dim=8 (tiny).  Uses real sqlite-vec + deterministic
FakeEmbedder — no API calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from second_brain.config import TypesCfg
from second_brain.daemon.index import DebouncedIndex
from second_brain.daemon.linker import EmbeddingLinker, LinkContext
from second_brain.daemon.pipeline import ingest_file
from second_brain.models import IngestStage, LinkDecision, TopicAction
from second_brain.slug import slugify
from second_brain.state import BrainStateStore
from second_brain.vectors.retrieval import (
    SearchHit,
    reciprocal_rank_fusion,
    search_brain,
    topic_match,
)
from second_brain.vectors.store import VectorStore

DIM = 8


# -- stubs (same pattern as test_phase1b) -------------------------------------


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
    (cfg.brain_root / "00-inbox").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / "50-sources").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / "90-wiki").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / ".brain").mkdir(parents=True, exist_ok=True)


def _write_inbox(cfg: _FakeCfg, name: str, content: str) -> Path:
    p = cfg.brain_root / "00-inbox" / name
    p.write_text(content, encoding="utf-8")
    return p


class FakeClient:
    """Fake OpenRouter client returning configurable payloads."""

    def __init__(self, payload: dict | None = None):
        self.payload = payload or {
            "tldr": "Test summary.",
            "topics": [
                {
                    "name": "Test Topic",
                    "action": "new",
                    "target_slug": "",
                    "confidence": 0.9,
                    "merged_section": "Test merged content.",
                }
            ],
        }

    async def chat_completion(self, *args, **kwargs) -> dict:  # noqa: ARG002
        return {
            "choices": [{"message": {"content": json.dumps(self.payload)}}]
        }

    async def close(self) -> None:
        pass


class FakeEmbedder:
    """Deterministic embedder for testing — no API calls.

    Supports ``set_vector(text, vec)`` to control what a specific input
    returns.  Unregistered texts get a default deterministic vector.
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self._vectors: dict[str, list[float]] = {}
        self.query_vec: list[float] | None = None

    def set_vector(self, text: str, vec: list[float]) -> None:
        self._vectors[text] = vec

    async def embed_one(self, text: str) -> list[float]:
        if text in self._vectors:
            return self._vectors[text]
        return [0.1 * (i + 1) for i in range(self.dim)]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_one(t) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        if self.query_vec is not None:
            return self.query_vec
        return await self.embed_one(query)


# -- Helper factory for a vector in R^DIM -------------------------------------


def _vec(first: float, rest: float = 0.0) -> list[float]:
    """Deterministic DIM-dimensional vector."""
    return [first] + [rest] * (DIM - 1)


# -- TestRetrieval ------------------------------------------------------------


class TestRetrieval:
    """Direct unit tests for the retrieval module."""

    def test_rrf_agreement_boost(self) -> None:
        """A rowid present in both lists ranks above one in only one list."""
        vec_hits = [(1, 0.9), (2, 0.8)]
        fts_hits = [(1, 0.2), (3, 0.3)]

        fused = reciprocal_rank_fusion(vec_hits, fts_hits, k=60)
        scores = dict(fused)

        # Rowid 1 appears in both -> highest RRF score
        assert fused[0][0] == 1
        assert scores[1] > scores[2]
        assert scores[1] > scores[3]

    def test_rrf_ordering(self) -> None:
        """RRF results are sorted by fused score descending."""
        vec_hits = [(10, 0.9), (20, 0.8)]
        fts_hits = [(20, 0.1), (30, 0.2)]

        fused = reciprocal_rank_fusion(vec_hits, fts_hits, k=60)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_empty_lists(self) -> None:
        """RRF handles empty input lists gracefully."""
        assert reciprocal_rank_fusion([], []) == []
        assert reciprocal_rank_fusion([(1, 0.9)], []) == [(1, 1.0 / 61.0)]
        assert reciprocal_rank_fusion([], [(1, 0.2)]) == [(1, 1.0 / 61.0)]


# -- TestEmbeddingLinker ------------------------------------------------------


class TestEmbeddingLinker:
    """EmbeddingLinker MATCH vs NEW decisions."""

    @pytest.mark.asyncio
    async def test_match_when_similar(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path, model="test", dim=DIM)
        embedder = FakeEmbedder(DIM)

        # Seed one topic centroid
        centroid = _vec(1.0)
        store.upsert_source_chunks(
            "seed-src", "existing-topic",
            [("seed", centroid)],
        )
        store.add_topic_member("existing-topic", "seed-src")
        store.recompute_centroid("existing-topic")

        # FakeEmbedder returns a vector near centroid for the matching name
        embedder.set_vector("Existing Topic", centroid)

        linker = EmbeddingLinker(embedder, store, threshold=0.70)

        ctx = LinkContext(brain_store=type("S", (), {"state": type("", (), {"topics": {}})()})())  # noqa: E501
        decisions_in = [
            LinkDecision(
                name="Existing Topic",
                action=TopicAction.NEW,
                target_slug="",
                confidence=0.8,
                merged_section="Content.",
            )
        ]
        result = await linker.link(decisions_in, ctx)
        assert result[0].action == TopicAction.MATCH
        assert result[0].target_slug == "existing-topic"

    @pytest.mark.asyncio
    async def test_new_when_dissimilar(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path, model="test", dim=DIM)
        embedder = FakeEmbedder(DIM)

        # Seed one topic centroid
        centroid = _vec(1.0)
        store.upsert_source_chunks(
            "seed-src", "existing-topic",
            [("seed", centroid)],
        )
        store.add_topic_member("existing-topic", "seed-src")
        store.recompute_centroid("existing-topic")

        # FakeEmbedder returns a far vector for the new name (sim ~= 0)
        embedder.set_vector("New Concept", _vec(0.0))

        linker = EmbeddingLinker(embedder, store, threshold=0.70)

        ctx = LinkContext(brain_store=type("S", (), {"state": type("", (), {"topics": {}})()})())  # noqa: E501
        decisions_in = [
            LinkDecision(
                name="New Concept",
                action=TopicAction.NEW,
                target_slug="",
                confidence=0.6,
                merged_section="New content.",
            )
        ]
        result = await linker.link(decisions_in, ctx)
        assert result[0].action == TopicAction.NEW
        assert result[0].target_slug == slugify("New Concept")

    @pytest.mark.asyncio
    async def test_no_centroids_new(self, tmp_path: Path) -> None:
        """When no centroids exist, every candidate is NEW."""
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path, model="test", dim=DIM)
        embedder = FakeEmbedder(DIM)

        linker = EmbeddingLinker(embedder, store, threshold=0.70)
        ctx = LinkContext(brain_store=type("S", (), {"state": type("", (), {"topics": {}})()})())  # noqa: E501
        decisions_in = [
            LinkDecision(
                name="Anything",
                action=TopicAction.NEW,
                target_slug="",
                confidence=0.5,
                merged_section="Whatever.",
            )
        ]
        result = await linker.link(decisions_in, ctx)
        assert result[0].action == TopicAction.NEW
        assert result[0].target_slug == "anything"


# -- TestSearchBrain ----------------------------------------------------------


class TestSearchBrain:
    """Hybrid search integration tests."""

    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path, model="test", dim=DIM)
        embedder = FakeEmbedder(DIM)

        vec_a = _vec(1.0)  # matches query vector
        vec_b = _vec(0.0)  # dissimilar to query vector

        a_rowids = store.upsert_source_chunks(
            "source-a", "topic-a",
            [("apple pie recipe", vec_a)],
        )
        b_rowids = store.upsert_source_chunks(
            "source-b", "topic-b",
            [("oranges are citrus", vec_b)],
        )

        # embed_query returns vec_a -> vectorially matches source A
        embedder.query_vec = vec_a

        hits = await search_brain(
            "oranges", store, embedder, k=5, merge_k=20,
        )

        assert len(hits) >= 2
        hit_rowids = {h.rowid for h in hits}
        assert a_rowids[0] in hit_rowids, "A should appear (vector match)"
        assert b_rowids[0] in hit_rowids, "B should appear (FTS match)"

        for hit in hits:
            assert isinstance(hit, SearchHit)
            assert hit.source_id in ("source-a", "source-b")
            assert isinstance(hit.score, float)
            assert isinstance(hit.text, str)

    @pytest.mark.asyncio
    async def test_search_hit_fields(self, tmp_path: Path) -> None:
        """SearchHit has all expected fields populated."""
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path, model="test", dim=DIM)
        embedder = FakeEmbedder(DIM)

        vec = _vec(0.5)
        rowids = store.upsert_source_chunks(
            "src", "topic-slug",
            [("hello world text", vec)],
        )

        embedder.query_vec = vec
        hits = await search_brain("hello", store, embedder, k=5)

        assert len(hits) == 1
        hit = hits[0]
        assert hit.rowid == rowids[0]
        assert hit.source_id == "src"
        assert hit.topic_slug == "topic-slug"
        assert hit.text == "hello world text"
        assert hit.score > 0.0


# -- TestTopicMatch -----------------------------------------------------------


class TestTopicMatch:
    """topic_match edge cases."""

    def test_no_topics_returns_none(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path, model="test", dim=DIM)

        result = topic_match(_vec(1.0), store, threshold=0.7)
        assert result == (None, 0.0)

    def test_above_threshold(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path, model="test", dim=DIM)

        centroid = _vec(1.0)
        store.upsert_source_chunks("s", "t", [("x", centroid)])
        store.add_topic_member("t", "s")
        store.recompute_centroid("t")

        result = topic_match(centroid, store, threshold=0.7)
        assert result[0] == "t"
        assert result[1] >= 0.7

    def test_below_threshold(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path, model="test", dim=DIM)

        centroid = _vec(1.0)
        store.upsert_source_chunks("s", "t", [("x", centroid)])
        store.add_topic_member("t", "s")
        store.recompute_centroid("t")

        far_vec = _vec(0.0)
        result = topic_match(far_vec, store, threshold=0.7)
        assert result[0] is None
        assert result[1] < 0.7


# -- TestE2EPipelineWithEmbeddings --------------------------------------------


class TestE2EPipelineWithEmbeddings:
    """End-to-end pipeline run with real VectorStore + EmbeddingLinker."""

    @pytest.mark.asyncio
    async def test_ingest_with_embeddings(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)

        client = FakeClient(
            payload={
                "tldr": "A test topic about testing.",
                "topics": [
                    {
                        "name": "Test Topic",
                        "action": "new",
                        "target_slug": "",
                        "confidence": 0.9,
                        "merged_section": "Test Topic content.",
                    },
                ],
            }
        )

        embedder = FakeEmbedder(DIM)
        db_path = tmp_path / ".brain" / "embeddings.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        vec_store = VectorStore(db_path, model="test-embed", dim=DIM)

        linker = EmbeddingLinker(embedder, vec_store, threshold=0.7)
        state_store = BrainStateStore.load(cfg)
        index = DebouncedIndex(cfg, state_store)

        path = _write_inbox(
            cfg, "test-note.md", "# Test Note\n\nThis is test content."
        )
        stage = await ingest_file(
            path, cfg, state_store, client, linker, index,
            embedder=embedder, vec_store=vec_store,
        )
        assert stage == IngestStage.DONE

        # -- assertions ------------------------------------------------

        # 50-sources file written
        sources = list((cfg.brain_root / "50-sources").iterdir())
        assert len(sources) == 1

        # 90-wiki page created
        wiki_files = list((cfg.brain_root / "90-wiki").iterdir())
        assert len(wiki_files) == 1

        # state.json confirms DONE
        state_path = cfg.brain_root / ".brain" / "state.json"
        state_data = json.loads(state_path.read_text(encoding="utf-8"))
        src_id = list(state_data["sources"].keys())[0]

        # Vector store has topic member for the source
        assert vec_store.topic_member_count("test-topic") >= 1

        # Vector store has a centroid computed
        topic_vec = await embedder.embed_one("Test Topic")
        best = vec_store.best_topic_for_vector(topic_vec)
        assert best is not None, "centroid should have been computed"
        assert best[0] == "test-topic"

        # Source should appear among member source IDs
        member_ids = vec_store.member_source_ids("test-topic")
        assert src_id in member_ids

        vec_store.close()
