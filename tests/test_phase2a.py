"""Phase 2A tests — sqlite-vec store + Embedder cache, no API calls needed.

All vectors use dim=8 (tiny).  Tests use real sqlite-vec on the local
database file — no mocking or network.
"""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from second_brain.vectors.embed import Embedder
from second_brain.vectors.store import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    VectorStore,
    chunk_text,
)

DIM = 8


# ── helpers ──────────────────────────────────────────────────────────────────


def _vec(seed: int) -> list[float]:
    """Deterministic 8-dim vector for testing."""
    return [float(seed + i) for i in range(DIM)]


# ── TestVectorStore ──────────────────────────────────────────────────────────


class TestVectorStore:
    """Real sqlite-vec backed VectorStore tests."""

    # ── 1. Schema + model registry ───────────────────────────────────────

    def test_schema_and_model_registry(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"

        store = VectorStore(db_path, model="model-a", dim=DIM)
        assert store.active_dim() == DIM
        store.close()

        store2 = VectorStore(db_path, model="model-b", dim=16)
        assert store2.active_dim() == 16

        rows = store2.db.execute(
            "SELECT model, active FROM model_registry ORDER BY id"
        ).fetchall()
        assert dict(rows[0]) == {"model": "model-a", "active": 0}
        assert dict(rows[1]) == {"model": "model-b", "active": 1}

        # Tables exist
        tables = {
            r["name"]
            for r in store2.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' OR type='virtual'"
            ).fetchall()
        }
        expected = {
            "model_registry",
            "source_chunks_vec",
            "source_chunks_meta",
            "source_chunks_fts",
            "topic_centroids_vec",
            "topic_centroids_meta",
            "topic_members",
            "vec_tombstones",
        }
        assert expected.issubset(tables)
        store2.close()

    # ── 2. chunk_text ────────────────────────────────────────────────────

    def test_chunk_text_short(self) -> None:
        assert chunk_text("short text") == ["short text"]

    def test_chunk_text_empty(self) -> None:
        assert chunk_text("") == []

    def test_chunk_text_overlap(self) -> None:
        # 2x-size text; sliding window step = size - overlap → 3 windows.
        text = "A" * CHUNK_SIZE_CHARS + "B" * CHUNK_SIZE_CHARS
        chunks = chunk_text(text)
        assert len(chunks) == 3  # noqa: PLR2004
        assert all(len(c) <= CHUNK_SIZE_CHARS for c in chunks)
        assert chunks[0] == "A" * CHUNK_SIZE_CHARS  # first window [0:size]
        # each subsequent window overlaps the previous by `overlap` chars
        for i in range(len(chunks) - 1):
            assert chunks[i + 1][:CHUNK_OVERLAP_CHARS] == chunks[i][-CHUNK_OVERLAP_CHARS:]
        # the final window reaches the end of the text
        assert chunks[-1] == text[len(text) - len(chunks[-1]) :]

    def test_chunk_text_lengths(self) -> None:
        text = "hello world " * 500  # ~6000 chars
        chunks = chunk_text(text)
        for c in chunks:
            assert len(c) <= CHUNK_SIZE_CHARS

    # ── 3. upsert_source_chunks + vector_search_chunks ───────────────────

    def test_upsert_and_vector_search(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path, model="test", dim=DIM)

        v1 = _vec(0)  # [0,1,2,3,4,5,6,7]
        v2 = _vec(10)  # [10,11,12,13,14,15,16,17]
        rowids = store.upsert_source_chunks(
            source_id="src1",
            topic_slug="topic-a",
            chunks=[("first chunk", v1), ("second chunk", v2)],
        )
        assert len(rowids) == 2  # noqa: PLR2004

        # Search with v1 → first rowid should be top
        results = store.vector_search_chunks(v1, k=5)
        assert len(results) >= 1
        top_rowid, top_sim = results[0]
        assert top_rowid == rowids[0]
        assert top_sim == pytest.approx(1.0, abs=1e-5)

        # Verify metadata
        meta = store.get_chunk(rowids[0])
        assert meta is not None
        assert meta["source_id"] == "src1"
        assert meta["text"] == "first chunk"
        assert meta["chunk_idx"] == 0

        store.close()

    # ── 4. centroid ──────────────────────────────────────────────────────

    def test_centroid(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path, model="test", dim=DIM)

        v_a = _vec(0)
        v_b = _vec(1)
        store.upsert_source_chunks(
            source_id="src-a", topic_slug="t",
            chunks=[("a", v_a)],
        )
        store.upsert_source_chunks(
            source_id="src-b", topic_slug="t",
            chunks=[("b", v_b)],
        )
        store.add_topic_member("t", "src-a")
        store.add_topic_member("t", "src-b")

        centroid = store.recompute_centroid("t")
        assert centroid is not None

        expected = np.mean([np.array(v_a), np.array(v_b)], axis=0).tolist()
        assert centroid == pytest.approx(expected, abs=1e-5)

        # vector_search_topics with centroid → topic slug top
        results = store.vector_search_topics(centroid, k=5)
        assert results[0][0] == "t"
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

        store.close()

    # ── 5. best_topic_for_vector ─────────────────────────────────────────

    def test_best_topic_for_vector(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path, model="test", dim=DIM)

        # No centroids yet
        assert store.best_topic_for_vector(_vec(0)) is None

        # Add one
        v = _vec(0)
        store.upsert_source_chunks(
            source_id="src", topic_slug="t",
            chunks=[("x", v)],
        )
        store.add_topic_member("t", "src")
        centroid = store.recompute_centroid("t")
        assert centroid is not None
        best = store.best_topic_for_vector(v)
        assert best is not None
        assert best[0] == "t"
        assert best[1] == pytest.approx(1.0, abs=1e-5)

        store.close()

    # ── 6. fts_search ────────────────────────────────────────────────────

    def test_fts_search(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path, model="test", dim=DIM)

        v = _vec(0)
        rowids = store.upsert_source_chunks(
            source_id="src-fts", topic_slug="fts",
            chunks=[("walrus whiskers are magnificent", v)],
        )

        results = store.fts_search("walrus", k=5)
        assert any(r[0] == rowids[0] for r in results)

        store.close()

    # ── 7. tombstone_source ──────────────────────────────────────────────

    def test_tombstone_source(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path, model="test", dim=DIM)

        v = _vec(0)
        rowids = store.upsert_source_chunks(
            source_id="src-del", topic_slug="del",
            chunks=[("delete me", v)],
        )
        store.tombstone_source("src-del")

        # Vector search no longer finds it
        results = store.vector_search_chunks(v, k=5)
        assert rowids[0] not in {r[0] for r in results}

        # Tombstone recorded
        tombstones = store.db.execute(
            "SELECT source_id, table_name FROM vec_tombstones"
        ).fetchall()
        assert len(tombstones) >= 1
        assert tombstones[0]["source_id"] == "src-del"

        store.close()

    # ── 8. read_only ─────────────────────────────────────────────────────

    def test_read_only(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"

        # 1. Create writable store with tables
        write_store = VectorStore(db_path, model="test", dim=DIM)
        write_store.close()

        # 2. Snapshot schema before opening read-only
        conn = sqlite3.connect(str(db_path))
        before = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' OR type='virtual'"
            ).fetchall()
        }
        conn.close()

        # 3. Open read-only and close
        ro = VectorStore(db_path, model="test", dim=DIM, read_only=True)
        ro.close()

        # 4. Schema unchanged
        conn = sqlite3.connect(str(db_path))
        after = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' OR type='virtual'"
            ).fetchall()
        }
        conn.close()
        assert before == after

        # 5. Open read-only again and query meta table
        ro = VectorStore(db_path, model="test", dim=DIM, read_only=True)
        rows = ro.db.execute(
            "SELECT model FROM model_registry WHERE active = 1"
        ).fetchall()
        assert len(rows) > 0
        assert rows[0]["model"] == "test"

        # 6. Vector MATCH raises (extension not loaded)
        packed = struct.pack(f"{DIM}f", *[1.0] * DIM)
        with pytest.raises(sqlite3.OperationalError):
            ro.db.execute(
                "SELECT rowid, distance FROM source_chunks_vec "
                "WHERE embedding MATCH ? AND k = 10 ORDER BY distance",
                (packed,),
            ).fetchall()

        ro.close()


# ── TestEmbedder ─────────────────────────────────────────────────────────────


class TestEmbedder:
    """Embedder cache tests — no real API calls."""

    async def test_embedder_cache(self, tmp_path: Path) -> None:
        call_count = 0

        class FakeClient:
            async def embedding(self, model: str, input: str) -> list[float]:
                nonlocal call_count
                call_count += 1
                return [float(i) for i in range(8)]

        client = FakeClient()
        cfg = SimpleNamespace(
            brain_root=tmp_path,
            models=SimpleNamespace(embedding="test-model"),
        )
        embedder = Embedder(client, cfg)

        # First call → hits the fake
        vec1 = await embedder.embed_one("hello cache")
        assert vec1 == [float(i) for i in range(8)]
        assert call_count == 1

        # Second call with same text → cache hit, fake not called
        vec2 = await embedder.embed_one("hello cache")
        assert vec2 == [float(i) for i in range(8)]
        assert call_count == 1, "expected cache hit — fake should not be called"

        # Different text → cache miss, fake called
        vec3 = await embedder.embed_one("different text")
        assert vec3 == [float(i) for i in range(8)]
        assert call_count == 2  # noqa: PLR2004

        # Verify cache file exists
        cache_dir = tmp_path / ".brain" / "cache" / "embeddings"
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 2  # noqa: PLR2004

        # Cache content is valid JSON
        cached_vec = json.loads(cache_files[0].read_text(encoding="utf-8"))
        assert isinstance(cached_vec, list)
        assert len(cached_vec) == 8  # noqa: PLR2004

    async def test_embed_texts(self, tmp_path: Path) -> None:
        """embed_texts returns results in order, caching per text."""
        call_count = 0

        class FakeClient:
            async def embedding(self, model: str, input: str) -> list[float]:
                nonlocal call_count
                call_count += 1
                return [float(ord(input[0]) + i) for i in range(8)]

        client = FakeClient()
        cfg = SimpleNamespace(
            brain_root=tmp_path,
            models=SimpleNamespace(embedding="test-model"),
        )
        embedder = Embedder(client, cfg)

        texts = ["aaaa", "bbbb", "aaaa"]  # third is duplicate
        results = await embedder.embed_texts(texts)
        assert len(results) == 3  # noqa: PLR2004
        assert results[0] == results[2]  # same text → same vector
        assert call_count == 2  # noqa: PLR2004 — only "aaaa" and "bbbb" hit API

    async def test_embed_query_alias(self, tmp_path: Path) -> None:
        """embed_query is an alias of embed_one."""
        call_count = 0

        class FakeClient:
            async def embedding(self, model: str, input: str) -> list[float]:
                nonlocal call_count
                call_count += 1
                return [1.0] * 8

        client = FakeClient()
        cfg = SimpleNamespace(
            brain_root=tmp_path,
            models=SimpleNamespace(embedding="test-model"),
        )
        embedder = Embedder(client, cfg)

        vec = await embedder.embed_query("search query")
        assert vec == [1.0] * 8
        assert call_count == 1
