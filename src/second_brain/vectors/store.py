"""sqlite-vec vector store — daemon-owned writes with hybrid retrieval (§12.1, §12.6).

Schema
------
- **model_registry**: tracks active/inactive embedding models (blue/green swap).
- **source_chunks_vec**: ``vec0`` virtual table holding chunk embeddings.
- **source_chunks_meta**: per-chunk metadata (source, topic, text).
- **source_chunks_fts**: ``fts5`` virtual table for keyword search (hybrid).
- **topic_centroids_vec**: ``vec0`` virtual table for topic centroids.
- **topic_centroids_meta**: centroid metadata (slug, rowid, member count).
- **topic_members**: many-to-many mapping source -> topic.
- **vec_tombstones**: deletion audit log.

All write operations are exclusive to the daemon process.  Read-only
connections (used by the web UI) do NOT load the sqlite-vec extension and
can only query metadata tables — vector search is served via loopback HTTP
to the daemon (Phase 5).
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Any

import numpy as np
import sqlite_vec
import structlog

from second_brain.state import now_iso

log = structlog.get_logger(__name__)

DEFAULT_EMBED_DIM = 1536
CHUNK_SIZE_CHARS = 3200
CHUNK_OVERLAP_CHARS = 400


# -- chunking -----------------------------------------------------------------


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split *text* into overlapping character-window passages.

    Each chunk is at most *size* characters wide, sliding by
    ``size - overlap`` characters per step.  A text shorter than *size*
    produces a single chunk.  Empty input returns an empty list.

    Args:
        text: Input text to chunk.
        size: Maximum characters per chunk.
        overlap: Overlap in characters between consecutive chunks.

    Returns:
        A list of chunk strings in order.
    """
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = size - overlap
    if step <= 0:
        # overlap >= size — degenerate; return whole text as one chunk
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


# -- binary packing -----------------------------------------------------------


def _pack(vec: list[float], dim: int) -> bytes:
    """Pack a float vector into a binary BLOB for sqlite-vec."""
    return struct.pack(f"{dim}f", *vec)


def _unpack(blob: bytes, dim: int) -> list[float]:
    """Unpack a binary BLOB from sqlite-vec back into a float vector."""
    return list(struct.unpack(f"{dim}f", blob))


# -- VectorStore --------------------------------------------------------------


class VectorDimMismatchError(Exception):
    """Raised when an existing embeddings.db was built with a different dim.

    Swapping embedding models changes the vector dimension and the vec0 schema is
    fixed at creation. Resolve by deleting ``.brain/embeddings.db`` (and
    re-running ingest) or via a blue/green embedding swap (§12.6).
    """


class VectorStore:
    """sqlite-vec backed vector store with hybrid (vector + FTS) retrieval.

    **Thread-safety:** not guaranteed — the daemon owns all writes.  The web UI
    should open a **read-only** connection (does NOT load the extension) and use
    loopback HTTP to the daemon for vector search (Phase 5).

    Args:
        db_path: Path to the SQLite database file.
        model: Embedding model name (registered in ``model_registry``).
        dim: Embedding dimension.  Must match the model.
        read_only: If True, open the database in read-only mode without
            loading the sqlite-vec extension.  Only metadata tables are
            queryable.
    """

    def __init__(
        self,
        db_path: Path,
        model: str,
        dim: int = DEFAULT_EMBED_DIM,
        *,
        read_only: bool = False,
    ) -> None:
        self.model = model
        self.dim = dim
        self.read_only = read_only
        self.db_path = db_path

        if read_only:
            self.db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        else:
            self.db = sqlite3.connect(str(db_path))
            self.db.enable_load_extension(True)
            self.db.load_extension(sqlite_vec.loadable_path())
            self.db.row_factory = sqlite3.Row
            # Detect a dim mismatch with an existing db before mutating schema.
            has_reg = self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='model_registry'"
            ).fetchone()
            if has_reg:
                prior = self.db.execute(
                    "SELECT dim FROM model_registry WHERE active=1 "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if prior is not None and prior["dim"] != dim:
                    self.db.close()
                    raise VectorDimMismatchError(
                        f"embeddings.db stored dim={prior['dim']} but model "
                        f"{model!r} probes dim={dim}; delete .brain/embeddings.db "
                        f"or run a blue/green swap (§12.6)"
                    )
            self._init_schema()
            self._register_model(model, dim)

        self.db.row_factory = sqlite3.Row
        log.debug(
            "vector_store_opened",
            path=str(db_path),
            model=model,
            dim=dim,
            read_only=read_only,
        )

    # -- schema -----------------------------------------------------------

    def _init_schema(self) -> None:
        """Idempotently create all tables and virtual tables.

        Safe to call multiple times — all statements use ``IF NOT EXISTS``.
        The vec0 table dimension is baked into the schema at creation time.
        """
        self.db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS source_chunks_vec
            USING vec0(embedding float[{self.dim}])
        """)
        self.db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS topic_centroids_vec
            USING vec0(embedding float[{self.dim}])
        """)
        self.db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS source_chunks_fts
            USING fts5(text, source_id UNINDEXED, topic_slug UNINDEXED)
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS model_registry (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                model       TEXT NOT NULL,
                dim         INTEGER NOT NULL,
                active      INTEGER NOT NULL DEFAULT 0,
                registered  TEXT NOT NULL
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS source_chunks_meta (
                rowid       INTEGER PRIMARY KEY,
                source_id   TEXT NOT NULL,
                topic_slug  TEXT,
                chunk_idx   INTEGER NOT NULL,
                text        TEXT NOT NULL
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS topic_centroids_meta (
                slug            TEXT PRIMARY KEY,
                rowid           INTEGER NOT NULL,
                member_count    INTEGER NOT NULL DEFAULT 0,
                updated         TEXT NOT NULL
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS topic_members (
                source_id   TEXT NOT NULL,
                topic_slug  TEXT NOT NULL,
                PRIMARY KEY (source_id, topic_slug)
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS vec_tombstones (
                rowid       INTEGER NOT NULL,
                source_id   TEXT,
                table_name  TEXT NOT NULL,
                ts          TEXT NOT NULL
            )
        """)

    def _register_model(self, model: str, dim: int) -> None:
        """Register an embedding model as active, deactivating all others."""
        self.db.execute("UPDATE model_registry SET active = 0")
        self.db.execute(
            "INSERT INTO model_registry (model, dim, active, registered) "
            "VALUES (?, ?, 1, ?)",
            (model, dim, now_iso()),
        )
        self.db.commit()

    # -- model registry queries -------------------------------------------

    def active_dim(self) -> int:
        """Return the dimension of the currently active embedding model."""
        row = self.db.execute(
            "SELECT dim FROM model_registry WHERE active = 1 "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["dim"] if row else self.dim

    # -- chunk write ------------------------------------------------------

    def upsert_source_chunks(
        self,
        source_id: str,
        topic_slug: str,
        chunks: list[tuple[str, list[float]]],
    ) -> list[int]:
        """Insert or overwrite all chunks for a source in one transaction.

        Each entry in *chunks* is a ``(text, embedding_vector)`` pair.
        The vec0 rowid auto-generated by the INSERT is shared with the
        metadata and FTS tables.

        Returns:
            The list of assigned rowids in insertion order.
        """
        rowids: list[int] = []
        try:
            for i, (text, vec) in enumerate(chunks):
                cur = self.db.execute(
                    "INSERT INTO source_chunks_vec (embedding) VALUES (?)",
                    (_pack(vec, self.dim),),
                )
                rowid = cur.lastrowid
                assert rowid is not None, "vec0 INSERT did not return a rowid"
                rowids.append(rowid)
                self.db.execute(
                    "INSERT INTO source_chunks_meta "
                    "(rowid, source_id, topic_slug, chunk_idx, text) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (rowid, source_id, topic_slug, i, text),
                )
                self.db.execute(
                    "INSERT INTO source_chunks_fts "
                    "(text, source_id, topic_slug) VALUES (?, ?, ?)",
                    (text, source_id, topic_slug),
                )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        return rowids

    # -- topic membership -------------------------------------------------

    def add_topic_member(self, topic_slug: str, source_id: str) -> bool:
        """Register *source_id* as a member of the topic.

        Returns:
            True if the row was inserted, False if already present.
        """
        cur = self.db.execute(
            "INSERT OR IGNORE INTO topic_members (source_id, topic_slug) "
            "VALUES (?, ?)",
            (source_id, topic_slug),
        )
        self.db.commit()
        return cur.rowcount > 0

    def member_source_ids(self, topic_slug: str) -> list[str]:
        """Return all source IDs belonging to *topic_slug*."""
        rows = self.db.execute(
            "SELECT source_id FROM topic_members WHERE topic_slug = ?",
            (topic_slug,),
        ).fetchall()
        return [r["source_id"] for r in rows]

    # -- centroids --------------------------------------------------------

    def recompute_centroid(self, topic_slug: str) -> list[float] | None:
        """Recompute the centroid vector for *topic_slug* from member chunks.

        Averages all chunk embeddings of sources belonging to the topic.
        Updates the ``topic_centroids_vec`` and ``topic_centroids_meta``
        tables via :meth:`upsert_topic_centroid`.

        Returns:
            The centroid vector, or ``None`` if the topic has no members.
        """
        src_ids = self.member_source_ids(topic_slug)
        if not src_ids:
            return None

        placeholders = ",".join("?" for _ in src_ids)
        meta_rows = self.db.execute(
            f"SELECT rowid FROM source_chunks_meta "
            f"WHERE source_id IN ({placeholders})",
            src_ids,
        ).fetchall()

        vectors: list[np.ndarray] = []
        for mr in meta_rows:
            blob_row = self.db.execute(
                "SELECT embedding FROM source_chunks_vec WHERE rowid = ?",
                (mr["rowid"],),
            ).fetchone()
            if blob_row is not None:
                vec = _unpack(blob_row["embedding"], self.dim)
                vectors.append(np.array(vec, dtype=np.float32))

        if not vectors:
            return None

        mean_vec = np.mean(vectors, axis=0).tolist()
        self.upsert_topic_centroid(topic_slug, mean_vec, len(src_ids))
        return mean_vec

    def upsert_topic_centroid(
        self,
        slug: str,
        centroid: list[float],
        member_count: int,
    ) -> None:
        """Replace the centroid for *slug* with a new vector.

        Deletes the old vec0 entry and metadata row, then inserts fresh.
        """
        try:
            old = self.db.execute(
                "SELECT rowid FROM topic_centroids_meta WHERE slug = ?",
                (slug,),
            ).fetchone()
            if old is not None:
                self.db.execute(
                    "DELETE FROM topic_centroids_vec WHERE rowid = ?",
                    (old["rowid"],),
                )
                self.db.execute(
                    "DELETE FROM topic_centroids_meta WHERE slug = ?",
                    (slug,),
                )

            cur = self.db.execute(
                "INSERT INTO topic_centroids_vec (embedding) VALUES (?)",
                (_pack(centroid, self.dim),),
            )
            new_rowid = cur.lastrowid
            assert new_rowid is not None, "vec0 INSERT did not return a rowid"
            self.db.execute(
                "INSERT INTO topic_centroids_meta "
                "(slug, rowid, member_count, updated) VALUES (?, ?, ?, ?)",
                (slug, new_rowid, member_count, now_iso()),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    # -- chunk topic assignment -------------------------------------------

    def set_chunk_topic(self, source_id: str, topic_slug: str) -> None:
        """Set the topic for all chunks of a given source."""
        self.db.execute(
            "UPDATE source_chunks_meta SET topic_slug = ? WHERE source_id = ?",
            (topic_slug, source_id),
        )
        self.db.execute(
            "UPDATE source_chunks_fts SET topic_slug = ? WHERE source_id = ?",
            (topic_slug, source_id),
        )
        self.db.commit()

    # -- deletion / tombstoning -------------------------------------------

    def tombstone_source(self, source_id: str) -> None:
        """Remove all chunks for *source_id* and record tombstones.

        Deletes from vec0, meta, and FTS tables in a single transaction
        and logs every removed vec0 rowid to ``vec_tombstones`` for
        audit / recovery.
        """
        try:
            meta_rows = self.db.execute(
                "SELECT rowid FROM source_chunks_meta WHERE source_id = ?",
                (source_id,),
            ).fetchall()
            for mr in meta_rows:
                rid = mr["rowid"]
                self.db.execute(
                    "INSERT INTO vec_tombstones "
                    "(rowid, source_id, table_name, ts) VALUES (?, ?, ?, ?)",
                    (rid, source_id, "source_chunks_vec", now_iso()),
                )
                self.db.execute(
                    "DELETE FROM source_chunks_vec WHERE rowid = ?",
                    (rid,),
                )
                self.db.execute(
                    "DELETE FROM source_chunks_meta WHERE rowid = ?",
                    (rid,),
                )
            self.db.execute(
                "DELETE FROM source_chunks_fts WHERE source_id = ?",
                (source_id,),
            )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    # -- vector search ----------------------------------------------------

    def vector_search_chunks(
        self,
        query_vec: list[float],
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """ANN search over chunk embeddings.

        Returns:
            ``[(rowid, cosine_similarity), ...]`` sorted by relevance
            (highest similarity first).
        """
        packed = _pack(query_vec, self.dim)
        rows = self.db.execute(
            "SELECT rowid, distance FROM source_chunks_vec "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (packed, k),
        ).fetchall()
        return [(r["rowid"], 1.0 - r["distance"]) for r in rows]

    def vector_search_topics(
        self,
        query_vec: list[float],
        k: int = 10,
    ) -> list[tuple[str, float]]:
        """ANN search over topic centroids.

        Returns:
            ``[(slug, cosine_similarity), ...]`` sorted by relevance.
        """
        packed = _pack(query_vec, self.dim)
        rows = self.db.execute(
            "SELECT m.slug, v.distance "
            "FROM topic_centroids_vec v "
            "JOIN topic_centroids_meta m ON m.rowid = v.rowid "
            "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance",
            (packed, k),
        ).fetchall()
        return [(r["slug"], 1.0 - r["distance"]) for r in rows]

    def best_topic_for_vector(
        self,
        vec: list[float],
    ) -> tuple[str, float] | None:
        """Return the ``(slug, similarity)`` of the closest topic centroid.

        Returns ``None`` when no centroids exist.
        """
        results = self.vector_search_topics(vec, k=1)
        return results[0] if results else None

    # -- FTS / keyword search ---------------------------------------------

    def fts_search(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Full-text search over chunk text via FTS5.

        Returns:
            ``[(rowid, bm25_score), ...]`` sorted by BM25 (lower is better).
        """
        rows = self.db.execute(
            "SELECT rowid, bm25(source_chunks_fts) AS score "
            "FROM source_chunks_fts "
            "WHERE source_chunks_fts MATCH ? "
            "ORDER BY score LIMIT ?",
            (query, k),
        ).fetchall()
        return [(r["rowid"], r["score"]) for r in rows]

    # -- chunk metadata queries -------------------------------------------

    def get_chunk(self, rowid: int) -> dict[str, Any] | None:
        """Return the metadata row for a chunk, or ``None``."""
        row = self.db.execute(
            "SELECT * FROM source_chunks_meta WHERE rowid = ?",
            (rowid,),
        ).fetchone()
        return dict(row) if row else None

    def topic_member_count(self, topic_slug: str) -> int:
        """Return the number of sources belonging to *topic_slug*."""
        row = self.db.execute(
            "SELECT COUNT(*) AS cnt FROM topic_members WHERE topic_slug = ?",
            (topic_slug,),
        ).fetchone()
        return row["cnt"]

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.db.close()
        log.debug("vector_store_closed", path=str(self.db_path))

    def __enter__(self) -> VectorStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
