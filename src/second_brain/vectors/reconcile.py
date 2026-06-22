"""Startup reconciliation for vector, FTS, centroid, and tombstone drift."""

from __future__ import annotations

from dataclasses import dataclass, field

from second_brain.frontmatter import split_frontmatter
from second_brain.vectors.store import VectorStore, chunk_text


@dataclass
class VectorReconcileReport:
    missing_source_embeddings: list[str] = field(default_factory=list)
    orphan_embeddings: list[str] = field(default_factory=list)
    stale_embeddings: list[str] = field(default_factory=list)
    missing_topic_centroids: list[str] = field(default_factory=list)
    stale_topic_centroids: list[str] = field(default_factory=list)
    fts_mismatches: list[str] = field(default_factory=list)
    tombstone_inconsistencies: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(getattr(self, field_name) for field_name in self.__dataclass_fields__)


async def reconcile_vector_index(
    cfg,
    store,
    vec_store: VectorStore,
    embedder=None,
) -> VectorReconcileReport:
    """Repair drift between state, source markdown, vector rows, FTS, and centroids."""
    report = VectorReconcileReport()
    db = vec_store.db

    rows = db.execute(
        "SELECT source_id, source_hash, embedding_model FROM source_chunks_meta"
    ).fetchall()
    indexed_sources = {row["source_id"] for row in rows}
    state_sources = set(store.state.sources)

    for source_id in sorted(indexed_sources - state_sources):
        vec_store.tombstone_source(source_id)
        report.orphan_embeddings.append(source_id)

    for source_id in sorted(state_sources):
        source = store.state.sources[source_id]
        source_rows = [
            row for row in rows
            if row["source_id"] == source_id
        ]
        if not source_rows:
            report.missing_source_embeddings.append(source_id)
            await _rebuild_source_embeddings(cfg, store, vec_store, embedder, source_id)
            continue
        if any(
            (row["source_hash"] and row["source_hash"] != source.sha256)
            or (row["embedding_model"] and row["embedding_model"] != vec_store.model)
            or (source.embedding_model and source.embedding_model != vec_store.model)
            for row in source_rows
        ):
            vec_store.tombstone_source(source_id)
            report.stale_embeddings.append(source_id)
            await _rebuild_source_embeddings(cfg, store, vec_store, embedder, source_id)

    _repair_fts(db, report)
    _repair_topic_members(store, vec_store)
    _repair_centroids(store, vec_store, report)
    _repair_tombstones(db, report)
    db.commit()
    return report


async def _rebuild_source_embeddings(
    cfg,
    store,
    vec_store: VectorStore,
    embedder,
    source_id: str,
) -> None:
    if embedder is None:
        return
    source = store.state.sources.get(source_id)
    if source is None:
        return
    source_path = cfg.brain_root / "50-sources" / f"{source_id}.md"
    if not source_path.is_file():
        return
    _meta, body = split_frontmatter(source_path.read_text(encoding="utf-8"))
    chunks = chunk_text(body)
    if not chunks:
        return
    vectors = await embedder.embed_texts(chunks)
    topics = source.topics or [""]
    vec_store.upsert_source_chunks(
        source_id,
        topics[0],
        list(zip(chunks, vectors, strict=False)),
        source_hash=source.sha256,
        embedding_model=vec_store.model,
    )
    for topic_slug in source.topics:
        vec_store.add_topic_member(topic_slug, source_id)
    source.embedding_model = vec_store.model


def _repair_fts(db, report: VectorReconcileReport) -> None:
    meta_rows = db.execute(
        "SELECT rowid, text, source_id, topic_slug FROM source_chunks_meta"
    ).fetchall()
    fts_rows = db.execute("SELECT rowid, source_id FROM source_chunks_fts").fetchall()
    fts_rowids = {row["rowid"] for row in fts_rows}
    meta_rowids = {row["rowid"] for row in meta_rows}
    if fts_rowids != meta_rowids:
        report.fts_mismatches.append("source_chunks_fts")
        db.execute("DELETE FROM source_chunks_fts")
        for row in meta_rows:
            db.execute(
                "INSERT INTO source_chunks_fts(rowid, text, source_id, topic_slug) "
                "VALUES (?, ?, ?, ?)",
                (row["rowid"], row["text"], row["source_id"], row["topic_slug"]),
            )


def _repair_topic_members(store, vec_store: VectorStore) -> None:
    for source_id, source in store.state.sources.items():
        for topic_slug in source.topics:
            vec_store.add_topic_member(topic_slug, source_id)


def _repair_centroids(store, vec_store: VectorStore, report: VectorReconcileReport) -> None:
    existing = {
        row["slug"]: row
        for row in vec_store.db.execute("SELECT slug, member_count FROM topic_centroids_meta")
    }
    for slug, topic in store.state.topics.items():
        expected = len(set(topic.sources))
        current = existing.get(slug)
        if current is None:
            report.missing_topic_centroids.append(slug)
            vec_store.recompute_centroid(slug)
        elif current["member_count"] != expected:
            report.stale_topic_centroids.append(slug)
            vec_store.recompute_centroid(slug)


def _repair_tombstones(db, report: VectorReconcileReport) -> None:
    rows = db.execute(
        "SELECT DISTINCT t.source_id FROM vec_tombstones t "
        "JOIN source_chunks_meta m ON m.source_id = t.source_id"
    ).fetchall()
    for row in rows:
        report.tombstone_inconsistencies.append(row["source_id"])
        db.execute("DELETE FROM vec_tombstones WHERE source_id = ?", (row["source_id"],))
