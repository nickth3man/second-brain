"""Hybrid retrieval using reciprocal rank fusion (§10, §12.1).

Provides :func:`search_brain` for the agentic-RAG query flow and
:func:`topic_match` for the embedding-based linking layer (topic-merge
is vector-only per §12.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchHit:
    """A single chunk-level search result from ``search_brain``."""

    rowid: int
    source_id: str
    topic_slug: str | None
    text: str
    score: float


def reciprocal_rank_fusion(
    vector_hits: list[tuple[int, float]],
    fts_hits: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge two ranked lists via Reciprocal Rank Fusion (RRF).

    Args:
        vector_hits: ``[(rowid, similarity), ...]`` from vector search
            (similarity is higher = better).
        fts_hits: ``[(rowid, bm25_score), ...]`` from FTS search
            (bm25_score is lower = better).
        k: RRF constant (default 60 per §12.1).

    Returns:
        ``[(rowid, fused_score), ...]`` sorted by fused score descending.
    """
    # Rank each list by its native ordering (0 = best)
    vec_ranked = sorted(vector_hits, key=lambda x: x[1], reverse=True)
    fts_ranked = sorted(fts_hits, key=lambda x: x[1])

    scores: dict[int, float] = {}
    for rank, (rowid, _) in enumerate(vec_ranked):
        scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (rowid, _) in enumerate(fts_ranked):
        scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def search_brain(
    query: str,
    store: Any,  # VectorStore
    embedder: Any,  # object with embed_query()
    *,
    k: int = 5,
    merge_k: int = 20,
) -> list[SearchHit]:
    """Hybrid search over the brain's chunk store.

    1. Embed the query with *embedder*.
    2. Run vector + FTS search, each returning *merge_k* candidates.
    3. Fuse via :func:`reciprocal_rank_fusion` and take top *k*.
    4. Fetch metadata for each hit and return ``SearchHit`` objects.

    Args:
        query: Free-text search query.
        store: A :class:`~second_brain.vectors.store.VectorStore` instance.
        embedder: An object with ``async embed_query(str) -> list[float]``.
        k: Number of final results to return.
        merge_k: Number of candidates to fetch from each search leg.

    Returns:
        A list of :class:`SearchHit` objects, highest score first.
    """
    qvec = await embedder.embed_query(query)
    vhits = store.vector_search_chunks(qvec, k=merge_k)
    fhits = store.fts_search(query, k=merge_k)
    fused = reciprocal_rank_fusion(vhits, fhits)[:k]

    results: list[SearchHit] = []
    for rowid, score in fused:
        chunk = store.get_chunk(rowid)
        if chunk is None:
            continue
        results.append(
            SearchHit(
                rowid=rowid,
                source_id=chunk["source_id"],
                topic_slug=chunk.get("topic_slug"),
                text=chunk["text"],
                score=score,
            )
        )
    return results


def topic_match(
    candidate_vec: list[float],
    vec_store: Any,  # VectorStore
    threshold: float,
) -> tuple[str | None, float]:
    """Match a candidate embedding against existing topic centroids.

    Topic-merge is vector-only per §12.1 (no lexical overlap expected).

    Args:
        candidate_vec: The embedding vector to match.
        vec_store: A :class:`~second_brain.vectors.store.VectorStore` instance.
        threshold: Minimum cosine similarity to consider a match.

    Returns:
        ``(slug, similarity)`` if the best match meets the threshold,
        else ``(None, similarity)``.  Similarity is 0.0 when no centroids
        exist.
    """
    best = vec_store.best_topic_for_vector(candidate_vec)
    if best and best[1] >= threshold:
        return best
    return (None, best[1] if best else 0.0)
