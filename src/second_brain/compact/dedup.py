"""Near-duplicate source detection (Track 7-1a in §11).

Pairs of sources whose embedding cosine >= threshold are surfaced for
manual review.  No auto-merge — surfacing is passive (§11 7-2a).

For larger brains this uses a deterministic SimHash-style LSH pre-filter so
normal compaction does not compare every pair at 1000+ source scale.

References
----------
- ARCHITECTURE.md §11 (anti-graveyard: near-dup embedding cosine >=0.95
  cross-link + badge)
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

LSH_SOURCE_THRESHOLD = 1000
LSH_PLANES = 16


def cosine(a: list[float], b: list[float]) -> float:
    """Return the cosine similarity between two vectors.

    Returns ``0.0`` when either vector is zero-magnitude.
    """
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    dot = float(np.dot(a_arr, b_arr))
    norm_a = float(np.linalg.norm(a_arr))
    norm_b = float(np.linalg.norm(b_arr))
    if math.isclose(norm_a, 0.0) or math.isclose(norm_b, 0.0):
        return 0.0
    return dot / (norm_a * norm_b)


async def find_near_duplicates(
    cfg: object,
    store: object,
    vec_store: object,  # noqa: ARG001
    embedder: object,
    threshold: float = 0.95,
) -> list[tuple[str, str, float]]:
    """Find pairs of sources whose embeddings are near-duplicates.

    This function is the O(n²) batch detector used by manual/CLI
    compaction passes.  For per-file pipeline use, prefer
    :func:`find_near_duplicates_for_source` (O(n) per ingest).

    For each source, embeds a representative text (reads the
    ``50-sources/{source_id}.md`` file body) and compares pairwise.

    Returns:
        ``[(source_id_a, source_id_b, cosine_similarity), ...]`` for
        every pair with ``similarity >= threshold``, sorted descending
        by similarity.
    """
    source_ids = list(store.state.sources.keys())
    if len(source_ids) < 2:
        return []

    # Read source file bodies as representative texts.
    texts: dict[str, str] = {}
    for sid in source_ids:
        src_path = cfg.brain_root / "50-sources" / f"{sid}.md"
        if src_path.exists():
            texts[sid] = src_path.read_text(encoding="utf-8")

    # Skip sources whose files we couldn't read.
    valid = [sid for sid in source_ids if sid in texts]
    if len(valid) < 2:
        return []

    # Embed all texts.
    embeddings: dict[str, list[float]] = {}
    for sid in valid:
        embeddings[sid] = await embedder.embed_one(texts[sid])

    if len(valid) >= LSH_SOURCE_THRESHOLD:
        candidates = _lsh_candidate_pairs(valid, embeddings)
    else:
        candidates = (
            (valid[i], valid[j])
            for i in range(len(valid))
            for j in range(i + 1, len(valid))
        )

    pairs: list[tuple[str, str, float]] = []
    for a, b in candidates:
        sim = cosine(embeddings[a], embeddings[b])
        if sim >= threshold:
            pairs.append((a, b, sim))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def _lsh_candidate_pairs(
    source_ids: list[str],
    embeddings: dict[str, list[float]],
) -> set[tuple[str, str]]:
    """Return candidate pairs from deterministic random-hyperplane buckets."""
    dim = len(next(iter(embeddings.values())))
    rng = np.random.default_rng(17)
    planes = rng.normal(size=(LSH_PLANES, dim))
    buckets: dict[str, list[str]] = {}
    for sid in source_ids:
        vec = np.array(embeddings[sid], dtype=np.float64)
        bits = "".join("1" if float(np.dot(vec, plane)) >= 0 else "0" for plane in planes)
        # Prefix buckets improve recall for very similar vectors that cross one
        # late hyperplane while keeping candidate sets bounded.
        for width in (8, 12, 16):
            buckets.setdefault(bits[:width], []).append(sid)

    pairs: set[tuple[str, str]] = set()
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = sorted((bucket[i], bucket[j]))
                pairs.add((a, b))
    return pairs


# Module-level in-process cache of per-source representative embeddings.
# Keyed on ``(source_id, sha256)`` where the sha is the SHA-256 of the
# source file body, so any re-write of the source file (which changes its
# sha) invalidates the cached embedding automatically.
_SOURCE_EMBEDDING_CACHE: dict[tuple[str, str], list[float]] = {}


async def find_near_duplicates_for_source(
    cfg,
    store,
    embedder,
    vec_store,
    new_source_id: str,
    new_embedding: list[float],
    threshold: float = 0.95,
) -> list[tuple[str, float]]:
    """Find existing sources that are near-duplicates of *new_source*.

    Compares the new source's embedding against every existing source's
    representative embedding.  O(n) per ingest — suitable for the
    per-file pipeline.

    Two-tier source-embedding strategy:

    1. **Prefer** ``vec_store.source_centroid(sid)`` when it returns a
       non-None vector (sources whose chunks are already in the store).
       This avoids a re-embedding round trip.
    2. **Fall back** to re-embedding the source file body via
       ``embedder.embed_one(text)`` only for legacy sources whose chunks
       are not in the vector store yet.  The fallback path is cached in
       ``_SOURCE_EMBEDDING_CACHE`` keyed on ``(source_id, sha256)`` so
       rewrites invalidate the cache automatically.

    Args:
        cfg: Brain config (used to resolve ``50-sources/<id>.md`` for
            the fallback path).
        store: Brain state store — iterated over ``state.sources``.
        embedder: Embedding client used only for the fallback path.
        vec_store: Vector store consulted first via
            :meth:`VectorStore.source_centroid`.
        new_source_id: The source id of the newly ingested file (skipped
            in the comparison loop).
        new_embedding: Representative embedding of the new source (mean
            of its chunk embeddings).
        threshold: Minimum cosine similarity to flag as a near-dup.

    Returns:
        ``[(existing_source_id, cosine_similarity), ...]`` for every
        existing source with similarity >= *threshold*, sorted
        descending by similarity.
    """
    hits: list[tuple[str, float]] = []
    for sid in store.state.sources:
        if sid == new_source_id:
            continue

        # Tier 1: prefer the vector-store centroid when available.
        existing_vec = None
        if vec_store is not None:
            try:
                existing_vec = vec_store.source_centroid(sid)
            except Exception:
                existing_vec = None

        # Tier 2: fall back to text re-embedding for legacy sources
        # (no chunks in the store yet).
        if existing_vec is None:
            src_path = cfg.brain_root / "50-sources" / f"{sid}.md"
            if not src_path.exists():
                continue
            try:
                text = src_path.read_text(encoding="utf-8")
            except OSError:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cache_key = (sid, digest)
            existing_vec = _SOURCE_EMBEDDING_CACHE.get(cache_key)
            if existing_vec is None:
                try:
                    existing_vec = await embedder.embed_one(text)
                except Exception:
                    # Skip sources that cannot be embedded rather than
                    # failing the whole ingest — near-dup surfacing is
                    # passive (§11).
                    continue
                _SOURCE_EMBEDDING_CACHE[cache_key] = existing_vec

        sim = cosine(new_embedding, existing_vec)
        if sim >= threshold:
            hits.append((sid, sim))
    hits.sort(key=lambda x: x[1], reverse=True)
    return hits
