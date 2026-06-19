"""Near-duplicate source detection (Track 7-1a in §11).

Pairs of sources whose embedding cosine >= threshold are surfaced for
manual review.  No auto-merge — surfacing is passive (§11 7-2a).

Scalability TODO
----------------
The current implementation is O(n^2) over sources, which is fine for MVP
(<1k sources).  For larger brains, a locality-sensitive hashing (LSH)
pre-filter should be added.

References
----------
- ARCHITECTURE.md §11 (anti-graveyard: near-dup embedding cosine >=0.95
  cross-link + badge)
"""

from __future__ import annotations

import math

import numpy as np


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

    # Pairwise comparison.
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a, b = valid[i], valid[j]
            sim = cosine(embeddings[a], embeddings[b])
            if sim >= threshold:
                pairs.append((a, b, sim))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs
