"""On-disk cached OpenRouter embeddings client (§12.5).

Caches every (model, text) pair as a JSON blob in
``.brain/cache/embeddings/<sha256>.json`` so repeated or overlapping calls
hit the filesystem instead of the API.  Cache writes use ``write_atomic``
for crash-safety.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from second_brain.atomicio import write_atomic

DEFAULT_EMBED_DIM = 1536


def _cache_path(cfg, model: str, text: str) -> Path:
    """Return the cache file path for a *(model, text)* pair.

    The path is deterministic: SHA-256 of ``model + text`` hex-encoded
    with a ``.json`` suffix, stored under
    ``cfg.brain_root / ".brain/cache/embeddings"``.
    """
    key = (model + text).encode()
    digest = hashlib.sha256(key).hexdigest()
    return cfg.brain_root / ".brain" / "cache" / "embeddings" / f"{digest}.json"


class Embedder:
    """Lightweight embedding facade backed by an OpenRouter client and disk cache.

    Usage::

        embedder = Embedder(openrouter_client, cfg)
        vec = await embedder.embed_one("Some text to embed")
        vecs = await embedder.embed_texts(["first", "second"])
    """

    def __init__(self, client, cfg) -> None:
        self.client = client
        self.model = cfg.models.embedding
        self.dim = DEFAULT_EMBED_DIM
        self.cfg = cfg

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single text string, using on-disk cache when available.

        Returns:
            The embedding vector as a ``list[float]``.
        """
        cache_path = _cache_path(self.cfg, self.model, text)
        if cache_path.is_file():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        vec = await self.client.embedding(self.model, text)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(cache_path, json.dumps(vec))
        return vec

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed every string in *texts*, returning results in the same order.

        Each call goes through :meth:`embed_one` so the cache is checked per
        text — repeated or duplicate texts hit disk instead of the API.
        """
        return [await self.embed_one(t) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a search query string.

        Currently an alias of :meth:`embed_one`.  Future phases may add
        query-specific prefixes or instruction tuning.
        """
        return await self.embed_one(query)
