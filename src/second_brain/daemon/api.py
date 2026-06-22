"""Daemon loopback HTTP API — exposes search_brain to the web UI (§12.1).

The daemon owns the writeable VectorStore. The web UI opens read-only and
calls this API for vector search. Bound to 127.0.0.1 only.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    k: int = 10


class SearchHitResponse(BaseModel):
    source_id: str
    topic_slug: str
    text: str
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHitResponse]


def create_daemon_app(vec_store: Any, embedder: Any, cfg: Any) -> FastAPI:
    """Build the daemon's FastAPI sub-app with /search_brain, /compact, /health.

    Args:
        vec_store: The daemon-owned writeable VectorStore (used for reads
            here — writes happen only via the pipeline).
        embedder: The daemon's Embedder instance. Its ``.client`` attribute
            is the OpenRouterClient the daemon created; compaction reuses it.
        cfg: Application Config (used by the ``/compact`` endpoint).

    Returns:
        A FastAPI sub-app exposing ``POST /search_brain``, ``POST /compact``
        and ``GET /health``.
    """
    app = FastAPI(title="Second Brain Daemon API", version="0.1.0")

    @app.post("/search_brain", response_model=SearchResponse)
    async def search_brain_endpoint(req: SearchRequest) -> SearchResponse:
        from second_brain.vectors.retrieval import search_brain

        try:
            results = await search_brain(
                req.query, vec_store, embedder, k=req.k
            )
        except Exception as exc:
            # Never leak a 500 / traceback to the web UI. Return 503 so the
            # caller can fall back to a degraded path (e.g. title search).
            kind = (
                "embedder"
                if "embed" in type(exc).__name__.lower()
                else "unknown"
            )
            raise HTTPException(
                status_code=503,
                detail=f"search failed: {kind}",
            ) from exc
        return SearchResponse(
            hits=[
                SearchHitResponse(
                    source_id=h.source_id,
                    topic_slug=h.topic_slug or "",
                    text=h.text,
                    score=h.score,
                )
                for h in results
            ]
        )

    @app.post("/compact")
    async def compact_endpoint() -> dict[str, Any]:
        """Run one compaction pass using the daemon's writeable store.

        Routes ``brain compact`` through the single writer (§12.1) so the
        CLI never opens its own writeable VectorStore while the daemon is
        running.
        """
        from second_brain.compact.compaction import run_compaction
        from second_brain.state import BrainStateStore, now_iso

        try:
            store = BrainStateStore.load(cfg)
            summary = await run_compaction(
                cfg,
                store,
                vec_store,
                # The daemon's embedder wraps the OpenRouterClient; reuse it
                # for synthesis rewrites instead of building a new one.
                embedder.client,
                merge_threshold=cfg.compaction.merge_threshold,
            )
            # Phase 4: a manual compact counts as a compaction run for
            # scheduling purposes — reset the counter + timestamp so the
            # daemon scheduler doesn't immediately fire again (§8).
            store.state.sources_since_compaction = 0
            store.state.last_compaction_ts = now_iso()
            store.save()
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"compaction failed: {type(exc).__name__}",
            ) from exc
        return {"summary": summary}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    return app


async def start_daemon_server(
    app: FastAPI, host: str = "127.0.0.1", port: int = 8001
) -> None:
    """Start a uvicorn server for the daemon API in the running event loop.

    Must be ``await``ed inside an asyncio task so the watcher pipeline can
    run concurrently. Cancelling the task stops the server.
    """
    import uvicorn

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        # The daemon owns signal handling; uvicorn must not install its own
        # handlers (they'd hijack Ctrl-C / SIGTERM from the watcher loop).
        install_signal_handlers=False,
    )
    server = uvicorn.Server(config)
    await server.serve()
