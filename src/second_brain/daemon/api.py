"""Daemon loopback HTTP API — exposes search_brain to the web UI (§12.1).

The daemon owns the writeable VectorStore. The web UI opens read-only and
calls this API for vector search. Bound to 127.0.0.1 only.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
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
    """Build the daemon's FastAPI sub-app with /search_brain and /health.

    Args:
        vec_store: The daemon-owned writeable VectorStore (used for reads
            here — writes happen only via the pipeline).
        embedder: The daemon's Embedder instance.
        cfg: Application Config (kept for future endpoints; unused today).

    Returns:
        A FastAPI sub-app exposing ``POST /search_brain`` and ``GET /health``.
    """
    app = FastAPI(title="Second Brain Daemon API", version="0.1.0")

    @app.post("/search_brain", response_model=SearchResponse)
    async def search_brain_endpoint(req: SearchRequest) -> SearchResponse:
        from second_brain.vectors.retrieval import search_brain

        results = await search_brain(req.query, vec_store, embedder, k=req.k)
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

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
