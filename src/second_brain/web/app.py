"""FastAPI web UI — §10 (query/chat), §12.4 (stack), §12.7 (hardening).

Routes
------
GET  /            — index with topic list + recent sources
GET  /topic/{slug} — rendered wiki topic page (§12.4 pipeline)
GET  /search?q=   — hybrid search (semantic -> title-substring fallback)
GET  /health      — structural health check (§11)

Read-only: the UI NEVER writes to the brain.
Bound to 127.0.0.1 only (§12.7).
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from second_brain.config import load_config
from second_brain.state import BrainStateStore

# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_store: BrainStateStore | None = None


def _get_store(cfg) -> BrainStateStore:
    global _store  # noqa: PLW0603
    if _store is None:
        _store = BrainStateStore.load(cfg)
    return _store


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "templates"
STATIC = HERE / "static"


def create_app(cfg=None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        cfg: Optional config object. If ``None``, ``load_config()`` is called.

    Returns:
        A configured FastAPI instance ready for ``uvicorn.run``.
    """
    if cfg is None:
        cfg = load_config()

    app = FastAPI(title="Second Brain", version="0.1.0")

    # Jinja2 templates
    templates = Jinja2Templates(directory=str(TEMPLATES))

    # Static files
    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    # -- helpers -----------------------------------------------------------

    def _store_singleton() -> BrainStateStore:
        return _get_store(cfg)

    # -- routes ------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        store = _store_singleton()
        topics = sorted(
            (
                {
                    "slug": slug,
                    "title": t.title,
                    "source_count": len(t.sources),
                    "updated": t.updated,
                }
                for slug, t in store.state.topics.items()
            ),
            key=lambda x: x["title"].lower(),
        )

        # Recent sources: top 10 by ingested date (descending)
        sorted_sources = sorted(
            store.state.sources.items(),
            key=lambda kv: kv[1].ingested,
            reverse=True,
        )[:10]
        recent = []
        for sid, src in sorted_sources:
            topic_slug = src.topics[0] if src.topics else None
            topic_title = (
                store.state.topics[topic_slug].title
                if topic_slug and topic_slug in store.state.topics
                else None
            )
            recent.append(
                {
                    "source_id": sid,
                    "topic_slug": topic_slug,
                    "topic_title": topic_title,
                    "ingested": src.ingested,
                }
            )

        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "source_count": len(store.state.sources),
                "topic_count": len(store.state.topics),
                "topics": topics,
                "recent_sources": recent,
            },
        )

    @app.get("/topic/{slug}", response_class=HTMLResponse)
    async def topic(request: Request, slug: str) -> HTMLResponse:
        store = _store_singleton()
        from second_brain.web.render import RenderedPage, render_topic_page

        action_create = request.query_params.get("action") == "create"

        if action_create:
            page = RenderedPage(
                slug=slug,
                title=slug.replace("-", " ").title(),
                html_body=(
                    "<p><em>This page does not exist yet. "
                    "Write a <code>90-wiki/{slug}.md</code> file "
                    "to create it.</em></p>"
                ),
                infobox=None,
                breadcrumbs=[("Home", "/"), (slug, f"/topic/{slug}")],
                see_also=[],
            )
        else:
            try:
                page = render_topic_page(slug, store)
            except FileNotFoundError as err:
                raise StarletteHTTPException(
                    status_code=404,
                    detail=f"Topic '{slug}' not found. Add ?action=create to view the placeholder.",
                ) from err

        return templates.TemplateResponse(
            request,
            "topic.html",
            {
                "page": page,
            },
        )

    @app.get("/search", response_class=HTMLResponse)
    async def search(
        request: Request,
        q: str = Query("", min_length=0),
    ) -> HTMLResponse:
        store = _store_singleton()
        hits: list[dict] = []
        fallback_note: str | None = None

        if q.strip():
            # Semantic search via the daemon loopback HTTP API (§12.1).
            # The web UI MUST NOT open a writeable VectorStore.
            try:
                import httpx

                daemon_url = (
                    f"http://{cfg.daemon.http_host}:"
                    f"{cfg.daemon.http_port}/search_brain"
                )
                async with httpx.AsyncClient() as http:
                    resp = await http.post(
                        daemon_url,
                        json={"query": q, "k": 10},
                        timeout=10.0,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                for hit in data.get("hits", []):
                    snippet = textwrap.shorten(
                        (hit.get("text") or "").replace("\n", " "),
                        width=160,
                        placeholder="...",
                    )
                    hits.append(
                        {
                            "source_id": hit.get("source_id", ""),
                            "topic_slug": hit.get("topic_slug", ""),
                            "text": snippet,
                            "score": hit.get("score", 0.0),
                        }
                    )
            except Exception:
                # Fallback: title-substring search (e.g. daemon not running).
                fallback_note = (
                    "Semantic search unavailable (daemon not running on "
                    f"{cfg.daemon.http_host}:{cfg.daemon.http_port}). "
                    "Showing title matches instead."
                )
                q_lower = q.lower()
                for slug, t in store.state.topics.items():
                    if q_lower in t.title.lower():
                        hits.append(
                            {
                                "source_id": "",
                                "topic_slug": slug,
                                "text": t.title,
                                "score": 0.0,
                            }
                        )
                hits.sort(key=lambda x: x["topic_slug"] or "")

        return templates.TemplateResponse(
            request,
            "search.html",
            {
                "query": q,
                "hits": hits,
                "fallback_note": fallback_note,
            },
        )

    @app.get("/health", response_class=HTMLResponse)
    async def health(request: Request) -> HTMLResponse:
        store = _store_singleton()
        from second_brain.compact.eval import render_health_markdown, run_health_check

        report = run_health_check(cfg, store)
        health_md = render_health_markdown(report)

        return templates.TemplateResponse(
            request,
            "health.html",
            {
                "report": report,
                "health_md": health_md,
            },
        )

    # -- chat routes (Phase 6, §10, §12.4) -------------------------------

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_page(request: Request) -> HTMLResponse:
        """Render the chat UI page."""
        return templates.TemplateResponse(request, "chat.html", {})

    @app.get("/api/chat")
    async def chat_api(q: str = Query("", min_length=0)) -> StreamingResponse:
        """SSE endpoint for the chat agent.

        Returns a ``text/event-stream`` with typed events matching §12.4.
        Retrieval goes through the daemon loopback HTTP API (§12.1); the
        web UI never opens a writeable VectorStore.
        """
        if not q.strip():
            empty = json.dumps(
                {"type": "done", "error": "empty query"}, ensure_ascii=False
            )
            return StreamingResponse(
                iter([f"data: {empty}\n\n"]),
                media_type="text/event-stream",
            )

        import httpx

        from second_brain.chat import chat_stream
        from second_brain.openrouter_client import OpenRouterClient

        daemon_url = (
            f"http://{cfg.daemon.http_host}:"
            f"{cfg.daemon.http_port}/search_brain"
        )

        async def _remote_search(query: str, k: int) -> list[dict]:
            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    daemon_url,
                    json={"query": query, "k": k},
                    timeout=10.0,
                )
                resp.raise_for_status()
                return resp.json().get("hits", [])

        async def event_gen() -> AsyncGenerator[str, None]:
            client: OpenRouterClient | None = None
            try:
                client = OpenRouterClient(cfg)
                store = _store_singleton()
                async for event in chat_stream(
                    q.strip(),
                    cfg,
                    store,
                    vec_store=None,
                    embedder=None,
                    client=client,
                    k=8,
                    search_fn=_remote_search,
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:
                # No API key / model error / etc. -> end the stream gracefully
                # so the client always receives a terminal event.
                err = json.dumps(
                    {"type": "done", "error": str(exc)}, ensure_ascii=False
                )
                yield f"data: {err}\n\n"
            finally:
                if client is not None:
                    await client.close()

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
        )

    return app


# ---------------------------------------------------------------------------
# Server runner
# ---------------------------------------------------------------------------


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the Uvicorn server bound to *host*:*port*.

    Always binds to 127.0.0.1 (§12.7) -- never 0.0.0.0.
    """
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port)
