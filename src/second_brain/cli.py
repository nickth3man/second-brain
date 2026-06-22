"""Second Brain CLI — ``brain`` entry point.

Usage::

    brain init      # scaffold folders + keyring
    brain watch     # start file-watcher daemon (Phase 1)
    brain ingest   # ingest one file (Phase 1/2)
    brain search   # hybrid search (Phase 2)
    brain rebuild  # rebuild wiki (Phase 5)
    brain compact  # run compaction pass (Phase 4)
    brain health   # run health check (Phase 4)
    brain ask      # interactive chat (Phase 6)
    brain embedding-swap # blue/green embedding model swap
    brain --version # show version
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from second_brain import __version__
from second_brain.config import load_config
from second_brain.log import configure_logging

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    invoke_without_command=True,
    help="Second Brain — self-improving knowledge base.",
)


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    if version:
        typer.echo(f"second-brain v{__version__}")
        raise typer.Exit()


@app.command()
def init() -> None:
    """Initialize the Second Brain folder structure and keyring."""
    cfg = load_config()

    # Store API key in keyring if configured
    if cfg.privacy.api_key_source == "keyring":
        import keyring

        existing = keyring.get_password("second-brain", "openrouter")
        if not existing:
            key = typer.prompt("Enter your OpenRouter API key", hide_input=True)
            keyring.set_password("second-brain", "openrouter", key)
            typer.echo("API key stored in system keyring.")

    # Create tracked directories
    for dirname in ("00-inbox", "50-sources", "90-wiki", ".brain"):
        (cfg.brain_root / dirname).mkdir(parents=True, exist_ok=True)

    typer.echo(
        "Enable the 4 model-group ZDR toggles at "
        "https://openrouter.ai/settings/privacy (one-time)."
    )
    try:
        from second_brain.openrouter_client import OpenRouterClient

        status = asyncio.run(OpenRouterClient(cfg).verify_zdr_status())
        typer.echo(status["message"])
    except Exception as exc:
        typer.echo(
            "Account-level ZDR remains manually unconfirmed; "
            f"automatic verification unavailable ({exc})."
        )
    typer.echo("Second Brain initialized successfully.")


@app.command()
def watch() -> None:
    """Start the file-watcher daemon (Phase 1 — text-only ingestion loop)."""
    from second_brain.daemon.pipeline import run_daemon

    cfg = load_config()
    configure_logging(cfg.brain_root)
    typer.echo(f"Watching {cfg.brain_root / '00-inbox'}/ ... Ctrl-C to stop.")
    asyncio.run(run_daemon(cfg))


def _render_stage_table(progress: list[dict]) -> None:
    """Render the compact per-stage table for ``brain ingest`` output.

    Fixed-width columns: Stage (12) | Model (30) | Status (7) | Notes.
    An empty ``model`` string renders as a blank cell (not ``None``).
    Rows are printed in the order they appear in ``progress``.
    """
    typer.echo(f"{'Stage':<12}{'Model':<30}{'Status':<7}Notes")
    typer.echo("-" * 60)
    for row in progress:
        stage = str(row.get("stage", ""))
        model = row.get("model") or ""
        status = str(row.get("status", ""))
        notes = row.get("notes") or ""
        typer.echo(f"{stage:<12}{model:<30}{status:<7}{notes}")


@app.command()
def ingest(
    path: Annotated[
        Path, typer.Argument(help="Path to the file to ingest (one-shot, for testing).")
    ],
) -> None:
    """Ingest a single file (one-shot, without the watcher)."""

    async def _ingest_one() -> tuple[str, list[dict]]:
        from second_brain.daemon.index import DebouncedIndex
        from second_brain.daemon.linker import EmbeddingLinker
        from second_brain.daemon.pipeline import ingest_file
        from second_brain.openrouter_client import OpenRouterClient
        from second_brain.state import BrainStateStore
        from second_brain.vectors.embed import Embedder
        from second_brain.vectors.store import VectorStore

        cfg = load_config()
        configure_logging(cfg.brain_root)
        client = OpenRouterClient(cfg)
        embedder = Embedder(client, cfg)
        dim = await embedder.ensure_dim()
        vec_store = VectorStore(
            cfg.brain_root / ".brain/embeddings.db",
            cfg.models.embedding,
            dim=dim,
        )
        store = BrainStateStore.load(cfg)
        linker = EmbeddingLinker(
            embedder, vec_store, cfg.ingestion.merge_threshold
        )
        index = DebouncedIndex(cfg, store)
        try:
            progress: list[dict] = []
            stage = await ingest_file(
                path, cfg, store, client, linker, index,
                embedder=embedder, vec_store=vec_store,
                progress=progress,
            )
            await index.flush_now()
            return str(stage), progress
        finally:
            vec_store.close()
            await client.close()

    stage, progress = asyncio.run(_ingest_one())
    typer.echo(f"{path.name}: {stage}")
    if progress:
        typer.echo("")
        _render_stage_table(progress)


async def _daemon_search(
    cfg, query: str, *, k: int = 10, timeout: float = 5.0,
) -> list[dict] | None:
    """Try the daemon HTTP search endpoint.

    Returns the hits list on success, or ``None`` if the daemon is not
    reachable (ConnectError / TimeoutException). Single-writer invariant
    (§12.1): the daemon owns the writeable VectorStore.
    """
    import httpx

    try:
        url = (
            f"http://{cfg.daemon.http_host}:"
            f"{cfg.daemon.http_port}/search_brain"
        )
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=0.5)
        ) as http:
            resp = await http.post(url, json={"query": query, "k": k})
            resp.raise_for_status()
            return resp.json().get("hits", [])
    except (httpx.ConnectError, httpx.TimeoutException):
        return None


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query.")],
) -> None:
    """Search the brain using hybrid retrieval (Phase 2)."""

    async def _search() -> None:
        from second_brain.state import BrainStateStore

        cfg = load_config()

        # Try the daemon first (single-writer invariant, §12.1).
        remote_hits = await _daemon_search(cfg, query, k=5)
        if remote_hits is not None:
            if not remote_hits:
                typer.echo("No results.")
                return
            for i, hit in enumerate(remote_hits, 1):
                snippet = (hit.get("text") or "")[:120].replace("\n", " ")
                typer.echo(
                    f"{i}. [{hit.get('source_id', '')}] "
                    f"(topic={hit.get('topic_slug')}, "
                    f"score={float(hit.get('score', 0.0)):.4f})\n"
                    f"   {snippet}\n"
                )
            return

        # Query commands are read-only (§10). If the daemon is down, do not
        # open sqlite locally in write mode; degrade to title matches.
        q_lower = query.lower()
        store = BrainStateStore.load(cfg)
        matches = [
            (slug, topic.title)
            for slug, topic in store.state.topics.items()
            if q_lower in topic.title.lower()
        ]
        if not matches:
            typer.echo("Semantic search unavailable (daemon not running). No title matches.")
            return
        typer.echo("Semantic search unavailable (daemon not running). Showing title matches.")
        for i, (slug, title) in enumerate(sorted(matches), 1):
            typer.echo(f"{i}. {title} (topic={slug})")

    asyncio.run(_search())


@app.command()
def rebuild(
    from_sources: Annotated[
        bool, typer.Option("--from-sources", help="Fast rebuild from 50-sources/ (re-link only).")
    ] = False,
    from_inbox: Annotated[
        bool, typer.Option("--from-inbox", help="Deep rebuild from 00-inbox/ (re-extract).")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would be rebuilt; do not write.")
    ] = False,
) -> None:
    """Rebuild the wiki from sources or inbox (Phase 5 — §12.6 escape hatch)."""

    async def _rebuild() -> None:
        import httpx

        from second_brain.openrouter_client import OpenRouterClient
        from second_brain.rebuild import rebuild_from_inbox, rebuild_from_sources

        cfg = load_config()
        configure_logging(cfg.brain_root)

        # Validate exactly one of --from-sources / --from-inbox.
        # xor: both-set or both-unset are invalid.
        if from_sources == from_inbox:
            typer.echo(
                "Usage: brain rebuild --from-sources | --from-inbox "
                "(exactly one required)."
            )
            raise typer.Exit(2)

        # Daemon-running guard (§12.1 single-writer invariant).
        try:
            url = (
                f"http://{cfg.daemon.http_host}:"
                f"{cfg.daemon.http_port}/health"
            )
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(5.0, connect=0.5)
            ) as http:
                await http.get(url)
            typer.echo(
                f"Daemon is running on port {cfg.daemon.http_port}; "
                "stop it before rebuild (single-writer invariant, §12.1)."
            )
            raise typer.Exit(1)
        except (httpx.ConnectError, httpx.TimeoutException):
            pass  # Daemon is down — proceed.

        client = OpenRouterClient(cfg)
        try:
            if from_sources:
                plan = await rebuild_from_sources(cfg, client, dry_run=dry_run)
            else:
                plan = await rebuild_from_inbox(cfg, client, dry_run=dry_run)
        except RuntimeError as e:
            typer.echo(f"Rebuild failed: {e}")
            raise typer.Exit(1) from e
        finally:
            await client.close()

        typer.echo(f"Rebuild mode: {plan.mode}")
        typer.echo(f"Dry run: {plan.dry_run}")
        typer.echo(f"Sources seen: {plan.sources_seen}")
        typer.echo(f"Sources skipped: {plan.sources_skipped}")
        typer.echo(f"Topics before: {plan.topics_before}")
        typer.echo(f"Topics after: {plan.topics_after}")
        if plan.snapshot_dir:
            typer.echo(f"Snapshot: {plan.snapshot_dir}")

    asyncio.run(_rebuild())


@app.command()
def compact() -> None:
    """Run one compaction pass (Phase 4 — merge similar topics)."""

    async def _compact() -> None:
        import httpx

        from second_brain.compact.eval import run_health_check
        from second_brain.state import BrainStateStore

        cfg = load_config()
        configure_logging(cfg.brain_root)

        # Try the daemon first (single-writer invariant, §12.1). Compaction
        # is slow (LLM synthesis rewrites), so allow a long timeout.
        try:
            url = (
                f"http://{cfg.daemon.http_host}:"
                f"{cfg.daemon.http_port}/compact"
            )
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=2.0)
            ) as http:
                resp = await http.post(url)
                resp.raise_for_status()
            summary = resp.json().get("summary", {})
        except (httpx.ConnectError, httpx.TimeoutException):
            # Daemon not running — fall back to a local writeable store.
            from second_brain.compact.compaction import run_compaction
            from second_brain.openrouter_client import OpenRouterClient
            from second_brain.vectors.embed import Embedder
            from second_brain.vectors.store import VectorStore

            client = OpenRouterClient(cfg)
            embedder = Embedder(client, cfg)
            dim = await embedder.ensure_dim()
            vec_store = VectorStore(
                cfg.brain_root / ".brain/embeddings.db",
                cfg.models.embedding,
                dim=dim,
            )
            store = BrainStateStore.load(cfg)
            try:
                summary = await run_compaction(
                    cfg, store, vec_store, client,
                    merge_threshold=cfg.compaction.merge_threshold,
                )
            finally:
                vec_store.close()
                await client.close()

        # After compaction (either path): reload state and run the health
        # check. The daemon path saved state server-side, so a fresh load
        # sees the post-compaction topics.
        store = BrainStateStore.load(cfg)
        report = run_health_check(cfg, store)

        typer.echo(f"Compaction complete: {summary['merges']} merges.")
        for a, b, sim in summary["pairs"]:
            typer.echo(f"  {b} -> {a} (sim={sim:.4f})")
        typer.echo(
            f"Health: {report['source_count']} sources, "
            f"{report['topic_count']} topics."
        )

    asyncio.run(_compact())


@app.command()
def health() -> None:
    """Run a health check and print a readable summary (Phase 4)."""
    from second_brain.compact.eval import render_health_markdown, run_health_check
    from second_brain.state import BrainStateStore

    cfg = load_config()
    store = BrainStateStore.load(cfg)
    report = run_health_check(cfg, store)

    typer.echo(f"Sources: {report['source_count']}")
    typer.echo(f"Topics: {report['topic_count']}")

    o = report["orphans"]
    typer.echo(
        f"Orphans: {len(o['sources'])} sources, {len(o['topics'])} topics"
    )
    typer.echo(f"Broken links: {len(report['broken_links'])}")
    typer.echo(f"Empty extractions: {len(report['empty_extractions'])}")
    typer.echo(f"Stale topics (>90d): {len(report['stale_topics'])}")
    typer.echo(f"Avg confidence: {report['avg_confidence']:.3f}")
    typer.echo(f"Schema violations: {len(report['schema_violations'])}")

    health_md = render_health_markdown(report)
    typer.echo("---")
    typer.echo(health_md)


@app.command("embedding-swap")
def embedding_swap(
    model: Annotated[str, typer.Argument(help="New OpenRouter embedding model slug.")],
) -> None:
    """Swap embedding models using the blue/green workflow (§12.6)."""

    async def _swap() -> None:
        from second_brain.openrouter_client import OpenRouterClient
        from second_brain.state import BrainStateStore
        from second_brain.vectors.swap import swap_embeddings

        cfg = load_config()
        configure_logging(cfg.brain_root)
        store = BrainStateStore.load(cfg)
        client = OpenRouterClient(cfg)
        try:
            result = await swap_embeddings(cfg, store, client, new_model=model)
        finally:
            await client.close()
        typer.echo(
            f"Embedding swap {result.status}: {result.model} "
            f"dim={result.dim} old_score={result.old_score:.3f} "
            f"new_score={result.new_score:.3f}"
        )

    asyncio.run(_swap())


@app.command()
def serve(
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port to bind (127.0.0.1 only).")
    ] = 8000,
) -> None:
    """Start the web UI server (Phase 5B).  Bound to 127.0.0.1 only."""
    from second_brain.web.app import run_server

    cfg = load_config()
    configure_logging(cfg.brain_root)
    typer.echo(
        f"Serving the brain at http://127.0.0.1:{port} (Ctrl-C to stop)."
    )
    run_server(port=port)


@app.command()
def ask(
    query: Annotated[str, typer.Argument(help="Question to ask the brain.")],
) -> None:
    """Ask a question via the chat agent (Phase 6)."""

    async def _ask() -> None:
        from second_brain.chat import chat_stream
        from second_brain.openrouter_client import OpenRouterClient
        from second_brain.state import BrainStateStore

        def _emit(event: dict) -> None:
            t = event["type"]
            if t == "thinking":
                typer.echo("[thinking] " + event["content"])
            elif t == "reasoning_delta":
                typer.echo("(think) " + event["content"], nl=False)
            elif t == "tool_call":
                typer.echo(
                    "[tool] " + event["tool"] + " -> args=" + str(event["args"])
                )
            elif t == "tool_result":
                hits = event.get("hits", [])
                err = event.get("error")
                if err:
                    typer.echo("[tool] " + event["tool"] + " -> error=" + err)
                else:
                    typer.echo(
                        "[tool] " + event["tool"]
                        + " -> " + str(len(hits)) + " hits"
                    )
                    for h in hits:
                        typer.echo(
                            "  " + h["source_id"]
                            + " topic=" + str(h["topic_slug"])
                            + " score=" + str(h["score"])
                        )
            elif t == "answer_delta":
                typer.echo(event["content"], nl=False)
            elif t == "done":
                typer.echo("")
                typer.echo("[done]")

        cfg = load_config()
        configure_logging(cfg.brain_root)
        client = OpenRouterClient(cfg)
        try:
            store = BrainStateStore.load(cfg)

            # Try the daemon first (single-writer invariant, §12.1).
            remote_hits = await _daemon_search(cfg, query, k=8)

            if remote_hits is not None:
                async def _search_fn(q: str, k: int) -> list[dict]:
                    return await _daemon_search(cfg, q, k=k) or []

                async for event in chat_stream(
                    query,
                    cfg,
                    store,
                    vec_store=None,
                    embedder=None,
                    client=client,
                    k=8,
                    search_fn=_search_fn,
                ):
                    _emit(event)
                return

            async def _empty_search_fn(q: str, k: int) -> list[dict]:  # noqa: ARG001
                return []

            async for event in chat_stream(
                query,
                cfg,
                store,
                vec_store=None,
                embedder=None,
                client=client,
                k=8,
                search_fn=_empty_search_fn,
            ):
                _emit(event)
        finally:
            await client.close()

    asyncio.run(_ask())
