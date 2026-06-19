"""Second Brain CLI — ``brain`` entry point.

Usage::

    brain init      # scaffold folders + keyring
    brain watch     # start file-watcher daemon (Phase 1)
    brain ingest   # ingest one file (Phase 1/2)
    brain search   # hybrid search (Phase 2)
    brain rebuild  # rebuild wiki (Phase 4)
    brain compact  # run compaction pass (Phase 4)
    brain health   # run health check (Phase 4)
    brain ask      # interactive chat (Phase 6)
    brain --version # show version
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from second_brain import __version__
from second_brain.config import load_config

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
    typer.echo("Second Brain initialized successfully.")


@app.command()
def watch() -> None:
    """Start the file-watcher daemon (Phase 1 — text-only ingestion loop)."""
    from second_brain.daemon.pipeline import run_daemon

    cfg = load_config()
    typer.echo(f"Watching {cfg.brain_root / '00-inbox'}/ ... Ctrl-C to stop.")
    asyncio.run(run_daemon(cfg))


@app.command()
def ingest(
    path: Annotated[
        Path, typer.Argument(help="Path to the file to ingest (one-shot, for testing).")
    ],
) -> None:
    """Ingest a single file (one-shot, without the watcher)."""

    async def _ingest_one() -> str:
        from second_brain.daemon.index import DebouncedIndex
        from second_brain.daemon.linker import EmbeddingLinker
        from second_brain.daemon.pipeline import ingest_file
        from second_brain.openrouter_client import OpenRouterClient
        from second_brain.state import BrainStateStore
        from second_brain.vectors.embed import Embedder
        from second_brain.vectors.store import VectorStore

        cfg = load_config()
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
            stage = await ingest_file(
                path, cfg, store, client, linker, index,
                embedder=embedder, vec_store=vec_store,
            )
            await index.flush_now()
            return str(stage)
        finally:
            vec_store.close()
            await client.close()

    stage = asyncio.run(_ingest_one())
    typer.echo(f"{path.name}: {stage}")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query.")],
) -> None:
    """Search the brain using hybrid retrieval (Phase 2)."""

    async def _search() -> None:
        from second_brain.openrouter_client import OpenRouterClient
        from second_brain.vectors.embed import Embedder
        from second_brain.vectors.retrieval import search_brain
        from second_brain.vectors.store import VectorStore

        cfg = load_config()
        client = OpenRouterClient(cfg)
        embedder = Embedder(client, cfg)
        dim = await embedder.ensure_dim()
        vec_store = VectorStore(
            cfg.brain_root / ".brain/embeddings.db",
            cfg.models.embedding,
            dim=dim,
        )
        try:
            hits = await search_brain(query, vec_store, embedder, k=5)
            if not hits:
                typer.echo("No results.")
                return
            for i, hit in enumerate(hits, 1):
                snippet = hit.text[:120].replace("\n", " ")
                typer.echo(
                    f"{i}. [{hit.source_id}] "
                    f"(topic={hit.topic_slug}, score={hit.score:.4f})\n"
                    f"   {snippet}\n"
                )
        finally:
            vec_store.close()
            await client.close()

    asyncio.run(_search())


@app.command()
def rebuild() -> None:
    """Rebuild wiki from sources or inbox (Phase 4)."""
    typer.echo("Not yet implemented (Phase 4).")


@app.command()
def compact() -> None:
    """Run one compaction pass (Phase 4 — merge similar topics)."""

    async def _compact() -> None:
        from second_brain.compact.compaction import run_compaction
        from second_brain.compact.eval import render_health_markdown, run_health_check
        from second_brain.openrouter_client import OpenRouterClient
        from second_brain.state import BrainStateStore
        from second_brain.vectors.embed import Embedder
        from second_brain.vectors.store import VectorStore

        cfg = load_config()
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
                merge_threshold=cfg.ingestion.merge_threshold,
            )

            report = run_health_check(cfg, store)
            health_md = render_health_markdown(report)

            # Append health section to INDEX.md
            index_path = cfg.brain_root / "INDEX.md"
            if index_path.exists():
                text = index_path.read_text(encoding="utf-8")
                text += "\n\n" + health_md
                from second_brain.atomicio import write_atomic

                write_atomic(index_path, text)

            typer.echo(f"Compaction complete: {summary['merges']} merges.")
            for a, b, sim in summary["pairs"]:
                typer.echo(f"  {b} -> {a} (sim={sim:.4f})")
            typer.echo(
                f"Health: {report['source_count']} sources, "
                f"{report['topic_count']} topics."
            )
        finally:
            vec_store.close()
            await client.close()

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


@app.command()
def serve(
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port to bind (127.0.0.1 only).")
    ] = 8000,
) -> None:
    """Start the web UI server (Phase 5B).  Bound to 127.0.0.1 only."""
    from second_brain.web.app import run_server

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
        from second_brain.vectors.embed import Embedder
        from second_brain.vectors.store import VectorStore

        cfg = load_config()
        client = OpenRouterClient(cfg)
        try:
            embedder = Embedder(client, cfg)
            dim = await embedder.ensure_dim()
            vec_store = VectorStore(
                cfg.brain_root / ".brain/embeddings.db",
                cfg.models.embedding,
                dim=dim,
            )
            store = BrainStateStore.load(cfg)
            try:
                async for event in chat_stream(
                    query, cfg, store, vec_store, embedder, client, k=8
                ):
                    t = event["type"]
                    if t == "thinking":
                        typer.echo("[thinking] " + event["content"])
                    elif t == "reasoning_delta":
                        typer.echo("(think) " + event["content"], nl=False)
                    elif t == "tool_call":
                        typer.echo(
                            "[tool] " + event["tool"]
                            + " -> args=" + str(event["args"])
                        )
                    elif t == "tool_result":
                        hits = event.get("hits", [])
                        err = event.get("error")
                        if err:
                            typer.echo(
                                "[tool] " + event["tool"]
                                + " -> error=" + err
                            )
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
            finally:
                vec_store.close()
        finally:
            await client.close()

    asyncio.run(_ask())
