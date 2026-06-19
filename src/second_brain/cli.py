"""Second Brain CLI — ``brain`` entry point.

Usage::

    brain init      # scaffold folders + keyring
    brain watch     # start file-watcher daemon (Phase 1)
    brain rebuild   # rebuild wiki (Phase 4)
    brain ask       # interactive chat (Phase 6)
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
    """Start the file-watcher daemon (Phase 1 — text-only ingestion loop)."""
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
        from second_brain.daemon.linker import SlugLinker
        from second_brain.daemon.pipeline import ingest_file
        from second_brain.openrouter_client import OpenRouterClient
        from second_brain.state import BrainStateStore

        cfg = load_config()
        client = OpenRouterClient(cfg)
        store = BrainStateStore.load(cfg)
        linker = SlugLinker()
        index = DebouncedIndex(cfg, store)
        try:
            stage = await ingest_file(
                path, cfg, store, client, linker, index
            )
            await index.flush_now()
            return str(stage)
        finally:
            await client.close()

    stage = asyncio.run(_ingest_one())
    typer.echo(f"{path.name}: {stage}")


@app.command()
def rebuild() -> None:
    """Rebuild wiki from sources or inbox (Phase 4)."""
    typer.echo("Not yet implemented (Phase 4).")


@app.command()
def ask() -> None:
    """Ask a question via the chat interface (Phase 6)."""
    typer.echo("Not yet implemented (Phase 6).")
