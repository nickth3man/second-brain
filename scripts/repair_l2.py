"""One-shot L2 state repair script.

Clears state entries for 2026-06-22-test-note and 2026-06-22-games,
removes their 50-sources/ files, and re-ingests the originals so the
source front-matter topics list is consistent with wiki state.

Run ONCE after initial batch ingestion exposed the mismatch.
"""
from __future__ import annotations

import asyncio

SOURCE_IDS_TO_CLEAR = [
    "2026-06-22-test-note",
    "2026-06-22-games",
]


def clear_state_and_sources() -> None:
    from second_brain.config import load_config
    from second_brain.state import BrainStateStore

    cfg = load_config()
    store = BrainStateStore.load(cfg)

    for source_id in SOURCE_IDS_TO_CLEAR:
        if source_id in store.state.sources:
            print(f"Removing source from state: {source_id}")
            del store.state.sources[source_id]

        # Also remove this source from any topic's sources list
        for slug, topic in store.state.topics.items():
            if source_id in topic.sources:
                print(f"  Removing {source_id} from topic '{slug}' sources list")
                topic.sources = [s for s in topic.sources if s != source_id]

        # Remove the 50-sources/ file so the source_id slug is freed for re-ingest
        source_file = cfg.brain_root / "50-sources" / f"{source_id}.md"
        if source_file.exists():
            print(f"Deleting 50-sources file: {source_file.name}")
            source_file.unlink()

    store.save()
    print("State saved.")
    return cfg


async def reingest_files(cfg) -> None:
    from second_brain.daemon.index import DebouncedIndex
    from second_brain.daemon.linker import EmbeddingLinker
    from second_brain.daemon.pipeline import ingest_file
    from second_brain.openrouter_client import OpenRouterClient
    from second_brain.state import BrainStateStore
    from second_brain.vectors.embed import Embedder
    from second_brain.vectors.store import VectorStore

    files = [
        cfg.brain_root / "00-inbox" / "test-note.md",
        cfg.brain_root / "00-inbox" / "Games.csv",
    ]

    store = BrainStateStore.load(cfg)
    client = OpenRouterClient(cfg)
    embedder = Embedder(client, cfg)
    dim = await embedder.ensure_dim()
    vec_store = VectorStore(cfg.brain_root / ".brain/embeddings.db", cfg.models.embedding, dim=dim)
    linker = EmbeddingLinker(embedder, vec_store, cfg.ingestion.merge_threshold)
    index = DebouncedIndex(cfg, store)

    try:
        for f in files:
            if not f.exists():
                print(f"WARNING: {f.name} not found, skipping.")
                continue
            progress: list[dict] = []
            stage = await ingest_file(
                f, cfg, store, client, linker, index,
                embedder=embedder, vec_store=vec_store,
                progress=progress,
            )
            print(f"\n{f.name}: {stage}")
            print(f"{'Stage':<12}{'Model':<30}{'Status':<7}Notes")
            print("-" * 60)
            for row in progress:
                print(
                    f"{row.get('stage', ''):<12}"
                    f"{(row.get('model') or ''):<30}"
                    f"{row.get('status', ''):<7}"
                    f"{row.get('notes') or ''}"
                )
        await index.flush_now()
    finally:
        vec_store.close()
        await client.close()


if __name__ == "__main__":
    cfg = clear_state_and_sources()
    asyncio.run(reingest_files(cfg))
    print("\nL2 repair complete.")
