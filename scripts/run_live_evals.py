"""One-shot live eval runner. Run after ingestion + compaction."""
import asyncio, json, os
from pathlib import Path

os.environ["SECOND_BRAIN_RUN_LLM_EVALS"] = "1"

async def main() -> None:
    from second_brain.config import load_config
    from second_brain.compact.eval import run_higher_level_evals
    from second_brain.openrouter_client import OpenRouterClient
    from second_brain.state import BrainStateStore
    from second_brain.vectors.embed import Embedder
    from second_brain.vectors.store import VectorStore

    cfg = load_config()
    store = BrainStateStore.load(cfg)
    client = OpenRouterClient(cfg)
    embedder = Embedder(client, cfg)
    dim = await embedder.ensure_dim()
    vec_store = VectorStore(
        cfg.brain_root / ".brain/embeddings.db",
        cfg.models.embedding,
        dim=dim,
    )
    try:
        result = await run_higher_level_evals(
            cfg, store, client, vec_store=vec_store, embedder=embedder
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error running higher level evals: {e}")
        import traceback
        traceback.print_exc()
    finally:
        vec_store.close()
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
