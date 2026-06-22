"""Clear VTT source state and 50-sources files for fresh re-ingest."""
from __future__ import annotations

from second_brain.config import load_config
from second_brain.state import BrainStateStore


def main() -> None:
    cfg = load_config()
    store = BrainStateStore.load(cfg)
    vtt_ids = sorted(
        p.stem
        for p in (cfg.brain_root / "50-sources").glob("2026-06-22-0*-every*.md")
    )
    print(f"Clearing {len(vtt_ids)} VTT sources: {vtt_ids}")
    for source_id in vtt_ids:
        if source_id in store.state.sources:
            del store.state.sources[source_id]
            print(f"  removed state: {source_id}")
        for topic in store.state.topics.values():
            if source_id in topic.sources:
                topic.sources = [s for s in topic.sources if s != source_id]
        src_file = cfg.brain_root / "50-sources" / f"{source_id}.md"
        if src_file.exists():
            src_file.unlink()
            print(f"  deleted: {src_file.name}")
    store.save()
    print("State saved.")


if __name__ == "__main__":
    main()
