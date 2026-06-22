"""Blue/green embedding model swap (§12.6)."""

from __future__ import annotations

import json
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from second_brain.frontmatter import split_frontmatter
from second_brain.state import BrainStateStore, now_iso
from second_brain.vectors.embed import Embedder
from second_brain.vectors.store import VectorStore, chunk_text

QUALITY_ROLLBACK_THRESHOLD = 0.10

ScoreFn = Callable[[Path], Awaitable[float]]


@dataclass(frozen=True)
class EmbeddingSwapResult:
    status: str
    old_score: float
    new_score: float
    model: str
    dim: int


async def swap_embeddings(
    cfg,
    store: BrainStateStore,
    client,
    *,
    new_model: str,
    dim: int | None = None,
    score_fn: ScoreFn | None = None,
) -> EmbeddingSwapResult:
    """Build a new embedding DB, validate it, atomically swap, or rollback.

    The active ``.brain/embeddings.db`` is never overwritten in place. A new
    database is built as ``embeddings.new/embeddings.db`` and moved into place
    only after shadow eval passes. ``state.json`` carries
    ``embedding_swap_in_progress`` so startup can resume/clean an interrupted
    swap.
    """
    brain = cfg.brain_root / ".brain"
    active_db = brain / "embeddings.db"
    new_dir = brain / "embeddings.new"
    rollback_dir = brain / "embeddings.rollback"
    new_db = new_dir / "embeddings.db"

    if store.state.embedding_swap_in_progress:
        recover_embedding_swap(cfg, store)

    old_model = store.state.embedding_model or getattr(cfg.models, "embedding", "")
    old_dim = store.state.embedding_dim or 0
    store.state.embedding_swap_in_progress = {
        "from_model": old_model,
        "to_model": new_model,
        "started": now_iso(),
        "stage": "building",
    }
    store.save()

    if new_dir.exists():
        shutil.rmtree(new_dir)
    new_dir.mkdir(parents=True)

    embedder = Embedder(client, cfg)
    embedder.model = new_model
    if dim is None:
        dim = await embedder.ensure_dim()
    else:
        embedder.dim = dim
        embedder._dim_probed = True  # noqa: SLF001

    vec_store = VectorStore(new_db, new_model, dim=dim)
    try:
        for source_id, source in store.state.sources.items():
            source_path = cfg.brain_root / "50-sources" / f"{source_id}.md"
            if not source_path.is_file():
                continue
            _meta, body = split_frontmatter(source_path.read_text(encoding="utf-8"))
            chunks = chunk_text(body)
            embeddings = await embedder.embed_texts(chunks)
            for topic_slug in source.topics or [""]:
                vec_store.upsert_source_chunks(
                    source_id,
                    topic_slug,
                    list(zip(chunks, embeddings, strict=False)),
                    source_hash=source.sha256,
                    embedding_model=new_model,
                )
                if topic_slug:
                    vec_store.add_topic_member(topic_slug, source_id)
                    vec_store.recompute_centroid(topic_slug)
    finally:
        vec_store.close()

    if score_fn is None:
        test_set = build_or_load_shadow_test_set(cfg, store)
        old_score = await evaluate_embedding_db(
            active_db,
            cfg,
            old_model,
            old_dim or dim,
            client,
            test_set,
        )
        new_score = await evaluate_embedding_db(new_db, cfg, new_model, dim, client, test_set)
    else:
        old_score = await _score(active_db, score_fn)
        new_score = await _score(new_db, score_fn)
    _persist_shadow_eval(cfg, old_model, new_model, old_score, new_score)
    if old_score > 0 and new_score < old_score * (1.0 - QUALITY_ROLLBACK_THRESHOLD):
        shutil.rmtree(new_dir, ignore_errors=True)
        store.state.embedding_swap_in_progress = None
        store.save()
        return EmbeddingSwapResult("rolled_back", old_score, new_score, new_model, dim)

    store.state.embedding_swap_in_progress = {
        **(store.state.embedding_swap_in_progress or {}),
        "stage": "swapping",
    }
    store.save()
    if rollback_dir.exists():
        shutil.rmtree(rollback_dir)
    rollback_dir.mkdir(parents=True)
    if active_db.exists():
        shutil.move(str(active_db), str(rollback_dir / "embeddings.db"))
    shutil.move(str(new_db), str(active_db))
    shutil.rmtree(new_dir, ignore_errors=True)

    store.state.embedding_model = new_model
    store.state.embedding_dim = dim
    for source in store.state.sources.values():
        source.embedding_model = new_model
    store.state.embedding_swap_in_progress = None
    store.save()
    shutil.rmtree(rollback_dir, ignore_errors=True)
    return EmbeddingSwapResult("swapped", old_score, new_score, new_model, dim)


def recover_embedding_swap(cfg, store: BrainStateStore) -> bool:
    """Complete or clean an interrupted swap without mutating inbox files."""
    brain = cfg.brain_root / ".brain"
    active_db = brain / "embeddings.db"
    new_dir = brain / "embeddings.new"
    rollback_db = brain / "embeddings.rollback" / "embeddings.db"
    changed = False
    if not active_db.exists() and rollback_db.exists():
        shutil.move(str(rollback_db), str(active_db))
        changed = True
    shutil.rmtree(new_dir, ignore_errors=True)
    shutil.rmtree(brain / "embeddings.rollback", ignore_errors=True)
    if store.state.embedding_swap_in_progress is not None:
        store.state.embedding_swap_in_progress = None
        store.save()
        changed = True
    return changed


async def _score(db_path: Path, score_fn: ScoreFn | None) -> float:
    if score_fn is None:
        return 1.0 if db_path.exists() else 0.0
    return float(await score_fn(db_path))


def build_or_load_shadow_test_set(cfg, store: BrainStateStore) -> list[dict[str, str]]:
    """Build or load representative cached retrieval eval cases."""
    path = cfg.brain_root / ".brain" / "evals" / "embedding-shadow-testset.json"
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cases = data.get("cases", [])
            if cases:
                return cases
        except json.JSONDecodeError:
            pass
    cases: list[dict[str, str]] = []
    for source_id in sorted(store.state.sources):
        source_path = cfg.brain_root / "50-sources" / f"{source_id}.md"
        if not source_path.is_file():
            continue
        _meta, body = split_frontmatter(source_path.read_text(encoding="utf-8"))
        query = _first_meaningful_line(body)
        if query:
            cases.append({"query": query[:500], "expected_source_id": source_id})
        if len(cases) >= 50:
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": cases}, indent=2), encoding="utf-8")
    return cases


async def evaluate_embedding_db(
    db_path: Path,
    cfg,
    model: str,
    dim: int,
    client,
    test_set: list[dict[str, str]],
) -> float:
    """Score retrieval hit@5 over a cached representative test set."""
    if not test_set:
        return 1.0 if db_path.exists() else 0.0
    if not db_path.exists() or dim <= 0:
        return 0.0
    embedder = Embedder(client, cfg)
    embedder.model = model
    embedder.dim = dim
    embedder._dim_probed = True  # noqa: SLF001
    vec_store = VectorStore(db_path, model, dim=dim)
    hits = 0
    try:
        for case in test_set:
            query_vec = await embedder.embed_one(case["query"])
            rowids = vec_store.vector_search_chunks(query_vec, k=5)
            source_ids = {
                chunk["source_id"]
                for rowid, _score in rowids
                if (chunk := vec_store.get_chunk(rowid)) is not None
            }
            if case["expected_source_id"] in source_ids:
                hits += 1
    finally:
        vec_store.close()
    return hits / len(test_set)


def _first_meaningful_line(body: str) -> str:
    for line in body.splitlines():
        clean = line.strip().lstrip("#-*` ")
        if len(clean) >= 20 and not clean.startswith("---"):
            return clean
    return ""


def _persist_shadow_eval(
    cfg,
    old_model: str,
    new_model: str,
    old_score: float,
    new_score: float,
) -> None:
    path = cfg.brain_root / ".brain" / "evals" / "embedding-shadow-eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "old_model": old_model,
                "new_model": new_model,
                "old_score": old_score,
                "new_score": new_score,
                "rollback_threshold": QUALITY_ROLLBACK_THRESHOLD,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
