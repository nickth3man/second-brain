"""Regression tests for architecture-audit remediations."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from second_brain.compact.eval import render_health_markdown, run_health_check
from second_brain.daemon.extract import (
    MAP_CHUNK_OVERLAP_TOKENS,
    MAP_CHUNK_TOKENS,
    ExtractionError,
    build_raptor_tree,
    chunk_for_extraction,
    extract,
    plan_extraction,
)
from second_brain.daemon.normalize import normalize_text
from second_brain.frontmatter import dump_frontmatter, split_frontmatter
from second_brain.models import SourceState
from second_brain.state import BrainStateStore, reconcile_filesystem


class FakeClient:
    async def close(self) -> None:
        pass


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        brain_root=tmp_path,
        types=SimpleNamespace(
            text=["txt", "md"],
            code=[],
            structured=[],
            vision=[],
            pdf=[],
            office=[],
            web=[],
            ebook=[],
            audio=[],
            video=[],
        ),
        ingestion=SimpleNamespace(max_audio_minutes=60),
        models=SimpleNamespace(stt="stt"),
    )


async def test_source_frontmatter_has_schema_version(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    inbox = tmp_path / "00-inbox" / "note.txt"
    inbox.parent.mkdir(parents=True)
    inbox.write_text("hello", encoding="utf-8")

    path, _body = await normalize_text(
        inbox,
        "note",
        "abc",
        "2026-06-22T00:00:00Z",
        "text",
        cfg,
        FakeClient(),
    )

    meta, _ = split_frontmatter(path.read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1


def test_reconcile_restores_state_from_filesystem(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    source_dir = tmp_path / "50-sources"
    wiki_dir = tmp_path / "90-wiki"
    source_dir.mkdir(parents=True)
    wiki_dir.mkdir(parents=True)

    source_text = dump_frontmatter(
        {
            "schema_version": 1,
            "source": "00-inbox/note.txt",
            "type": "text",
            "ingested": "2026-06-22T00:00:00Z",
            "sha256": "abc",
            "tokens": 3,
            "topics": ["topic-a"],
        },
        "# Note\n",
    )
    (source_dir / "note.md").write_text(source_text, encoding="utf-8")

    wiki_text = dump_frontmatter(
        {
            "schema_version": 1,
            "title": "Topic A",
            "slug": "topic-a",
            "type": "concept",
            "created": "2026-06-22",
            "updated": "2026-06-22",
            "confidence": 0.8,
            "related": [],
        },
        "# Topic A\n",
    )
    (wiki_dir / "topic-a.md").write_text(wiki_text, encoding="utf-8")

    store = BrainStateStore.load(cfg)
    changed = reconcile_filesystem(cfg, store)

    assert changed is True
    assert "note" in store.state.sources
    assert "topic-a" in store.state.topics
    assert store.state.topics["topic-a"].sources == ["note"]


def test_reconcile_removes_missing_derived_records(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    store = BrainStateStore.load(cfg)
    store.state.sources["missing"] = SourceState(sha256="abc", raw="00-inbox/missing.txt")
    store.ensure_topic("missing-topic", "Missing Topic")
    store.save()

    changed = reconcile_filesystem(cfg, store)

    assert changed is True
    assert store.state.sources == {}
    assert store.state.topics == {}


def test_reconcile_repairs_stale_edges_and_membership(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    source_dir = tmp_path / "50-sources"
    wiki_dir = tmp_path / "90-wiki"
    source_dir.mkdir(parents=True)
    wiki_dir.mkdir(parents=True)
    (source_dir / "note.md").write_text(
        dump_frontmatter(
            {
                "source": "00-inbox/note.txt",
                "type": "text",
                "ingested": "2026-06-22T00:00:00Z",
                "sha256": "abc",
                "tokens": 1,
                "topics": ["topic-a"],
            },
            "body",
        ),
        encoding="utf-8",
    )
    (wiki_dir / "topic-a.md").write_text(
        dump_frontmatter(
            {
                "title": "Topic A",
                "type": "concept",
                "created": "2026-06-22",
                "updated": "2026-06-22",
                "related": ["topic-b", "missing"],
            },
            "# Topic A\n[[Topic B]]",
        ),
        encoding="utf-8",
    )
    (wiki_dir / "topic-b.md").write_text(
        dump_frontmatter(
            {
                "title": "Topic B",
                "type": "concept",
                "created": "2026-06-22",
                "updated": "2026-06-22",
            },
            "# Topic B\n",
        ),
        encoding="utf-8",
    )

    store = BrainStateStore.load(cfg)
    assert reconcile_filesystem(cfg, store) is True

    assert store.state.sources["note"].topics == ["topic-a"]
    assert store.state.topics["topic-a"].sources == ["note"]
    assert store.state.topics["topic-a"].links_to == ["topic-b"]
    assert store.state.topics["topic-b"].linked_from == ["topic-a"]


def test_long_source_plan_uses_map_reduce_without_truncation() -> None:
    body = "a" * (20_000 * 4)
    plan = plan_extraction(body)

    assert plan.strategy == "map_reduce"
    assert "".join(chunk[:1] for chunk in plan.chunks)
    assert "[truncated" not in plan.chunks[0]


def test_chunk_for_extraction_uses_800_token_windows_with_overlap() -> None:
    body = "".join(str(i % 10) for i in range(10_000))
    chunks = chunk_for_extraction(body)
    size = MAP_CHUNK_TOKENS * 4
    overlap = MAP_CHUNK_OVERLAP_TOKENS * 4

    assert len(chunks[0]) == size
    assert chunks[0][-overlap:] == chunks[1][:overlap]


def test_raptor_plan_for_very_long_sources() -> None:
    body = "a" * (210_000 * 4)
    assert plan_extraction(body).strategy == "raptor"


def test_raptor_tree_preserves_traceability_and_recurses() -> None:
    summaries = [f"alpha summary {idx}" for idx in range(12)]
    root = build_raptor_tree(summaries, group_size=3, context_token_budget=1)

    assert root.children
    assert root.chunk_ids == tuple(range(12))
    assert all(child.chunk_ids for child in root.children)
    assert max(child.level for child in root.children) >= 1


class ReducingClient:
    def __init__(self) -> None:
        self.reduce_calls = 0

    async def chat_completion(self, model, messages, **kwargs):  # noqa: ARG002
        content = messages[-1]["content"]
        if "REDUCE PASS" in content:
            self.reduce_calls += 1
            name = "Reduced Topic"
        else:
            name = "Mapped Topic"
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"tldr":"ok","topics":[{"name":"'
                            + name
                            + '","action":"new","target_slug":"","confidence":0.9,'
                            '"merged_section":"section"}]}'
                        )
                    }
                }
            ]
        }


class MalformedReduceClient(ReducingClient):
    async def chat_completion(self, model, messages, **kwargs):  # noqa: ARG002
        content = messages[-1]["content"]
        if "REDUCE PASS" in content:
            self.reduce_calls += 1
            return {"choices": [{"message": {"content": '{"bad": true}'}}]}
        return await super().chat_completion(model, messages, **kwargs)


def _extract_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        extraction=SimpleNamespace(
            enable_healing=False,
            primary_model="primary",
            repair_model="repair",
            confidence_floor=0.6,
            require_parameters=True,
        ),
        models=SimpleNamespace(text="text"),
        privacy=SimpleNamespace(zdr=True, block_training_providers=True),
    )


async def test_map_reduce_calls_explicit_reduce_pass(monkeypatch) -> None:
    monkeypatch.setattr(
        "second_brain.daemon.extract.DIRECT_TOKEN_LIMIT",
        10,
    )
    monkeypatch.setattr(
        "second_brain.daemon.extract.MAP_REDUCE_TOKEN_LIMIT",
        1000,
    )
    client = ReducingClient()

    result = await extract(client, _extract_cfg(), "a" * 200, {}, source_id="s")

    assert client.reduce_calls == 1
    assert result.topics[0].name == "Reduced Topic"


async def test_malformed_reduce_output_fails_without_partial(monkeypatch) -> None:
    monkeypatch.setattr(
        "second_brain.daemon.extract.DIRECT_TOKEN_LIMIT",
        10,
    )
    monkeypatch.setattr(
        "second_brain.daemon.extract.MAP_REDUCE_TOKEN_LIMIT",
        1000,
    )
    client = MalformedReduceClient()

    with pytest.raises(ExtractionError):
        await extract(client, _extract_cfg(), "a" * 200, {}, source_id="s")


async def test_video_keyframe_vision_merges_with_transcript(tmp_path: Path, monkeypatch) -> None:
    from second_brain.parse import video

    video_path = tmp_path / "clip.mp4"
    frame_path = tmp_path / "frame.png"
    video_path.write_bytes(b"video")
    frame_path.write_bytes(b"png")
    cfg = SimpleNamespace(
        brain_root=tmp_path,
        ingestion=SimpleNamespace(
            video_keyframe_vision=True,
            video_keyframe_max_frames=1,
            video_keyframe_cadence_seconds=30,
        ),
        models=SimpleNamespace(vision="vision", stt="stt"),
    )

    async def fake_audio(path, cfg, client):  # noqa: ARG001
        return "spoken transcript"

    def fake_run(args, *args_, **kwargs):  # noqa: ANN001, ARG001
        out = Path(args[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"mp3")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(video.subprocess, "run", fake_run)
    monkeypatch.setattr(video, "parse_audio", fake_audio)
    monkeypatch.setattr(video, "_extract_keyframes", lambda path, cfg, sha: [frame_path])

    class VisionClient:
        async def vision_describe(self, model, images, prompt, *, mime="image/png"):  # noqa: ARG002
            return "diagram visible"

    result = await video.parse_video(video_path, cfg, VisionClient())

    assert "## Transcript" in result
    assert "spoken transcript" in result
    assert "## Visual Observations" in result
    assert "diagram visible" in result


def test_sensitive_provider_requires_zdr(tmp_path: Path) -> None:
    from second_brain.openrouter_client import OpenRouterClient, SensitiveRoutingError

    cfg = _cfg(tmp_path)
    cfg.openrouter = SimpleNamespace(base_url="https://openrouter.ai/api/v1", api_key="x")
    cfg.models.chat = "chat"
    cfg.models.vision = "vision"
    cfg.models.embedding = "embedding"
    cfg.privacy = SimpleNamespace()
    cfg.privacy.zdr = False
    cfg.privacy.block_training_providers = False
    cfg.privacy.api_key_source = "config"
    cfg.extraction = SimpleNamespace()
    cfg.extraction.require_parameters = False
    client = OpenRouterClient(cfg)

    with pytest.raises(SensitiveRoutingError):
        client._zdr_provider(sensitive=True)  # noqa: SLF001


def test_sensitive_provider_payload_is_strict(tmp_path: Path) -> None:
    from second_brain.openrouter_client import OpenRouterClient

    cfg = _cfg(tmp_path)
    cfg.openrouter = SimpleNamespace(base_url="https://openrouter.ai/api/v1", api_key="x")
    cfg.models.chat = "chat"
    cfg.models.vision = "vision"
    cfg.models.embedding = "embedding"
    cfg.privacy = SimpleNamespace()
    cfg.privacy.zdr = True
    cfg.privacy.block_training_providers = False
    cfg.privacy.api_key_source = "config"
    cfg.extraction = SimpleNamespace()
    cfg.extraction.require_parameters = False
    provider = OpenRouterClient(cfg)._zdr_provider(sensitive=True)  # noqa: SLF001

    assert provider["zdr"] is True
    assert provider["require_parameters"] is True
    assert provider["data_collection"] == "deny"


async def test_vector_reconcile_repairs_index_drift(tmp_path: Path) -> None:
    from second_brain.frontmatter import dump_frontmatter
    from second_brain.vectors.reconcile import reconcile_vector_index
    from second_brain.vectors.store import VectorStore

    cfg = _cfg(tmp_path)
    source_dir = tmp_path / "50-sources"
    source_dir.mkdir(parents=True)
    (source_dir / "note.md").write_text(
        dump_frontmatter(
            {"source": "00-inbox/note.txt", "sha256": "newhash", "topics": ["topic-a"]},
            "# Note\nmeaningful body for embedding repair",
        ),
        encoding="utf-8",
    )
    store = BrainStateStore.load(cfg)
    store.state.sources["note"] = SourceState(
        sha256="newhash",
        raw="00-inbox/note.txt",
        topics=["topic-a"],
        embedding_model="old-model",
    )
    store.ensure_topic("topic-a", "Topic A")
    store.state.topics["topic-a"].sources = ["note"]

    vec_store = VectorStore(tmp_path / ".brain" / "embeddings.db", "new-model", dim=3)
    vec_store.upsert_source_chunks(
        "orphan",
        "topic-a",
        [("orphan text", [1.0, 0.0, 0.0])],
        source_hash="orphan",
        embedding_model="new-model",
    )
    vec_store.upsert_source_chunks(
        "note",
        "topic-a",
        [("old text", [0.0, 1.0, 0.0])],
        source_hash="oldhash",
        embedding_model="old-model",
    )
    vec_store.db.execute("DELETE FROM source_chunks_fts")
    vec_store.upsert_topic_centroid("topic-a", [0.0, 1.0, 0.0], member_count=99)
    vec_store.db.execute(
        "INSERT INTO vec_tombstones(rowid, source_id, table_name, ts) VALUES (?, ?, ?, ?)",
        (999, "note", "source_chunks_vec", "2026-06-22T00:00:00Z"),
    )
    vec_store.db.commit()

    class FakeEmbedder:
        async def embed_texts(self, texts):  # noqa: ANN001
            return [[1.0, 0.0, 0.0] for _ in texts]

    try:
        report = await reconcile_vector_index(cfg, store, vec_store, FakeEmbedder())
        assert report.orphan_embeddings == ["orphan"]
        assert report.stale_embeddings == ["note"]
        assert report.fts_mismatches == ["source_chunks_fts"]
        assert report.stale_topic_centroids == ["topic-a"]
        assert report.tombstone_inconsistencies == ["note"]
        remaining = vec_store.db.execute(
            "SELECT source_id FROM vec_tombstones WHERE source_id = 'note'"
        ).fetchall()
        assert remaining == []
        assert vec_store.source_centroid("note") is not None
    finally:
        vec_store.close()


def test_health_surfaces_sampled_metrics(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    store = BrainStateStore.load(cfg)
    store.state.sources["s1"] = SourceState(sha256="abc", raw="00-inbox/a.txt")
    store.save()
    changelog = tmp_path / ".brain" / "changelog.jsonl"
    changelog.write_text(
        '{"ts":"2026-06-22T00:00:00Z","kind":"chat","scores":{"faithfulness":0.8},"usage":{"cost_usd":0.25}}\n'
        '{"ts":"2026-06-22T00:00:00Z","kind":"merge","scores":{"merge_reversibility_pass":true}}\n',
        encoding="utf-8",
    )

    report = run_health_check(cfg, store)
    md = render_health_markdown(report)

    assert report["mean_faithfulness_7d"] == 0.8
    assert report["merge_reversibility_pass_rate_7d"] == 1.0
    assert report["cost_per_active_source_7d"] == 0.25
    assert "Mean faithfulness (7d)" in md


async def test_higher_level_evals_persist_and_surface_skip_status(tmp_path: Path) -> None:
    from second_brain.compact.eval import run_higher_level_evals

    cfg = _cfg(tmp_path)
    cfg.eval = SimpleNamespace(golden_set_dir=".brain/golden")
    store = BrainStateStore.load(cfg)
    result = await run_higher_level_evals(cfg, store)
    report = run_health_check(cfg, store)
    md = render_health_markdown(report)

    assert result["l2"]["status"] == "complete"
    assert result["l3"]["status"] == "skipped"
    assert result["l4"]["status"] == "skipped"
    assert (tmp_path / ".brain" / "evals" / "latest.json").exists()
    assert report["l2_status"] == "complete"
    assert "L3 judge" in md


def test_topic_source_cosine_metric_written(tmp_path: Path) -> None:
    from second_brain.compact.eval import (
        _topic_source_cosine_mean,
        write_topic_source_cosine_metric,
    )
    from second_brain.vectors.store import VectorStore

    cfg = _cfg(tmp_path)
    store = BrainStateStore.load(cfg)
    store.state.sources["note"] = SourceState(
        sha256="abc",
        raw="00-inbox/note.txt",
        topics=["topic-a"],
    )
    store.ensure_topic("topic-a", "Topic A")

    vec_store = VectorStore(tmp_path / ".brain" / "embeddings.db", "test-model", dim=3)
    vec_store.upsert_source_chunks(
        "note",
        "topic-a",
        [("hello world", [1.0, 0.0, 0.0])],
    )
    vec_store.add_topic_member("topic-a", "note")
    vec_store.upsert_topic_centroid("topic-a", [1.0, 0.0, 0.0], member_count=1)

    try:
        write_topic_source_cosine_metric(cfg, store, vec_store)
        mean = _topic_source_cosine_mean(tmp_path)
        assert mean is not None
        assert mean == pytest.approx(1.0)
    finally:
        vec_store.close()


async def test_l3_judge_skipped_when_no_content(tmp_path: Path) -> None:
    from second_brain.compact.eval import _run_l3_judge

    cfg = _cfg(tmp_path)
    cfg.models = SimpleNamespace(
        text="anthropic/claude-3.5-sonnet",
        judge="openai/gpt-4o",
    )
    store = BrainStateStore.load(cfg)

    result = await _run_l3_judge(cfg, store, object())

    assert result["status"] == "skipped"
    assert result["reason"] == "no content to judge"


async def test_l4_golden_set_computes_pass_rate(tmp_path: Path) -> None:
    import json

    from second_brain.compact.eval import _run_l4_golden_set
    from second_brain.vectors.store import VectorStore

    cfg = _cfg(tmp_path)
    cfg.eval = SimpleNamespace(golden_set_dir=".brain/golden")
    golden_dir = tmp_path / ".brain" / "golden"
    golden_dir.mkdir(parents=True)
    (golden_dir / "case1.json").write_text(
        json.dumps({"query": "hello world", "expected_source_id": "note"}),
        encoding="utf-8",
    )

    vec_store = VectorStore(tmp_path / ".brain" / "embeddings.db", "test-model", dim=3)
    vec_store.upsert_source_chunks(
        "note",
        "topic-a",
        [("hello world text", [1.0, 0.0, 0.0])],
    )

    class FakeEmbedder:
        async def embed_texts(self, texts):  # noqa: ANN001
            return [[1.0, 0.0, 0.0] for _ in texts]

    try:
        result = await _run_l4_golden_set(cfg, vec_store, FakeEmbedder())
        assert result["status"] == "complete"
        assert result["golden_cases"] == 1
        assert "pass_rate" in result
        assert result["pass_rate"] == 1.0
    finally:
        vec_store.close()


def test_l1_hash_stability_metric(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    raw = tmp_path / "00-inbox" / "note.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("stable", encoding="utf-8")
    import hashlib

    store = BrainStateStore.load(cfg)
    store.state.sources["note"] = SourceState(
        sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
        raw="00-inbox/note.txt",
    )

    report = run_health_check(cfg, store)

    assert report["hash_stability_pass_rate"] == 1.0


def test_lsh_near_duplicate_prefilter_scales_to_1000_sources() -> None:
    import math

    from second_brain.compact.dedup import _lsh_candidate_pairs

    source_ids = [f"s{i:04d}" for i in range(1000)]
    embeddings = {}
    for idx, source_id in enumerate(source_ids):
        if idx < 2:
            embeddings[source_id] = [1.0, 0.0, 0.0, 0.0]
            continue
        angle = (idx / 998) * 2 * math.pi
        embeddings[source_id] = [math.cos(angle), math.sin(angle), 0.0, 0.0]

    pairs = _lsh_candidate_pairs(source_ids, embeddings)

    assert ("s0000", "s0001") in pairs
    assert len(pairs) < (1000 * 999) // 2


async def test_zdr_verification_reports_verified_and_manual_account(tmp_path: Path) -> None:
    from second_brain.openrouter_client import OpenRouterClient

    cfg = _cfg(tmp_path)
    cfg.openrouter = SimpleNamespace(base_url="https://openrouter.ai/api/v1", api_key="x")
    cfg.privacy = SimpleNamespace()
    cfg.privacy.api_key_source = "config"

    class Response:
        status_code = 200

    class Http:
        async def get(self, path):  # noqa: ARG002
            return Response()

    client = OpenRouterClient(cfg)
    client._client = Http()  # noqa: SLF001

    status = await client.verify_zdr_status()

    assert status["request_level_zdr_endpoint"] == "verified"
    assert status["account_level_zdr"] == "manual_unconfirmed"


async def test_zdr_verification_reports_unavailable(tmp_path: Path) -> None:
    from second_brain.openrouter_client import OpenRouterClient

    cfg = _cfg(tmp_path)
    cfg.openrouter = SimpleNamespace(base_url="https://openrouter.ai/api/v1", api_key="x")
    cfg.privacy = SimpleNamespace()
    cfg.privacy.api_key_source = "config"

    class Http:
        async def get(self, path):  # noqa: ARG002
            raise RuntimeError("offline")

    client = OpenRouterClient(cfg)
    client._client = Http()  # noqa: SLF001

    status = await client.verify_zdr_status()

    assert status["request_level_zdr_endpoint"] == "unavailable"
    assert status["account_level_zdr"] == "manual_unconfirmed"


def test_cli_query_fallback_does_not_open_vector_store() -> None:
    import second_brain.cli as cli

    source = inspect.getsource(cli.search)
    assert "VectorStore(" not in source
    assert "Semantic search unavailable" in source


def test_web_app_uses_lifespan_not_deprecated_on_event() -> None:
    text = Path("src/second_brain/web/app.py").read_text(encoding="utf-8")
    assert "@app.on_event" not in text
    assert "lifespan=" in text


def test_architecture_status_no_longer_ready_to_build() -> None:
    text = Path("ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "ready to build" not in text


@pytest.mark.skipif(
    not pytest.importorskip("os").environ.get("OPENROUTER_API_KEY")
    or not pytest.importorskip("os").environ.get("SECOND_BRAIN_LIVE_OPENROUTER"),
    reason=(
        "opt-in live OpenRouter smoke test requires OPENROUTER_API_KEY and "
        "SECOND_BRAIN_LIVE_OPENROUTER"
    ),
)
async def test_live_openrouter_chat_stream_contract_opt_in(tmp_path: Path) -> None:
    """Opt-in provider validation placeholder; normal test runs never call network."""
    from second_brain.chat import chat_stream

    cfg = _cfg(tmp_path)
    cfg.openrouter = SimpleNamespace(base_url="https://openrouter.ai/api/v1", api_key="")
    cfg.models.chat = "anthropic/claude-3.5-haiku"
    cfg.privacy.api_key_source = "env"
    cfg.privacy.zdr = True
    cfg.privacy.block_training_providers = False
    cfg.extraction.require_parameters = True
    store = BrainStateStore.load(cfg)
    events = []
    async for event in chat_stream("Say hello briefly.", cfg, store):
        events.append(event["type"])
    assert "thinking" in events
    assert "done" in events


def test_pyproject_declares_literal_chat_stack() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"pydantic-ai"' in text
    assert '"fastapi>=0.135"' in text


async def test_l3_judge_scores_with_mock_client(tmp_path: Path) -> None:
    """Happy-path: mock client returns valid JSON; scoring pipeline works end-to-end."""
    from second_brain.compact.eval import _run_l3_judge
    from second_brain.frontmatter import dump_frontmatter
    from second_brain.models import SourceState

    cfg = _cfg(tmp_path)
    cfg.models = SimpleNamespace(
        text="anthropic/claude-3.5-sonnet",
        judge="openai/gpt-4o",
    )

    # Seed a source with real body content and a matching wiki page.
    source_dir = tmp_path / "50-sources"
    wiki_dir = tmp_path / "90-wiki"
    source_dir.mkdir(parents=True)
    wiki_dir.mkdir(parents=True)

    (source_dir / "note.md").write_text(
        dump_frontmatter(
            {
                "source": "00-inbox/note.txt",
                "type": "text",
                "ingested": "2026-06-22T00:00:00Z",
                "sha256": "abc",
                "tokens": 20,
                "topics": ["topic-a"],
            },
            "# Note\nThe Lakers won the 2009 NBA championship.",
        ),
        encoding="utf-8",
    )
    (wiki_dir / "topic-a.md").write_text(
        dump_frontmatter(
            {
                "title": "Topic A",
                "type": "concept",
                "created": "2026-06-22",
                "updated": "2026-06-22",
            },
            "# Topic A\n\n## Synthesis\nThe Los Angeles Lakers won the 2009 title.\n\n## Sources\n",
        ),
        encoding="utf-8",
    )

    store = BrainStateStore.load(cfg)
    store.state.sources["note"] = SourceState(
        sha256="abc",
        raw="00-inbox/note.txt",
        topics=["topic-a"],
    )
    store.ensure_topic("topic-a", "Topic A")

    class MockClient:
        async def chat_completion_clean(self, model, messages, **kwargs):  # noqa: ANN001, ARG002
            return (None, '{"score": 0.9, "notes": "well grounded"}')

    result = await _run_l3_judge(cfg, store, MockClient())

    assert result["status"] == "complete"
    assert result["model"] == "openai/gpt-4o"
    assert result["score"] == pytest.approx(0.9)
    assert result["samples"] == 1
    assert result["raw"][0]["score"] == pytest.approx(0.9)


async def test_l3_judge_scores_with_fenced_json_response(tmp_path: Path) -> None:
    """_parse_judge_score handles JSON wrapped in markdown code fences."""
    from second_brain.compact.eval import _parse_judge_score

    fenced = '```json\n{"score": 0.75, "notes": "supported"}\n```'
    assert _parse_judge_score(fenced) == pytest.approx(0.75)

    plain = '{"score": 0.5, "notes": "partial"}'
    assert _parse_judge_score(plain) == pytest.approx(0.5)

    malformed = "I cannot score this."
    assert _parse_judge_score(malformed) is None


def test_l3_extract_synthesis_stops_at_known_sections(tmp_path: Path) -> None:
    """_extract_synthesis must treat ## headings inside synthesis as content,
    only stopping at Sources/Related/etc."""
    from second_brain.compact.eval import _extract_synthesis

    wiki_body = (
        "# Topic\n\n"
        "## Synthesis\n"
        "## 2009 NBA Season\n"
        "The Lakers won.\n"
        "## Key Players\n"
        "LeBron James led scoring.\n\n"
        "## Sources\n"
        "- source entry\n"
    )
    synthesis = _extract_synthesis(wiki_body)
    assert "Lakers won" in synthesis
    assert "LeBron James" in synthesis
    assert "Sources" not in synthesis
    assert "source entry" not in synthesis


def test_citation_format_pass_rate_detects_continuation_links(tmp_path: Path) -> None:
    """Pass rate should be 1.0 when source links are on the '  -> [source](...)' line."""
    from second_brain.compact.eval import _citation_format_pass_rate
    from second_brain.frontmatter import dump_frontmatter

    wiki_dir = tmp_path / "90-wiki"
    wiki_dir.mkdir()
    (wiki_dir / "topic.md").write_text(
        dump_frontmatter(
            {"title": "Topic", "type": "concept", "created": "2026-06-22", "updated": "2026-06-22"},
            (
                "# Topic\n\n"
                "## Synthesis\nSome content.\n\n"
                "## Sources\n"
                "- **[2026-06-22]** Topic\n"
                "  -> [source](../50-sources/2026-06-22-note.md)\n"
                "  > tldr line\n\n"
                "## Related\n"
            ),
        ),
        encoding="utf-8",
    )

    rate = _citation_format_pass_rate(tmp_path)
    assert rate == pytest.approx(1.0)


def test_citation_format_pass_rate_misses_missing_link(tmp_path: Path) -> None:
    """Pass rate should be 0.0 when a source entry has no [source](...) link."""
    from second_brain.compact.eval import _citation_format_pass_rate
    from second_brain.frontmatter import dump_frontmatter

    wiki_dir = tmp_path / "90-wiki"
    wiki_dir.mkdir()
    (wiki_dir / "topic.md").write_text(
        dump_frontmatter(
            {"title": "Topic", "type": "concept", "created": "2026-06-22", "updated": "2026-06-22"},
            (
                "# Topic\n\n"
                "## Sources\n"
                "- **[2026-06-22]** Topic\n"
                "  > tldr line (no source link here)\n\n"
            ),
        ),
        encoding="utf-8",
    )

    rate = _citation_format_pass_rate(tmp_path)
    assert rate == pytest.approx(0.0)
