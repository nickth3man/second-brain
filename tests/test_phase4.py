"""Phase 4 tests — health check, cosine, compaction merge, near-dup dedup.

No network, no API.  Uses ``_FakeCfg`` / ``FakeClient`` / ``FakeEmbedder``
stubs (same pattern as test_phase1b / test_phase2b).  Real sqlite-vec with
dim=8 for vector tests.

References
----------
- ARCHITECTURE.md §8 (compaction)
- ARCHITECTURE.md §11 (anti-graveyard, near-dup, health report)
- ARCHITECTURE.md §12.5 item 4a (eval MVP slice)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from second_brain.compact.compaction import run_compaction
from second_brain.compact.dedup import cosine, find_near_duplicates
from second_brain.compact.eval import render_health_markdown, run_health_check
from second_brain.config import TypesCfg
from second_brain.models import IngestStage, PageType, SourceState, TopicState
from second_brain.state import BrainStateStore
from second_brain.vectors.store import VectorStore

DIM = 8
MERGE_THRESHOLD = 0.85
DEDUP_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Stubs (same pattern as test_phase1b / test_phase2b)
# ---------------------------------------------------------------------------


@dataclass
class _FakeExtraction:
    primary_model: str = "test-model"
    repair_model: str = "test-repair-model"
    enable_healing: bool = False
    deadletter_dir: str = ".brain/deadletter"
    max_attempts: int = 2
    require_parameters: bool = False
    confidence_floor: float = 0.6
    quarantine_dir: str = ".brain/quarantine"


@dataclass
class _FakeModels:
    text: str = "test-model"
    vision: str = "test-vision"
    embedding: str = "test-embed"
    stt: str = "test-stt"
    chat: str = "test-chat"
    judge: str = "test-judge"


@dataclass
class _FakeIngestion:
    merge_threshold: float = MERGE_THRESHOLD
    pdf_dpi: int = 200
    pdf_image_format: str = "png"
    pdf_alpha: bool = False
    vision_max_images_per_request: int = 8
    vision_max_edge_px: int = 2048
    max_audio_minutes: int = 120


@dataclass
class _FakeCfg:
    brain_root: Path
    types: TypesCfg
    extraction: _FakeExtraction = field(default_factory=_FakeExtraction)
    models: _FakeModels = field(default_factory=_FakeModels)
    ingestion: _FakeIngestion = field(default_factory=_FakeIngestion)


def _make_cfg(tmp_path: Path) -> _FakeCfg:
    return _FakeCfg(
        brain_root=tmp_path,
        types=TypesCfg(
            text=["md", "txt", "markdown"],
            code=["py", "js", "ts"],
            structured=["json", "yaml", "toml"],
            vision=[],
            pdf=[],
            office=[],
            web=[],
            ebook=[],
            audio=[],
            video=[],
        ),
    )


def _ensure_dirs(cfg: _FakeCfg) -> None:
    (cfg.brain_root / "00-inbox").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / "50-sources").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / "90-wiki").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / ".brain").mkdir(parents=True, exist_ok=True)


def _vec(first: float, second: float = 0.0) -> list[float]:
    """Deterministic DIM-dimensional vector."""
    return [first] + [second] + [0.0] * (DIM - 2)


class FakeClient:
    """Fake OpenRouter client returning configurable payloads."""

    def __init__(self, payload: dict | None = None):
        self.payload = payload or {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Merged synthesis content integrating both topics with "
                            "enough detail to pass validation gates."
                        )
                    }
                }
            ]
        }

    async def chat_completion(
        self,
        model: str,  # noqa: ARG002
        messages: list[dict],  # noqa: ARG002
        *,
        response_format: dict | None = None,  # noqa: ARG002
        extra_body: dict | None = None,  # noqa: ARG002
        stream: bool = False,  # noqa: ARG002
    ) -> dict:
        return self.payload

    async def chat_completion_clean(
        self,
        model: str,  # noqa: ARG002
        messages: list[dict],  # noqa: ARG002
        *,
        response_format: dict | None = None,  # noqa: ARG002
        extra_body: dict | None = None,  # noqa: ARG002
    ) -> tuple[str | None, str]:
        from second_brain.reasoning import strip_think

        content = self.payload["choices"][0]["message"]["content"]
        return strip_think(content)

    async def close(self) -> None:
        pass


class FakeEmbedder:
    """Deterministic embedder — no API calls."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self._vectors: dict[str, list[float]] = {}

    def set_vector(self, text: str, vec: list[float]) -> None:
        self._vectors[text] = vec

    async def embed_one(self, text: str) -> list[float]:
        if text in self._vectors:
            return self._vectors[text]
        return [0.0] * self.dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_one(t) for t in texts]

    async def embed_query(self, query: str) -> list[float]:
        return await self.embed_one(query)


# ---------------------------------------------------------------------------
# TestHealthCheck
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """run_health_check with edge cases."""

    def test_healthy_empty(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)
        report = run_health_check(cfg, store)

        assert report["source_count"] == 0
        assert report["topic_count"] == 0
        assert report["orphans"]["sources"] == []
        assert report["orphans"]["topics"] == []
        assert report["broken_links"] == []
        assert report["near_duplicates"] == []
        assert report["empty_extractions"] == []
        assert report["stale_topics"] == []
        assert report["avg_confidence"] == 0.0
        assert report["schema_violations"] == []

    def test_normal_topic(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)

        store.ensure_topic("healthy", "Healthy Topic", PageType.CONCEPT, 0.8)
        store.add_source_to_topic("healthy", "src-1")

        report = run_health_check(cfg, store)
        assert report["topic_count"] == 1
        assert report["orphans"]["topics"] == []
        assert report["orphans"]["sources"] == []
        assert report["avg_confidence"] == 0.8

    def test_orphan_source(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)

        store.state.sources["src-orphan"] = SourceState(
            sha256="abc", raw="test", topics=[]
        )

        report = run_health_check(cfg, store)
        assert "src-orphan" in report["orphans"]["sources"]

    def test_orphan_topic(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)

        store.ensure_topic("orphan", "Orphan Topic")
        # No sources added -> orphan

        report = run_health_check(cfg, store)
        assert "orphan" in report["orphans"]["topics"]

    def test_broken_link(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)

        store.ensure_topic("a", "Topic A")
        # Manually add a broken link
        store.state.topics["a"].links_to.append("nonexistent")

        report = run_health_check(cfg, store)
        assert ("a", "nonexistent") in report["broken_links"]

    def test_stale_topic(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)

        old = (datetime.now(UTC) - timedelta(days=100)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        store.state.topics["stale"] = TopicState(
            title="Stale Topic", created=old, updated=old
        )

        report = run_health_check(cfg, store)
        assert "stale" in report["stale_topics"]

    def test_failed_source(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)

        store.state.sources["failed-src"] = SourceState(
            sha256="def", raw="fail", stage=IngestStage.FAILED, topics=[]
        )

        report = run_health_check(cfg, store)
        assert "failed-src" in report["empty_extractions"]

    def test_avg_confidence_empty(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)
        report = run_health_check(cfg, store)
        assert report["avg_confidence"] == 0.0

    def test_avg_confidence_mixed(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)

        store.state.topics["a"] = TopicState(
            title="A", confidence=1.0, created="2026-01-01", updated="2026-01-01"
        )
        store.state.topics["b"] = TopicState(
            title="B", confidence=0.5, created="2026-01-01", updated="2026-01-01"
        )

        report = run_health_check(cfg, store)
        assert report["avg_confidence"] == 0.75


# ---------------------------------------------------------------------------
# TestRenderHealthMarkdown
# ---------------------------------------------------------------------------


class TestRenderHealthMarkdown:
    def test_renders_section(self) -> None:
        report = {
            "source_count": 10,
            "topic_count": 5,
            "orphans": {"sources": ["s1"], "topics": ["t1"]},
            "broken_links": [("a", "b")],
            "near_duplicates": [],
            "empty_extractions": ["e1"],
            "stale_topics": ["old"],
            "avg_confidence": 0.75,
            "schema_violations": [],
        }
        md = render_health_markdown(report)
        assert md.startswith("## Brain Health")
        assert "10" in md
        assert "5" in md
        assert "**Broken links**: 1" in md
        assert "0.750" in md

    def test_empty_report(self) -> None:
        report = {
            "source_count": 0,
            "topic_count": 0,
            "orphans": {"sources": [], "topics": []},
            "broken_links": [],
            "near_duplicates": [],
            "empty_extractions": [],
            "stale_topics": [],
            "avg_confidence": 0.0,
            "schema_violations": [],
        }
        md = render_health_markdown(report)
        assert "0" in md
        assert "0.000" in md


# ---------------------------------------------------------------------------
# TestCosine
# ---------------------------------------------------------------------------


class TestCosine:
    def test_identical(self) -> None:
        assert cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 1.0

    def test_orthogonal(self) -> None:
        assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_vector(self) -> None:
        assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
        assert cosine([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_known_angle(self) -> None:
        # 45-degree angle -> cos = ~0.707
        sim = cosine([1.0, 0.0], [1.0, 1.0])
        assert abs(sim - (1 / (2**0.5))) < 1e-9


# ---------------------------------------------------------------------------
# TestCompactionMerge
# ---------------------------------------------------------------------------


class TestCompactionMerge:
    """Integration test for ``run_compaction`` with a real VectorStore."""

    @pytest.mark.asyncio
    async def test_merge_similar_topics(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)

        store = BrainStateStore.load(cfg)

        # Seed two topics with near-identical member vectors.
        db_path = tmp_path / ".brain" / "embeddings.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        vec_store = VectorStore(db_path, model="test", dim=DIM)

        # vec_a and vec_b are very similar (cosine ~= 0.99)
        vec_a = _vec(1.0, 0.0)
        vec_b = _vec(0.99, 0.141)

        # Topic A: 2 sources
        store.ensure_topic("topic-a", "Topic A", PageType.CONCEPT, 0.8)
        store.add_source_to_topic("topic-a", "src-a1")
        store.add_source_to_topic("topic-a", "src-a2")
        vec_store.upsert_source_chunks("src-a1", "topic-a", [("text a1", vec_a)])
        vec_store.upsert_source_chunks("src-a2", "topic-a", [("text a2", vec_a)])
        vec_store.add_topic_member("topic-a", "src-a1")
        vec_store.add_topic_member("topic-a", "src-a2")

        # Topic B: 1 source
        store.ensure_topic("topic-b", "Topic B", PageType.CONCEPT, 0.6)
        store.add_source_to_topic("topic-b", "src-b1")
        vec_store.upsert_source_chunks("src-b1", "topic-b", [("text b1", vec_b)])
        vec_store.add_topic_member("topic-b", "src-b1")

        # Write wiki pages.
        _write_wiki_page(cfg, "topic-a", "Topic A",
                         "## Synthesis\nOriginal A synthesis.\n",
                         2, 0.8)
        _write_wiki_page(cfg, "topic-b", "Topic B",
                         "## Synthesis\nOriginal B synthesis.\n",
                         1, 0.6)

        # Add a link from topic-a to nonexistent (to test broken link later)
        store.state.topics["topic-a"].links_to.append("ghost-topic")

        store.save()
        vec_store.recompute_centroid("topic-a")
        vec_store.recompute_centroid("topic-b")

        # Run compaction with FakeClient
        client = FakeClient()

        summary = await run_compaction(
            cfg, store, vec_store, client,
            merge_threshold=MERGE_THRESHOLD,
        )

        # -- assertions --------------------------------------------------

        # 1. A merge happened
        assert summary["merges"] == 1
        assert summary["pairs"][0][0] == "topic-a"  # a has more sources
        assert summary["pairs"][0][1] == "topic-b"
        assert summary["merged_into"]["topic-b"] == "topic-a"

        # 2. topic-a now has 3 sources (a1 + a2 + b1)
        assert len(store.state.topics["topic-a"].sources) == 3
        assert "src-b1" in store.state.topics["topic-a"].sources

        # 3. topic-b's wiki page has a redirect note
        b_path = cfg.brain_root / "90-wiki" / "topic-b.md"
        assert b_path.exists()  # NEVER deleted (§8)
        b_text = b_path.read_text(encoding="utf-8")
        assert "Merged into" in b_text
        assert "topic-a" in b_text

        # 4. topic-a's synthesis was rewritten via FakeClient
        a_path = cfg.brain_root / "90-wiki" / "topic-a.md"
        a_text = a_path.read_text(encoding="utf-8")
        assert "Merged synthesis content" in a_text

        # 5. Changelog has merge + rewrite entries
        changelog_path = cfg.brain_root / ".brain" / "changelog.jsonl"
        lines = changelog_path.read_text(encoding="utf-8").strip().split("\n")
        merge_entries = [
            json.loads(ln)
            for ln in lines
            if '"kind": "compact"' in ln and '"action": "merge"' in ln
        ]
        rewrite_entries = [
            json.loads(ln)
            for ln in lines
            if '"kind": "compact"' in ln and '"action": "rewrite_synthesis"' in ln
        ]
        assert len(merge_entries) == 1
        assert merge_entries[0]["from"] == "topic-b"
        assert merge_entries[0]["into"] == "topic-a"
        assert len(rewrite_entries) >= 1

        # 6. Confidence was recomputed (weighted)
        assert store.state.topics["topic-a"].confidence > 0.0

        # 7. topic-b's state still exists (never deleted)
        assert "topic-b" in store.state.topics

        # 8. vec_store centroid for topic-a still exists
        centroid = vec_store.recompute_centroid("topic-a")
        assert centroid is not None

        vec_store.close()

    @pytest.mark.asyncio
    async def test_no_merge_below_threshold(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)

        store = BrainStateStore.load(cfg)
        db_path = tmp_path / ".brain" / "embeddings.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        vec_store = VectorStore(db_path, model="test", dim=DIM)

        # Dissimilar vectors (orthogonal)
        store.ensure_topic("topic-x", "Topic X")
        store.ensure_topic("topic-y", "Topic Y")
        store.add_source_to_topic("topic-x", "src-x")
        store.add_source_to_topic("topic-y", "src-y")

        vec_store.upsert_source_chunks("src-x", "topic-x",
                                       [("x", _vec(1.0, 0.0))])
        vec_store.upsert_source_chunks("src-y", "topic-y",
                                       [("y", _vec(0.0, 1.0))])
        vec_store.add_topic_member("topic-x", "src-x")
        vec_store.add_topic_member("topic-y", "src-y")
        vec_store.recompute_centroid("topic-x")
        vec_store.recompute_centroid("topic-y")

        client = FakeClient()
        summary = await run_compaction(
            cfg, store, vec_store, client,
            merge_threshold=MERGE_THRESHOLD,
        )
        assert summary["merges"] == 0
        vec_store.close()


# ---------------------------------------------------------------------------
# TestFindNearDuplicates
# ---------------------------------------------------------------------------


class TestFindNearDuplicates:
    """Near-dup detection with controlled FakeEmbedder."""

    @pytest.mark.asyncio
    async def test_detects_near_duplicates(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)

        # Register 3 sources.
        store.state.sources["src-a"] = SourceState(
            sha256="aaa", raw="src-a.md", topics=["t1"]
        )
        store.state.sources["src-b"] = SourceState(
            sha256="bbb", raw="src-b.md", topics=["t1"]
        )
        store.state.sources["src-c"] = SourceState(
            sha256="ccc", raw="src-c.md", topics=["t2"]
        )

        # Write source files.
        (cfg.brain_root / "50-sources" / "src-a.md").write_text(
            "Source A content", encoding="utf-8"
        )
        (cfg.brain_root / "50-sources" / "src-b.md").write_text(
            "Source B content similar to A", encoding="utf-8"
        )
        (cfg.brain_root / "50-sources" / "src-c.md").write_text(
            "Completely different content", encoding="utf-8"
        )

        # FakeEmbedder: src-a & src-b return near-identical vectors (collinear);
        # src-c is orthogonal so it should NOT be flagged.
        embedder = FakeEmbedder(DIM)
        embedder.set_vector(
            "Source A content",
            [0.95] + [0.0] * (DIM - 1),
        )
        embedder.set_vector(
            "Source B content similar to A",
            [0.96] + [0.0] * (DIM - 1),
        )
        embedder.set_vector(
            "Completely different content",
            [0.0] + [1.0] + [0.0] * (DIM - 2),  # orthogonal to the others
        )

        db_path = tmp_path / "dedup.db"
        vec_store = VectorStore(db_path, model="test", dim=DIM)

        pairs = await find_near_duplicates(
            cfg, store, vec_store, embedder, threshold=DEDUP_THRESHOLD,
        )

        # Should detect src-a and src-b as near duplicates (sim >= 0.95).
        pair_slugs = {(p[0], p[1]) for p in pairs}
        pair_slugs_rev = {(p[1], p[0]) for p in pairs}
        assert ("src-a", "src-b") in pair_slugs or ("src-a", "src-b") in pair_slugs_rev

        # src-c should not appear in any pair.
        c_pairs = [p for p in pairs if "src-c" in (p[0], p[1])]
        assert len(c_pairs) == 0

        assert all(sim >= DEDUP_THRESHOLD for _, _, sim in pairs)

        vec_store.close()

    @pytest.mark.asyncio
    async def test_no_pairs_when_only_one_source(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)

        store.state.sources["src-only"] = SourceState(
            sha256="aaa", raw="sole.md", topics=["t1"]
        )
        (cfg.brain_root / "50-sources" / "src-only.md").write_text(
            "Sole content", encoding="utf-8"
        )

        embedder = FakeEmbedder(DIM)
        db_path = tmp_path / "dedup-empty.db"
        vec_store = VectorStore(db_path, model="test", dim=DIM)

        pairs = await find_near_duplicates(
            cfg, store, vec_store, embedder, threshold=DEDUP_THRESHOLD,
        )
        assert pairs == []

        vec_store.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_wiki_page(
    cfg: _FakeCfg,
    slug: str,
    title: str,
    body: str,
    source_count: int,
    confidence: float,
) -> None:
    """Write a minimal wiki page for testing."""
    from second_brain.atomicio import write_atomic
    from second_brain.frontmatter import dump_frontmatter

    meta = {
        "title": title,
        "slug": slug,
        "type": "concept",
        "tags": [],
        "aliases": [],
        "created": "2026-06-19",
        "updated": "2026-06-19",
        "source_count": source_count,
        "confidence": confidence,
        "related": [],
    }
    page_text = f"# {title}\n\n{body}\n\n## Sources\n\n## Open questions\n- \n\n## Related\n"
    write_atomic(
        cfg.brain_root / "90-wiki" / f"{slug}.md",
        dump_frontmatter(meta, page_text),
    )
