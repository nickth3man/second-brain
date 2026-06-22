"""End-to-end tests for Phase 1B (daemon pipeline).

All tests use a fake OpenRouter client — no network, no API key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from second_brain.config import (
    TypesCfg,
)
from second_brain.daemon.extract import (
    ExtractionError,
    build_messages,
    extract,
    schema_for_strict,
)
from second_brain.openrouter_client import OpenRouterAPIError
from second_brain.daemon.index import DebouncedIndex
from second_brain.daemon.linker import LinkContext, SlugLinker
from second_brain.daemon.normalize import (
    estimate_tokens,
    sha256_of_file,
    source_id_for,
)
from second_brain.daemon.pipeline import ingest_file
from second_brain.daemon.router import is_temp_file, route
from second_brain.models import (
    IngestStage,
    LibrarianOutput,
    LinkDecision,
    TopicAction,
)
from second_brain.state import BrainStateStore

# -- helpers -------------------------------------------------------------------


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
    merge_threshold: float = 0.7
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


class FakeClient:
    """Fake OpenRouter client with configurable payloads and failure modes.

    Failures raise :class:`OpenRouterAPIError` to match the real client
    contract — the production ``extract()`` now catches that type per §12.2
    (was previously ``httpx.HTTPStatusError``).
    """

    def __init__(self, payload: dict | None = None, fail_times: int = 0):
        self.payload = payload or {
            "tldr": "Test summary.",
            "topics": [
                {
                    "name": "Test Topic",
                    "action": "new",
                    "target_slug": "",
                    "confidence": 0.9,
                    "merged_section": "Test merged content.",
                }
            ],
        }
        self.fail_times = fail_times
        self.call_count = 0

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        *,
        response_format: dict | None = None,
        extra_body: dict | None = None,
        stream: bool = False,
    ) -> dict:
        self.call_count += 1
        if self.call_count <= self.fail_times:
            # Simulate a 5xx server error
            raise OpenRouterAPIError(
                status=500,
                endpoint="/chat/completions",
                body='{"error":{"name":"ServerError"}}',
                error_name="ServerError",
            )
        return {
            "choices": [{"message": {"content": json.dumps(self.payload)}}]
        }

    async def close(self) -> None:
        pass


def _ensure_dirs(cfg) -> None:
    """Create the brain directory structure needed by the pipeline."""
    (cfg.brain_root / "00-inbox").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / "50-sources").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / "90-wiki").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / ".brain").mkdir(parents=True, exist_ok=True)


def _write_inbox(cfg, name: str, content: str) -> Path:
    """Write a test file into the inbox and return its path."""
    p = cfg.brain_root / "00-inbox" / name
    p.write_text(content, encoding="utf-8")
    return p


# -- tests ---------------------------------------------------------------------


class TestRouter:
    def test_is_temp_file(self) -> None:
        assert is_temp_file("~$doc.docx")
        assert is_temp_file(".goutputstream-1234")
        assert is_temp_file("download.crdownload")
        assert is_temp_file("data.part")
        assert is_temp_file(".swapfile.swp")
        assert is_temp_file("temp.tmp")
        assert is_temp_file("~tempfile~")
        assert is_temp_file("backup.bak")
        assert not is_temp_file("notes.md")
        assert not is_temp_file("readme.txt")
        assert not is_temp_file("script.py")

    def test_route_known(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        assert route(".md", cfg) == "text"
        assert route(".txt", cfg) == "text"
        assert route(".py", cfg) == "code"
        assert route(".json", cfg) == "structured"

    def test_route_unknown(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        with pytest.raises(ValueError, match="Unknown extension"):
            route(".pdf", cfg)


class TestNormalize:
    def test_source_id_for(self, tmp_path: Path) -> None:
        path = tmp_path / "My Test File.md"
        path.write_text("hello world", encoding="utf-8")
        sid = source_id_for(path, "hello world", "2026-06-19T10:00:00Z")
        assert sid.startswith("2026-06-19-")
        assert "my-test-file" in sid

    def test_source_id_fallback(self, tmp_path: Path) -> None:
        # Stem with only special chars -> slugified empty -> fallback to body words
        path = tmp_path / "___!!!.md"
        path.write_text("hello world foo bar baz qux", encoding="utf-8")
        sid = source_id_for(path, "hello world foo bar baz qux", "2026-06-19T10:00:00Z")
        # Should fall back to first 6 words
        assert "hello-world-foo-bar-baz-qux" in sid

    def test_sha256_of_file(self, tmp_path: Path) -> None:
        p = tmp_path / "test.txt"
        p.write_bytes(b"hello")
        h = sha256_of_file(p)
        assert len(h) == 64  # SHA-256 hex digest length

    def test_estimate_tokens(self) -> None:
        text = "a" * 100
        assert estimate_tokens(text) == 25  # 100 // 4


class TestExtract:
    def test_build_messages_truncation(self) -> None:
        long_body = "x" * 100_000  # ~25k tokens -> triggers truncation
        msgs = build_messages(long_body, {})
        user_msg = msgs[1]["content"]
        assert "[truncated for extraction" in user_msg
        assert len(user_msg) < 70_000

    def test_schema_for_strict(self) -> None:
        schema = schema_for_strict()
        assert schema["type"] == "json_schema"
        assert schema["json_schema"]["strict"] is True
        assert schema["json_schema"]["name"] == "librarian_output"

    @pytest.mark.asyncio
    async def test_extract_success(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        client = FakeClient()
        result = await extract(client, cfg, "test body", {})
        assert isinstance(result, LibrarianOutput)
        assert result.tldr == "Test summary."
        assert len(result.topics) == 1

    @pytest.mark.asyncio
    async def test_extract_failure_then_fallback(self, tmp_path: Path) -> None:
        """First call fails (5xx), second (repair) succeeds."""
        cfg = _make_cfg(tmp_path)
        client = FakeClient(fail_times=1)
        result = await extract(client, cfg, "test body", {})
        assert isinstance(result, LibrarianOutput)
        assert client.call_count == 2  # primary + repair

    @pytest.mark.asyncio
    async def test_extract_total_failure(self, tmp_path: Path) -> None:
        """Both primary and repair fail -> ExtractionError."""
        cfg = _make_cfg(tmp_path)
        client = FakeClient(fail_times=2)
        with pytest.raises(ExtractionError):
            await extract(client, cfg, "test body", {})

    @pytest.mark.asyncio
    async def test_extract_garbage_json(self, tmp_path: Path) -> None:
        """Garbage response content -> ExtractionError."""
        cfg = _make_cfg(tmp_path)

        class GarbageClient:
            call_count = 0

            async def chat_completion(self, *a, **kw):
                self.call_count += 1
                return {
                    "choices": [
                        {"message": {"content": "this is not valid librarian json"}}
                    ]
                }

            async def close(self):
                pass

        client = GarbageClient()
        with pytest.raises(ExtractionError):
            await extract(client, cfg, "test body", {})
        assert client.call_count == 2  # primary + repair


class TestSlugLinker:
    @pytest.mark.asyncio
    async def test_new_topic(self) -> None:
        from second_brain.state import BrainState

        store = type("S", (), {"state": BrainState()})()  # empty state
        linker = SlugLinker()
        decisions = [
            LinkDecision(
                name="Brand New Concept",
                action=TopicAction.NEW,
                target_slug="",
                confidence=0.8,
                merged_section="Some content.",
            )
        ]
        result = await linker.link(decisions, LinkContext(brain_store=store))
        assert result[0].action == TopicAction.NEW
        assert result[0].target_slug == "brand-new-concept"

    @pytest.mark.asyncio
    async def test_match_existing(self) -> None:
        from second_brain.models import BrainState, TopicState

        state = BrainState()
        state.topics["existing-topic"] = TopicState(
            title="Existing Topic", created="2026-01-01", updated="2026-01-01"
        )
        store = type("S", (), {"state": state})()
        linker = SlugLinker()
        decisions = [
            LinkDecision(
                name="Existing Topic",
                action=TopicAction.NEW,
                target_slug="",
                confidence=0.8,
                merged_section="Content.",
            )
        ]
        result = await linker.link(decisions, LinkContext(brain_store=store))
        assert result[0].action == TopicAction.MATCH
        assert result[0].target_slug == "existing-topic"


class TestPipeline:
    @pytest.mark.asyncio
    async def test_e2e_text_ingest(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        client = FakeClient(
            payload={
                "tldr": "A test source about something.",
                "topics": [
                    {
                        "name": "Test Concept",
                        "action": "new",
                        "target_slug": "",
                        "confidence": 0.9,
                        "merged_section": "Test Concept is about testing.",
                    },
                    {
                        "name": "Testing Methods",
                        "action": "new",
                        "target_slug": "",
                        "confidence": 0.7,
                        "merged_section": "Testing methods include various approaches.",
                    },
                ],
            }
        )
        store = BrainStateStore.load(cfg)
        linker = SlugLinker()
        index = DebouncedIndex(cfg, store)

        path = _write_inbox(cfg, "my-test-note.md", "# My Note\n\nThis is a test.")
        stage = await ingest_file(path, cfg, store, client, linker, index)
        assert stage == IngestStage.DONE

        # Assert 50-sources/ has one file
        sources = list((cfg.brain_root / "50-sources").iterdir())
        assert len(sources) == 1
        src_text = sources[0].read_text(encoding="utf-8")
        assert "source:" in src_text
        assert "sha256:" in src_text
        assert "## Summary" in src_text

        # Assert 90-wiki/ has two topic pages
        wiki_files = list((cfg.brain_root / "90-wiki").iterdir())
        assert len(wiki_files) == 2
        wiki_text = wiki_files[0].read_text(encoding="utf-8")
        assert "## Synthesis" in wiki_text
        assert "## Sources" in wiki_text

        # Assert state.json exists and source is DONE
        state_path = cfg.brain_root / ".brain" / "state.json"
        assert state_path.exists()
        state_data = json.loads(state_path.read_text(encoding="utf-8"))
        assert len(state_data["sources"]) == 1
        src_id = list(state_data["sources"].keys())[0]
        assert state_data["sources"][src_id]["stage"] == "done"
        assert len(state_data["topics"]) == 2

        # Assert changelog has done line
        changelog_path = cfg.brain_root / ".brain" / "changelog.jsonl"
        assert changelog_path.exists()
        lines = changelog_path.read_text(encoding="utf-8").strip().split("\n")
        done_entries = [ln for ln in lines if '"done"' in ln]
        assert len(done_entries) == 1

    @pytest.mark.asyncio
    async def test_dedup(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        client = FakeClient()
        store = BrainStateStore.load(cfg)
        linker = SlugLinker()
        index = DebouncedIndex(cfg, store)

        content = "# Dedup Test\n\nSame content twice."
        path = _write_inbox(cfg, "dedup-test.md", content)

        # First ingest
        stage1 = await ingest_file(path, cfg, store, client, linker, index)
        assert stage1 == IngestStage.DONE

        # Second ingest of same content
        stage2 = await ingest_file(path, cfg, store, client, linker, index)
        assert stage2 == IngestStage.DONE

        # Only one source file created
        sources = list((cfg.brain_root / "50-sources").iterdir())
        assert len(sources) == 1

    @pytest.mark.asyncio
    async def test_merge_vs_spawn(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)

        # Pre-seed a topic whose slug matches a proposed name
        from second_brain.models import PageType

        store.ensure_topic("test-concept", "Test Concept", PageType.CONCEPT, 0.5)
        store.save()

        # Create the existing wiki page for the pre-seeded topic
        from second_brain.daemon.wiki import write_new_topic
        from second_brain.models import LinkDecision, TopicAction

        existing_decision = LinkDecision(
            name="Test Concept",
            action=TopicAction.MATCH,
            target_slug="test-concept",
            confidence=0.5,
            merged_section="Original synthesis.",
        )
        write_new_topic(
            cfg, store, "test-concept", "Test Concept",
            existing_decision, "prev-source", "2026-06-19T10:00:00Z",
        )

        client = FakeClient(
            payload={
                "tldr": "More about testing.",
                "topics": [
                    {
                        "name": "Test Concept",
                        "action": "new",
                        "target_slug": "",
                        "confidence": 0.85,
                        "merged_section": "Additional synthesis content.",
                    },
                    {
                        "name": "New Unseen Topic",
                        "action": "new",
                        "target_slug": "",
                        "confidence": 0.6,
                        "merged_section": "Brand new synthesis.",
                    },
                ],
            }
        )
        linker = SlugLinker()
        index = DebouncedIndex(cfg, store)

        path = _write_inbox(cfg, "second-note.md", "# Second Note\nMore content.")
        stage = await ingest_file(path, cfg, store, client, linker, index)
        assert stage == IngestStage.DONE

        # Test Concept page should now have a "### From" section (merge)
        tc_path = cfg.brain_root / "90-wiki" / "test-concept.md"
        tc_text = tc_path.read_text(encoding="utf-8")
        assert "### From" in tc_text
        assert "Original synthesis" in tc_text
        assert "Additional synthesis" in tc_text

        # New Unseen Topic should be a new page (spawn)
        nu_path = cfg.brain_root / "90-wiki" / "new-unseen-topic.md"
        assert nu_path.exists()
        nu_text = nu_path.read_text(encoding="utf-8")
        assert "## Synthesis" in nu_text

        # source_count should have incremented for test-concept
        from second_brain.frontmatter import split_frontmatter

        meta, _ = split_frontmatter(tc_text)
        assert meta.get("source_count") == 2
        assert meta.get("confidence") == 0.675  # weighted mean of 0.5 (1 src) and 0.85

    @pytest.mark.asyncio
    async def test_extract_failure_pipeline(self, tmp_path: Path) -> None:
        """Pipeline catches extraction failure -> FAILED + deadletter."""
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)

        class FailingClient:
            call_count = 0

            async def chat_completion(self, *a, **kw):
                self.call_count += 1
                return {
                    "choices": [{"message": {"content": "NOT JSON"}}]
                }

            async def close(self):
                pass

        store = BrainStateStore.load(cfg)
        linker = SlugLinker()
        index = DebouncedIndex(cfg, store)

        path = _write_inbox(cfg, "garbage.md", "# Garbage\nWon't parse.")
        stage = await ingest_file(
            path, cfg, store, FailingClient(), linker, index
        )
        assert stage == IngestStage.FAILED

        # Raw file should be in deadletter
        dead_dir = cfg.brain_root / cfg.extraction.deadletter_dir
        assert (dead_dir / "garbage.md").exists()

    @pytest.mark.asyncio
    async def test_index_flush(self, tmp_path: Path) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        store = BrainStateStore.load(cfg)

        # Add a source and topic to the store
        from second_brain.models import SourceState

        store.record_source(
            "2026-06-19-test-src",
            SourceState(
                sha256="abc",
                raw="00-inbox/test.md",
                type="text",
                ingested="2026-06-19T10:00:00Z",
                topics=["test-topic"],
            ),
        )
        store.ensure_topic("test-topic", "Test Topic")
        store.add_source_to_topic("test-topic", "2026-06-19-test-src")
        store.save()

        index = DebouncedIndex(cfg, store)
        await index.flush_now()

        index_path = cfg.brain_root / "INDEX.md"
        assert index_path.exists()
        text = index_path.read_text(encoding="utf-8")
        assert "# Second Brain" in text
        assert "1 sources · 1 topics" in text
        assert "## Recent" in text
        assert "## Topics" in text
        assert "[Test Topic](90-wiki/test-topic.md)" in text

    @pytest.mark.asyncio
    async def test_route_unsupported(self, tmp_path: Path) -> None:
        """Unsupported file types (e.g., .pdf in Phase 1) fail gracefully."""
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)
        client = FakeClient()
        store = BrainStateStore.load(cfg)
        linker = SlugLinker()
        index = DebouncedIndex(cfg, store)

        path = _write_inbox(cfg, "doc.pdf", "fake pdf content")
        stage = await ingest_file(path, cfg, store, client, linker, index)
        assert stage == IngestStage.FAILED



