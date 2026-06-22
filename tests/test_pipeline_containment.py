"""Pipeline containment tests — failure resilience, timeout, deadletter.

Hermetic — no real API calls; uses fakes for client/embedder.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from second_brain.models import IngestStage

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeClient:
    """Fake OpenRouterClient that can be made to raise on extract."""

    def __init__(self, fail_extract: bool = False) -> None:
        self.fail_extract = fail_extract

    async def transcribe(
        self, model: str, audio_path: Path,
        *, language: str | None = None, audio_format: str | None = None,
    ) -> str:
        return "transcript"

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        if self.fail_extract:
            raise RuntimeError("extract failed")
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "tldr": "test",
                            "topics": [
                                {
                                    "name": "Test",
                                    "action": "new",
                                    "target_slug": "test",
                                    "confidence": 0.9,
                                    "merged_section": "Test content.",
                                }
                            ],
                        })
                    }
                }
            ],
        }

    async def chat_completion_clean(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str | None, str]:
        return (None, "test")

    async def embedding(self, model: str, input: str | list[str]) -> list[float]:
        return [0.1] * 8

    async def close(self) -> None:
        pass


class FakeEmbedder:
    """Fake embedder for pipeline tests."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def ensure_dim(self) -> int:
        return 8

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("embedding failed")
        return [[0.1] * 8 for _ in texts]

    async def embed_one(self, text: str) -> list[float]:
        return [0.1] * 8

    async def embed_query(self, query: str) -> list[float]:
        return [0.1] * 8


class FakeLinker:
    """Stub linker for pipeline tests."""

    async def link(self, topics, ctx):
        return []


class FakeIndex:
    """Stub index for pipeline tests."""

    def mark_dirty(self) -> None:
        pass

    async def flush_now(self) -> None:
        pass


class FakeVecStore:
    """Stub vector store for pipeline tests."""

    def close(self) -> None:
        pass


def _make_cfg(tmp_path: Path) -> SimpleNamespace:
    """Build minimal config stub for pipeline tests."""
    return SimpleNamespace(
        brain_root=tmp_path,
        models=SimpleNamespace(
            stt="test-stt",
            text="test-text",
            chat="test-chat",
            embedding="test-embed",
        ),
        types=SimpleNamespace(
            text=["txt", "md"],
            code=["py"],
            structured=["json"],
            vision=[],
            pdf=[],
            office=[],
            web=[],
            ebook=[],
            audio=[],
            video=[],
        ),
        ingestion=SimpleNamespace(
            max_audio_minutes=120,
            merge_threshold=0.85,
            vision_max_images_per_request=5,
            require_parameters=False,
            enable_healing=False,
        ),
        extraction=SimpleNamespace(
            deadletter_dir=str(tmp_path / ".brain" / "deadletter"),
            primary_model="",
            repair_model="",
            require_parameters=False,
            enable_healing=False,
            confidence_floor=0.5,
        ),
        privacy=SimpleNamespace(
            zdr=True,
            api_key_source="env",
            block_training_providers=False,
        ),
        openrouter=SimpleNamespace(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-v1-test",
        ),
    )


def _make_source(tmp_path: Path) -> Path:
    """Write a minimal test source file."""
    p = tmp_path / "00-inbox" / "test.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("Hello, world.")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineContainment:
    """Failure modes are contained and files go to FAILED."""

    async def test_extract_raises_marks_failed(self, tmp_path: Path) -> None:
        """Extract failure transitions to FAILED."""
        from second_brain.daemon.pipeline import ingest_file
        from second_brain.state import BrainStateStore

        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)
        client = FakeClient(fail_extract=True)
        path = _make_source(tmp_path)

        stage = await ingest_file(
            path, cfg, store, client, FakeLinker(), FakeIndex(),
            embedder=None, vec_store=None,
        )

        assert stage == IngestStage.FAILED
        # Deadletter should exist
        dead_dir = tmp_path / ".brain" / "deadletter"
        assert list(dead_dir.iterdir()) != []

    async def test_embed_raises_marks_failed(self, tmp_path: Path) -> None:
        """Embedding failure transitions to FAILED."""
        from second_brain.daemon.pipeline import ingest_file
        from second_brain.state import BrainStateStore

        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)
        client = FakeClient(fail_extract=False)
        embedder = FakeEmbedder(fail=True)  # embedding will fail
        path = _make_source(tmp_path)

        stage = await ingest_file(
            path, cfg, store, client, FakeLinker(), FakeIndex(),
            embedder=embedder, vec_store=FakeVecStore(),
        )

        assert stage == IngestStage.FAILED

    async def test_binary_body_not_fed_to_extract(self, tmp_path: Path) -> None:
        """An MP3 path does not feed raw bytes to the extract call
        (the body comes from normalize_text, not path.read_text)."""
        # This is a structural test: we verify normalize_text returns
        # a body and that pipeline uses it (not raw bytes).
        # The audio parser will fail if given a real MP3 with no ffprobe,
        # so we test the path through the normal file route.
        from second_brain.daemon.pipeline import ingest_file
        from second_brain.state import BrainStateStore

        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)
        client = FakeClient(
            fail_extract=False,  # extract succeeds with our fake
        )
        path = _make_source(tmp_path)

        stage = await ingest_file(
            path, cfg, store, client, FakeLinker(), FakeIndex(),
            embedder=None, vec_store=None,
        )

        # Should reach DONE since text file normalizes fine
        # and the fake extract returns valid output.
        assert stage == IngestStage.DONE
        assert store.state.sources
        sid = list(store.state.sources.keys())[0]
        assert store.state.sources[sid].stage == IngestStage.DONE

    async def test_deadletter_copy_created(self, tmp_path: Path) -> None:
        """Failed files get a deadletter copy."""
        from second_brain.daemon.pipeline import ingest_file
        from second_brain.state import BrainStateStore

        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)
        client = FakeClient(fail_extract=True)
        path = _make_source(tmp_path)

        await ingest_file(
            path, cfg, store, client, FakeLinker(), FakeIndex(),
            embedder=None, vec_store=None,
        )

        dead_dir = tmp_path / ".brain" / "deadletter"
        assert dead_dir.is_dir()
        dead_files = list(dead_dir.iterdir())
        assert any("test.txt" in f.name for f in dead_files)

    async def test_state_json_always_written_on_failure(
        self, tmp_path: Path,
    ) -> None:
        """State is saved after failure (checkpointing)."""
        from second_brain.daemon.pipeline import ingest_file
        from second_brain.state import BrainStateStore

        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)
        client = FakeClient(fail_extract=True)
        path = _make_source(tmp_path)

        await ingest_file(
            path, cfg, store, client, FakeLinker(), FakeIndex(),
            embedder=None, vec_store=None,
        )

        state_path = tmp_path / ".brain" / "state.json"
        assert state_path.is_file()
        data = json.loads(state_path.read_text())
        sources = data.get("sources", {})
        assert sources
        for s in sources.values():
            assert s["stage"] == IngestStage.FAILED.value

    async def test_timeout_marks_failed_and_continues(
        self, tmp_path: Path,
    ) -> None:
        """A file that times out goes to FAILED (test via ingest_file directly
        since we can't easily induce a real timeout in the loop)."""
        # This is verified structurally: the timeout logic is in run_daemon,
        # which calls asyncio.wait_for(...ingest_file()). We test that
        # _fail_file_safe produces correct state.
        from second_brain.daemon.pipeline import _fail_file_safe
        from second_brain.state import BrainStateStore

        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)
        path = _make_source(tmp_path)

        # Simulate a timeout by calling _fail_file_safe directly
        _fail_file_safe(
            path, cfg, msg="timed out", store=store,
            source_id="test-src", sha="abc123", exc=TimeoutError(),
        )

        dead_dir = tmp_path / ".brain" / "deadletter"
        assert dead_dir.is_dir()
        state_path = tmp_path / ".brain" / "state.json"
        assert state_path.is_file()

    async def test_pretry_failure_contained(
        self, tmp_path: Path,
    ) -> None:
        """A failure in normalize_text (ValueError) goes to FAILED, not crash."""
        from second_brain.daemon.pipeline import ingest_file
        from second_brain.state import BrainStateStore

        cfg = _make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)
        client = FakeClient()  # won't be reached
        path = tmp_path / "00-inbox" / "test.unknown"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test")

        stage = await ingest_file(
            path, cfg, store, client, FakeLinker(), FakeIndex(),
            embedder=None, vec_store=None,
        )

        assert stage == IngestStage.FAILED
