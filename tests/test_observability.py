"""Observability tests — logging, key redaction, audit trails.

Uses ``structlog.testing.capture_logs`` to verify structured log events.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import structlog
from structlog.testing import capture_logs

from second_brain.openrouter_client import OpenRouterAPIError

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------


class TestLogConfig:
    """configure_logging idempotency and file output."""

    def test_disk_log_file_written(self, tmp_path: Path) -> None:
        """configure_logging(tmp_path) creates a log file with a JSON line."""
        from second_brain.log import configure_logging

        log_path = configure_logging(tmp_path)
        assert log_path.exists()

        # Write a log line and verify it appears as JSON.
        log = structlog.get_logger(__name__)
        log.info("test_message", key="secret_value")

        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert lines
        parsed = json.loads(lines[-1])
        assert parsed["event"] == "test_message"
        assert parsed["key"] == "***"

    def test_log_config_idempotent(self, tmp_path: Path) -> None:
        """Calling configure_logging twice does not add duplicate handlers."""
        from second_brain.log import configure_logging

        p1 = configure_logging(tmp_path)
        p2 = configure_logging(tmp_path)
        assert p1 == p2

        log = structlog.get_logger(__name__)
        with capture_logs() as cap:
            log.info("second_call")
        assert len(cap) == 1
        assert cap[0]["event"] == "second_call"


# ---------------------------------------------------------------------------
# Key redaction
# ---------------------------------------------------------------------------


class TestKeyRedaction:
    """Sensitive fields are redacted in log output."""

    def test_api_key_never_in_logs(self, tmp_path: Path) -> None:
        """Fields matching key/token/auth/password are redacted in the disk log."""
        from second_brain.log import configure_logging

        log_path = configure_logging(tmp_path)
        log = structlog.get_logger(__name__)
        log.info("test", api_key="sk-or-v1-fake", token="abc", normal="visible")

        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert lines
        last_line = json.loads(lines[-1])
        # The redaction processor runs on the disk log, so the key field
        # should be "***", not the raw value.
        assert last_line["api_key"] == "***", f"Expected redacted, got: {last_line['api_key']}"
        assert last_line["token"] == "***"
        assert last_line["normal"] == "visible"


# ---------------------------------------------------------------------------
# OpenRouter observability
# ---------------------------------------------------------------------------


class TestOpenRouterObservability:
    """Request/response/error events are emitted."""

    async def test_request_response_events_emitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """openrouter.request and openrouter.response events appear."""
        from unittest.mock import AsyncMock, MagicMock

        import httpx

        from second_brain.openrouter_client import OpenRouterClient

        cfg = _fake_cfg(tmp_path)
        cfg.privacy.zdr = True
        client = OpenRouterClient(cfg)
        client._client = AsyncMock(spec=httpx.AsyncClient)

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.text = '{"data":[{"embedding":[0.1]}]}'
        mock_resp.content = mock_resp.text.encode()
        mock_resp.json.return_value = {"data": [{"embedding": [0.1]}]}
        client._client.post.return_value = mock_resp  # type: ignore[assignment]

        with capture_logs() as cap:
            await client.embedding("test-model", "hello")

        events = [c["event"] for c in cap]
        assert "openrouter.request" in events
        assert "openrouter.response" in events

    async def test_error_event_has_body(
        self, tmp_path: Path,
    ) -> None:
        """Error events include the response body preview."""
        from unittest.mock import AsyncMock, MagicMock

        import httpx

        from second_brain.openrouter_client import OpenRouterClient

        cfg = _fake_cfg(tmp_path)
        client = OpenRouterClient(cfg)
        client._client = AsyncMock(spec=httpx.AsyncClient)

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.text = '{"error":{"name":"ZodError","message":"bad request"}}'
        mock_resp.content = mock_resp.text.encode()
        mock_resp.json.return_value = {"error": {"name": "ZodError", "message": "bad request"}}
        client._client.post.return_value = mock_resp  # type: ignore[assignment]

        with capture_logs() as cap, pytest.raises(OpenRouterAPIError):
            await client.embedding("test-model", "hello")

        error_events = [c for c in cap if c["event"] == "openrouter.error"]
        assert error_events
        assert "error_body_preview" in error_events[0]

    def test_400_body_in_exception_and_log(self) -> None:
        """OpenRouterAPIError string includes status, endpoint, and body."""
        err = OpenRouterAPIError(
            status=400,
            endpoint="/audio/transcriptions",
            body='{"error":{"name":"ZodError"}}',
            error_name="ZodError",
        )
        assert "400" in str(err)
        assert "ZodError" in str(err)
        assert "/audio/transcriptions" in str(err)


# ---------------------------------------------------------------------------
# Pipeline observability
# ---------------------------------------------------------------------------


class TestPipelineObservability:
    """Pipeline stage events are emitted."""

    async def test_pipeline_stage_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stage start/end events appear during ingest."""
        from second_brain.daemon.pipeline import ingest_file
        from second_brain.state import BrainStateStore

        # Minimal setup: text file that routes to the text parser.
        cfg = _fake_cfg(tmp_path)
        store = BrainStateStore.load(cfg)

        # Use a fake client that works.
        client = _FakeClient()
        path = tmp_path / "00-inbox" / "test.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Hello, world.")

        with capture_logs() as cap:
            await ingest_file(
                path, cfg, store, client, _FakeLinker(), _FakeIndex(),
                embedder=None, vec_store=None,
            )

        events = [c["event"] for c in cap]
        # Should have stage.start and stage.end events.
        assert "pipeline.stage.start" in events
        assert "pipeline.stage.end" in events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_cfg(tmp_path: Path):
    from types import SimpleNamespace

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


class _FakeClient:
    """Fake client for pipeline log tests."""

    def __init__(self) -> None:
        self.fail_extract = False

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        *,
        response_format=None,
        extra_body=None,
        stream=False,
    ):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"tldr":"test","topics":[{"name":"T",'
                            '"action":"new","target_slug":"t",'
                            '"confidence":0.9,"merged_section":"M."}]}'
                        )
                    }
                }
            ],
        }

    async def chat_completion_clean(
        self, model, messages, *, response_format=None, extra_body=None,
    ):
        return (None, "test")

    async def embedding(self, model, input):
        return [0.1] * 8

    async def close(self):
        pass


class _FakeLinker:
    async def link(self, topics, ctx):
        return []


class _FakeIndex:
    def mark_dirty(self):
        pass

    async def flush_now(self):
        pass
