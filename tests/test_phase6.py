"""Phase 6 tests — §12.4 Pydantic AI chat event bridge."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from second_brain.chat import chat_once, chat_stream
from second_brain.state import BrainStateStore


class FakeAgent:
    """Sentinel agent used by patched event stream tests."""


async def _fake_event_stream(agent: Any, query: str, cfg: Any) -> AsyncIterator[dict[str, Any]]:
    assert isinstance(agent, FakeAgent)
    assert query == "what is RAG?"
    assert cfg.models.chat == "test-chat-model"
    yield {"type": "thinking", "content": "Starting agent loop..."}
    yield {"type": "tool_call", "tool": "search_brain", "args": {"query": query, "k": 5}}
    yield {
        "type": "tool_result",
        "tool": "search_brain",
        "hits": [{"source_id": "src1", "topic_slug": "rag", "score": 1.0}],
    }
    yield {"type": "reasoning_delta", "content": "checking"}
    yield {"type": "answer_delta", "content": "Hello"}
    yield {"type": "answer_delta", "content": ", world."}
    yield {"type": "done"}


class _FakeCfg:
    def __init__(self, brain_root: Path) -> None:
        self.brain_root = brain_root
        self.daemon = SimpleNamespace(http_host="127.0.0.1", http_port=8001)
        self.models = SimpleNamespace(chat="test-chat-model")
        self.openrouter = SimpleNamespace(base_url="https://openrouter.ai/api/v1", api_key="sk")
        self.privacy = SimpleNamespace(
            zdr=True,
            block_training_providers=False,
            api_key_source="config",
        )
        self.extraction = SimpleNamespace(require_parameters=True)


class TestPydanticAgentBridge:
    async def test_chat_stream_uses_agent_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import second_brain.chat.agent as chat_agent

        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        fake_agent = FakeAgent()
        monkeypatch.setattr(chat_agent, "_stream_pydantic_events", _fake_event_stream)

        events = [
            ev
            async for ev in chat_stream(
                "what is RAG?",
                cfg,
                store,
                vec_store=None,
                embedder=None,
                client=None,
                agent=fake_agent,
            )
        ]

        assert [e["type"] for e in events] == [
            "thinking",
            "tool_call",
            "tool_result",
            "reasoning_delta",
            "answer_delta",
            "answer_delta",
            "done",
        ]
        assert events[1]["tool"] == "search_brain"
        assert events[2]["hits"][0]["source_id"] == "src1"

    async def test_chat_once_concatenates_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import second_brain.chat.agent as chat_agent

        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        monkeypatch.setattr(chat_agent, "_stream_pydantic_events", _fake_event_stream)

        answer = await chat_once("what is RAG?", cfg, store, agent=FakeAgent())

        assert answer == "Hello, world."


class TestChatRoutes:
    @pytest.fixture
    def app_with_chat(self, tmp_path: Path):
        from second_brain.web.app import create_app

        cfg_fake = _FakeCfg(tmp_path)
        return create_app(cfg_fake)

    @pytest.fixture
    def client(self, app_with_chat):
        import second_brain.web.app as web_app

        web_app._store = None  # noqa: SLF001
        return TestClient(app_with_chat)

    def test_chat_page_returns_200(self, client: TestClient) -> None:
        resp = client.get("/chat")
        assert resp.status_code == 200
        assert 'id="chat-log"' in resp.text

    def test_chat_api_empty_query(self, client: TestClient) -> None:
        resp = client.get("/api/chat?q=")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert '"type":"done"' in body or '"type": "done"' in body
        assert '"error"' in body

    def test_chat_api_returns_sse_headers(self, client: TestClient) -> None:
        resp = client.get("/api/chat?q=test", timeout=5)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
