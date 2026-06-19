"""Phase 6 tests — chat agent + SSE streaming + web route.

Tests the retrieve-then-answer loop with a fake streaming client,
a real (tmp) VectorStore seeded with chunks, and a FakeEmbedder.
No live server or API calls.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from second_brain.chat import chat_once, chat_stream
from second_brain.state import BrainStateStore
from second_brain.vectors.store import VectorStore

DIM = 8


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeStreamClient:
    """Fake OpenRouterClient that returns canned streaming deltas."""

    def __init__(self) -> None:
        self.chat_calls: list[tuple[str, list[dict]]] = []

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        self.chat_calls.append((model, messages))
        for chunk in ["Hello", ", ", "world."]:
            yield chunk

    async def close(self) -> None:
        pass


class BrokenEmbedder:
    """An embedder that always raises."""

    async def embed_query(self, query: str) -> list[float]:
        msg = "embedding failed"
        raise RuntimeError(msg)


class FakeEmbedder:
    """Fixed-dimension embedder that returns deterministic vectors."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    async def ensure_dim(self) -> int:
        return self.dim

    async def embed_query(self, query: str) -> list[float]:
        # Deterministic vector based on query length
        return [float(len(query) + i) for i in range(self.dim)]


def _vec(seed: int) -> list[float]:
    """Deterministic DIM-dim vector."""
    return [float(seed + i) for i in range(DIM)]


def _seed_store_and_vec(tmp_path: Path) -> tuple[BrainStateStore, VectorStore]:
    """Seed a BrainStateStore + VectorStore with one source containing 2 chunks."""
    cfg_fake = SimpleNamespace(brain_root=tmp_path)
    store = BrainStateStore.load(cfg_fake)
    store.ensure_topic("test-topic", "Test Topic")
    store.record_source(
        "src1",
        SimpleNamespace(  # type: ignore[arg-type]
            sha256="abc", topics=["test-topic"], raw="", embedding_model="test"
        ),
    )
    store.save()

    vec_store = VectorStore(
        tmp_path / "embeddings.db",
        model="test-model",
        dim=DIM,
    )
    vec_store.upsert_source_chunks(
        source_id="src1",
        topic_slug="test-topic",
        chunks=[
            ("Chunk one about RAG and vector search.", _vec(0)),
            ("Chunk two about embedding models.", _vec(1)),
        ],
    )
    vec_store.add_topic_member("test-topic", "src1")
    return store, vec_store


class _FakeCfg:
    """Minimal config stub for the web app (chat route only tests empty-query path)."""

    def __init__(self, brain_root: Path) -> None:
        self.brain_root = brain_root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEventSequence:
    """Verify the correct event sequence from chat_stream."""

    async def test_event_sequence(self, tmp_path: Path) -> None:
        store, vec_store = _seed_store_and_vec(tmp_path)
        client = FakeStreamClient()
        embedder = FakeEmbedder()
        cfg = SimpleNamespace(
            models=SimpleNamespace(chat="test-chat-model"),
        )

        events: list[dict[str, Any]] = []
        async for ev in chat_stream(
            "what is RAG?", cfg, store, vec_store, embedder, client, k=5
        ):
            events.append(ev)

        # Assert sequence
        types = [e["type"] for e in events]
        assert types[0] == "thinking"
        assert types[1] == "tool_call"
        assert types[2] == "tool_result"
        # answer_delta appears one or more times
        assert all(t == "answer_delta" for t in types[3:-1])
        assert types[-1] == "done"

        # tool_result should contain hits referencing src1
        tool_result = events[2]
        assert tool_result["tool"] == "search_brain"
        assert len(tool_result["hits"]) > 0
        hit_ids = {h["source_id"] for h in tool_result["hits"]}
        assert "src1" in hit_ids

        # Concatenated answer
        answer = "".join(
            e["content"] for e in events if e["type"] == "answer_delta"
        )
        assert answer == "Hello, world."

        # Verify the chat model received a grounded prompt
        assert len(client.chat_calls) == 1
        call_model, call_msgs = client.chat_calls[0]
        assert call_model == "test-chat-model"
        assert len(call_msgs) == 2  # system + user
        assert "Question: what is RAG?" in call_msgs[1]["content"]
        assert "Retrieved sources" in call_msgs[1]["content"]

        vec_store.close()

    async def test_no_context_fallback(self, tmp_path: Path) -> None:
        """When search raises, tool_result has error but stream continues."""
        store, vec_store = _seed_store_and_vec(tmp_path)
        client = FakeStreamClient()
        broken = BrokenEmbedder()
        cfg = SimpleNamespace(
            models=SimpleNamespace(chat="test-chat-model"),
        )

        events: list[dict[str, Any]] = []
        async for ev in chat_stream(
            "test", cfg, store, vec_store, broken, client, k=5
        ):
            events.append(ev)

        types = [e["type"] for e in events]
        assert types[0] == "thinking"
        assert types[1] == "tool_call"
        # tool_result has error + empty hits
        assert types[2] == "tool_result"
        tool_result = events[2]
        assert tool_result["tool"] == "search_brain"
        assert len(tool_result["hits"]) == 0
        assert "error" in tool_result
        assert "embedding failed" in tool_result["error"]
        # answer deltas still arrive
        assert all(t == "answer_delta" for t in types[3:-1])
        assert types[-1] == "done"
        answer = "".join(
            e["content"] for e in events if e["type"] == "answer_delta"
        )
        assert answer == "Hello, world."

        vec_store.close()

    async def test_chat_once(self, tmp_path: Path) -> None:
        """chat_once concatenates answer deltas."""
        store, vec_store = _seed_store_and_vec(tmp_path)
        client = FakeStreamClient()
        embedder = FakeEmbedder()
        cfg = SimpleNamespace(
            models=SimpleNamespace(chat="test-chat-model"),
        )

        answer = await chat_once(
            "greet me", cfg, store, vec_store, embedder, client
        )
        assert answer == "Hello, world."

        vec_store.close()


class TestChatRoutes:
    """Web UI chat routes (GET /chat, GET /api/chat)."""

    @pytest.fixture
    def app_with_chat(self, tmp_path: Path):
        """Build a FastAPI app with seeded brain."""
        from second_brain.web.app import create_app

        cfg_fake = _FakeCfg(tmp_path)
        _seed_store_and_vec(tmp_path)
        application = create_app(cfg_fake)
        return application

    @pytest.fixture
    def client(self, app_with_chat):
        """TestClient with cleared singleton."""
        import second_brain.web.app as web_app

        web_app._store = None  # noqa: SLF001
        return TestClient(app_with_chat)

    def test_chat_page_returns_200(self, client: TestClient) -> None:
        """GET /chat returns the chat UI page."""
        resp = client.get("/chat")
        assert resp.status_code == 200
        assert 'id="chat-log"' in resp.text

    def test_chat_api_empty_query(self, client: TestClient) -> None:
        """Empty query returns a single error done event (no API calls needed)."""
        resp = client.get("/api/chat?q=")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        body = resp.text
        assert '"type":"done"' in body or '"type": "done"' in body
        assert '"error"' in body

    def test_chat_api_returns_sse_headers(self, client: TestClient) -> None:
        """GET /api/chat with a query returns text/event-stream (content may be truncated
        without a real API key, but the route starts and sets correct headers)."""
        resp = client.get("/api/chat?q=test", timeout=5)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
