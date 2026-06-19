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
    """Fake OpenRouterClient that returns canned streaming deltas (dict format)."""

    def __init__(self) -> None:
        self.chat_calls: list[tuple[str, list[dict]]] = []

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        self.chat_calls.append((model, messages))
        for chunk in ["Hello", ", ", "world."]:
            yield {"reasoning": None, "content": chunk}

    async def chat_completion_clean(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str | None, str]:
        return (None, "Hello, world.")

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


# ---------------------------------------------------------------------------
# Reasoning-specific fakes
# ---------------------------------------------------------------------------


class FakeStreamReasoningClient:
    """Fake that streams reasoning via ``delta.reasoning``, then clean answer."""

    def __init__(self) -> None:
        self.chat_calls: list[tuple[str, list[dict]]] = []

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        self.chat_calls.append((model, messages))
        yield {"reasoning": "thinking...", "content": None}
        yield {"reasoning": None, "content": "answer"}

    async def close(self) -> None:
        pass


class FakeStreamThinkLeakClient:
    """Fake that simulates a model leaking ``<think>`` (minimax-style).

    The fake chat_completion_stream uses ThinkSplitter internally (as the
    real client does), yielding already-separated reasoning/content dicts.
    """

    def __init__(self) -> None:
        self.chat_calls: list[tuple[str, list[dict]]] = []

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        from second_brain.reasoning import ThinkSplitter

        self.chat_calls.append((model, messages))
        splitter = ThinkSplitter()
        # Use no trailing whitespace for deterministic test expectations.
        raw_content = "<think>deep</think>answer"
        for r_piece, c_piece in splitter.feed(raw_content):
            if r_piece:
                yield {"reasoning": r_piece, "content": None}
            if c_piece:
                yield {"reasoning": None, "content": c_piece}
        r_rem, c_rem = splitter.flush()
        if r_rem:
            yield {"reasoning": r_rem, "content": None}
        if c_rem:
            yield {"reasoning": None, "content": c_rem}

    async def close(self) -> None:
        pass


class FakeClientClean:
    """Fake that returns raw content with <think> for chat_completion_clean test."""

    def __init__(self) -> None:
        self.chat_calls: list[tuple[str, list[dict]]] = []

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> Any:
        self.chat_calls.append((model, messages))
        return {"choices": [{"message": {"content": "<think>r</think>answer"}}]}

    async def chat_completion_clean(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str | None, str]:
        from second_brain.reasoning import strip_think

        resp = await self.chat_completion(
            model,
            messages,
            response_format=response_format,
            extra_body=extra_body,
        )
        raw = resp["choices"][0]["message"]["content"]
        return strip_think(raw)

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Reasoning tests
# ---------------------------------------------------------------------------


class TestReasoning:
    """Verify reasoning_delta emission and clean answer extraction."""

    async def test_reasoning_delta_emitted(self, tmp_path: Path) -> None:
        """When delta.reasoning is present, reasoning_delta is emitted."""
        store, vec_store = _seed_store_and_vec(tmp_path)
        client = FakeStreamReasoningClient()
        embedder = FakeEmbedder()
        cfg = SimpleNamespace(
            models=SimpleNamespace(chat="test-chat-model"),
        )

        events: list[dict[str, Any]] = []
        async for ev in chat_stream(
            "test", cfg, store, vec_store, embedder, client, k=5
        ):
            events.append(ev)

        types = [e["type"] for e in events]
        # Sequence: thinking, tool_call, tool_result, reasoning_delta, answer_delta, done
        assert types[0] == "thinking"
        assert types[1] == "tool_call"
        assert types[2] == "tool_result"
        assert "reasoning_delta" in types
        assert "answer_delta" in types
        assert types[-1] == "done"

        # Verify reasoning_delta content
        reasoning = "".join(
            e["content"] for e in events if e["type"] == "reasoning_delta"
        )
        assert reasoning == "thinking..."

        # Verify answer content is clean
        answer = "".join(
            e["content"] for e in events if e["type"] == "answer_delta"
        )
        assert answer == "answer"

        vec_store.close()

    async def test_think_leak_stripped(self, tmp_path: Path) -> None:
        """When <think> leaks into content, it is stripped, not in answer."""
        store, vec_store = _seed_store_and_vec(tmp_path)
        client = FakeStreamThinkLeakClient()
        embedder = FakeEmbedder()
        cfg = SimpleNamespace(
            models=SimpleNamespace(chat="test-chat-model"),
        )

        events: list[dict[str, Any]] = []
        async for ev in chat_stream(
            "test", cfg, store, vec_store, embedder, client, k=5
        ):
            events.append(ev)

        # Verify no <think> in any answer_delta
        for e in events:
            if e["type"] == "answer_delta":
                assert "<think>" not in e["content"]

        # Concatenated answer should be clean
        answer = "".join(
            e["content"] for e in events if e["type"] == "answer_delta"
        )
        assert answer == "answer"

        # Reasoning delta should contain the stripped think block
        reasoning = "".join(
            e["content"] for e in events if e["type"] == "reasoning_delta"
        )
        assert reasoning == "deep"

        vec_store.close()

    async def test_chat_once_clean_with_think_leak(self, tmp_path: Path) -> None:
        """chat_once returns only the clean answer (no <think>)."""
        store, vec_store = _seed_store_and_vec(tmp_path)
        client = FakeStreamThinkLeakClient()
        embedder = FakeEmbedder()
        cfg = SimpleNamespace(
            models=SimpleNamespace(chat="test-chat-model"),
        )

        answer = await chat_once(
            "test", cfg, store, vec_store, embedder, client
        )
        assert "<think>" not in answer
        assert answer == "answer"

        vec_store.close()

    async def test_chat_completion_clean(self) -> None:
        """chat_completion_clean strips <think> from raw content."""
        client = FakeClientClean()
        r, c = await client.chat_completion_clean(
            "test-model",
            [{"role": "user", "content": "hello"}],
        )
        assert r == "r"
        assert c == "answer"


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
