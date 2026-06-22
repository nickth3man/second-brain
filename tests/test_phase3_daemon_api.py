"""Phase 3 (gap closure) — daemon loopback HTTP API (§12.1).

Tests the daemon-owned search_brain endpoint that the web UI and CLI use to
satisfy the single-writer DB invariant: only the daemon process opens a
writeable VectorStore; everyone else goes through this loopback HTTP API.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from second_brain.daemon.api import create_daemon_app
from second_brain.vectors.retrieval import SearchHit

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Trivial embedder stub."""

    async def embed_query(self, query: str) -> list[float]:
        return [0.0] * 8


class _FakeVecStore:
    """Vec store stub returning one canned hit."""

    def vector_search_chunks(self, vec: list[float], k: int = 10) -> list[tuple[int, float]]:
        return [(1, 0.9)]

    def fts_search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        return []

    def get_chunk(self, rowid: int) -> dict[str, Any] | None:
        return {
            "source_id": "src-daemon",
            "topic_slug": "test-topic",
            "text": "hello world from the daemon-owned store",
        }


@pytest.fixture
def daemon_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Build a TestClient for the daemon app with a stubbed search_brain."""
    # Patch search_brain inside retrieval to return a canned SearchHit.
    async def _fake_search(query, store, embedder, *, k=10, merge_k=20):
        return [
            SearchHit(
                rowid=1,
                source_id="src-daemon",
                topic_slug="test-topic",
                text="hello world from the daemon-owned store",
                score=0.42,
            )
        ]

    monkeypatch.setattr(
        "second_brain.vectors.retrieval.search_brain", _fake_search
    )
    app = create_daemon_app(_FakeVecStore(), _FakeEmbedder(), SimpleNamespace())
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDaemonApi:
    """Daemon loopback HTTP endpoints."""

    def test_health(self, daemon_client: TestClient) -> None:
        resp = daemon_client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_search_brain_returns_hits(self, daemon_client: TestClient) -> None:
        resp = daemon_client.post(
            "/search_brain", json={"query": "hello", "k": 5}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "hits" in data
        assert len(data["hits"]) == 1
        hit = data["hits"][0]
        assert hit["source_id"] == "src-daemon"
        assert hit["topic_slug"] == "test-topic"
        assert hit["text"] == "hello world from the daemon-owned store"
        assert hit["score"] == pytest.approx(0.42)

    def test_search_brain_default_k(self, daemon_client: TestClient) -> None:
        """k defaults to 10 per the SearchRequest model."""
        resp = daemon_client.post(
            "/search_brain", json={"query": "anything"}
        )
        assert resp.status_code == 200
        assert "hits" in resp.json()

    def test_search_brain_handles_null_topic_slug(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A chunk with no topic_slug should serialize as empty string, not null."""
        async def _fake_search(query, store, embedder, *, k=10, merge_k=20):
            return [
                SearchHit(
                    rowid=2,
                    source_id="orphan-src",
                    topic_slug=None,
                    text="chunk with no topic",
                    score=0.1,
                )
            ]

        monkeypatch.setattr(
            "second_brain.vectors.retrieval.search_brain", _fake_search
        )
        app = create_daemon_app(_FakeVecStore(), _FakeEmbedder(), SimpleNamespace())
        client = TestClient(app)

        resp = client.post("/search_brain", json={"query": "x"})
        assert resp.status_code == 200
        hit = resp.json()["hits"][0]
        assert hit["topic_slug"] == ""
        assert hit["source_id"] == "orphan-src"

    def test_search_brain_returns_503_on_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When search_brain raises, the endpoint returns 503 (not 500).

        The response body must NOT leak a traceback (§12.7 hardening).
        """
        async def _boom(query, store, embedder, *, k=10, merge_k=20):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "second_brain.vectors.retrieval.search_brain", _boom
        )
        app = create_daemon_app(_FakeVecStore(), _FakeEmbedder(), SimpleNamespace())
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/search_brain", json={"query": "x"})
        assert resp.status_code == 503
        body = resp.text
        assert "Traceback" not in body
        assert "search failed" in body


class TestChatAgentSearchFn:
    """The chat agent's search_fn parameter (the remote-search path)."""

    async def test_chat_stream_with_search_fn(self, tmp_path: Path) -> None:
        """When search_fn is provided, the agent uses it instead of vec_store."""
        from second_brain.chat import chat_stream

        class FakeStreamClient:
            async def chat_completion_stream(
                self, model, messages, *, extra_body=None
            ) -> AsyncIterator[dict[str, Any]]:
                yield {"reasoning": None, "content": "Answer."}

            async def close(self) -> None:
                pass

        async def _search_fn(query: str, k: int) -> list[dict]:
            return [
                {
                    "source_id": "via-daemon",
                    "topic_slug": "remote-topic",
                    "text": "remote snippet",
                    "score": 0.9,
                }
            ]

        cfg = SimpleNamespace(models=SimpleNamespace(chat="test-chat-model"))
        store = SimpleNamespace()

        events: list[dict[str, Any]] = []
        async for ev in chat_stream(
            "hi",
            cfg,
            store,
            vec_store=None,
            embedder=None,
            client=FakeStreamClient(),
            k=5,
            search_fn=_search_fn,
        ):
            events.append(ev)

        types = [e["type"] for e in events]
        assert types[0] == "thinking"
        assert types[1] == "tool_call"
        tool_result = events[2]
        assert tool_result["type"] == "tool_result"
        assert len(tool_result["hits"]) == 1
        assert tool_result["hits"][0]["source_id"] == "via-daemon"
        assert tool_result["hits"][0]["snippet"] == "remote snippet"
        assert types[-1] == "done"

    async def test_chat_stream_search_fn_error_handled(
        self, tmp_path: Path
    ) -> None:
        """When search_fn raises, tool_result has error + empty hits (no crash)."""
        from second_brain.chat import chat_stream

        class FakeStreamClient:
            async def chat_completion_stream(
                self, model, messages, *, extra_body=None
            ) -> AsyncIterator[dict[str, Any]]:
                yield {"reasoning": None, "content": "answer"}

            async def close(self) -> None:
                pass

        async def _broken_search(query: str, k: int) -> list[dict]:
            msg = "daemon 503"
            raise RuntimeError(msg)

        cfg = SimpleNamespace(models=SimpleNamespace(chat="m"))
        events: list[dict[str, Any]] = []
        async for ev in chat_stream(
            "q",
            cfg,
            SimpleNamespace(),
            vec_store=None,
            embedder=None,
            client=FakeStreamClient(),
            k=5,
            search_fn=_broken_search,
        ):
            events.append(ev)

        tool_result = next(e for e in events if e["type"] == "tool_result")
        assert tool_result["hits"] == []
        assert "daemon 503" in tool_result["error"]
