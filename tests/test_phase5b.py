"""Phase 5B — FastAPI web UI tests.

Uses FastAPI TestClient (no live server).  Seeds a tmp brain_root with
2 topics and one 90-wiki page file containing a ``[[wikilink]]`` with
front-matter, then exercises every route.

No network, no API key required.  Search falls back to title-substring match.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from second_brain.frontmatter import dump_frontmatter
from second_brain.state import BrainStateStore
from second_brain.web.app import create_app

# ---------------------------------------------------------------------------
# Stub config (same pattern as test_phase5a.py)
# ---------------------------------------------------------------------------


class _FakeCfg:
    """Minimal config stub — satisfies web routes' ``cfg.brain_root`` + ``cfg.daemon``."""

    def __init__(self, brain_root: Path) -> None:
        self.brain_root = brain_root
        # Web routes build the daemon loopback URL from these (§12.1).
        self.daemon = SimpleNamespace(http_host="127.0.0.1", http_port=8001)


def _seed_store(tmp_path: Path) -> BrainStateStore:
    """Create a BrainStateStore with 2 topics and a wiki page, then save."""
    cfg_fake = _FakeCfg(tmp_path)
    store = BrainStateStore.load(cfg_fake)
    store.ensure_topic("test-topic", "Test Topic")
    store.ensure_topic("other", "Other Topic")
    store.record_link("other", "test-topic")
    store.save()  # persist so the web app loads the topics

    # Create the wiki page on disk
    wiki_dir = tmp_path / "90-wiki"
    wiki_dir.mkdir(exist_ok=True)
    page = wiki_dir / "test-topic.md"
    meta = {
        "title": "Test",
        "type": "concept",
        "created": "2026-06-01",
        "confidence": 0.9,
        "sources": ["src1", "src2"],
    }
    body = "\n## Synthesis\n\nHello [[Other Topic]] world.\n"
    page.write_text(dump_frontmatter(meta, body))

    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app(tmp_path: Path):
    """Build a FastAPI TestClient app with a seeded tmp brain_root."""
    cfg_fake = _FakeCfg(tmp_path)
    _seed_store(tmp_path)
    application = create_app(cfg_fake)
    return application


@pytest.fixture
def client(app):
    """FastAPI TestClient wrapping the seeded app."""
    # Clear the module-level _store singleton so the app loads fresh
    import second_brain.web.app as web_app

    web_app._store = None  # noqa: SLF001
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIndex:
    """GET /"""

    def test_index_200(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.text
        # Contains the topic title
        assert "Test Topic" in body
        # Contains "Topics" heading
        assert "Topics" in body

    def test_index_shows_totals(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        # Should mention source count and topic count
        body = resp.text
        assert "0 sources" in body or "sources" in body
        assert "2 topics" in body


class TestTopic:
    """GET /topic/{slug}"""

    def test_existing_topic_200(self, client: TestClient) -> None:
        resp = client.get("/topic/test-topic")
        assert resp.status_code == 200
        body = resp.text
        # Contains the <h1> title
        assert "<h1>Test</h1>" in body or "<h1>Test" in body
        # Rendered wikilink (blue link for "Other Topic")
        assert 'href="/topic/other"' in body
        assert 'class="wikilink"' in body
        # Infobox present (concept type)
        assert 'class="infobox"' in body

    def test_missing_topic_404(self, client: TestClient) -> None:
        resp = client.get("/topic/nonexistent")
        assert resp.status_code == 404

    def test_create_placeholder(self, client: TestClient) -> None:
        resp = client.get("/topic/nonexistent?action=create")
        assert resp.status_code == 200
        body = resp.text
        assert "does not exist yet" in body

    def test_topic_breadcrumbs(self, client: TestClient) -> None:
        resp = client.get("/topic/test-topic")
        assert resp.status_code == 200
        # Breadcrumbs contain "Home"
        assert "Home" in resp.text

    def test_topic_see_also(self, client: TestClient) -> None:
        resp = client.get("/topic/test-topic")
        assert resp.status_code == 200
        # "other" links to test-topic -> should appear in See also
        assert "See also" in resp.text
        assert "Other Topic" in resp.text


class TestSearch:
    """GET /search?q=..."""

    def test_empty_query_returns_page(self, client: TestClient) -> None:
        resp = client.get("/search")
        assert resp.status_code == 200
        assert "Search" in resp.text

    def test_search_fallback_title_match(self, client: TestClient) -> None:
        """Without an API key, falls back to title-substring matching."""
        resp = client.get("/search?q=test")
        assert resp.status_code == 200
        body = resp.text
        # Should include the fallback note
        assert "Semantic search unavailable" in body or "title matches" in body or "result" in body
        # Should not crash

    def test_search_no_results(self, client: TestClient) -> None:
        resp = client.get("/search?q=xyznonexistent12345")
        assert resp.status_code == 200
        body = resp.text
        assert "No results" in body or "0 result" in body

    def test_search_uses_daemon_when_reachable(
        self, app
    ) -> None:
        """When the daemon HTTP endpoint is reachable, /search uses its hits
        and does NOT fall back to title-substring matching.

        Verifies the §12.1 single-writer invariant: the web UI goes through
        the daemon loopback API rather than opening a writeable VectorStore.

        The shared httpx client is lifted to ``app.state`` at startup
        (Item 6b), so we mock the transport at client construction time and
        drive startup via the TestClient context manager.
        """
        import httpx

        import second_brain.web.app as web_app

        web_app._store = None  # noqa: SLF001 — fresh store load

        daemon_payload = {
            "hits": [
                {
                    "source_id": "src-daemon",
                    "topic_slug": "test-topic",
                    "text": "daemon-powered snippet about Test Topic",
                    "score": 0.42,
                }
            ]
        }

        def _handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/search_brain"
            return httpx.Response(200, json=daemon_payload)

        transport = httpx.MockTransport(_handler)
        # The shared client is created in the startup event; patch
        # AsyncClient construction so the startup-built client uses the mock.
        original_init = httpx.AsyncClient.__init__

        def _patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs.setdefault("transport", transport)
            return original_init(self, *args, **kwargs)

        httpx.AsyncClient.__init__ = _patched_init  # type: ignore[method-assign]
        try:
            # Context manager triggers lifespan startup (builds the shared
            # daemon_client) and shutdown (closes it).
            with TestClient(app) as client:
                resp = client.get("/search?q=anything")
        finally:
            httpx.AsyncClient.__init__ = original_init  # type: ignore[method-assign]

        assert resp.status_code == 200
        body = resp.text
        assert "src-daemon" in body
        # Should NOT show the fallback note when the daemon answered.
        assert "daemon not running" not in body


class TestHealth:
    """GET /health"""

    def test_health_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.text
        # Contains "Brain Health" heading
        assert "Brain Health" in body
        # Contains report fields
        assert "Topics" in body
        assert "Sources" in body
