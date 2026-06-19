"""Phase 1A data-contract tests — no network, no API key needed.

Tests models, front-matter, state persistence, backup recovery, and the state
machine.  Every test uses ``tmp_path`` or an in-memory fake.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from second_brain.frontmatter import dump_frontmatter, split_frontmatter
from second_brain.models import (
    BrainState,
    IngestStage,
    LibrarianOutput,
    LinkDecision,
    PageType,
    SourceState,
    TopicAction,
    TopicState,
)
from second_brain.state import BrainStateStore, now_iso

# ── helpers ──────────────────────────────────────────────────────────────────


class _FakeCfg:
    """Minimal config stub — satisfies BrainStateStore's ``cfg.brain_root``."""

    def __init__(self, brain_root: Path) -> None:
        self.brain_root = brain_root


# ── front-matter ─────────────────────────────────────────────────────────────


class TestFrontmatter:
    """split_frontmatter / dump_frontmatter round-trips and edge cases."""

    def test_round_trip_no_frontmatter(self) -> None:
        body = "# Just a heading\n\nSome content.\n"
        meta, parsed_body = split_frontmatter(body)
        assert meta == {}
        assert parsed_body == body

    def test_round_trip_with_frontmatter(self) -> None:
        original_meta = {"title": "Test", "tags": ["a", "b"]}
        original_body = "# Test\n\nBody text.\n"
        dumped = dump_frontmatter(original_meta, original_body)
        parsed_meta, parsed_body = split_frontmatter(dumped)
        assert parsed_meta == original_meta
        assert parsed_body == original_body

    def test_nested_list_value(self) -> None:
        meta = {"sources": [{"name": "A", "year": 2024}, {"name": "B", "year": 2025}]}
        body = "nested\n"
        dumped = dump_frontmatter(meta, body)
        parsed_meta, parsed_body = split_frontmatter(dumped)
        assert parsed_meta == meta
        assert parsed_body == body

    def test_empty_frontmatter(self) -> None:
        text = "---\n---\nbody content\n"
        meta, body = split_frontmatter(text)
        assert meta == {}
        assert body == "body content\n"

    def test_no_closing_delimiter_is_body(self) -> None:
        text = "---\nkey: val\nno closing marker\n"
        meta, body = split_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_malformed_yaml_raises(self) -> None:
        text = "---\n  invalid yaml: [\n---\nbody\n"
        with pytest.raises(ValueError, match="Malformed YAML"):
            split_frontmatter(text)

    def test_dump_ensures_trailing_newline(self) -> None:
        result = dump_frontmatter({"x": 1}, "no newline")
        assert result.endswith("\n")


# ── state store ──────────────────────────────────────────────────────────────


class TestBrainStateStoreLoad:
    """Save then reload — topics round-trip."""

    def test_save_and_reload(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore(cfg)
        store.ensure_topic("rag", "RAG & Vector Search")
        store.save()

        store2 = BrainStateStore.load(cfg)
        assert "rag" in store2.state.topics
        assert store2.state.topics["rag"].title == "RAG & Vector Search"

    def test_load_missing_directory(self, tmp_path: Path) -> None:
        """Loading on a non-existent .brain/ dir succeeds with fresh state."""
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        assert isinstance(store.state, BrainState)
        assert len(store.state.topics) == 0


class TestBackupRecovery:
    """Recovery chain: corrupt primary → falls back to .bak."""

    def test_recover_from_bak(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore(cfg)
        store.ensure_topic("survivor", "Survivor Topic")
        store.save()

        # Corrupt the primary
        store.path.write_text("{garbage}")

        # Load should recover from .bak
        store2 = BrainStateStore.load(cfg)
        assert "survivor" in store2.state.topics
        assert store2.state.topics["survivor"].title == "Survivor Topic"

    def test_recover_from_bak_1(self, tmp_path: Path) -> None:
        """Fall through primary .bak → .bak-1."""
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore(cfg)

        # Write primary, save (creates .bak)
        store.ensure_topic("deep", "Deep Survivor")
        store.save()

        # Overwrite primary with valid-but-older state, then save again
        # so that .bak has no topic and .bak-1 holds the topic.
        store2 = BrainStateStore(cfg)
        store2.save()  # now .bak = clean state (no topics)

        # Corrupt primary + .bak
        store2.path.write_text("{garbage}")
        (store2.path.parent / f"{store2.path.name}.bak").write_text("{garbage}")

        # Restore a topic into .bak-1 manually
        (store2.path.parent / f"{store2.path.name}.bak-1").write_text(
            json.dumps({
                "schema_version": 1,
                "topics": {
                    "deep": {
                        "title": "Deep Survivor",
                        "type": "concept",
                        "tags": [],
                        "aliases": [],
                        "sources": [],
                        "links_to": [],
                        "linked_from": [],
                        "confidence": 0.0,
                        "created": "2026-06-19T00:00:00Z",
                        "updated": "2026-06-19T00:00:00Z",
                    },
                },
                "sources": {},
                "updated": "",
            })
        )

        store3 = BrainStateStore.load(cfg)
        assert "deep" in store3.state.topics
        assert store3.state.topics["deep"].title == "Deep Survivor"


class TestAllFailFresh:
    """When primary + all backups are corrupt → fresh empty state."""

    def test_all_corrupt_returns_fresh(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore(cfg)
        store.ensure_topic("lost", "Lost Topic")
        store.save()

        # Corrupt everything
        store.path.write_text("garbage")
        for suffix in (".bak", ".bak-1", ".bak-2"):
            (store.path.parent / f"{store.path.name}{suffix}").write_text("garbage")

        store2 = BrainStateStore.load(cfg)
        assert isinstance(store2.state, BrainState)
        assert len(store2.state.topics) == 0


class TestTransition:
    """State machine transitions (Idempotent)."""

    def test_moves_stage(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        st = SourceState(sha256="abc", raw="inbox/test.md")
        store.record_source("src1", st)
        assert st.stage == IngestStage.SEEN

        store.transition("src1", IngestStage.NORMALIZED)
        assert st.stage == IngestStage.NORMALIZED

    def test_error_on_failed(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        st = SourceState(sha256="abc", raw="inbox/test.md")
        store.record_source("src1", st)

        store.transition("src1", IngestStage.FAILED, error="timeout")
        assert st.stage == IngestStage.FAILED
        assert st.error == "timeout"

    def test_idempotent_missing_source(self, tmp_path: Path) -> None:
        """Transition on nonexistent source is a no-op."""
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.transition("nonexistent", IngestStage.DONE)  # should not raise


class TestEnsureTopic:
    """ensure_topic is idempotent."""

    def test_creates_and_returns(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        t = store.ensure_topic("my-slug", "My Title", page_type=PageType.PROJECT, confidence=0.8)
        assert t.title == "My Title"
        assert t.type == PageType.PROJECT
        assert t.confidence == 0.8
        assert t.created == t.updated

    def test_second_call_returns_same(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        t1 = store.ensure_topic("slug", "Original", confidence=0.5)
        created_original = t1.created

        t2 = store.ensure_topic("slug", "Overwritten?", confidence=0.9)
        assert t2 is t1
        assert t2.title == "Original"  # unchanged
        assert t2.created == created_original


class TestAddSourceToTopic:
    """add_source_to_topic deduplicates and returns bool."""

    def test_adds_source(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.ensure_topic("t", "Topic")
        assert store.add_source_to_topic("t", "src1") is True
        assert store.state.topics["t"].sources == ["src1"]

    def test_dedup_returns_false(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.ensure_topic("t", "Topic")
        store.add_source_to_topic("t", "src1")
        assert store.add_source_to_topic("t", "src1") is False  # already present
        assert store.state.topics["t"].sources == ["src1"]

    def test_missing_topic_returns_false(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        assert store.add_source_to_topic("nonexistent", "src1") is False


class TestRecordLink:
    """record_link populates links_to + linked_from symmetrically."""

    def test_bidirectional(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.ensure_topic("a", "Topic A")
        store.ensure_topic("b", "Topic B")

        store.record_link("a", "b")
        assert "b" in store.state.topics["a"].links_to
        assert "a" in store.state.topics["b"].linked_from

    def test_ignores_missing(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.ensure_topic("a", "Topic A")
        # 'b' does not exist — no error
        store.record_link("a", "b")
        assert "b" not in store.state.topics["a"].links_to

    def test_dedup(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.ensure_topic("a", "A")
        store.ensure_topic("b", "B")
        store.record_link("a", "b")
        store.record_link("a", "b")  # second call — no-op
        assert store.state.topics["a"].links_to == ["b"]


class TestAppendChangelog:
    """Changelog writes one JSON line per call."""

    def test_writes_one_line_per_call(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.append_changelog({"action": "merge", "from": "a", "into": "b"})
        store.append_changelog({"action": "rewrite", "topic": "x"})
        lines = store.changelog_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        entry0 = json.loads(lines[0])
        assert entry0["action"] == "merge"
        assert "ts" in entry0

    def test_survives_across_calls(self, tmp_path: Path) -> None:
        cfg = _FakeCfg(tmp_path)
        store = BrainStateStore.load(cfg)
        store.append_changelog({"event": "first"})
        store2 = BrainStateStore.load(cfg)  # new instance, same file
        store2.append_changelog({"event": "second"})
        lines = store2.changelog_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2


class TestLibrarianOutputStrict:
    """LibrarianOutput validates strict JSON and rejects extra fields."""

    def test_validates_valid_payload(self) -> None:
        payload = {
            "tldr": "A summary of the source.",
            "topics": [
                {
                    "name": "Machine Learning",
                    "action": "match",
                    "target_slug": "machine-learning",
                    "confidence": 0.85,
                    "merged_section": "## ML section\n\nContent.",
                },
            ],
        }
        lo = LibrarianOutput.model_validate(payload)
        assert lo.tldr == "A summary of the source."
        assert len(lo.topics) == 1
        assert lo.topics[0].action == TopicAction.MATCH

    def test_model_validate_json_valid(self) -> None:
        """model_validate_json accepts a well-formed strict payload."""
        payload = json.dumps({
            "tldr": "Valid summary.",
            "topics": [
                {
                    "name": "Deep Learning",
                    "action": "match",
                    "target_slug": "deep-learning",
                    "confidence": 0.92,
                    "merged_section": "## DL\n\nContent.",
                },
            ],
        })
        lo = LibrarianOutput.model_validate_json(payload)
        assert lo.tldr == "Valid summary."
        assert lo.topics[0].name == "Deep Learning"

    def test_rejects_extra_fields(self) -> None:
        payload = """{"tldr": "x", "topics": [], "extra_field": "should be rejected"}"""
        with pytest.raises(ValidationError):
            LibrarianOutput.model_validate_json(payload)

    def test_model_dump_round_trip(self) -> None:
        """Serialise and deserialise via JSON."""
        original = LibrarianOutput(
            tldr="test",
            topics=[
                LinkDecision(
                    name="AI",
                    action=TopicAction.NEW,
                    target_slug="ai",
                    confidence=0.9,
                    merged_section="# AI\n",
                ),
            ],
        )
        data = original.model_dump(mode="json")
        restored = LibrarianOutput.model_validate(data)
        assert restored.tldr == "test"
        assert restored.topics[0].name == "AI"


class TestNowIso:
    """now_iso returns ISO 8601 with Z suffix."""

    def test_format(self) -> None:
        ts = now_iso()
        assert ts.endswith("Z")
        assert "T" in ts
        # Rough check: it should be ~20 chars ("2026-06-19T14:23:01Z" = 20)
        assert len(ts) == 20


class TestTopicStateDefaults:
    """TopicState defaults match §4.4."""

    def test_defaults(self) -> None:
        ts = TopicState(title="Test", created="2026-01-01", updated="2026-06-01")
        assert ts.type == PageType.CONCEPT
        assert ts.tags == []
        assert ts.confidence == 0.0


class TestBrainStateSchemaVersion:
    """schema_version defaults to 1."""

    def test_default(self) -> None:
        bs = BrainState()
        assert bs.schema_version == 1
