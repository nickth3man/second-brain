"""Phase 2 follow-up tests — near-dup pipeline integration test.

End-to-end ``ingest_file`` run that exercises the Phase 2.1 near-dup
branch (``embedder is not None and vec_store is not None``).  Two text
files with very similar content are ingested in sequence; the second
ingest should detect the first as a near-duplicate.

No network, no API.  Uses real sqlite-vec with dim=8 + deterministic
fake embedder (pattern from ``test_phase2b.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from second_brain.config import TypesCfg
from second_brain.daemon.index import DebouncedIndex
from second_brain.daemon.linker import EmbeddingLinker
from second_brain.daemon.pipeline import ingest_file
from second_brain.models import IngestStage
from second_brain.state import BrainStateStore
from second_brain.vectors.store import VectorStore

DIM = 8


# ---------------------------------------------------------------------------
# Stubs (same pattern as test_phase2b)
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


def _ensure_dirs(cfg: _FakeCfg) -> None:
    (cfg.brain_root / "00-inbox").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / "50-sources").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / "90-wiki").mkdir(parents=True, exist_ok=True)
    (cfg.brain_root / ".brain").mkdir(parents=True, exist_ok=True)


def _write_inbox(cfg: _FakeCfg, name: str, content: str) -> Path:
    p = cfg.brain_root / "00-inbox" / name
    p.write_text(content, encoding="utf-8")
    return p


class FakeClient:
    """Fake OpenRouter client returning a fixed extraction payload."""

    def __init__(self, payload: dict | None = None) -> None:
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

    async def chat_completion(self, *args, **kwargs) -> dict:  # noqa: ARG002
        return {
            "choices": [{"message": {"content": json.dumps(self.payload)}}]
        }

    async def close(self) -> None:
        pass


class ConstantEmbedder:
    """Deterministic embedder returning a fixed unit vector for any text.

    Because every chunk of every source gets the same vector, the
    source-centroid tier of ``find_near_duplicates_for_source`` returns
    that same vector for the first source, and the second source's mean
    chunk embedding is identical -> cosine similarity == 1.0.
    """

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        # Unit vector along axis 0.
        self._vec = [1.0] + [0.0] * (dim - 1)

    async def embed_one(self, text: str) -> list[float]:  # noqa: ARG002
        return list(self._vec)

    async def embed_texts(
        self, texts: list[str]
    ) -> list[list[float]]:
        return [list(self._vec) for _ in texts]

    async def embed_query(self, query: str) -> list[float]:  # noqa: ARG002
        return list(self._vec)

    async def ensure_dim(self) -> int:
        return self.dim


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestPipelineNearDup:
    """End-to-end ``ingest_file`` exercising the near-dup branch."""

    @pytest.mark.asyncio
    async def test_second_ingest_flags_near_duplicate(
        self, tmp_path: Path
    ) -> None:
        cfg = _make_cfg(tmp_path)
        _ensure_dirs(cfg)

        client = FakeClient(
            payload={
                "tldr": "Near-dup integration test.",
                "topics": [
                    {
                        "name": "Shared Concept",
                        "action": "new",
                        "target_slug": "",
                        "confidence": 0.9,
                        "merged_section": "Synthesis body.",
                    }
                ],
            }
        )

        embedder = ConstantEmbedder(DIM)
        db_path = tmp_path / ".brain" / "embeddings.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        vec_store = VectorStore(db_path, model="test-embed", dim=DIM)

        linker = EmbeddingLinker(embedder, vec_store, threshold=0.7)
        store = BrainStateStore.load(cfg)
        index = DebouncedIndex(cfg, store)

        # Two inbox files with very similar content (identical headings
        # and body text -> identical chunk embeddings under ConstantEmbedder).
        body_one = "# Shared Topic\n\nThis is the shared body content."
        body_two = "# Shared Topic\n\nThis is the shared body content, again."
        path_one = _write_inbox(cfg, "note-one.md", body_one)
        path_two = _write_inbox(cfg, "note-two.md", body_two)

        stage_one = await ingest_file(
            path_one, cfg, store, client, linker, index,
            embedder=embedder, vec_store=vec_store,
        )
        assert stage_one == IngestStage.DONE

        first_source_id = next(
            sid for sid, s in store.state.sources.items()
            if s.sha256 and s.stage == IngestStage.DONE
        )

        # Sanity: the first source has chunks in the vec store, so the
        # centroid tier will be exercised by the second ingest.
        assert vec_store.source_centroid(first_source_id) is not None

        stage_two = await ingest_file(
            path_two, cfg, store, client, linker, index,
            embedder=embedder, vec_store=vec_store,
        )
        assert stage_two == IngestStage.DONE

        second_source_id = next(
            sid for sid, s in store.state.sources.items()
            if sid != first_source_id and s.stage == IngestStage.DONE
        )

        # The second source's near_duplicates must contain the first.
        near_dups = store.state.sources[second_source_id].near_duplicates
        assert first_source_id in near_dups, (
            f"expected {first_source_id!r} in near_duplicates, "
            f"got {near_dups!r}"
        )

        # Changelog must have a near_dup_detected entry for the second.
        changelog_path = cfg.brain_root / ".brain" / "changelog.jsonl"
        assert changelog_path.exists()
        lines = changelog_path.read_text(encoding="utf-8").strip().split("\n")
        near_dup_entries = [
            json.loads(ln)
            for ln in lines
            if '"action": "near_dup_detected"' in ln
        ]
        assert len(near_dup_entries) >= 1
        last_nd = near_dup_entries[-1]
        assert last_nd["source"] == second_source_id
        assert first_source_id in last_nd["near_duplicates"]

        vec_store.close()
