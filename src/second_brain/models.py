"""Pydantic v2 data contracts for the Second Brain.

Defines every schema used by the daemon pipeline (Phase 1B) and the state
persistence layer. Every field name and default here is a load-bearing contract.

See §4 (Data Formats), §4.4 (state.json), §4.6 (page types), §5 (pipeline),
§12.2 (structured output) in ARCHITECTURE.md.
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

# -- StrEnum helpers ----------------------------------------------------------


class IngestStage(StrEnum):
    """Per-source pipeline stage (§12.3 state machine)."""

    SEEN = auto()
    HASHING = auto()
    NORMALIZED = auto()
    EXTRACTED = auto()
    LINKED = auto()
    WIKI_MERGED = auto()
    INDEXED = auto()
    DONE = auto()
    FAILED = auto()


class TopicAction(StrEnum):
    """Librarian decision: match existing topic or propose new one."""

    MATCH = auto()
    NEW = auto()


class PageType(StrEnum):
    """Typed infobox schemas (§4.6). Default is CONCEPT."""

    CONCEPT = auto()
    PERSON = auto()
    WORK = auto()
    PROJECT = auto()
    TOOL = auto()
    PLACE = auto()
    EVENT = auto()
    NOTE = auto()


# -- Front-matter / ingestion schemas ----------------------------------------


class SourceMeta(BaseModel):
    """Front-matter block written into every 50-sources/*.md file (§4.1)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    source: str
    type: str
    ingested: str
    sha256: str
    tokens: int
    topics: list[str] = []


class TopicCandidate(BaseModel):
    """Raw candidate topic before the linking/rerank pass."""

    name: str
    confidence: float
    tldr: str | None = None


class LinkDecision(BaseModel):
    """Per-topic entry in the librarian's strict JSON output (§12.2)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    action: TopicAction
    target_slug: str
    confidence: float
    merged_section: str


class LibrarianOutput(BaseModel):
    """Top-level structured output from the extract+link LLM call (§5 skeleton).

    This model is serialised to JSON schema for OpenRouter ``response_format``
    with ``extra="forbid"`` so the model cannot invent extra fields.
    """

    model_config = ConfigDict(extra="forbid")

    tldr: str
    topics: list[LinkDecision]


# -- State persistence schemas (§4.4) ----------------------------------------


class TopicState(BaseModel):
    """A single topic node in the knowledge graph."""

    model_config = ConfigDict(extra="ignore")

    title: str
    type: PageType = PageType.CONCEPT
    tags: list[str] = []
    aliases: list[str] = []
    sources: list[str] = []
    links_to: list[str] = []
    linked_from: list[str] = []
    confidence: float = 0.0
    created: str
    updated: str


class SourceState(BaseModel):
    """Per-source entry in the registry with pipeline tracking."""

    model_config = ConfigDict(extra="ignore")

    sha256: str
    topics: list[str] = []
    near_duplicates: list[str] = []
    raw: str
    embedding_model: str | None = None
    stage: IngestStage = IngestStage.SEEN
    tokens: int = 0
    type: str = "text"
    ingested: str = ""
    error: str | None = None
    partial: bool = False


class BrainState(BaseModel):
    """Root state object — one per ``.brain/state.json`` (§4.4)."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    topics: dict[str, TopicState] = {}
    sources: dict[str, SourceState] = {}
    updated: str = ""
    # Phase 4 scheduler fields (§8 Track 5-2a). Defaults keep older
    # ``state.json`` files loadable (backward compat via ``extra="ignore"``).
    last_compaction_ts: str = ""        # ISO 8601 of last compaction run
    sources_since_compaction: int = 0   # counter, reset on compaction

    @model_validator(mode="before")
    @classmethod
    def _migrate(cls, data: Any) -> Any:
        """Transparent in-load migration hook (§12.6).

        Version 1 migration is intentionally small: old state files without a
        root schema marker are treated as v1, and the old ``last_updated`` key
        is accepted as ``updated`` when present.
        """
        if not isinstance(data, dict):
            return data
        migrated = dict(data)
        migrated.setdefault("schema_version", 1)
        if "updated" not in migrated and "last_updated" in migrated:
            migrated["updated"] = migrated["last_updated"]
        return migrated
