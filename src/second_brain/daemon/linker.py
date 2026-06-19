"""Linking layer — app decides match/new based on slug existence or embedding similarity.

Phase 1 uses naive slug-match (:class:`SlugLinker`).
Phase 2 swaps to :class:`EmbeddingLinker` that uses cosine similarity
>= ``cfg.ingestion.merge_threshold`` to decide merge vs. spawn.
Both implement the ``Linker`` protocol so the swap is a drop-in replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from second_brain.models import LinkDecision, TopicAction
from second_brain.slug import slugify

if TYPE_CHECKING:
    from second_brain.state import BrainStateStore
    from second_brain.vectors.embed import Embedder
    from second_brain.vectors.store import VectorStore


@dataclass
class LinkContext:
    """Context object passed into ``Linker.link()``.

    Carries the brain state store plus optional embedding infrastructure
    so the linker can decide match vs. new on semantic or lexical grounds.
    """

    brain_store: BrainStateStore
    embedder: Embedder | None = None
    vec_store: VectorStore | None = None
    source_id: str = ""
    source_chunks: list[tuple[str, list[float]]] = field(default_factory=list)


class Linker(Protocol):
    """Protocol for the linking layer.

    Phase 1 :class:`SlugLinker` — slug-existence check.
    Phase 2 :class:`EmbeddingLinker` — embed candidate names and compare
    against existing topic centroids.
    """

    async def link(
        self,
        decisions: list[LinkDecision],
        ctx: LinkContext,
    ) -> list[LinkDecision]:
        ...


class SlugLinker:
    """Naive slug-match linker.

    Slugifies each candidate name and checks whether that slug already
    exists in the store's topic registry.  If it does -> ``MATCH``, else
    ``NEW``.

    This is the Phase 1 fallback — no embeddings needed.
    """

    async def link(
        self,
        decisions: list[LinkDecision],
        ctx: LinkContext,
    ) -> list[LinkDecision]:
        result: list[LinkDecision] = []
        store = ctx.brain_store
        for d in decisions:
            slug = slugify(d.name)
            action = (
                TopicAction.MATCH
                if slug in store.state.topics
                else TopicAction.NEW
            )
            result.append(
                LinkDecision(
                    name=d.name,
                    action=action,
                    target_slug=slug,
                    confidence=d.confidence,
                    merged_section=d.merged_section,
                )
            )
        return result


class EmbeddingLinker:
    """Embedding-based linker.

    Embeds each candidate name and compares cosine similarity against
    existing topic centroids.  Similarity >= threshold -> ``MATCH``,
    else ``NEW`` (with a slugified name for the new topic).
    """

    def __init__(
        self,
        embedder: Any,
        vec_store: Any,
        threshold: float = 0.70,
    ) -> None:
        self.embedder = embedder
        self.vec_store = vec_store
        self.threshold = threshold

    async def link(
        self,
        decisions: list[LinkDecision],
        ctx: LinkContext,  # noqa: ARG002 — uses own embedder/vec_store
    ) -> list[LinkDecision]:
        result: list[LinkDecision] = []
        for d in decisions:
            vec = await self.embedder.embed_one(d.name)
            best = self.vec_store.best_topic_for_vector(vec)
            if best is not None and best[1] >= self.threshold:
                action = TopicAction.MATCH
                target_slug = best[0]
            else:
                action = TopicAction.NEW
                target_slug = slugify(d.name)
            result.append(
                LinkDecision(
                    name=d.name,
                    action=action,
                    target_slug=target_slug,
                    confidence=d.confidence,
                    merged_section=d.merged_section,
                )
            )
        return result
