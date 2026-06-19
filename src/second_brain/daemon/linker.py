"""Linking layer — app decides match/new based on slug existence.

Phase 1 uses naive slug-match (no embeddings yet).  Phase 2 will swap to an
``EmbeddingLinker`` that uses cosine similarity ≥ ``cfg.ingestion.merge_threshold``
to decide merge vs. spawn.  Both implement the ``Linker`` protocol so the
swap is a drop-in replacement.
"""

from __future__ import annotations

from typing import Protocol

from second_brain.models import LinkDecision, TopicAction
from second_brain.slug import slugify


class Linker(Protocol):
    """Protocol for the linking layer.

    Phase 2 will provide :class:`EmbeddingLinker` that embeds candidate
    names and compares against existing topic centroids.
    """

    def link(
        self,
        decisions: list[LinkDecision],
        store,  # BrainStateStore
    ) -> list[LinkDecision]:
        ...


class SlugLinker:
    """Naive slug-match linker.

    App-slugifies each candidate name and checks whether that slug already
    exists in the store's topic registry.  If it does → ``MATCH``, else
    ``NEW``.
    """

    def link(
        self,
        decisions: list[LinkDecision],
        store,
    ) -> list[LinkDecision]:
        result: list[LinkDecision] = []
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
