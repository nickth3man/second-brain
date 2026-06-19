"""Page index — resolves wikilink targets and computes backlinks.

Built from ``BrainStateStore`` at render time.  Every topic registered
in the store becomes a ``PageEntry`` that the wikilink renderer queries
for existence, self-detection, and backlink computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from second_brain.slug import slugify
from second_brain.state import BrainStateStore


@dataclass
class PageEntry:
    """A single topic entry in the page index.

    Mirrors ``TopicState`` fields relevant to link resolution.
    """

    slug: str
    title: str
    type: str
    aliases: list[str] = field(default_factory=list)
    links_to: list[str] = field(default_factory=list)
    linked_from: list[str] = field(default_factory=list)


class PageIndex:
    """Wikilink resolution index built from a ``BrainStateStore``.

    Supports slug-based existence checks, title/alias-based resolution
    (case-insensitive), and backlink computation.
    """

    def __init__(self, entries: dict[str, PageEntry]) -> None:
        self.entries = entries

    @classmethod
    def from_store(cls, store: BrainStateStore) -> PageIndex:
        """Build a ``PageIndex`` from every topic in *store*.state."""
        entries: dict[str, PageEntry] = {}
        for slug, ts in store.state.topics.items():
            entries[slug] = PageEntry(
                slug=slug,
                title=ts.title,
                type=str(ts.type),
                aliases=list(ts.aliases),
                links_to=list(ts.links_to),
                linked_from=list(ts.linked_from),
            )
        return cls(entries)

    def slug_for_target(self, target: str) -> str:
        """Return the slug that *target* would produce without checking existence."""
        return slugify(target)

    def resolve(self, target: str) -> str | None:
        """Resolve a wikilink target to its canonical slug.

        Tries in order:
        1. Slugify the target and look it up directly.
        2. Match the target (case-insensitive) against every entry's title.
        3. Match the target (case-insensitive) against every entry's aliases.

        Returns:
            The resolved slug, or ``None`` if no match is found.
        """
        # 1. Direct slug match
        slug = slugify(target)
        if slug in self.entries:
            return slug

        # 2. Title match (case-insensitive)
        target_lower = target.lower()
        for entry in self.entries.values():
            if entry.title.lower() == target_lower:
                return entry.slug

        # 3. Alias match (case-insensitive)
        for entry in self.entries.values():
            for alias in entry.aliases:
                if alias.lower() == target_lower:
                    return entry.slug

        return None

    def exists(self, slug: str) -> bool:
        """Return ``True`` if the slug is registered in the index."""
        return slug in self.entries

    def backlinks(self, slug: str) -> list[str]:
        """Return slugs of pages whose ``links_to`` includes *slug*.

        These are topics that link to the given slug — the ``## See also``
        candidates.
        """
        return [s for s, e in self.entries.items() if slug in e.links_to]
