"""Slugify — §10 slug rule for topic page names.

Rule: lowercase; replace spaces and underscores with hyphens; drop apostrophes
and parentheses; collapse runs of non-alphanumeric (except hyphen) into single
hyphens; strip leading/trailing hyphens.

Worked examples from ARCHITECTURE §10:
    "The Tourist Trap"            -> "the-tourist-trap"
    "Barrel (The Tourist Trap)"   -> "barrel-the-tourist-trap"
    "Members' NPCs"               -> "members-npcs"
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9-]+")


def slugify(name: str) -> str:
    """Convert a human-readable topic name into a URL-safe slug."""
    s = name.lower()
    s = s.replace(" ", "-").replace("_", "-")
    s = s.replace("'", "").replace("(", "").replace(")", "")
    s = _NON_ALNUM.sub("-", s)
    s = s.strip("-")
    return s
