"""YAML front-matter split / dump helpers.

Used by the normalise stage (writing 50-sources/ files) and the wiki-writer
(reading / writing 90-wiki/ pages). Relies on PyYAML for safe parsing.
"""

from __future__ import annotations

import yaml

# -- public helpers -----------------------------------------------------------


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Separate YAML front-matter from the markdown body.

    Front-matter opens with a line whose stripped content is ``---`` and closes
    with the next such line. Parsing is **line-based** so an empty block
    (``---\\n---\\nbody``) is detected correctly — a substring search for
    ``\\n---\\n`` misses it because the closing delimiter overlaps the opening.

    The YAML block is parsed with ``yaml.safe_load`` (returns ``{}`` when empty).

    Returns:
        ``(meta_dict, body)``.  When no front-matter is detected the meta dict
        is empty and the full text is returned as body.

    Raises:
        ValueError: wrapping any ``yaml.YAMLError`` with context about which
            file or input was being parsed.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, text

    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            yaml_block = "".join(lines[1:i])
            body = "".join(lines[i + 1 :])
            try:
                meta = yaml.safe_load(yaml_block) or {}
                if not isinstance(meta, dict):
                    raise ValueError(
                        f"Front-matter must be a mapping, got {type(meta).__name__}"
                    )
            except yaml.YAMLError as exc:
                raise ValueError(f"Malformed YAML front-matter: {exc}") from exc
            return dict(meta), body

    # Opening "---" with no closing delimiter -> not valid front-matter.
    return {}, text


def dump_frontmatter(meta: dict, body: str) -> str:
    """Wrap *meta* as YAML front-matter and append *body*.

    Always produces::

        ---
        <yaml>
        ---
        <body>

    A single trailing newline is ensured on *body*.
    """
    yaml_str = yaml.safe_dump(
        dict(meta),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    body = body.rstrip("\n") + "\n"
    return f"---\n{yaml_str}---\n{body}"
