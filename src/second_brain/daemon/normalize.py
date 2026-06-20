"""Source normalization — write 50-sources/*.md with front-matter (§4.1, §5 [2]).

Dispatches parsing via :func:`parse_to_markdown` based on the pipeline stage.
Phase 3 Wave 1 handles text, code, structured, office, web, and ebook stages.
Wave 2 adds pdf, vision, audio, and video.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from second_brain.atomicio import write_atomic
from second_brain.config import Config
from second_brain.frontmatter import dump_frontmatter
from second_brain.models import SourceMeta
from second_brain.openrouter_client import OpenRouterClient
from second_brain.slug import slugify


def source_id_for(path: Path, body: str, ingested_iso: str) -> str:
    """Build a deterministic source ID: ``<YYYY-MM-DD>-<slug>``.

    The slug is derived from the filename stem via :func:`slugify`.  If the
    stem is empty (or produces an empty slug), the first six words of *body*
    are used instead.  The slug is truncated to 60 characters.
    """
    date_part = ingested_iso[:10]
    stem = path.stem
    slug = slugify(stem)
    if not slug:
        words = body.split()
        slug = slugify(" ".join(words[:6]))
    slug = slug[:60]
    return f"{date_part}-{slug}"


def sha256_of_file(path: Path) -> str:
    """Streaming SHA-256 digest of *path* (1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ``len(text) // 4``."""
    return len(text) // 4


async def normalize_text(
    path: Path,
    source_id: str,
    sha: str,
    ingested_iso: str,
    stage: str,
    cfg: Config,
    client: OpenRouterClient,
) -> tuple[Path, str]:
    """Normalise a raw inbox file into a ``50-sources/<source_id>.md`` file.

    The output file has YAML front-matter (:class:`SourceMeta`) followed by
    the parsed body and an empty ``## Summary`` section.

    Parsing is dispatched to the appropriate parser via
    :func:`parse_to_markdown` based on *stage*.

    Args:
        client: OpenRouter client (needed for Wave-2 vision/STT parsers;
            Wave-1 parsers ignore it).

    Returns:
        ``(dst_path, body)`` where *body* is the parsed markdown body.

    Raises:
        NotImplementedError: if *stage* is a Wave-2 stage (pdf, vision, audio,
            video).
        ValueError: if *stage* is unknown.
    """
    from second_brain.parse import parse_to_markdown

    body = await parse_to_markdown(path, stage, cfg, client)

    try:
        rel = path.resolve().relative_to(cfg.brain_root.resolve()).as_posix()
    except ValueError:
        rel = str(path)

    meta = SourceMeta(
        source=rel,
        type=stage,
        ingested=ingested_iso,
        sha256=sha,
        tokens=estimate_tokens(body),
    )

    output = f"{body}\n\n## Summary\n> (pending)\n"
    content = dump_frontmatter(meta.model_dump(mode="json"), output)

    dst = cfg.brain_root / "50-sources" / f"{source_id}.md"
    write_atomic(dst, content)
    return (dst, body)
