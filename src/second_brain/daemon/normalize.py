"""Source normalization — write 50-sources/*.md with front-matter (§4.1, §5 [2]).

Handles text/code/structured stages only in Phase 1. Multimodal stages
(PDF, vision, office, audio, video) raise ``ValueError``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from second_brain.atomicio import write_atomic
from second_brain.config import Config
from second_brain.frontmatter import dump_frontmatter
from second_brain.models import SourceMeta
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
) -> Path:
    """Normalise a raw inbox file into a ``50-sources/<source_id>.md`` file.

    The output file has YAML front-matter (:class:`SourceMeta`) followed by
    the raw body and an empty ``## Summary`` section.

    Raises:
        ValueError: if *stage* is not one of ``text``, ``code``, or
            ``structured`` (unsupported in Phase 1).
    """
    supported = {"text", "code", "structured"}
    if stage not in supported:
        raise ValueError(f"unsupported in Phase 1: {stage}")

    body = path.read_text(encoding="utf-8", errors="replace")

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
    return dst
