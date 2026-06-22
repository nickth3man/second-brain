"""Structured extraction via LLM — Librarian output with JSON schema (§12.2).

Defines the system prompt, message builder, strict-JSON-schema wrapper, and
the extract function with the long-source strategy from §12.2.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import structlog
from pydantic import ValidationError

from second_brain.daemon.normalize import estimate_tokens
from second_brain.models import LibrarianOutput
from second_brain.openrouter_client import (
    CreditExhaustedError,
    OpenRouterAPIError,
)

log = structlog.get_logger(__name__)

DIRECT_TOKEN_LIMIT = 16_000
MAP_REDUCE_TOKEN_LIMIT = 200_000
MAP_CHUNK_TOKENS = 800
MAP_CHUNK_OVERLAP_TOKENS = 100
CHARS_PER_TOKEN = 4

# -- prompt --------------------------------------------------------------------

LIBRARIAN_SYSTEM_PROMPT = """\
You are the librarian for a personal second brain.

INPUT:
- A normalized source in markdown.
- Existing topic titles for context.

JOB:
1. Summarize the source in <=2 sentences.
2. Propose 3–7 topics this source belongs to.
3. For each topic: output its name, confidence (0.0–1.0), and a markdown
   section to merge into that topic's Synthesis.
4. Set `action` to "new" and `target_slug` to empty string for every topic.
   (The application decides linking via the linker, not the model.)

CONSTRAINTS:
- Never invent facts not in the source.
- Quote sparingly; prefer compression.
- If unsure about a link, confidence < 0.6.
"""

STRUCTURED_SYSTEM_PROMPT = """\
You are summarizing a structured/tabular data file for a personal second brain.

INPUT:
- A compact markdown summary of the dataset (columns, types, sample rows, shape).
- Existing topic titles for context.

JOB:
1. Produce **exactly ONE** topic that faithfully describes the dataset.
2. Name the topic after the dataset — a concise descriptive title derived
   from the content + filename (e.g. "NBA Team Game Stats — Data Dictionary"
   or "Player Per-Game Box Scores").
3. The topic's `merged_section`: a faithful markdown synthesis describing:
   - What the dataset is.
   - What each row represents (granularity).
   - The column groups, key metrics, and units.
   - What analysis or questions the dataset supports.
   Compress; do NOT dump rows.
4. Set `action` to "new" and `target_slug` to empty string.
5. Confidence should be 0.85+ (it's a direct description).

CONSTRAINTS:
- NEVER invent columns, metrics, or analysis capabilities not present.
- Quote sparingly; prefer compression.
"""

# -- exceptions ----------------------------------------------------------------


class ExtractionError(Exception):
    """Extraction failed after all retry/fallback attempts."""


class ConfidenceFloorError(Exception):
    """Raised when ALL extracted topics fall below the configured confidence_floor.

    Caught by the pipeline to quarantine the source. Per §12.2 — quarantine, no merge.
    """

    def __init__(self, source_id: str, n_topics: int, max_confidence: float, floor: float) -> None:
        self.source_id = source_id
        self.n_topics = n_topics
        self.max_confidence = max_confidence
        self.floor = floor
        super().__init__(
            f"all {n_topics} topics below confidence_floor={floor} (max={max_confidence})"
        )


@dataclass(frozen=True)
class ExtractionPlan:
    strategy: str
    chunks: list[str]


@dataclass(frozen=True)
class RaptorNode:
    """A deterministic RAPTOR hierarchy node with source-chunk traceability."""

    node_id: str
    level: int
    text: str
    chunk_ids: tuple[int, ...]
    children: tuple[RaptorNode, ...] = field(default_factory=tuple)


# -- helpers -------------------------------------------------------------------


def build_messages(
    source_body: str,
    existing_titles: dict[str, str],
    *,
    source_type: str | None = None,
) -> list[dict]:
    """Build the messages list for the librarian chat completion.

    Args:
        source_type: If ``"structured"``, uses the structured-data prompt
            (exactly ONE topic). Otherwise uses the default librarian prompt
            (3–7 topics).
    """
    prompt = (
        STRUCTURED_SYSTEM_PROMPT
        if source_type == "structured"
        else LIBRARIAN_SYSTEM_PROMPT
    )

    parts = [f"SOURCE:\n\n{source_body}"]
    if existing_titles:
        titles_str = "\n".join(
            f"- {title}" for _slug, title in existing_titles.items()
        )
        parts.append(f"\n\nEXISTING TOPICS:\n{titles_str}")

    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def chunk_for_extraction(
    source_body: str,
    *,
    chunk_tokens: int = MAP_CHUNK_TOKENS,
    overlap_tokens: int = MAP_CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Chunk text into approximate token windows for map-reduce extraction."""
    if not source_body:
        return []
    size = chunk_tokens * CHARS_PER_TOKEN
    overlap = overlap_tokens * CHARS_PER_TOKEN
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")
    chunks: list[str] = []
    start = 0
    while start < len(source_body):
        end = min(start + size, len(source_body))
        chunks.append(source_body[start:end])
        if end == len(source_body):
            break
        start += step
    return chunks


def build_raptor_tree(
    chunk_summaries: list[str],
    *,
    group_size: int = 6,
    context_token_budget: int = DIRECT_TOKEN_LIMIT,
) -> RaptorNode:
    """Build a deterministic RAPTOR-style summary tree.

    The tree clusters related summaries by stable lexical signatures, groups
    them into bounded parent nodes, and preserves every original chunk id in
    each ancestor. This pure function is intentionally LLM-free so hierarchy
    shape and traceability are unit-testable.
    """
    if not chunk_summaries:
        raise ExtractionError("raptor tree requires at least one chunk summary")
    if group_size < 2:
        raise ValueError("group_size must be >= 2")

    nodes = [
        RaptorNode(
            node_id=f"L0-C{idx}",
            level=0,
            text=summary,
            chunk_ids=(idx,),
        )
        for idx, summary in enumerate(chunk_summaries)
    ]
    level = 0
    while len(nodes) > 1 and (
        sum(estimate_tokens(node.text) for node in nodes) > context_token_budget
        or level == 0
    ):
        level += 1
        nodes = _cluster_raptor_nodes(nodes, level=level, group_size=group_size)
    return RaptorNode(
        node_id=f"L{level + 1}-ROOT",
        level=level + 1,
        text="\n\n".join(node.text for node in nodes),
        chunk_ids=tuple(chunk for node in nodes for chunk in node.chunk_ids),
        children=tuple(nodes),
    )


def _cluster_raptor_nodes(
    nodes: list[RaptorNode],
    *,
    level: int,
    group_size: int,
) -> list[RaptorNode]:
    buckets: dict[str, list[RaptorNode]] = {}
    for node in nodes:
        buckets.setdefault(_summary_signature(node.text), []).append(node)

    parents: list[RaptorNode] = []
    for signature in sorted(buckets):
        bucket = buckets[signature]
        bucket.sort(key=lambda node: node.chunk_ids)
        for start in range(0, len(bucket), group_size):
            group = bucket[start:start + group_size]
            chunk_ids = tuple(chunk for child in group for chunk in child.chunk_ids)
            parent_text = _parent_summary_text(signature, group)
            parents.append(
                RaptorNode(
                    node_id=f"L{level}-N{len(parents)}",
                    level=level,
                    text=parent_text,
                    chunk_ids=chunk_ids,
                    children=tuple(group),
                )
            )
    parents.sort(key=lambda node: node.chunk_ids)
    return parents


def _summary_signature(text: str) -> str:
    words = [
        word
        for word in re.findall(r"[a-z0-9]{4,}", text.lower())
        if word not in {"this", "that", "with", "from", "source", "chunk", "summary"}
    ]
    return words[0] if words else "misc"


def _parent_summary_text(signature: str, children: list[RaptorNode]) -> str:
    traces = ", ".join(f"chunk {idx}" for child in children for idx in child.chunk_ids)
    body = "\n".join(f"- {child.text}" for child in children)
    return f"Cluster: {signature}\nTrace: {traces}\n{body}"


def plan_extraction(source_body: str) -> ExtractionPlan:
    """Return direct, map-reduce, or raptor-style extraction plan (§12.2)."""
    token_count = estimate_tokens(source_body)
    if token_count < DIRECT_TOKEN_LIMIT:
        return ExtractionPlan("direct", [source_body])
    chunks = chunk_for_extraction(source_body)
    if token_count < MAP_REDUCE_TOKEN_LIMIT:
        return ExtractionPlan("map_reduce", chunks)
    return ExtractionPlan("raptor", chunks)


def _merge_outputs(outputs: list[LibrarianOutput], *, strategy: str) -> LibrarianOutput:
    """Deterministically reduce chunk-level LibrarianOutput objects."""
    if not outputs:
        raise ExtractionError(f"{strategy} extraction produced no chunk outputs")
    seen: set[tuple[str, str]] = set()
    topics = []
    for out in outputs:
        for topic in out.topics:
            key = (topic.name.lower(), topic.target_slug)
            if key in seen:
                continue
            seen.add(key)
            topics.append(topic)
            if len(topics) >= 7:
                break
        if len(topics) >= 7:
            break
    tldr = " ".join(out.tldr.strip() for out in outputs if out.tldr.strip())
    if not topics:
        raise ExtractionError(f"{strategy} extraction returned zero topics")
    return LibrarianOutput(tldr=tldr[:2000], topics=topics)


def _outputs_to_reduce_markdown(outputs: list[LibrarianOutput], *, label: str) -> str:
    sections = [
        f"{label}\n\n"
        "Reduce these structured chunk summaries into one valid librarian output."
    ]
    for idx, out in enumerate(outputs, 1):
        sections.append(
            "## Mapped Summary "
            f"{idx}\nTrace: chunk {idx - 1}\n"
            f"TLDR: {out.tldr}\n"
            + "\n".join(
                (
                    f"- Topic: {topic.name}\n"
                    f"  Confidence: {topic.confidence}\n"
                    f"  Section: {topic.merged_section}"
                )
                for topic in out.topics
            )
        )
    return "\n\n".join(sections)


def _raptor_final_markdown(root: RaptorNode) -> str:
    return (
        "RAPTOR FINAL REDUCE\n\n"
        "Use the hierarchical summaries below. Preserve claims only when they "
        "are supported by traced chunks.\n\n"
        f"Root trace chunks: {', '.join(str(idx) for idx in root.chunk_ids)}\n\n"
        f"{root.text}"
    )


def schema_for_strict() -> dict:
    """Wrap :class:`LibrarianOutput`'s JSON Schema for OpenRouter strict mode.

    Returns a dict suitable for ``response_format``:
    ``{"type": "json_schema", "json_schema": {"name": …, "strict": True, "schema": …}}``
    """
    schema = LibrarianOutput.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "librarian_output",
            "strict": True,
            "schema": schema,
        },
    }


# -- main entry point ----------------------------------------------------------


async def extract(
    client,
    cfg,
    source_body: str,
    existing_titles: dict[str, str],
    *,
    source_type: str | None = None,
    source_id: str = "",
) -> LibrarianOutput:
    """Extract structured output from a source via the librarian LLM.

    Args:
        source_type: If ``"structured"``, uses the structured-data prompt
            (exactly ONE topic). Otherwise uses the default librarian prompt
            (3–7 topics).
        source_id: Optional caller-supplied source identifier used in
            structured log events and stamped on :class:`ConfidenceFloorError`.
            Defaults to ``""`` when the caller does not supply it.

    Attempts the **primary model** first, then falls back to the **repair
    model** on 5xx / 4xx (other than 402) / parse errors (§12.2: "4xx →
    next model"; 5xx → next model; parse/validation → next model).

    Raises:
        CreditExhaustedError: OpenRouter credit exhausted — stops the daemon
            (§12.3).  Propagates immediately, no repair attempt.
        ExtractionError: both attempts failed (primary + repair).
        ConfidenceFloorError: every returned topic is below
            ``cfg.extraction.confidence_floor`` — pipeline quarantines.
    """
    extra_body = (
        {"plugins": [{"id": "response-healing"}]}
        if cfg.extraction.enable_healing
        else {}
    )
    model = cfg.extraction.primary_model or cfg.models.text
    repair_model = cfg.extraction.repair_model
    floor = cfg.extraction.confidence_floor

    def _emit_end(model_name: str, attempt_n: int, out: LibrarianOutput, t0: float) -> None:
        """Emit ``extract.attempt.end`` plus ``extract.attempt.empty`` when applicable."""
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        end_kw: dict[str, object] = {
            "source_id": source_id,
            "model": model_name,
            "attempt": attempt_n,
            "latency_ms": latency_ms,
            "n_topics": len(out.topics),
        }
        if out.topics:
            confs = [t.confidence for t in out.topics]
            end_kw["confidence_min"] = round(min(confs), 4)
            end_kw["confidence_max"] = round(max(confs), 4)
            log.info("extract.attempt.end", **end_kw)
            return
        log.warning(
            "extract.attempt.empty",
            source_id=source_id,
            model=model_name,
            reason="zero_topics",
        )
        log.info("extract.attempt.end", **end_kw)

    async def _try_body(model_name: str, attempt_n: int, body: str) -> LibrarianOutput:
        log.info(
            "extract.attempt.start",
            source_id=source_id,
            model=model_name,
            source_type=source_type or "text",
            attempt=attempt_n,
        )
        t0 = time.perf_counter()
        messages = build_messages(body, existing_titles, source_type=source_type)
        resp = await client.chat_completion(
            model_name,
            messages,
            response_format=schema_for_strict(),
            extra_body=extra_body,
        )
        content = resp["choices"][0]["message"]["content"]
        # Structured output (json_schema) is self-protecting: a leading
        # <think>...</think> would break JSON parse -> existing repair
        # fallback handles it. No need to strip here.
        out = LibrarianOutput.model_validate_json(content)
        _emit_end(model_name, attempt_n, out, t0)
        # Confidence-floor enforcement — only when at least one topic exists.
        # An empty topics list is a separate (existing) failure mode.
        if out.topics:
            max_conf = max(t.confidence for t in out.topics)
            if max_conf < floor:
                log.warning(
                    "extract.confidence_floor.breached",
                    source_id=source_id,
                    n_topics=len(out.topics),
                    max_confidence=round(max_conf, 4),
                    confidence_floor=floor,
                    model=model_name,
                )
                raise ConfidenceFloorError(
                    source_id=source_id,
                    n_topics=len(out.topics),
                    max_confidence=max_conf,
                    floor=floor,
                )
        return out

    async def _try(model_name: str, attempt_n: int) -> LibrarianOutput:
        plan = plan_extraction(source_body)
        if plan.strategy == "direct":
            return await _try_body(model_name, attempt_n, plan.chunks[0])

        total_chunks = len(plan.chunks)
        log.info(
            "extract.long_source.start",
            source_id=source_id,
            strategy=plan.strategy,
            chunks=total_chunks,
        )
        outputs: list[LibrarianOutput] = []
        for idx, chunk in enumerate(plan.chunks, 1):
            log.info(
                "extract.chunk.start",
                source_id=source_id,
                strategy=plan.strategy,
                chunk=idx,
                total=total_chunks,
            )
            wrapped = (
                f"LONG SOURCE {plan.strategy.upper()} MAP CHUNK {idx}/{total_chunks}\n\n"
                f"{chunk}"
            )
            outputs.append(await _try_body(model_name, attempt_n, wrapped))
            log.info(
                "extract.chunk.done",
                source_id=source_id,
                strategy=plan.strategy,
                chunk=idx,
                total=total_chunks,
                topics_so_far=sum(len(o.topics) for o in outputs),
            )

        if plan.strategy == "map_reduce":
            reduce_body = _outputs_to_reduce_markdown(
                outputs,
                label="MAP-REDUCE REDUCE PASS",
            )
            return await _try_body(model_name, attempt_n, reduce_body)

        raptor_root = build_raptor_tree(
            [_outputs_to_reduce_markdown([out], label="RAPTOR LEAF SUMMARY") for out in outputs]
        )
        return await _try_body(model_name, attempt_n, _raptor_final_markdown(raptor_root))

    primary_error_type = "Unknown"

    # -- Primary attempt ------------------------------------------------
    try:
        return await _try(model, 1)
    except CreditExhaustedError:
        # 402: stops the daemon (§12.3) — never repair.
        raise
    except OpenRouterAPIError as e:
        # §12.2: 4xx (other than 402) → next model; 5xx → next model.
        # 402 is already caught above and re-raised.  The OpenRouter client
        # internally retries 429/500/502/503 with backoff + Retry-After
        # honour, so by the time we see an OpenRouterAPIError the request
        # has already exhausted its in-client retries.
        if e.status < 500:
            log.info(
                "extract.primary.4xx_repair_fallback",
                source_id=source_id,
                primary_model=model,
                repair_model=repair_model,
                status=e.status,
                error_name=e.error_name,
            )
        primary_error_type = type(e).__name__
        # Fall through to repair.
    except (json.JSONDecodeError, ValidationError) as e:
        primary_error_type = type(e).__name__
        # Parse error -> fallback

    # -- Repair fallback -------------------------------------------------
    log.warning(
        "extract.repair_fallback",
        source_id=source_id,
        primary_model=model,
        repair_model=repair_model,
        error_type=primary_error_type,
    )
    try:
        return await _try(repair_model, 2)
    except CreditExhaustedError:
        raise
    except (json.JSONDecodeError, ValidationError, OpenRouterAPIError) as e:
        raise ExtractionError(
            f"Extraction failed after repair fallback: {e}"
        ) from e
