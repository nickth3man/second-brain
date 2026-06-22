"""Structured extraction via LLM — Librarian output with JSON schema (§12.2).

Defines the system prompt, message builder, strict-JSON-schema wrapper, and
the extract function with a single repair fallback.

Phase 1 does **not** implement map-reduce or RAPTOR — one stuffed call + one
repair call is the slice.
"""

from __future__ import annotations

import json
import time

import structlog
from pydantic import ValidationError

from second_brain.daemon.normalize import estimate_tokens
from second_brain.models import LibrarianOutput
from second_brain.openrouter_client import (
    CreditExhaustedError,
    OpenRouterAPIError,
)

log = structlog.get_logger(__name__)

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


# -- helpers -------------------------------------------------------------------


def build_messages(
    source_body: str,
    existing_titles: dict[str, str],
    *,
    source_type: str | None = None,
) -> list[dict]:
    """Build the messages list for the librarian chat completion.

    Truncates *source_body* if its estimated token count exceeds 16 000
    (§12.2: map-reduce deferred in Phase 1).

    Args:
        source_type: If ``"structured"``, uses the structured-data prompt
            (exactly ONE topic). Otherwise uses the default librarian prompt
            (3–7 topics).
    """
    if estimate_tokens(source_body) > 16000:
        source_body = (
            source_body[:64000]
            + "\n\n[truncated for extraction — map-reduce deferred §12.2]"
        )

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

    async def _try(model_name: str, attempt_n: int) -> LibrarianOutput:
        log.info(
            "extract.attempt.start",
            source_id=source_id,
            model=model_name,
            source_type=source_type or "text",
            attempt=attempt_n,
        )
        t0 = time.perf_counter()
        messages = build_messages(source_body, existing_titles, source_type=source_type)
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
