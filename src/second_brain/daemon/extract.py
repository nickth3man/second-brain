"""Structured extraction via LLM — Librarian output with JSON schema (§12.2).

Defines the system prompt, message builder, strict-JSON-schema wrapper, and
the extract function with a single repair fallback.

Phase 1 does **not** implement map-reduce or RAPTOR — one stuffed call + one
repair call is the slice.
"""

from __future__ import annotations

import json

import httpx
from pydantic import ValidationError

from second_brain.daemon.normalize import estimate_tokens
from second_brain.models import LibrarianOutput
from second_brain.openrouter_client import CreditExhaustedError

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
) -> LibrarianOutput:
    """Extract structured output from a source via the librarian LLM.

    Args:
        source_type: If ``"structured"``, uses the structured-data prompt
            (exactly ONE topic). Otherwise uses the default librarian prompt
            (3–7 topics).

    Attempts the **primary model** first, then falls back to the **repair
    model** on 5xx / parse errors.

    Raises:
        CreditExhaustedError: OpenRouter credit exhausted — stops the daemon.
        ExtractionError: both attempts failed.
        httpx.HTTPStatusError: a 4xx status (other than 402) — immediate
            abort, no fallback.
    """
    extra_body = (
        {"plugins": [{"id": "response-healing"}]}
        if cfg.extraction.enable_healing
        else {}
    )
    model = cfg.extraction.primary_model or cfg.models.text
    repair_model = cfg.extraction.repair_model

    async def _try(model_name: str) -> LibrarianOutput:
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
        return LibrarianOutput.model_validate_json(content)

    # -- Primary attempt ------------------------------------------------
    try:
        return await _try(model)
    except CreditExhaustedError:
        raise
    except httpx.HTTPStatusError as e:
        if e.response.status_code < 500:  # 4xx (402 already caught above)
            raise  # immediate abort, no fallback
        # 5xx -> fall through to repair
    except (json.JSONDecodeError, ValidationError):
        pass  # Parse error -> fallback

    # -- Repair fallback -------------------------------------------------
    try:
        return await _try(repair_model)
    except CreditExhaustedError:
        raise
    except (json.JSONDecodeError, ValidationError, httpx.HTTPStatusError) as e:
        raise ExtractionError(
            f"Extraction failed after repair fallback: {e}"
        ) from e
