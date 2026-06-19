"""Client-side stripping of ``<think>...</think>`` from LLM content.

Minimax ``minimax-m3`` (and potentially other reasoning models) leak
chain-of-thought into the ``content`` field as ``<think>...</think>`` markers.
Request-side params (``reasoning_split``, ``reasoning``) do NOT reliably
remove the leak, so the only robust fix is client-side stripping.

These helpers are IDENTITY for any content with no ``<think>`` tag —
zero behavior change for non-reasoning models.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Pattern: non-greedy match of the first <think>...</think> block.
_THINK_RE = re.compile(
    re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE),
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# One-shot (non-streaming) helper
# ---------------------------------------------------------------------------


def strip_think(text: str) -> tuple[str | None, str]:
    """Separate any ``<think>...</think>`` block from *text*.

    Args:
        text: Raw content that may contain a ``<think>`` block.

    Returns:
        ``(reasoning, clean_content)`` where *reasoning* is ``None`` when
        no think block is found.
    """
    if THINK_OPEN not in text:
        return (None, text)

    m = _THINK_RE.search(text)
    if m is not None:
        # Well-formed <think>...</think> block.
        # Strip leading whitespace/newlines left by the removed block.
        reasoning = m.group(1)
        clean = text[: m.start()] + text[m.end() :].lstrip()
        return (reasoning, clean)

    # Unclosed <think> — no </think> found.
    if text.startswith(THINK_OPEN):
        # Entire remainder is reasoning.
        return (text[len(THINK_OPEN) :], "")
    # Mid-response <think> with no close → leave unchanged.
    return (None, text)


# ---------------------------------------------------------------------------
# Streaming splitter
# ---------------------------------------------------------------------------


def _partial_prefix(text: str, tag: str) -> int:
    """Return length of longest prefix of *tag* that *text* ends with.

    Excludes a complete match (the caller already checked for that).
    Returns 0 when no partial prefix is found.
    """
    max_len = min(len(text), len(tag) - 1)
    for i in range(max_len, 0, -1):
        if text.endswith(tag[:i]):
            return i
    return 0


class ThinkSplitter:
    """Stateful stream splitter that separates ``<think>`` blocks from content.

    Handles tags that span chunk boundaries by holding back a tail buffer
    of up to 8 characters (the length of ``</think>``).

    Example::

        splitter = ThinkSplitter()
        for chunk in raw_stream:
            for reasoning_piece, content_piece in splitter.feed(chunk):
                ...
        reasoning, content = splitter.flush()
    """

    def __init__(self) -> None:
        self.inside_think: bool = False
        self.tail_buffer: str = ""

    def feed(self, chunk: str) -> Iterator[tuple[str | None, str | None]]:
        """Feed a chunk of text and yield ``(reasoning_piece, content_piece)``.

        At most one of the two tuple elements is non-``None`` per yield
        (the reasoning and content streams are separated).
        """
        combined = self.tail_buffer + chunk
        self.tail_buffer = ""

        while combined:
            if self.inside_think:
                idx = combined.find(THINK_CLOSE)
                if idx == -1:
                    # No close tag yet; check for partial close at end.
                    hold = _partial_prefix(combined, THINK_CLOSE)
                    if hold:
                        self.tail_buffer = combined[-hold:]
                        combined = combined[:-hold]
                    if combined:
                        yield (combined, None)
                    break
                # Found </think> — emit reasoning before it, then exit.
                if idx > 0:
                    yield (combined[:idx], None)
                combined = combined[idx + len(THINK_CLOSE) :]
                self.inside_think = False
                # Continue processing the remainder.
            else:
                idx = combined.find(THINK_OPEN)
                if idx == -1:
                    # No open tag; check for partial open at end.
                    hold = _partial_prefix(combined, THINK_OPEN)
                    if hold:
                        self.tail_buffer = combined[-hold:]
                        combined = combined[:-hold]
                    if combined:
                        yield (None, combined)
                    break
                # Found <think> — emit content before it, then enter.
                if idx > 0:
                    yield (None, combined[:idx])
                combined = combined[idx + len(THINK_OPEN) :]
                self.inside_think = True
                # Continue processing the remainder.

    def flush(self) -> tuple[str | None, str]:
        """Finalise the stream.

        Returns:
            ``(reasoning, content)`` with any remaining buffered text.
        """
        remainder = self.tail_buffer
        self.tail_buffer = ""
        if self.inside_think:
            return (remainder, "")
        return (None, remainder)
