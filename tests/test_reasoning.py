"""Unit tests for ``second_brain.reasoning`` — strip_think & ThinkSplitter.

Covers all edge cases specified in the oracle, including streaming
tag-boundary crossing and the cardinal identity invariant for
non-think content.
"""

from __future__ import annotations

from second_brain.reasoning import ThinkSplitter, strip_think

# ---------------------------------------------------------------------------
# strip_think — one-shot, non-streaming
# ---------------------------------------------------------------------------


class TestStripThink:
    """Identity for non-think content; correct extraction for think content."""

    def test_no_think_tag_identity(self) -> None:
        """No <think> → identity: (None, text) unchanged."""
        r, c = strip_think("Hello, world.")
        assert r is None
        assert c == "Hello, world."

    def test_well_formed_think(self) -> None:
        """Well-formed <think>r</think>answer → ("r", "answer")."""
        r, c = strip_think("<think>r</think>answer")
        assert r == "r"
        assert c == "answer"

    def test_whitespace_after_block_stripped(self) -> None:
        """Leading whitespace/newlines after the block are stripped."""
        r, c = strip_think("<think>deep</think>\n\n  Here is the answer.")
        assert r == "deep"
        assert c == "Here is the answer."

    def test_no_content_after_think(self) -> None:
        """<think>...</think> with nothing after → content empty."""
        r, c = strip_think("<think>reasoning</think>")
        assert r == "reasoning"
        assert c == ""

    def test_think_with_content_before(self) -> None:
        """Content before <think> is preserved."""
        r, c = strip_think("Before.<think>r</think>After.")
        assert r == "r"
        assert c == "Before.After."

    def test_multi_paragraph_think(self) -> None:
        """Multi-line think block with DOTALL."""
        r, c = strip_think("<think>line1\nline2</think>out")
        assert r == "line1\nline2"
        assert c == "out"

    def test_unclosed_think_at_offset_zero(self) -> None:
        """Unclosed <think> at offset 0 → all reasoning, content empty."""
        r, c = strip_think("<think>this is all reasoning")
        assert r == "this is all reasoning"
        assert c == ""

    def test_unclosed_think_mid_text_unchanged(self) -> None:
        """Unclosed <think> mid-text → unchanged (reasoning None)."""
        r, c = strip_think("Some text <think>unclosed")
        assert r is None
        assert c == "Some text <think>unclosed"

    def test_multiple_think_blocks(self) -> None:
        """Only the first <think>...</think> block is extracted."""
        r, c = strip_think("<think>first</think>mid<think>second</think>end")
        assert r == "first"
        assert c == "mid<think>second</think>end"

    def test_no_think_at_all(self) -> None:
        """Completely plain text unchanged."""
        r, c = strip_think("Just plain text with no tags.")
        assert r is None
        assert c == "Just plain text with no tags."

    def test_empty_string(self) -> None:
        """Empty string → (None, '')."""
        r, c = strip_think("")
        assert r is None
        assert c == ""


# ---------------------------------------------------------------------------
# ThinkSplitter — streaming
# ---------------------------------------------------------------------------


def _collect(splitter: ThinkSplitter, chunks: list[str]) -> tuple[str, str]:
    """Feed chunks through a splitter and collect reasoning + content."""
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    for chunk in chunks:
        for r, c in splitter.feed(chunk):
            if r is not None:
                reasoning_parts.append(r)
            if c is not None:
                content_parts.append(c)
    r_rem, c_rem = splitter.flush()
    if r_rem:
        reasoning_parts.append(r_rem)
    if c_rem:
        content_parts.append(c_rem)
    return ("".join(reasoning_parts), "".join(content_parts))


class TestThinkSplitter:
    """Streaming splitter across chunk boundaries."""

    def test_no_think_identity(self) -> None:
        """No think at all → content passes through verbatim."""
        r, c = _collect(ThinkSplitter(), ["Hello", ", ", "world."])
        assert r == ""
        assert c == "Hello, world."

    def test_think_whole_chunk(self) -> None:
        """<think> in a single chunk."""
        r, c = _collect(ThinkSplitter(), ["<think>r</think>answer"])
        assert r == "r"
        assert c == "answer"

    def test_tag_split_across_chunks(self) -> None:
        """<think> tag spanning 4 chunks."""
        r, c = _collect(
            ThinkSplitter(),
            ["<thi", "nk>r", "</t", "hink>answer"],
        )
        assert r == "r"
        assert c == "answer"

    def test_think_split_across_two_chunks(self) -> None:
        """Open tag split at boundary."""
        r, c = _collect(ThinkSplitter(), ["<think", ">r</think>a"])
        assert r == "r"
        assert c == "a"

    def test_close_split_across_two_chunks(self) -> None:
        """Close tag split at boundary."""
        r, c = _collect(ThinkSplitter(), ["<think>r</thi", "nk>a"])
        assert r == "r"
        assert c == "a"

    def test_content_before_and_after_think(self) -> None:
        """Content before <think> is emitted as content."""
        r, c = _collect(
            ThinkSplitter(),
            ["Before.<think>r</think>After."],
        )
        assert r == "r"
        assert c == "Before.After."

    def test_flush_inside_think(self) -> None:
        """Stream ends inside <think>: all buffered text becomes reasoning.

        The splitter emits reasoning incrementally during feed(); flush() only
        holds a final remainder (empty when text was already streamed).
        """
        splitter = ThinkSplitter()
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for chunk in ["<think>still", " thinking"]:
            for r, c in splitter.feed(chunk):
                if r:
                    reasoning_parts.append(r)
                if c:
                    content_parts.append(c)
        r_rem, c_rem = splitter.flush()
        if r_rem:
            reasoning_parts.append(r_rem)
        if c_rem:
            content_parts.append(c_rem)
        assert "".join(reasoning_parts) == "still thinking"
        assert "".join(content_parts) == ""

    def test_multi_chunk_content_no_think(self) -> None:
        """Multiple chunks with no think at all."""
        r, c = _collect(ThinkSplitter(), ["abc", "def", "ghi"])
        assert r == ""
        assert c == "abcdefghi"

    def test_empty_chunks(self) -> None:
        """Empty chunks do not produce output."""
        r, c = _collect(ThinkSplitter(), ["", "", ""])
        assert r == ""
        assert c == ""

    def test_only_open_tag(self) -> None:
        """Only opening <think> (no close) → all reasoning."""
        r, c = _collect(ThinkSplitter(), ["<think>just reasoning"])
        assert r == "just reasoning"
        assert c == ""

    def test_think_with_content_after_flush(self) -> None:
        """Well-formed think followed by flush is fine."""
        r, c = _collect(ThinkSplitter(), ["<think>r</think>content"])
        assert r == "r"
        assert c == "content"

    def test_partial_open_tag_at_end_of_chunk(self) -> None:
        """Partial <think at chunk end held back, completed next chunk."""
        r, c = _collect(ThinkSplitter(), ["text <thi", "nk>r</think>end"])
        assert r == "r"
        assert c == "text end"

    def test_reasoning_delta_yielded_as_it_arrives(self) -> None:
        """Reasoning yielded before content completes (streaming)."""
        splitter = ThinkSplitter()
        # Simulate a model that streams thinking then answer
        results: list[tuple[str | None, str | None]] = []
        for chunk in ["<think>step", " by step</think>", "Final answer."]:
            for pair in splitter.feed(chunk):
                results.append(pair)
        r_rem, c_rem = splitter.flush()
        if r_rem:
            results.append((r_rem, None))
        if c_rem:
            results.append((None, c_rem))

        reasoning_str = "".join(r for r, _ in results if r is not None)
        content_str = "".join(c for _, c in results if c is not None)
        assert reasoning_str == "step by step"
        assert content_str == "Final answer."
