"""WebVTT transcript normalization — strip timing tags and deduplicate cues."""

from __future__ import annotations

import re

# Inline timing tags: <00:00:00.640>, <c>, </c>, <c.colorXXX>, </c>
_TIMING_TAG_RE = re.compile(r"<\d{2}:\d{2}:\d{2}\.\d{3}>|</?c(?:\.\w+)?>")
# Cue header line: "HH:MM:SS.mmm --> HH:MM:SS.mmm ..."
_CUE_HEADER_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->")
# VTT file header metadata
_HEADER_RE = re.compile(r"^(WEBVTT|Kind:|Language:)")


def vtt_to_transcript(text: str) -> str:
    """Convert raw WebVTT text to a clean deduplicated transcript.

    Strips all WebVTT formatting: cue timestamps, inline word-timing tags,
    color markup, and file header metadata. Deduplicates the rolling-subtitle
    echo lines that YouTube's auto-captions produce (each sentence appears
    2-3x in consecutive cues as the caption window scrolls).

    Returns plain flowing prose joined with spaces.
    """
    lines = text.splitlines()
    transcript_lines: list[str] = []
    prev = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADER_RE.match(stripped):
            continue
        if _CUE_HEADER_RE.match(stripped):
            continue
        clean = _TIMING_TAG_RE.sub("", stripped).strip()
        if not clean:
            continue
        if clean == prev:
            continue
        transcript_lines.append(clean)
        prev = clean

    return " ".join(transcript_lines)
