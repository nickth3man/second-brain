"""Phase 3 Wave 2b tests — STT parsers (audio + video) (§6, §12.7).

No real ffmpeg or API calls — subprocess calls are monkeypatched.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess as _sp
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# -- stubs --------------------------------------------------------------------


@dataclass
class _FakeModels:
    stt: str = "test-stt-model"


@dataclass
class _FakeIngestion:
    max_audio_minutes: int = 120


@dataclass
class _FakeCfg:
    brain_root: Path
    models: _FakeModels = field(default_factory=_FakeModels)
    ingestion: _FakeIngestion = field(default_factory=_FakeIngestion)


class FakeSTTClient:
    """Stub OpenRouter client that returns predictable transcripts.

    Attributes:
        call_count: Incremented on every ``transcribe`` call.
        fail_on_call: If set, the client raises ``RuntimeError`` on the
            n-th call (1-indexed).
    """

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.call_count = 0
        self.fail_on_call = fail_on_call

    async def transcribe(
        self,
        model: str,
        audio_path: Path,
        *,
        language: str | None = None,
        audio_format: str | None = None,
    ) -> str:
        self.call_count += 1
        if self.fail_on_call is not None and self.call_count == self.fail_on_call:
            raise RuntimeError("STT API error")
        return f"transcript:{audio_path.name}"


# -- helpers ------------------------------------------------------------------


def _fake_extract_chunk(
    path: Path,
    start: float,
    duration: float,
    out_path: Path,
) -> None:
    """Write a dummy audio file instead of running ffmpeg."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"dummy audio")


def _fake_probe_1200(_path: Path) -> float:
    """Return 1200 seconds (20 min → 2 chunks of 15 min)."""
    return 1200.0


# -- Audio tests --------------------------------------------------------------


class TestAudioChunking:
    """parse_audio chunking, resumability, and failure tolerance."""

    async def test_chunks_and_concatenates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from second_brain.parse.audio import CHUNK_MINUTES, parse_audio

        monkeypatch.setattr(
            "second_brain.parse.audio._probe_duration_seconds",
            _fake_probe_1200,
        )
        monkeypatch.setattr(
            "second_brain.parse.audio._extract_chunk",
            _fake_extract_chunk,
        )

        # 1200s with CHUNK_MINUTES=4 -> ceil(1200/(4*60)) = ceil(5) = 5 chunks
        expected_chunks = max(1, math.ceil(1200 / (CHUNK_MINUTES * 60)))

        client = FakeSTTClient()
        cfg = _FakeCfg(brain_root=tmp_path)
        p = tmp_path / "test.mp3"
        p.write_bytes(b"audio content")
        sha = hashlib.sha256(b"audio content").hexdigest()[:16]

        result = await parse_audio(p, cfg, client)

        assert client.call_count == expected_chunks
        for k in range(expected_chunks):
            assert f"transcript:chunk_{k}.mp3" in result

        prog_path = tmp_path / ".brain" / "cache" / f"{sha}.audio_progress.json"
        assert prog_path.is_file()
        assert json.loads(prog_path.read_text()) == list(range(expected_chunks))

    async def test_resume_skips_done_chunks(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from second_brain.parse.audio import CHUNK_MINUTES, parse_audio

        monkeypatch.setattr(
            "second_brain.parse.audio._probe_duration_seconds",
            _fake_probe_1200,
        )
        monkeypatch.setattr(
            "second_brain.parse.audio._extract_chunk",
            _fake_extract_chunk,
        )

        expected_chunks = max(1, math.ceil(1200 / (CHUNK_MINUTES * 60)))

        client = FakeSTTClient()
        cfg = _FakeCfg(brain_root=tmp_path)
        p = tmp_path / "test.mp3"
        p.write_bytes(b"resumable audio")
        sha = hashlib.sha256(b"resumable audio").hexdigest()[:16]

        # Pre-write progress indicating chunk 0 is done
        prog_path = tmp_path / ".brain" / "cache" / f"{sha}.audio_progress.json"
        prog_path.parent.mkdir(parents=True, exist_ok=True)
        prog_path.write_text("[0]")

        result = await parse_audio(p, cfg, client)

        assert client.call_count == expected_chunks - 1
        assert "transcript:chunk_0.mp3" not in result
        assert json.loads(prog_path.read_text()) == list(range(expected_chunks))

    async def test_chunk_failure_appends_partial_sentinel(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from second_brain.parse.audio import CHUNK_MINUTES, parse_audio

        monkeypatch.setattr(
            "second_brain.parse.audio._probe_duration_seconds",
            _fake_probe_1200,
        )
        monkeypatch.setattr(
            "second_brain.parse.audio._extract_chunk",
            _fake_extract_chunk,
        )

        expected_chunks = max(1, math.ceil(1200 / (CHUNK_MINUTES * 60)))

        # Fail on the second transcribe call; partial success produces sentinel.
        # With fail_on_call=2: chunk 0 succeeds, chunk 1 fails, chunks 2+ succeed.
        client = FakeSTTClient(fail_on_call=2)
        cfg = _FakeCfg(brain_root=tmp_path)
        p = tmp_path / "test.mp3"
        p.write_bytes(b"faulty audio")
        sha = hashlib.sha256(b"faulty audio").hexdigest()[:16]

        result = await parse_audio(p, cfg, client)

        # First chunk succeeded
        assert "transcript:chunk_0.mp3" in result
        # Partial sentinel appended
        assert "sb:partial" in result
        # Only one chunk (the second) should have failed
        assert "sb:partial 1" in result
        # Old-style error note should NOT be present
        assert "transcribe failed" not in result
        assert client.call_count == expected_chunks
        # All chunks marked done in progress
        prog_path = tmp_path / ".brain" / "cache" / f"{sha}.audio_progress.json"
        assert json.loads(prog_path.read_text()) == list(range(expected_chunks))


# -- Video tests --------------------------------------------------------------


class TestVideoDelegation:
    """parse_video extracts audio and delegates to parse_audio."""

    async def test_delegates_to_audio_parser(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from second_brain.parse.video import parse_video

        content = b"fake video content"
        p = tmp_path / "test.mp4"
        p.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()[:16]

        cfg = _FakeCfg(brain_root=tmp_path)

        # Monkeypatch ffmpeg subprocess to create a dummy MP3 instead
        def _fake_ffmpeg_run(
            args: list[str],
            *args_: object,
            **kwargs: object,
        ) -> _sp.CompletedProcess:
            out_path = Path(args[-1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"dummy mp3")
            return _sp.CompletedProcess(args, returncode=0)

        monkeypatch.setattr(
            "second_brain.parse.video.subprocess.run",
            _fake_ffmpeg_run,
        )

        async def _fake_parse_audio(
            _path: Path,
            _cfg: object,
            _client: object,
        ) -> str:
            return "fake video transcript"

        monkeypatch.setattr(
            "second_brain.parse.video.parse_audio",
            _fake_parse_audio,
        )

        client = object()  # not used — parse_audio is monkeypatched
        result = await parse_video(p, cfg, client)

        assert result == "fake video transcript"

        # Temp audio file should have been cleaned up
        expected_tmp = tmp_path / ".brain" / "cache" / f"{sha}_audio.mp3"
        assert not expected_tmp.exists()
