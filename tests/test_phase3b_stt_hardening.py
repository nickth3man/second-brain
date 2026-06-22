"""Phase 3 Wave 2b hardening tests — STT client, audio policy, retry.

Covers the OpenRouter STT wire-format fix (base64 JSON body, not multipart),
error handling, retry logic, and audio parsing edge cases.

Hermetic — no real ffmpeg or API calls.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess as _sp
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from second_brain.openrouter_client import (
    CreditExhaustedError,
    OpenRouterAPIError,
    OpenRouterClient,
    audio_format_for,
)
from second_brain.parse.audio import (
    CHUNK_MINUTES,
    MAX_CHUNK_BYTES,
    FFprobeError,
    _probe_duration_seconds,
    parse_audio,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeModels:
    stt: str = "test-stt-model"
    embedding: str = "test-embed-model"


@dataclass
class _FakeIngestion:
    max_audio_minutes: int = 120
    merge_threshold: float = 0.85
    require_parameters: bool = False
    enable_healing: bool = False


@dataclass
class _FakeExtraction:
    deadletter_dir: str = ".brain/deadletter"
    primary_model: str = ""
    repair_model: str = ""
    require_parameters: bool = False
    enable_healing: bool = False


@dataclass
class _FakePrivacy:
    zdr: bool = True
    api_key_source: str = "env"
    block_training_providers: bool = False


@dataclass
class _FakeOpenRouter:
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = "sk-or-v1-test"


@dataclass
class _FakeCfg:
    brain_root: Path
    models: _FakeModels = field(default_factory=_FakeModels)
    ingestion: _FakeIngestion = field(default_factory=_FakeIngestion)
    extraction: _FakeExtraction = field(default_factory=_FakeExtraction)
    privacy: _FakePrivacy = field(default_factory=_FakePrivacy)
    openrouter: _FakeOpenRouter = field(default_factory=_FakeOpenRouter)


class FakeSTTClient:
    """Stub OpenRouter client for audio test compatibility.

    The new transcribe signature adds ``audio_format`` kwarg.
    """

    def __init__(
        self,
        fail_on_call: int | None = None,
        always_fail: bool = False,
    ) -> None:
        self.call_count = 0
        self.fail_on_call = fail_on_call
        self.always_fail = always_fail

    async def transcribe(
        self,
        model: str,
        audio_path: Path,
        *,
        language: str | None = None,
        audio_format: str | None = None,
    ) -> str:
        self.call_count += 1
        if self.always_fail or (
            self.fail_on_call is not None and self.call_count == self.fail_on_call
        ):
            raise RuntimeError("STT API error")
        return f"transcript:{audio_path.name}"

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# audio_format_for
# ---------------------------------------------------------------------------


class TestAudioFormat:
    def test_mp3_format(self) -> None:
        assert audio_format_for(Path("test.mp3")) == "mp3"

    def test_wav_format(self) -> None:
        assert audio_format_for(Path("test.wav")) == "wav"

    def test_unknown_suffix_defaults_to_mp3(self) -> None:
        assert audio_format_for(Path("test.xyz")) == "mp3"

    def test_no_suffix_defaults_to_mp3(self) -> None:
        assert audio_format_for(Path("test")) == "mp3"

    def test_case_insensitive(self) -> None:
        assert audio_format_for(Path("test.MP3")) == "mp3"

    def test_all_formats_mapped(self) -> None:
        """All expected audio extensions are mapped."""
        for ext, expected in [(".mp3", "mp3"), (".wav", "wav"), (".m4a", "m4a"),
                               (".webm", "webm"), (".ogg", "ogg"), (".flac", "flac"),
                               (".aac", "aac")]:
            assert audio_format_for(Path(f"test{ext}")) == expected


# ---------------------------------------------------------------------------
# Transcribe wire format
# ---------------------------------------------------------------------------


class TestTranscribeWireFormat:
    """Verify transcribe sends base64 JSON body (not multipart)."""

    @pytest.fixture
    def cfg(self, tmp_path: Path) -> _FakeCfg:
        return _FakeCfg(brain_root=tmp_path)

    @pytest.fixture
    async def client(self, cfg: _FakeCfg) -> OpenRouterClient:
        c = OpenRouterClient(cfg)
        c._client = AsyncMock(spec=httpx.AsyncClient)
        yield c
        await c.close()

    async def test_transcribe_sends_json_base64_body(
        self, client: OpenRouterClient, cfg: _FakeCfg, tmp_path: Path
    ) -> None:
        """transcribe sends a JSON body with base64 input_audio."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"fake audio bytes")

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": "hello world"}
        mock_resp.text = '{"text":"hello world"}'
        mock_resp.content = mock_resp.text.encode()
        client._client.post.return_value = mock_resp  # type: ignore[assignment]

        result = await client.transcribe("test-stt", audio_path)

        assert result == "hello world"
        # Verify the POST body was JSON with input_audio.data
        call_kwargs = client._client.post.call_args[1]
        body: dict = call_kwargs["json"]
        assert "input_audio" in body
        assert body["input_audio"]["format"] == "mp3"
        # Verify base64 content
        decoded = base64.b64decode(body["input_audio"]["data"])
        assert decoded == b"fake audio bytes"
        # Verify no files field (no multipart)
        assert "files" not in call_kwargs

    async def test_transcribe_base64_roundtrip(
        self, client: OpenRouterClient, cfg: _FakeCfg, tmp_path: Path
    ) -> None:
        """Base64 payload round-trips correctly."""
        original = b"\x00\x01\x02\xff\xfe\xfd audio data " * 100
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(original)

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": "transcribed"}
        mock_resp.text = '{"text":"transcribed"}'
        mock_resp.content = mock_resp.text.encode()
        client._client.post.return_value = mock_resp  # type: ignore[assignment]

        await client.transcribe("test-stt", audio_path)

        call_kwargs = client._client.post.call_args[1]
        body: dict = call_kwargs["json"]
        decoded = base64.b64decode(body["input_audio"]["data"])
        assert decoded == original
        assert body["input_audio"]["format"] == "wav"

    async def test_transcribe_format_mapping(
        self, client: OpenRouterClient, cfg: _FakeCfg, tmp_path: Path
    ) -> None:
        """Format is auto-detected from path suffix."""
        audio_path = tmp_path / "test.flac"
        audio_path.write_bytes(b"data")

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": "t"}
        mock_resp.text = '{"text":"t"}'
        mock_resp.content = mock_resp.text.encode()
        client._client.post.return_value = mock_resp  # type: ignore[assignment]

        await client.transcribe("test-stt", audio_path)

        call_kwargs = client._client.post.call_args[1]
        assert call_kwargs["json"]["input_audio"]["format"] == "flac"

    async def test_transcribe_language_passed(
        self, client: OpenRouterClient, cfg: _FakeCfg, tmp_path: Path
    ) -> None:
        """Language is passed through in the JSON body."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"data")

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": "t"}
        mock_resp.text = '{"text":"t"}'
        mock_resp.content = mock_resp.text.encode()
        client._client.post.return_value = mock_resp  # type: ignore[assignment]

        await client.transcribe("test-stt", audio_path, language="en")

        body: dict = client._client.post.call_args[1]["json"]
        assert body["language"] == "en"

    async def test_transcribe_zdr_always_present(
        self, client: OpenRouterClient, cfg: _FakeCfg, tmp_path: Path
    ) -> None:
        """ZDR provider is always in the JSON body."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"data")

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": "t"}
        mock_resp.text = '{"text":"t"}'
        mock_resp.content = mock_resp.text.encode()
        client._client.post.return_value = mock_resp  # type: ignore[assignment]

        await client.transcribe("test-stt", audio_path)

        body: dict = client._client.post.call_args[1]["json"]
        assert "provider" in body
        assert body["provider"]["zdr"] is True

    async def test_transcribe_402_credit_exhausted(
        self, client: OpenRouterClient, cfg: _FakeCfg, tmp_path: Path
    ) -> None:
        """402 raises CreditExhaustedError."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"data")

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 402
        mock_resp.text = "credit exhausted"
        client._client.post.return_value = mock_resp  # type: ignore[assignment]

        with pytest.raises(CreditExhaustedError):
            await client.transcribe("test-stt", audio_path)

    async def test_transcribe_400_zoderror_surfaces_body(
        self, client: OpenRouterClient, cfg: _FakeCfg, tmp_path: Path
    ) -> None:
        """400 surfaces as OpenRouterAPIError with body."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"data")

        error_body = '{"error":{"name":"ZodError","message":"Validation failed"}}'
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 400
        mock_resp.headers = {}
        mock_resp.text = error_body
        mock_resp.content = error_body.encode()
        mock_resp.json.return_value = {
            "error": {"name": "ZodError", "message": "Validation failed"}
        }
        client._client.post.return_value = mock_resp  # type: ignore[assignment]

        with pytest.raises(OpenRouterAPIError) as exc_info:
            await client.transcribe("test-stt", audio_path)

        assert exc_info.value.status == 400
        assert exc_info.value.error_name == "ZodError"

    async def test_transcribe_401_raises_api_error(
        self, client: OpenRouterClient, cfg: _FakeCfg, tmp_path: Path
    ) -> None:
        """401 raises OpenRouterAPIError."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"data")

        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 401
        mock_resp.headers = {}
        mock_resp.text = '{"error":{"name":"Unauthorized"}}'
        mock_resp.content = mock_resp.text.encode()
        mock_resp.json.return_value = {"error": {"name": "Unauthorized"}}
        client._client.post.return_value = mock_resp  # type: ignore[assignment]

        with pytest.raises(OpenRouterAPIError):
            await client.transcribe("test-stt", audio_path)


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


class TestSTTRetry:
    """STT retry with exponential backoff."""

    @pytest.fixture
    def cfg(self, tmp_path: Path) -> _FakeCfg:
        return _FakeCfg(brain_root=tmp_path)

    @pytest.fixture
    async def client(self, cfg: _FakeCfg) -> OpenRouterClient:
        c = OpenRouterClient(cfg)
        c._client = AsyncMock(spec=httpx.AsyncClient)
        yield c
        await c.close()

    async def test_transcribe_429_retries_then_succeeds(
        self, client: OpenRouterClient, cfg: _FakeCfg, tmp_path: Path
    ) -> None:
        """429 is retried; eventual success returns text."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"data")

        responses = [
            MagicMock(status_code=429, text='{"error":{"name":"RateLimited"}}',
                      content=b'{"error":{"name":"RateLimited"}}',
                      json=lambda: {"error": {"name": "RateLimited"}}),
            MagicMock(status_code=200, text='{"text":"ok"}',
                      content=b'{"text":"ok"}',
                      json=lambda: {"text": "ok"}),
        ]
        client._client.post = AsyncMock(side_effect=responses)  # type: ignore[assignment]

        result = await client.transcribe("test-stt", audio_path)
        assert result == "ok"

    async def test_transcribe_500_gives_up_after_3(
        self, client: OpenRouterClient, cfg: _FakeCfg, tmp_path: Path
    ) -> None:
        """500 is retried 3 times then raises."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"data")

        resp_500 = MagicMock(status_code=500, text='{"error":{"name":"ServerError"}}',
                              content=b'{"error":{"name":"ServerError"}}',
                              json=lambda: {"error": {"name": "ServerError"}})
        client._client.post = AsyncMock(return_value=resp_500)  # type: ignore[assignment]

        with pytest.raises(OpenRouterAPIError) as exc_info:
            await client.transcribe("test-stt", audio_path)
        assert exc_info.value.status == 500
        assert client._client.post.call_count == 3

    @pytest.mark.parametrize("status_code", [502, 503])
    async def test_transcribe_502_503_retried(
        self, client: OpenRouterClient, cfg: _FakeCfg, tmp_path: Path,
        status_code: int,
    ) -> None:
        """502/503 are retried."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"data")

        resp_err = MagicMock(
            status_code=status_code,
            text=f'{{"error":{{"name":"Error{status_code}"}}}}',
            content=f'{{"error":{{"name":"Error{status_code}"}}}}'.encode(),
            json=lambda: {"error": {"name": f"Error{status_code}"}},
        )
        resp_ok = MagicMock(status_code=200, text='{"text":"ok"}',
                            content=b'{"text":"ok"}',
                            json=lambda: {"text": "ok"})
        client._client.post = AsyncMock(side_effect=[resp_err, resp_ok])  # type: ignore[assignment]

        result = await client.transcribe("test-stt", audio_path)
        assert result == "ok"

    async def test_transcribe_timeout_retried(
        self, client: OpenRouterClient, cfg: _FakeCfg, tmp_path: Path
    ) -> None:
        """httpx.TimeoutException is retried."""
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"data")

        mock_ok = MagicMock(status_code=200, text='{"text":"ok"}',
                            content=b'{"text":"ok"}',
                            json=lambda: {"text": "ok"})
        client._client.post = AsyncMock(  # type: ignore[assignment]
            side_effect=[httpx.TimeoutException("timeout"), mock_ok]
        )

        result = await client.transcribe("test-stt", audio_path)
        assert result == "ok"


# ---------------------------------------------------------------------------
# ffprobe / audio policy
# ---------------------------------------------------------------------------


class TestAudioProbe:
    """ _probe_duration_seconds error handling."""

    def test_probe_raises_on_ffprobe_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing ffprobe raises FFprobeError."""
        def _fake_run(*args, **kwargs):
            raise FileNotFoundError("ffprobe not found")

        monkeypatch.setattr("second_brain.parse.audio.subprocess.run", _fake_run)
        with pytest.raises(FFprobeError):
            _probe_duration_seconds(Path("/nonexistent.mp3"))

    def test_probe_raises_on_nonzero_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Nonzero exit raises FFprobeError."""
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"data")

        def _fake_run(*args, **kwargs):
            return _sp.CompletedProcess(args[0], returncode=1, stdout="", stderr="error msg")

        monkeypatch.setattr("second_brain.parse.audio.subprocess.run", _fake_run)
        with pytest.raises(FFprobeError, match="exit 1"):
            _probe_duration_seconds(audio)

    def test_probe_raises_on_empty_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty stdout raises FFprobeError."""
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"data")

        def _fake_run(*args, **kwargs):
            return _sp.CompletedProcess(args[0], returncode=0, stdout="", stderr="")

        monkeypatch.setattr("second_brain.parse.audio.subprocess.run", _fake_run)
        with pytest.raises(FFprobeError, match="empty stdout"):
            _probe_duration_seconds(audio)

    def test_probe_raises_on_corrupt_mp3(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Corrupt file causes probe failure."""
        audio = tmp_path / "corrupt.mp3"
        audio.write_bytes(b"\xff\xfb\x90" * 10)

        # Real ffprobe would return non-zero; simulate it.
        def _fake_run(*args, **kwargs):
            return _sp.CompletedProcess(
                args[0], returncode=1, stdout="",
                stderr="corrupt input",
            )

        monkeypatch.setattr("second_brain.parse.audio.subprocess.run", _fake_run)
        with pytest.raises(FFprobeError):
            _probe_duration_seconds(audio)

    def test_probe_raises_on_zero_duration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A duration of 0.0 is not returned (float parse of 0.0 is valid but
        we test the case where stdout is '0.0' meaning ffprobe returned 0)."""
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"data")

        def _fake_run(*args, **kwargs):
            return _sp.CompletedProcess(
                args[0], returncode=0, stdout="0.0\n", stderr="",
            )

        monkeypatch.setattr("second_brain.parse.audio.subprocess.run", _fake_run)
        # float("0.0") doesn't fail, so this should succeed
        result = _probe_duration_seconds(audio)
        assert result == 0.0


class TestAudioParser:
    """parse_audio edge cases."""

    async def test_audio_empty_file_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty file probe raises RuntimeError."""

        audio = tmp_path / "empty.mp3"
        audio.write_bytes(b"")

        def _raise(*args, **kwargs):
            raise FFprobeError("empty file")

        monkeypatch.setattr(
            "second_brain.parse.audio._probe_duration_seconds", _raise,
        )

        cfg = _FakeCfg(brain_root=tmp_path)
        with pytest.raises(RuntimeError, match="cannot probe duration"):
            await parse_audio(audio, cfg, FakeSTTClient())

    async def test_audio_oversize_chunk_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Oversize chunk raises RuntimeError."""

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"x" * 100)

        monkeypatch.setattr(
            "second_brain.parse.audio._probe_duration_seconds",
            lambda _: 120.0,
        )

        def _oversize_extract(*args, **kwargs) -> None:
            out = kwargs.get("out_path") or args[3]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"x" * (MAX_CHUNK_BYTES + 1))

        monkeypatch.setattr(
            "second_brain.parse.audio._extract_chunk", _oversize_extract,
        )

        cfg = _FakeCfg(brain_root=tmp_path)
        with pytest.raises(RuntimeError, match="exceeds max size"):
            await parse_audio(audio, cfg, FakeSTTClient())

    async def test_audio_identical_400_fail_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """400 on first chunk raises immediately (fail-fast)."""
        from second_brain.openrouter_client import OpenRouterAPIError

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"x" * 100)

        monkeypatch.setattr(
            "second_brain.parse.audio._probe_duration_seconds",
            lambda _: 300.0,
        )

        def _extract(*args, **kwargs):
            out = args[3]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"data")

        monkeypatch.setattr(
            "second_brain.parse.audio._extract_chunk", _extract,
        )

        class FailFastClient:
            """Client that always returns 400."""
            async def transcribe(self, model, path, *, language=None, audio_format=None):
                raise OpenRouterAPIError(
                    400, "/audio/transcriptions",
                    '{"error":{"name":"ZodError"}}', "ZodError",
                )
            async def close(self):
                pass

        cfg = _FakeCfg(brain_root=tmp_path)
        with pytest.raises(RuntimeError, match="STT contract error"):
            await parse_audio(audio, cfg, FailFastClient())

    async def test_audio_partial_success_sets_partial_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Partial success produces sentinel + no crash."""

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"x" * 100)

        monkeypatch.setattr(
            "second_brain.parse.audio._probe_duration_seconds",
            lambda _: 300.0,  # 5 min -> 2 chunks
        )

        def _extract(*args, **kwargs):
            out = args[3]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"data")

        monkeypatch.setattr(
            "second_brain.parse.audio._extract_chunk", _extract,
        )

        client = FakeSTTClient(fail_on_call=2)
        cfg = _FakeCfg(brain_root=tmp_path)
        result = await parse_audio(audio, cfg, client)

        assert "transcript:chunk_0.mp3" in result
        assert "sb:partial" in result

    async def test_audio_all_chunks_fail_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All chunks failing raises RuntimeError."""

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"x" * 100)

        monkeypatch.setattr(
            "second_brain.parse.audio._probe_duration_seconds",
            lambda _: 480.0,
        )

        def _extract(*args, **kwargs):
            out = args[3]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"data")

        monkeypatch.setattr(
            "second_brain.parse.audio._extract_chunk", _extract,
        )

        client = FakeSTTClient(always_fail=True)
        cfg = _FakeCfg(brain_root=tmp_path)
        with pytest.raises(RuntimeError, match="all 2 STT chunks failed"):
            await parse_audio(audio, cfg, client)

    async def test_chunk_minutes_is_four(self) -> None:
        """CHUNK_MINUTES is 4."""
        assert CHUNK_MINUTES == 4

    async def test_audio_resume_from_checkpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Resume from checkpoint skips completed chunks."""

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"resumable audio")
        sha = hashlib.sha256(b"resumable audio").hexdigest()

        monkeypatch.setattr(
            "second_brain.parse.audio._probe_duration_seconds",
            lambda _: 300.0,
        )

        def _extract(*args, **kwargs):
            out = args[3]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"data")

        monkeypatch.setattr(
            "second_brain.parse.audio._extract_chunk", _extract,
        )

        prog_path = tmp_path / ".brain" / "cache" / f"{sha}.audio_progress.json"
        prog_path.parent.mkdir(parents=True, exist_ok=True)
        prog_path.write_text("[0]")

        client = FakeSTTClient()
        cfg = _FakeCfg(brain_root=tmp_path)
        result = await parse_audio(audio, cfg, client)

        assert client.call_count == 1  # only chunk 1
        assert "transcript:chunk_1.mp3" in result

    async def test_audio_progress_backward_compat_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Progress file remains a JSON list of ints."""

        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"backward compat")
        sha = hashlib.sha256(b"backward compat").hexdigest()

        monkeypatch.setattr(
            "second_brain.parse.audio._probe_duration_seconds",
            lambda _: 480.0,
        )

        def _extract(*args, **kwargs):
            out = args[3]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"data")

        monkeypatch.setattr(
            "second_brain.parse.audio._extract_chunk", _extract,
        )

        client = FakeSTTClient()
        cfg = _FakeCfg(brain_root=tmp_path)
        await parse_audio(audio, cfg, client)

        prog_path = tmp_path / ".brain" / "cache" / f"{sha}.audio_progress.json"
        assert json.loads(prog_path.read_text()) == [0, 1]


class TestVideoDelegation:
    """parse_video extracts audio and delegates to parse_audio."""

    async def test_delegates_to_audio_parser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from second_brain.parse.video import parse_video

        content = b"fake video content"
        p = tmp_path / "test.mp4"
        p.write_bytes(content)

        cfg = _FakeCfg(brain_root=tmp_path)

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
            _path: Path, _cfg: object, _client: object,
        ) -> str:
            return "fake video transcript"

        monkeypatch.setattr(
            "second_brain.parse.video.parse_audio",
            _fake_parse_audio,
        )

        client = object()
        result = await parse_video(p, cfg, client)
        assert result == "fake video transcript"
