"""Video parser — audio-based STT with deferred vision (§6).

Extracts the audio track via ffmpeg and delegates to :func:`parse_audio`
for chunked STT transcription.  Keyframe extraction + vision description
is intentionally deferred (the transcript is the primary signal for
knowledge ingestion).
"""

from __future__ import annotations

import contextlib
import hashlib
import subprocess
from pathlib import Path

import structlog

from second_brain.config import Config
from second_brain.openrouter_client import OpenRouterClient
from second_brain.parse.audio import parse_audio

logger = structlog.get_logger(__name__)


async def parse_video(
    path: Path,
    cfg: Config,
    client: OpenRouterClient,
) -> str:
    """Transcribe a video file by extracting audio and delegating to STT.

    Workflow:
        1. Compute a SHA-based key for the temp audio file.
        2. Extract the audio track to ``.brain/cache/<sha>_audio.mp3`` via
           ffmpeg (re-encode to MP3 for Whisper compatibility).
        3. Delegate to :func:`parse_audio` for chunked STT.
        4. Best-effort cleanup of the temp MP3.

    Keyframe → vision description is **deferred** (see §6).

    Args:
        path: Path to the video file.
        cfg: App config.
        client: OpenRouter client.

    Returns:
        The transcribed text (or an error message on extraction failure).
    """
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    cache = cfg.brain_root / ".brain" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tmp_mp3 = cache / f"{sha}_audio.mp3"

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-acodec",
                "libmp3lame",
                "-q:a",
                "4",
                str(tmp_mp3),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return (
                f"[video audio extraction failed: ffmpeg exit "
                f"{result.returncode}]"
            )
    except Exception as e:
        return f"[video audio extraction failed: {e}]"

    transcript = await parse_audio(tmp_mp3, cfg, client)

    # Best-effort cleanup
    with contextlib.suppress(Exception):
        tmp_mp3.unlink(missing_ok=True)

    return transcript
