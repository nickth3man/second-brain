"""Audio parser — ffmpeg-chunked STT with resumability (§6, §12.7).

Splits long audio into fixed-duration chunks, transcribes each via
OpenRouter's /audio/transcriptions (Whisper-family), and persists
per-chunk progress to a JSON checkpoint file for crash recovery.

Chunk-boundary artifacts (clipped words at seams) are an accepted
trade-off per Architecture Decision 3a (§12.7).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import structlog

from second_brain.config import Config
from second_brain.openrouter_client import OpenRouterClient

logger = structlog.get_logger(__name__)

CHUNK_MINUTES = 15


# -- helpers ------------------------------------------------------------------


def _probe_duration_seconds(path: Path) -> float:
    """Return the media duration in seconds via ffprobe.

    Runs ``ffprobe -v error -show_entries format=duration
    -of default=noprint_wrappers=1:nokey=1 <path>``.

    Returns:
        Duration in seconds, or ``0.0`` on any failure.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0.0
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _extract_chunk(
    path: Path,
    start: float,
    duration: float,
    out_path: Path,
) -> None:
    """Extract a segment of audio via ffmpeg (fast seek, re-encode to MP3).

    Command::

        ffmpeg -y -ss <start> -i <path> -t <duration>
               -vn -acodec libmp3lame -q:a 4 <out_path>

    ``-ss`` before ``-i`` enables fast seeking.  Re-encoding to MP3 ensures
    compatibility with Whisper-family models.

    Raises:
        RuntimeError: if ffmpeg returns a non-zero exit code.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(path),
            "-t",
            str(duration),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "4",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg chunk extraction failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )


def _progress_path(cfg: Config, sha: str) -> Path:
    """Return the path to the per-file progress checkpoint.

    The checkpoint is a JSON list of completed chunk indices stored at::

        <brain_root>/.brain/cache/<sha>.audio_progress.json
    """
    return cfg.brain_root / ".brain" / "cache" / f"{sha}.audio_progress.json"


# -- main entry point ---------------------------------------------------------


async def parse_audio(
    path: Path,
    cfg: Config,
    client: OpenRouterClient,
) -> str:
    """Transcribe an audio file using chunked STT with resumability.

    Workflow:
        1. SHA-based dedup key (first 16 hex chars).
        2. Probe duration with ffprobe.
        3. Warn (but continue) if duration exceeds ``max_audio_minutes``.
        4. Compute chunk boundaries (``CHUNK_MINUTES`` per chunk).
        5. Load completed-chunk set from progress file (if any).
        6. For each incomplete chunk: extract via ffmpeg, transcribe via
           OpenRouter, mark done, persist progress.
        7. Concatenate chunk transcripts.  Clean up temp chunk directory.

    Args:
        path: Path to the audio file.
        cfg: App config (used for cache paths, model name, max minutes).
        client: OpenRouter client for STT.

    Returns:
        The full concatenated transcript.
    """
    sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    cache_root = cfg.brain_root / ".brain" / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    dur = _probe_duration_seconds(path)
    dur_minutes = dur / 60.0
    if dur_minutes > cfg.ingestion.max_audio_minutes:
        logger.warning(
            "audio_duration_exceeds_max",
            duration_minutes=round(dur_minutes, 1),
            max_minutes=cfg.ingestion.max_audio_minutes,
        )

    chunk_sec = CHUNK_MINUTES * 60
    n_chunks = max(1, math.ceil(dur / chunk_sec)) if dur > 0 else 1

    # Load progress checkpoint
    prog_path = _progress_path(cfg, sha)
    done: set[int] = set()
    if prog_path.is_file():
        try:
            done = set(json.loads(prog_path.read_text()))
        except Exception:
            done = set()

    # Temp chunk directory
    chunk_dir = cache_root / f"{sha}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[str] = []

    for k in range(n_chunks):
        if k in done:
            continue

        start = k * chunk_sec
        tmp_chunk = chunk_dir / f"chunk_{k}.mp3"

        try:
            _extract_chunk(path, start, chunk_sec, tmp_chunk)
            text = await client.transcribe(cfg.models.stt, tmp_chunk)
            chunks.append(text)
            done.add(k)
            prog_path.write_text(json.dumps(sorted(done)))
        except Exception as e:
            chunks.append(f"\n[chunk {k + 1} transcribe failed: {e}]\n")
            done.add(k)
            prog_path.write_text(json.dumps(sorted(done)))

    # Best-effort cleanup of temp chunk directory
    try:
        import shutil

        shutil.rmtree(chunk_dir, ignore_errors=True)
    except Exception:
        pass

    return "\n".join(chunks)
