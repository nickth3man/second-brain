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

from second_brain.atomicio import write_atomic
from second_brain.config import Config
from second_brain.openrouter_client import (
    CreditExhaustedError,
    OpenRouterAPIError,
    OpenRouterClient,
)

logger = structlog.get_logger(__name__)

CHUNK_MINUTES = 4
MAX_CHUNK_BYTES = 8_000_000


# -- exceptions --------------------------------------------------------------


class FFprobeError(RuntimeError):
    """Raised when ffprobe fails to probe audio duration."""


# -- helpers ------------------------------------------------------------------


def _probe_duration_seconds(path: Path) -> float:
    """Return the media duration in seconds via ffprobe.

    Runs ``ffprobe -v error -show_entries format=duration
    -of default=noprint_wrappers=1:nokey=1 <path>``.

    Returns:
        Duration in seconds.

    Raises:
        FFprobeError: on any failure (nonzero exit, empty stdout, parse error).
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
            raise FFprobeError(
                f"ffprobe failed (exit {result.returncode}): "
                f"{result.stderr.strip() or 'empty stdout'}"
            )
        return float(result.stdout.strip())
    except FFprobeError:
        raise
    except Exception as exc:
        raise FFprobeError(f"ffprobe exception: {exc}") from exc


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

    Args:
        cfg: App configuration (used for ``brain_root``).
        sha: Full SHA-256 hex digest of the file.
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
        1. SHA-based dedup key (full SHA-256 hex digest).
        2. Probe duration with ffprobe (raises on failure).
        3. Warn (but continue) if duration exceeds ``max_audio_minutes``.
        4. Compute chunk boundaries (``CHUNK_MINUTES`` per chunk).
        5. Load completed-chunk set from progress file (if any).
        6. For each incomplete chunk: extract via ffmpeg, transcribe via
           OpenRouter (with built-in retry), mark done, persist progress.
        7. Concatenate chunk transcripts.  Clean up temp chunk directory.

    Args:
        path: Path to the audio file.
        cfg: App config (used for cache paths, model name, max minutes).
        client: OpenRouter client for STT.

    Returns:
        The full concatenated transcript.

    Raises:
        FFprobeError: if the audio file cannot be probed.
        RuntimeError: if all chunks fail, chunk is oversize, or contract error.
    """
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    cache_root = cfg.brain_root / ".brain" / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    try:
        dur = _probe_duration_seconds(path)
    except FFprobeError as exc:
        logger.warning("audio.ffprobe_failed", error=str(exc))
        raise RuntimeError(f"cannot probe duration: {exc}") from exc

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
    failed_chunks: list[int] = []

    for k in range(n_chunks):
        if k in done:
            continue

        start = k * chunk_sec
        tmp_chunk = chunk_dir / f"chunk_{k}.mp3"

        try:
            _extract_chunk(path, start, chunk_sec, tmp_chunk)
        except Exception as e:
            # Extraction failure (ffmpeg) — treat as chunk failure.
            failed_chunks.append(k)
            logger.warning("audio.chunk_extract_failed", chunk=k, error=str(e))
            chunks.append("")
            done.add(k)
            write_atomic(prog_path, json.dumps(sorted(done)))
            continue

        # Size check — raise OUTSIDE the main except so oversize propagates.
        if tmp_chunk.stat().st_size > MAX_CHUNK_BYTES:
            raise RuntimeError(
                f"chunk {k} exceeds max size: "
                f"{tmp_chunk.stat().st_size} > {MAX_CHUNK_BYTES}"
            )

        try:
            text = await client.transcribe(cfg.models.stt, tmp_chunk)
            chunks.append(text)
            done.add(k)
            write_atomic(prog_path, json.dumps(sorted(done)))
            logger.info("audio.chunk_done", chunk=k, chars=len(text))
        except OpenRouterAPIError as e:
            if e.status == 400 and k == 0:
                raise RuntimeError(f"STT contract error (400): {e}") from e
            failed_chunks.append(k)
            logger.warning(
                "audio.chunk_failed",
                chunk=k,
                status=e.status,
                error_name=e.error_name,
            )
            chunks.append("")
            done.add(k)
            write_atomic(prog_path, json.dumps(sorted(done)))
        except CreditExhaustedError:
            raise
        except Exception as e:
            failed_chunks.append(k)
            logger.warning(
                "audio.chunk_failed",
                chunk=k,
                error_type=type(e).__name__,
            )
            chunks.append("")
            done.add(k)
            write_atomic(prog_path, json.dumps(sorted(done)))

    # Best-effort cleanup of temp chunk directory
    try:
        import shutil

        shutil.rmtree(chunk_dir, ignore_errors=True)
    except Exception:
        pass

    n_failed = len(failed_chunks)
    if n_failed == n_chunks:
        raise RuntimeError(f"all {n_chunks} STT chunks failed")

    sentinel = ""
    if n_failed:
        sentinel = f"\n<!-- sb:partial {n_failed}/{n_chunks} chunks failed -->"

    return "\n".join(c for c in chunks if c) + sentinel
