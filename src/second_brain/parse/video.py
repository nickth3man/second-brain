"""Video parser — audio-based STT with optional keyframe vision (§6).

Extracts the audio track via ffmpeg and delegates to :func:`parse_audio`
for chunked STT transcription. Optional bounded keyframe extraction augments
the transcript with visual observations from the configured vision model.
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
    """Transcribe a video file and optionally add bounded keyframe vision.

    Workflow:
        1. Compute a SHA-based key for the temp audio file.
        2. Extract the audio track to ``.brain/cache/<sha>_audio.mp3`` via
           ffmpeg (re-encode to MP3 for Whisper compatibility).
        3. Delegate to :func:`parse_audio` for chunked STT.
        4. Extract representative keyframes if enabled and describe them with
           the configured OpenRouter vision model.
        5. Best-effort cleanup of temporary media files.

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
    observations = []
    if getattr(cfg.ingestion, "video_keyframe_vision", True):
        frames = _extract_keyframes(path, cfg, sha)
        observations = await _describe_keyframes(frames, cfg, client)

    # Best-effort cleanup
    with contextlib.suppress(Exception):
        tmp_mp3.unlink(missing_ok=True)
    for frame in frames if "frames" in locals() else []:
        with contextlib.suppress(Exception):
            frame.unlink(missing_ok=True)

    return _merge_video_markdown(transcript, observations)


def _extract_keyframes(path: Path, cfg: Config, sha: str) -> list[Path]:
    """Extract bounded representative keyframes with ffmpeg."""
    max_frames = max(0, int(getattr(cfg.ingestion, "video_keyframe_max_frames", 8)))
    if max_frames == 0:
        return []
    cadence = max(1, int(getattr(cfg.ingestion, "video_keyframe_cadence_seconds", 120)))
    cache = cfg.brain_root / ".brain" / "cache" / f"{sha}_keyframes"
    cache.mkdir(parents=True, exist_ok=True)
    pattern = cache / "frame_%03d.png"
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vf",
                f"fps=1/{cadence},scale='min(1280,iw)':-2",
                "-frames:v",
                str(max_frames),
                str(pattern),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("video.keyframes_failed", returncode=result.returncode)
            return []
    except Exception as exc:
        logger.warning("video.keyframes_failed", error=str(exc))
        return []
    return sorted(cache.glob("frame_*.png"))[:max_frames]


async def _describe_keyframes(
    frames: list[Path],
    cfg: Config,
    client: OpenRouterClient,
) -> list[str]:
    """Describe frames one at a time to stay under provider image limits."""
    observations: list[str] = []
    for idx, frame in enumerate(frames, 1):
        try:
            text = await client.vision_describe(
                cfg.models.vision,
                [frame.read_bytes()],
                (
                    "Describe visible information in this video keyframe for "
                    f"knowledge ingestion. Frame {idx} of {len(frames)}. "
                    "Mention on-screen text, diagrams, people, objects, and "
                    "scene changes. Do not infer unsupported facts."
                ),
            )
        except Exception as exc:
            logger.warning("video.keyframe_vision_failed", frame=str(frame), error=str(exc))
            continue
        observations.append(f"Frame {idx}: {text.strip()}")
    return observations


def _merge_video_markdown(transcript: str, observations: list[str]) -> str:
    if not observations:
        return transcript
    lines = ["## Transcript", transcript.strip()]
    if observations:
        lines.extend(["", "## Visual Observations", *[f"- {obs}" for obs in observations]])
    return "\n\n".join(part for part in lines if part)
