"""Image vision parser (§6).

Reads an image file and sends it to an OpenRouter vision model for description
and OCR.  No explicit resize — vision models downscale internally to ~2048 px
longest edge (§6).
"""

from __future__ import annotations

from pathlib import Path

from second_brain.config import Config
from second_brain.openrouter_client import OpenRouterClient

# Suffix-to-MIME mapping.  Anything not in this table defaults to ``image/png``.
_MIME_MAP: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
}


async def parse_image(path: Path, cfg: Config, client: OpenRouterClient) -> str:
    """Describe an image and transcribe any visible text via a vision model.

    Args:
        path: Path to the image file.
        cfg: App configuration (``models.vision``).
        client: OpenRouter client for vision API calls.

    Returns:
        The model's description / OCR transcription, or an error note wrapped
        in brackets on failure.
    """
    mime = _MIME_MAP.get(path.suffix.lstrip(".").lower(), "image/png")
    data = path.read_bytes()
    try:
        return await client.vision_describe(
            cfg.models.vision,
            [data],
            "Describe this image and transcribe any visible text (OCR). "
            "Return a faithful textual representation.",
            mime=mime,
        )
    except Exception as e:
        return f"[image parse failed: {e}]"
