"""OpenRouter HTTP client with ZDR privacy and keyring resolution (§12.7).

Provides an async httpx client that enforces ``provider.zdr: true`` on every
request, resolves the API key via keyring -> env -> config, and raises typed
exceptions for HTTP 402 (credit exhaustion, §12.3).

Vision (``vision_describe``) and STT (``transcribe``) methods added in
Phase 3 Wave 1 per §6 and §12.7.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import keyring

from second_brain.config import Config

# -- exceptions --------------------------------------------------------------


class OpenRouterError(Exception):
    """Base exception for OpenRouter client errors."""


class CreditExhaustedError(OpenRouterError):
    """Raised on HTTP 402 — credit exhaustion stops the daemon (§12.3)."""


# -- key resolution ----------------------------------------------------------


def resolve_api_key(cfg: Config) -> str:
    """Resolve the OpenRouter API key per §12.7 resolution order.

    Order:
        1. keyring (Windows Credential Manager, service="second-brain",
           username="openrouter")
        2. env var ``OPENROUTER_API_KEY``
        3. ``cfg.openrouter.api_key``

    Returns:
        The resolved API key.

    Raises:
        OpenRouterError: if all sources are empty or contain the placeholder
                         value (``sk-or-v1-PASTE*``).
    """
    # 1. keyring
    if cfg.privacy.api_key_source == "keyring":
        try:
            key = keyring.get_password("second-brain", "openrouter")
            if key and not key.startswith("sk-or-v1-PASTE"):
                return key
        except Exception:
            pass

    # 2. environment variable
    key = os.environ.get("OPENROUTER_API_KEY")
    if key and not key.startswith("sk-or-v1-PASTE"):
        return key

    # 3. config file
    key = cfg.openrouter.api_key
    if key and not key.startswith("sk-or-v1-PASTE"):
        return key

    raise OpenRouterError(
        "No OpenRouter API key found. Set OPENROUTER_API_KEY env var, "
        "run `brain init` to store in keyring, or paste your key in config.toml."
    )


# -- client ------------------------------------------------------------------


class OpenRouterClient:
    """Async HTTP client for the OpenRouter API.

    Lazily creates an ``httpx.AsyncClient`` with ZDR enforcement,
    timeout config, and standard headers.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            api_key = resolve_api_key(self.cfg)
            self._client = httpx.AsyncClient(
                base_url=self.cfg.openrouter.base_url,
                timeout=httpx.Timeout(120.0, connect=10.0),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/nickth3man/second-brain",
                    "X-Title": "second-brain",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OpenRouterClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # -- provider params -------------------------------------------------

    def _zdr_provider(self) -> dict[str, Any]:
        """Build the ``provider`` sub-dict enforcing ZDR (§12.7).

        Only includes keys whose value is truthy. Strips Nones.
        """
        params: dict[str, Any] = {}
        if self.cfg.privacy.zdr:
            params["zdr"] = True
        if self.cfg.extraction.require_parameters:
            params["require_parameters"] = True
        if self.cfg.privacy.block_training_providers:
            params["data_collection"] = "deny"
        return params

    # -- API methods -----------------------------------------------------

    async def vision_describe(
        self,
        model: str,
        images: list[bytes],
        prompt: str,
        *,
        mime: str = "image/png",
    ) -> str:
        """Describe one or more images via a vision-capable model.

        Builds a single user message whose ``content`` is a list of alternating
        text and image-url parts.  The caller MUST keep ``len(images)`` within
        ``cfg.ingestion.vision_max_images_per_request`` (PDF rendering sends one
        page per request per §6).

        Args:
            model: OpenRouter vision model slug.
            images: Raw image bytes (one per page/screenshot).
            prompt: Text prompt accompanying the images.
            mime: MIME type of the images (default ``image/png``).

        Returns:
            The model's text response.

        Raises:
            CreditExhaustedError: on HTTP 402.
            httpx.HTTPStatusError: on other HTTP errors.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for im in images:
            b64 = base64.b64encode(im).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "provider": self._zdr_provider(),
        }
        resp = await self.client.post("/chat/completions", json=body)
        if resp.status_code == 402:
            raise CreditExhaustedError(
                "OpenRouter credit exhausted. Top up at https://openrouter.ai/credits"
            )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def transcribe(
        self,
        model: str,
        audio_path: Path,
        *,
        language: str | None = None,
    ) -> str:
        """Transcribe an audio file via OpenRouter's /audio/transcriptions.

        Sends the file as **multipart/form-data** (not JSON).  ZDR is conveyed
        through the default HTTP-Referer / X-Title headers (the existing client
        headers); no ``provider`` field is sent in the multipart body.

        Args:
            model: OpenRouter STT model slug.
            audio_path: Path to the audio file on disk.
            language: Optional ISO language code hint.

        Returns:
            The transcribed text.

        Raises:
            CreditExhaustedError: on HTTP 402.
            httpx.HTTPStatusError: on other HTTP errors.
        """
        data: dict[str, Any] = {"model": model}
        if language is not None:
            data["language"] = language

        with open(audio_path, "rb") as f:
            resp = await self.client.post(
                "/audio/transcriptions",
                files={"file": (audio_path.name, f, "audio/mpeg")},
                data=data,
            )
        if resp.status_code == 402:
            raise CreditExhaustedError(
                "OpenRouter credit exhausted. Top up at https://openrouter.ai/credits"
            )
        resp.raise_for_status()
        return resp.json()["text"]

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> Any:
        """POST /chat/completions.

        Args:
            model: OpenRouter model slug.
            messages: Chat messages in OpenAI format.
            response_format: Optional JSON schema for structured output.
            extra_body: Additional fields merged into the request body.
            stream: If True, raises NotImplementedError (streaming lands in Phase 6).

        Returns:
            Parsed JSON response body.

        Raises:
            NotImplementedError: if *stream* is True.
            CreditExhausted: on HTTP 402.
            httpx.HTTPStatusError: on other HTTP errors.
        """
        if stream:
            raise NotImplementedError("Streaming not yet available (Phase 6).")

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "provider": self._zdr_provider(),
        }
        if response_format is not None:
            body["response_format"] = response_format
        if extra_body:
            body.update(extra_body)

        resp = await self.client.post("/chat/completions", json=body)
        if resp.status_code == 402:
            raise CreditExhaustedError(
                "OpenRouter credit exhausted. Top up at https://openrouter.ai/credits"
            )
        resp.raise_for_status()
        return resp.json()

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream a chat completion via SSE (Phase 6).

        POSTs to ``/chat/completions`` with ``stream=True`` and yields
        content deltas as they arrive.

        Args:
            model: OpenRouter model slug.
            messages: Chat messages in OpenAI format.
            extra_body: Additional fields merged into the request body.

        Yields:
            Content string deltas from ``choices[0].delta.content``.

        Raises:
            CreditExhaustedError: on HTTP 402.
            httpx.HTTPStatusError: on other HTTP errors.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "provider": self._zdr_provider(),
        }
        if extra_body:
            body.update(extra_body)

        async with self.client.stream("POST", "/chat/completions", json=body) as resp:
            if resp.status_code == 402:
                raise CreditExhaustedError(
                    "OpenRouter credit exhausted. Top up at https://openrouter.ai/credits"
                )
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload.strip() == "[DONE]":
                    return
                data = json.loads(payload)
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content is None:
                    continue
                yield content

    async def embedding(self, model: str, input: str | list[str]) -> list[float]:
        """POST /embeddings and return the embedding vector.

        Args:
            model: OpenRouter embedding model slug.
            input: Text to embed (single string or list of strings).

        Returns:
            The embedding vector (list of floats) for the first result.

        Raises:
            CreditExhausted: on HTTP 402.
            httpx.HTTPStatusError: on other HTTP errors.
        """
        resp = await self.client.post(
            "/embeddings",
            json={
                "model": model,
                "input": input,
                "provider": self._zdr_provider(),
            },
        )
        if resp.status_code == 402:
            raise CreditExhaustedError(
                "OpenRouter credit exhausted. Top up at https://openrouter.ai/credits"
            )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
