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
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import keyring
import structlog
import tenacity

from second_brain.config import Config
from second_brain.reasoning import ThinkSplitter, strip_think

log = structlog.get_logger(__name__)

# -- exceptions --------------------------------------------------------------


class OpenRouterError(Exception):
    """Base exception for OpenRouter client errors."""


class CreditExhaustedError(OpenRouterError):
    """Raised on HTTP 402 — credit exhaustion stops the daemon (§12.3)."""


class OpenRouterAPIError(OpenRouterError):
    """Raised on non-402 HTTP errors with structured context.

    Attributes:
        status: HTTP status code.
        endpoint: The API endpoint path (e.g. ``"/chat/completions"``).
        body: The raw response text (truncated to 2000 chars).
        error_name: Parsed ``error.name`` from the response body, if any.
    """

    def __init__(
        self,
        status: int,
        endpoint: str,
        body: str,
        error_name: str | None = None,
    ) -> None:
        self.status = status
        self.endpoint = endpoint
        self.body = body[:2000]
        self.error_name = error_name
        preview = body[:500]
        msg = f"OpenRouter API error {status} on {endpoint}"
        if error_name:
            msg += f" ({error_name})"
        msg += f": {preview}"
        super().__init__(msg)


# -- STT retry constants -----------------------------------------------------

_RETRYABLE_STATUSES: set[int] = {429, 500, 502, 503}
MAX_STT_RETRIES = 3

# -- audio format mapping ---------------------------------------------------


_AUDIO_FORMAT_MAP: dict[str, str] = {
    ".mp3": "mp3",
    ".wav": "wav",
    ".m4a": "m4a",
    ".webm": "webm",
    ".ogg": "ogg",
    ".flac": "flac",
    ".aac": "aac",
}


def audio_format_for(path: Path) -> str:
    """Return the audio container string for *path* based on its suffix.

    Defaults to ``"mp3"`` for unknown extensions.
    """
    return _AUDIO_FORMAT_MAP.get(path.suffix.lower(), "mp3")


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

    # -- shared boundary --------------------------------------------------

    async def _post(
        self,
        endpoint: str,
        body: dict[str, Any],
        *,
        model: str,
        req_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST to *endpoint* with structured logging and error wrapping.

        All non-streaming API methods route through this boundary so that
        request/response/error observability is consistent.

        Args:
            endpoint: API path (e.g. ``"/chat/completions"``).
            body: JSON-serialisable request body (includes ``provider``).
            model: Model slug for logging context.
            req_meta: Optional metadata dict for the log line (NEVER include
                payload content, only counts/sizes).

        Returns:
            Parsed JSON response dict.

        Raises:
            CreditExhaustedError: on HTTP 402.
            OpenRouterAPIError: on other >=400 responses.
        """
        meta = req_meta or {}
        log.info(
            "openrouter.request",
            endpoint=endpoint,
            model=model,
            zdr=bool(self.cfg.privacy.zdr),
            **meta,
        )

        t0 = time.perf_counter()
        resp = await self.client.post(endpoint, json=body)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        if resp.status_code == 402:
            log.error(
                "openrouter.error",
                endpoint=endpoint,
                model=model,
                status=402,
                latency_ms=latency_ms,
                error_name="CreditExhausted",
            )
            raise CreditExhaustedError(
                "OpenRouter credit exhausted. Top up at https://openrouter.ai/credits"
            )

        if resp.status_code >= 400:
            error_name: str | None = None
            try:
                err_data = resp.json()
                error_name = err_data.get("error", {}).get("name")
            except Exception:
                pass
            log.error(
                "openrouter.error",
                endpoint=endpoint,
                model=model,
                status=resp.status_code,
                latency_ms=latency_ms,
                error_name=error_name,
                error_message=(
                    resp.json().get("error", {}).get("message", "")
                    if error_name
                    else ""
                ),
                error_body_preview=resp.text[:500],
            )
            raise OpenRouterAPIError(
                status=resp.status_code,
                endpoint=endpoint,
                body=resp.text,
                error_name=error_name,
            )

        log.info(
            "openrouter.response",
            endpoint=endpoint,
            model=model,
            status=resp.status_code,
            latency_ms=latency_ms,
            resp_bytes=len(resp.content),
            text_preview=(resp.text or "")[:200],
        )
        return resp.json()

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
            The model's text response (``<think>``-stripped).

        Raises:
            CreditExhaustedError: on HTTP 402.
            OpenRouterAPIError: on other HTTP errors.
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
        resp = await self._post(
            "/chat/completions",
            body,
            model=model,
            req_meta={"n_images": len(images)},
        )
        raw = resp["choices"][0]["message"]["content"]
        # Strip <think>...</think> — reasoning models leak CoT into content.
        _, clean_content = strip_think(raw)
        return clean_content

    async def transcribe(
        self,
        model: str,
        audio_path: Path,
        *,
        language: str | None = None,
        audio_format: str | None = None,
    ) -> str:
        """Transcribe an audio file via OpenRouter's /audio/transcriptions.

        Sends the audio as **base64 inline JSON** (not multipart) — OpenRouter
        rejects multipart with ``ZodError``.  ZDR is enforced via the
        ``provider`` sub-dict in the JSON body.

        Retries on ``httpx.TimeoutException``, ``httpx.ConnectError``, or
        ``OpenRouterAPIError`` with retryable status (429/500/502/503).
        Non-retryable statuses (400/401/402/404) raise immediately.
        402 remains :class:`CreditExhaustedError`.

        Args:
            model: OpenRouter STT model slug.
            audio_path: Path to the audio file on disk.
            language: Optional ISO language code hint.
            audio_format: Audio container format (e.g. ``"mp3"``, ``"wav"``).
                Auto-detected from *audio_path* suffix if not given.

        Returns:
            The transcribed text.

        Raises:
            CreditExhaustedError: on HTTP 402 (non-retryable).
            OpenRouterAPIError: on non-retryable HTTP errors.
        """
        raw = audio_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")  # NO data: URI prefix
        fmt = audio_format or audio_format_for(audio_path)

        body: dict[str, Any] = {
            "model": model,
            "input_audio": {"data": b64, "format": fmt},
            "provider": self._zdr_provider(),
        }
        if language is not None:
            body["language"] = language

        def _is_retryable(exc: Exception) -> bool:
            if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
                return True
            return (
                isinstance(exc, OpenRouterAPIError)
                and exc.status in _RETRYABLE_STATUSES
            )

        async for attempt in tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(MAX_STT_RETRIES),
            wait=tenacity.wait_exponential_jitter(initial=1, max=10),
            retry=tenacity.retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                resp = await self._post(
                    "/audio/transcriptions",
                    body,
                    model=model,
                    req_meta={"audio_bytes": len(raw), "format": fmt},
                )
                return resp["text"]

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
            OpenRouterAPIError: on other HTTP errors.
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

        return await self._post(
            "/chat/completions",
            body,
            model=model,
            req_meta={"input_chars": sum(len(m.get("content", "")) for m in messages)},
        )

    async def chat_completion_clean(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str | None, str]:
        """Like :meth:`chat_completion` but strips ``<think>`` from content.

        Returns:
            ``(reasoning, clean_content)`` — the ``<think>`` block (if any)
            is separated as *reasoning* and the remainder as *clean_content*.
        """
        resp = await self.chat_completion(
            model,
            messages,
            response_format=response_format,
            extra_body=extra_body,
        )
        raw = resp["choices"][0]["message"]["content"]
        return strip_think(raw)

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        extra_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a chat completion via SSE (Phase 6), separating reasoning.

        POSTs to ``/chat/completions`` with ``stream=True`` and yields
        dicts with ``"reasoning"`` and ``"content"`` keys (one of which
        may be ``None`` per yield).

        If the API provides a structured ``delta.reasoning`` field it is
        used directly; otherwise any ``<think>...</think>`` block found in
        ``delta.content`` is stripped client-side via :class:`ThinkSplitter`.

        Args:
            model: OpenRouter model slug.
            messages: Chat messages in OpenAI format.
            extra_body: Additional fields merged into the request body.

        Yields:
            ``{"reasoning": str | None, "content": str | None}`` dicts.

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

        splitter = ThinkSplitter()

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
                    break
                data = json.loads(payload)
                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                api_reasoning: str | None = delta.get("reasoning")
                content: str | None = delta.get("content")

                if api_reasoning:
                    yield {"reasoning": api_reasoning, "content": None}

                if content:
                    for r_piece, c_piece in splitter.feed(content):
                        if api_reasoning:
                            # API already covered reasoning; don't duplicate.
                            if c_piece:
                                yield {"reasoning": None, "content": c_piece}
                        else:
                            yield {"reasoning": r_piece, "content": c_piece}

        # Flush any remaining buffered text.
        r_remainder, c_remainder = splitter.flush()
        if r_remainder:
            yield {"reasoning": r_remainder, "content": None}
        if c_remainder:
            yield {"reasoning": None, "content": c_remainder}

    async def embedding(self, model: str, input: str | list[str]) -> list[float]:
        """POST /embeddings and return the embedding vector.

        Args:
            model: OpenRouter embedding model slug.
            input: Text to embed (single string or list of strings).

        Returns:
            The embedding vector (list of floats) for the first result.

        Raises:
            CreditExhausted: on HTTP 402.
            OpenRouterAPIError: on other HTTP errors.
        """
        body = {
            "model": model,
            "input": input,
            "provider": self._zdr_provider(),
        }
        resp = await self._post(
            "/embeddings",
            body,
            model=model,
            req_meta={
                "input_chars": len(input) if isinstance(input, str) else sum(len(t) for t in input)
            },
        )
        return resp["data"][0]["embedding"]
