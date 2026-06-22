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
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import contextmanager
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
        retry_after: Server-provided ``Retry-After`` value in seconds, if
            present (RFC 9110 §10.2.3). ``None`` when the header is absent
            or unparseable. Retriers honour this with at least this delay
            (§12.2).
    """

    def __init__(
        self,
        status: int,
        endpoint: str,
        body: str,
        error_name: str | None = None,
        *,
        retry_after: int | None = None,
    ) -> None:
        self.status = status
        self.endpoint = endpoint
        self.body = body[:2000]
        self.error_name = error_name
        self.retry_after = retry_after
        preview = body[:500]
        msg = f"OpenRouter API error {status} on {endpoint}"
        if error_name:
            msg += f" ({error_name})"
        msg += f": {preview}"
        super().__init__(msg)


class SensitiveRoutingError(OpenRouterError):
    """Raised when sensitive content would be sent without strict ZDR routing."""


# -- STT retry constants -----------------------------------------------------

_RETRYABLE_STATUSES: set[int] = {429, 500, 502, 503, 504}

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


def is_sensitive_path(cfg: Config, path: Path) -> bool:
    """Return True when *path* is under a configured sensitive inbox path."""
    try:
        rel = path.resolve().relative_to(cfg.brain_root.resolve()).as_posix().lower()
    except Exception:
        rel = path.as_posix().lower()
    if rel.startswith("00-inbox/sensitive/"):
        return True
    privacy = getattr(cfg, "privacy", None)
    patterns = getattr(privacy, "sensitive_patterns", []) or []
    return any(pattern.strip("/").lower() in rel for pattern in patterns)


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
        self._sensitive_mode = False

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

    @contextmanager
    def sensitive_mode(self) -> Any:
        """Temporarily require strict ZDR routing for every request."""
        old = self._sensitive_mode
        self._sensitive_mode = True
        try:
            yield
        finally:
            self._sensitive_mode = old

    def _zdr_provider(self, *, sensitive: bool = False) -> dict[str, Any]:
        """Build the ``provider`` sub-dict enforcing ZDR (§12.7).

        Only includes keys whose value is truthy. Strips Nones.
        """
        strict = sensitive or self._sensitive_mode
        if strict and not self.cfg.privacy.zdr:
            raise SensitiveRoutingError(
                "sensitive content requires privacy.zdr=true; refusing non-ZDR routing"
            )
        params: dict[str, Any] = {}
        if self.cfg.privacy.zdr or strict:
            params["zdr"] = True
        if self.cfg.extraction.require_parameters or strict:
            params["require_parameters"] = True
        if self.cfg.privacy.block_training_providers or strict:
            params["data_collection"] = "deny"
        return params

    async def verify_zdr_status(self) -> dict[str, Any]:
        """Return honest ZDR verification status for ``brain init``.

        OpenRouter's documented ``/endpoints/zdr`` endpoint previews the impact
        of request-level ZDR routing. It does not prove account-level dashboard
        toggles are enabled, so those remain manual/unconfirmed.
        """
        try:
            resp = await self.client.get("/endpoints/zdr")
        except Exception as exc:
            return {
                "request_level_zdr_endpoint": "unavailable",
                "account_level_zdr": "manual_unconfirmed",
                "message": f"Could not verify OpenRouter ZDR preview endpoint: {exc}",
            }
        if resp.status_code == 200:
            return {
                "request_level_zdr_endpoint": "verified",
                "account_level_zdr": "manual_unconfirmed",
                "message": (
                    "OpenRouter ZDR endpoint preview is reachable; account-level "
                    "privacy toggles still require manual confirmation."
                ),
            }
        return {
            "request_level_zdr_endpoint": "unverified",
            "account_level_zdr": "manual_unconfirmed",
            "message": f"OpenRouter ZDR preview returned HTTP {resp.status_code}.",
        }

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
            # Parse Retry-After (§12.2) — RFC 9110 §10.2.3 allows either a
            # delta-seconds integer or an HTTP-date; we only honour the
            # integer form (the OpenRouter docs specify seconds).
            retry_after: int | None = None
            raw = resp.headers.get("retry-after")
            if raw:
                try:
                    retry_after = int(raw)
                except ValueError:
                    retry_after = None
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
                retry_after=retry_after,
            )
            raise OpenRouterAPIError(
                status=resp.status_code,
                endpoint=endpoint,
                body=resp.text,
                error_name=error_name,
                retry_after=retry_after,
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

    # -- shared retry helper ---------------------------------------------

    async def _with_retry(
        self,
        coro_factory: Callable[[], Awaitable[dict[str, Any]]],
        *,
        model: str,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Run *coro_factory* with retry on transient OpenRouter errors (§12.2).

        Args:
            coro_factory: Zero-arg callable that returns a fresh awaitable
                on each invocation.  It must call :meth:`_post` so that
                status codes / timeouts are raised as
                :class:`OpenRouterAPIError` / ``httpx`` exceptions.  The
                factory is invoked once per attempt; the awaitable returned
                by an earlier invocation MUST NOT be re-awaited.
            model: Model slug for log context.
            max_attempts: Total attempts (including the first) before the
                last exception is reraised.  Defaults to ``3``.

        Retries on:

        - :class:`OpenRouterAPIError` whose ``status`` is in
          ``{429, 500, 502, 503, 504}``
        - :class:`httpx.TimeoutException`
        - :class:`httpx.ConnectError`

        Other 4xx responses (400/401/403/404) are not retried and bubble up
        to the caller — the §12.2 contract is "4xx → next model" (the
        caller is responsible for switching to the repair model).

        Honours ``retry_after`` from the most recent failure (waits at
        least that many seconds before the next attempt) and otherwise
        uses ``tenacity.wait_exponential_jitter(initial=1, max=10)``.

        Raises:
            The last exception of a retryable kind, reraised verbatim.
            Non-retryable exceptions (e.g. :class:`CreditExhaustedError`,
            400/401/403/404 :class:`OpenRouterAPIError`) propagate
            immediately on first attempt.
        """
        exp_jitter = tenacity.wait_exponential_jitter(initial=1, max=10)

        def _is_retryable(exc: BaseException) -> bool:
            if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
                return True
            return (
                isinstance(exc, OpenRouterAPIError)
                and exc.status in _RETRYABLE_STATUSES
            )

        def _wait(retry_state: tenacity.RetryCallState) -> float:
            # The wait function is called BEFORE the next attempt, with
            # retry_state.outcome already populated with the previous
            # attempt's exception.  Honour server-supplied retry_after
            # when present (§12.2), otherwise fall back to exponential
            # backoff with jitter.
            outcome = retry_state.outcome
            if outcome is not None and outcome.failed:
                exc = outcome.exception()
                if (
                    isinstance(exc, OpenRouterAPIError)
                    and exc.retry_after is not None
                    and exc.retry_after > 0
                ):
                    return float(exc.retry_after)
            return exp_jitter(retry_state)

        async for attempt in tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(max_attempts),
            wait=_wait,
            retry=tenacity.retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                try:
                    return await coro_factory()
                except OpenRouterAPIError as e:
                    if _is_retryable(e):
                        log.warning(
                            "openrouter.retry",
                            model=model,
                            status=e.status,
                            attempt=attempt.retry_state.attempt_number,
                            max_attempts=max_attempts,
                            retry_after=e.retry_after,
                        )
                    raise

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
        resp = await self._with_retry(
            lambda: self._post(
                "/chat/completions",
                body,
                model=model,
                req_meta={"n_images": len(images)},
            ),
            model=model,
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

        Retries via :meth:`_with_retry` on ``httpx.TimeoutException``,
        ``httpx.ConnectError``, or ``OpenRouterAPIError`` with retryable
        status (429/500/502/503/504).  Non-retryable statuses
        (400/401/402/404) raise immediately.  402 remains
        :class:`CreditExhaustedError`.

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

        # §12.7 — ``whisper-1`` has no ZDR endpoint; the default
        # ``whisper-large-v3`` model is served by Groq/Together which both
        # support ZDR.  Force routing through those providers via
        # ``provider.only`` (in addition to the global ZDR toggle).  Users
        # who override ``stt`` in config.toml may end up with a model that
        # *isn't* on either provider — the ``provider.only`` list will be
        # ignored by OpenRouter when no provider in the list serves the
        # requested model, falling back to the next cheapest available.
        stt_provider = self._zdr_provider(sensitive=is_sensitive_path(self.cfg, audio_path))
        stt_provider["only"] = ["groq", "together"]

        body: dict[str, Any] = {
            "model": model,
            "input_audio": {"data": b64, "format": fmt},
            "provider": stt_provider,
        }
        if language is not None:
            body["language"] = language

        resp = await self._with_retry(
            lambda: self._post(
                "/audio/transcriptions",
                body,
                model=model,
                req_meta={"audio_bytes": len(raw), "format": fmt},
            ),
            model=model,
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

        return await self._with_retry(
            lambda: self._post(
                "/chat/completions",
                body,
                model=model,
                req_meta={
                    "input_chars": sum(len(m.get("content", "")) for m in messages)
                },
            ),
            model=model,
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
        resp = await self._with_retry(
            lambda: self._post(
                "/embeddings",
                body,
                model=model,
                req_meta={
                    "input_chars": len(input)
                    if isinstance(input, str)
                    else sum(len(t) for t in input)
                },
            ),
            model=model,
        )
        return resp["data"][0]["embedding"]
