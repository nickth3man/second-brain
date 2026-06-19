"""OpenRouter HTTP client with ZDR privacy and keyring resolution (§12.7).

Provides an async httpx client that enforces ``provider.zdr: true`` on every
request, resolves the API key via keyring → env → config, and raises typed
exceptions for HTTP 402 (credit exhaustion, §12.3).
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import keyring

from second_brain.config import Config

# ── exceptions ──────────────────────────────────────────────────────────────


class OpenRouterError(Exception):
    """Base exception for OpenRouter client errors."""


class CreditExhaustedError(OpenRouterError):
    """Raised on HTTP 402 — credit exhaustion stops the daemon (§12.3)."""


# ── key resolution ──────────────────────────────────────────────────────────


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


# ── client ──────────────────────────────────────────────────────────────────


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

    # ── provider params ─────────────────────────────────────────────────

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

    # ── API methods ─────────────────────────────────────────────────────

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
