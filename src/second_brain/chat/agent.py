"""Pydantic AI chat agent with native tool streaming (§10, §12.4).

The chat path is an actual agent loop: the model can call ``search_brain``,
``get_topic``, and ``get_sources`` tools, and the native Pydantic AI stream
events are mapped onto the app's SSE schema.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from second_brain.openrouter_client import resolve_api_key
from second_brain.reasoning import ThinkSplitter

SearchFn = Callable[[str, int], Awaitable[list[dict[str, Any]]]]

SYSTEM_PROMPT = """\
You are the librarian for a personal second brain.

Use the available tools to investigate before answering:
- search_brain(query, k): retrieve relevant passages.
- get_topic(slug): fetch a full topic page.
- get_sources(topic_slug): list sources attached to a topic.

Answer with grounded synthesis and inline citations using [source_id] and
[[topic_slug]]. If the brain does not contain enough evidence, say so.
"""


def _openrouter_extra_body(cfg: Any) -> dict[str, Any]:
    provider: dict[str, Any] = {}
    if getattr(cfg.privacy, "zdr", True):
        provider["zdr"] = True
    if getattr(cfg.privacy, "block_training_providers", False):
        provider["data_collection"] = "deny"
    if getattr(cfg.extraction, "require_parameters", True):
        provider["require_parameters"] = True
    return {"extra_body": {"provider": provider}}


def _normalise_hit(hit: dict[str, Any]) -> dict[str, Any]:
    text = hit.get("text") or hit.get("snippet") or ""
    return {
        "source_id": hit.get("source_id", ""),
        "topic_slug": hit.get("topic_slug"),
        "score": float(hit.get("score", 0.0)),
        "snippet": text[:500],
    }


def _topic_markdown(store: Any, slug: str) -> str:
    path = store.cfg.brain_root / "90-wiki" / f"{slug}.md"
    return path.read_text(encoding="utf-8")


def _build_agent(cfg: Any, store: Any, search_fn: SearchFn | None):
    """Create the §12.4 Pydantic AI agent with plain tools."""
    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "pydantic-ai is required for chat (§12.4). Install project dependencies."
        ) from exc

    model = OpenAIChatModel(
        cfg.models.chat,
        provider=OpenAIProvider(
            base_url=cfg.openrouter.base_url,
            api_key=resolve_api_key(cfg),
        ),
    )
    agent = Agent(model, system_prompt=SYSTEM_PROMPT, output_type=str)

    @agent.tool_plain
    async def search_brain(query: str, k: int = 8) -> list[dict[str, Any]]:
        if search_fn is None:
            return []
        hits = await search_fn(query, k)
        return [_normalise_hit(h) for h in hits]

    @agent.tool_plain
    async def get_topic(slug: str) -> str:
        try:
            return _topic_markdown(store, slug)
        except OSError:
            return f"Topic {slug!r} not found."

    @agent.tool_plain
    async def get_sources(topic_slug: str) -> list[dict[str, Any]]:
        topic = store.state.topics.get(topic_slug)
        if topic is None:
            return []
        results: list[dict[str, Any]] = []
        for source_id in topic.sources:
            src = store.state.sources.get(source_id)
            if src is None:
                continue
            results.append(
                {
                    "source_id": source_id,
                    "raw": src.raw,
                    "topics": src.topics,
                    "ingested": src.ingested,
                    "type": src.type,
                }
            )
        return results

    return agent


def _tool_result_payload(event: Any) -> dict[str, Any]:
    content = getattr(getattr(event, "part", None), "content", None)
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = content
    else:
        parsed = content
    payload: dict[str, Any] = {"type": "tool_result", "hits": []}
    if isinstance(parsed, list):
        payload["hits"] = parsed
    else:
        payload["result"] = parsed
    return payload


async def _stream_pydantic_events(
    agent: Any,
    query: str,
    cfg: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Map Pydantic AI stream events onto the app's SSE schema."""
    from pydantic_ai import (
        AgentRunResultEvent,
        FunctionToolCallEvent,
        FunctionToolResultEvent,
        PartDeltaEvent,
        TextPartDelta,
        ThinkingPartDelta,
    )

    splitter = ThinkSplitter()
    answer_started = False
    yield {"type": "thinking", "content": "Starting agent loop..."}

    async with agent.run_stream_events(
        query,
        model_settings=_openrouter_extra_body(cfg),
    ) as stream:
        async for event in stream:
            if isinstance(event, PartDeltaEvent):
                delta = event.delta
                if isinstance(delta, ThinkingPartDelta):
                    content = getattr(delta, "content_delta", "")
                    if content:
                        yield {"type": "reasoning_delta", "content": content}
                elif isinstance(delta, TextPartDelta):
                    content = getattr(delta, "content_delta", "")
                    for reasoning, answer in splitter.feed(content):
                        if reasoning:
                            yield {"type": "reasoning_delta", "content": reasoning}
                        if answer:
                            answer_started = True
                            yield {"type": "answer_delta", "content": answer}
            elif isinstance(event, FunctionToolCallEvent):
                part = event.part
                yield {
                    "type": "tool_call",
                    "tool": part.tool_name,
                    "args": part.args,
                }
            elif isinstance(event, FunctionToolResultEvent):
                payload = _tool_result_payload(event)
                payload["tool"] = getattr(getattr(event, "part", None), "tool_name", "")
                yield payload
            elif isinstance(event, AgentRunResultEvent):
                output = getattr(getattr(event, "result", None), "output", None)
                if isinstance(output, str) and not answer_started:
                    for reasoning, answer in splitter.feed(output):
                        if reasoning:
                            yield {"type": "reasoning_delta", "content": reasoning}
                        if answer:
                            answer_started = True
                            yield {"type": "answer_delta", "content": answer}

    reasoning, answer = splitter.flush()
    if reasoning:
        yield {"type": "reasoning_delta", "content": reasoning}
    if answer:
        yield {"type": "answer_delta", "content": answer}
    yield {"type": "done"}


async def chat_stream(
    query: str,
    cfg: Any,
    store: Any,
    vec_store: Any | None = None,  # retained for API compatibility; unused by §12.4
    embedder: Any | None = None,   # retained for API compatibility; unused by §12.4
    client: Any = None,            # retained for API compatibility; unused by §12.4
    *,
    k: int = 8,  # noqa: ARG001 - default tool k is controlled by the model/tool args
    search_fn: SearchFn | None = None,
    agent: Any | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run the §12.4 Pydantic AI agent and stream typed SSE events."""
    del vec_store, embedder, client
    active_agent = agent or _build_agent(cfg, store, search_fn)
    async for event in _stream_pydantic_events(active_agent, query, cfg):
        yield event


async def chat_once(
    query: str,
    cfg: Any,
    store: Any,
    vec_store: Any | None = None,
    embedder: Any | None = None,
    client: Any = None,
    *,
    search_fn: SearchFn | None = None,
    agent: Any | None = None,
) -> str:
    """Return the concatenated final answer text."""
    parts: list[str] = []
    async for event in chat_stream(
        query,
        cfg,
        store,
        vec_store,
        embedder,
        client,
        search_fn=search_fn,
        agent=agent,
    ):
        if event["type"] == "answer_delta":
            parts.append(event["content"])
    return "".join(parts)
