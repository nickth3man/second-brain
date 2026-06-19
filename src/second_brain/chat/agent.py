"""Retrieve-then-answer chat agent with streaming events (§10, §12.4).

Implements the agentic RAG flow: retrieve-then-answer (not model-initiated
tool-calling) so it works with any chat model.  Yields typed event dicts
matching the SSE schema from §12.4.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from second_brain.vectors.retrieval import search_brain


async def chat_stream(
    query: str,
    cfg: Any,
    store: Any,
    vec_store: Any,
    embedder: Any,
    client: Any,
    *,
    k: int = 8,
) -> AsyncIterator[dict[str, Any]]:
    """Retrieve-then-answer with streaming events.

    Yields event dicts (each has a ``type`` key) in this order:

    1. ``thinking`` — reasoning-in-progress status.
    2. ``tool_call`` — the search_brain invocation.
    3. ``tool_result`` — the retrieved hits (or error).
    4. ``reasoning_delta`` — zero or more model CoT tokens (only on
       reasoning models; interleaved with ``answer_delta``).
    5. ``answer_delta`` — one or more content tokens from the LLM.
    6. ``done`` — stream complete.

    Args:
        query: The user's question.
        cfg: Application config (needs ``cfg.models.chat``).
        store: BrainStateStore (unused in this phase, kept for future tools).
        vec_store: VectorStore for search.
        embedder: Embedder with ``embed_query``.
        client: OpenRouterClient with ``chat_completion_stream``.
        k: Number of search hits to retrieve.

    Yields:
        Event dicts.
    """
    # 1. thinking
    yield {"type": "thinking", "content": "Searching the brain..."}

    # 2. tool_call
    yield {"type": "tool_call", "tool": "search_brain", "args": {"query": query, "k": k}}

    # 3. tool_result (may fail gracefully)
    hits: list[dict[str, Any]] = []
    try:
        results = await search_brain(query, vec_store, embedder, k=k)
        for h in results:
            hits.append(
                {
                    "source_id": h.source_id,
                    "topic_slug": h.topic_slug,
                    "score": h.score,
                    "snippet": h.text[:300],
                }
            )
        yield {"type": "tool_result", "tool": "search_brain", "hits": hits}
    except Exception as exc:
        yield {
            "type": "tool_result",
            "tool": "search_brain",
            "hits": [],
            "error": str(exc),
        }

    # 4. Build grounded prompt
    system_msg = (
        "You are the librarian for a personal second brain. "
        "Answer the user's question using ONLY the provided sources when relevant. "
        "Cite sources inline as [source_id] and topics as [[topic_slug]]. "
        "If the sources don't cover it, say so."
    )

    if hits:
        passages = "\n\n".join(
            f"[{h['source_id']}] (topic: [[{h['topic_slug']}]], "
            f"score: {h['score']:.4f})\n{h['snippet']}"
            for h in hits
        )
        user_content = (
            f"Question: {query}\n\n"
            f"Retrieved sources:\n{passages}"
        )
    else:
        user_content = (
            f"Question: {query}\n\n"
            "No relevant sources found in the brain. Answer from your own "
            "knowledge, but note the limited grounding."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

    # 5. Stream answer deltas (with optional reasoning from reasoning models)
    async for piece in client.chat_completion_stream(cfg.models.chat, messages):
        if piece.get("reasoning"):
            yield {"type": "reasoning_delta", "content": piece["reasoning"]}
        if piece.get("content"):
            yield {"type": "answer_delta", "content": piece["content"]}

    # 6. done
    yield {"type": "done"}


async def chat_once(
    query: str,
    cfg: Any,
    store: Any,
    vec_store: Any,
    embedder: Any,
    client: Any,
) -> str:
    """Convenience wrapper that returns the full answer text (for CLI).

    Consumes :func:`chat_stream` and concatenates all ``answer_delta``
    content into a single string.
    """
    parts: list[str] = []
    async for event in chat_stream(query, cfg, store, vec_store, embedder, client):
        if event["type"] == "answer_delta":
            parts.append(event["content"])
    return "".join(parts)
