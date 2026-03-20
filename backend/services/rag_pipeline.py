"""
RAG pipeline: retrieves relevant chunks from the vector store, then streams a
Claude-generated answer via Server-Sent Events.
"""

import json
import logging
from typing import List, AsyncGenerator

import anthropic

from config import settings
from models import ChatMessage, RetrievedChunk, SSEChunk
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

# ── Prompt helpers ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise, helpful assistant with access to a curated knowledge base.

Guidelines:
- Answer based **only** on the provided context. Do not fabricate information.
- When the context contains the answer, be concise and accurate.
- Cite sources inline using **[Source N]** notation (e.g., "According to [Source 1]...").
- If the context does not contain enough information to answer, say so honestly.
- Use markdown formatting where it improves readability (lists, bold, code blocks).
"""


def _build_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i}] ({chunk.source}, chunk {chunk.chunk_index})\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def _build_user_message(question: str, context: str) -> str:
    return (
        f"## Context\n\n{context}\n\n"
        f"---\n\n"
        f"## Question\n\n{question}"
    )


def _sse(event: SSEChunk) -> str:
    """Serialise an SSEChunk to an SSE-formatted string."""
    return f"data: {event.model_dump_json()}\n\n"


def _cap_history(history: List[ChatMessage]) -> List[ChatMessage]:
    """Keep only the last N turns to prevent token overflow."""
    max_turns = settings.max_history_turns
    if len(history) <= max_turns:
        return history
    # Always keep pairs (user + assistant) so we don't cut mid-turn
    trimmed = history[-max_turns:]
    # Ensure we start with a user message
    if trimmed and trimmed[0].role != "user":
        trimmed = trimmed[1:]
    logger.info("Trimmed conversation history to %d turns", len(trimmed))
    return trimmed


# ── Public API ─────────────────────────────────────────────────────────────────

async def stream_rag_response(
    question: str,
    vector_store: VectorStore,
    conversation_history: List[ChatMessage],
    top_k: int,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted strings:

      data: {"type":"sources","sources":[...]}

      data: {"type":"token","content":"Hello"}
      ...
      data: {"type":"done"}

    or on error:

      data: {"type":"error","error":"..."}
    """
    # 1. Retrieve relevant chunks ───────────────────────────────────────────────
    try:
        results = vector_store.query(query_text=question, n_results=top_k)
    except Exception as exc:
        logger.exception("Vector store query failed")
        yield _sse(SSEChunk(type="error", error=str(exc)))
        return

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: List[RetrievedChunk] = []
    for doc, meta, dist in zip(docs, metas, distances):
        # Filter out low-relevance chunks (cosine distance: lower = more similar)
        if dist > settings.relevance_threshold:
            logger.debug("Skipping chunk with distance %.3f (threshold=%.3f)", dist, settings.relevance_threshold)
            continue
        retrieved.append(
            RetrievedChunk(
                text=doc,
                source=meta.get("filename", "unknown"),
                chunk_index=meta.get("chunk_index", 0),
                distance=round(float(dist), 4),
            )
        )

    # 2. Emit sources first (so the UI can render them immediately) ────────────
    yield _sse(SSEChunk(type="sources", sources=retrieved))

    if not retrieved:
        yield _sse(
            SSEChunk(
                type="token",
                content=(
                    "I couldn't find relevant information in the knowledge base "
                    "to answer your question."
                ),
            )
        )
        yield _sse(SSEChunk(type="done"))
        return

    # 3. Build messages ─────────────────────────────────────────────────────────
    context = _build_context(retrieved)
    user_content = _build_user_message(question, context)

    history = _cap_history(conversation_history)
    messages: List[dict] = []
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": user_content})

    # 4. Stream from Claude ─────────────────────────────────────────────────────
    try:
        with _client.messages.stream(
            model=settings.claude_model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        yield _sse(SSEChunk(type="token", content=event.delta.text))
    except anthropic.APIError as exc:
        logger.exception("Claude API error")
        yield _sse(SSEChunk(type="error", error=f"Claude API error: {exc.message}"))
        return
    except Exception as exc:
        logger.exception("Unexpected error during generation")
        yield _sse(SSEChunk(type="error", error=str(exc)))
        return

    yield _sse(SSEChunk(type="done"))
