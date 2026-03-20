"""
Chat endpoint with Server-Sent Events streaming:
  POST /api/chat/stream  — RAG query, streams tokens + sources via SSE
  POST /api/chat         — RAG query, returns full response (non-streaming)
"""

import logging
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from limiter import limiter
from models import ChatRequest, ChatResponse, RetrievedChunk
from services.rag_pipeline import stream_rag_response

logger = logging.getLogger(__name__)

router = APIRouter()


def _vector_store(request: Request):
    vs = request.app.state.vector_store
    if vs is None:
        raise HTTPException(status_code=503, detail="Vector store is still loading, please retry in a moment.")
    return vs


# ── Streaming endpoint (primary) ───────────────────────────────────────────────

@router.post("/stream")
@limiter.limit(settings.chat_rate_limit)
async def chat_stream(body: ChatRequest, request: Request):
    """
    Stream the RAG response as Server-Sent Events.

    Event shapes:
      {"type": "sources",  "sources": [...]}       — emitted first
      {"type": "token",    "content": "..."}        — one per token
      {"type": "done"}                              — signals end
      {"type": "error",    "error":  "..."}         — on failure
    """
    vs = _vector_store(request)
    top_k = body.top_k or settings.top_k

    async def event_generator():
        async for chunk in stream_rag_response(
            question=body.question,
            vector_store=vs,
            conversation_history=body.conversation_history,
            top_k=top_k,
        ):
            # Respect client disconnects
            if await request.is_disconnected():
                logger.info("Client disconnected; aborting stream")
                break
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


# ── Non-streaming endpoint (convenience) ──────────────────────────────────────

@router.post("", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
async def chat(body: ChatRequest, request: Request):
    """
    Collect the full streamed response and return it as a single JSON object.
    Useful for non-browser clients or testing.
    """
    vs = _vector_store(request)
    top_k = body.top_k or settings.top_k

    answer_parts: list[str] = []
    sources: list[RetrievedChunk] = []

    async for raw in stream_rag_response(
        question=body.question,
        vector_store=vs,
        conversation_history=body.conversation_history,
        top_k=top_k,
    ):
        # raw is "data: {...}\n\n"
        payload = raw.removeprefix("data: ").strip()
        if not payload:
            continue

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        match event.get("type"):
            case "token":
                answer_parts.append(event.get("content", ""))
            case "sources":
                sources = [RetrievedChunk(**s) for s in event.get("sources", [])]
            case "error":
                raise HTTPException(status_code=502, detail=event.get("error"))

    return ChatResponse(
        answer="".join(answer_parts),
        sources=sources,
        model=settings.claude_model,
    )
