"""Chat routes, including the token-by-token SSE stream.

Pipeline per request:
    profile -> language detection -> semantic cache -> RAG retrieval
            -> system prompt -> Gemini stream -> cache write + profile update
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.services import cache, lang_detect, llm, memory, rag
from app.services.prompts import build_system_prompt

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


class Message(BaseModel):
    role: str = Field(description='"user" or "assistant"')
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = Field(default_factory=list)
    # Browser-generated stable id; enables cross-session memory without accounts.
    user_id: str = "anonymous"
    use_rag: bool = True
    use_cache: bool = True


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.get("/profile")
async def get_profile(user_id: str = "anonymous") -> JSONResponse:
    """What we remember about a returning user."""
    return JSONResponse(memory.get_profile(user_id))


@router.delete("/profile")
async def delete_profile(user_id: str = "anonymous") -> JSONResponse:
    memory.reset_profile(user_id)
    return JSONResponse({"status": "reset", "user_id": user_id})


@router.get("/cache/stats")
async def cache_stats() -> JSONResponse:
    return JSONResponse(cache.stats())


@router.delete("/cache")
async def clear_cache() -> JSONResponse:
    cache.clear()
    return JSONResponse({"status": "cleared"})


async def _event_stream(request: ChatRequest) -> AsyncIterator[str]:
    """Generate the SSE frames for one chat turn.

    Frame order: meta -> token* -> done, or meta -> error.
    """
    message = request.message.strip()
    if not message:
        yield _sse("error", {"message": "Empty message."})
        return

    # --- 1. Returning-user profile -------------------------------------------
    profile = memory.get_profile(request.user_id)

    # --- 2. Language detection (script pass, then LLM fallback if romanized) --
    try:
        detection = await lang_detect.detect(
            message, preferred_languages=profile.get("preferred_languages") or None
        )
    except Exception as exc:  # noqa: BLE001 - detection must never kill the turn
        log.warning("Detection failed, defaulting to English: %s", exc)
        detection = lang_detect.DetectionResult(languages=["en"], method="fallback")

    # --- 3. RAG retrieval ------------------------------------------------------
    chunks: list[rag.RetrievedChunk] = []
    has_documents = rag.stats()["has_documents"]
    if request.use_rag and has_documents:
        try:
            chunks = await rag.retrieve(message)
        except Exception as exc:  # noqa: BLE001
            log.warning("Retrieval failed, answering without context: %s", exc)

    # --- 4. Semantic cache probe ----------------------------------------------
    cached = None
    if request.use_cache:
        try:
            cached = await cache.lookup(message, has_documents=bool(chunks))
        except Exception as exc:  # noqa: BLE001
            log.warning("Cache lookup failed: %s", exc)

    # --- 5. Metadata frame, sent before any token so the UI can update at once -
    yield _sse(
        "meta",
        {
            "detection": detection.to_dict(),
            "cache_hit": cached is not None,
            "cache_similarity": cached["similarity"] if cached else None,
            "rag_chunks": [
                {"source": chunk.source, "score": chunk.score, "preview": chunk.text[:160]}
                for chunk in chunks
            ],
            "returning_user": profile.get("returning", False),
        },
    )

    memory.record_turn(request.user_id, detection.languages, detection.register)

    # --- 6a. Cache hit: replay without calling Gemini at all -------------------
    if cached is not None:
        # Re-chunked so the UI animates identically to a live response. No tokens
        # are billed here — this is a replay, not a generation.
        text = cached["response"]
        for index in range(0, len(text), 24):
            yield _sse("token", {"t": text[index : index + 24]})
            await asyncio.sleep(0.012)
        yield _sse("done", {"cached": True, "chars": len(text)})
        return

    # --- 6b. Live generation ---------------------------------------------------
    system_prompt = build_system_prompt(
        detection, chunks, returning_user=profile.get("returning", False)
    )
    # Last 10 turns only: keeps the prompt small, which matters on a free tier
    # billed by token.
    messages = [turn.model_dump() for turn in request.history[-10:]]
    messages.append({"role": "user", "content": message})

    collected: list[str] = []
    try:
        async for piece in llm.stream_chat(messages, system_prompt):
            collected.append(piece)
            # Yielded immediately — nothing is buffered, so the client sees tokens
            # as the model produces them.
            yield _sse("token", {"t": piece})
    except llm.RateLimitError as exc:
        yield _sse("error", {"message": str(exc), "kind": "rate_limit"})
        return
    except llm.NotConfiguredError as exc:
        yield _sse("error", {"message": str(exc), "kind": "not_configured"})
        return
    except llm.LLMError as exc:
        yield _sse("error", {"message": str(exc), "kind": "llm_error"})
        return

    full_response = "".join(collected)
    yield _sse("done", {"cached": False, "chars": len(full_response)})

    # --- 7. Cache write, after the client already has the full answer ---------
    if request.use_cache and full_response.strip():
        try:
            await cache.store(message, full_response, has_documents=bool(chunks))
        except Exception as exc:  # noqa: BLE001
            log.warning("Cache store failed: %s", exc)


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream a reply as Server-Sent Events, token by token."""
    return StreamingResponse(
        _event_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tells nginx / App Runner style proxies not to buffer the response,
            # which would silently destroy the streaming behaviour.
            "X-Accel-Buffering": "no",
        },
    )
