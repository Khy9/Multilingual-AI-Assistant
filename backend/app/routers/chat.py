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

from fastapi import APIRouter, HTTPException
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
    # Browser-generated too, so a brand-new chat needs no round-trip before its
    # first message. The row is created lazily once that message succeeds.
    conversation_id: str | None = None
    # None = auto-detect (the default behaviour). Otherwise a key from
    # lang_detect.VALID_OVERRIDES; anything else is ignored, never trusted.
    language_override: str | None = None


class OverrideUpdate(BaseModel):
    language_override: str | None = None


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


@router.get("/languages")
async def supported_languages() -> JSONResponse:
    """The language modes a client may pin, so the UI's dropdown is generated from
    the same whitelist the server validates against and cannot drift from it."""
    return JSONResponse(
        {
            "options": [
                {
                    "value": key,
                    "codes": codes,
                    "label": " + ".join(
                        lang_detect.LANGUAGE_NAMES.get(code, code) for code in codes
                    ),
                }
                for key, codes in lang_detect.VALID_OVERRIDES.items()
            ]
        }
    )


# --- Saved conversations ------------------------------------------------------
# Every route is scoped by user_id. That is not authentication — user_id is a
# client-supplied localStorage value and anyone can send any id — it just stops one
# browser's list from showing another's. Noted as a limitation in README.md.


@router.get("/conversations")
async def list_conversations(user_id: str = "anonymous") -> JSONResponse:
    return JSONResponse({"conversations": memory.list_conversations(user_id)})


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, user_id: str = "anonymous") -> JSONResponse:
    conversation = memory.get_conversation(conversation_id, user_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="No such conversation.")
    return JSONResponse(conversation)


@router.patch("/conversations/{conversation_id}")
async def update_conversation(
    conversation_id: str, payload: OverrideUpdate, user_id: str = "anonymous"
) -> JSONResponse:
    """Persist a language-mode change made without sending a message.

    404 here is normal, not an error state: a chat with no messages yet has no row,
    and the client holds the choice until its first message creates one.
    """
    override = payload.language_override
    if override is not None and override not in lang_detect.VALID_OVERRIDES:
        raise HTTPException(status_code=400, detail=f"Unknown language option {override!r}.")

    if not memory.set_language_override(conversation_id, user_id, override):
        raise HTTPException(status_code=404, detail="No such conversation.")
    return JSONResponse({"status": "ok", "language_override": override})


@router.delete("/conversations/{conversation_id}/messages")
async def clear_conversation(conversation_id: str, user_id: str = "anonymous") -> JSONResponse:
    """Empty one conversation but keep it — the UI's "Clear this chat"."""
    if not memory.clear_conversation_messages(conversation_id, user_id):
        raise HTTPException(status_code=404, detail="No such conversation.")
    return JSONResponse({"status": "cleared", "conversation_id": conversation_id})


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, user_id: str = "anonymous") -> JSONResponse:
    if not memory.delete_conversation(conversation_id, user_id):
        raise HTTPException(status_code=404, detail="No such conversation.")
    return JSONResponse({"status": "deleted", "conversation_id": conversation_id})


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

    # --- 2. Language: manual override, else two-stage detection ---------------
    # A valid override also skips the stage-2 LLM classifier entirely, saving one
    # Gemini call per turn on all-Latin input.
    detection = None
    if request.language_override:
        detection = lang_detect.detection_from_override(request.language_override, message)
        if detection is None:
            log.warning(
                "Ignoring unknown language_override %r; falling back to auto-detect.",
                request.language_override,
            )

    if detection is None:
        try:
            detection = await lang_detect.detect(
                message, preferred_languages=profile.get("preferred_languages") or None
            )
        except Exception as exc:  # noqa: BLE001 - detection must never kill the turn
            log.warning("Detection failed, defaulting to English: %s", exc)
            detection = lang_detect.DetectionResult(languages=["en"], method="fallback")

    # Only what the override actually resolved to gets persisted, so an ignored
    # value cannot come back as a stored setting.
    effective_override = request.language_override if detection.method == "manual" else None

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
            "conversation_id": request.conversation_id,
        },
    )

    # Auto-detected turns only. A manual override is a per-conversation choice, not
    # evidence of how this user writes, so it must not bias detection elsewhere.
    if detection.method != "manual":
        memory.record_turn(request.user_id, detection.languages, detection.register)

    def persist(reply: str) -> None:
        """Save the completed exchange. Called on both the cache-replay and live
        paths, and never on the error paths — a failed turn would otherwise leave
        an orphan user message the client does not have."""
        if not request.conversation_id or not reply.strip():
            return
        try:
            memory.append_turn(
                request.conversation_id,
                request.user_id,
                message,
                reply,
                language_override=effective_override,
            )
        except Exception as exc:  # noqa: BLE001 - persistence must not kill the turn
            log.warning("Could not persist conversation turn: %s", exc)

    # --- 6a. Cache hit: replay without calling Gemini at all -------------------
    if cached is not None:
        # Re-chunked so the UI animates identically to a live response. No tokens
        # are billed here — this is a replay, not a generation.
        text = cached["response"]
        for index in range(0, len(text), 24):
            yield _sse("token", {"t": text[index : index + 24]})
            await asyncio.sleep(0.012)
        persist(text)
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
    persist(full_response)
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
