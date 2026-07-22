"""LLM provider wrapper (Google Gemini).

This is the ONLY module that imports the Gemini SDK for generation. Routers speak
plain dicts and plain strings, so swapping in another provider means rewriting
this file and nothing else. See ARCHITECTURE.md.

SDK reference (current as of this build):
    pip install google-genai
    from google import genai
    client = genai.Client(api_key=...)
    async for chunk in await client.aio.models.generate_content_stream(...)
    client.models.embed_content(model="gemini-embedding-001", contents=[...])
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Iterable

from google import genai
from google.genai import types

from app.core.config import get_settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base class for provider failures that routers are expected to handle."""


class RateLimitError(LLMError):
    """Gemini free tier exhausted (HTTP 429 RESOURCE_EXHAUSTED).

    Raised instead of letting the SDK exception bubble up, so the chat route can
    emit a clean SSE error frame rather than a 500 traceback.
    """


class NotConfiguredError(LLMError):
    """No GEMINI_API_KEY in the environment."""


_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Lazily build the shared client so importing this module never needs a key."""
    global _client
    settings = get_settings()
    if not settings.llm_configured:
        raise NotConfiguredError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env in the project "
            "root and add your key from https://aistudio.google.com/apikey"
        )
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _is_rate_limit(exc: Exception) -> bool:
    """The SDK surfaces quota errors as ClientError with status 429 /
    RESOURCE_EXHAUSTED. Match on both, since the exception class name has
    changed across SDK versions."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return True
    blob = f"{type(exc).__name__} {exc}".upper()
    return "429" in blob or "RESOURCE_EXHAUSTED" in blob or "QUOTA" in blob


def _wrap(exc: Exception) -> LLMError:
    if _is_rate_limit(exc):
        return RateLimitError(
            "Gemini free-tier rate limit reached. Wait a minute and try again "
            "(Flash-Lite allows 15 requests/minute, 1000/day)."
        )
    return LLMError(f"LLM request failed: {exc}")


def _to_contents(messages: Iterable[dict]) -> list[types.Content]:
    """Map neutral {"role", "content"} dicts to SDK types.

    Gemini uses "model" for the assistant role; anything that is not a user turn
    is normalised to that.
    """
    contents: list[types.Content] = []
    for message in messages:
        text = (message.get("content") or "").strip()
        if not text:
            continue
        role = "user" if message.get("role") == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
    return contents


async def stream_chat(
    messages: list[dict],
    system_prompt: str,
    *,
    temperature: float = 0.7,
) -> AsyncIterator[str]:
    """Yield response text chunk-by-chunk as the model produces it.

    Each chunk is yielded the moment it arrives — nothing is buffered or joined
    here, which is what makes the SSE endpoint genuinely token-by-token.
    """
    settings = get_settings()
    client = get_client()

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
    )

    try:
        stream = await client.aio.models.generate_content_stream(
            model=settings.gemini_model,
            contents=_to_contents(messages),
            config=config,
        )
        async for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yield text
    except Exception as exc:  # noqa: BLE001 - deliberately funnelled into LLMError
        raise _wrap(exc) from exc


async def complete(prompt: str, system_prompt: str = "", *, temperature: float = 0.0) -> str:
    """One-shot, non-streaming generation.

    Used by the language-detection fallback, where we want a single small JSON
    response rather than a stream.
    """
    settings = get_settings()
    client = get_client()

    config = types.GenerateContentConfig(temperature=temperature)
    if system_prompt:
        config.system_instruction = system_prompt

    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=config,
        )
        return (response.text or "").strip()
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed one or more texts with Gemini's multilingual embedding model.

    Multilingual is the whole point: a Telugu query and an English document chunk
    land near each other in the same vector space, which is what makes
    cross-language retrieval work without a translation step.
    """
    if not texts:
        return []

    settings = get_settings()
    client = get_client()

    def _call() -> list[list[float]]:
        response = client.models.embed_content(model=settings.embed_model, contents=texts)
        return [list(item.values) for item in response.embeddings]

    try:
        # embed_content is sync in the SDK; keep the event loop free.
        return await asyncio.to_thread(_call)
    except Exception as exc:  # noqa: BLE001
        raise _wrap(exc) from exc


async def embed_one(text: str) -> list[float]:
    vectors = await embed_texts([text])
    return vectors[0] if vectors else []
