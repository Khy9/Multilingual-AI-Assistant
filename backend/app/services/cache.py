"""Semantic response cache.

Exact-string caching is close to useless for a chat assistant: "What is the refund
policy?", "what's the refund policy" and "refund policy enti?" are the same question
but three different strings. So we cache on MEANING — embed the query, compare
cosine similarity against previously-answered queries, and replay the stored answer
above a threshold.

Cost note: a cache probe still costs one embedding call, but embeddings are far
cheaper than generation and have a much higher free-tier ceiling, so on the Gemini
free tier this trades an expensive quota for a cheap one.

Threshold note: the cutoff lives in config.cache_similarity_threshold and was
calibrated by measuring real pairs, not picked by intuition (see the comment there
for the data). It is deliberately strict. Serving a cached answer to a question
that merely *resembles* a previous one is the worst failure this module can
produce — the user gets a confidently wrong answer with no indication anything went
wrong. A miss just costs one API call.

Scaling note: this is a JSON file loaded into memory — fine for a single-container
student project with hundreds of entries. To scale, swap the internals of
lookup()/store() for Redis (RediSearch vector index) or any vector DB. The two
function signatures are the seam; nothing outside this module changes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.services import llm

log = logging.getLogger(__name__)

_lock = threading.Lock()
_entries: list[dict[str, Any]] | None = None

# Beyond this the file is trimmed oldest-first, to bound both memory and load time.
MAX_ENTRIES = 500


def _load() -> list[dict[str, Any]]:
    global _entries
    if _entries is not None:
        return _entries

    settings = get_settings()
    if settings.cache_file.exists():
        try:
            _entries = json.loads(settings.cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Cache file unreadable (%s); starting empty.", exc)
            _entries = []
    else:
        _entries = []
    return _entries


def _persist() -> None:
    settings = get_settings()
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.cache_file.write_text(
            json.dumps(_entries or [], ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:  # pragma: no cover
        log.warning("Could not persist cache: %s", exc)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _scope_key(has_documents: bool) -> str:
    """Answers grounded in an uploaded document must not be replayed for a user
    with no document loaded (and vice versa) — the correct answer differs."""
    return "rag" if has_documents else "plain"


async def lookup(query: str, *, has_documents: bool = False) -> dict[str, Any] | None:
    """Return a cached entry for a semantically-similar past query, else None."""
    settings = get_settings()
    entries = _load()
    scope = _scope_key(has_documents)

    candidates = [e for e in entries if e.get("scope") == scope and e.get("embedding")]
    if not candidates:
        return None

    try:
        query_vector = np.asarray(await llm.embed_one(query), dtype=np.float32)
    except llm.LLMError as exc:
        # A cache miss is always safe; never block a chat on cache infrastructure.
        log.warning("Cache lookup skipped (embedding failed): %s", exc)
        return None

    if query_vector.size == 0:
        return None

    best: dict[str, Any] | None = None
    best_score = 0.0
    for entry in candidates:
        score = _cosine(query_vector, np.asarray(entry["embedding"], dtype=np.float32))
        if score > best_score:
            best_score, best = score, entry

    if best is not None and best_score >= settings.cache_similarity_threshold:
        log.info("Semantic cache HIT (similarity %.3f) for %r", best_score, query[:60])
        return {
            "response": best["response"],
            "similarity": round(best_score, 4),
            "original_query": best["query"],
        }

    log.info("Semantic cache MISS (best similarity %.3f) for %r", best_score, query[:60])
    return None


async def store(query: str, response: str, *, has_documents: bool = False) -> None:
    """Write a completed response into the cache. Called after a stream finishes."""
    if not query.strip() or not response.strip():
        return

    try:
        embedding = await llm.embed_one(query)
    except llm.LLMError as exc:
        log.warning("Cache store skipped (embedding failed): %s", exc)
        return

    if not embedding:
        return

    with _lock:
        entries = _load()
        entries.append(
            {
                "query": query,
                "response": response,
                "embedding": embedding,
                "scope": _scope_key(has_documents),
                "created_at": time.time(),
            }
        )
        if len(entries) > MAX_ENTRIES:
            del entries[: len(entries) - MAX_ENTRIES]
        _persist()


def stats() -> dict[str, int]:
    entries = _load()
    return {
        "total": len(entries),
        "plain": sum(1 for e in entries if e.get("scope") == "plain"),
        "rag": sum(1 for e in entries if e.get("scope") == "rag"),
    }


def clear() -> None:
    global _entries
    with _lock:
        _entries = []
        _persist()
