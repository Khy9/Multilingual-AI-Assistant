# Architecture

## Shape of the system

One FastAPI process serves the API *and* the frontend. There is no separate web server, no message
queue and no external database — everything persists to one directory that Docker mounts as a
volume.

```
                          ┌─────────────────────────────┐
  Browser ───────────────►│  FastAPI (app/main.py)      │
  frontend/index.html     │  · mounts frontend/ at /    │
  frontend/script.js      │  · /health                  │
                          └──────────┬──────────────────┘
                                     │
                   ┌─────────────────┴──────────────────┐
                   │                                    │
          routers/chat.py                       routers/documents.py
          POST /chat/stream (SSE)               POST /documents/upload
                   │                                    │
                   ▼                                    ▼
   ┌───────────────────────────────┐            services/rag.py
   │ 1. memory.get_profile()       │            (chunk → embed → store)
   │ 2. lang_detect.detect()       │
   │ 3. rag.retrieve()             │
   │ 4. cache.lookup()             │
   │ 5. prompts.build_system_prompt│
   │ 6. llm.stream_chat()  ────────┼──────────► Google Gemini API
   │ 7. cache.store() + memory     │
   └───────────────────────────────┘
                   │
                   ▼
        data/  ── chroma/            (vector store)
               ── semantic_cache.json (cached answers)
               ── memory.sqlite3      (user language profiles)
```

## Request lifecycle: `POST /chat/stream`

Implemented in `app/routers/chat.py::_event_stream`. Order matters.

1. **Profile lookup** (`memory.get_profile`) — what language pair does this user usually write in?
   Used only to break ties later; never to override evidence in the message itself.
2. **Language detection** (`lang_detect.detect`) — two stages, described below.
3. **Retrieval** (`rag.retrieve`) — skipped entirely if no document has been uploaded.
4. **Cache probe** (`cache.lookup`) — scoped by whether document context is in play, because the
   right answer to the same question differs with and without a document loaded.
5. **`meta` frame emitted** — the UI updates its language badge and shows which excerpts were used
   *before* any answer text arrives.
6. **Generation** — either a cache replay (no Gemini call at all) or `llm.stream_chat`, whose
   chunks are forwarded to the client the instant they arrive.
7. **Write-back** — the finished answer goes into the cache, and the turn is folded into the user's
   profile.

Every step from 1–4 is wrapped so a failure degrades the response rather than killing the request:
detection failure falls back to English, retrieval failure answers without context, cache failure
is treated as a miss.

### SSE frame protocol

```
event: meta
data: {"detection": {...}, "cache_hit": false, "rag_chunks": [...]}

event: token
data: {"t": "Namaste"}

event: done
data: {"cached": false, "chars": 412}
```

Errors terminate the stream with `event: error` carrying a `kind` of `rate_limit`,
`not_configured` or `llm_error`. The route never returns a 500 for a provider failure — by the time
generation starts, HTTP headers have already been sent, so an exception would just truncate the
stream silently.

## Modules

| Module | Responsibility |
|---|---|
| `core/config.py` | All configuration and secrets, via `pydantic-settings`. Derives every storage path from one `data_dir` so Docker mounts a single volume. |
| `services/llm.py` | The **only** module importing the Gemini SDK for generation. Exposes `stream_chat`, `complete`, `embed_texts`. Translates provider errors into `RateLimitError` / `NotConfiguredError` / `LLMError`. |
| `services/lang_detect.py` | Two-stage detection plus register detection. |
| `services/rag.py` | Chunking, PDF/text extraction, embedding, Chroma persistence, retrieval. |
| `services/cache.py` | Embedding-similarity response cache. |
| `services/memory.py` | SQLite store of each user's language/register history. |
| `services/prompts.py` | Assembles the system prompt from detection + register + retrieved context + few-shot examples. |
| `routers/chat.py` | SSE streaming endpoint and profile/cache admin routes. |
| `routers/documents.py` | Upload validation and RAG status. |

## The two-stage language detector

This is the core technical idea, in `services/lang_detect.py`.

**The problem.** `langdetect`, fastText and CLD3 all assume one language per document and classify
from character n-grams. Two failure modes follow:

- *Code-mixing*: "Naaku ee document lo revenue figures kavali" is Telugu grammar with English
  nouns. A single-label classifier must pick one, and is wrong either way.
- *Romanization*: that sentence contains zero Telugu characters. To a script- or n-gram-based
  detector it is just Latin text, and it will guess something like Portuguese with high confidence.

**Stage 1 — `detect_scripts()`.** Classify each token by Unicode block (Telugu `U+0C00–0C7F`,
Devanagari `U+0900–097F`, Latin). Free, instant, offline, and runs on every request. If any
non-Latin script is present, the answer is conclusive and we stop here — no API call.

**Stage 2 — `classify_with_llm()`.** Only reached when the text is *entirely* Latin, i.e. exactly
the case where script evidence cannot separate English from romanized Telugu/Hindi. One cheap
non-streaming Gemini call returns JSON naming the languages present. An LLM handles this well
because it reads words, not character statistics.

If stage 2 is unavailable — no key, 429, or unparseable output — `_heuristic_roman_languages()`
matches known function words (`meeku`, `kavali`, `chahiye`, `kaise`) offline. Note that `yaar` and
`bhai` are deliberately *excluded* from those lists: they are so absorbed into everyday Indian
English that they indicate casual register, not Hindi grammar.

The split is deliberate: the cheap path never depends on the network, and the expensive path is
only paid for when it is genuinely needed.

## Cross-language retrieval

`gemini-embedding-001` is multilingual, so "third quarter revenue", "మూడవ త్రైమాసిక ఆదాయం" and
"third quarter revenue enta?" land near each other in one shared vector space. A Telugu query
therefore retrieves English chunks directly — **no translation step exists in this pipeline**, and
adding one would only lose information.

Chroma is used purely as a vector store: `rag.py` computes embeddings itself and passes them to
`collection.add(embeddings=...)`, bypassing Chroma's default ONNX embedding function. That keeps
the multilingual model in charge of retrieval quality and avoids downloading a local ONNX model.

The system prompt then explicitly tells the model that excerpts may be in a different language from
the question, and to answer in the *user's* language regardless.

## Swappable interfaces

Each of these is isolated behind a narrow function surface, so replacing it touches one file:

| Concern | Current | Interface to preserve | Swap to |
|---|---|---|---|
| **LLM provider** | Google Gemini | `stream_chat(messages, system_prompt)` yielding `str`; `complete()`; `embed_texts()` | OpenAI, Anthropic, local Ollama. Routers pass plain `{"role","content"}` dicts and never see SDK types. |
| **Vector store** | ChromaDB (embedded, on-disk) | `ingest_document()`, `retrieve()`, `stats()`, `clear()` | FAISS, pgvector, Pinecone, Qdrant |
| **Cache backend** | JSON file + numpy cosine | `lookup()`, `store()` (both async) | Redis with RediSearch vector index; the async signatures already accommodate a network round-trip |
| **User memory** | stdlib `sqlite3` | `get_profile()`, `record_turn()`, `reset_profile()` | Postgres, DynamoDB |

The seam that matters most is the LLM one. `services/llm.py` is the single place `from google import
genai` appears for generation — nothing else in the codebase knows which provider is behind it.

## Scaling notes

Honest limits of the current design, and what each would become:

- **Semantic cache** is an in-process list persisted to JSON, scanned linearly. Fine for hundreds of
  entries in one container; becomes Redis + a vector index beyond that, and would need to be shared
  rather than per-process once there is more than one replica.
- **User memory** uses one SQLite file with a process-level lock — single-writer by construction.
  Multiple replicas need a real database.
- **Chroma** runs embedded, sharing the container's disk and memory. A multi-replica deployment
  needs Chroma in server mode or a hosted vector DB, since the volume cannot be safely shared.

In short: this is deliberately a single-container design. Every one of those limits is a
consequence of that choice, and each is confined to one module.
