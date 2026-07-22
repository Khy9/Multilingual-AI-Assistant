# Multilingual AI Assistant

A retrieval-augmented chat assistant built for the way people in Hyderabad actually type: Telugu,
Hindi and English mixed inside a single sentence, with the regional language usually written in
Latin script — "Tenglish" and "Hinglish".

```
Naaku ee document lo revenue figures kavali, cheppandi
Aapko ye report kaise chahiye, PDF format mein bhejun kya?
ఈ report లో ఏముంది cheppandi
```

Every one of those sentences breaks a standard NLP pipeline. This project is an attempt to handle
them properly rather than to wrap a translation API.

---

## 1. What this is, and why a normal chatbot doesn't solve it

### The problem

A user uploads an English company policy document and asks, in Telugu, what the leave policy is.
Three things have to work at once, and off-the-shelf tooling fails at all three.

**A language detector has to know what language the question is in.** The standard detectors
(`langdetect`, fastText, CLD3) make two assumptions that are both wrong here:

- *One language per document.* "Naaku ee document lo revenue figures kavali" is Telugu grammar
  wrapped around English nouns. A single-label classifier has to pick one, and is wrong whichever
  it picks.
- *Language is inferable from character statistics.* Romanized Telugu contains no Telugu
  characters. A detector sees only Latin letters and returns Portuguese, Indonesian or Somali —
  confidently. Confidence scores don't help, because the model isn't uncertain; it's wrong.

**Retrieval has to cross a language boundary.** The question is in Telugu, the document is in
English. The obvious fix — translate the query first, then search — adds a lossy step, an extra API
call of latency, and a new failure mode where a mistranslated query silently retrieves the wrong
passage.

**The reply has to sound like the person who asked.** "Report pampu" and "Meeru daya chesi report
pampandi" both mean *send me the report*, at very different politeness levels. Most assistants
flatten both into the same neutral textbook prose, which reads as subtly wrong — and translating
*up* in formality misrepresents a speaker just as badly as translating down.

### Why not just use a translator

A translate-then-answer pipeline gives up on all three. It has to pick a source language before it
can translate (problem one), it introduces a lossy hop before retrieval (problem two), and machine
translation normalizes register by design — it is built to produce standard prose (problem three).
Code-mixed input is not a translation task. It is a detection, retrieval and generation task where
the language boundary is inside the sentence.

---

## 2. How it actually works

### 2.1 Two-stage language detection

`app/services/lang_detect.py`

The insight is that the two hard cases need different tools, and only one of them costs an API
call.

**Stage 1 — Unicode script pass.** `detect_scripts()` tokenizes the text and classifies each token
by Unicode block: Telugu (`U+0C00–U+0C7F`), Devanagari (`U+0900–U+097F`), Latin, or other. A token
counts toward *every* script it contains, so `Telugu-లో` registers as both. This is offline,
instant, free, and runs on every request.

If any non-Latin script is present, detection is finished — script evidence is conclusive
(`method: "script"`, confidence 0.99). Telugu script means Telugu. There is nothing to guess.

**Stage 2 — LLM classifier, only for all-Latin text.** This is the case where script tells you
nothing: `"Naaku ee document lo revenue kavali"` and `"Send me the revenue figures"` are both 100%
Latin. `classify_with_llm()` asks Gemini to name the languages present and return strict JSON
(`{"languages": ["te-rom","en"], "code_mixed": true, "register": "casual"}`). An LLM handles this
well because it reads *words*, not character n-grams — the exact thing statistical detectors
cannot do.

**Stage 2 fallback — offline keyword heuristic.** If the classifier fails for any reason (no API
key, HTTP 429, non-JSON output — the parser strips markdown fences and extracts the first JSON
object before giving up), `_heuristic_roman_languages()` matches against curated sets of romanized
function words: `naaku`, `meeku`, `kavali`, `cheppu`, `enduku` for Telugu; `aapko`, `mujhe`,
`kya`, `kaise`, `chahiye` for Hindi. These were chosen to *not* collide with common English words.
Detection degrades; it never fails.

One deliberate detail: `yaar` and `bhai` are excluded from the Hindi markers. They are so absorbed
into everyday Indian English ("hey yaar, send me the file") that they signal a casual *register*,
not Hindi grammar. They live in `CASUAL_MARKERS` instead.

**Register detection** (`detect_register()`) is a separate cheap pass over the same tokens,
comparing hits against formal markers (`kindly`, `sir`, `gaaru`, `andi`, `kripya`) and casual ones
(`yaar`, `bro`, `plz`, `ra`, `anna`). Ties break on surface style: twelve-plus words with no chat
punctuation reads formal.

The result carries `method` (`script` / `llm` / `heuristic` / `profile`), which the UI surfaces as
a chip — you can always see which path made the call.

### 2.2 The RAG pipeline, end to end

`app/services/rag.py`

```
upload → extract text → chunk (900 chars, 150 overlap) → embed (one batched call)
       → store in Chroma with {source, doc_id, chunk_index}

query  → embed query → cosine search top-4 → format excerpts → inject into system prompt
```

**Extraction.** `.pdf` goes through `pypdf`; text formats are decoded by trying UTF-8, UTF-16, then
latin-1. Scanned image-only PDFs produce no text and are rejected with an explicit message — there
is no OCR in this project.

**Chunking.** 900 characters with 150 characters of overlap, but the split point is not blind.
`chunk_text()` looks for a paragraph break in the last 40% of the window, then a sentence boundary
(including the Devanagari danda `।`), and only falls back to a hard cut. Overlap matters for
retrieval: a fact that straddles a boundary would otherwise be unfindable from either side.

**Embedding.** One `embed_texts()` call for the whole document rather than one per chunk — much
gentler on the free-tier requests-per-minute limit.

**Storage.** ChromaDB in persistent mode, cosine space. Note that Chroma is used *purely* as a
vector store: embeddings are computed by `app/services/llm.py` and passed in explicitly via
`collection.add(embeddings=...)`, with `embedding_function=None`. That deliberately bypasses
Chroma's default ONNX embedder, which would otherwise download a local model on first use and —
more importantly — would not be multilingual.

**Why cross-language Q&A needs no translation step.** `gemini-embedding-001` is a *multilingual*
embedding model. "third quarter revenue", "మూడవ త్రైమాసిక ఆదాయం" and "third quarter revenue enta?"
all map to nearby points in one shared vector space. So a Telugu query retrieves the relevant
**English** chunks directly — no translation, no extra hop, no lossy intermediate. The retrieved
English text then goes into the prompt, and the model is instructed to read context in whatever
language it is in and answer in the *user's* language.

That instruction is explicit in `prompts.py`, because it is not the obvious default behavior:

> *the excerpts may be in a DIFFERENT LANGUAGE from the question. That is expected. Read them in
> whatever language they are in, and answer in the USER'S language.*

Retrieval degrades safely: if the collection is empty or the query embedding fails, `retrieve()`
returns `[]` and the chat answers without document context rather than erroring.

### 2.3 Register and tone preservation

`app/services/prompts.py`

The system prompt is assembled per request from three inputs — detected languages, detected
register, and retrieved context — so the assistant's entire "personality" is auditable in one
file.

Rules alone don't hold tone; the model drifts back to neutral prose. What actually works is
few-shot examples showing the *same meaning at two registers*, embedded in every prompt:

| Source | Casual | Formal |
|---|---|---|
| "Send me the report." | "Report pampu." | "Meeru daya chesi report pampandi." |
| "I can't do this today." | "Aaj ye nahi ho payega yaar." | "Kshama kijiye, aaj yeh sambhav nahi hoga." |
| "Ee file lo em undo cheppu ra." | "Just tell me what's in this file." | *(wrong here)* "Kindly inform me of this document's contents." |

Each example carries a note explaining *what changed* — the formal Telugu adds `daya chesi` and the
respectful `-andi` verb ending; the casual Hindi keeps `yaar` while the formal opens with an
apology and drops slang. The third row is the instructive one: the source used `ra`, which is very
informal, so a polished English rendering misrepresents the speaker. Formality is preserved in
both directions, not maximized.

Alongside the examples, the prompt states the detected register explicitly and instructs the model
to mirror code-mixing "in roughly the same proportion rather than collapsing into one language",
and to reply in the user's *script* — romanized input gets a romanized reply, never an "upgrade"
to native script.

### 2.4 Semantic cache

`app/services/cache.py`

Exact-string caching is nearly useless for chat: "What is the refund policy?", "what's the refund
policy" and "refund policy enti?" are one question and three strings. So the cache keys on
*meaning* — embed the query, cosine-compare against previously answered queries, replay the stored
answer above a threshold.

A cache probe still costs one embedding call. That is the point: embeddings have a far higher
free-tier ceiling than generation, so this trades an expensive quota for a cheap one.

Entries are scoped `rag` or `plain` (`_scope_key()`). An answer grounded in an uploaded document
must never be replayed for a user who has no document loaded — the correct answer genuinely
differs.

**The threshold, and why 0.88.** The cutoff was calibrated by measuring real embedding pairs, not
picked by intuition. Measured cosine similarities against `gemini-embedding-001`:

| Pair | Similarity | Should it hit? |
|---|---|---|
| "third quarter revenue?" vs "Q3 revenue figures?" | 0.946 | yes |
| "notice period for director" vs "how long must a director…" | 0.934 | yes |
| "what is the leave policy?" vs "tell me about the leave policy" | 0.886 | yes |
| **"what is the leave policy?" vs "what is the SICK leave policy?"** | **0.860** | **no — different question** |
| "how many days annual leave" vs "yearly paid vacation entitlement" | 0.812 | yes |
| **"what is the leave policy?" vs "సెలవు విధానం ఏమిటి?"** | **0.808** | **yes — same question** |

The two classes overlap. A genuinely *different* question (sick leave, 0.860) scores **higher**
than a true cross-language paraphrase (0.808). No single threshold separates them — this is a
property of the embedding space, not a tuning failure.

Given that, the choice is which error to prefer. A wrong cache hit serves a confidently wrong
answer with no indication anything went wrong. A miss costs one API call. So the threshold sits at
**0.88**, just above the highest false pair, biasing hard toward correctness.

**The consequence, stated plainly: cross-language paraphrases do not hit the cache.** Asking the
same question in Telugu and then in English costs two generations. Lowering the threshold to 0.86
would catch some of those, but would also start serving "leave policy" answers to "sick leave
policy" questions. Fixing this properly needs a cheap LLM equivalence check on near-threshold
pairs — not a looser number.

Storage is a JSON file capped at 500 entries, trimmed oldest-first. `lookup()` and `store()` are
the seam: swapping in Redis or a vector DB changes nothing outside this module.

### 2.5 Streaming (SSE)

`app/routers/chat.py` → `frontend/script.js`

Replies stream token by token over Server-Sent Events. The endpoint is `POST /chat/stream`, which
is why `EventSource` is not used on the client — `EventSource` can only issue GET and cannot send a
JSON body. Instead the browser POSTs with `fetch()` and parses `text/event-stream` off the
`ReadableStream` reader manually, splitting on the blank-line frame separator and holding any
trailing partial frame in a buffer until the rest arrives.

Frame protocol — `meta` → `token`* → `done`, or `meta` → `error`:

| Event | Payload |
|---|---|
| `meta` | detection result, `cache_hit`, `cache_similarity`, `rag_chunks[]` (source, score, preview), `returning_user` |
| `token` | `{"t": "<text fragment>"}` |
| `done` | `{"cached": bool, "chars": int}` |
| `error` | `{"message": str, "kind": "rate_limit" \| "not_configured" \| "llm_error"}` |

`meta` is sent before any token so the UI can update the language badge and citation state
immediately, before text starts arriving. Nothing is buffered server-side: each chunk from
`llm.stream_chat()` is yielded the moment it arrives.

Response headers include `X-Accel-Buffering: no`, which stops nginx-style proxies (and App Runner)
from buffering the response — a proxy that buffers silently destroys streaming while leaving every
test that checks *content* passing.

Cache hits are re-chunked into 24-character pieces with a 12ms delay so a replay animates like a
live response. No tokens are billed for that path — it never calls Gemini.

Errors are delivered as SSE frames, not HTTP error codes. Once a `200` and the stream have begun,
you cannot go back and change the status, so a mid-stream rate limit arrives as
`event: error` with `kind: "rate_limit"`.

### 2.6 Manual language override

`app/services/lang_detect.py` → `app/routers/chat.py`

Detection is good, not infallible, so the sidebar has a language picker. "Auto-detect" is the
default and is exactly the behavior described above; picking anything else pins the language for
that conversation and skips detection entirely — including the stage-2 LLM call, so an overridden
turn costs one fewer Gemini request.

The options are not free text. `VALID_OVERRIDES` whitelists the nine combinations the detector can
actually produce (`en`, `te`, `hi`, `te-rom`, `hi-rom`, and the four `+ English` pairs), and the UI
builds its dropdown from `GET /chat/languages` so the list cannot drift from what the server
accepts. That whitelist is load-bearing rather than cosmetic: `prompts._describe_languages()` falls
back to the raw code for anything it does not recognise, so an unvalidated override string would be
client-controlled text inside the system prompt. Unknown values are logged and ignored, falling
back to auto-detect.

Two deliberate limits on what an override changes:

- **Register is still detected from the text.** The override is about language; formality is
  orthogonal, and someone writing politely in pinned Telugu should not be flattened.
- **Overridden turns do not update the user profile.** The profile means "what this user actually
  writes in", inferred from observed text. Folding an explicit pick into it would let one
  conversation's setting bias auto-detection everywhere else.

### 2.7 Persistent memory and saved conversations

`app/services/memory.py`

One stdlib `sqlite3` file holds two things, both keyed by a browser-generated `user_id` stored in
`localStorage` — no accounts, no login flow.

**`user_profile`** — cross-session language memory.

Each turn folds into rolling `Counter`s of observed languages and registers. `get_profile()`
returns the two most-used language codes as `preferred_languages`.

The profile is used narrowly, and that restraint is deliberate. In `lang_detect.detect()`, the
remembered pair is applied **only** when the current message produced plain English at confidence
below 0.7 — a tie-break for a short ambiguous message like "ok" or "thanks", never an override of
positive evidence in the text. If you write in Telugu, you get Telugu, regardless of history.
Memory failures are logged and swallowed; a chat is never failed over a nice-to-have.

**`conversation` / `conversation_message`** — saved chats, so history survives a refresh and the
sidebar can switch between past conversations.

- The `conversation_id` is generated **client-side**, like `user_id`, so a new chat is usable with
  no round-trip. The row is created **lazily** on the first successful turn, which means a chat you
  opened but never used leaves nothing behind.
- Turns are persisted after the reply completes, on both the live and cache-replay paths, and never
  on an error path — a failed turn would otherwise leave an orphan user message the client does not
  have.
- Titles are the first user message truncated to 60 characters on a word boundary. No LLM call: it
  would add latency and quota cost to every new conversation for a cosmetic string. Scripts without
  spaces (Telugu, Devanagari) fall through to a hard cut.
- **The language override lives on the conversation row, not the session.** That is what makes the
  two features compose: opening an old chat restores its language mode along with its messages, so
  a conversation you pinned to Tenglish stays Tenglish even if the session has since moved to
  auto-detect.
- "Clear this chat" empties one conversation but keeps it (and its language mode), resetting the
  title so the next message re-titles it. Deleting is the ✕ on its row — a separate, confirmed
  action.

Every conversation route is scoped by `user_id`. That is **not** authentication — `user_id` is a
client-supplied value and anyone can send any id — it only stops one browser's list from showing
another's. See §9.

### 2.8 Voice input and output

`frontend/script.js`

Both use the browser's Web Speech API, and support is genuinely uneven — so every control
feature-detects and **hides itself** rather than presenting a button that silently does nothing.

**Speech recognition** (mic button) is Chrome/Edge/Safari only, webkit-prefixed in practice. The
locale is `en-IN`: it handles Indian-accented English *and* the English half of code-mixed speech,
whereas `te-IN` recognition is unavailable on most platforms. Permission denial produces a
readable message rather than a silent no-op.

**Text-to-speech** comes in two forms sharing one implementation:

- the **global toggle** in the composer, which auto-reads each reply as it finishes streaming;
- a **per-message play button** in the chip row under every assistant reply, for replaying any
  message on demand — including ones from earlier in the conversation.

Both route through the same `speak()` function, so voice selection is identical. A voice is chosen
from the language of the message being read — `te-IN`, `hi-IN`, or `en-IN` — then degrades in
order: exact locale → same base language → any English → browser default. Telugu voices are absent
on most desktops, so this commonly lands on English, and the status bar says so explicitly instead
of quietly reading Telugu text with an English voice.

Only one message is ever audible: starting playback cancels whatever was already speaking and
resets that message's button. Each button reads its *own* message's detected language, so replaying
an older Telugu answer still picks a Telugu voice after the conversation has moved on to English.

---

## 3. Request flow: what happens when you press Send

A single trace, from keystroke to streamed reply.

```
Browser (frontend/script.js)
  │  POST /chat/stream  {message, history[-10:], user_id}
  ▼
chat.py :: _event_stream()
  │
  ├─1─ memory.get_profile(user_id) ......... sqlite3 → preferred_languages, returning?
  │
  ├─2─ lang_detect.detect(message) ......... Unicode script pass (offline, always)
  │        └─ all-Latin? ──► classify_with_llm() ──► Gemini (JSON)
  │                             └─ failed? ──► offline keyword heuristic
  │
  ├─3─ rag.stats() → has_documents?
  │        └─ yes ──► rag.retrieve(message) ─► embed query ─► Chroma cosine top-4
  │
  ├─4─ cache.lookup(message, has_documents) ─► embed query ─► cosine vs stored
  │                                             ≥ 0.88 → hit
  │
  ├─5─ yield  event: meta   {detection, cache_hit, rag_chunks, returning_user}
  │    memory.record_turn(user_id, languages, register)
  │
  ├─6a─ CACHE HIT  ──► replay stored text in 24-char frames ──► event: done {cached:true}
  │                    (Gemini is never called)
  │
  └─6b─ MISS ──► prompts.build_system_prompt(detection, chunks, returning_user)
                    │   base rules + few-shot register examples
                    │   + detected language/register lines
                    │   + document excerpts (if any)
                    ▼
                 llm.stream_chat() ──► Gemini generate_content_stream
                    │
                    └─ per chunk: yield  event: token {"t": "..."}
                                          ▼
                       event: done {cached:false, chars:N}
                       cache.store(message, full_response, scope)
```

Client-side, `meta` updates the language badge and the metadata chips, each `token` is appended to
the bubble immediately (which is what makes it visibly stream), and on completion the citation
`<details>` block is attached listing every retrieved excerpt with its source file and similarity
score.

Every stage between 1 and 4 is wrapped so that failure degrades rather than kills the turn:
detection failure defaults to English, retrieval failure answers without context, cache failure is
a miss.

---

## 4. Tech stack, and why

| Choice | Why |
|---|---|
| **FastAPI** | Native async, which this needs — SSE streaming, concurrent Gemini calls. `StreamingResponse` handles SSE without extra machinery, and OpenAPI docs come free at `/docs`. |
| **Gemini `gemini-3.5-flash-lite`** | Genuinely free tier, no credit card. Flash-Lite has the most generous limits of the family. Strong on Indian languages, which matters more here than raw benchmark scores. |
| **`gemini-embedding-001`** | The load-bearing choice. Multilingual embeddings are what make cross-language retrieval work *without* a translation step — an English-only embedder would force translate-then-search. |
| **ChromaDB** | Embedded, persistent, zero infrastructure — no server to run, no account. Used purely as a vector store with externally supplied embeddings. |
| **SQLite (stdlib)** | Cross-session memory needs durable key-value storage; that is all. No dependency, survives restarts, lives on the same mounted volume as the vector store. |
| **NumPy** | Cosine similarity for the semantic cache. Already present transitively. |
| **Vanilla JS frontend, no framework** | The UI is one page. React/Vite would add a build step, a `node_modules` tree and a second deploy artifact to produce the same result. No build step also means the static files served *are* the source. |
| **`google-genai`** | The current Google Gen AI SDK. Not `google-generativeai`, which is retired. |

**One process serves both API and UI.** FastAPI mounts `frontend/` via `StaticFiles` at `/` in
`app/main.py` — mounted last, so it never shadows the API routes. One process, one port, one
container: no nginx, no CORS configuration, no second billable service. The trade-off is no CDN for
static assets, which is irrelevant at this scale.

---

## 5. Project structure

```
multilingual-ai-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py               FastAPI entrypoint; /health; mounts frontend/ at /
│   │   ├── core/
│   │   │   └── config.py         Pydantic settings, .env loading, path resolution,
│   │   │                         cache threshold + calibration data
│   │   ├── routers/
│   │   │   ├── chat.py           POST /chat/stream (SSE), conversations, languages,
│   │   │   │                     profile + cache endpoints
│   │   │   └── documents.py      upload, status, delete-one, clear-all
│   │   └── services/
│   │       ├── lang_detect.py    two-stage detector + register detection
│   │       ├── rag.py            extract → chunk → embed → Chroma → retrieve
│   │       ├── prompts.py        system prompt assembly + few-shot register examples
│   │       ├── cache.py          semantic cache (cosine, scoped, JSON-persisted)
│   │       ├── memory.py         sqlite3 user profiles + saved conversations
│   │       └── llm.py            ONLY module importing the Gemini SDK
│   ├── tests/
│   │   ├── test_lang_detect.py   prints detection for 6 representative inputs
│   │   └── test_streaming.py     proves SSE is token-by-token, not batched
│   ├── data/                     git-ignored: chroma/, semantic_cache.json, memory.sqlite3
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html                sidebar + chat shell
│   ├── style.css                 dark terminal-adjacent theme, single accent
│   └── script.js                 SSE parsing, upload, voice, drawer
├── sample_docs/                  two unrelated docs for testing retrieval
├── docker-compose.yml
├── .env.example
├── ARCHITECTURE.md               module map, request lifecycle, swap points
└── DEPLOY.md                     AWS App Runner, budget alerts first
```

The dependency direction is one-way: `routers → services → llm → SDK`. Routers speak plain dicts
and strings; `llm.py` is the only file that imports the Gemini SDK, so changing provider means
rewriting that one file.

---

## 6. Setup and running

### Python version

**Python 3.11 or 3.12 is required.** ChromaDB's dependency chain (`onnxruntime`, `pypika`) has no
Python 3.14 wheels, and `pypika` uses `ast.Str`, which 3.14 removed. Installation fails on 3.14.

### Get a free Gemini API key

1. Go to <https://aistudio.google.com/apikey> and sign in with a Google account.
2. Click **Create API key**. No credit card required.
3. Put it in `.env` as `GEMINI_API_KEY=...`.

> #### ⚠ Do not enable billing on that Google Cloud project
>
> This is the most common way to accidentally start paying. The Gemini free tier applies to
> projects **without** billing enabled. The moment a billing account is linked to the project your
> key belongs to, that project is silently promoted to the paid tier — same key, same code, no
> warning, and requests start being charged. If you need a paid project for something else, create
> a **separate** project for this key.

Quotas are **per project**, not per key — generating more keys in the same project does not raise
your ceiling. Exceeding one returns HTTP `429 RESOURCE_EXHAUSTED`, which the app catches and
surfaces as a readable message rather than a traceback.

> **Retired models still appear in the model list.** `gemini-2.5-flash-lite` is returned by
> `client.models.list()` but responds `404 — no longer available to new users` for recently created
> keys. This project defaulted to it originally and had to be changed. If you hit that 404, set
> `GEMINI_MODEL=gemini-flash-lite-latest` (a floating alias) or list what your key can actually
> use:
>
> ```python
> from google import genai
> for m in genai.Client(api_key="...").models.list():
>     print(m.name, m.supported_actions)
> ```

### Local run

```bash
cd multilingual-ai-assistant
cp .env.example .env            # Windows: copy .env.example .env
#   then edit .env and paste your key into GEMINI_API_KEY

cd backend
python3.11 -m venv venv         # Windows: py -3.11 -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

On Windows you can skip activating the venv and call the interpreter directly:

```powershell
cd "multilingual-ai-assistant\backend"
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Then open **<http://127.0.0.1:8000>**.

> **Open the URL, not the file.** The UI is served by the same process as the API. Double-clicking
> `frontend/index.html` opens it as `file://`, where its stylesheet, script and every API call fail
> — the page renders as unstyled plain text. The app detects this and shows a warning bar telling
> you to use the URL.

### Where `.env` goes

The project root, beside `docker-compose.yml`. `backend/.env` also works — `config.py` checks both,
and the root file wins if both exist.

Overridable settings (all optional, defaults in `app/core/config.py`): `GEMINI_MODEL`,
`EMBED_MODEL`, `DATA_DIR`, `CHUNK_SIZE` (900), `CHUNK_OVERLAP` (150), `RAG_TOP_K` (4),
`CACHE_SIMILARITY_THRESHOLD` (0.88).

### Docker

```bash
cd multilingual-ai-assistant
cp .env.example .env    # then add your key
docker compose up --build
```

Open <http://127.0.0.1:8000>. Stop with `Ctrl+C` or `docker compose down`. The vector store, cache
and user memory persist in the `assistant-data` volume; `docker compose down -v` deletes them.

> **Honest caveat: the Docker path was never build-tested.** Docker was not installed on the
> machine this was built on. `Dockerfile` and `docker-compose.yml` are written and reviewed — the
> build context is the project root because the image needs both `backend/` and `frontend/`, the
> paths line up with what `config.py` expects, and the container runs as a non-root user — but they
> have **not** been built or run. The local non-Docker path *was* fully tested. Treat Docker as
> unverified.

---

## 7. API reference

Generated from the live OpenAPI schema at `/openapi.json`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat/stream` | Chat, streamed as SSE. Body: `{message, history[], user_id, conversation_id, language_override, use_rag, use_cache}`. Frames: `meta` → `token`* → `done` \| `error` |
| `GET` | `/chat/languages` | The nine pinnable language modes; the UI's dropdown is built from this |
| `GET` | `/chat/conversations?user_id=` | Saved chats: id, title, updated_at, message_count, language_override |
| `GET` | `/chat/conversations/{id}?user_id=` | One conversation's full history + its language mode |
| `PATCH` | `/chat/conversations/{id}?user_id=` | Set `language_override` without sending a message. `400` on an unknown option; `404` if the chat has no messages yet |
| `DELETE` | `/chat/conversations/{id}/messages?user_id=` | Empty a conversation, keep it ("Clear this chat") |
| `DELETE` | `/chat/conversations/{id}?user_id=` | Delete a conversation and its messages |
| `GET` | `/chat/profile?user_id=` | Remembered language pair and message count for a returning user |
| `DELETE` | `/chat/profile?user_id=` | Forget a user's profile |
| `GET` | `/chat/cache/stats` | Cache entry counts (`total`, `plain`, `rag`) |
| `DELETE` | `/chat/cache` | Empty the semantic cache |
| `POST` | `/documents/upload` | Ingest `.txt` `.md` `.pdf` `.csv` `.json`, max 5 MB. Returns `{doc_id, filename, chunks, characters}` |
| `GET` | `/documents/status` | What is in the vector store: total chunks + per-document rollup |
| `DELETE` | `/documents/{doc_id}` | Remove one document's chunks. `404` if the id is unknown |
| `DELETE` | `/documents` | Drop every stored chunk |
| `GET` | `/health` | Liveness, plus `llm_configured` (a boolean — never the key) and the configured model names |

Interactive docs at <http://127.0.0.1:8000/docs>.

Upload errors are typed rather than generic: `400` unsupported type / empty file / no extractable
text, `413` too large, `429` rate limited, `503` no API key, `502` other provider failure.

---

## 8. What's verified, and what isn't

Measured on this build against the live Gemini API. This table is deliberately not padded — see
section 9 for the other side.

| Check | Status | Evidence |
|---|---|---|
| Token-by-token SSE streaming | ✅ verified | `tests/test_streaming.py` — many frames spread over a non-zero delivery window; first text visible well before total wall clock |
| Two-stage language detection, 6 cases | ✅ verified | `tests/test_lang_detect.py` — script pass for native script, LLM classifier for romanized, correct `method` reported per case |
| Cross-language retrieval, 2 documents | ✅ verified | 4/4 queries ranked the correct document first, with two unrelated docs loaded |
| Script fidelity | ✅ verified | Telugu-script question → Telugu-script answer (574 Telugu vs 72 Latin characters) |
| Register preservation | ✅ verified | Casual Tenglish → *"…padutundi, bro!"*; formal English → *"Sir, …"* |
| Semantic cache | ✅ verified | Paraphrase hit at 0.886 (3.0s replay vs 5.7s live); a distinct question correctly missed |
| Persistent memory | ✅ verified | Remembered language pair survived a server restart |
| 429 handling | ✅ verified | A real 429 from Google produced a clean SSE error frame, no crash |
| Frontend renders over HTTP | ✅ verified | Headless-browser screenshot: sidebar, theme, badges, streaming bubbles all render; assets return `200` |
| Docker build | ⏸ **not run** | Docker was not installed on the build machine |
| Automated test suite | ⏸ **none** | `tests/` holds two runnable diagnostic scripts, not a pytest suite. No CI, no assertions, no coverage |
| Load / concurrency | ⏸ **not tested** | Single-user local use only |
| Browsers other than Chromium | ⏸ **not tested** | Voice paths especially |

On the two test scripts: `test_lang_detect.py` and `test_streaming.py` *print* what happened and,
for streaming, give a verdict. They are diagnostics you read, not assertions that pass or fail in
CI. Calling them a test suite would overstate what exists.

### Trying it yourself

**Cross-language retrieval.** Upload **both** `sample_docs/company_policy.txt` and
`sample_docs/cricket_notes.txt` by dragging them onto the sidebar dropzone. Two *unrelated*
documents is the point — with only one loaded, top-k retrieval returns everything and proves
nothing. Then ask:

- Telugu script: `సెలవు విధానం ఏమిటి?` ("what is the leave policy?")
- Tenglish: `Pitch prepare cheyyadaniki enni days padutundi?`

Each should pull chunks from the *correct* file. Expand the sources block under the reply to see
which excerpts were used, from which document, with similarity scores.

**Detection and streaming.**

```bash
cd backend
python tests/test_lang_detect.py     # runs with or without an API key
python tests/test_streaming.py       # needs the server running
```

---

## 9. Known limitations

**Cross-language paraphrases miss the cache.** By design, explained in §2.4: the same question in
Telugu and English scores 0.808, below the 0.88 threshold, so it costs a second generation. The
alternative — lowering the threshold — would start serving wrong answers to similar-but-different
questions. The real fix is an LLM equivalence check on near-threshold pairs.

**Docker is unverified.** Written and reviewed, never built. See §6.

**Voice support is uneven and mostly out of the app's control.** Speech recognition is
Chrome/Edge/Safari only. `te-IN` recognition is unavailable on most platforms, so the mic runs at
`en-IN`. Telugu TTS voices are absent on most desktops, so playback commonly falls back to Hindi or
English. The app detects and reports this rather than hiding it, but cannot fix it.

**Stage 2 detection costs an API call.** Every all-Latin message hits Gemini for classification
before the actual answer is generated. That is two calls per turn against a 15 req/min free tier.
The offline heuristic exists as a fallback, but it is a keyword matcher and less accurate.

**No OCR.** Scanned or image-only PDFs yield no text and are rejected explicitly.

**Conversations are readable by anyone who guesses a `user_id`.** Every conversation route is
scoped by `user_id`, but that value is generated by the browser and trusted as sent — there is no
auth. Saved chats are stored in plain text in SQLite. Fine for a local project; do not expose this
publicly with real content in it.

**A refresh opens a new chat rather than resuming the last one.** History is persisted and
reachable from the sidebar, but the app does not remember which conversation you had open.

**Prompt context is still the client's last 10 turns.** Saving history did not change what gets
sent to the model: the browser posts `history[-10:]` with each request. A conversation loaded from
the sidebar therefore carries its full transcript into the prompt only up to that window.

**Register detection is lexical.** A keyword-and-punctuation heuristic, not a model. Sarcasm,
mixed signals, and formality carried by grammar rather than vocabulary will be misread.

**Single-process, single-node.** The semantic cache is an in-memory list persisted to JSON, guarded
by a thread lock. Fine for one container; two replicas would each hold their own copy. Chroma is
embedded, and SQLite writes are serialized by one lock. Scaling means externalizing all three —
which is why each sits behind a narrow function-level seam.

**No authentication.** `user_id` is a `localStorage` value generated by the browser and trusted as
sent. Anyone can pass any id. Fine for a local project; not something to expose publicly as-is.

**Document scope is global.** All uploaded documents live in one Chroma collection shared by every
user. There is no per-user isolation.

---

## Security

- Secrets come from environment variables only. `GEMINI_API_KEY` is read in `app/core/config.py`
  and used only inside `app/services/llm.py`.
- The key is never sent to the frontend. `/health` reports `llm_configured: true|false` — a
  boolean, not the value.
- `.env` is git-ignored; only `.env.example`, with placeholders, is committed.
- No key is baked into the Docker image; it is injected at runtime via `env_file` locally or the
  App Runner configuration in production.

## Further documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — module map, request lifecycle, and which interfaces are
  designed to be swapped.
- **[DEPLOY.md](DEPLOY.md)** — AWS App Runner deployment, with budget alerts set up first.
