"""RAG: document ingestion and cross-language retrieval.

Cross-language retrieval, and why it needs no translation step
-------------------------------------------------------------
A naive pipeline would translate the query to the document's language before
searching. We don't. `gemini-embedding-001` is a MULTILINGUAL embedding model:
"revenue for the third quarter", "మూడవ త్రైమాసిక ఆదాయం" and "third quarter revenue
enta?" all map to nearby points in one shared vector space. So a Telugu or Tenglish
query retrieves the relevant ENGLISH chunks directly, and Gemini then answers in
the user's language using that English context.

Chroma is used purely as a vector store: we compute embeddings ourselves and pass
them to collection.add(embeddings=...). That deliberately bypasses Chroma's default
ONNX embedding function — it keeps the multilingual model in charge of retrieval
quality and avoids downloading a local ONNX model at first use.

Swap note: the public surface is ingest_document() / retrieve() / stats() / clear().
Replacing Chroma with FAISS, pgvector or Pinecone means rewriting this file only.
"""

from __future__ import annotations

import io
import logging
import os
import re
import uuid
from dataclasses import dataclass

# Must be set before chromadb is imported: its telemetry client is built at import
# time, and a posthog version mismatch otherwise spams "capture() takes 1
# positional argument but 3 were given" errors on every operation. We disable
# telemetry anyway; this just makes the opt-out apply early enough to be quiet.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.services import llm

log = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
_client: chromadb.ClientAPI | None = None


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


def _collection() -> chromadb.Collection:
    global _client
    settings = get_settings()
    if _client is None:
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    # embedding_function=None: we always supply vectors explicitly (see module docstring).
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def chunk_text(text: str, *, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph then sentence breaks.

    Overlap matters for retrieval: a fact split across a boundary would otherwise be
    unfindable from either side.
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Prefer a paragraph break, then a sentence end, within the last 40%.
            window_start = start + int(chunk_size * 0.6)
            breakpoint = text.rfind("\n\n", window_start, end)
            if breakpoint == -1:
                match = None
                for match in re.finditer(r"[.!?।]\s", text[window_start:end]):
                    pass  # keep the last match in the window
                if match:
                    breakpoint = window_start + match.end()
            if breakpoint > start:
                end = breakpoint

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def extract_text(filename: str, raw: bytes) -> str:
    """Pull plain text out of a .pdf, or decode a text file."""
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


async def ingest_document(filename: str, raw: bytes) -> dict:
    """Extract, chunk, embed and store a document. Returns an ingestion summary."""
    text = extract_text(filename, raw)
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError(
            f"No extractable text found in {filename!r}. "
            "Scanned/image-only PDFs need OCR, which this project does not include."
        )

    # One embedding call for the whole document rather than one per chunk — a lot
    # gentler on the free-tier request-per-minute limit.
    embeddings = await llm.embed_texts(chunks)

    doc_id = uuid.uuid4().hex[:12]
    collection = _collection()
    collection.add(
        ids=[f"{doc_id}-{index}" for index in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[
            {"source": filename, "doc_id": doc_id, "chunk_index": index}
            for index in range(len(chunks))
        ],
    )

    log.info("Ingested %s: %d chunks, %d characters", filename, len(chunks), len(text))
    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks": len(chunks),
        "characters": len(text),
    }


async def retrieve(query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
    """Fetch the chunks most relevant to `query`, regardless of language.

    Returns [] rather than raising if nothing is stored or embedding fails — the
    chat route then answers without document context instead of erroring.
    """
    settings = get_settings()
    top_k = top_k or settings.rag_top_k

    collection = _collection()
    if collection.count() == 0:
        return []

    try:
        query_embedding = await llm.embed_one(query)
    except llm.LLMError as exc:
        log.warning("Retrieval skipped (embedding failed): %s", exc)
        return []

    if not query_embedding:
        return []

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    chunks: list[RetrievedChunk] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        chunks.append(
            RetrievedChunk(
                text=document,
                source=str((metadata or {}).get("source", "unknown")),
                # Collection uses cosine space, so similarity = 1 - distance.
                score=round(1.0 - float(distance), 4),
            )
        )
    return chunks


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks for injection into the system prompt."""
    if not chunks:
        return ""
    parts = [
        f"[Excerpt {index}, from {chunk.source}]\n{chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(parts)


def stats() -> dict:
    """Per-document rollup for the sidebar: one row per uploaded file with its
    doc_id (needed for per-document delete) and its own chunk count."""
    try:
        collection = _collection()
        count = collection.count()
        docs: dict[str, dict] = {}
        if count:
            stored = collection.get(include=["metadatas"], limit=count)
            for metadata in stored.get("metadatas") or []:
                if not metadata:
                    continue
                doc_id = str(metadata.get("doc_id") or "")
                source = str(metadata.get("source") or "document")
                # Group by doc_id when present so two files sharing a name stay
                # distinct; fall back to filename for pre-doc_id data.
                key = doc_id or source
                entry = docs.setdefault(
                    key, {"doc_id": doc_id, "filename": source, "chunks": 0}
                )
                entry["chunks"] += 1
        documents = sorted(docs.values(), key=lambda d: d["filename"].lower())
        return {"chunks": count, "documents": documents, "has_documents": count > 0}
    except Exception as exc:  # noqa: BLE001 - status endpoint must never 500
        log.warning("Could not read RAG stats: %s", exc)
        return {"chunks": 0, "documents": [], "has_documents": False}


def delete_document(doc_id: str) -> int:
    """Remove every chunk belonging to one uploaded document.

    Returns the number of chunks removed (0 if the doc_id was unknown), so the
    route can answer 404 rather than silently succeeding.
    """
    collection = _collection()
    try:
        existing = collection.get(where={"doc_id": doc_id})
        removed = len(existing.get("ids") or [])
        if removed:
            collection.delete(where={"doc_id": doc_id})
        return removed
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not delete document %s: %s", doc_id, exc)
        raise


def clear() -> None:
    """Drop every stored chunk. Used by the UI's 'clear documents' control."""
    _collection()  # ensures _client is initialised
    assert _client is not None
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001 - already-absent collection is fine
        log.warning("Could not delete collection: %s", exc)
