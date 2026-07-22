"""Document upload and RAG collection status."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services import llm, rag

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv", ".json"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB — plenty of text, gentle on free-tier quota


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Chunk, embed and store an uploaded document for cross-language retrieval."""
    filename = file.filename or "upload"
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {extension or '(none)'}. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(raw) // 1024} KB; the limit is {MAX_UPLOAD_BYTES // 1024} KB.",
        )

    try:
        summary = await rag.ingest_document(filename, raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except llm.RateLimitError as exc:
        # 429 passed through honestly so the UI can tell the user to wait.
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except llm.NotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Ingestion failed for %s", filename)
        raise HTTPException(status_code=500, detail=f"Could not ingest document: {exc}") from exc

    return JSONResponse({"status": "ok", **summary})


@router.get("/status")
async def documents_status() -> JSONResponse:
    """What is currently in the vector store — drives the UI's document badge."""
    return JSONResponse(rag.stats())


@router.delete("")
async def clear_documents() -> JSONResponse:
    rag.clear()
    return JSONResponse({"status": "cleared"})


@router.delete("/{doc_id}")
async def delete_document(doc_id: str) -> JSONResponse:
    """Remove a single uploaded document by the doc_id returned from /status."""
    try:
        removed = rag.delete_document(doc_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Could not delete document: {exc}") from exc
    if removed == 0:
        raise HTTPException(status_code=404, detail="No such document.")
    return JSONResponse({"status": "deleted", "doc_id": doc_id, "removed": removed})
