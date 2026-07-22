"""FastAPI entrypoint.

Design note: this single process serves BOTH the JSON/SSE API and the frontend as
static files. One container, one port, no CORS config, no separate nginx. See
README.md for why that trade-off suits a free-tier deployment.
"""

import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import FRONTEND_DIR, get_settings
from app.routers import chat, documents

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

settings = get_settings()

app = FastAPI(
    title="Multilingual AI Assistant",
    description=(
        "A RAG-powered assistant for code-mixed regional language input "
        "(Tenglish / Hinglish), with cross-language document retrieval."
    ),
    version="1.0.0",
)

app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/health", tags=["system"])
async def health() -> JSONResponse:
    """Liveness probe. Also reports whether a Gemini key is present WITHOUT
    revealing any part of the key itself."""
    return JSONResponse(
        {
            "status": "ok",
            "llm_configured": settings.llm_configured,
            "chat_model": settings.gemini_model,
            "embed_model": settings.embed_model,
        }
    )


# --- Frontend (mounted last so it never shadows the API routes) ---------------
if FRONTEND_DIR.is_dir():

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")
else:  # pragma: no cover - only hit if the image was built without the frontend
    log.warning("Frontend directory not found at %s; serving API only.", FRONTEND_DIR)
