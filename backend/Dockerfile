# Python 3.11 is required, not incidental: ChromaDB's dependencies (onnxruntime,
# pypika) have no wheels for 3.14, and pypika uses ast.Str which 3.14 removed.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so edits to source code don't invalidate the pip layer.
COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/app ./app
COPY backend/tests ./tests

# The API also serves the UI, so the frontend must be inside the image. The path
# matches what config.py expects: FRONTEND_DIR is the sibling of the backend dir.
COPY frontend /frontend

# Persisted data (Chroma, semantic cache, memory DB) lives here. docker-compose
# mounts a volume over it so it survives container restarts.
RUN mkdir -p /app/data

# Run as a non-root user, and make sure it owns the writable directory.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app/data
USER appuser

EXPOSE 8000

# NOTE: no secrets are baked in. GEMINI_API_KEY arrives at runtime, via
# docker-compose env_file locally or the App Runner config in production.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
