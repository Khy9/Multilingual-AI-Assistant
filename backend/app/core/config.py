"""Application configuration.

Every secret and tunable comes from the environment (or a local .env file that is
git-ignored). Nothing is hardcoded here, and GEMINI_API_KEY is never exposed to
any route that the frontend can reach.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Both locations are supported so the project works whether you keep .env
        # beside docker-compose.yml (project root, the documented default) or
        # inside backend/. Later entries win, so the project-root file takes
        # precedence if both exist.
        env_file=(BACKEND_DIR / ".env", PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Provider credentials -------------------------------------------------
    # Empty string rather than a hard failure: the app must still boot without a
    # key so /health, the UI and the language-detection fallbacks stay testable.
    gemini_api_key: str = ""

    # --- Models ---------------------------------------------------------------
    # Flash-Lite has the most generous free-tier limits. Note that older Flash-Lite
    # generations get retired for NEW API keys while still appearing in
    # client.models.list() — gemini-2.5-flash-lite now returns 404 for new keys.
    # If this model ever 404s, run: client.models.list() and pick a current
    # flash-lite, or use the floating alias gemini-flash-lite-latest.
    gemini_model: str = "gemini-3.5-flash-lite"
    embed_model: str = "gemini-embedding-001"

    # --- Storage --------------------------------------------------------------
    # Single directory so docker-compose only needs to mount one volume.
    data_dir: Path = BACKEND_DIR / "data"

    # --- RAG tuning -----------------------------------------------------------
    chunk_size: int = 900
    chunk_overlap: int = 150
    rag_top_k: int = 4

    # --- Semantic cache -------------------------------------------------------
    # Cosine similarity above this counts as "the same question, asked again".
    #
    # Calibrated empirically against gemini-embedding-001, not guessed. Measured
    # cosine similarities:
    #     0.946  "third quarter revenue?"     vs "Q3 revenue figures?"      (same)
    #     0.934  "notice period for director" vs "how long must a director…" (same)
    #     0.886  "what is the leave policy?"  vs "tell me about the leave policy"
    #     0.860  "what is the leave policy?"  vs "what is the SICK leave policy?"  <-- DIFFERENT
    #     0.812  "how many days annual leave" vs "yearly paid vacation entitlement"
    #     0.808  "what is the leave policy?"  vs "సెలవు విధానం ఏమిటి?"
    #
    # The classes overlap: a distinct question (sick leave, 0.860) scores higher
    # than a true cross-language paraphrase (0.808). No threshold separates them
    # perfectly, so we bias toward correctness — 0.88 sits just above the highest
    # false pair. A wrong cache hit serves a WRONG answer; a miss costs one API
    # call. Consequence: cross-language paraphrases do NOT hit the cache. Fixing
    # that properly needs a cheap LLM equivalence check on near-threshold pairs,
    # not a looser number.
    cache_similarity_threshold: float = 0.88

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def cache_file(self) -> Path:
        return self.data_dir / "semantic_cache.json"

    @property
    def memory_db(self) -> Path:
        return self.data_dir / "memory.sqlite3"

    @property
    def llm_configured(self) -> bool:
        return bool(self.gemini_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
