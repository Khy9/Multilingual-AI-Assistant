"""Persistent multi-session user memory.

Remembers which language pair a user actually writes in, so a returning user gets
sensible defaults before they have typed enough for detection to be confident.

Storage is stdlib sqlite3 — no extra dependency, survives restarts, and lives on
the same mounted volume as the vector store. At larger scale this would become a
real database behind the same four functions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import Counter
from typing import Any

from app.core.config import get_settings

log = logging.getLogger(__name__)

# sqlite3 connections are not safe to share across threads without care; a single
# lock is more than adequate at this scale.
_lock = threading.Lock()
_initialised = False

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_profile (
    user_id            TEXT PRIMARY KEY,
    language_counts    TEXT NOT NULL DEFAULT '{}',
    register_counts    TEXT NOT NULL DEFAULT '{}',
    message_count      INTEGER NOT NULL DEFAULT 0,
    last_seen          TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _connect() -> sqlite3.Connection:
    global _initialised
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.memory_db)
    conn.row_factory = sqlite3.Row
    if not _initialised:
        conn.executescript(_SCHEMA)
        conn.commit()
        _initialised = True
    return conn


def get_profile(user_id: str) -> dict[str, Any]:
    """Return the stored profile, or an empty-but-valid one for a new user."""
    empty = {
        "user_id": user_id,
        "preferred_languages": [],
        "preferred_register": None,
        "message_count": 0,
        "returning": False,
    }
    if not user_id:
        return empty

    try:
        with _lock, _connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_profile WHERE user_id = ?", (user_id,)
            ).fetchone()
    except sqlite3.Error as exc:  # pragma: no cover - disk/permission problems
        log.warning("Could not read profile for %s: %s", user_id, exc)
        return empty

    if row is None:
        return empty

    language_counts = Counter(json.loads(row["language_counts"]))
    register_counts = Counter(json.loads(row["register_counts"]))

    # "Preferred" = the languages this user has actually used most, capped at two
    # so we describe a pair (e.g. Telugu-romanized + English), not a long tail.
    preferred = [code for code, _ in language_counts.most_common(2)]
    preferred_register = register_counts.most_common(1)[0][0] if register_counts else None

    return {
        "user_id": user_id,
        "preferred_languages": preferred,
        "preferred_register": preferred_register,
        "message_count": row["message_count"],
        "returning": row["message_count"] > 0,
    }


def record_turn(user_id: str, languages: list[str], register: str) -> None:
    """Fold one observed message into the user's rolling profile."""
    if not user_id or not languages:
        return

    try:
        with _lock, _connect() as conn:
            row = conn.execute(
                "SELECT language_counts, register_counts, message_count "
                "FROM user_profile WHERE user_id = ?",
                (user_id,),
            ).fetchone()

            language_counts = Counter(json.loads(row["language_counts"])) if row else Counter()
            register_counts = Counter(json.loads(row["register_counts"])) if row else Counter()
            message_count = row["message_count"] if row else 0

            language_counts.update(languages)
            if register:
                register_counts.update([register])

            conn.execute(
                """
                INSERT INTO user_profile
                    (user_id, language_counts, register_counts, message_count, last_seen)
                VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(user_id) DO UPDATE SET
                    language_counts = excluded.language_counts,
                    register_counts = excluded.register_counts,
                    message_count   = excluded.message_count,
                    last_seen       = excluded.last_seen
                """,
                (
                    user_id,
                    json.dumps(dict(language_counts)),
                    json.dumps(dict(register_counts)),
                    message_count + 1,
                ),
            )
            conn.commit()
    except sqlite3.Error as exc:  # pragma: no cover
        # Memory is a nice-to-have; never fail a chat request over it.
        log.warning("Could not record turn for %s: %s", user_id, exc)


def reset_profile(user_id: str) -> None:
    try:
        with _lock, _connect() as conn:
            conn.execute("DELETE FROM user_profile WHERE user_id = ?", (user_id,))
            conn.commit()
    except sqlite3.Error as exc:  # pragma: no cover
        log.warning("Could not reset profile for %s: %s", user_id, exc)
