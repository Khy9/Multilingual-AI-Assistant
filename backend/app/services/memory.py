"""Persistent user memory and saved conversations.

Two things live here, in one SQLite file on the same mounted volume as the vector
store:

  1. user_profile   - which language pair a user actually writes in, so a returning
                      user gets sensible defaults before they have typed enough for
                      detection to be confident.
  2. conversation / conversation_message
                    - saved chats, so history survives a page refresh and a user can
                      switch between past conversations. Each conversation also owns
                      its own language mode (auto-detect, or a manual override), which
                      is why the override is stored HERE rather than per session:
                      switching conversations must switch language mode with it.

Storage is stdlib sqlite3 — no extra dependency, survives restarts. At larger scale
this becomes a real database behind the same functions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import Counter
from contextlib import contextmanager
from typing import Any, Iterator

from app.core.config import get_settings

log = logging.getLogger(__name__)

# sqlite3 connections are not safe to share across threads without care; a single
# lock is more than adequate at this scale.
_lock = threading.Lock()
_initialised = False

DEFAULT_TITLE = "New chat"
TITLE_MAX_CHARS = 60
# Bounds a single listing response; far beyond anything a real user accumulates.
MAX_LISTED_CONVERSATIONS = 200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_profile (
    user_id            TEXT PRIMARY KEY,
    language_counts    TEXT NOT NULL DEFAULT '{}',
    register_counts    TEXT NOT NULL DEFAULT '{}',
    message_count      INTEGER NOT NULL DEFAULT 0,
    last_seen          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversation (
    conversation_id    TEXT PRIMARY KEY,
    user_id            TEXT NOT NULL,
    title              TEXT NOT NULL DEFAULT 'New chat',
    -- NULL means auto-detect. Otherwise a key from lang_detect.VALID_OVERRIDES.
    language_override  TEXT,
    created_at         TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_conversation_user
    ON conversation (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS conversation_message (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id    TEXT NOT NULL,
    role               TEXT NOT NULL,
    content            TEXT NOT NULL,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_message_conversation
    ON conversation_message (conversation_id, id);
"""


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success, and always close it.

    The `with conn` block commits or rolls back but does NOT close, so the close
    lives in the finally — otherwise connections accumulate for the process
    lifetime, which the conversation endpoints would make noticeable.
    """
    global _initialised
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(settings.memory_db)
    conn.row_factory = sqlite3.Row
    try:
        if not _initialised:
            # Every statement is CREATE ... IF NOT EXISTS, so this doubles as the
            # migration path: existing databases gain the new tables untouched.
            conn.executescript(_SCHEMA)
            conn.commit()
            _initialised = True
        with conn:
            yield conn
    finally:
        conn.close()


# --- User profile -------------------------------------------------------------


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
        with _lock, _db() as conn:
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
    """Fold one observed message into the user's rolling profile.

    Only ever called for AUTO-DETECTED turns. A manual language override is a
    per-conversation choice, not evidence about how this user writes, so folding it
    in here would let one overridden conversation bias auto-detection everywhere
    else.
    """
    if not user_id or not languages:
        return

    try:
        with _lock, _db() as conn:
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
    except sqlite3.Error as exc:  # pragma: no cover
        # Memory is a nice-to-have; never fail a chat request over it.
        log.warning("Could not record turn for %s: %s", user_id, exc)


def reset_profile(user_id: str) -> None:
    try:
        with _lock, _db() as conn:
            conn.execute("DELETE FROM user_profile WHERE user_id = ?", (user_id,))
    except sqlite3.Error as exc:  # pragma: no cover
        log.warning("Could not reset profile for %s: %s", user_id, exc)


# --- Conversations ------------------------------------------------------------


def make_title(text: str) -> str:
    """Derive a conversation title from its first user message.

    Plain truncation on a word boundary — deliberately no LLM call, which would add
    latency and quota cost to every new conversation for a cosmetic string.
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return DEFAULT_TITLE
    if len(cleaned) <= TITLE_MAX_CHARS:
        return cleaned

    cut = cleaned[:TITLE_MAX_CHARS]
    space = cut.rfind(" ")
    # Only break on a word boundary if that leaves a reasonable amount of text;
    # scripts without spaces (Telugu, Devanagari) fall through to a hard cut.
    if space >= TITLE_MAX_CHARS // 2:
        cut = cut[:space]
    return cut.rstrip(" ,.;:-") + "…"


def append_turn(
    conversation_id: str,
    user_id: str,
    user_message: str,
    assistant_message: str,
    *,
    language_override: str | None = None,
) -> None:
    """Persist one completed exchange, creating the conversation if it is new.

    Conversations are created lazily here rather than by an explicit endpoint, so a
    chat the user opened but never sent anything in leaves no empty row behind.
    """
    if not conversation_id or not user_id:
        return

    try:
        with _lock, _db() as conn:
            conn.execute(
                """
                INSERT INTO conversation
                    (conversation_id, user_id, title, language_override)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    updated_at        = datetime('now'),
                    language_override = excluded.language_override,
                    -- Keep a title once set, except after a clear, which resets it
                    -- to the default so the next message re-titles the chat.
                    title = CASE
                        WHEN conversation.title = ? THEN excluded.title
                        ELSE conversation.title
                    END
                """,
                (
                    conversation_id,
                    user_id,
                    make_title(user_message),
                    language_override,
                    DEFAULT_TITLE,
                ),
            )
            conn.executemany(
                "INSERT INTO conversation_message (conversation_id, role, content) "
                "VALUES (?, ?, ?)",
                [
                    (conversation_id, "user", user_message),
                    (conversation_id, "assistant", assistant_message),
                ],
            )
    except sqlite3.Error as exc:  # pragma: no cover
        log.warning("Could not append turn to conversation %s: %s", conversation_id, exc)


def list_conversations(user_id: str) -> list[dict[str, Any]]:
    """Every saved conversation for one user, most recently updated first."""
    if not user_id:
        return []

    try:
        with _lock, _db() as conn:
            rows = conn.execute(
                """
                SELECT c.conversation_id, c.title, c.language_override,
                       c.created_at, c.updated_at,
                       (SELECT COUNT(*) FROM conversation_message m
                         WHERE m.conversation_id = c.conversation_id) AS message_count
                  FROM conversation c
                 WHERE c.user_id = ?
                 ORDER BY c.updated_at DESC, c.created_at DESC
                 LIMIT ?
                """,
                (user_id, MAX_LISTED_CONVERSATIONS),
            ).fetchall()
    except sqlite3.Error as exc:  # pragma: no cover
        log.warning("Could not list conversations for %s: %s", user_id, exc)
        return []

    return [dict(row) for row in rows]


def get_conversation(conversation_id: str, user_id: str) -> dict[str, Any] | None:
    """One conversation with its full message history, or None if it is not this
    user's (there is no auth, so every read is scoped by user_id)."""
    if not conversation_id or not user_id:
        return None

    try:
        with _lock, _db() as conn:
            row = conn.execute(
                "SELECT conversation_id, title, language_override, created_at, updated_at "
                "FROM conversation WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if row is None:
                return None

            messages = conn.execute(
                "SELECT role, content, created_at FROM conversation_message "
                "WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
    except sqlite3.Error as exc:  # pragma: no cover
        log.warning("Could not read conversation %s: %s", conversation_id, exc)
        return None

    return {**dict(row), "messages": [dict(message) for message in messages]}


def clear_conversation_messages(conversation_id: str, user_id: str) -> bool:
    """Empty one conversation without deleting it.

    This is what the UI's "Clear this chat" does: the conversation stays in the
    sidebar (and keeps its language mode), but its history is gone and the title
    resets so the next message re-titles it.
    """
    if not conversation_id or not user_id:
        return False

    try:
        with _lock, _db() as conn:
            owned = conn.execute(
                "SELECT 1 FROM conversation WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if owned is None:
                return False

            conn.execute(
                "DELETE FROM conversation_message WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.execute(
                "UPDATE conversation SET title = ?, updated_at = datetime('now') "
                "WHERE conversation_id = ?",
                (DEFAULT_TITLE, conversation_id),
            )
            return True
    except sqlite3.Error as exc:  # pragma: no cover
        log.warning("Could not clear conversation %s: %s", conversation_id, exc)
        return False


def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """Delete a conversation and its messages. Both tables are cleared explicitly
    rather than via ON DELETE CASCADE, which SQLite only honours when
    `PRAGMA foreign_keys=ON` is set on every connection."""
    if not conversation_id or not user_id:
        return False

    try:
        with _lock, _db() as conn:
            cursor = conn.execute(
                "DELETE FROM conversation WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
            if cursor.rowcount == 0:
                return False
            conn.execute(
                "DELETE FROM conversation_message WHERE conversation_id = ?",
                (conversation_id,),
            )
            return True
    except sqlite3.Error as exc:  # pragma: no cover
        log.warning("Could not delete conversation %s: %s", conversation_id, exc)
        return False


def set_language_override(
    conversation_id: str, user_id: str, language_override: str | None
) -> bool:
    """Persist a language-mode change made without sending a message.

    Returns False when the conversation does not exist yet — a chat with no messages
    has no row, and the client simply holds the choice until its first message
    creates one.
    """
    if not conversation_id or not user_id:
        return False

    try:
        with _lock, _db() as conn:
            cursor = conn.execute(
                "UPDATE conversation SET language_override = ?, updated_at = datetime('now') "
                "WHERE conversation_id = ? AND user_id = ?",
                (language_override, conversation_id, user_id),
            )
            return cursor.rowcount > 0
    except sqlite3.Error as exc:  # pragma: no cover
        log.warning("Could not set language override on %s: %s", conversation_id, exc)
        return False
