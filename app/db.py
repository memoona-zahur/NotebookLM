"""Database access: connection pool, schema, and session-scoped queries.

Everything persistent lives in one Postgres: source records, chunk text, chunk
vectors (pgvector), chat messages, and session metadata. One store means one
backup and one thing to reason about at 3am.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

from . import config

_pool = None


@dataclass
class SessionRow:
    id: str
    name: str
    created_at: datetime
    updated_at: datetime


def dsn() -> str:
    return os.getenv("DATABASE_URL", config.DEFAULT_DATABASE_URL)


def pool():
    """Lazily built connection pool. Rebuilt when DATABASE_URL changes."""
    global _pool
    if _pool is None:
        from psycopg_pool import ConnectionPool

        _pool = ConnectionPool(
            dsn(),
            min_size=1,
            max_size=int(os.getenv("DB_POOL_MAX", "8")),
            timeout=10.0,
            open=True,
        )
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection() -> Iterator:
    """Borrow a connection and always give it back.

    `commit` rolls back on exception, so a failed request cannot leave a
    half-written source behind.
    """
    with pool().connection() as conn:
        with conn.transaction():
            yield conn


SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- A session is the unit of work: its own sources and its own chat history.
CREATE TABLE IF NOT EXISTS sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    -- At most one row may carry this, enforced by the partial unique index
    -- below. Identifying the default session by a flag rather than by name
    -- means a user can call their own session whatever they like without
    -- colliding with it.
    is_default  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE sessions ADD COLUMN IF NOT EXISTS is_default BOOLEAN NOT NULL DEFAULT FALSE;
CREATE UNIQUE INDEX IF NOT EXISTS sessions_one_default_idx
    ON sessions (is_default) WHERE is_default;

CREATE TABLE IF NOT EXISTS sources (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id    UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    kind          TEXT NOT NULL,
    pages         INTEGER NOT NULL DEFAULT 0,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    numeric_count INTEGER NOT NULL DEFAULT 0,
    storage_path  TEXT NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sources_session_idx ON sources(session_id);

-- Applied after the CREATE TABLE statements so a database created by an older
-- version picks the column up. IF NOT EXISTS keeps this safe to re-run.
ALTER TABLE sources ADD COLUMN IF NOT EXISTS storage_path TEXT NOT NULL DEFAULT '';

-- One row per indexed chunk. `embedding` is the pgvector column searched by
-- ANN index; `text` is kept so a retrieved row needs no second lookup.
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    source_id     UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    page          INTEGER NOT NULL DEFAULT 0,
    heading       TEXT NOT NULL DEFAULT '',
    text          TEXT NOT NULL,
    numeric_heavy BOOLEAN NOT NULL DEFAULT FALSE,
    embedding     vector({dim}) NOT NULL,
    UNIQUE (source_id, position)
);
CREATE INDEX IF NOT EXISTS chunks_source_idx ON chunks(source_id);

-- Chat history, per session. role is constrained so a row can never hold a
-- 'system' turn that would compete with the grounding instructions.
CREATE TABLE IF NOT EXISTS messages (
    id         BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_session_idx ON messages(session_id, id);
"""


def init_schema() -> None:
    """Apply the schema. Idempotent, so it is safe on every boot."""
    with pool().connection() as conn:
        conn.execute(SCHEMA.format(dim=config.EMBED_DIM))
        # HNSW gives approximate nearest-neighbour search over the vectors.
        # Created outside the DDL string because IF NOT EXISTS on an index that
        # already exists is still fine, but the vector() cast needs the column
        # to exist first.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops)"
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


def create_session(name: str) -> SessionRow:
    with connection() as conn:
        row = conn.execute(
            "INSERT INTO sessions (name) VALUES (%s) RETURNING id, name, created_at, updated_at",
            (name.strip() or "Untitled session",),
        ).fetchone()
    return SessionRow(*row)


def list_sessions() -> list[SessionRow]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT id, name, created_at, updated_at FROM sessions "
            "ORDER BY updated_at DESC"
        ).fetchall()
    return [SessionRow(*row) for row in rows]


def get_session(session_id: str) -> SessionRow | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT id, name, created_at, updated_at FROM sessions WHERE id = %s",
            (session_id,),
        ).fetchone()
    return SessionRow(*row) if row else None


def rename_session(session_id: str, name: str) -> SessionRow | None:
    with connection() as conn:
        row = conn.execute(
            "UPDATE sessions SET name = %s, updated_at = now() WHERE id = %s "
            "RETURNING id, name, created_at, updated_at",
            (name.strip() or "Untitled session", session_id),
        ).fetchone()
    return SessionRow(*row) if row else None


def delete_session(session_id: str) -> bool:
    # ON DELETE CASCADE clears the session's sources, chunks and messages.
    with connection() as conn:
        row = conn.execute(
            "DELETE FROM sessions WHERE id = %s RETURNING id", (session_id,)
        ).fetchone()
    return row is not None


def touch_session(session_id: str) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE sessions SET updated_at = now() WHERE id = %s", (session_id,)
        )


DEFAULT_SESSION_NAME = "My session"


def ensure_default_session() -> SessionRow:
    """Return the default session, creating it once if it is not there yet.

    Two guards, because this runs on every request that omits `session_id`:
    the partial unique index makes a duplicate impossible, and the ON CONFLICT
    clause means a concurrent creator loses the race gracefully instead of
    raising a unique violation.
    """
    with connection() as conn:
        row = conn.execute(
            "SELECT id, name, created_at, updated_at FROM sessions WHERE is_default LIMIT 1"
        ).fetchone()
    if row:
        return SessionRow(*row)

    with connection() as conn:
        row = conn.execute(
            "INSERT INTO sessions (name, is_default) VALUES (%s, TRUE) "
            "ON CONFLICT (is_default) WHERE is_default "
            "DO UPDATE SET updated_at = sessions.updated_at "
            "RETURNING id, name, created_at, updated_at",
            (DEFAULT_SESSION_NAME,),
        ).fetchone()
    return SessionRow(*row)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def add_message(session_id: str, role: str, content: str) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
            (session_id, role, content),
        )


def recent_messages(session_id: str, limit: int) -> list[dict]:
    """Oldest-first window of the most recent `limit` turns."""
    if limit <= 0:
        return []
    with connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM ("
            "  SELECT role, content, id FROM messages WHERE session_id = %s"
            "  ORDER BY id DESC LIMIT %s"
            ") recent ORDER BY id",
            (session_id, limit),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def clear_messages(session_id: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))


def now() -> datetime:
    return datetime.now(timezone.utc)