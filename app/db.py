"""Database access: connection pool, migrations, and session-scoped queries.

Everything persistent lives in one Postgres: source records, chunk text, chunk
vectors (pgvector), chat messages, and session metadata. One store means one
backup and one thing to reason about at 3am.

The schema is owned by Alembic in `migrations/`, not by this module; see
`migrate()` below.
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


# The schema itself lives in migrations/versions, not here. Alembic applies it in
# order and records what it did, so a column rename or a data backfill is
# expressible and a deployment knows which revision a database is on. Boot-time
# DDL cannot do either.


def alembic_config():
    """Alembic config bound to this application's DSN."""
    from alembic.config import Config

    cfg = Config(str(config.BASE_DIR / "alembic.ini"))
    # Left unset on purpose: env.py falls back to dsn(), which reads DATABASE_URL
    # and so follows the same environment the app does.
    cfg.set_main_option("sqlalchemy.url", "")
    return cfg


def migrate() -> str:
    """Apply every pending migration. Returns the revision now at head.

    Called on boot so a fresh volume is usable with no manual step and an
    existing one is upgraded in place. Alembic is a no-op when already current,
    so this is safe on every start.
    """
    from alembic import command

    cfg = alembic_config()
    command.upgrade(cfg, "head")
    return current_revision() or ""


def current_revision() -> str | None:
    """Which revision this database is on, or None if it has never been migrated."""
    with pool().connection() as conn:
        row = conn.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone()
    return row[0] if row else None


def head_revision() -> str | None:
    from alembic.script import ScriptDirectory

    return ScriptDirectory.from_config(alembic_config()).get_current_head()


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