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

from psycopg.types.json import Json

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


def add_message(
    session_id: str,
    role: str,
    content: str,
    citations: list | None = None,
    evidence: dict | None = None,
) -> None:
    """Append a turn, keeping the evidence an answer was based on.

    `citations` and `evidence` are what make `[1]` in `content` mean something on
    a later read. They are optional rather than required so the same function
    records a user turn, and so a caller that genuinely has nothing to attach is
    not forced to invent two empty objects.
    """
    with connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, citations, evidence) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                session_id,
                role,
                content,
                Json(citations) if citations is not None else None,
                Json(evidence) if evidence is not None else None,
            ),
        )


def recent_messages(session_id: str, limit: int) -> list[dict]:
    """Oldest-first window of the most recent `limit` turns.

    An assistant turn carries its citations and evidence back out, so a
    reopened session renders the same provenance the original response did.
    Absent columns - a transcript written before this migration - come back as
    no keys at all rather than as `null`, because the client already treats a
    missing key and a null one the same way, and omitting is less surprising to
    read in an API response.
    """
    if limit <= 0:
        return []
    with connection() as conn:
        rows = conn.execute(
            "SELECT role, content, citations, evidence FROM ("
            "  SELECT role, content, citations, evidence, id FROM messages "
            "  WHERE session_id = %s"
            "  ORDER BY id DESC LIMIT %s"
            ") recent ORDER BY id",
            (session_id, limit),
        ).fetchall()
    turns = []
    for role, content, citations, evidence in rows:
        turn = {"role": role, "content": content}
        if citations:
            turn["citations"] = citations
        if evidence:
            turn["evidence"] = evidence
        turns.append(turn)
    return turns


def clear_messages(session_id: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))


# ---------------------------------------------------------------------------
# Usage ledger
# ---------------------------------------------------------------------------


def record_usage_event(
    kind: str,
    model: str,
    session_id: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    latency_ms: float = 0.0,
) -> None:
    """Write one billable call to the ledger.

    Best effort by design: this records what a call cost, it is not the call
    itself. An exception here must never turn a successful web search into a
    failed request, so it is swallowed rather than propagated - the cost would be
    missing, but the user's pages are already indexed and re-fetching them to
    retry a log line would cost more than the line is worth.
    """
    try:
        with connection() as conn:
            conn.execute(
                "INSERT INTO usage_events "
                "(session_id, kind, model, prompt_tokens, completion_tokens, latency_ms) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (session_id, kind, model, prompt_tokens, completion_tokens, latency_ms),
            )
    except Exception:  # noqa: BLE001 - a ledger failure must not fail the work
        return


def usage_totals(session_id: str | None = None) -> dict:
    """Total tokens spent, across both places cost is recorded.

    Chat turns keep their cost in `messages.evidence` so it reads back with the
    transcript; web searches keep theirs in `usage_events` because they are not
    turns. Summing only one of the two would report a total that is missing
    either every conversation or every search, which is worse than no total
    because it looks complete.

    Two queries rather than one union: the shapes differ (one side is a JSONB
    blob, the other a table), and readable beats clever for a number that has to
    survive being questioned.
    """
    where = "WHERE m.session_id = %s" if session_id else ""
    params = (session_id,) if session_id else ()
    with connection() as conn:
        row = conn.execute(
            f"""
            SELECT coalesce(sum((m.evidence->'cost'->>'prompt_tokens')::bigint), 0),
                   coalesce(sum((m.evidence->'cost'->>'completion_tokens')::bigint), 0),
                   count(*)
            FROM messages m
            {where}
            """,
            params,
        ).fetchone()
        llm_in, llm_out, turns = row

        row = conn.execute(
            """
            SELECT coalesce(sum(prompt_tokens), 0)::bigint,
                   coalesce(sum(completion_tokens), 0)::bigint,
                   count(*)::bigint,
                   coalesce(sum(latency_ms), 0)::float
            FROM usage_events
            """ + ("WHERE session_id = %s" if session_id else ""),
            params,
        ).fetchone()
        web_in, web_out, searches, web_ms = row

    return {
        "prompt_tokens": int(llm_in) + int(web_in),
        "completion_tokens": int(llm_out) + int(web_out),
        "llm_prompt_tokens": int(llm_in),
        "llm_completion_tokens": int(llm_out),
        "web_prompt_tokens": int(web_in),
        "web_completion_tokens": int(web_out),
        "turns": int(turns),
        "searches": int(searches),
        "web_search_ms": round(float(web_ms), 1),
    }


def now() -> datetime:
    return datetime.now(timezone.utc)