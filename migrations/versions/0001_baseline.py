"""Baseline: sessions, sources, chunks, messages.

This revision reproduces the schema the application was creating with boot-time
DDL, so an existing database can be stamped at this revision instead of being
rebuilt from scratch.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-01-15
"""

from alembic import op
from sqlalchemy import text

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def _dim() -> int:
    """Vector width from config, so the column always matches the model."""
    from app import config

    return config.EMBED_DIM


def upgrade() -> None:
    # Extensions first: gen_random_uuid() and vector() both depend on these, and
    # in a managed Postgres they usually need to be enabled by an admin.
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute(
        """
        CREATE TABLE sessions (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name        TEXT NOT NULL,
            is_default  BOOLEAN NOT NULL DEFAULT FALSE,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    # Partial unique index: it constrains the single default session without
    # forbidding several ordinary sessions, and the WHERE clause is what
    # ON CONFLICT (is_default) WHERE is_default must match later.
    op.execute(
        "CREATE UNIQUE INDEX sessions_one_default_idx "
        "ON sessions (is_default) WHERE is_default"
    )

    op.execute(
        """
        CREATE TABLE sources (
            id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id    UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            name          TEXT NOT NULL,
            kind          TEXT NOT NULL,
            pages         INTEGER NOT NULL DEFAULT 0,
            chunk_count   INTEGER NOT NULL DEFAULT 0,
            numeric_count INTEGER NOT NULL DEFAULT 0,
            storage_path  TEXT NOT NULL DEFAULT '',
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX sources_session_idx ON sources(session_id)")

    op.execute(
        f"""
        CREATE TABLE chunks (
            id            BIGSERIAL PRIMARY KEY,
            source_id     UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            position      INTEGER NOT NULL,
            page          INTEGER NOT NULL DEFAULT 0,
            heading       TEXT NOT NULL DEFAULT '',
            text          TEXT NOT NULL,
            numeric_heavy BOOLEAN NOT NULL DEFAULT FALSE,
            embedding     vector({_dim()}) NOT NULL,
            UNIQUE (source_id, position)
        )
        """
    )
    op.execute("CREATE INDEX chunks_source_idx ON chunks(source_id)")
    # HNSW for approximate nearest-neighbour search over the embeddings. Without
    # it every query is a sequential scan of the whole vector column.
    op.execute(
        "CREATE INDEX chunks_embedding_hnsw "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )

    op.execute(
        """
        CREATE TABLE messages (
            id         BIGSERIAL PRIMARY KEY,
            session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            -- Constrained so a row can never hold a 'system' turn that would
            -- compete with the grounding instructions.
            role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content    TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    # (session_id, id) rather than (session_id) alone: a transcript is read as
    # "the last N turns", so id is the ordering column that has to be covered.
    op.execute("CREATE INDEX messages_session_idx ON messages(session_id, id)")


def downgrade() -> None:
    # Reverse dependency order. The tables drop their indexes with them, but the
    # default-session partial index is listed explicitly so a failure here is
    # unambiguous about what went wrong.
    op.execute("DROP TABLE IF EXISTS messages")
    op.execute("DROP TABLE IF EXISTS chunks")
    op.execute("DROP TABLE IF EXISTS sources")
    op.execute("DROP INDEX IF EXISTS sessions_one_default_idx")
    op.execute("DROP TABLE IF EXISTS sessions")