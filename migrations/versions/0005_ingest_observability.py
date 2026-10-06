"""Say what ingest actually did, and notice a file you already have.

Two columns that answer two questions that had no answer before.

`content_hash` is a SHA-256 of the uploaded bytes. Re-uploading the same PDF
three times used to create three sources, index it three times, and triple the
answer's context; nothing in the store could tell that the second copy was the
first copy. The hash is checked inside the session, not globally: two people
uploading the same public document in different notebooks are not duplicates of
each other, and a global uniqueness rule would make one of them unable to index
their own file. NULL means "not hashed" - web sources and rows created before
this migration - and NULL never matches, which is the safe direction.

`token_max` and `fit_splits` are the ingestion report. `CHUNK_SIZE` is a
character ceiling, but the embedding model enforces a wordpiece ceiling, and
before this migration nothing recorded whether the second one had been hit or
what it cost. `token_max` is the largest wordpiece count of any chunk in the
source, so the report can say "341 of 512" rather than implying everything fit;
`fit_splits` is how many extra cuts were made to make it fit. Both default to 0
so an existing row reads as "not measured" rather than as a confident zero -
the distinction matters, and the default is honest about it.

Backfill is deliberately skipped: recomputing these would mean re-parsing every
stored source on migration, and a source that predates the report is better
described as "unmeasured" than as measured with a tokenizer that may since have
changed.
"""

from alembic import op

revision = "0005_ingest_observability"
down_revision = "0004_usage_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE sources ADD COLUMN content_hash TEXT")
    op.execute("ALTER TABLE sources ADD COLUMN token_max INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE sources ADD COLUMN fit_splits INTEGER NOT NULL DEFAULT 0")
    # The dedup lookup is always "this session, this hash". Without the composite
    # index the check is a scan of every source the session owns, which is the
    # whole notebook on a large one.
    op.execute(
        "CREATE INDEX sources_session_hash_idx ON sources(session_id, content_hash)"
        " WHERE content_hash IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS sources_session_hash_idx")
    op.execute("ALTER TABLE sources DROP COLUMN IF EXISTS fit_splits")
    op.execute("ALTER TABLE sources DROP COLUMN IF EXISTS token_max")
    op.execute("ALTER TABLE sources DROP COLUMN IF EXISTS content_hash")
