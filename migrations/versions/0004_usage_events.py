"""Record every billable call that is not a chat turn.

Answer/summarize/chat cost already survives in `messages.evidence`, because it
is attached to the turn that caused it and is read back with the transcript.
Web search is not a turn - nothing says "hi" when a page is fetched - so its
tokens had nowhere to land. They were computed, returned in the HTTP response,
and then dropped, which meant any total spend figure was silently short by
every search the user had ever run.

A ledger table rather than a column on `sessions` because a search can happen
with no session, its own row is what makes "cost per search" answerable, and
totals are a sum rather than an increment that can drift.

`session_id` is nullable and uses ON DELETE SET NULL rather than CASCADE: the
spend happened whether or not the notebook still exists. Cascading here would
make the ledger quietly lose rows on delete, and a cost record that erases
itself is not a record.
"""

from alembic import op

revision = "0004_usage_events"
down_revision = "0003_source_url"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE usage_events (
            id                BIGSERIAL PRIMARY KEY,
            session_id        UUID REFERENCES sessions(id) ON DELETE SET NULL,
            -- 'web_search' today; a chat call would be 'llm' if it is ever
            -- recorded here too. Kept as free text so the ledger does not need
            -- a migration to admit a new kind of spend.
            kind              TEXT NOT NULL,
            model             TEXT NOT NULL DEFAULT '',
            prompt_tokens     INTEGER NOT NULL DEFAULT 0,
            completion_tokens INTEGER NOT NULL DEFAULT 0,
            latency_ms        REAL NOT NULL DEFAULT 0,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX usage_events_session_idx ON usage_events(session_id, created_at)")
    op.execute("CREATE INDEX usage_events_kind_idx ON usage_events(kind)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS usage_events")
