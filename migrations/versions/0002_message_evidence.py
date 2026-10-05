"""Store the evidence with the answer, not just the answer.

`messages` kept only `role` and `content`, so an assistant turn came back from
the database as a string. The `[n]` markers survived, because they are in the
text, and the citation cards did not, because they were only ever in the HTTP
response. A user who asked a question, read a cited answer, closed the tab and
came back found the provenance gone - which quietly contradicts the one promise
this app makes. The evidence was not missing; it had been thrown away on the way
to disk.

Both new columns are JSONB and both are nullable. Nullable because a user turn
has no citations and must not be made to carry two empty objects, and JSONB
rather than more tables because this is read with its parent row and never
queried on its own: `WHERE evidence->>'verdict' = 'no_match'` is not a question
this app asks, and a join per citation to answer it would cost more than it
could ever return.

Revision ID: 0002_message_evidence
Revises: 0001_baseline
Create Date: 2026-10-05
"""

from alembic import op

revision = "0002_message_evidence"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # IF NOT EXISTS because the baseline revision may be re-stamped onto a
    # database that a previous version of this code already created.
    op.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS citations JSONB")
    op.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS evidence JSONB")


def downgrade() -> None:
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS evidence")
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS citations")