"""Remember where a source came from.

`sources` recorded only `name` and `storage_path`. That is enough for an upload,
where the name is the user's filename and the path is ours. It is not enough for
a page found by web search, where the two facts that matter - which site it came
from and what its address is - are not in either column. A source called
"ROUGE (metric)" with a UUID path cannot be re-read, re-fetched, or checked
against the site it claims to be, and the user has no way to tell a page pulled
from a real publisher from one that arrived through a redirect.

So `url` is nullable and defaults to empty: an uploaded file has no address, and
making that say "no URL" rather than inventing one keeps `name` and `url`
answering different questions. It is stored rather than derived because the
address is needed after the fact - a citation should be able to link out to the
page it came from, and a source list should be able to show the origin - and a
page that can only be identified while it is being fetched cannot do either.

Revision ID: 0003_source_url
Revises: 0002_message_evidence
Create Date: 2026-10-05
"""

from alembic import op

revision = "0003_source_url"
down_revision = "0002_message_evidence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE sources ADD COLUMN IF NOT EXISTS url TEXT NOT NULL DEFAULT ''")


def downgrade() -> None:
    op.execute("ALTER TABLE sources DROP COLUMN IF EXISTS url")