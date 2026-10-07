"""Enough on the row to open the original file and to say what was cut.

`file_bytes` is the size of the upload as it was stored. It could be read with
a `stat` on every listing, but the sources endpoint runs on every status poll
and a listing that touches the filesystem for each row is a disk round trip
hiding inside what looks like a database read. Recorded once at ingest instead.

`size_splits` counts cuts forced by `CHUNK_SIZE`, the character ceiling, as the
companion to `fit_splits` which counts cuts forced by the embedding model's
wordpiece window. Only the second one used to be counted, which made the report
half an answer: a reader could see how often the token ceiling fired and had no
way to see how often the character one did - and the honest finding, that it
almost never does, is exactly the sort of claim that needs a number attached to
it rather than an adjective.

Both default to nothing rather than to 0, and are not backfilled, for the same
reason as 0005: a source indexed before this migration is unmeasured. NULL and
0 mean different things here in a way they do not for `token_max` - a chunk with
zero wordpieces cannot exist, so 0 is unambiguous there, while "zero cuts" is
the ordinary outcome for a new file. A default of 0 would therefore report the
common case and the unknown case identically.
"""

from alembic import op

revision = "0006_source_view"
down_revision = "0005_ingest_observability"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE sources ADD COLUMN file_bytes BIGINT")
    op.execute("ALTER TABLE sources ADD COLUMN size_splits INTEGER")


def downgrade() -> None:
    op.execute("ALTER TABLE sources DROP COLUMN IF EXISTS size_splits")
    op.execute("ALTER TABLE sources DROP COLUMN IF EXISTS file_bytes")
