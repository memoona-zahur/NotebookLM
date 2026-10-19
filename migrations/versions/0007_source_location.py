"""Where each chunk sits in its source, so a citation can open the document there.

A citation used to point at a chunk, and a chunk only knew its page. To show the
original document with the answering text marked, the store has to be able to
say *where in the document* a chunk came from:

- `sources.parsed_text` is the text the parser produced for the whole document
  (blocks joined with a blank line). It is what non-PDF formats are shown as -
  the original bytes of a DOCX or a CSV are not something a browser can mark up -
  and it is what the chunk offsets below index into.
- `sources.page_offsets` is, for a PDF, the offset in `parsed_text` at which each
  page starts (element 0 is page 1). It turns a document offset into a
  page-local one, which is what picking between repeated phrases on one page
  needs. NULL for formats without pages.
- `chunks.char_start` / `char_end` are absolute offsets of the chunk in
  `parsed_text`. Absolute, never a running sum of lengths, so two chunks that
  overlap cover the same region of the same text and resolve to the same place.

All four are nullable and not backfilled, as in 0005 and 0006: a source indexed
before this migration has no recorded position, and NULL says so. A 0 would be a
claim ("starts at the beginning"). Re-indexing a source fills them in and keeps
its id, so old citations keep working.
"""

from alembic import op

revision = "0007_source_location"
down_revision = "0006_source_view"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE sources ADD COLUMN parsed_text TEXT")
    op.execute("ALTER TABLE sources ADD COLUMN page_offsets INTEGER[]")
    op.execute("ALTER TABLE chunks ADD COLUMN char_start INTEGER")
    op.execute("ALTER TABLE chunks ADD COLUMN char_end INTEGER")


def downgrade() -> None:
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS char_end")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS char_start")
    op.execute("ALTER TABLE sources DROP COLUMN IF EXISTS page_offsets")
    op.execute("ALTER TABLE sources DROP COLUMN IF EXISTS parsed_text")
