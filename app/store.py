import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from . import config
from . import db
from . import parsers
from .embeddings import embed_query, embed_texts, score_against_query
from .lexical import BM25, fuse
from .rerank import rerank

# How much of a highlight response one request may return. A common word can
# appear in every chunk of every document, and the browser is worse off for
# rendering all of it.
OCCURRENCE_LIMIT = 40
# Per chunk, so one repetitive passage cannot crowd out every other source.
OCCURRENCE_PER_CHUNK = 10
MAX_TERM_CHARS = 80


def _occurrence_pattern(term: str) -> re.Pattern[str] | None:
    """Match `term` case-insensitively, on word boundaries where it is a word.

    Highlighting 'rate' inside 'generate' is noise, and a bare substring search
    for '4' lights up every number in the corpus. Boundaries go on only at the
    edges that start or end on a word character, so a phrase like '(old)' or
    '4.8 ' still matches exactly as written.
    """
    term = term.strip()
    if not term:
        return None
    lead = r"\b" if term[0].isalnum() or term[0] == "_" else ""
    trail = r"\b" if term[-1].isalnum() or term[-1] == "_" else ""
    return re.compile(f"{lead}{re.escape(term)}{trail}", re.IGNORECASE)


def _ilike(term: str) -> str:
    """A pattern for ILIKE, with the wildcards in the user's term escaped."""
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _discard(storage_path) -> None:
    """Delete a stored file, but only if the store is the thing that owns it.

    `add` records whatever path it was handed, and that is correct: the route
    generates a UUID name inside `UPLOAD_DIR` precisely so the store can own and
    later delete it. It also means anything else that indexes files through the
    store - the evaluation harnesses point it straight at `experiments/corpus/` -
    would have its originals deleted by an ordinary session delete.

    That already happened once: running the gold evaluation and then deleting
    its session removed 24 committed corpus documents, and `git status` showed
    them as deletions with nothing in the code to explain why. So the rule is
    enforced here instead of left to every caller's good intentions: the store
    only ever deletes inside its own upload directory. A path outside it is left
    on disk, because leaking one file is a smaller failure than losing a
    committed corpus to a DELETE endpoint.
    """
    if not storage_path:
        return
    path = Path(storage_path)
    try:
        inside = path.resolve().is_relative_to(config.UPLOAD_DIR.resolve())
    except OSError:
        return
    if inside:
        path.unlink(missing_ok=True)


def _content_hash(path: Path) -> str:
    """SHA-256 of a file's bytes, read in pieces so a 100 MB PDF is not slurped.

    Byte-exact rather than "the text we parsed", because the check has to run
    before parsing - it is the check that decides whether to parse at all.
    """
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for piece in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(piece)
    except OSError:
        return ""
    return digest.hexdigest()


def _file_bytes(path: Path) -> int | None:
    """Size of the stored upload, or None if it is not there to measure.

    None rather than 0 because 0 bytes is a claim about a file and this is the
    absence of one. The distinction survives into the UI, which shows "size not
    recorded" for None and would otherwise have to print "0 KB" for a source
    whose file it never measured.
    """
    try:
        return Path(path).stat().st_size
    except OSError:
        return None


@dataclass
class Source:
    id: str
    name: str
    kind: str
    pages: int = 0
    chunks: int = 0
    numeric: int = 0
    # Empty for an uploaded file, which has no address; set when the source came
    # from a web search, so the page can be identified and linked later.
    url: str = ""
    # The ingestion report. Largest wordpiece count of any chunk, and how many
    # extra cuts were made to fit the embedding model's window - the two numbers
    # that say whether the index lost anything on the way in. 0 means "not
    # measured" for a source indexed before this was recorded.
    token_max: int = 0
    fit_splits: int = 0
    avg_chars: int = 0
    max_chars: int = 0
    # Cuts forced by the character ceiling, the companion to `fit_splits`.
    # Recorded per source so the expanded row can say which ceiling fired on
    # *this* file rather than on the whole notebook. None, not 0: a source
    # indexed before this was recorded has no answer, and "no cuts" is a real
    # and common answer that must not be confused with no answer at all.
    size_splits: int | None = None
    # Size of the stored upload in bytes, taken once at ingest. Feeding the
    # "View" control and the file facts row without stat-ing on every listing.
    file_bytes: int | None = None
    created_at: str = ""
    # True only on the return value of an `add` that recognised content the
    # session already had. Never stored; false for every source in a listing.
    duplicate: bool = False


# Every list-returning query builds its row with `Source(*row)`, so the column
# order here *is* the field order above. Two queries used to spell this out
# separately, which is one accidental reordering away from silently reading
# `token_max` into `fit_splits` - both are ints, so nothing would raise.
_SOURCE_SELECT = """
    SELECT s.id::text, s.name, s.kind, s.pages, s.chunk_count,
           s.numeric_count, s.url, s.token_max, s.fit_splits,
           COALESCE(c.avg_chars, 0)::int, COALESCE(c.max_chars, 0)::int,
           s.size_splits, s.file_bytes,
           to_char(s.created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')
    FROM sources s
    LEFT JOIN LATERAL (
        SELECT AVG(LENGTH(text)) AS avg_chars,
               MAX(LENGTH(text)) AS max_chars
        FROM chunks WHERE source_id = s.id
    ) c ON TRUE
"""


@dataclass
class SearchResult:
    """Outcome of one retrieval, including why it was or was not confident."""

    hits: list[dict]
    best_score: float
    considered: int
    min_score: float
    floor_applied: float = 0.0
    numeric_share: float = 0.0
    damped: bool = False
    mode: str = "dense"

    @property
    def relevant(self) -> bool:
        return bool(self.hits)

    def confidence(self) -> str:
        """Bucket the best score so the UI can show an honest strength label."""
        if not self.hits:
            return "none"
        if self.best_score >= 0.5:
            return "high"
        if self.best_score >= 0.35:
            return "medium"
        return "low"


class VectorStore:
    """Chunks, embeddings and BM25, scoped to one session.

    Dense search is a pgvector query; lexical search is an in-memory BM25 over
    the session's chunks, rebuilt lazily and invalidated when the chunk set
    changes. BM25 has no natural persistent form, and the sessions here are
    small enough that rebuilding is cheaper than maintaining an index.
    """

    def __init__(self) -> None:
        self._lexical: dict[str, BM25] = {}
        self._revision: dict[str, int] = {}

    # -- caching -----------------------------------------------------------

    def _session_revision(self, session_id: str) -> int:
        with db.connection() as conn:
            row = conn.execute(
                "SELECT count(*)::bigint FROM chunks WHERE source_id IN "
                "(SELECT id FROM sources WHERE session_id = %s)",
                (session_id,),
            ).fetchone()
        return int(row[0])

    def _lexical_for(self, session_id: str) -> BM25:
        """BM25 over the session's chunks, rebuilt only when it changed."""
        revision = self._session_revision(session_id)
        if self._revision.get(session_id) != revision:
            with db.connection() as conn:
                rows = conn.execute(
                    "SELECT c.id::text, c.text FROM chunks c "
                    "JOIN sources s ON s.id = c.source_id WHERE s.session_id = %s",
                    (session_id,),
                ).fetchall()
            index = BM25()
            index.fit([row[0] for row in rows], [row[1] for row in rows])
            self._lexical[session_id] = index
            self._revision[session_id] = revision
        return self._lexical[session_id]

    # -- writes ------------------------------------------------------------

    def add(self, path: Path, display_name: str, session_id: str, url: str = "") -> Source:
        """Index a file into a session, unless the session already has it.

        The content hash is taken and checked *before* parsing, because parsing
        and embedding is the expensive part and "you already uploaded this" is
        the one outcome that makes it pointless. The duplicate check is scoped to
        the session: two notebooks may both legitimately hold the same file, and
        a cross-session rule would tell the second user they cannot index their
        own document.

        On a match the stored file is discarded (it is a second copy of a file
        the store already owns) and the *existing* source is returned, flagged
        `duplicate`, so the caller reports it as a no-op rather than as a new
        source with a new id that answers would then cite inconsistently.
        """
        digest = _content_hash(path)
        if digest:
            existing = self._find_duplicate(session_id, digest)
            if existing is not None:
                _discard(path)
                return existing

        stats = parsers.IngestStats()
        blocks = parsers.parse(path, stats=stats)
        name = display_name or path.name

        keep: list[tuple] = []
        numeric = 0
        for position, block in enumerate(blocks):
            if not block.text.strip():
                continue
            heavy = parsers.is_numeric_heavy(block.text)
            if heavy:
                numeric += 1
            keep.append((position, block.page, block.heading, block.text, heavy))

        chunk_count = 0
        source_id = None
        kind = path.suffix.lower().lstrip(".")
        with db.connection() as conn:
            source = conn.execute(
                "INSERT INTO sources (session_id, name, kind, pages, storage_path, url, "
                "content_hash, token_max, fit_splits, size_splits, file_bytes) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "RETURNING id, to_char(created_at AT TIME ZONE 'UTC', "
                "'YYYY-MM-DD HH24:MI:SS')",
                (session_id, name, kind, len(blocks), str(path), url or "",
                 digest or None, stats.token_max, stats.fit_splits,
                 stats.size_splits, _file_bytes(path)),
            ).fetchone()
            source_id = source[0]
            created_at = source[1]
            if keep:
                vectors = embed_texts([row[3] for row in keep])
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks (source_id, position, page, heading, text, "
                        "numeric_heavy, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        [
                            (source_id, row[0], row[1], row[2], row[3], row[4], vector)
                            for row, vector in zip(keep, vectors)
                        ],
                    )
                chunk_count = len(keep)
            conn.execute(
                "UPDATE sources SET chunk_count = %s, numeric_count = %s WHERE id = %s",
                (chunk_count, numeric, source_id),
            )

        self._invalidate(session_id)
        return Source(
            id=str(source_id),
            name=name,
            kind=kind,
            pages=len(blocks),
            chunks=chunk_count,
            numeric=numeric,
            url=url or "",
            token_max=stats.token_max,
            fit_splits=stats.fit_splits,
            avg_chars=stats.avg_chars,
            max_chars=stats.max_chars,
            size_splits=stats.size_splits,
            file_bytes=_file_bytes(path),
            created_at=created_at,
        )

    def _find_duplicate(self, session_id: str, digest: str) -> Source | None:
        """The source in this session whose bytes hash to `digest`, if any.

        Any parse failure on the existing row falls through to None rather than
        aborting the upload: a row that cannot be read should not become a wall
        that stops the user indexing their file, so they get a fresh copy.
        """
        try:
            with db.connection() as conn:
                row = conn.execute(
                    _SOURCE_SELECT
                    + "WHERE s.session_id = %s AND s.content_hash = %s "
                    "ORDER BY s.created_at LIMIT 1",
                    (session_id, digest),
                ).fetchone()
        except Exception:  # noqa: BLE001
            return None
        if row is None:
            return None
        return Source(*row, duplicate=True)

    def reindex(self, source_id: str, session_id: str) -> Source | None:
        """Re-parse and re-embed a source in place, from the copy on disk.

        Why this exists: the upload path deliberately keeps a file it managed to
        parse, so "the session can be re-indexed without a re-upload" - but
        nothing ever called it back. Without this the only way to pick up a
        changed `CHUNK_SIZE` or `EMBED_MODEL` is to delete the source and go
        find the original file again.

        The source id is kept, not minted. Persisted `messages.citations` point
        at chunk ids and source ids, so a new id would silently detach every
        past answer's links from the passage it quoted. `created_at` is kept too,
        so the source list does not reorder under the user.

        Parse and embed happen *before* the transaction opens. If either fails
        the existing chunks are still there and the source is untouched; deleting
        first would leave a readable file indexed as nothing, which is the
        silent-empty-index failure this app refuses elsewhere.
        """
        with db.connection() as conn:
            row = conn.execute(
                "SELECT name, storage_path, url, "
                "to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS') "
                "FROM sources WHERE id = %s AND session_id = %s",
                (source_id, session_id),
            ).fetchone()
        if row is None:
            return None
        name, stored, url, created_at = row
        path = Path(stored)
        if not stored or not path.exists():
            raise parsers.UnreadableDocument(
                "The original file for this source is no longer on disk, so it "
                "cannot be re-read. Upload it again to index it."
            )

        stats = parsers.IngestStats()
        blocks = parsers.parse(path, stats=stats)
        digest = _content_hash(path)
        keep: list[tuple] = []
        numeric = 0
        for position, block in enumerate(blocks):
            if not block.text.strip():
                continue
            heavy = parsers.is_numeric_heavy(block.text)
            if heavy:
                numeric += 1
            keep.append((position, block.page, block.heading, block.text, heavy))

        vectors = embed_texts([row[3] for row in keep]) if keep else []

        with db.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
            if keep:
                with conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO chunks (source_id, position, page, heading, text, "
                        "numeric_heavy, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        [
                            (source_id, row[0], row[1], row[2], row[3], row[4], vector)
                            for row, vector in zip(keep, vectors)
                        ],
                    )
            conn.execute(
                "UPDATE sources SET chunk_count = %s, numeric_count = %s, pages = %s, "
                "content_hash = %s, token_max = %s, fit_splits = %s, "
                "size_splits = %s, file_bytes = %s "
                "WHERE id = %s",
                (len(keep), numeric, len(blocks), digest or None,
                 stats.token_max, stats.fit_splits, stats.size_splits,
                 _file_bytes(path), source_id),
            )

        self._invalidate(session_id)
        return Source(
            id=source_id,
            name=name,
            kind=path.suffix.lower().lstrip("."),
            pages=len(blocks),
            chunks=len(keep),
            numeric=numeric,
            url=url or "",
            token_max=stats.token_max,
            fit_splits=stats.fit_splits,
            avg_chars=stats.avg_chars,
            max_chars=stats.max_chars,
            size_splits=stats.size_splits,
            file_bytes=_file_bytes(path),
            created_at=created_at,
        )

    def remove(self, source_id: str, session_id: str) -> bool:
        with db.connection() as conn:
            row = conn.execute(
                "DELETE FROM sources WHERE id = %s AND session_id = %s "
                "RETURNING id, storage_path",
                (source_id, session_id),
            ).fetchone()
        self._invalidate(session_id)
        if row is None:
            return False
        # The database row is gone; drop the uploaded file with it so the
        # volume does not quietly fill up.
        _discard(row[1])
        return True

    def clear(self, session_id: str) -> None:
        with db.connection() as conn:
            # ON DELETE CASCADE drops the chunk rows too.
            paths = conn.execute(
                "DELETE FROM sources WHERE session_id = %s RETURNING storage_path",
                (session_id,),
            ).fetchall()
        self._invalidate(session_id)
        for (path,) in paths:
            _discard(path)

    def _invalidate(self, session_id: str) -> None:
        self._lexical.pop(session_id, None)
        self._revision.pop(session_id, None)

    def forget(self, session_id: str) -> None:
        """Drop every cache for a deleted session."""
        self._invalidate(session_id)

    def storage_path(self, source_id: str, session_id: str) -> Path | None:
        """Where the original upload for a source lives, if we still have it."""
        with db.connection() as conn:
            row = conn.execute(
                "SELECT storage_path FROM sources WHERE id = %s AND session_id = %s",
                (source_id, session_id),
            ).fetchone()
        return Path(row[0]) if row and row[0] else None

    def original_file(self, source_id: str, session_id: str) -> tuple[str, Path] | None:
        """The display name and stored path of a source's original, or None.

        Separate from `storage_path` because serving a file needs the name the
        user knows it by, and the name is a user-visible string while the path is
        not. Both are scoped to the session: owning a source id is not the same
        as owning the source, and every other source route works that way.
        """
        with db.connection() as conn:
            row = conn.execute(
                "SELECT name, storage_path FROM sources "
                "WHERE id = %s AND session_id = %s",
                (source_id, session_id),
            ).fetchone()
        if not row or not row[1]:
            return None
        return row[0], Path(row[1])

    def source_chunks(self, source_id: str, session_id: str) -> list[dict] | None:
        """Every stored chunk of one source in document order, or None.

        None means the source is not in this session - the same 404 the other
        per-source routes give, because owning a source id is not the same as
        owning its chunks. An empty list is a different answer: a source that
        exists and has nothing indexed under it.

        What comes back is what was stored, not a re-derivation of it. `page`
        and `heading` are the parent block's, because that is what survived the
        split - which sub-piece of a parent a chunk is, and whether the size or
        the wordpiece ceiling cut it, was never recorded, and inventing it after
        the fact would be the opposite of what this endpoint is for. `chars` is
        measured here rather than kept as a column, for the same reason
        `avg_chars` is: it is a function of `text`, and a second copy could
        drift from the first.
        """
        with db.connection() as conn:
            owned = conn.execute(
                "SELECT 1 FROM sources WHERE id = %s AND session_id = %s",
                (source_id, session_id),
            ).fetchone()
            if owned is None:
                return None
            rows = conn.execute(
                """
                SELECT position, page, heading, text, numeric_heavy
                FROM chunks
                WHERE source_id = %s
                ORDER BY position
                """,
                (source_id,),
            ).fetchall()
        return [
            {
                "position": row[0],
                "page": row[1],
                "heading": row[2],
                "text": row[3],
                "chars": len(row[3]),
                "numeric": bool(row[4]),
            }
            for row in rows
        ]

    # -- reads -------------------------------------------------------------

    def sources(self, session_id: str) -> list[Source]:
        """The session's sources, each with the ingestion report attached.

        `avg_chars` and `max_chars` are computed from the stored chunks rather
        than kept as columns: they are a function of `chunks`, so a column would
        be a second copy of a fact that could drift from the first. `file_bytes`
        is the opposite - it is a property of a file that may since have been
        deleted, so it has to be captured when the file is there.
        """
        with db.connection() as conn:
            rows = conn.execute(
                _SOURCE_SELECT + "WHERE s.session_id = %s ORDER BY s.created_at",
                (session_id,),
            ).fetchall()
        return [Source(*row) for row in rows]

    def stats(self, session_id: str) -> dict:
        """Session totals plus the ingestion report.

        The report exists because "indexed" is not one claim. A notebook can be
        fully indexed and still contain chunks the embedding model truncated, or
        contain a third numeric tables that damp to nothing on retrieval; those
        are different problems with different fixes, and neither is visible from
        a chunk count.
        """
        records = self.sources(session_id)
        chunks = sum(s.chunks for s in records)
        numeric = sum(s.numeric for s in records)
        measured = [s for s in records if s.token_max > 0]
        weighted = sum(s.avg_chars * s.chunks for s in records)
        return {
            "sources": [vars(s) for s in records],
            "chunks": chunks,
            "numeric": numeric,
            "ingest": {
                "chunks": chunks,
                "numeric": numeric,
                "numeric_pct": round(100 * numeric / chunks, 1) if chunks else 0.0,
                "avg_chars": int(weighted / chunks) if chunks else 0,
                "max_chars": max((s.max_chars for s in records), default=0),
                # Largest wordpiece count seen, against the window the model
                # enforces. Reported as a ratio rather than a bare number so the
                # reader can see how close the ceiling is without knowing what
                # the ceiling is.
                "token_max": max((s.token_max for s in records), default=0),
                "token_window": config.EMBED_MAX_TOKENS,
                # Cuts made only to satisfy the token window. A non-zero figure
                # is not a defect; it is a document whose characters do not map
                # onto wordpieces the way CHUNK_SIZE assumes.
                "fit_splits": sum(s.fit_splits for s in records),
                # Sources indexed before this was recorded. Their figures are
                # 0, and 0 here means "unknown", so they are counted apart.
                "unmeasured": len(records) - len(measured),
            },
        }

    def find_occurrences(
        self,
        term: str,
        session_id: str,
        limit: int = OCCURRENCE_LIMIT,
        per_chunk: int = OCCURRENCE_PER_CHUNK,
    ) -> dict:
        """Every place `term` appears in this session's indexed text.

        Returns the character spans rather than the terms, so the caller marks
        what was matched instead of searching again and risking a different
        answer. Rows come back ordered by source name then chunk position, which
        is what makes a location stable enough to link to twice.
        """
        pattern = _occurrence_pattern(term)
        if pattern is None:
            return {"term": term, "occurrences": [], "count": 0, "truncated": False}

        term = term.strip()[:MAX_TERM_CHARS]
        with db.connection() as conn:
            rows = conn.execute(
                """
                SELECT s.name, s.kind, c.position, c.page, c.heading, c.text
                FROM chunks c
                JOIN sources s ON s.id = c.source_id
                WHERE s.session_id = %s AND c.text ILIKE %s ESCAPE '\\'
                ORDER BY s.name, c.position
                """,
                (session_id, _ilike(term)),
            ).fetchall()

        occurrences: list[dict] = []
        total = 0
        truncated = False
        for name, kind, position, page, heading, text in rows:
            spans = [m.span() for m in pattern.finditer(text)]
            if not spans:
                continue  # ILIKE and the word-boundary pattern disagree
            total += len(spans)
            if len(occurrences) >= limit:
                truncated = True
                continue
            occurrences.append(
                {
                    "source": name,
                    "kind": kind,
                    "position": position,
                    "page": page,
                    "heading": heading,
                    "text": text,
                    "matches": [list(span) for span in spans[:per_chunk]],
                    "count": len(spans),
                }
            )
        return {
            "term": term,
            "occurrences": occurrences,
            "count": total,
            "truncated": truncated,
        }

    # -- retrieval ---------------------------------------------------------

    def search(
        self,
        query: str,
        session_id: str,
        top_k: int | None = None,
        min_score: float | None = None,
        max_per_source: int | None = None,
    ) -> list[dict]:
        """Relevant chunks only, best first. See search_detailed for metadata."""
        return self.search_detailed(
            query, session_id, top_k, min_score, max_per_source
        ).hits

    def search_detailed(
        self,
        query: str,
        session_id: str,
        top_k: int | None = None,
        min_score: float | None = None,
        max_per_source: int | None = None,
    ) -> SearchResult:
        limit = max(1, top_k or config.TOP_K)
        floor = config.MIN_SCORE if min_score is None else min_score
        cap = max(
            1,
            min(
                config.MAX_PER_SOURCE if max_per_source is None else max_per_source,
                limit,
            ),
        )

        pool_size = max(limit, limit * 4)
        query_vector = embed_query(query)

        # Dense candidates from pgvector. Scoped to the session so one session's
        # documents can never surface in another's answers.
        with db.connection() as conn:
            dense_rows = conn.execute(
                "SELECT c.id::text, c.text, c.page, c.heading, c.numeric_heavy, "
                "       s.id::text AS source_id, s.name, "
                "       c.embedding <=> %s::vector AS distance "
                "FROM chunks c JOIN sources s ON s.id = c.source_id "
                "WHERE s.session_id = %s "
                "ORDER BY c.embedding <=> %s::vector LIMIT %s",
                (query_vector, session_id, query_vector, pool_size),
            ).fetchall()

        if not dense_rows:
            return SearchResult(hits=[], best_score=0.0, considered=0, min_score=floor)

        # pgvector returns cosine *distance* (0 = identical, 2 = opposite).
        # Retrieval has always been calibrated on cosine *similarity*, so the
        # conversion happens here and nowhere else.
        by_id: dict[str, dict] = {}

        def remember(chunk_id: str, source_id: str, name: str, page: int,
                     heading: str, text: str, numeric: bool, score: float) -> None:
            if chunk_id in by_id:
                by_id[chunk_id]["score"] = round(score, 4)
                return
            by_id[chunk_id] = {
                "chunk_id": chunk_id,
                "text": text,
                "source": name,
                "page": page,
                "source_id": source_id,
                "heading": heading or "",
                "numeric_heavy": bool(numeric),
                "score": round(score, 4),
                "support": 0.0,
                "rescued": False,
            }

        for chunk_id, text, page, heading, numeric, source_id, name, distance in dense_rows:
            remember(chunk_id, source_id, name, page, heading, text, numeric, 1.0 - float(distance))

        mode = "dense"
        dense_ranked = sorted(by_id.values(), key=lambda hit: -hit["score"])[:pool_size]

        lexical = self._lexical_for(session_id)
        if config.HYBRID_ENABLED and lexical.ids:
            mode = "hybrid"
            lexical_ranked = lexical.top(query, pool_size)
            # A chunk surfaced only by BM25 has no dense score yet, but the
            # admission decision is made on the dense scale, so it must be
            # measured rather than assumed.
            unseen = [cid for cid, _bm25, _cov in lexical_ranked if cid not in by_id]
            if unseen:
                with db.connection() as conn:
                    rows = conn.execute(
                        "SELECT c.id::text, c.text, c.page, c.heading, c.numeric_heavy, "
                        "       s.id::text, s.name, c.embedding <=> %s::vector "
                        "FROM chunks c JOIN sources s ON s.id = c.source_id "
                        "WHERE c.id = ANY(%s::bigint[])",
                        (query_vector, [int(cid) for cid in unseen]),
                    ).fetchall()
                measured = score_against_query(
                    query_vector, [row[1] for row in rows]
                )
                for row, score in zip(rows, measured):
                    remember(row[0], row[5], row[6], row[2], row[3], row[1], row[4], score)

            covers = dict(
                (cid, cov) for cid, _s, cov in lexical.top(query, len(lexical.ids))
            )
            for chunk_id, _bm25, _cov in lexical_ranked:
                if chunk_id in by_id:
                    by_id[chunk_id]["support"] = covers.get(chunk_id, 0.0)

            fused = fuse(
                [(hit["chunk_id"], hit["score"]) for hit in dense_ranked],
                [(cid, score) for cid, score, _c in lexical_ranked],
                k=config.RRF_K,
            )
            ordered = [
                by_id[chunk_id]
                for chunk_id, _score in sorted(fused.items(), key=lambda kv: -kv[1])
                if chunk_id in by_id
            ]

            # A weak dense match with no term overlap is almost always a false
            # positive; a strong term match with a mediocre dense score is almost
            # always a real answer the embedding under-scored. Both directions are
            # admitted explicitly rather than by moving one global threshold,
            # which the eval corpus shows cannot work. The lexical rescue is only
            # offered in the default regime: a caller that passes a stricter
            # floor than MIN_SCORE is asking for strict, and gets strict.
            rescue_allowed = floor <= config.DEFAULT_MIN_SCORE
            open_floor = floor <= 0.0
            for hit in ordered:
                strong = hit["score"] >= config.DENSE_STRONG
                supported = hit["support"] >= config.COVERAGE_MIN
                if open_floor:
                    hit["admitted"] = "dense"
                elif strong:
                    hit["admitted"] = "dense"
                elif hit["score"] >= floor and supported:
                    hit["admitted"] = "both"
                elif rescue_allowed and supported and hit["score"] >= config.BM25_RESCUE_MIN:
                    hit["admitted"] = "lexical"
                    hit["rescued"] = True
                else:
                    hit["admitted"] = None
            ordered = [hit for hit in ordered if hit["admitted"]]
        else:
            ordered = list(dense_ranked)
            for hit in ordered:
                hit["admitted"] = "dense"

        # A numeric-heavy chunk (a bare data table) is semantically flat and will
        # match almost any query. Damp it so it cannot outrank real prose - but
        # only when prose is actually present. In a corpus that is entirely
        # numeric (a CSV, a log, a metrics dump) the damping is uniform, so
        # relative ranking is untouched and those documents still work.
        considered_hits = list(by_id.values())
        numeric_share = (
            sum(1 for hit in considered_hits if hit["numeric_heavy"]) / len(considered_hits)
            if considered_hits
            else 0.0
        )
        damped = bool(considered_hits) and numeric_share < config.NUMERIC_PENALTY_SHARE
        if damped:
            for hit in ordered:
                if hit["numeric_heavy"]:
                    hit["raw_score"] = hit["score"]
                    hit["score"] = round(hit["score"] * config.NUMERIC_DAMPING, 4)

        if not ordered:
            return SearchResult(
                hits=[], best_score=0.0, considered=len(considered_hits),
                min_score=floor, floor_applied=floor,
                numeric_share=round(numeric_share, 4), damped=damped, mode=mode,
            )

        best_score = max(hit["score"] for hit in ordered)
        # A chunk admitted on lexical evidence has already passed an independent
        # test, so the dense floor must not reject it again. Everything else
        # faces the absolute floor plus a share of the best hit, which prunes the
        # weak tail a single strong match drags along.
        effective_floor = max(floor, best_score * config.MIN_RATIO)

        def admitted(hit: dict) -> bool:
            if hit["rescued"]:
                return hit["score"] >= config.BM25_RESCUE_MIN
            return hit["score"] >= effective_floor

        hits = [hit for hit in ordered if admitted(hit)]
        if not hits:
            return SearchResult(
                hits=[], best_score=0.0, considered=len(considered_hits),
                min_score=floor, floor_applied=effective_floor,
                numeric_share=round(numeric_share, 4), damped=damped, mode=mode,
            )

        # Re-rank only what survived the floor, and only the head of it. The
        # floor is the dense model's calibrated decision and reranking before it
        # would not change any of those decisions - the floor still reads
        # `score` - so it would pay the cost and buy nothing. After it, the only
        # thing that changes is order, which is the whole point.
        #
        # Only the top RERANK_TOP_K of the pool are taken in cross-encoder order;
        # the rest keep their fused order. The cross-encoder is more accurate on
        # the head, where the difference matters, and deferring to the cheap
        # pass deeper in costs nothing because nothing past that point is
        # usually read.
        if config.RERANK_ENABLED and len(hits) > 1:
            pool = hits[: max(2, config.RERANK_POOL)]
            reranked = rerank(query, pool)
            head = reranked[: max(1, config.RERANK_TOP_K)]
            promoted = {id(hit) for hit in head}
            rest = [hit for hit in pool if id(hit) not in promoted]
            hits = head + rest + hits[len(pool):]
            if any("rerank" in hit for hit in head):
                mode = f"{mode}+rerank"

        kept: list[dict] = []
        taken: set[int] = set()
        per_source: dict[str, int] = {}

        def take(index: int) -> bool:
            taken.add(index)
            hit = hits[index]
            per_source[hit["source_id"]] = per_source.get(hit["source_id"], 0) + 1
            hit.pop("admitted", None)
            kept.append(hit)
            return len(kept) == limit

        for index, hit in enumerate(hits):
            if per_source.get(hit["source_id"], 0) >= cap:
                continue
            if take(index):
                return _result(kept, best_score, hits, floor, effective_floor,
                               numeric_share, damped, mode)

        # Backfill: a single-source notebook would otherwise be capped at
        # MAX_PER_SOURCE chunks regardless of TOP_K. The cap is a preference for
        # diversity, never a reason to return fewer passages than asked for.
        for index, hit in enumerate(hits):
            if len(kept) >= limit:
                break
            if index not in taken:
                take(index)

        return _result(kept, best_score, hits, floor, effective_floor,
                       numeric_share, damped, mode)


def _result(kept, best_score, hits, floor, effective_floor, numeric_share, damped, mode) -> SearchResult:
    return SearchResult(
        hits=kept,
        best_score=best_score,
        considered=len(hits),
        min_score=floor,
        floor_applied=effective_floor,
        numeric_share=round(numeric_share, 4),
        damped=damped,
        mode=mode,
    )


store = VectorStore()