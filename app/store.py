from dataclasses import dataclass
from pathlib import Path

from . import config
from . import db
from . import parsers
from .embeddings import embed_query, embed_texts, score_against_query
from .lexical import BM25, fuse


@dataclass
class Source:
    id: str
    name: str
    kind: str
    pages: int = 0
    chunks: int = 0
    numeric: int = 0


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

    def add(self, path: Path, display_name: str, session_id: str) -> Source:
        blocks = parsers.parse(path)
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
        with db.connection() as conn:
            source = conn.execute(
                "INSERT INTO sources (session_id, name, kind, pages, storage_path) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (session_id, name, path.suffix.lower().lstrip("."),
                 len(blocks), str(path)),
            ).fetchone()
            source_id = source[0]
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
            kind=path.suffix.lower().lstrip("."),
            pages=len(blocks),
            chunks=chunk_count,
            numeric=numeric,
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
        if row[1]:
            Path(row[1]).unlink(missing_ok=True)
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
            if path:
                Path(path).unlink(missing_ok=True)

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

    # -- reads -------------------------------------------------------------

    def sources(self, session_id: str) -> list[Source]:
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT id::text, name, kind, pages, chunk_count, numeric_count "
                "FROM sources WHERE session_id = %s ORDER BY created_at",
                (session_id,),
            ).fetchall()
        return [Source(*row) for row in rows]

    def stats(self, session_id: str) -> dict:
        records = self.sources(session_id)
        return {
            "sources": [vars(s) for s in records],
            "chunks": sum(s.chunks for s in records),
            "numeric": sum(s.numeric for s in records),
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