import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import config
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
class Notebook:
    sources: dict[str, Source] = field(default_factory=dict)

    def stats(self) -> dict:
        return {
            "sources": [vars(s) for s in self.sources.values()],
            "chunks": sum(s.chunks for s in self.sources.values()),
            "numeric": sum(s.numeric for s in self.sources.values()),
        }


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
    def __init__(self) -> None:
        from qdrant_client import QdrantClient

        self._client = QdrantClient(":memory:")
        self._ready = False
        self._lock = threading.Lock()
        self.notebook = Notebook()
        self._lexical = BM25()
        self._chunks: dict[str, str] = {}
        self._meta: dict[str, dict] = {}

    def stats(self) -> dict:
        return self.notebook.stats()

    def _ensure(self) -> None:
        if self._ready:
            return
        from qdrant_client.models import Distance, VectorParams

        with self._lock:
            if self._ready:
                return
            self._client.create_collection(
                collection_name=config.COLLECTION,
                vectors_config=VectorParams(
                    size=config.EMBED_DIM, distance=Distance.COSINE
                ),
            )
            self._ready = True

    def add(self, path: Path, display_name: str | None = None) -> Source:
        from qdrant_client.models import PointStruct

        self._ensure()
        blocks = parsers.parse(path)
        source_id = uuid.uuid4().hex[:12]
        name = display_name or path.name

        payloads: list[dict] = []
        vectors: list[list[float]] = []
        numeric = 0

        for position, block in enumerate(blocks):
            if not block.text.strip():
                continue
            heavy = parsers.is_numeric_heavy(block.text)
            if heavy:
                numeric += 1
            payloads.append(
                {
                    "source_id": source_id,
                    "source": name,
                    "page": block.page,
                    "heading": block.heading,
                    "text": block.text,
                    "numeric_heavy": heavy,
                    "chunk_id": f"{source_id}:{position}",
                }
            )

        if payloads:
            vectors = embed_texts([p["text"] for p in payloads])
            self._client.upsert(
                collection_name=config.COLLECTION,
                points=[
                    PointStruct(id=uuid.uuid4().hex, vector=vector, payload=payload)
                    for vector, payload in zip(vectors, payloads)
                ],
            )
            self._chunks.update({p["chunk_id"]: p["text"] for p in payloads})
            self._meta.update({p["chunk_id"]: p for p in payloads})
            self._reindex_lexical()

        source = Source(
            id=source_id,
            name=name,
            kind=path.suffix.lower().lstrip("."),
            pages=sum(1 for block in blocks if block.page),
            chunks=len(payloads),
            numeric=numeric,
        )
        self.notebook.sources[source_id] = source
        return source

    def remove(self, source_id: str) -> bool:
        if source_id not in self.notebook.sources:
            return False
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        self._ensure()
        self._client.delete(
            collection_name=config.COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="source_id", match=MatchValue(value=source_id))]
                )
            ),
        )
        del self.notebook.sources[source_id]
        prefix = f"{source_id}:"
        self._chunks = {k: v for k, v in self._chunks.items() if not k.startswith(prefix)}
        self._meta = {k: v for k, v in self._meta.items() if not k.startswith(prefix)}
        self._reindex_lexical()
        return True

    def clear(self) -> None:
        self._ensure()
        self._client.delete_collection(collection_name=config.COLLECTION)
        self._ready = False
        self.notebook = Notebook()
        self._chunks = {}
        self._meta = {}
        self._lexical = BM25()

    def _reindex_lexical(self) -> None:
        ids = list(self._chunks)
        self._lexical.fit(ids, [self._chunks[i] for i in ids])

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Relevant chunks only, best first. See search_detailed for metadata."""
        return self.search_detailed(query, top_k).hits

    def search_detailed(
        self,
        query: str,
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

        if not self._ready or not self.notebook.sources:
            return SearchResult(
                hits=[], best_score=0.0, considered=0, min_score=floor
            )

        pool = max(limit, limit * 4)
        points = self._client.query_points(
            collection_name=config.COLLECTION,
            query=embed_query(query),
            limit=pool,
            with_payload=True,
        ).points

        # Candidates are the union of what each retriever likes. Taking only the
        # dense pool would hide the chunk that lexical search is most confident
        # about, which in a long document is often far outside the dense top-k.
        by_id: dict[str, dict] = {}

        def remember(chunk_id: str, payload: dict, score: float) -> None:
            if chunk_id in by_id:
                by_id[chunk_id]["score"] = round(score, 4)
                return
            by_id[chunk_id] = {
                "chunk_id": chunk_id,
                "text": payload["text"],
                "source": payload["source"],
                "page": payload["page"],
                "source_id": payload["source_id"],
                "heading": payload.get("heading", ""),
                "numeric_heavy": bool(payload.get("numeric_heavy")),
                "score": round(score, 4),
                "support": 0.0,
                "rescued": False,
            }

        for point in points:
            remember(point.payload["chunk_id"], point.payload, float(point.score))

        mode = "dense"
        dense_ranked = sorted(by_id.values(), key=lambda hit: -hit["score"])[:pool]

        if config.HYBRID_ENABLED and self._chunks:
            mode = "hybrid"
            lexical_ranked = self._lexical.top(query, pool)
            # Lexical-only candidates are absent from the dense pool, so their
            # dense score is unknown. It still has to be measured: the decision
            # to admit or reject is made on the dense scale.
            covers = dict(
                (cid, cov) for cid, _s, cov in self._lexical.top(query, len(self._chunks))
            )
            unseen = [
                cid
                for cid, _bm25, _cov in lexical_ranked
                if cid not in by_id and cid in self._meta
            ]
            if unseen:
                query_vector = embed_query(query)
                for chunk_id, score in zip(
                    unseen,
                    score_against_query(query_vector, [self._chunks[cid] for cid in unseen]),
                ):
                    remember(chunk_id, self._meta[chunk_id], score)
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
            # positive; a strong term match with a mediocre dense score is
            # almost always a real answer the embedding under-scored. Both
            # directions are admitted explicitly rather than by moving one
            # global threshold, which the eval corpus shows cannot work.
            # The lexical rescue is only offered in the default regime: a caller
            # that passes a stricter floor than MIN_SCORE is asking for strict,
            # and that request is honoured in full.
            rescue_allowed = floor <= config.DEFAULT_MIN_SCORE
            # A floor of zero is an explicit request for no relevance gating at
            # all, so the support requirement is skipped entirely.
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
                hits=[],
                best_score=0.0,
                considered=len(considered_hits),
                min_score=floor,
                floor_applied=floor,
                numeric_share=round(numeric_share, 4),
                damped=damped,
                mode=mode,
            )

        best_score = max(hit["score"] for hit in ordered)
        # A chunk admitted on lexical evidence has already passed an
        # independent test, so the dense floor must not reject it again.
        # Everything else faces the absolute floor plus a share of the best hit,
        # which prunes the weak tail a single strong match drags along.
        effective_floor = max(floor, best_score * config.MIN_RATIO)

        def admitted(hit: dict) -> bool:
            if hit["rescued"]:
                return hit["score"] >= config.BM25_RESCUE_MIN
            return hit["score"] >= effective_floor

        hits = [hit for hit in ordered if admitted(hit)]
        if not hits:
            return SearchResult(
                hits=[],
                best_score=0.0,
                considered=len(considered_hits),
                min_score=floor,
                floor_applied=effective_floor,
                numeric_share=round(numeric_share, 4),
                damped=damped,
                mode=mode,
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
                return _result(kept, best_score, hits, floor, effective_floor, numeric_share, damped, mode)

        # Backfill: a single-source notebook would otherwise be capped at
        # MAX_PER_SOURCE chunks regardless of TOP_K. The cap is a preference for
        # diversity, never a reason to return fewer passages than asked for.
        for index, hit in enumerate(hits):
            if len(kept) >= limit:
                break
            if index not in taken:
                take(index)

        return _result(kept, best_score, hits, floor, effective_floor, numeric_share, damped, mode)


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
