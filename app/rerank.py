"""Second-stage re-ranking: a cross-encoder reads the query and the chunk together.

Why a second stage at all: the first stage - dense cosine plus BM25, fused by
reciprocal rank fusion - scores the query and the chunk *separately* and combines
two numbers. It cannot tell that a chunk shares every keyword with the question
while answering a different one. A cross-encoder reads them jointly, so it can,
and it is the standard fix for exactly that failure.

Why it is not the first stage: it scores one (query, chunk) pair at a time, which
means it cannot shortlist. Ranking a whole session's chunks with it would be
orders of magnitude more expensive than the vector search it replaces, so it runs
on the fused shortlist - recall from the cheap pass, precision from the expensive
one.

Off by default until it is measured, and the failure mode is written down: the
model downloads on first use (~90 MB), and if it cannot be reached the request
must still answer. A re-ranker that can block retrieval is worse than no
re-ranker, so every failure here degrades to "leave the order alone".
"""

from __future__ import annotations

import threading

from . import config

_lock = threading.Lock()
_model = None
_checked = False


def _encoder():
    """The cross-encoder, or None if it is unavailable. Resolved once.

    Importing and loading a model costs real time, and a machine with no network
    is not going to fix itself on the next query, so the failure is cached along
    with the success. `None` means "reranking is off for this process", which is
    reported rather than retried.
    """
    global _model, _checked
    with _lock:
        if _checked:
            return _model
        _checked = True
        if not config.RERANK_ENABLED:
            return None
        try:
            from sentence_transformers import CrossEncoder

            _model = CrossEncoder(config.RERANK_MODEL)
        except Exception:  # noqa: BLE001
            # Observability rule applied to ranking: a feature that can take the
            # request down is worse than the feature being absent.
            _model = None
        return _model


def available() -> bool:
    """Whether re-ranking will actually run on the next query."""
    return _encoder() is not None


def rerank(query: str, hits: list[dict]) -> list[dict]:
    """Reorder `hits` by cross-encoder relevance. Never raises.

    Only the order changes. Each hit's `score` stays the dense cosine value,
    because that is the scale MIN_SCORE was calibrated on and the one the floor
    already used; swapping it for a cross-encoder logit would make the reported
    number mean two different things depending on whether reranking ran. The
    re-ranker's own score is added alongside as `rerank`, so the reason for an
    ordering is visible rather than implied.
    """
    if not hits or len(hits) < 2:
        return hits

    model = _encoder()
    if model is None:
        return hits

    pool = hits[: max(2, config.RERANK_POOL)]
    try:
        scores = model.predict([(query, hit["text"]) for hit in pool])
    except Exception:  # noqa: BLE001
        return hits

    for hit, score in zip(pool, scores):
        hit["rerank"] = round(float(score), 4)
    # `zip` truncates on a short prediction, and a hit left without `rerank`
    # would sort to the end through `None`; -inf keeps a hit we failed to score
    # in its existing place relative to scored ones.
    for hit in pool:
        hit.setdefault("rerank", float("-inf"))

    pool.sort(key=lambda hit: -hit["rerank"])
    return pool + hits[len(pool):]


def status() -> dict:
    """What reranking is doing, for /api/status. Reported, not assumed."""
    model = _encoder()
    return {
        "enabled": config.RERANK_ENABLED and model is not None,
        "configured": config.RERANK_ENABLED,
        "model": config.RERANK_MODEL,
        "pool": config.RERANK_POOL,
        # "loaded" vs "enabled" is the difference between "the flag is on" and
        # "the model is here". A reviewer seeing enabled=false with
        # configured=true knows the download failed.
        "loaded": model is not None,
        "reason": None if model is not None else (
            "disabled by RERANK_ENABLED=0" if not config.RERANK_ENABLED
            else "model unavailable (download or import failed)"
        ),
    }
