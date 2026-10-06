"""Response shapes shared by more than one route.

These are here rather than in `main.py` because two questions come up whenever
an evidence payload is being assembled - "what exactly does the client learn
about this answer?" and "why is there both a confidence and three scores?" - and
they are answered once, here, instead of in three handlers that could drift.

`evidence` in particular is the app's honesty contract with the client. It is
the single place where retrieval statistics, the citation audit and the token
cost are merged, so there is no way for one route to report a cost and another
to forget.
"""

from fastapi import HTTPException, Request

from . import config, db, usage
from .llm import detect_injection
from .store import SearchResult


def session_or_404(session_id: str) -> db.SessionRow:
    """Load a session, or fail with 404 rather than a KeyError.

    Every session-scoped route starts here, which is what makes the scoping
    auditable: there is one place where "does this session exist" is decided.
    """
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session


def resolve_session(session_id: str | None) -> db.SessionRow:
    """The session a request is about.

    A missing `session_id` is not an error: it falls back to the default
    session, so the very first page load works before anything has been created.
    A *wrong* one is an error, because silently answering against somebody else's
    documents would be worse than refusing.
    """
    return session_or_404(session_id) if session_id else db.ensure_default_session()


def base_url(request: Request) -> str:
    """Origin the browser should use to call the API.

    Taken from the request so the app works unchanged behind a proxy or on a
    different port, with BASE_URL as an override.
    """
    if config.BASE_URL:
        return config.BASE_URL
    return str(request.base_url).rstrip("/")


def evidence(
    search: SearchResult,
    audit,
    verdict: str,
    cost: usage.Request | None = None,
) -> dict:
    """Combine retrieval stats, the citation audit, and cost into one summary.

    `cost` is optional because the refusal path has a real measurement to report
    (retrieval happened, the model did not) rather than nothing to say.
    """
    summary = {
        "verdict": verdict,
        "confidence": search.confidence(),
        "retrieval": search.mode,
        "best_score": search.best_score,
        "min_score": search.min_score,
        "floor_applied": search.floor_applied,
        "numeric_damped": search.damped,
        "numeric_share": search.numeric_share,
        "considered": search.considered,
        "returned": len(search.hits),
        "passages": audit.passages,
        "cited": audit.cited,
        "invalid": audit.invalid,
        "ungrounded": audit.ungrounded,
        # Passages that tried to instruct the model rather than inform it. They
        # are still cited and still counted; this only says the upload was
        # adversarial, so a surprising answer can be traced to the source.
        "injection": detect_injection(search.hits),
    }
    if cost is not None:
        summary["cost"] = cost.as_dict()
        # Latency is recorded here, at the one place every retrieval-bearing
        # path builds its evidence, so answered and refused requests land in the
        # same window. Measuring only the answers that came back would report a
        # P95 over exactly the requests that already succeeded.
        usage.record(cost)
    return summary


def session_payload(request: Request, session: db.SessionRow) -> dict:
    return {
        "id": str(session.id),
        "name": session.name,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "history": db.recent_messages(str(session.id), config.HISTORY_TURNS),
        "api_base": base_url(request),
    }