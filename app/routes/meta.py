"""Everything that describes the app rather than its data.

`/api/status` is the only endpoint that takes no action, and that is deliberate:
it is what the UI shows in the top bar and what a reviewer hits first. It
reports every knob that changes behaviour - provider, model, embedding model,
the relevance floor, hybrid search, numeric damping - plus the current tracing
state, so "is anything leaving this machine?" is answered by the app rather than
by reading `.env`.
"""

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from .. import config, tracing
from ..parsers import SUPPORTED
from ..payloads import base_url, resolve_session
from ..store import store

router = APIRouter(tags=["meta"])


@router.get("/api/status")
def status(request: Request, session_id: str | None = None) -> dict:
    session = resolve_session(session_id)
    web_available, web_reason = config.web_search_available()
    return {
        "provider": config.resolved_provider(),
        "model": config.resolved_model(),
        "embed_model": config.EMBED_MODEL,
        "min_score": config.MIN_SCORE,
        "min_ratio": config.MIN_RATIO,
        "top_k": config.TOP_K,
        "max_per_source": config.MAX_PER_SOURCE,
        "hybrid": config.HYBRID_ENABLED,
        "numeric_damping": config.NUMERIC_DAMPING,
        "tracing": tracing.status(),
        # Whether the web-search card is live, and why not when it is not. The UI
        # shows the reason rather than a dead control, for the same reason this
        # endpoint reports the relevance floor: a reviewer should not have to read
        # .env to find out why a feature is unavailable.
        "web_search": {
            "available": web_available,
            "reason": web_reason,
            "model": config.WEB_SEARCH_MODEL,
            "max_pages": config.WEB_SEARCH_MAX_PAGES,
        },
        "supported_types": sorted(SUPPORTED),
        "session": {"id": str(session.id), "name": session.name},
        "api_base": base_url(request),
        **store.stats(str(session.id)),
    }


@router.get("/")
def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")