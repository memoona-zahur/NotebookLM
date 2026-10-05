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
        "supported_types": sorted(SUPPORTED),
        "session": {"id": str(session.id), "name": session.name},
        "api_base": base_url(request),
        **store.stats(str(session.id)),
    }


@router.get("/")
def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")