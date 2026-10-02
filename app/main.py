from contextlib import asynccontextmanager
from pathlib import Path
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, db
from .parsers import SUPPORTED
from .store import SearchResult, store

ALLOWED = SUPPORTED

MAX_QUESTION_CHARS = 4000
MAX_SESSION_NAME_CHARS = 120

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Applied on every boot and idempotent, so an existing volume is untouched
    # and a fresh one is usable without a manual migration step.
    db.init_schema()
    db.ensure_default_session()
    yield
    # Without this the pool's connections are only reclaimed when the process
    # dies, which matters because uvicorn runs several workers under a reload.
    db.close_pool()


app = FastAPI(title="Local NotebookLM", version="2.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    """A question plus the session it belongs to.

    There is deliberately no `history` field. The transcript is read from the
    database on the server, so a client cannot inject turns it never asked -
    including a forged `system` turn aimed at the grounding instructions.
    """

    question: str = Field(max_length=MAX_QUESTION_CHARS)


class SummarizeRequest(BaseModel):
    instruction: str = Field(default="", max_length=2000)


class SessionRequest(BaseModel):
    name: str = Field(default="", max_length=MAX_SESSION_NAME_CHARS)


def _session_or_404(session_id: str) -> db.SessionRow:
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session


def _base_url(request: Request) -> str:
    """Origin the browser should use to call the API.

    Taken from the request so the app works unchanged behind a proxy or on a
    different port, with BASE_URL as an override.
    """
    if config.BASE_URL:
        return config.BASE_URL
    return str(request.base_url).rstrip("/")


def _evidence(search: SearchResult, audit, verdict: str) -> dict:
    """Combine retrieval stats and the citation audit into one honest summary."""
    return {
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
    }


def _session_payload(request: Request, session: db.SessionRow) -> dict:
    return {
        "id": str(session.id),
        "name": session.name,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "history": db.recent_messages(str(session.id), config.HISTORY_TURNS),
        "api_base": _base_url(request),
    }


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@app.get("/api/sessions")
def list_sessions() -> dict:
    # No history here: listing every session with its transcript would grow
    # without bound. The UI fetches a session when it selects one.
    return {
        "sessions": [
            {"id": str(s.id), "name": s.name,
             "created_at": s.created_at, "updated_at": s.updated_at}
            for s in db.list_sessions()
        ]
    }


@app.post("/api/sessions")
def create_session(request: Request, payload: SessionRequest) -> dict:
    return _session_payload(request, db.create_session(payload.name))


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, request: Request) -> dict:
    return _session_payload(request, _session_or_404(session_id))


@app.patch("/api/sessions/{session_id}")
def rename_session(session_id: str, request: Request, payload: SessionRequest) -> dict:
    session = db.rename_session(session_id, payload.name)
    if session is None:
        raise HTTPException(404, "Session not found")
    return _session_payload(request, session)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    _session_or_404(session_id)
    # Remove the uploads belonging to this session before the rows cascade away;
    # afterwards there is nothing left to say which files they were.
    store.clear(session_id)
    db.delete_session(session_id)
    store.forget(session_id)
    return {"deleted": session_id}


@app.post("/api/sessions/{session_id}/messages/clear")
def clear_history(session_id: str) -> dict:
    _session_or_404(session_id)
    db.clear_messages(session_id)
    return {"cleared": session_id}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


@app.get("/api/status")
def status(request: Request, session_id: str | None = None) -> dict:
    session = (
        _session_or_404(session_id) if session_id else db.ensure_default_session()
    )
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
        "supported_types": sorted(ALLOWED),
        "session": {"id": str(session.id), "name": session.name},
        "api_base": _base_url(request),
        **store.stats(str(session.id)),
    }


@app.post("/api/sources")
async def add_source(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = None,
) -> dict:
    original = Path(file.filename or "upload")
    suffix = original.suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type: {suffix or 'unknown'}")

    session = (
        _session_or_404(session_id) if session_id else db.ensure_default_session()
    )

    target = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(await file.read())

    try:
        source = store.add(target, display_name=original.name, session_id=str(session.id))
    except Exception as exc:  # noqa: BLE001
        # A file we could not parse is not worth keeping; one we parsed is
        # retained so the session can be re-indexed without a re-upload.
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read file: {exc}") from exc

    return {"source": vars(source), **store.stats(str(session.id))}


@app.delete("/api/sources/{source_id}")
def delete_source(source_id: str, session_id: str | None = None) -> dict:
    session = (
        _session_or_404(session_id) if session_id else db.ensure_default_session()
    )
    if not store.remove(source_id, str(session.id)):
        raise HTTPException(404, "Source not found")
    return store.stats(str(session.id))


@app.delete("/api/sources")
def delete_all(session_id: str | None = None) -> dict:
    session = (
        _session_or_404(session_id) if session_id else db.ensure_default_session()
    )
    store.clear(str(session.id))
    return store.stats(str(session.id))


# ---------------------------------------------------------------------------
# Ask / summarize
# ---------------------------------------------------------------------------


@app.post("/api/ask")
def ask(request: Request, payload: AskRequest, session_id: str | None = None) -> dict:
    from .llm import NO_MATCH, Grounded, LLMUnavailable, answer as run_answer

    question = payload.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty")

    session = (
        _session_or_404(session_id) if session_id else db.ensure_default_session()
    )

    # History is read from the session, never from the request body: a client
    # that supplies its own turns could otherwise inject context the server did
    # not record.
    history = db.recent_messages(str(session.id), config.HISTORY_TURNS)
    search = store.search_detailed(question, session_id=str(session.id))

    # Relevance floor. Passing passages the model would have to guess from is how
    # a grounded assistant turns into a confident liar, so stop here instead.
    if not search.relevant:
        db.add_message(str(session.id), "user", question)
        db.add_message(str(session.id), "assistant", NO_MATCH)
        db.touch_session(str(session.id))
        return {
            "answer": NO_MATCH,
            "citations": [],
            "evidence": _evidence(search, Grounded(NO_MATCH, [], [], 0), "no_match"),
        }

    try:
        audit = run_answer(question, search.hits, history)
    except LLMUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc

    db.add_message(str(session.id), "user", question)
    db.add_message(str(session.id), "assistant", audit.text)
    db.touch_session(str(session.id))

    return {
        "answer": audit.text,
        "citations": search.hits,
        "evidence": _evidence(search, audit, "answered"),
    }


@app.post("/api/summarize")
def summarize(request: Request, payload: SummarizeRequest, session_id: str | None = None) -> dict:
    from .llm import Grounded, LLMUnavailable, summarize as run_summary

    session = (
        _session_or_404(session_id) if session_id else db.ensure_default_session()
    )
    query = payload.instruction.strip() or "key points, themes and conclusions"
    search = store.search_detailed(query, session_id=str(session.id), top_k=12)

    if not search.relevant:
        # Don't spend an LLM call on passages we already judged irrelevant, and
        # don't imply an empty session when the real problem is a bad query.
        empty = not store.stats(str(session.id))["sources"]
        message = (
            "Nothing to summarize yet - upload a source first."
            if empty
            else "None of the indexed sources match that instruction. Try a broader topic."
        )
        return {
            "summary": message,
            "citations": [],
            "evidence": _evidence(search, Grounded(message, [], [], 0), "no_match"),
        }

    try:
        audit = run_summary(search.hits, payload.instruction)
    except LLMUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc

    return {
        "summary": audit.text,
        "citations": search.hits,
        "evidence": _evidence(search, audit, "answered"),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")