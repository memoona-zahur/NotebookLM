from contextlib import asynccontextmanager
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, db, tracing, usage
from .llm import detect_injection, last_usage
from .parsers import SUPPORTED, UnreadableDocument
from .store import SearchResult, store
from . import store as store_module

ALLOWED = SUPPORTED

MAX_QUESTION_CHARS = 4000

# Uploads are copied to disk in pieces this size rather than one `file.read()`,
# which would hold the whole file in memory with no ceiling on it.
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_SESSION_NAME_CHARS = 120

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Brings a fresh volume up to head and leaves an existing one where it
    # already is, so neither needs a manual migration step before first use.
    db.migrate()
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


def _evidence(
    search: SearchResult,
    audit,
    verdict: str,
    cost: usage.Request | None = None,
) -> dict:
    """Combine retrieval stats, the citation audit, and cost into one summary.

    `cost` is optional because the refusal path has a real measurement to report
    (retrieval happened, the model did not) rather than nothing to say.
    """
    evidence = {
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
        evidence["cost"] = cost.as_dict()
    return evidence


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
        # Reported so "is anything leaving this machine?" is a question the app
        # answers rather than one the user has to answer by reading .env.
        "tracing": tracing.status(),
        "supported_types": sorted(ALLOWED),
        "session": {"id": str(session.id), "name": session.name},
        "api_base": _base_url(request),
        **store.stats(str(session.id)),
    }


@app.get("/api/occurrences")
def occurrences(term: str, request: Request, session_id: str | None = None) -> dict:
    """Where a word appears in this session's sources.

    Scoped to one session like every other read: a highlight must not surface
    text from a document the user never opened.
    """
    term = (term or "").strip()
    if not term:
        raise HTTPException(400, "No word given")
    if len(term) > store_module.MAX_TERM_CHARS:
        raise HTTPException(
            400, f"Word too long (max {store_module.MAX_TERM_CHARS} characters)"
        )
    session = (
        _session_or_404(session_id) if session_id else db.ensure_default_session()
    )
    return store.find_occurrences(term, str(session.id))


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
    # Streamed in bounded chunks and cut off at MAX_UPLOAD_BYTES. `await
    # file.read()` loads the whole upload into memory with nothing stopping it,
    # so one large file could take the process down.
    await _write_upload(file, target)

    try:
        source = store.add(target, display_name=original.name, session_id=str(session.id))
    except UnreadableDocument as exc:
        # Readable file, unusable content. The parser's message says what to do
        # about it, so it is not flattened into a generic parse failure.
        target.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        # A file we could not parse is not worth keeping; one we parsed is
        # retained so the session can be re-indexed without a re-upload.
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read file: {exc}") from exc

    return {"source": vars(source), **store.stats(str(session.id))}


async def _write_upload(file: UploadFile, target: Path) -> None:
    """Write the upload to disk in bounded chunks, refusing oversized files.

    The partial file is removed on any failure, including the oversize case, so
    a rejected upload leaves nothing behind.
    """
    limit = config.MAX_UPLOAD_BYTES
    written = 0
    try:
        with target.open("wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(
                        413,
                        f"File is larger than {limit // (1024 * 1024)} MB. "
                        f"Split it, or raise MAX_UPLOAD_BYTES.",
                    )
                out.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise


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
    # Retrieval and generation are traced as separate spans, and that separation
    # is the reason to trace at all: a bad answer caused by bad retrieval is
    # indistinguishable from a bad answer caused by a bad model unless the two
    # steps are separable. Off by default, and no document text unless asked.
    with tracing.ask_span(question, str(session.id)) as span:
        # Timed on its own: retrieval is local CPU work and generation is a
        # network call, so a single total would hide which one was slow.
        started = time.perf_counter()
        with tracing.retrieval_span(question) as retrieval:
            search = store.search_detailed(question, session_id=str(session.id))
            retrieval_ms = (time.perf_counter() - started) * 1000
            retrieval.record(
                best_score=search.best_score,
                min_score=search.min_score,
                relevant=search.relevant,
                considered=search.considered,
                returned=len(search.hits),
                injection=detect_injection(search.hits),
            )

        # Relevance floor. Passing passages the model would have to guess from is
        # how a grounded assistant turns into a confident liar, so stop here.
        if not search.relevant:
            db.add_message(str(session.id), "user", question)
            db.add_message(str(session.id), "assistant", NO_MATCH)
            db.touch_session(str(session.id))
            refusal = usage.not_called(
                config.resolved_model(),
                retrieval_ms,
                "no_match",
                len(search.hits),
            )
            span.record(verdict="no_match", cost=refusal.as_dict())
            return {
                "answer": NO_MATCH,
                "citations": [],
                "evidence": _evidence(
                    search, Grounded(NO_MATCH, [], [], 0), "no_match", refusal
                ),
            }

        with tracing.generation_span(question, search.hits) as generation:
            try:
                audit = run_answer(question, search.hits, history)
            except LLMUnavailable as exc:
                raise HTTPException(503, str(exc)) from exc
            reported = last_usage()
            generation.record(
                cited=audit.cited,
                invalid=audit.invalid,
                ungrounded=audit.ungrounded,
                passages=audit.passages,
                **reported.as_dict(),
            )

        db.add_message(str(session.id), "user", question)
        db.add_message(str(session.id), "assistant", audit.text)
        db.touch_session(str(session.id))

        cost = usage.Request(
            model=reported.model or config.resolved_model(),
            usage=reported,
            retrieval_ms=retrieval_ms,
            generation_ms=reported.latency_ms,
            passages_sent=audit.passages,
            context_chars=sum(len(str(h.get("text") or "")) for h in search.hits),
            verdict="answered",
        )
        span.record(
            verdict="answered",
            cited=audit.cited,
            ungrounded=audit.ungrounded,
            cost=cost.as_dict(),
        )
        return {
            "answer": audit.text,
            "citations": search.hits,
            "evidence": _evidence(search, audit, "answered", cost),
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


# The built bundle references its assets by absolute path (/assets/app.js), so
# they are served from /assets as well as from /static. Missing on purpose would
# mean a blank page with no explanation, so this fails loudly at boot instead.
if not config.ASSET_DIR.is_dir():
    raise RuntimeError(
        f"No built frontend at {config.ASSET_DIR}. "
        "Run `npm install && npm run build` in frontend/."
    )

app.mount("/assets", StaticFiles(directory=config.ASSET_DIR), name="assets")