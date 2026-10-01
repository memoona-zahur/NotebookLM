from pathlib import Path
from typing import Literal
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config
from .parsers import SUPPORTED
from .store import SearchResult, store

ALLOWED = SUPPORTED

MAX_QUESTION_CHARS = 4000
MAX_TURN_CHARS = 8000
MAX_HISTORY_TURNS = 20

app = FastAPI(title="Local NotebookLM", version="1.1.0")


class Turn(BaseModel):
    """One prior chat turn.

    `role` is a Literal, so a client cannot smuggle in a `system` turn to
    override the grounding instructions. Anything else is a 422.
    """

    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_TURN_CHARS)


class AskRequest(BaseModel):
    question: str = Field(max_length=MAX_QUESTION_CHARS)
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)


class SummarizeRequest(BaseModel):
    instruction: str = Field(default="", max_length=2000)


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


@app.get("/api/status")
def status() -> dict:
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
        **store.stats(),
    }


@app.post("/api/sources")
async def add_source(file: UploadFile = File(...)) -> dict:
    original = Path(file.filename or "upload")
    suffix = original.suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type: {suffix or 'unknown'}")

    target = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    target.write_bytes(await file.read())

    try:
        source = store.add(target, display_name=original.name)
    except Exception as exc:  # noqa: BLE001
        target.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read file: {exc}") from exc

    return {"source": vars(source), **store.stats()}


@app.delete("/api/sources/{source_id}")
def delete_source(source_id: str) -> dict:
    if not store.remove(source_id):
        raise HTTPException(404, "Source not found")
    return store.stats()


@app.delete("/api/sources")
def delete_all() -> dict:
    store.clear()
    return store.stats()


@app.post("/api/ask")
def ask(payload: AskRequest) -> dict:
    from .llm import NO_MATCH, Grounded, LLMUnavailable, answer as run_answer

    question = payload.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty")

    search = store.search_detailed(question)

    # Relevance floor. Passing passages the model would have to guess from is how
    # a grounded assistant turns into a confident liar, so stop here instead.
    if not search.relevant:
        return {
            "answer": NO_MATCH,
            "citations": [],
            "evidence": _evidence(search, Grounded(NO_MATCH, [], [], 0), "no_match"),
        }

    try:
        audit = run_answer(question, search.hits, [t.model_dump() for t in payload.history])
    except LLMUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc

    return {
        "answer": audit.text,
        "citations": search.hits,
        "evidence": _evidence(search, audit, "answered"),
    }


@app.post("/api/summarize")
def summarize(payload: SummarizeRequest) -> dict:
    from .llm import Grounded, LLMUnavailable, summarize as run_summary

    query = payload.instruction.strip() or "key points, themes and conclusions"
    search = store.search_detailed(query, top_k=12)

    if not search.relevant:
        # Don't spend an LLM call on passages we already judged irrelevant, and
        # don't imply an empty notebook when the real problem is a bad query.
        empty_notebook = not store.stats()["sources"]
        message = (
            "Nothing to summarize yet - upload a source first."
            if empty_notebook
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
