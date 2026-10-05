"""Asking questions.

The handlers here are intentionally thin: parse, delegate to `answering`, map
the one domain error to an HTTP status. All the grounding policy lives in
`app/answering.py`, and keeping this file that small is how you can claim that
without the reader having to take it on faith.
"""

from fastapi import APIRouter, HTTPException, Request

from .. import answering
from ..llm import LLMUnavailable
from ..payloads import resolve_session
from ..schemas import AskRequest, SummarizeRequest

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/ask")
def ask(request: Request, payload: AskRequest, session_id: str | None = None) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(400, "Question is empty")
    session = resolve_session(session_id)
    try:
        return answering.ask(question, session)
    except LLMUnavailable as exc:
        # 503, not 500: the request was fine, the upstream provider was not, and
        # those want different handling from a retry policy.
        raise HTTPException(503, str(exc)) from exc


@router.post("/summarize")
def summarize(
    request: Request, payload: SummarizeRequest, session_id: str | None = None
) -> dict:
    session = resolve_session(session_id)
    try:
        return answering.summarize(payload.instruction, session)
    except LLMUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc