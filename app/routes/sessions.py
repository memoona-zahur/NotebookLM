"""Session lifecycle: create, list, open, rename, delete, clear history.

A session is the isolation boundary. Every source and every message belongs to
exactly one, and nothing in the app reads across that line - which is the
property that makes "my documents never touch anyone else's" true rather than
aspirational.

Note what deletion has to do in the right order: the files on disk, then the
rows, then the caches. After the rows go there is nothing left to say which
uploads belonged to this session, so the uploads have to be identified first.
"""

from fastapi import APIRouter, HTTPException, Request

from .. import db
from ..payloads import session_or_404, session_payload
from ..schemas import SessionRequest
from ..store import store

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
def list_sessions() -> dict:
    # No history here: listing every session with its transcript would grow
    # without bound. The UI fetches a session when it selects one.
    return {
        "sessions": [
            {
                "id": str(s.id),
                "name": s.name,
                "created_at": s.created_at,
                "updated_at": s.updated_at,
            }
            for s in db.list_sessions()
        ]
    }


@router.post("")
def create_session(request: Request, payload: SessionRequest) -> dict:
    return session_payload(request, db.create_session(payload.name))


@router.get("/{session_id}")
def get_session(session_id: str, request: Request) -> dict:
    return session_payload(request, session_or_404(session_id))


@router.patch("/{session_id}")
def rename_session(session_id: str, request: Request, payload: SessionRequest) -> dict:
    session = db.rename_session(session_id, payload.name)
    if session is None:
        raise HTTPException(404, "Session not found")
    return session_payload(request, session)


@router.delete("/{session_id}")
def delete_session(session_id: str) -> dict:
    session_or_404(session_id)
    # Remove the uploads belonging to this session before the rows cascade away;
    # afterwards there is nothing left to say which files they were.
    store.clear(session_id)
    db.delete_session(session_id)
    store.forget(session_id)
    return {"deleted": session_id}


@router.post("/{session_id}/messages/clear")
def clear_history(session_id: str) -> dict:
    session_or_404(session_id)
    db.clear_messages(session_id)
    return {"cleared": session_id}