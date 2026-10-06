"""Uploading, removing and searching inside sources.

The upload path is the one place in the app that accepts a file from the
network, so it is the one place with a size ceiling and a stream rather than a
read. Everything else is bounded by a Pydantic field or by a query parameter.

Web search also accepts a network request, and it is bounded differently: the
query is a Pydantic field, but the *responses* are bounded in
`app/websearch.py`, because their sizes are not known until they are read.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from .. import config
from ..parsers import SUPPORTED, UnreadableDocument
from ..payloads import resolve_session
from ..store import MAX_TERM_CHARS, store
from ..webingest import ingest

router = APIRouter(prefix="/api", tags=["sources"])

ALLOWED = SUPPORTED

# Uploads are copied to disk in pieces this size rather than one `file.read()`,
# which would hold the whole file in memory with no ceiling on it.
UPLOAD_CHUNK_BYTES = 1024 * 1024


class WebSearchRequest(BaseModel):
    """A web search request.

    Bounded here rather than in `ingest` so an oversized `limit` is rejected
    before a search - the most expensive call in the app - is started, and so the
    ceiling the user is told about is the one enforced.
    """

    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=20)


@router.get("/occurrences")
def occurrences(term: str, request: Request, session_id: str | None = None) -> dict:
    """Where a word appears in this session's sources.

    Scoped to one session like every other read: a highlight must not surface
    text from a document the user never opened.
    """
    term = (term or "").strip()
    if not term:
        raise HTTPException(400, "No word given")
    if len(term) > MAX_TERM_CHARS:
        raise HTTPException(400, f"Word too long (max {MAX_TERM_CHARS} characters)")
    session = resolve_session(session_id)
    return store.find_occurrences(term, str(session.id))


@router.post("/sources")
async def add_source(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = None,
) -> dict:
    original = Path(file.filename or "upload")
    suffix = original.suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type: {suffix or 'unknown'}")

    session = resolve_session(session_id)

    # The stored name is a generated id, not the uploaded one: the user's
    # filename appears only in the database. So a file called
    # `../../../etc/passwd` cannot escape the upload directory, and two uploads
    # of the same name cannot overwrite each other.
    target = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    await _write_upload(file, target)

    try:
        source = store.add(
            target, display_name=original.name, session_id=str(session.id)
        )
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

    # `duplicate` is surfaced at the top level as well as on the source, because
    # it is the one field that changes what this response *means*: not "a source
    # was added" but "your file matched one you already had, nothing changed".
    # A client that only read `chunks` would otherwise report a new upload.
    return {
        "source": vars(source),
        "duplicate": source.duplicate,
        **store.stats(str(session.id)),
    }


@router.post("/sources/web")
def add_web_source(body: WebSearchRequest, session_id: str | None = None) -> dict:
    """Search the web and index what it finds into this notebook.

    Returns 200 even when some or all pages failed. A search that found five
    results and indexed three is a partial success, and reporting it as an error
    would throw away the three real sources the user just gained along with the
    information about the two that did not work - which is in `failed`, per URL,
    and shown to them.

    400 is reserved for "this cannot run at all" (no key, wrong provider, empty
    query), and 502 for the search itself failing upstream. Both are conditions
    where nothing was indexed.
    """
    session = resolve_session(session_id)
    outcome = ingest(body.query, str(session.id), body.limit)
    return {"web_search": outcome.as_dict(), **store.stats(str(session.id))}


@router.post("/sources/{source_id}/reindex")
def reindex_source(source_id: str, session_id: str | None = None) -> dict:
    """Re-read a source already on disk and rebuild its index in place.

    The upload route promises this in its own error comment - a file that
    parsed is kept "so the session can be re-indexed without a re-upload" - and
    until now nothing honoured it. The practical case is a settings change:
    `CHUNK_SIZE` and `EMBED_MODEL` only take effect on new text, so an existing
    notebook silently keeps the chunks it was built with.

    The id and the created_at survive, because past answers' citations point at
    them. 400 when the file is gone or unreadable, 404 when the source is not in
    this session.
    """
    session = resolve_session(session_id)
    try:
        source = store.reindex(source_id, str(session.id))
    except UnreadableDocument as exc:
        raise HTTPException(400, str(exc)) from exc
    if source is None:
        raise HTTPException(404, "Source not found")
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


@router.delete("/sources/{source_id}")
def delete_source(source_id: str, session_id: str | None = None) -> dict:
    session = resolve_session(session_id)
    # Scoped to the session: a source id from another notebook is a 404 here,
    # not a deletion. Without this check any valid id would delete anything.
    if not store.remove(source_id, str(session.id)):
        raise HTTPException(404, "Source not found")
    return store.stats(str(session.id))


@router.delete("/sources")
def delete_all(session_id: str | None = None) -> dict:
    session = resolve_session(session_id)
    store.clear(str(session.id))
    return store.stats(str(session.id))