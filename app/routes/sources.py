"""Uploading, removing and searching inside sources.

The upload path is the one place in the app that accepts a file from the
network, so it is the one place with a size ceiling and a stream rather than a
read. Everything else is bounded by a Pydantic field or by a query parameter.

Web search also accepts a network request, and it is bounded differently: the
query is a Pydantic field, but the *responses* are bounded in
`app/websearch.py`, because their sizes are not known until they are read.
"""

import mimetypes
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import config
from ..parsers import SUPPORTED, UnreadableDocument, count_tokens
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


# Types a browser will execute rather than display. Serving a user's uploaded
# HTML as its own content type would hand that document the app's origin - any
# script in it could then call the API as the person viewing it. Displayed as
# plain text instead, which is what "show me what you indexed" actually wants.
SCRIPTABLE = {
    "text/html",
    "application/xhtml+xml",
    "image/svg+xml",
    "text/xml",
    "application/xml",
}


def _header_filename(name: str) -> str:
    """A display name made safe for one header value.

    The name is user-controlled, and a CR or a quote in it would be header
    injection rather than a download. Everything outside a small allowlist is
    replaced, the result is truncated, and it always falls back to something
    non-empty - an empty filename= is a malformed header.
    """
    safe = re.sub(r"[^A-Za-z0-9 ._()\[\]-]", "_", name).strip(" .") or "source"
    return safe[:120]


@router.get("/sources/{source_id}/file")
def source_file(source_id: str, session_id: str | None = None) -> FileResponse:
    """The original upload behind a source, so indexing can be checked by eye.

    "Indexed" is a claim, and this is the evidence: the document the parser was
    handed, served as it was stored. Web sources are served too - the fetched
    page as well as the link back to it - so the control is not a second class
    of source.

    Two refusals, for different reasons. A source in another session is a 404
    that says nothing about whether it exists, because owning an id is not the
    same as owning the source. A path outside `UPLOAD_DIR` is a 404 as well:
    the path comes from the row rather than the request so there is nothing to
    traverse, but the store is handed paths by its callers, and a boundary
    check costs nothing next to trusting every caller forever.
    """
    session = resolve_session(session_id)
    found = store.original_file(source_id, str(session.id))
    if found is None:
        raise HTTPException(404, "That source is not in this notebook.")
    name, path = found
    try:
        inside = path.resolve().is_relative_to(config.UPLOAD_DIR.resolve())
    except OSError:
        inside = False
    if not inside:
        raise HTTPException(404, "That file cannot be served.")
    if not path.is_file():
        raise HTTPException(404, "The original file is no longer on disk.")

    media_type, _ = mimetypes.guess_type(path.name)
    media_type = media_type or "application/octet-stream"
    if media_type in SCRIPTABLE:
        media_type = "text/plain; charset=utf-8"
    return FileResponse(
        path,
        media_type=media_type,
        headers={
            # Inline, so a PDF opens in the browser's viewer instead of being
            # downloaded - the point of a View button is to look at the thing.
            "Content-Disposition": f'inline; filename="{_header_filename(name)}"',
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/sources/{source_id}/chunks")
def source_chunks(source_id: str, session_id: str | None = None) -> dict:
    """Every chunk of one source, sized, so the split can be checked by eye.

    The whole feature exists because "indexed" hides the part a reader would
    want to see: how their document became 264 pieces, how big each one came
    out, and how close any of them sit to what the model keeps. A count says
    none of that, and re-opening the PDF says less, because the split is not in
    the PDF.

    Wordpieces are counted here rather than stored, so they can never disagree
    with what ingest recorded - `count_tokens` is the same call ingest makes.
    That costs one tokenizer pass per chunk, in a worker thread, which is why
    the route is a plain `def`.
    """
    session = resolve_session(session_id)
    records = store.source_chunks(source_id, str(session.id))
    if records is None:
        raise HTTPException(404, "That source is not in this notebook.")
    for row in records:
        row["wordpieces"] = count_tokens(row["text"])
    return {
        "source_id": source_id,
        "ceiling": {
            "chars": config.CHUNK_SIZE,
            "wordpieces": config.EMBED_MAX_TOKENS,
        },
        "chunks": records,
    }


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