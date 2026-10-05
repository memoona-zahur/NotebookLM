"""Turning a web search into notebook sources.

This is the second half of `app/websearch.py`, and it is a separate file because
the two halves have opposite failure policies. `websearch` looks and fetches, and
its failures are collected: a bot-walled PDF must not cost the user the four
pages that worked. This module's failures are fatal, because once the bytes are
in hand a page that will not parse is a problem with the *source*, exactly as it
would be for an upload.

Every page found this way goes through `store.add`, the same path an upload
takes. That is the whole design decision: a web page becomes a source rather than
becoming model prose, so it is chunked, embedded, retrieved against, cited with
a page reference, persisted across reloads and deleted by the same delete button
as any uploaded file. Anything that answered straight from search snippets would
be uncitable, unverifiable and gone when the session closed.

Fetched pages are written into `UPLOAD_DIR` under generated names for the same
reason uploads are: the URL is user-influenced, and a URL is not a safe filename.
`_discard` in the store only deletes inside that directory, so deleting a
web-sourced page cleans up its copy and deleting the corpus cannot reach
anything else.
"""

import time
import uuid
from dataclasses import dataclass, field

from fastapi import HTTPException

from . import config, websearch
from .parsers import UnreadableDocument
from .store import store


@dataclass
class IngestedSource:
    """One page that made it into the notebook."""

    id: str
    name: str
    url: str
    kind: str
    chunks: int
    pages: int


@dataclass
class WebIngest:
    """The result of one search: what was added, and what did not work."""

    query: str
    added: list[IngestedSource] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    search_ms: float = 0.0
    fetch_ms: float = 0.0
    cost: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "added": [vars(s) for s in self.added],
            "failed": self.failed,
            "search_ms": round(self.search_ms, 1),
            "fetch_ms": round(self.fetch_ms, 1),
            "cost": self.cost,
            # The count that matters to the user: what is now in the notebook,
            # not what was attempted.
            "added_count": len(self.added),
        }


def ingest(query: str, session_id: str, limit: int) -> WebIngest:
    """Search the web for `query` and index the results into `session_id`.

    Raises `HTTPException` only for failures that make the whole request
    meaningless - no search key, no provider, an empty query. Everything else
    comes back in `failed`, because a search that found five pages and could read
    three of them is a success with a caveat, not an error.
    """
    usable, reason = config.web_search_available()
    if not usable:
        raise HTTPException(400, f"Web search is unavailable: {reason}")
    if not (query or "").strip():
        raise HTTPException(400, "No search query given.")

    try:
        candidates, cost = websearch.find(query, limit)
    except websearch.WebSearchUnavailable as exc:
        raise HTTPException(502, str(exc)) from exc

    result = WebIngest(
        query=query.strip(),
        search_ms=cost.pop("search_ms", 0.0),
        cost=cost,
    )

    if not candidates:
        return result

    started = time.perf_counter()
    for candidate in candidates:
        try:
            page = websearch.fetch(candidate)
        except websearch.WebSearchUnavailable as exc:
            result.failed.append({"url": candidate.url, "reason": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001
            # One page raising something unforeseen must not abandon the rest.
            result.failed.append({"url": candidate.url, "reason": f"Could not read page: {exc}"})
            continue

        try:
            source = _store_page(page, session_id)
        except UnreadableDocument as exc:
            result.failed.append({"url": page.url, "reason": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001
            result.failed.append({"url": page.url, "reason": f"Could not index page: {exc}"})
            continue

        result.added.append(
            IngestedSource(
                id=source.id,
                name=source.name,
                url=page.url,
                kind=source.kind,
                chunks=source.chunks,
                pages=source.pages,
            )
        )

    result.fetch_ms = (time.perf_counter() - started) * 1000
    return result


def _store_page(page: websearch.FetchedPage, session_id: str) -> object:
    """Write a fetched page to disk and index it like an upload.

    The write is inside the try in `ingest`, and the file is removed again if
    parsing refuses the content, so a rejected page leaves nothing behind - the
    same rule the upload route follows.
    """
    target = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{page.suffix}"
    target.write_bytes(page.content)
    try:
        return store.add(target, display_name=page.title, session_id=session_id, url=page.url)
    except Exception:
        target.unlink(missing_ok=True)
        raise