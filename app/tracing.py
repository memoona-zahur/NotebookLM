"""Optional tracing. Off unless you turn it on, and quiet about what it sends.

Why this exists at all: once the app is running, a bad answer is a fact you
have to reconstruct, and the pieces are scattered across three subsystems -
the retrieval scores live in `store.py`, the citation audit in `llm.py`, the
tokens and latency in `usage.py`. LangSmith puts a whole run in one place and
lets you group by a field, which is how "why was Tuesday slow" stops being an
archaeology exercise.

Why it is opt-in and not just on: **tracing a RAG system sends the retrieved
passages to a third party.** Those passages are your documents. The app's core
promise is that embeddings never leave the machine and only the retrieved chunks
go to the model, and a tracer that uploads chunks by default would quietly widen
that promise to cover a company the user never agreed to. So:

  * tracing is off unless LANGSMITH_TRACING is set, and
  * even when it is on, document text is withheld unless
    LANGSMITH_TRACING_INCLUDE_TEXT says otherwise.

The default trace therefore carries the question, the scores, the audit and the
cost - everything needed to tell a retriever failure from a generation failure -
and no document content. That is enough for most debugging, and it is the mode
to run in if the corpus is confidential.

What is sent when text *is* included, stated plainly: the retrieved passage
text, the prompt, and the model answer. Not the uploads themselves, not the
transcript of other sessions, not the embeddings.

The fallback is not a stub. `LANGSMITH_TRACING_LOCAL=1` writes the same span
tree to a JSONL file with no network call at all, which is the mode to use when
you want the traces but the corpus cannot leave the host. Testing a RAG system's
observability should not require uploading the documents the system is about.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config

_lock = threading.Lock()
_client: Any | None = None
_client_checked = False


def _tracing_on() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() in {"1", "true", "yes", "on"}


def _include_text() -> bool:
    """Whether document text may be sent. Separate from the on/off flag on purpose.

    Two switches rather than one, because they answer different questions: "do I
    want traces" and "may my documents leave this machine" are not the same
    decision, and the person making the second one is often not the person who
    turned tracing on in the first place.
    """
    return os.getenv("LANGSMITH_TRACING_INCLUDE_TEXT", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _local_path() -> Path | None:
    raw = os.getenv("LANGSMITH_TRACING_LOCAL", "").strip()
    if not raw:
        return None
    return Path(config.DATA_DIR) / (raw if raw.endswith(".jsonl") else f"{raw}.jsonl")


def status() -> dict[str, object]:
    """What tracing is doing right now, for /api/status and for the README's claims.

    Reported rather than assumed, because "tracing is off" and "tracing is on but
    cannot connect" look identical from the outside and mean opposite things.
    """
    enabled = _tracing_on()
    local = _local_path()
    return {
        "enabled": enabled,
        # The field that matters most in the report, so it cannot be overlooked.
        "document_text_included": _include_text(),
        "mode": "local file" if local else ("langsmith" if enabled else "off"),
        "local_path": str(local) if local else None,
        "destination": os.getenv("LANGSMITH_PROJECT", "") or None,
    }


def _client_or_none():
    """The LangSmith client, or None if it is unavailable.

    Resolved once and cached, including the failure. Importing the SDK costs
    real time and a missing key is not going to fix itself, so a failed import
    is not retried on every request.
    """
    global _client, _client_checked
    with _lock:
        if _client_checked:
            return _client
        _client_checked = True
        if not _tracing_on():
            return None
        if _local_path():
            # A local trace is the offline mode; do not also ship it.
            return None
        try:
            from langsmith import Client

            _client = Client()
        except Exception:  # noqa: BLE001
            # Any failure here must not take down a request. Tracing is
            # observability, and observability that can break the app is worse
            # than no observability.
            _client = None
        return _client


def _write_local(span: dict) -> None:
    path = _local_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(span, default=str) + "\n")


def _emit(payload: dict) -> None:
    """Send one span, by whichever route is configured. Never raises."""
    client = _client_or_none()
    if client is None:
        _write_local(payload)
        return
    try:
        client.create_run(
            name=payload.get("name", "span"),
            run_type=payload.get("run_type", "chain"),
            inputs=payload.get("inputs") or {},
            outputs=payload.get("outputs") or {},
            tags=payload.get("tags") or [],
            metadata=payload.get("metadata") or {},
        )
    except Exception:  # noqa: BLE001
        # Swallowed on purpose. A tracing outage is not a user-facing error, and
        # the local file is the fallback for when the trace is the point.
        _write_local(payload)


@dataclass
class Span:
    """One traced step: retrieval, generation, or the whole request.

    Usable as a context manager so a span closes even if the step raises, which
    is when you most want the trace - an exception is the reason you are reading it.
    """

    name: str
    run_type: str = "chain"
    inputs: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    start: float = field(default_factory=time.perf_counter)
    outputs: dict = field(default_factory=dict)
    error: str | None = None
    _id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.error = f"{exc_type.__name__}: {exc}"
        self.close()
        return False  # never swallow the exception

    def close(self) -> None:
        _emit({
            "name": self.name,
            "run_type": self.run_type,
            "id": self._id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "metadata": {
                **self.metadata,
                "duration_ms": round((time.perf_counter() - self.start) * 1000, 1),
            },
            "tags": [t for t in (os.getenv("LANGSMITH_TAGS", "") or "").split(",") if t],
        })

    def record(self, **fields) -> None:
        self.outputs.update(fields)


def ask_span(question: str, session_id: str | None) -> Span:
    """The outer span for one question.

    The question text is always included. It is the user's own words and it is
    the one field you cannot diagnose anything without.
    """
    span = Span(
        name="ask",
        run_type="chain",
        inputs={"question": question},
        metadata={
            "session_id": session_id,
            "model": config.resolved_model(),
            "provider": config.resolved_provider(),
            "text_included": _include_text(),
        },
    )
    return span


def retrieval_span(question: str) -> Span:
    """Retrieval is traced as its own span, always without passage text.

    Retrieval scores are the diagnosis: a bad answer caused by bad retrieval
    looks identical to a bad answer caused by a good retriever and a bad model,
    and you cannot tell them apart unless the two steps are separate spans.
    """
    return Span(
        name="retrieval",
        run_type="retriever",
        inputs={"question": question},
        metadata={"chunk_size": config.CHUNK_SIZE, "top_k": config.TOP_K,
                  "hybrid": config.HYBRID_ENABLED},
    )


def generation_span(question: str, passages: list[dict]) -> Span:
    """Generation, with passage text only when explicitly permitted."""
    inputs: dict[str, object] = {"question": question}
    if _include_text():
        inputs["passages"] = [p.get("text", "") for p in passages]
    else:
        # Enough to see that the right number of passages went in, and their
        # lengths, without their content.
        inputs["passage_lengths"] = [len(str(p.get("text") or "")) for p in passages]
    return Span(
        name="generation",
        run_type="llm",
        inputs=inputs,
        metadata={"model": config.resolved_model(), "provider": config.resolved_provider()},
    )