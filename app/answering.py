"""The ask pipeline: question in, evidence-backed answer out.

This is the app's core, separated from the HTTP layer for two reasons.

First, so the ordering is readable in one place. The sequence below - retrieve,
apply a relevance floor, refuse or generate, persist, measure - *is* the
grounding policy. When that policy lived inside a FastAPI handler it was mixed
with request parsing and response shaping, and reading it meant skipping over
code that had nothing to do with it.

Second, so the layers have honest dependencies. This module knows nothing about
requests, status codes or JSON. It raises `LLMUnavailable` and lets the route
translate that into a 503, because "the provider was unreachable" is a fact
about the world and "503" is a fact about HTTP.

What this module deliberately does not do: retry, fall back to a second
provider, or answer from anything but the retrieved passages. Each of those
would be a reasonable product decision and none of them belongs in a module
whose whole job is to make one policy legible.
"""

import time

from . import config, db, tracing, usage
from .llm import NO_MATCH, Grounded, LLMUnavailable, answer as run_answer
from .llm import last_usage
from .llm import summarize as run_summary
from .payloads import evidence
from .store import store


def ask(question: str, session: db.SessionRow) -> dict:
    """Answer `question` from `session`'s sources, or refuse.

    Returns the `answer` / `citations` / `evidence` triple the route serialises
    unchanged.
    """
    question = question.strip()
    session_id = str(session.id)

    # History is read from the session, never from the request body: a client
    # that supplies its own turns could otherwise inject context the server did
    # not record.
    history = db.recent_messages(session_id, config.HISTORY_TURNS)

    # Retrieval and generation are traced as separate spans, and that separation
    # is the reason to trace at all: a bad answer caused by bad retrieval is
    # indistinguishable from a bad answer caused by a bad model unless the two
    # steps are separable. Off by default, and no document text unless asked.
    with tracing.ask_span(question, session_id) as span:
        # Timed on its own: retrieval is local CPU work and generation is a
        # network call, so a single total would hide which one was slow.
        started = time.perf_counter()
        with tracing.retrieval_span(question) as retrieval:
            search = store.search_detailed(question, session_id=session_id)
            retrieval_ms = (time.perf_counter() - started) * 1000
            retrieval.record(
                best_score=search.best_score,
                min_score=search.min_score,
                relevant=search.relevant,
                considered=search.considered,
                returned=len(search.hits),
            )

        # Relevance floor. Passing passages the model would have to guess from is
        # how a grounded assistant turns into a confident liar, so stop here.
        if not search.relevant:
            return _refuse(question, session_id, search, retrieval_ms, span)

        with tracing.generation_span(question, search.hits) as generation:
            # Raises LLMUnavailable; the route turns that into a 503. Swallowing
            # it here would mean returning a 200 with no answer in it.
            audit = run_answer(question, search.hits, history)
            reported = last_usage()
            generation.record(
                cited=audit.cited,
                invalid=audit.invalid,
                ungrounded=audit.ungrounded,
                passages=audit.passages,
                **reported.as_dict(),
            )

        cost = usage.Request(
            model=reported.model or config.resolved_model(),
            usage=reported,
            retrieval_ms=retrieval_ms,
            generation_ms=reported.latency_ms,
            passages_sent=audit.passages,
            context_chars=sum(len(str(h.get("text") or "")) for h in search.hits),
            verdict="answered",
        )
        summary = evidence(search, audit, "answered", cost)

        # The evidence is persisted, not merely returned. The `[n]` markers are
        # inside `content`, so writing the text without these leaves a reopened
        # session showing markers that point at nothing - which is exactly the
        # bug where citations stopped being clickable after a reload.
        db.add_message(session_id, "user", question)
        db.add_message(
            session_id,
            "assistant",
            audit.text,
            citations=search.hits,
            evidence=summary,
        )
        db.touch_session(session_id)

        span.record(
            verdict="answered",
            cited=audit.cited,
            ungrounded=audit.ungrounded,
            cost=cost.as_dict(),
        )
        return {
            "answer": audit.text,
            "citations": search.hits,
            "evidence": summary,
        }


def _refuse(question, session_id, search, retrieval_ms, span) -> dict:
    """The no-match path.

    A refusal is still persisted, and so is the reason for it. A refusal has no
    citations by definition, so the only thing that would be lost by storing an
    empty list is nothing; but the evidence carries the score that failed the
    floor, and a transcript that says "no match" without saying "0.00 against a
    0.25 floor" cannot be debugged a week later.
    """
    # The model was not called, so the cost is a measured zero rather than an
    # absent field. "Retrieval ran, nothing was generated" is exactly the kind of
    # fact that gets lost when a metric is omitted instead of set to zero.
    refusal = usage.not_called(
        config.resolved_model(), retrieval_ms, "no_match", len(search.hits)
    )
    summary = evidence(
        search, Grounded(NO_MATCH, [], [], 0), "no_match", refusal
    )

    db.add_message(session_id, "user", question)
    db.add_message(
        session_id, "assistant", NO_MATCH, citations=[], evidence=summary
    )
    db.touch_session(session_id)

    span.record(verdict="no_match", cost=refusal.as_dict())
    return {"answer": NO_MATCH, "citations": [], "evidence": summary}


def summarize(instruction: str, session: db.SessionRow) -> dict:
    """Summarise a session's sources, honouring an optional instruction."""
    session_id = str(session.id)
    query = instruction.strip() or "key points, themes and conclusions"
    search = store.search_detailed(query, session_id=session_id, top_k=12)

    if not search.relevant:
        # Don't spend an LLM call on passages we already judged irrelevant, and
        # don't imply an empty session when the real problem is a bad query.
        empty = not store.stats(session_id)["sources"]
        message = (
            "Nothing to summarize yet - upload a source first."
            if empty
            else "None of the indexed sources match that instruction. Try a broader topic."
        )
        return {
            "summary": message,
            "citations": [],
            "evidence": evidence(search, Grounded(message, [], [], 0), "no_match"),
        }

    audit = run_summary(search.hits, instruction)
    return {
        "summary": audit.text,
        "citations": search.hits,
        "evidence": evidence(search, audit, "answered"),
    }


# Re-exported so callers do not have to know that `summarize` above is the
# pipeline's and `llm.summarize` is the provider's. Two functions with one name
# in one module is a trap, and this alias is the smaller evil.
__all__ = ["ask", "summarize", "LLMUnavailable"]