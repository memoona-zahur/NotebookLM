"""What one question actually cost.

A grounding system has to answer "why did that answer say it?" and "what does a
question cost?" before anyone will trust it. The retrieval side already reports
`best_score`, `considered` and `returned` on every answer; nothing reported the
generation side, so an answer could be correct, cited, and quietly expensive with
no way to see it.

Three things are tracked, and they answer different questions:

- **Tokens.** What you pay for. Sent, received, and the total, plus what the
  retrieved passages cost to send. Sent tokens dominate, because the context is
  the retrieved passages and the system prompt, not the question.
- **Latency.** Split into retrieval and generation, because they have different
  fixes. Retrieval is CPU-bound and local; generation is a network call. A slow
  answer that is slow in retrieval is a chunking problem, and a slow answer that
  is slow in generation is a provider problem.
- **Whether the model was called at all.** A question whose relevance floor
  rejects it costs nothing and should say so. Reporting a cost for a refusal
  would be misleading.

Providers report token counts differently and some report nothing, so
`estimated` is a separate flag on the reading rather than something blended into
the numbers. A number you cannot attribute to a measurement is worse than no
number, because it looks equally trustworthy.

Usage is read from the provider response, never inferred from string length,
when the provider supplies it. Where it does not (Ollama in some versions),
`estimated` is set and the numbers are a character-count approximation. The
char-per-token ratio is the one OpenAI documents for English text; it is coarse
for code and for other languages, which is why it is labelled rather than hidden.
"""

from __future__ import annotations

import collections
import math
import threading
from dataclasses import dataclass, field

from . import config

# OpenAI's documented approximation for English text. Only used when a provider
# reports no usage at all; a real count always wins.
CHARS_PER_TOKEN = 4


@dataclass
class Usage:
    """The cost of one model call.

    Attributes:
        model: provider model name, so a number can be attributed.
        prompt_tokens: tokens sent, including the system prompt and context.
        completion_tokens: tokens received.
        estimated: True when the counts came from character length rather than
            from the provider, because a provider that reports nothing is not
            evidence of a zero.
        latency_ms: wall time for the call.
        called: False when the relevance floor stopped the request. Then the
            token counts are zero and that zero is the finding.
    """

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False
    latency_ms: float = 0.0
    called: bool = True

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated": self.estimated,
            "latency_ms": round(self.latency_ms, 1),
            "called": self.called,
        }


@dataclass
class Request:
    """One whole question, end to end.

    Latency is kept per stage rather than as one number, because the two stages
    fail for unrelated reasons and a single total hides which one it was.
    """

    model: str
    usage: Usage = field(default_factory=lambda: Usage(model=""))
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    passages_sent: int = 0
    context_chars: int = 0
    verdict: str = "answered"

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "verdict": self.verdict,
            "passages_sent": self.passages_sent,
            "context_chars": self.context_chars,
            "retrieval_ms": round(self.retrieval_ms, 1),
            "generation_ms": round(self.generation_ms, 1),
            "total_ms": round(self.total_ms, 1),
            **self.usage.as_dict(),
        }


def not_called(model: str, retrieval_ms: float, verdict: str, passages: int) -> Request:
    """A question answered without the model, which is the cheapest answer.

    Worth recording rather than omitting: how often the relevance floor holds is a
    direct measure of whether the retriever is discriminating or just refusing.
    """
    return Request(
        model=model,
        usage=Usage(model=model, called=False),
        retrieval_ms=retrieval_ms,
        verdict=verdict,
        passages_sent=passages,
    )


def from_response(
    model: str,
    messages: list[dict],
    response,
    latency_ms: float,
    output: str = "",
) -> Usage:
    """Read usage off an OpenAI-compatible response, falling back to an estimate.

    `_openai_compatible` is shared by Groq and OpenAI, so both expose `usage`
    when they expose it at all. Anything unexpected is treated as "reported
    nothing" rather than propagated as an AttributeError at the call site.
    """
    reported = getattr(response, "usage", None)
    prompt = getattr(reported, "prompt_tokens", None)
    completion = getattr(reported, "completion_tokens", None)

    if isinstance(prompt, int) and isinstance(completion, int):
        return Usage(
            model=model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            estimated=False,
            latency_ms=latency_ms,
        )

    return estimate(model, messages, latency_ms, output)


def estimate(model: str, messages: list[dict], latency_ms: float, output: str = "") -> Usage:
    """Approximate usage from character counts, and say that is what it is.

    Both sides are estimated. Estimating only the prompt would report a
    completion cost of zero, which reads as "the answer was free" - a wrong
    number that understates cost, which is the direction that hides a problem.
    """
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    return Usage(
        model=model,
        # A few tokens of envelope per message for role and separators.
        prompt_tokens=max(1, chars // CHARS_PER_TOKEN) + len(messages),
        completion_tokens=max(1, len(output) // CHARS_PER_TOKEN) if output else 0,
        estimated=True,
        latency_ms=latency_ms,
    )

# -- latency percentiles ---------------------------------------------------

# A mean is the wrong shape for this. One provider hiccup on a 5 ms retrieval
# window moves the average and says nothing about whether the ordinary request
# was fine, while a P95 is exactly the claim a user makes: "it usually answers
# quickly, and the slow ones are this slow".
def _percentile(sorted_values: list[float], quantile: float) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty list.

    Nearest-rank rather than interpolated: with a few hundred samples the
    interpolated figure implies a precision the sample does not have, and
    nearest-rank always returns a value that was actually observed.
    """
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(quantile / 100 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


class LatencyWindow:
    """The last N latency samples, for P50/P95 over the recent past.

    A ring buffer rather than an accumulator because the question is always "is
    it slow *now*". Percentiles over the process lifetime answer "was it slow
    once in March", which is true, unactionable, and the reason most latency
    dashboards get ignored.
    """

    def __init__(self, size: int) -> None:
        self.size = max(1, size)
        self._series: dict[str, collections.deque[float]] = {}
        self._lock = threading.Lock()

    def add(self, **samples: float) -> None:
        """Record one or more stage latencies in milliseconds. Zeros are skipped.

        A zero here is not "instant"; it is "this stage did not run" - a refusal
        has no generation and a greeting has no retrieval. Mixing those into the
        distribution would make generation look faster than it is, which is the
        one direction this metric must not be wrong in.
        """
        with self._lock:
            for name, value in samples.items():
                if value is None or value <= 0:
                    continue
                series = self._series.setdefault(
                    name, collections.deque(maxlen=self.size)
                )
                series.append(float(value))

    def percentiles(self) -> dict:
        with self._lock:
            snapshot = {
                name: sorted(series) for name, series in self._series.items()
            }
        out = {"window": self.size, "samples": {}}
        for name, values in snapshot.items():
            out["samples"][name] = {
                "count": len(values),
                "p50": round(_percentile(values, 50), 1),
                "p95": round(_percentile(values, 95), 1),
                "max": round(values[-1], 1) if values else 0.0,
            }
        return out


window = LatencyWindow(config.LATENCY_WINDOW)


def record(request: Request) -> None:
    """Add one completed request to the latency window.

    Called wherever a `Request` is built, so every path - answered, refused,
    conversational - is measured the same way. Measuring only the happy path
    would report a P95 built entirely from the requests that already worked.
    """
    window.add(
        retrieval_ms=request.retrieval_ms,
        generation_ms=request.generation_ms,
        total_ms=request.total_ms,
    )


def latency_report() -> dict:
    """P50/P95 per stage, for /api/status."""
    return window.percentiles()
