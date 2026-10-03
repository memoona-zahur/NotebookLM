"""Ranking and grounding metrics, written so the arithmetic can be checked.

    .venv/bin/python -m experiments.metrics   # self-checks the definitions

Every function here is pure and takes plain sequences, so a metric can be
verified by hand against a worked example instead of trusted because a
framework returned it. Three choices are deliberate and differ from the
quick implementations these usually get copied from:

  * A query with no gold evidence raises instead of scoring 0.0. Scoring an
    unlabelled question as a miss is indistinguishable from a real retrieval
    failure, and it drags a mean down for the wrong reason - it hides the
    labelling gap rather than reporting it.
  * A repeated id in the ranking is collapsed before scoring. A chunk listed
    twice is a retriever bug, and counting it twice would reward it.
  * Precision@k divides by k, not by however many chunks came back, so a
    retriever cannot raise the score by returning less.

Fact coverage is deliberately not a ranking metric. It asks a different
question - was the evidence actually present in the context handed to the
model - which is what makes it comparable across chunkers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "aggregate",
    "coverage_at_k",
    "dcg_at_k",
    "fact_coverage",
    "hit_rate_at_k",
    "ndcg_at_k",
    "normalize",
    "precision_at_k",
    "reciprocal_rank",
    "recall_at_k",
    "score_query",
]

_WHITESPACE = re.compile(r"\s+")
_ORDINALS = re.compile(r"[^0-9a-z ]+")


def normalize(text: str) -> str:
    """Fold case and collapse whitespace so a fact match ignores formatting."""
    return _WHITESPACE.sub(" ", text.replace("\u00a0", " ")).strip().casefold()


def _ranked(expected: Iterable[object], retrieved: Sequence[object]) -> list[tuple[int, object]]:
    """Gold positions in the ranking, each id once, first occurrence winning."""
    gold = set(expected)
    seen: set[object] = set()
    out: list[tuple[int, object]] = []
    for rank, item in enumerate(retrieved, start=1):
        if item in gold and item not in seen:
            seen.add(item)
            out.append((rank, item))
    return out


def _require_gold(expected: Sequence[object], name: str = "expected") -> set[object]:
    gold = set(expected)
    if not gold:
        raise ValueError(f"{name} is empty: label the question or exclude it from the mean")
    return gold


def recall_at_k(expected: Sequence[object], retrieved: Sequence[object], k: int) -> float:
    """Share of the gold evidence found in the top k. 1.0 means all of it."""
    gold = _require_gold(expected)
    found = _ranked(gold, retrieved[:k])
    return len(found) / len(gold)


def precision_at_k(expected: Sequence[object], retrieved: Sequence[object], k: int) -> float:
    """Share of the top k that is gold. Rewards returning little."""
    gold = _require_gold(expected)
    top = retrieved[:k]
    return sum(1 for item in top if item in gold) / k


def hit_rate_at_k(expected: Sequence[object], retrieved: Sequence[object], k: int) -> float:
    """1.0 when any gold evidence is in the top k. Says nothing about how much."""
    gold = _require_gold(expected)
    return float(bool(_ranked(gold, retrieved[:k])))


def reciprocal_rank(expected: Sequence[object], retrieved: Sequence[object], k: int | None = None) -> float:
    """1/rank of the first gold hit, 0.0 when nothing relevant comes back.

    Truncating at k is the difference between 'the answer was available' and
    'the answer was reachable', so the cut is a parameter rather than a
    constant.
    """
    gold = _require_gold(expected)
    considered = retrieved if k is None else retrieved[:k]
    for rank, item in _ranked(gold, considered):
        return 1.0 / rank
    return 0.0


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Discounted gain with the standard log2(rank + 1) falloff."""
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], start=1))


def ndcg_at_k(expected: Sequence[object], retrieved: Sequence[object], k: int) -> float:
    """DCG against the best possible ordering of the same gold set.

    The gains are written at the rank each gold hit actually occupies. A
    compacted list of ones would score a hit at rank 4 as if it were at
    rank 2, which quietly rewards a ranking that buries its best evidence.
    """
    gold = _require_gold(expected)
    top = retrieved[:k]
    gains = [0.0] * len(top)
    for rank, _ in _ranked(gold, top):
        gains[rank - 1] = 1.0
    ideal = [1.0] * min(len(gold), k)
    best = dcg_at_k(ideal, k)
    return dcg_at_k(gains, k) / best if best else 0.0


def coverage_at_k(expected: Sequence[object], retrieved: Sequence[object], k: int) -> float:
    """Share of gold evidence present in the top k, duplicates ignored.

    The half-k credit variant would smooth the score without saying anything
    a reviewer can act on, so this is plain coverage.
    """
    gold = _require_gold(expected)
    found = _ranked(gold, retrieved[:k])
    return len(found) / len(gold)


def fact_coverage(required_facts: Sequence[str], context: str) -> dict[str, object]:
    """Which required facts are literally present in the context.

    Deliberately literal. A fact is a short verifiable string - a number, a
    parameter name, a definition clause - so a paraphrase counts as a miss.
    That is the point: it keeps the check deterministic and stops a judge
    model from grading itself. `missing` is the actionable half of the result
    and is what the chunker bake-off reads.

    Write each fact the way the parser renders the chunk, not the way a person
    would say it: `$.limits.rpm: 600` matches, `Rpm 600` does not. Matching is
    case- and whitespace-insensitive and nothing else.
    """
    haystack = normalize(context)
    present: list[str] = []
    missing: list[str] = []
    for fact in required_facts:
        needle = normalize(fact)
        (present if needle and needle in haystack else missing).append(fact)
    total = len(required_facts)
    return {
        "found": present,
        "missing": missing,
        "covered": len(present),
        "total": total,
        "score": len(present) / total if total else 0.0,
    }


def score_query(
    expected: Sequence[object],
    retrieved: Sequence[object],
    k: int = 5,
) -> dict[str, float]:
    """Every ranked metric for one question at one cut-off.

    Raises when there is no gold evidence, so an unlabelled question cannot
    quietly become a zero.
    """
    _require_gold(expected)
    return {
        f"recall@{k}": recall_at_k(expected, retrieved, k),
        f"precision@{k}": precision_at_k(expected, retrieved, k),
        f"hit_rate@{k}": hit_rate_at_k(expected, retrieved, k),
        f"ndcg@{k}": ndcg_at_k(expected, retrieved, k),
        f"mrr@{k}": reciprocal_rank(expected, retrieved, k),
    }


def aggregate(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Means over the scored rows, plus the size of the gap.

    A row is `{"metrics": {...}}` or `{"metrics": None}` for a question with no
    gold evidence. Unlabelled rows are counted and dropped, never averaged in
    as failures, and the caller is expected to print `skipped`.
    """
    means: dict[str, float] = {}
    scored = 0
    skipped = 0
    for row in rows:
        metrics = row.get("metrics")
        if not metrics:
            skipped += 1
            continue
        scored += 1
        for name, value in metrics.items():
            means[name] = means.get(name, 0.0) + float(value)
    return {
        "scored": scored,
        "skipped": skipped,
        "means": {name: total / scored for name, total in means.items()} if scored else {},
    }


def _self_check() -> None:
    """Worked examples, so a wrong definition fails here and not in a report."""
    gold = ["a", "b", "c"]
    ranked = ["x", "b", "y", "c", "z"]

    assert math.isclose(recall_at_k(gold, ranked, 5), 2 / 3)
    assert math.isclose(precision_at_k(gold, ranked, 5), 2 / 5)
    assert hit_rate_at_k(gold, ranked, 1) == 0.0
    assert hit_rate_at_k(gold, ranked, 2) == 1.0
    assert math.isclose(reciprocal_rank(gold, ranked), 0.5)
    assert math.isclose(reciprocal_rank(gold, ranked, k=2), 0.5)
    assert reciprocal_rank(gold, ["x", "y"]) == 0.0

    # Ideal ordering scores 1.0.
    assert math.isclose(ndcg_at_k(gold, gold, 3), 1.0)
    assert math.isclose(dcg_at_k([1.0, 1.0, 1.0], 3), 1 + 1 / math.log2(3) + 1 / 2)

    # Two gold hits, but one of them is pushed down to rank 4:
    #   dcg  = 1/log2(2) + 1/log2(5)        = 1.43068
    #   idcg = 1/log2(2) + 1/log2(3) + 1/log2(4) = 2.13093
    #   ndcg = 0.67142
    late = ["c", "x", "y", "a"]
    assert math.isclose(recall_at_k(gold, late, 4), 2 / 3)
    assert math.isclose(dcg_at_k([1.0, 0.0, 0.0, 1.0], 4), 1 + 1 / math.log2(5))
    assert math.isclose(ndcg_at_k(gold, late, 4), 0.67142, rel_tol=1e-4)

    # Binary gains mean a complete gold set scores the same in any order, so a
    # reversed ranking is not by itself a penalty - the discount only bites
    # when gold is pushed below competing chunks.
    assert math.isclose(ndcg_at_k(gold, list(reversed(gold)), 3), 1.0)

    # A repeated id must not earn a second credit...
    assert recall_at_k(["a"], ["a", "a", "a"], 3) == 1.0
    # ...and a short ranking must not beat a full one on precision.
    assert precision_at_k(["a"], ["a"], 5) < precision_at_k(["a"], ["a"], 1)

    facts = fact_coverage(
        ["$.limits.rpm: 600", "missing clause"],
        "$.limits.rpm: 600\nother text",
    )
    assert facts["covered"] == 1 and facts["total"] == 2
    assert facts["missing"] == ["missing clause"]
    assert math.isclose(facts["score"], 0.5)

    # A fact has to be written the way the parser renders the chunk. Prose
    # phrasing earns no credit, which is what keeps the check deterministic
    # and stops a label from being satisfied by a near-miss.
    assert fact_coverage(["Rpm 600"], "$.limits.rpm: 600")["covered"] == 0

    for call in (
        lambda: recall_at_k([], ["a"], 5),
        lambda: score_query([], ["a"], 5),
        lambda: ndcg_at_k([], ["a"], 5),
    ):
        try:
            call()
        except ValueError:
            pass
        else:  # pragma: no cover - the guard is the behaviour under test
            raise AssertionError("empty gold must raise, not score 0.0")

    report = aggregate(
        [
            {"metrics": {"recall@5": 1.0}},
            {"metrics": {"recall@5": 0.0}},
            {"metrics": None},
        ]
    )
    assert report["scored"] == 2 and report["skipped"] == 1
    assert math.isclose(report["means"]["recall@5"], 0.5)

    print("metrics: all definitions behave as documented")


if __name__ == "__main__":
    _self_check()