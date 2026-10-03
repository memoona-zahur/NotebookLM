"""Draft a gold set from the evaluation corpus, for a human to verify.

    .venv/bin/python -m experiments.draft_gold --write
    .venv/bin/python -m experiments.draft_gold --out gold/gold_set.json

Produces questions paired with the evidence needed to answer them, so later
retrieval and chunking changes can be measured instead of argued about.

The candidate pool is every block of the source document, not the blocks a
retriever happened to return. That distinction is the whole point: when the
judge only sees its own top-k, anything the retriever missed can never be
labelled relevant, and the resulting Hit Rate measures the judge rather than
the system. Here a fact the retriever fails to find is still proposed, and it
shows up as a retrieval miss instead of disappearing.

Every fact is a verbatim substring of a parsed block, so `fact_coverage`
matches it literally and a paraphrase cannot quietly satisfy a label.

Nothing here is trusted. Output lands with verified_by_human: false and is
worth nothing until someone reads each fact against the document. The
difficulty field is a guess for the same reason.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import parsers  # noqa: E402
from app.lexical import content_terms  # noqa: E402
from experiments import corpus  # noqa: E402
from experiments.metrics import normalize  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "corpus"
DEFAULT_OUT = Path(__file__).resolve().parent / "gold" / "gold_set.json"

# How many questions to take per document, and how many traps overall. Kept
# small on purpose: every item costs a human verification, and an unverified
# gold set is worse than a short verified one.
QUESTIONS_PER_DOC = 2
TRAP_DOCS = 8

MAX_FACT_CHARS = 170
MIN_FACT_CHARS = 8

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE = re.compile(r"(?<=[,;:])\s+")
_RECORD_LINE = re.compile(r"^\s*[\w$.\[\]\"'-]+\s*[:=]")


def _shorten(piece: str) -> list[str]:
    """Split an over-long piece at clause boundaries.

    Dropping anything past MAX_FACT_CHARS would silently discard the longest
    and often most informative sentences - exactly the ones a long PDF line
    produces. Clause boundaries keep every part verbatim, so the substring
    still matches the chunk literally.
    """
    if len(piece) <= MAX_FACT_CHARS:
        return [piece]
    parts: list[str] = []
    for part in _CLAUSE.split(piece):
        part = part.strip()
        if not part:
            continue
        parts.extend(_shorten(part) if len(part) > MAX_FACT_CHARS else [part])
    return parts


def _spans(text: str) -> list[str]:
    """Candidate evidence spans: sentences or clause-sized pieces.

    Structured formats put a record across several lines - 'zone: coldroom'
    then 'reading: -19.1' - and neither line alone answers the question.
    Consecutive lines are also offered as one span so the pair can be the
    fact, capped at the same length so it stays checkable by eye.
    """
    parts: list[str] = []
    for part in _SENTENCE.split(text):
        part = part.strip()
        if part:
            parts.extend(_shorten(part))

    lines = [line for line in text.splitlines() if line.strip()]
    if sum(1 for line in lines if _RECORD_LINE.match(line)) >= 3:
        window: list[str] = []
        for line in lines:
            window.append(line)
            joined = "\n".join(window)
            if len(joined) > MAX_FACT_CHARS:
                window = window[1:]
                joined = "\n".join(window)
            if len(window) >= 2:
                parts.append(joined)
        parts.append("\n".join(lines))

    return [
        part
        for part in parts
        if MIN_FACT_CHARS <= len(part) <= MAX_FACT_CHARS * 2
    ]


def _overlap(terms: set[str], text: str) -> int:
    return len(terms & set(content_terms(text)))


def _scoring_text(block: parsers.Block) -> str:
    """What a question is matched against: heading included.

    For code, config and Markdown the searchable key is the heading - the
    symbol name, the config path, the section title - while the body is an
    implementation detail. Scoring the body alone reports a perfectly
    answerable question as unanswerable.
    """
    return f"{block.heading} {block.text}" if block.heading else block.text


def _rank_blocks(blocks: list[parsers.Block], question: str) -> list[int]:
    terms = set(content_terms(question))
    scored = [(i, _overlap(terms, _scoring_text(b))) for i, b in enumerate(blocks)]
    hits = [(i, score) for i, score in scored if score]
    # Strongest blocks first; ties keep document order so the proposal is
    # stable across runs.
    hits.sort(key=lambda pair: (-pair[1], pair[0]))
    return [i for i, _ in hits]


def _looks_like_evidence(span: str, heading: str) -> bool:
    """Reject furniture - titles, heading echoes, stubs - and keep the rest.

    A heading shares every topic word with its question, so lexical overlap
    ranks it first, yet a title is not an answerable fact. The test is shape
    rather than punctuation: a short span with no sentence ending and no
    number is a label, not an answer. Prose extracted from a PDF often has
    neither, so requiring a full stop would throw away real evidence.
    """
    if len(span) < MIN_FACT_CHARS:
        return False
    if heading:
        for part in re.split(r"\s*>\s*", heading):
            if part.strip() and normalize(span) == normalize(part):
                return False
    # A key with a real value is evidence however short it is, and it is the
    # most deterministic kind there is: 'pool_size: 25' needs no reading.
    if re.search(r"\d", span) and re.search(r"[:=]", span):
        return True
    # Code and query text terminates on ';' and states relations with '=', so
    # 'ORDER BY avg_seconds DESC;' is an answer even without a full stop.
    if span.rstrip().endswith(";") or "=" in span:
        return True
    words = span.split()
    if len(words) <= 3:
        return False
    if len(words) <= 8 and not re.search(r"[.!?]$", span) and not re.search(r"\d", span):
        return False
    return True


def _propose(
    question: str, blocks: list[parsers.Block], wanted: int
) -> tuple[list[int], list[str], str]:
    """Positions and verbatim facts that together answer the question.

    Positions come only from blocks that actually contributed an accepted
    fact. Padding the gold set with a block that holds no labelled evidence
    would make recall unreachable and precision look worse for no reason.

    The third element says why nothing was proposed, because "no lexical
    overlap" and "overlap found but no span read like evidence" are different
    problems and only a human can settle either.
    """
    terms = set(content_terms(question))
    positions = _rank_blocks(blocks, question)[:2]
    if not positions:
        return [], [], "no block shares a term with the question"

    candidates: list[tuple[int, int, str]] = []
    for position in positions:
        heading = blocks[position].heading
        for span in _spans(blocks[position].text):
            if not _looks_like_evidence(span, heading):
                continue
            candidates.append((_overlap(terms, span), position, span))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    facts: list[str] = []
    contributors: list[int] = []
    seen: set[str] = set()
    for score, position, span in candidates:
        key = normalize(span)
        if not score or key in seen:
            continue
        seen.add(key)
        facts.append(span)
        if position not in contributors:
            contributors.append(position)
        if len(facts) == wanted:
            break
    if not facts:
        return [], [], "matching blocks found, but no span read like a fact"
    return sorted(contributors), facts, ""


def _difficulty(question: str, facts: list[str]) -> str:
    """A rough label: how much of the answer sits in one place."""
    if not facts:
        return "unknown"
    if len(facts) == 1 and len(facts[0]) < 90:
        return "easy"
    return "medium" if len(facts) < 3 else "hard"


def build() -> dict:
    items: list[dict] = []
    counter = 0
    trap_budget = TRAP_DOCS

    for name, questions in corpus.QUESTIONS.items():
        path = CORPUS / name
        if not path.exists():
            continue
        blocks = parsers.parse(path)

        for question in questions["relevant"][:QUESTIONS_PER_DOC]:
            positions, facts, why = _propose(question, blocks, wanted=2)
            if not facts:
                # Worth surfacing rather than dropping: either the question is
                # unanswerable from this document, or the proposer missed it.
                items.append(
                    {
                        "id": f"G{counter:03d}",
                        "question": question,
                        "source": name,
                        "answerable": False,
                        "difficulty": "unknown",
                        "expected_positions": [],
                        "required_facts": [],
                        "verified_by_human": False,
                        "note": (
                            f"unresolved: {why}. A human must decide whether the "
                            "document answers this - do not read this as 'the "
                            "document does not say'"
                        ),
                    }
                )
                counter += 1
                continue
            items.append(
                {
                    "id": f"G{counter:03d}",
                    "question": question,
                    "source": name,
                    "answerable": True,
                    "difficulty": _difficulty(question, facts),
                    "expected_positions": positions,
                    "required_facts": facts,
                    "verified_by_human": False,
                }
            )
            counter += 1

        if trap_budget > 0:
            for question in questions["trap"][:1]:
                items.append(
                    {
                        "id": f"G{counter:03d}",
                        "question": question,
                        "source": name,
                        "answerable": False,
                        "difficulty": "trap",
                        "expected_positions": [],
                        "required_facts": [],
                        "verified_by_human": False,
                        "note": "unanswerable: must retrieve nothing above the floor",
                    }
                )
                counter += 1
                trap_budget -= 1

    return {
        "schema": 1,
        "generated_by": "experiments/draft_gold.py",
        "corpus": "experiments/corpus",
        "warning": (
            "Draft only. required_facts are verbatim spans proposed by lexical "
            "overlap, not by a reader. Each one must be checked against the "
            "source document and verified_by_human flipped to true before any "
            "number here is quoted anywhere."
        ),
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write", action="store_true", help="write the JSON instead of printing it")
    args = parser.parse_args()

    gold = build()
    answerable = sum(1 for item in gold["items"] if item["answerable"])
    traps = len(gold["items"]) - answerable
    thin = [i["id"] for i in gold["items"] if i["answerable"] and not i["required_facts"]]

    if not args.write:
        print(json.dumps(gold, indent=2, ensure_ascii=False))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gold, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  {len(gold['items'])} items: {answerable} answerable, {traps} unanswerable")
    print(f"  verified by a human: 0 - nothing in this file is evidence yet")
    if thin:
        print(f"  needs review first: {', '.join(thin)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())