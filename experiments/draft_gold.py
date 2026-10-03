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
import math
import re
import sys
from collections.abc import Callable, Sequence
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

# Words that ask for an answer chosen by comparing values rather than quoted
# from one place. Matched against question terms, so 'most' and 'min' count
# and 'monument' does not.
_COMPARISON = re.compile(
    r"^(highest|lowest|largest|smallest|greatest|best|worst|most|least|max|min|maximum|minimum|"
    r"warmest|coldest|fastest|slowest|average|total|count|compare|comparison|rank|ranking|top)$"
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# Comma and semicolon only. Splitting on a colon looks reasonable for prose
# and mangles key/value text: in '$.package.version: 0.9.3\n$.package.name: x'
# it swallows every following line, producing multi-key blobs as facts.
_CLAUSE = re.compile(r"(?<=[,;])\s+")
# A key at the start of a line: 'zone: coldroom', '$.learning_rate: 0.001'.
_RECORD_LINE = re.compile(r"^\s*(?:\$\.?)?[\w\"'-]+(?:\.[\w-]+)*\s*[:=]")


def _is_record_line(line: str) -> bool:
    """True for a 'key: value' line, false for code that merely contains ':'.

    A Java method reference like LineItem::declaredValue matches a naive ':'
    test, and treating those lines as records produced multi-line code blobs
    as gold facts. Requiring a key at the start of the line, with no '::'
    anywhere in it, keeps windows for tables and config and away from code.
    """
    return "::" not in line and _RECORD_LINE.match(line) is not None


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


def _record_key(line: str) -> str:
    """The field name of a 'key: value' line, or '' when it has none."""
    if not _is_record_line(line):
        return ""
    return re.split(r"[:=]", line, maxsplit=1)[0].strip().casefold()


def _record_spans(lines: list[str]) -> list[str]:
    """Split a key/value block into one span per record.

    A repeated field name starts a new record, which is how tabular data
    repeats its schema. Sliding a fixed-width window over the lines instead
    produced spans that straddled two records - 'sensor_id: TH-001' with
    'zone: storefront' but not the reading, so the fact named the zone without
    its temperature.
    """
    spans: list[str] = []
    current: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = _record_key(line)
        if current and key and key in seen:
            spans.append("\n".join(current))
            current, seen = [], set()
        current.append(line)
        if key:
            seen.add(key)
    if current:
        spans.append("\n".join(current))
    return spans


def _is_record_block(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    records = [line for line in lines if _is_record_line(line)]
    # Every line a record, not merely enough of them. A code block that happens
    # to contain three assignments is not a table, and treating it as one
    # produced multi-line source blobs as gold facts.
    return len(lines) >= 4 and len(records) >= 3 and len(records) >= 0.8 * len(lines)


def _spans(text: str) -> list[str]:
    """Candidate evidence spans: sentences, clause pieces, records, key lines.

    Structured formats put a record across several lines - 'zone: coldroom'
    then 'reading: -19.1' - and neither line alone answers the question, so the
    whole record is offered as one span alongside the individual lines.
    """
    parts: list[str] = []
    for part in _SENTENCE.split(text):
        part = part.strip()
        if part:
            parts.extend(_shorten(part))

    lines = [line for line in text.splitlines() if line.strip()]
    if _is_record_block(text):
        parts.extend(_record_spans(lines))

    return [
        part
        for part in parts
        if MIN_FACT_CHARS <= len(part) <= MAX_FACT_CHARS * 2
    ]


def _idf(spans: Sequence[str]) -> Callable[[str], float]:
    """Weight a term by how rare it is across the spans of one document.

    Counting shared terms cannot tell 'training' from 'rows', so a generic key
    beat the specific one on file order alone: 'How many training rows are
    there?' matched '$.training.epochs: 140' exactly as well as
    '$.data.train_rows: 4820000'. Down-weighting the terms a document uses
    everywhere is the insight BM25 rests on, and it hands the match to the
    discriminative term.
    """
    counts: dict[str, int] = {}
    for span in spans:
        for term in set(content_terms(span)):
            counts[term] = counts.get(term, 0) + 1
    total = max(1, len(spans))
    return lambda term: math.log(1.0 + total / (1.0 + counts.get(term, 0)))


def _overlap(terms: set[str], text: str, weight: Callable[[str], float] | None = None) -> float:
    shared = terms & set(content_terms(text))
    if weight is None:
        return float(len(shared))
    return sum(weight(term) for term in shared)


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
    # A comment describes code without being an answer, and its words match the
    # question as well as the code does - 'Cart totals with progressive
    # discount tiers' ranks top for a question about discount tiers.
    if re.match(r"\s*(//|#|--|/\*|\*)", span):
        return False
    if heading:
        for part in re.split(r"\s*>\s*", heading):
            if part.strip() and normalize(span) == normalize(part):
                return False
    # A key with a real value is evidence however short it is, and it is the
    # most deterministic kind there is: 'pool_size: 25' needs no reading.
    if re.search(r"\d", span) and re.search(r"[:=]", span):
        return True
    # Code and query text terminates on ';' or '{' and states relations with
    # '=', so 'ORDER BY avg_seconds DESC;' and a method signature are answers
    # even without a full stop.
    if span.rstrip().endswith((";", "{")) or "=" in span:
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

    # 'Which zone is warmest?' and 'Which region has the highest return rate?'
    # are answered by comparing records, not by quoting one. A lexical match
    # picks a plausible-looking record that is not the answer, and a wrong gold
    # fact is worse than an absent one, because it is indistinguishable from a
    # retriever failure once scoring starts.
    if any(_COMPARISON.fullmatch(term) for term in terms) and any(
        _is_record_block(blocks[p].text) for p in positions
    ):
        return [], [], "the question compares records, so a human must supply the fact"

    candidates: list[tuple[float, int, str]] = []
    # Rarity is measured over the whole document, not just the two blocks that
    # matched: a term can be rare in a neighbour and common everywhere else.
    all_spans = {index: _spans(block.text) for index, block in enumerate(blocks)}
    weight = _idf([span for spans in all_spans.values() for span in spans])
    for position in positions:
        heading = blocks[position].heading
        for span in all_spans[position]:
            if not _looks_like_evidence(span, heading):
                continue
            candidates.append((_overlap(terms, span, weight), position, span))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    # Keep only facts that match the question as strongly as the best one, and at
    # most one per block. Demanding a second fact unconditionally pads the gold
    # set with spans that are on-topic but do not answer anything, and every one
    # of those becomes an unmeetable recall requirement later. Two equally good
    # spans inside one block are usually the same fact twice - a config file
    # where every key repeats the section name ties on every key.
    if not candidates:
        return [], [], "matching blocks found, but no span read like a fact"
    best = candidates[0][0]

    facts: list[str] = []
    contributors: list[int] = []
    seen: set[str] = set()
    for score, position, span in candidates:
        key = normalize(span)
        if not score or score < best or key in seen or position in contributors:
            continue
        seen.add(key)
        facts.append(span)
        if position not in contributors:
            contributors.append(position)
        if len(facts) == wanted:
            break
    if not facts:
        return [], [], "matching blocks found, but no span read like a fact"

    # Every block holding the fact is gold, not just the one it was drafted
    # from. Overlapping chunks repeat their boundary text on purpose, and a
    # label that names only the first copy is unreachable: the retriever
    # returns the second copy, it is a correct answer, and recall calls it a
    # miss. Customs.java caught exactly that - 'requiresLicence' sits in two
    # chunks and the gold named one of them.
    holders = [
        index
        for index, block in enumerate(blocks)
        if any(normalize(fact) in normalize(block.text) for fact in facts)
    ]
    return sorted(holders) or sorted(contributors), facts, ""


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
            # One fact, not two. A second span of equal standing is usually a
            # near-duplicate rather than a second requirement, and every extra
            # fact becomes another way for recall@k to fail for a reason that
            # has nothing to do with retrieval. Where a question genuinely
            # needs two pieces of evidence, the human review adds them.
            positions, facts, why = _propose(question, blocks, wanted=1)
            if not facts:
                # Worth surfacing rather than dropping: either the question is
                # unanswerable from this document, or the proposer missed it.
                items.append(
                    {
                        "id": f"G{counter:03d}",
                        "question": question,
                        "source": name,
                        # Unknown, not False. Recording 'the document does not
                        # answer this' would be a claim nobody has checked, and
                        # it would turn a correct retrieval into a reported
                        # leak once scoring starts.
                        "answerable": None,
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


def _context_around(block_text: str, fact: str, width: int = 130) -> str:
    """The fact with its neighbouring text, so it can be judged in place."""
    at = normalize(fact)
    haystack = block_text
    # normalize() collapses whitespace, so find the offset on the raw text by
    # searching progressively longer prefixes rather than re-implementing it.
    for start in range(0, max(1, len(haystack) - 1)):
        if normalize(haystack[start : start + len(at)]) == at:
            haystack = haystack[start:]
            break
    lines = [line for line in haystack.splitlines() if line.strip()]
    kept: list[str] = []
    budget = width
    for line in lines:
        if budget <= 0:
            break
        kept.append(line[:budget])
        budget -= len(line)
    return "\n".join(kept)


def _support_count(question: str, fact: str) -> int:
    """How many of the question's content terms the fact actually contains.

    One is the weak case: the fact matched on a single shared word, which is
    how 'A surge is a short episode of dramatic acceleration' came to be
    offered as the answer to 'What causes a glacier surge?'. The count is a
    reading aid for the reviewer and nothing else.
    """
    terms = set(content_terms(question))
    shared = terms & set(content_terms(fact))
    return len(shared)


def review_sheet(gold: dict, path: Path) -> None:
    """Write the human review sheet: one judgement per item, context included."""
    blocks: dict[str, list] = {}
    out: list[str] = [
        "# Gold set review sheet",
        "",
        "For each item, one question: **would this text be an acceptable answer",
        "to that question?** If yes, approve it. If no, reject it.",
        "",
        "You do not need to open the source files. Each fact is quoted verbatim",
        "with the lines around it, which is all the context needed to judge.",
        "",
        "Three ways to answer a question count as correct here: the fact states",
        "the answer outright, or it contains the specific value asked for, or",
        "it is the sentence that defines the thing asked about. A fact that is",
        "merely on the same topic is not an answer.",
        "",
        "## Needs a fact written by hand",
        "",
    ]
    for item in gold["items"]:
        if item["answerable"] and not item["required_facts"]:
            out.append(f"- **{item['id']}** ({item['source']}) {item['question']}")
            out.append(f"  - {item.get('note', '')}")
    out += [
        "",
        "Read the document, then record the fact as the parser renders it:",
        "",
        '```',
        'python -m experiments.draft_gold --fact G027 "region: EMEA\\nreturn_rate_pct: 2.4"',
        "```",
        "",
        "The text is checked against the document, so a paraphrase is refused",
        "rather than stored as a fact that can never be matched.",
        "",
        "## Traps",
        "",
        "These ask about something the document does not cover. Check that the",
        "question really is unrelated to its document, then approve it. A trap",
        "that its own document answers is a broken trap.",
        "",
    ]
    for item in gold["items"]:
        if not item["answerable"]:
            out.append(f"- **{item['id']}** ({item['source']}) {item['question']}")

    out += ["", "## Answerable items", ""]
    risky: list[dict] = []
    for item in gold["items"]:
        if not item["required_facts"]:
            continue
        if item["source"] not in blocks:
            blocks[item["source"]] = parsers.parse(CORPUS / item["source"])
        parsed = blocks[item["source"]]
        block = parsed[item["expected_positions"][0]]
        support = _support_count(item["question"], item["required_facts"][0])
        (risky if support <= 1 else []).append(item)
        out.append(f"### {item['id']}  ({item['source']}, {item['difficulty']})")
        out.append("")
        out.append(f"**Q: {item['question']}**")
        out.append("")
        out.append("```")
        out.append(_context_around(block.text, item["required_facts"][0]))
        out.append("```")
        if support <= 1:
            out.append("")
            out.append(
                f"> Only {support} term of the question appears in this fact. Check "
                "it really answers the question and is not just on the same topic."
            )
        out.append("")

    out += [
        "## Recording your review",
        "",
        "Approve the ones that hold up:",
        "",
        "```",
        "python -m experiments.draft_gold --verify G000 G001 G004",
        "```",
        "",
        "Reject the ones that do not:",
        "",
        "```",
        "python -m experiments.draft_gold --reject G003",
        "```",
        "",
        "Rejecting removes the item. That is deliberate: a wrong fact left in",
        "place becomes a permanent miss that looks like a retriever failure, and",
        "it can be re-drafted once the question is understood.",
        "",
        f"{len(risky)} item(s) rest on a single shared term and are the ones most",
        "worth a second look.",
    ]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {path}  ({len(gold['items'])} items, {len(risky)} flagged for a second look)")


def _load_existing(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist yet; run --write first")
    return json.loads(path.read_text(encoding="utf-8"))


def _positions_holding(source: str, fact: str) -> list[int]:
    """Every block of the document that contains the fact verbatim.

    A hand-written fact is checked the same way a drafted one is, because a
    paraphrase can never be matched by fact_coverage. Better to refuse it here
    than to discover at scoring time that recall is unreachable.
    """
    needle = normalize(fact)
    blocks = parsers.parse(CORPUS / source)
    holders = [index for index, block in enumerate(blocks) if needle in normalize(block.text)]
    if not holders:
        raise ValueError(
            f"that text is not in {source} as the parser renders it. Copy the "
            "exact line, including punctuation and spacing."
        )
    return holders


def review(
    path: Path,
    verify: list[str],
    reject: list[str],
    facts: dict[str, str],
) -> dict:
    """Record a human review of the draft, in place, one item at a time.

    Approving and rejecting are separate commands on purpose. Approving says
    'this fact answers this question'; rejecting says the pairing is wrong
    enough that the item should go. Leaving a known-bad label in place to be
    'fixed later' is how a wrong answer becomes a permanent 0.0 nobody can
    explain, so there is no way to record a doubt - the item is rejected and
    can be re-drafted.
    """
    gold = _load_existing(path)
    items = gold["items"]
    by_id = {item["id"]: item for item in items}
    unknown = [i for i in (*verify, *reject, *facts) if i not in by_id]
    if unknown:
        raise ValueError(f"no such item: {unknown}")

    for item_id, fact in facts.items():
        item = by_id[item_id]
        item["required_facts"] = [fact]
        item["expected_positions"] = _positions_holding(item["source"], fact)
        item["difficulty"] = _difficulty(item["question"], item["required_facts"])
        if item["difficulty"] == "unknown":
            item["difficulty"] = "medium"
        item.pop("note", None)
        item["answerable"] = True
        print(f"  {item_id} fact accepted, cites positions {item['expected_positions']}")

    for item_id in verify:
        item = by_id[item_id]
        if not item["required_facts"]:
            raise ValueError(
                f"{item_id} has no fact to check. Supply one with "
                f"--fact {item_id} \"...\", or reject it with --reject {item_id}"
            )
        item["verified_by_human"] = True
        print(f"  {item_id} verified")

    dropped = set(reject)
    if dropped:
        gold["items"] = [item for item in items if item["id"] not in dropped]
        print(f"  rejected {len(dropped)}: {sorted(dropped)}")

    path.write_text(json.dumps(gold, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    remaining = sum(1 for i in gold["items"] if not i["verified_by_human"])
    print(f"\n{len(gold['items'])} items, {len(gold['items']) - remaining} verified, {remaining} left")
    return gold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write", action="store_true", help="write the JSON instead of printing it")
    parser.add_argument(
        "--force", action="store_true", help="regenerate even if verified items would be lost"
    )
    parser.add_argument("--verify", nargs="+", default=[], metavar="ID", help="mark items verified")
    parser.add_argument("--reject", nargs="+", default=[], metavar="ID", help="remove items entirely")
    parser.add_argument(
        "--review",
        type=Path,
        metavar="PATH",
        help="write a human review sheet with the context each fact needs",
    )
    parser.add_argument(
        "--fact",
        nargs="+",
        default=[],
        metavar=("ID", "TEXT"),
        help="attach a hand-written fact to an item; checked against the document",
    )
    args = parser.parse_args()

    if args.review:
        review_sheet(_load_existing(args.out), args.review)
        return 0

    if args.verify or args.reject or args.fact:
        pairs = args.fact
        if len(pairs) % 2:
            raise SystemExit("--fact takes ID and TEXT pairs")
        review(
            args.out,
            args.verify,
            args.reject,
            dict(zip(pairs[::2], pairs[1::2])),
        )
        return 0

    # Regenerating rebuilds every item from the corpus, so any flag a person
    # set by hand would be silently dropped. Refuse instead.
    if args.write and args.out.exists():
        kept = [i for i in _load_existing(args.out)["items"] if i.get("verified_by_human")]
        if kept and not args.force:
            raise SystemExit(
                f"{args.out} has {len(kept)} human-verified items. Regenerating "
                "discards that review; pass --force if you really mean to."
            )

    gold = build()
    answerable = sum(1 for item in gold["items"] if item["answerable"] is True)
    traps = sum(1 for item in gold["items"] if item["answerable"] is False)
    unknown = [i["id"] for i in gold["items"] if i["answerable"] is None]

    if not args.write:
        print(json.dumps(gold, indent=2, ensure_ascii=False))
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gold, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(
        f"  {len(gold['items'])} items: {answerable} answerable, {traps} traps, "
        f"{len(unknown)} answerability unknown"
    )
    print(f"  verified by a human: 0 - nothing in this file is evidence yet")
    if unknown:
        print(f"  needs a fact written by hand: {', '.join(unknown)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())