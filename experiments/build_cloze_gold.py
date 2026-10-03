"""A gold set that a machine can check, so no expert review is required.

    .venv/bin/python -m experiments.build_cloze_gold --write

Every item is a cloze: one line from a document with a value blanked out, and
a question that is that same line with a gap in it. The answer is the text that
filled the gap.

    line:  Vancomycin must be infused slowly over at least one hour to avoid ...
    Q:     Vancomycin must be infused slowly over at least _____ to avoid ...
    A:     one hour

The point is that correctness is decidable without reading the subject. Four
checks run on every item, and eval_gold re-runs them at scoring time, so an
item that no longer matches its document cannot be scored:

  1 the fact is a verbatim line of the parser's own output
  2 the answer appears in that line
  3 the answer does not appear in the question, so nothing is given away
  4 the question with the blank refilled reproduces the line exactly

A human reviewer would ask a fifth question - 'is this the answer a person
would give?' - and that is the honest limit of this file. These labels prove
the pipeline keeps a value attached to its key and can find the line again.
They do not prove the system understands questions.

What this is good for, specifically:

  * a value split from its key by a chunk boundary is caught immediately. That
    is not hypothetical: a sliding window over temperatures.tsv produced a fact
    naming a zone without its reading.
  * PDFs, Markdown and logs are covered at all, which key-value-only labelling
    would skip entirely.
  * it is reproducible. Re-running this regenerates the same items from the
    same documents, so a change in the numbers means retrieval changed.

Where the blanked value sits is chosen to be unambiguous: the answer must
occur exactly once in the line, and the line must be long enough that the
question still reads as a question.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import parsers  # noqa: E402
from experiments import corpus  # noqa: E402
from app.lexical import content_terms  # noqa: E402
from experiments.metrics import normalize  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
DEFAULT_OUT = HERE / "gold" / "cloze_set.json"

PER_DOCUMENT = 3
MIN_LINE_CHARS = 45
MAX_LINE_CHARS = 170

SPELLED = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    "eighty|ninety|hundred|thousand|million|billion"
)
# A number, or a number word, optionally hyphenated or spaced into a compound
# ('thirty-six', 'twenty three'). The trailing group has to wrap the whole
# alternation, not just its last branch: 'billion(?:[- ]one|...)' read as
# 'billion' followed by a group choosing one of the words, so 'thirty six'
# matched as two separate numbers and never picked up its unit.
NUMBER = rf"(?:\d+(?:[.,]\d+)?|{SPELLED}(?:[- ]{SPELLED})?)"
UNITS = (
    "degrees?|celsius|fahrenheit|minutes?|hours?|seconds?|days?|weeks?|months?|years?|"
    "rows?|kg|kg/ha|ha|metres?|meters?|m|km|cm|mm|%|percent|reviews?|items?|units?|"
    "replicates?|samples?|pages?|words?|geometries?|features?|epochs?|epochs|tries?"
)
# The unit alternation needs its own group: '\s+degrees?|celsius|months?' binds
# the \s+ to 'degrees?' alone, so 'months' then had to match without a leading
# space and the unit was silently dropped from every prose value.
BLANKABLE = re.compile(
    rf"\b(?P<value>{NUMBER}(?:\s+{NUMBER})*(?:\s+(?:{UNITS}))?)\b", re.I
)

# Furniture: headings, separators, imports, comment lines and bare numbers.
SKIP = re.compile(r"^[\s#=|*\-_`/\.\(\)\[\]{}]*$|^\s*(import|from|package|using)\s")

# A comment describes code without asserting anything, and '// Cart totals with
# progressive discount tiers' makes a poor answer to 'what is in this file'.
COMMENT = re.compile(r"^\s*(//|#|--|/\*|\*)")

# A log line's clock time is not a fact. Blanking part of it produced items
# whose answer was '38', which answers nothing.
TIMESTAMPED = re.compile(
    r"^(?:\d{4}-\d{2}-\d{2}[T ])?\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?Z?\b\s*"
)
DIGITS = re.compile(r"\d")

# 'key: value' and 'key = value' at the end of a line. The key is taken from
# the last match so that a log line yields 'model', not a fragment of its
# timestamp. These lines are short by nature - '$.training.learning_rate: 0.001'
# is 31 characters - and they are the most valuable items in the set, so they
# are not held to the prose length floor.
KEY_VALUE = re.compile(r"(?P<key>[$A-Za-z_][\w$.\[\]]*)\s*[:=]\s(?P<value>\S.*?)\s*$")
MAX_VALUE_CHARS = 60
# A key term appearing in at most this many documents is specific enough to
# identify its own value.
RARE_TERMS = 3


def _candidate_lines(blocks: list[parsers.Block]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for index, block in enumerate(blocks):
        for line in block.text.splitlines():
            line = line.strip()
            if SKIP.match(line) or COMMENT.match(line):
                continue
            if len(line) > MAX_LINE_CHARS:
                continue
            # A log line's clock time is not a fact, but the message after it is.
            # Dropping the prefix keeps 'model: rmse 412.7 kg/ha' available
            # instead of discarding every line of every log.
            line = TIMESTAMPED.sub("", line).strip()
            if len(line) < MIN_LINE_CHARS and not _field_value(line):
                continue
            out.append((index, line))
    return out


def _field_value(line: str) -> str | None:
    """The value of a 'key: value' line, if it is a clean single value."""
    matches = list(KEY_VALUE.finditer(line))
    if not matches:
        return None
    value = matches[-1].group("value").strip()
    if not value or len(value) > MAX_VALUE_CHARS:
        return None
    # A compound value is a record, not an answer. 'rmse 412.7 kg/ha, r2 0.71'
    # asks two questions at once and neither has one right answer.
    if len(value.split()) > 6 or ", " in value:
        return None
    return value


def _blanks(line: str) -> list[tuple[int, int, str]]:
    """Every span of the line worth blanking, with its offsets."""
    found: list[tuple[int, int, str]] = []
    for match in BLANKABLE.finditer(line):
        value = match.group("value").strip()
        before = line[match.start() - 2 : match.start()]
        after = line[match.end() : match.end() + 2]
        # 'sixty-six' must not split into 'sixty-_____' answering 'six', and
        # '10.4.2.9' must not become '10.4._____' answering '2.9'. A dot only
        # joins the number when digits sit either side of it: 'months.' ends a
        # sentence, and rejecting that would throw away every prose value.
        if before and (before[-1] in "-_/" or before[-1].isdigit()):
            continue
        if len(before) == 2 and before[-1] == "." and before[0].isdigit():
            continue
        if after and (after[0] in "-_" or after[0].isdigit()):
            continue
        if len(after) == 2 and after[0] == "." and after[1].isdigit():
            continue  # '10.4' of '10.4.2.9': the tail is digits, so it is an address
        # The answer has to be a value, not a stray word. A bare number word
        # with no unit reads as a guess; 'one' is not an answer to anything.
        has_unit = bool(re.search(rf"\s{UNITS}$", value, re.I))
        digits = len(re.findall(r"\d", value))
        if not has_unit and digits < 2:
            continue
        # Never blank the opening words: 'A glacier advances' must not become
        # '_____ glacier advances'.
        if match.start() == 0 or match.start() < 12:
            continue
        found.append((match.start("value"), match.end("value"), value))
    return found


def _cloze_from_line(line: str) -> tuple[str, str, str] | None:
    """Blank one value out of a line. Returns (kind, question, answer).

    Only values are blanked. An earlier version also blanked a single
    distinctive word, which produced 'held at a single temperature for _____ an
    hour' answered 'roughly' - a filler adverb that answers nothing. Every
    item here asks for a value the document states.
    """
    # Strip the clock here rather than trusting every caller to have done it.
    line = TIMESTAMPED.sub("", line).strip()

    # Numbers first. A 'key: value' match fires on prose that merely contains a
    # colon, and taking it first dropped whole documents from the set.
    best: tuple[int, int, str] | None = None
    for start, end, candidate in _blanks(line):
        if line.count(candidate) != 1:
            continue
        if best is None or len(candidate) > len(best[2]):
            best = (start, end, candidate)
    if best is not None:
        start, end, candidate = best
        return "value", line[:start] + "_____" + line[end:], candidate

    value = _field_value(line)
    if value is not None:
        at = line.rindex(value)
        return "field", line[:at] + "_____" + line[at + len(value) :], value
    return None


RECORD_LINE = re.compile(r"^\s*(?:\$\.?)?[\w\"'-]+(?:\.[\w-]+)*\s*[:=]\s*\S")
# When is a field's value worth asking for? A timestamp is not: it says when
# the row was written, not what it records.
META_FIELD = re.compile(r"(?:^|[._\s-])(?:timestamp|datetime|date|time|ts|created|updated)(?:$|[._\s-])", re.I)


def _records(blocks: list[parsers.Block]) -> list[tuple[int, list[str]]]:
    """Group a key/value block into records, split where a field name repeats.

    Tabular data repeats its schema, so a repeated field name starts a new
    record. Without this, 'zone: _____' would be the question for every zone
    in the file - storefront, warehouse, coldroom - and the label would stop
    being decisive, because any of them could be called correct. Naming the
    record fixes that: the question carries the other fields, so only one line
    in the document answers it.
    """
    out: list[tuple[int, list[str]]] = []
    for index, block in enumerate(blocks):
        lines = [line.strip() for line in block.text.splitlines() if line.strip()]
        if len(lines) < 4 or sum(1 for line in lines if RECORD_LINE.match(line)) < 0.8 * len(lines):
            continue
        current: list[str] = []
        seen: set[str] = set()
        for line in lines:
            key = line.split(":", 1)[0].strip().casefold()
            if current and key in seen:
                out.append((index, current))
                current, seen = [], set()
            current.append(line)
            seen.add(key)
        if current:
            out.append((index, current))
    return out


def _record_items(
    name: str, blocks: list[parsers.Block], per_document: int
) -> list[dict]:
    groups = _records(blocks)
    # A field that reads the same in every record carries no information, so
    # blanking it would produce 'what was the unit?' -> 'C'. Blanking a field
    # that varies asks the question the table is actually for.
    constant: set[str] = set()
    fields: dict[str, set[str]] = {}
    for _, group in groups:
        for line in group:
            matches = list(KEY_VALUE.finditer(line))
            if matches:
                fields.setdefault(matches[-1].group("key").casefold(), set()).add(
                    matches[-1].group("value").casefold()
                )
    for key, values in fields.items():
        if len(values) == 1:
            constant.add(key)
    constant |= {key for key in fields if META_FIELD.search(key)}

    built: list[dict] = []
    for position, group in groups:
        if len(built) >= per_document:
            break
        if len(group) < 3:
            continue
        body = "\n".join(group)
        if len(body) > MAX_LINE_CHARS * 3:
            continue
        # Blank the last varying numeric field: it is the measurement, and the
        # rest of the record names which measurement is wanted.
        target = next(
            (
                line
                for line in reversed(group)
                if _field_value(line)
                and _key_of(line) not in constant
                and any(ch.isdigit() for ch in _field_value(line))
            ),
            None,
        )
        if target is None:
            target = next(
                (line for line in reversed(group) if _field_value(line) and _key_of(line) not in constant),
                None,
            )
        if target is None:
            continue
        value = _field_value(target)
        if value is None or body.count(target) != 1:
            continue
        at = target.rindex(value)
        blanked = target[:at] + "_____" + target[at + len(value) :]
        built.append(
            {
                "source": name,
                "kind": "record_cloze",
                "question": body.replace(target, blanked, 1),
                "answer": value,
                "required_facts": [body],
                "expected_positions": [
                    i for i, b in enumerate(blocks) if normalize(body) in normalize(b.text)
                ],
            }
        )
    return built


def _key_of(line: str) -> str | None:
    matches = list(KEY_VALUE.finditer(line))
    return matches[-1].group("key").casefold() if matches else None


def _distinctive(key: str, df: dict[str, int], ceiling: int) -> bool:
    """Does this key name something specific, or is it a word like 'version'?

    '$.version: _____' asks 'which version?' and nothing in the question says
    which. No retriever can answer it, so scoring it measures noise. A key is
    usable when at least one of its terms is rare across the corpus.
    """
    terms = [term for term in content_terms(key) if term]
    if not terms:
        return False
    return any(df.get(term, 0) <= ceiling for term in terms)


def _items_for_document(
    name: str, per_document: int, df: dict[str, int] | None = None
) -> list[dict]:
    path = CORPUS / name
    if not path.exists():
        return []
    blocks = parsers.parse(path)
    df = df or {}
    # Tabular documents first: a record names itself and gives a decisive
    # question, where a bare field name repeated down a file does not.
    built = _record_items(name, blocks, per_document)
    if len(built) >= per_document:
        return built

    headings = {normalize(block.heading) for block in blocks if block.heading}
    keys_in_doc: dict[str, int] = {}
    for block in blocks:
        for line in block.text.splitlines():
            line = line.strip()
            for match in KEY_VALUE.finditer(line):
                key = match.group("key").casefold()
                keys_in_doc[key] = keys_in_doc.get(key, 0) + 1

    seen_questions: set[str] = set()
    for position, line in _candidate_lines(blocks):
        if len(built) >= per_document:
            break
        clozed = _cloze_from_line(line)
        if clozed is None:
            continue
        kind, question, answer = clozed
        if normalize(line) in headings:
            continue  # a heading is furniture, whatever follows the colon
        if kind == "field":
            matches = list(KEY_VALUE.finditer(line))
            if matches and keys_in_doc.get(matches[-1].group("key").casefold(), 0) > 1:
                # The same field name occurs in several places in this
                # document, so the field on its own does not say which value is
                # wanted. Record items cover the tabular case, where the other
                # fields of the record name it.
                continue
            if df and not _distinctive(matches[-1].group("key") if matches else "", df, RARE_TERMS):
                continue
        key = normalize(question)
        if key in seen_questions:
            continue
        seen_questions.add(key)
        holders = [i for i, b in enumerate(blocks) if normalize(line) in normalize(b.text)]
        built.append(
            {
                "source": name,
                "kind": f"{kind}_cloze",
                "question": question,
                "answer": answer,
                "required_facts": [line],
                "expected_positions": holders,
            }
        )
    return built


def _traps(all_text: str, count: int) -> list[dict]:
    """Unanswerable questions, verified by absence rather than by assertion.

    A trap is only valid if none of its content words appear anywhere in the
    corpus. Anything else could be answered by a real document, and a trap that
    leaks is reported as a retrieval success - the one failure mode a
    hand-written trap list cannot rule out.
    """
    lexicon = {term for term in content_terms(all_text)}
    candidates = [
        ("How do I replace a bicycle chain?", "replace a bicycle chain"),
        ("How do I change a car tyre?", "change a car tyre"),
        ("How do I knit a scarf?", "knit a scarf"),
        ("How do I make pasta?", "make pasta"),
        ("How do I bake bread?", "bake bread"),
        ("What is the capital of Peru?", "capital of Peru"),
        ("How do I service a bicycle wheel?", "service a bicycle wheel"),
        ("How do I repot a fern?", "repot a fern"),
        ("Who won the 1998 world cup?", "1998 world cup"),
        ("How do I tune a piano?", "tune a piano"),
    ]
    out: list[dict] = []
    for question, phrase in candidates:
        if len(out) >= count:
            break
        overlap = sorted(set(content_terms(phrase)) & lexicon)
        if overlap:
            continue
        out.append(
            {
                "source": None,
                "question": question,
                "answer": None,
                "required_facts": [],
                "expected_positions": [],
                "absent_terms": phrase,
            }
        )
    return out


def check_item(item: dict, blocks: dict[str, list]) -> list[str]:
    """The mechanical checks. An empty list means the item is sound."""
    problems: list[str] = []
    facts = item["required_facts"]
    if not facts:
        # A trap: every content term of the phrase must be absent corpus-wide.
        text = " ".join(b.text for parsed in blocks.values() for b in parsed)
        leaked = sorted(set(content_terms(item["absent_terms"])) & set(content_terms(text)))
        if leaked:
            problems.append(f"trap terms appear in the corpus: {leaked}")
        return problems

    source = item["source"]
    if source not in blocks:
        return [f"{source} was not parsed"]
    parsed = blocks[source]
    positions = item["expected_positions"]
    if not positions:
        problems.append("no cited position")
    for position in positions:
        if position >= len(parsed):
            problems.append(f"position {position} does not exist")
        elif normalize(facts[0]) not in normalize(parsed[position].text):
            problems.append(f"position {position} does not contain the fact")

    fact = facts[0]
    answer = item["answer"]
    if normalize(answer) not in normalize(fact):
        problems.append("the answer is not in the fact")
    if normalize(answer) in normalize(item["question"]):
        problems.append("the answer leaks into the question")
    if normalize(item["question"]).count("_____") != 1:
        problems.append("the question must contain exactly one blank")
    elif normalize(item["question"].replace("_____", answer)) != normalize(fact):
        problems.append("refilling the blank does not reproduce the fact")
    if len(positions) != len({*positions}):
        problems.append("duplicate positions")
    return problems


def build(per_document: int = PER_DOCUMENT, traps: int = 10) -> dict:
    corpus.build(CORPUS)
    names = sorted(p.name for p in CORPUS.iterdir() if p.is_file())
    blocks = {name: parsers.parse(CORPUS / name) for name in names}
    all_text = " ".join(b.text for parsed in blocks.values() for b in parsed)
    # Document frequency, so a key can be rejected for being generic.
    df: dict[str, int] = {}
    for parsed in blocks.values():
        for term in {*content_terms(" ".join(b.text for b in parsed))}:
            df[term] = df.get(term, 0) + 1

    items: list[dict] = []
    seen: set[str] = set()
    for name in names:
        for draft in _items_for_document(name, per_document, df):
            key = normalize(draft["question"])
            if key in seen:
                # Two documents quoting the same sentence would otherwise
                # produce two items that are the same question.
                continue
            seen.add(key)
            items.append(
                {
                    "id": f"C{len(items):03d}",
                    "question": draft["question"],
                    "answer": draft["answer"],
                    "source": draft["source"],
                    "answerable": True,
                    "kind": draft["kind"],
                    "label_provenance": f"machine-{draft['kind']}",
                    "expected_positions": draft["expected_positions"],
                    "required_facts": draft["required_facts"],
                    "verified_by_human": False,
                }
            )
    for draft in _traps(all_text, traps):
        items.append(
            {
                "id": f"C{len(items):03d}",
                "question": draft["question"],
                "answer": None,
                "source": "any document in the corpus",
                "answerable": False,
                "kind": "trap",
                "label_provenance": "machine-absence",
                "expected_positions": [],
                "required_facts": [],
                "absent_terms": draft["absent_terms"],
                "verified_by_human": False,
            }
        )

    problems = {item["id"]: check_item(item, blocks) for item in items}
    broken = {i: p for i, p in problems.items() if p}
    for item_id, issues in broken.items():
        print(f"  {item_id} failed: {issues}")

    return {
        "schema": 1,
        "generated_by": "experiments/build_cloze_gold.py",
        "corpus": "experiments/corpus",
        "warning": (
            "Machine-checked labels, not human-reviewed. Every item satisfies the "
            "four mechanical checks recorded in experiments.build_cloze_gold, and "
            "experiments.eval_gold re-runs them before scoring anything. That makes "
            "the labels reproducible and impossible to score stale; it does not make "
            "them expert-checked. These items prove a value survives chunking "
            "attached to its key and can be retrieved again - not that the system "
            "understands the question."
        ),
        "checked": len(items) - len(broken),
        "failed": len(broken),
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--per-document", type=int, default=PER_DOCUMENT)
    args = parser.parse_args()

    gold = build(args.per_document)
    if not args.write:
        print(json.dumps(gold, indent=2, ensure_ascii=False))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gold, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"  {gold['checked']} items pass their own checks, {gold['failed']} fail")
    print(f"  verified by a human: 0 - these are machine-checked, not reviewed")
    return 0 if not gold["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())