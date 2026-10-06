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
import itertools
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

# Six per document. Two-block items are built first and are the only kind that
# can score between zero and one, so the budget goes to them; past six the
# extra items are duplicates of what a document already says.
PER_DOCUMENT = 6
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
# A question has to be able to identify a block on its own: long enough to
# carry meaning, and carrying more than a term or two to match on.
MIN_QUESTION_CHARS = 24
MIN_QUESTION_TERMS = 3
# 'timeout: _____' is 16 characters and is still a good question: a field
# question is identified by its key, not by having a sentence around it, and
# the document-frequency check rejects generic keys. Holding it to the prose
# sentence floor threw away every short config line.
MIN_FIELD_QUESTION_CHARS = 12
# A key term appearing in at most this many documents is specific enough to
# identify its own value.
RARE_TERMS = 3


def _continuation(lines: list[str], at: int) -> str:
    """The text a wrapped line runs on into, bounded to its own paragraph.

    A value at the end of a physical line is usually continued on the next one:
    'The block narrows to nineteen metres' / 'in behind the ball'. Cutting the
    question at the newline gave 'The block narrows to _____', a fragment that
    identifies no block at all and reads like a typo.
    """
    tail: list[str] = []
    for nxt in lines[at + 1 : at + 4]:
        nxt = nxt.strip()
        if not nxt or COMMENT.match(nxt) or SKIP.match(nxt):
            break
        tail.append(nxt)
    return "\n" + "\n".join(tail) if tail else ""


def _candidate_lines(blocks: list[parsers.Block]) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for index, block in enumerate(blocks):
        lines = block.text.splitlines()
        for at, raw in enumerate(lines):
            line = raw.strip()
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
            out.append((index, line, _continuation(lines, at)))
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
    # A bracket or a parenthesis is not an answer. 'const TIERS = [' blanked to
    # '[', and 'median = _____' to '(': no reply could be marked against
    # either, because any of them contains them.
    if not re.search(r"\w", value):
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


KEYLIKE = re.compile(r"[$A-Za-z_][\w$\.\[\]]*")
BARE_KEY = re.compile(r"[$A-Za-z_][\w$\.\[\]]*:\Z")


def _owning_key(text: str, start: int) -> int | None:
    """Offset where the key owning the value at `start` begins.

    `key: value` puts the key first, so the key belongs in the question.
    Bounding the sentence on `': '` alone cut it off *at* the key and kept the
    *next* key instead, giving `_____ $.runtime.cpu_limit:` with `2048` as the
    answer. That reads as "the value of cpu_limit" while labelling
    memory_limit_mb's value, so a reader answers 1500 and is marked wrong.
    Every config line the corpus produced was mislabelled in that direction.
    """
    i = start
    while i > 0 and text[i - 1] in " \t":
        i -= 1
    if i == 0 or text[i - 1] not in ":=":
        return None
    i -= 1
    while i > 0 and text[i - 1] in " \t":
        i -= 1
    j = i
    while j > 0 and text[j - 1] not in " \t\n:":
        j -= 1
    if j == i or not KEYLIKE.fullmatch(text[j:i]):
        return None
    return j


def _window(line: str, start: int, end: int, value: str) -> str | None:
    """The question: the one sentence around the blank, not the whole line.

    A question built from the entire line quotes the block it is answered from,
    so lexical overlap alone nearly solves it and recall@5 sits at 1.000 for
    every chunker - the harness cannot tell a good retriever from a bad one.
    Cutting the question down to its sentence keeps the label mechanically
    checkable while making retrieval do real work: the rest of the block is
    evidence the retriever has to find on its own.

    Returns None when the sentence is too thin to identify anything.
    """
    left = 0
    for stop in (". ", "! ", "? ", "; ", ": "):
        at = line.rfind(stop, 0, start)
        if at != -1 and at + len(stop) > left:
            left = at + len(stop)
    right = len(line)
    for stop in (". ", "! ", "? ", "; ", ": "):
        at = line.find(stop, end)
        if at != -1 and at < right:
            right = at
    if right < len(line):
        right += 1
    while right < len(line) and line[right] in ".;:!?":
        right += 1
    key_at = _owning_key(line, start)
    if key_at is not None and key_at < left:
        left = key_at
    before, after = line[left:start].strip(), line[end:right].strip()
    # The next key must not ride along: a question ending in a key and no value
    # names a key whose value the answer is not.
    if BARE_KEY.fullmatch(after or ""):
        after = ""
    # The blank always survives, even when the value sits at either end of its
    # sentence: dropping it when a side was empty turned '0.0.0.0:8443' into
    # the question '0.0.0.0:', which is not a question at all.
    question = f"{before} _____ {after}".strip()
    if len(question) < MIN_QUESTION_CHARS:
        return None
    return question


def _cloze_from_line(
    line: str, continuation: str = ""
) -> tuple[str, str, str] | None:
    """Blank one value out of a line. Returns (kind, question, answer).

    Only values are blanked. An earlier version also blanked a single
    distinctive word, which produced 'held at a single temperature for _____ an
    hour' answered 'roughly' - a filler adverb that answers nothing. Every
    item here asks for a value the document states.

    `continuation` is the text a wrapped line runs on into. The sentence is cut
    from the pair, not from `line` alone, or a value on the last line of a
    paragraph loses everything that explains it.
    """
    # Strip the clock here rather than trusting every caller to have done it.
    line = TIMESTAMPED.sub("", line).strip()
    # `line` is a prefix of this, so the offsets below are valid in both.
    haystack = line + continuation

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
        # Prefer the value's own sentence. Fall back to the whole line when the
        # sentence is too thin to stand alone, which is what a config line looks
        # like: '$.training.epochs: 140' has no surrounding prose to cut down to.
        # _window already returns the blanked question, so it must not be blanked
        # a second time - doing so found no value and dropped the item.
        question = _window(haystack, start, end, candidate) or (
            line[:start] + "_____" + line[end:]
        )
        # A prose question of one or two terms cannot identify a block: 'of
        # _____ months' is a guess with a blank in it.
        if len({*content_terms(question)}) >= MIN_QUESTION_TERMS:
            return "value", question, candidate

    value = _field_value(line)
    if value is not None:
        at = line.rindex(value)
        question = line[:at] + "_____" + line[at + len(value) :]
        # A field question carries no prose term floor. A distinctive key is
        # exactly what identifies it - '$.limits.rpm' says which limit is meant
        # - and _distinctive rejects the ones that do not.
        if len(question) >= MIN_FIELD_QUESTION_CHARS:
            return "field", question, value
    return None


def _question_key(question: str) -> str | None:
    """The key the blank claims to ask about, when the question names one.

    Both shapes exist - `KEY: _____` and `_____ KEY:` - and both read to a
    person as "the value of KEY", so both have to agree with what the fact
    says KEY holds.
    """
    text = question.strip()
    if text.startswith("_____"):
        text = text[5:].strip()
    elif text.endswith("_____"):
        text = text[:-5].strip()
    else:
        return None
    text = text.strip().rstrip(":").strip()
    return text if text and KEYLIKE.fullmatch(text) else None


def _key_value(fact: str, key: str) -> str | None:
    """What the fact states `key` holds, when the fact is a key/value listing."""
    wanted = normalize(key)
    for line in fact.splitlines():
        match = KEY_VALUE.match(line.strip())
        if match and normalize(match.group("key")) == wanted:
            return match.group("value")
    return None


def _blank(window: str, value: str) -> str | None:
    """One blank, one value. The value must be unambiguous inside its window."""
    if window.count(value) != 1:
        return None
    at = window.index(value)
    return window[:at] + "_____" + window[at + len(value) :]


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


def _evidence(blocks: list[parsers.Block], window: str) -> tuple[str, list[int]] | None:
    """The block a question came from, as the evidence that must come back.

    The fact is the whole parser block, not the line the blank sits on. Citing
    only the line made fact_coverage nearly unfailable - the retriever had
    already been handed the answer's own sentence in the query. Citing the block
    asks whether the evidence around the value came back with it.
    """
    for block in blocks:
        if normalize(window) in normalize(block.text):
            fact = block.text.strip()
            positions = [
                i for i, other in enumerate(blocks) if normalize(fact) in normalize(other.text)
            ]
            return fact, positions
    return None


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
        evidence = _evidence(blocks, body)
        if evidence is None:
            continue
        fact, positions = evidence
        built.append(
            {
                "source": name,
                "kind": "record_cloze",
                "question": body.replace(target, blanked, 1),
                "answer": value,
                "answers": [value],
                "windows": [target],
                "required_facts": [fact],
                "expected_positions": positions,
            }
        )
    return built


def _key_of(line: str) -> str | None:
    matches = list(KEY_VALUE.finditer(line))
    return matches[-1].group("key").casefold() if matches else None


# A sentence that names a concept and then says what it is. The cue is what makes
# the label decidable: 'X is the unsorted sediment deposited directly from ice'
# has one answer for X, where a sentence with no cue does not.
DEFINITION_CUE = re.compile(
    r"\b(?:"
    r"defined as|(?:is|are|was|were) called|known as|refers to|means that|"
    r"states that|(?:is|are) the|is a set of|implies|"
    r"(?:is|are|was|were) (?:driven|caused|written|used|applied) (?:by|as|to|when)|"
    r"(?:is|are) (?:now )?(?:used|found|common|known)|occurs when|"
    r"functioned as|is distinguished from|carries|records|leaves"
    r")\b",
    re.I,
)
# A subject that opens a subordinate clause is not a definition. 'When a glacier
# retreats' answers nothing - the question would be a condition, not a name.
# Requiring the cue to sit in the sentence's first clause kills that case before
# any word-counting: the comma after 'retreats' puts 'leaves' in clause two.
SUBORDINATOR = re.compile(
    r"^(?:when|while|if|because|although|though|after|before|as|since|unless|"
    r"whereas|where|that|and|but|so|once|until)\b",
    re.I,
)
ARTICLE = re.compile(r"^(?:a|an|the)\s+", re.I)
SUBJECT_WORD = re.compile(r"^[A-Za-z][\w'-]*(?:\s+[A-Za-z][\w'-]*){0,3}$")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
# The whole clause has to survive the blank, or the question carries no meaning
# left to retrieve on. These are prose floors, not config-line ones.
MIN_DEFINITION_CHARS = 60
# 'A cadential six four was written as a bass note with a six and a four above
# it' is a clause, not a name. Four words is a term; more is a sentence the
# model would have to reproduce to fill the blank.
MAX_SUBJECT_WORDS = 4


def _definition_items(
    name: str, blocks: list[parsers.Block], per_document: int, df: dict[str, int]
) -> list[dict]:
    """Blank the subject of a definitional sentence.

    Prose documents have no field names and few quotable numbers, so every other
    kind here returns nothing for them and they drop out of the set. A
    definition is what a paragraph does have: 'Till is the unsorted,
    unstratified sediment deposited directly from the ice.' The cue carries the
    meaning, so the subject is recoverable and the label stays mechanically
    checkable by the same four rules as a value cloze.

    The guard that matters is requiring the subject to be rare across the corpus
    and to appear once in its own sentence. Blanking a common word produces
    '_____ is the unsorted sediment' answered 'till' from anywhere in the
    document, and blanking a word the sentence repeats produces a question that
    gives away its own answer.
    """
    built: list[dict] = []
    seen: set[str] = set()
    for block in blocks:
        if len(built) >= per_document:
            break
        for sentence in SENTENCE.split(block.text):
            if len(built) >= per_document:
                break
            sentence = " ".join(sentence.split())
            if len(sentence) < MIN_DEFINITION_CHARS:
                continue
            cue = DEFINITION_CUE.search(sentence)
            if cue is None:
                continue
            # The cue must open the sentence's first clause. A cue in clause two
            # means the text before it is a condition, not a name.
            if any(ch in sentence[: cue.start()] for ch in ",;:"):
                continue
            subject = sentence[: cue.start()].strip()
            subject = SUBORDINATOR.sub("", subject).strip()
            subject = ARTICLE.sub("", subject).strip()
            if not SUBJECT_WORD.match(subject):
                continue
            if len(subject.split()) > MAX_SUBJECT_WORDS:
                continue
            # A gerund phrase is an action, not a name. 'Binding too weakly
            # leaves the activation barrier intact' is answered by anything that
            # leaves it intact, so the label would score noise. A single gerund
            # is a process noun and stays: 'Sintering' is a name the document
            # defines, and it is a good answer.
            if len(subject.split()) > 1 and subject.split()[0].lower().endswith("ing"):
                continue
            # A one-word subject has to be a proper noun to be worth asking for.
            # 'Ice deforms by dislocation creep' answers 'ice', which nothing in
            # the question narrows down.
            terms = [term for term in content_terms(subject) if term]
            if not terms:
                continue
            if not (df.get(terms[0], 0) <= RARE_TERMS or subject[0].isupper()):
                continue
            # The blank has to be unambiguous inside its own sentence.
            if sentence.count(subject) != 1:
                continue
            question = sentence.replace(subject, "_____", 1)
            if normalize(subject) in normalize(question):
                continue
            if len({*content_terms(question)}) < MIN_QUESTION_TERMS:
                continue
            key = normalize(question)
            if key in seen:
                continue
            seen.add(key)
            evidence = _evidence(blocks, sentence)
            if evidence is None:
                continue
            fact, positions = evidence
            built.append(
                {
                    "source": name,
                    "kind": "definition_cloze",
                    "question": question,
                    "answer": subject,
                    "answers": [subject],
                    "windows": [sentence],
                    "required_facts": [fact],
                    "expected_positions": positions,
                }
            )
    return built


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


# A pair may use a shorter window than a prose item: its second half comes from
# another block, so the question as a whole is long enough to identify the
# document even when one blank sits on a short line. Below this it is config
# furniture - 'category: _____', 'owner: _____' - and a pair of those reads as a
# diff rather than a question.
MIN_PAIR_WINDOW_CHARS = 18
# Pairs before triples. A triple is the harder label, but pairs are the ones
# already known to fail, and triples read worse when a third blank is tacked on
# for no reason.
PAIR_BUDGET = 4
TRIPLE_BUDGET = 2

# An answer has to be a value a document states, not a fragment of code. Pairing
# reached into .py and .js files and produced 'last_error = _____' answered
# 'str) -> str:', which is not a fact anyone could answer.
CODEY = re.compile(r"->|\(|\)|\[\]|\{|\}|;|==|!=|=\s|\b(?:def|class|self|import|return)\b")
LITERALS = {"none", "true", "false", "null", "undefined", "nan", "nil"}


def _usable_pair_answer(answer: str) -> bool:
    """Whether an answer is worth asking for when it comes from a pair.

    Pairing raised the yield a lot, and most of what it picked up was not worth
    asking: 'const TIERS = _____' answered '[', 'count: _____' answered
    'int = SPATIAL_FOLDS):', and a reStructuredText underline answered a
    heading. All of it came from code and config lines, where the text either
    side of the colon is punctuation.
    """
    if re.search(r"[^\w\s]{3,}", answer):
        return False  # '===', '):', '[', '0;'
    if CODEY.search(answer):
        return False  # a signature or an assignment, not a value
    if answer.casefold().strip() in LITERALS:
        return False  # 'last_error = _____' answered 'None' asks nothing
    if not re.search(r"[A-Za-z]{3,}", answer) and not re.search(r"\d", answer):
        return False  # no word and no digit: nothing to ask for
    return True


def _multi_block_items(
    name: str,
    blocks: list[parsers.Block],
    per_document: int,
    df: dict[str, int],
    size: int,
    limit: int,
) -> list[dict]:
    """Questions whose answer needs size different blocks.

    A single-block cloze cannot discriminate between retrievers, and that is
    structural rather than a matter of tuning: the question has to name
    something in the block that answers it, so lexical overlap alone nearly
    solves it. Every single-block item scored recall@5 1.000 across every chunk
    size tried.

    More than one block gives partial credit. The retriever has to bring back
    every block, not just the most obvious one, which is the actual product
    failure this system has already had once - an answer that spans two sources
    used to cite only one. It also makes precision@k meaningful, because now more
    than one retrieved chunk is labelled correct. Three blocks go further and
    score 1/3 or 2/3 rather than only half.
    """
    built: list[dict] = []
    candidates: list[tuple[int, str, str]] = []
    for position, line, continuation in _candidate_lines(blocks):
        made = _cloze_from_line(line, continuation)
        if made is None:
            continue
        kind, question, answer = made
        window = question.replace("_____", answer)
        # Each half has to stand on its own. 'category: _____' or 'index: _____'
        # names a field, not a block, and a pair of those reads as a config diff
        # rather than a question.
        if len(question) < MIN_PAIR_WINDOW_CHARS or not _usable_pair_answer(answer):
            continue
        if kind == "field":
            # Take the key from the refilled window. The blanked side has no
            # value left for the 'key: value' pattern to match, so reading it
            # from the question returned nothing and dropped every field pair.
            key = _key_of(window)
            if df and not _distinctive(key or "", df, RARE_TERMS):
                continue
        candidates.append((position, window, answer))

    for combo in itertools.combinations(candidates, size):
        if len(built) >= min(limit, per_document):
            break
        # size blanks have to come from size different blocks and carry
        # size different answers, or the question has fewer right answers
        # than it appears to.
        if len({c[0] for c in combo}) != size or len({c[2] for c in combo}) != size:
            continue
        windows = [c[1] for c in combo]
        answers = [c[2] for c in combo]
        # Blocks that share most of their wording are one fact, not several.
        terms = [set(content_terms(w)) for w in windows]
        if any(
            len(one & two) > 0.6 * min(len(one), len(two))
            for one, two in itertools.combinations(terms, 2)
        ):
            continue
        blanked = [_blank(w, a) for w, a in zip(windows, answers)]
        if any(b is None for b in blanked):
            continue
        evidence = [_evidence(blocks, w) for w in windows]
        if any(part is None for part in evidence):
            continue
        facts = [part[0] for part in evidence]  # type: ignore[index]
        cited = sorted({p for part in evidence for p in part[1]})  # type: ignore[union-attr]
        # The evidence has to be size different blocks. Windows can land in
        # one block even when their lines came from different blocks, and an
        # item citing one block twice is a single-block item wearing a
        # multi-block label - it cannot score between zero and one.
        if len({normalize(f) for f in facts}) != size or len(cited) < size:
            continue
        built.append(
            {
                "source": name,
                "kind": "multi_block_cloze",
                "block_count": size,
                "question": "\n".join(blanked),
                "answer": answers[0],
                "answers": answers,
                "windows": windows,
                "required_facts": facts,
                "expected_positions": cited,
            }
        )
    return built


def _pair_items(
    name: str, blocks: list[parsers.Block], per_document: int, df: dict[str, int]
) -> list[dict]:
    """Pairs first, then triples: the only items that can score between 0 and 1."""
    built = _multi_block_items(name, blocks, per_document, df, 2, PAIR_BUDGET)
    if len(built) < per_document:
        built += _multi_block_items(
            name, blocks, per_document, df, 3, PAIR_BUDGET + TRIPLE_BUDGET
        )
    return built[:per_document]


def _items_for_document(
    name: str, per_document: int, df: dict[str, int] | None = None
) -> list[dict]:
    path = CORPUS / name
    if not path.exists():
        return []
    blocks = parsers.parse(path)
    df = df or {}
    # Two-block items first: they are the only kind here that can tell a good
    # retriever from a bad one, so they take priority over easier labels.
    built = _pair_items(name, blocks, per_document, df)
    if len(built) >= per_document:
        return built
    # Tabular documents next: a record names itself and gives a decisive
    # question, where a bare field name repeated down a file does not.
    built += _record_items(name, blocks, per_document - len(built))
    if len(built) >= per_document:
        return built
    # Prose last. A paragraph states no field and quotes no number, so both
    # cloze kinds above find nothing in it and the document is skipped entirely
    # - which is how catalysis.txt, glaciology.pdf and music.txt were absent from
    # a set claiming to cover every document in the corpus. Definitions are the
    # one prose shape that yields a decidable label: the cue names the concept,
    # so blanking the subject asks a question with one answer.
    built += _definition_items(name, blocks, per_document - len(built), df)
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
    for position, line, continuation in _candidate_lines(blocks):
        if len(built) >= per_document:
            break
        clozed = _cloze_from_line(line, continuation)
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
        evidence = _evidence(blocks, question.replace("_____", answer))
        if evidence is None:
            continue  # the question's own sentence is not in any parsed block
        fact, positions = evidence
        built.append(
            {
                "source": name,
                "kind": f"{kind}_cloze",
                "question": question,
                "answer": answer,
                "answers": [answer],
                "windows": [question.replace("_____", answer)],
                "required_facts": [fact],
                "expected_positions": positions,
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

    answers = item.get("answers") or [item["answer"]]
    windows = item.get("windows") or [item["question"].replace("_____", item["answer"])]
    if len(answers) != len(facts) or len(windows) != len(facts):
        problems.append("each fact needs exactly one answer and one window")
        return problems
    if normalize(item["question"]).count("_____") != len(facts):
        problems.append(f"the question must contain exactly {len(facts)} blanks")

    haystack = normalize(item["question"])
    for fact, answer, window in zip(facts, answers, windows):
        # Every fact must be a real block: cited positions must actually hold it.
        if not any(
            position < len(parsed) and normalize(fact) in normalize(parsed[position].text)
            for position in positions
        ):
            problems.append("no cited position contains its fact")
        if normalize(answer) not in normalize(fact):
            problems.append(f"the answer {answer!r} is not in its fact")
        if normalize(answer) in haystack:
            problems.append(f"the answer {answer!r} leaks into the question")
        # The window is the text the blank was cut from. It must be verbatim
        # text of the fact, and blanking it must reproduce part of the question,
        # or the label was written by something other than the parser's output.
        if normalize(window) not in normalize(fact):
            problems.append("the window is not text from its fact")
        # A blank standing in front of a key reads as "the value of that key",
        # so the answer has to be that key's value. Without this the checks are
        # blind to an off-by-one: '2048 $.runtime.cpu_limit:' is verbatim text
        # of the flattened fact, so window, answer and question all pass while
        # the item still labels the wrong line.
        named = _question_key(item["question"])
        if named is not None:
            stated = _key_value(fact, named)
            if stated is not None and normalize(stated) != normalize(answer):
                problems.append(
                    f"the question names {named!r}, whose value is {stated!r}, "
                    f"not {answer!r}"
                )
        blanked = _blank(window, answer)
        if blanked is None or normalize(blanked) not in haystack:
            problems.append("the window is not in the question")
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
                    "answers": draft["answers"],
                    "windows": draft["windows"],
                    "source": draft["source"],
                    "answerable": True,
                    "kind": draft["kind"],
                    "label_provenance": f"machine-{draft['kind']}",
                    "expected_positions": draft["expected_positions"],
                    "required_facts": draft["required_facts"],
                    # How many blocks a retriever has to bring back for this to
                    # be fully answered. Recorded per item because it is what
                    # decides whether the item can score between 0 and 1.
                    "block_count": len(draft["required_facts"]),
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
            "mechanical checks recorded in experiments.build_cloze_gold, and "
            "experiments.eval_gold re-runs them before scoring anything. That makes "
            "the labels reproducible and impossible to score stale; it does not make "
            "them expert-checked. These items prove a value survives chunking "
            "attached to its key and can be retrieved again - not that the system "
            "understands the question. A multi_block_cloze item needs two or three "
            "separate blocks, so it can be partly right; the rest have one right "
            "answer and measure only whether it was found. A definition_cloze item "
            "asks for the subject of a sentence that defines it, which is the only "
            "decidable question prose yields; it measures whether the defining "
            "sentence came back, not whether the term was understood."
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