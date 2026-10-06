"""Does the generated answer stay faithful to the passages it cites?

Retrieval can be perfect and the answer still wrong: the model can read the
right chunk, misread it, or blend it with something it remembered. The gold
harness measures whether the evidence reached the prompt (`fact_coverage` in
`metrics.py`); nothing measured what came back out. That is the gap this closes.

The three measurements here are the standard triad, deliberately computed
without a judge model:

  answer_relevancy   did the answer say the thing it was asked? Scored by
                     whether the required facts appear in the answer at all.
  faithfulness      is every claim traceable to a cited passage? Scored per
                     sentence, against the passages that sentence cites.
  citation_precision are the citations load-bearing? A `[n]` attached to
                     nothing is decoration, and decoration reads as evidence.

Why no LLM judge
----------------
An LLM judge is the usual answer and it was rejected here. It costs money per
evaluation, it is non-deterministic, and it is the same family of model as the
one being graded, so its errors correlate with the errors being measured. A
metric that fails for the same reason the system fails cannot be used to detect
that failure. `metrics.py` makes the same argument for its definitions.

The cost of this choice is real and worth stating: these are lexical proxies,
not entailment judgements. "The limit is 600 rpm" supports "the limit is 600
per minute" but not "the limit is 600 requests per hour" - the token overlap is
identical and only one is faithful. So a proxy can be fooled by a close
paraphrase that changes the meaning, and cannot detect semantic entailment that
shares no words. These numbers are a floor and a regression alarm, not proof of
grounding. That is the same standard the retrieval metrics are held to, and the
same reason they are written out in full here.

Every function is pure and takes plain sequences, so each definition can be
checked by hand against a worked example:

    .venv/bin/python -m experiments.generation
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .metrics import normalize

__all__ = [
    "CITATION_RE",
    "answer_relevancy",
    "citation_precision",
    "extract_citations",
    "faithfulness",
    "score_answer",
    "sentences",
    "support",
]

# The markers the app itself emits, canonicalised by `llm.validate_citations`
# before an answer reaches the UI. Only ASCII and the fullwidth/CJK variants,
# because those are the ones the validator rewrites to `[n]`.
CITATION_RE = re.compile(r"[\[［【]([0-9０-９]{1,3})[\]］】]")

# Sentences, not lines and not whitespace splits. A claim needs a subject and a
# predicate to be checkable, and a bare fragment is neither. Abbreviations and
# decimals are protected because "No. 5" and "3.5" are not sentence ends.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n{2,}")
_ABBREVIATIONS = {
    "no", "fig", "eq", "ref", "sec", "ch", "vol", "pp", "para", "approx",
    "e.g", "i.e", "vs", "mr", "mrs", "ms", "dr", "st", "cf", "al",
}

# Words carrying no evidential content. Stripping them keeps "the" in both
# strings from inflating overlap, which is how a lexical proxy most easily lies.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "has", "have", "he", "her", "his", "in", "is", "it", "its",
    "of", "on", "or", "she", "that", "the", "their", "there", "these", "they",
    "this", "to", "was", "were", "which", "who", "will", "with", "would",
    "you", "your", "am", "can", "do", "does", "did", "not", "no", "any",
    "all", "if", "then", "than", "so", "such", "when", "where", "while",
    "each", "into", "over", "under", "about", "after", "before", "also",
}

_WORD_RE = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")

# An unsupported sentence below this ratio is called a hallucination. The
# threshold is a guess, and it is documented as one: it is set high enough that
# a correct answer phrased differently from its source does not get flagged, and
# low enough that a claim with no real lexical relationship trips it. The
# sensitivity sweep in `eval_gold` is what would justify changing it, and until
# that sweep exists the number is reported alongside the per-sentence detail so
# a reader can see how close any given answer was.
SUPPORT_THRESHOLD = 0.34


def extract_citations(text: str) -> list[int]:
    """Every citation number in `text`, in order, duplicates kept.

    Order matters and duplicates are kept on purpose: `[3] [3]` is two citation
    markers that need two passages to support, and collapsing them would hide a
    sentence claiming two separate supports and finding one.
    """
    return [
        int(digit.translate({ord("０") + i: str(i) for i in range(10)}))
        for digit in CITATION_RE.findall(text or "")
    ]


def sentences(text: str) -> list[str]:
    """Split an answer into checkable claims.

    Line breaks and bullet markers are normalised first, so a list answer is
    scored claim by claim instead of as one long paragraph where a single
    unsupported clause is diluted by the rest.
    """
    if not text:
        return []
    cleaned = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", text.strip(), flags=re.M)
    parts: list[str] = []
    for chunk in _SENTENCE_SPLIT.split(cleaned):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts.append(chunk)
    return parts


def _words(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(normalize(text)) if w not in _STOPWORDS]


def support(claim: str, passages: Sequence[str]) -> tuple[float, list[str]]:
    """Fraction of a claim's content words present in the cited passages.

    Returns the ratio and the content words that were missing, because the ratio
    alone cannot be acted on and the missing words are the whole finding.

    A recall ratio over the claim's own vocabulary, not precision over the
    passage's: a passage is much larger than the claim it supports, so scoring
    precision would measure passage length and punish well-supported claims in
    dense passages.
    """
    wanted = _words(claim)
    if not wanted:
        # A claim with no content words ("It depends.") cannot be checked.
        # Reported as fully supported rather than zero, because calling it a
        # hallucination would be a false accusation based on a metric failure.
        return 1.0, []
    haystack = " ".join(passages)
    have = set(_words(haystack))
    missing = sorted({w for w in wanted if w not in have})
    return (len(wanted) - len(missing)) / len(wanted), missing



# Spelled-out numbers, so "sixty-six" and "66" are one value. Without this the
# metric marks a correct answer wrong for using digits, which is a formatting
# difference, not a knowledge difference.
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_HYPHEN_NUMBER = re.compile(r"\b([a-z]+)-([a-z]+)\b")

# Units, collapsed to a canonical token. An answer that writes "°C" and a label
# that writes "degrees Celsius" must not miss on unit vocabulary alone.
_UNIT_ALIASES = {
    "celsius": "c", "centigrade": "c",
    "fahrenheit": "f",
    "percent": "%", "percentage": "%",
    "minutes": "min", "mins": "min", "minute": "min",
    "seconds": "s", "secs": "s", "sec": "s", "second": "s",
    "hours": "h", "hrs": "h", "hour": "h",
    "days": "day", "weeks": "week", "months": "month", "years": "year",
    "kilograms": "kg", "kilogram": "kg", "grams": "g", "gram": "g",
    "milligrams": "mg", "milligram": "mg",
    "millimetres": "mm", "millimeter": "mm", "metres": "m", "meter": "m",
    "kilometres": "km", "kilometer": "km", "litres": "l", "liter": "l",
    "pounds": "lb", "pound": "lb", "ounces": "oz", "ounce": "oz",
}

# Words that carry no claim content. Broader than the stopword list because this
# runs over answer prose, where "figure" and "approximately" are noise.
_NOISE = _STOPWORDS | {
    "approximately", "roughly", "about", "figure", "table", "section",
    "according", "described", "source", "passage", "document", "chunk",
    "answer", "question", "following", "above", "below", "here", "there",
    # Unit scaffolding, not claim content. "degrees Celsius" and "°C" are one
    # value; counting "degrees" as a required content word would make a correct
    # answer miss on a word that carries no information.
    "degrees", "degree", "unit", "units", "value", "values", "amount",
    "at", "least", "most", "up", "out", "off", "per",
}


def _numbers_in(words: Sequence[str]) -> list[str]:
    """Canonicalise spelled-out numbers to digit strings.

    Handles "sixty", "sixty six" and "sixty-six" as 60, 66 and 66 respectively.
    Hyphenated pairs are read as tens-plus-units ("twenty-five" -> 25) and not
    as two separate numbers, because that is how they are written.
    """
    out: list[str] = []
    for word in words:
        if word in _NUMBER_WORDS:
            out.append(str(_NUMBER_WORDS[word]))
        else:
            out.append(word)

    joined: list[str] = []
    for word in out:
        if word in _TENS:
            joined.append(str(_TENS[word]))
        else:
            joined.append(word)

    merged: list[str] = []
    for match in _HYPHEN_NUMBER.finditer(" ".join(joined)):
        head, tail = match.group(1), match.group(2)
        if head in _TENS and tail in _NUMBER_WORDS and _NUMBER_WORDS[tail] < 10:
            value = _TENS[head] + _NUMBER_WORDS[tail]
            merged.append((match.start(), match.end(), str(value)))
    if not merged:
        return joined

    # Splice the hyphenated composites back in, left to right.
    text = " ".join(joined)
    for start, end, value in reversed(merged):
        text = text[:start] + value + text[end:]
    return text.split()


def _content(text: str) -> list[str]:
    """Content words with numbers and units canonicalised, deduplicated.

    A hyphenated compound contributes both the compound and its parts, so
    "short-lived" matches a fact reading "short". Keeping only the compound made
    a correct answer miss on wording, and only the parts would let "high" match
    inside "highlight", which is a different word entirely.
    """
    words = _numbers_in(_WORD_RE.findall(normalize(text)))
    seen: list[str] = []
    for word in words:
        for part in (word, *word.split("-")) if "-" in word else (word,):
            if not part or part in _NOISE:
                continue
            token = _UNIT_ALIASES.get(part, part)
            if token not in seen:
                seen.append(token)
    return seen


def _claims_with_citations(text: str) -> list[tuple[str, list[int]]]:
    """Pair each claim with the passages it points at.

    A citation marker at the end of a sentence is taken as supporting that
    sentence. A marker mid-sentence belongs to the sentence it appears in, which
    is why the split happens on sentences rather than on clauses.
    """
    out: list[tuple[str, list[int]]] = []
    for claim in sentences(text):
        found = extract_citations(claim)
        if found:
            out.append((claim, found))
    return out


def faithfulness(
    answer: str,
    passages: Sequence[Mapping[str, object] | str],
) -> dict[str, object]:
    """Every cited claim traced back to the passages it cites.

    A claim is only judged against the passages it names. Judging it against
    everything retrieved would reward the model for citing the wrong passage:
    the words would be present somewhere in the context and the answer would
    score well while being unauditable. This is the difference between an answer
    that is verifiable and an answer that is merely true.

    Returns per-claim detail alongside the score, because "faithfulness 0.82"
    alone cannot be acted on.
    """
    resolved = [
        p if isinstance(p, str) else str(p.get("text") or "") for p in passages
    ]
    claims = _claims_with_citations(answer)

    rows: list[dict[str, object]] = []
    unsupported: list[str] = []
    for claim, cited in claims:
        # An out-of-range citation names no passage, so nothing can support it.
        # `llm.validate_citations` normally strips these; scoring here catches
        # the case where an answer reached this function without passing it.
        usable = [resolved[n - 1] for n in cited if 1 <= n <= len(resolved)]
        ratio, missing = support(claim, usable)
        ok = ratio >= SUPPORT_THRESHOLD
        if not ok:
            unsupported.append(claim)
        rows.append({
            "claim": claim,
            "citations": cited,
            "support": round(ratio, 3),
            "missing": missing,
            "grounded": ok,
            "dangling": [n for n in cited if not (1 <= n <= len(resolved))],
        })

    cited_claims = len(claims)
    grounded = sum(1 for r in rows if r["grounded"])
    return {
        "score": grounded / cited_claims if cited_claims else 0.0,
        "cited_claims": cited_claims,
        "grounded": grounded,
        "unsupported": unsupported,
        "claims": rows,
        "threshold": SUPPORT_THRESHOLD,
    }


def citation_precision(answer: str, passages: Sequence[Mapping[str, object] | str]) -> dict[str, object]:
    """Of the citations the model emitted, how many are load-bearing.

    Precision here is over citations, not over claims: a claim that cites three
    passages when one suffices spends context and invites the reader to check
    evidence that was never used. Uncited claims are excluded rather than counted
    as failures - they are a faithfulness problem, not a citation problem, and
    double-counting them would let one bad sentence sink two metrics.
    """
    faith = faithfulness(answer, passages)
    total = sum(len(r["citations"]) for r in faith["claims"])  # type: ignore[index]
    if not total:
        return {
            "score": 0.0,
            "citations": 0,
            "load_bearing": 0,
            "unused": [],
        }

    # A citation is load-bearing when removing its passage would break the claim
    # it appears on. Probed one at a time, because a claim citing [1] and [3]
    # where only [3] matters has one used citation and one decoration.
    resolved = [
        p if isinstance(p, str) else str(p.get("text") or "") for p in passages
    ]
    used = 0
    unused: list[int] = []
    for row in faith["claims"]:  # type: ignore[union-attr]
        claim, cited = str(row["claim"]), list(row["citations"])  # type: ignore[arg-type]
        for n in cited:
            others = [resolved[m - 1] for m in cited if m != n and 1 <= m <= len(resolved)]
            alone, _ = support(claim, [resolved[n - 1]] if 1 <= n <= len(resolved) else [])
            # Still supported without this passage, and supported with it: the
            # passage adds nothing the others did not already provide.
            if alone < SUPPORT_THRESHOLD and support(claim, others)[0] >= SUPPORT_THRESHOLD:
                unused.append(n)
            else:
                used += 1
    return {
        "score": used / total,
        "citations": total,
        "load_bearing": used,
        "unused": sorted(set(unused)),
    }


def answer_relevancy(
    answer: str,
    required_facts: Sequence[str],
) -> dict[str, object]:
    """Did the answer say the thing it was asked?

    Scored on **content words, not literal substrings**, and the reason is worth
    stating because it was found the hard way. `fact_coverage` in `metrics.py`
    matches gold facts literally, and that is correct there: it asks whether the
    evidence reached the prompt, and the chunk text is the chunk text. It does
    not transfer to answers. A correct answer paraphrases and normalises units,
    so a fact labelled "mash at sixty-six degrees Celsius" is missed by the
    perfectly good answer "mash at 66 °C favours a balanced body". Measured that
    way, six out of six perfect answers scored 0.0.

    So relevance counts content words of each required fact found in the answer,
    with numbers and units canonicalised first so "sixty-six", "66" and "66 °C"
    are one value rather than four mismatches. A fact is covered when a majority
    of its content words are present, which tolerates rephrasing without
    tolerating a different answer.

    Three signals, because "did it answer the question" has two distinct answers
    and one number cannot carry both:

      substance  paraphrase-tolerant. The fraction of the *union* of all
                 required-fact content words that appear in the answer. A
                 correct answer that rewords the fact scores well here, so this
                 is the signal for "did it address the substance".
      score      strict. The fraction of required facts covered, each requiring
                 all of its content words. Rewording drops it, so this is the
                 signal for "did it land every labelled fact", and it is the one
                 that catches a changed unit.
      verbatim   exact-substring, kept as the strictest bound. The gap between
                 verbatim and substance is a direct measurement of how much the
                 system paraphrases, which is worth knowing rather than hiding.

    Both are needed because either alone is misleading. Strict alone reports the
    paraphrase rate as if it were a quality problem; tolerant alone passed
    "600 requests per hour" against a "600 rpm" fact. The pair brackets the truth
    from both sides.

    `kind` is derived from `score` and reported as a distribution rather than a
    mean, because partial has a different fix from missed - a longer context or
    better retrieval, versus a model that did not understand the question.

    `no_required_facts` is a distinct kind, not a zero: there is nothing to be
    relevant to, and averaging in a 0.0 would punish a question nobody labelled.
    """
    if not required_facts:
        return {
            "score": 0.0,
            "substance": 0.0,
            "verbatim": 0.0,
            "found": [],
            "missing": [],
            "total": 0,
            "kind": "no_required_facts",
        }

    # Citation markers are stripped before any content comparison. Otherwise
    # "[1]" contributes the token "1" to the answer, which can then satisfy a
    # fact that happens to contain the digit 1 - a silent false pass that looks
    # like a correct answer.
    body = CITATION_RE.sub(" ", answer or "")
    answer_words = set(_content(body))
    haystack = normalize(body)

    wanted_all: set[str] = set()
    for fact in required_facts:
        wanted_all.update(_content(fact))

    found: list[str] = []
    missing: list[str] = []
    verbatim = 0
    for fact in required_facts:
        wanted = _content(fact)
        if wanted and all(word in answer_words for word in wanted):
            found.append(fact)
        else:
            missing.append(fact)
        if normalize(fact) in haystack:
            verbatim += 1

    total = len(required_facts)
    covered = len(found)
    if not (answer or "").strip():
        kind = "refused"
    elif covered == total:
        kind = "complete"
    elif covered:
        kind = "partial"
    else:
        kind = "missed"

    return {
        "score": covered / total,
        "substance": (
            sum(1 for word in wanted_all if word in answer_words) / len(wanted_all)
            if wanted_all
            else 0.0
        ),
        "verbatim": verbatim / total,
        "found": found,
        "missing": missing,
        "total": total,
        "kind": kind,
    }


def score_answer(
    answer: str,
    passages: Sequence[Mapping[str, object] | str],
    required_facts: Sequence[str] = (),
    verdict: str = "answered",
) -> dict[str, object]:
    """All three metrics for one answer, with the refusal kept separate.

    An answer that correctly refuses carries no claims, so faithfulness over it
    is 0.0 and averaging that in would punish the safest possible behaviour.
    `verdict` is passed in rather than sniffed from the answer text, because the
    server already decided it - and matching on a phrase like "I could not" is a
    guess that breaks the day the wording changes.
    """
    refused = verdict != "answered"
    faith = faithfulness(answer, passages)
    return {
        "relevancy": answer_relevancy(answer, required_facts),
        "faithfulness": faith,
        "citations": citation_precision(answer, passages),
        "uncited_claims": max(0, len(sentences(answer)) - faith["cited_claims"]),  # type: ignore[operator]
        "refused": refused,
        "verdict": verdict,
    }


def _tokens(text: str) -> list[str]:
    """Lowercased alphanumeric runs.

    Camel case is left whole on purpose: the cloze answers are Java identifiers
    like `declarationNumber`, and splitting them at the case boundary would let
    any sentence containing both halves count as a match.
    """
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _has_run(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    return any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def reference_score(answer: str, expected: Sequence[str] | None) -> dict[str, object]:
    """Did the answer contain the value the document actually holds?

    The answer key, not the evidence: `expected` is the text that filled a gap
    in the source document, so the label comes from the document and not from
    anything the retriever or the generator produced. That is what makes this
    the one non-circular correctness check available here, and why it is
    reported beside faithfulness rather than folded into it.

    Lexical for the same reason a judge model is refused at the top of this
    file: embedding similarity or an LLM judge puts the model under test inside
    its own grading. The cost of that choice is stated rather than hidden -- a
    correct paraphrase of a short value scores 0, so a low number here means
    "did not contain the words", which is weaker than "did not know".

    `exact` is the expected value appearing as a contiguous run of words in the
    answer. `partial` is every word of the expected value appearing somewhere,
    in any order: that catches an answer that has the right parts but scatters
    them, which is the usual shape of a hedged or rambling reply. A refusal
    scores neither.
    """
    found = _tokens(answer)
    exact: str | None = None
    partial: str | None = None
    for raw in expected or ():
        want = _tokens(raw)
        if not want:
            continue
        if exact is None and _has_run(found, want):
            exact = str(raw)
        elif exact is None and partial is None and all(w in found for w in want):
            partial = str(raw)
        if exact is not None:
            break
    return {
        "exact": exact is not None,
        "partial": partial is not None and exact is None,
        "covered": exact is not None or partial is not None,
        "matched": exact or partial,
        "expected": [str(x) for x in (expected or ())],
    }


def _self_check() -> None:
    """Worked examples, so a wrong definition fails here and not in a report."""
    passages = [
        "Standard returns are accepted within 30 days of delivery.",
        "The contract was signed in March 2019.",
    ]

    # A claim copied from its source is fully supported.
    faith = faithfulness(
        "Standard returns are accepted within 30 days of delivery [1].", passages
    )
    assert faith["score"] == 1.0, faith
    assert faith["unsupported"] == [], faith

    # A claim from nowhere is not, and names what is missing.
    faith = faithfulness("Refunds take up to 90 days [1].", passages)
    assert faith["score"] == 0.0, faith
    assert "90" in faith["claims"][0]["missing"], faith

    # The load-bearing test: citing both passages when one says it all is
    # precision 0.5, which is the whole point of the metric.
    prec = citation_precision(
        "Standard returns are accepted within 30 days of delivery [1][2].", passages
    )
    assert prec["citations"] == 2, prec
    assert prec["load_bearing"] == 1, prec
    assert prec["unused"] == [2], prec

    # A citation pointing past the end of the list supports nothing, and must not
    # crash or be silently treated as [1].
    faith = faithfulness("Standard returns are accepted within 30 days [9].", passages)
    assert faith["score"] == 0.0, faith
    assert faith["claims"][0]["dangling"] == [9], faith

    # The answer key is matched as a run of words, case- and punctuation-blind.
    ref = reference_score(
        "The declaration number is this.declarationNumber = declarationNumber;",
        ["declarationNumber;"],
    )
    assert ref["exact"] is True and ref["covered"] is True, ref

    # Every word present but scattered is only partial -- close enough to see,
    # not close enough to call correct.
    ref = reference_score("Grant it for months, 12 days at a time", ["12 months"])
    assert ref["exact"] is False and ref["partial"] is True, ref
    assert ref["covered"] is True, ref

    # A refusal is neither. It must not borrow a pass from an empty answer.
    ref = reference_score("", ["12 months"])
    assert ref["covered"] is False and ref["matched"] is None, ref

    # No label, no claim: traps carry no answer key and must stay unscored.
    assert reference_score("anything", None)["covered"] is False
    assert reference_score("anything", ["", "  "])["covered"] is False

    # Fullwidth markers are the same citation, not a different one.
    assert extract_citations("30 days ［1］ and 【２】") == [1, 2]

    # Relevancy distinguishes complete from partial, which have different fixes.
    # Digits must match a spelled-out label, or a correct answer scores 0.
    spelled = answer_relevancy(
        "A single-infusion mash at 66 °C favours a balanced body [1].",
        ["mash at sixty-six degrees Celsius favours a balanced body."],
    )
    assert spelled["kind"] == "complete", spelled
    assert spelled["score"] == 1.0, spelled
    # And the strict signal is reported separately: it did not match verbatim,
    # which is information rather than an error.
    assert spelled["verbatim"] == 0.0, spelled

    # The case a majority threshold gets wrong in the dangerous direction: the
    # answer shares most of the fact's vocabulary and changes the unit, so it is
    # confidently wrong and must not read as covered.
    wrong_unit = answer_relevancy("The limit is 600 requests per hour [1].", ["The limit is 600 rpm."])
    assert wrong_unit["kind"] == "missed", wrong_unit

    # A citation marker must not count as answer content: otherwise "[1]" supplies
    # the token "1" and can satisfy a fact that merely contains that digit.
    cited = answer_relevancy("The limit is 600 rpm [1].", ["600 rpm 1"])
    assert "1" not in _content(CITATION_RE.sub(" ", "The limit is 600 rpm [1].")), cited

    # A partial answer is a partial, not a miss - but only when there is more than
    # one fact to be partial about. A single-fact item is complete or missed;
    # `kind: partial` never fires on one, which is worth knowing before reading
    # a distribution and wondering where the partials went.
    part = answer_relevancy(
        "The wort is boiled for at least 60 minutes [1].",
        ["After the wort is drained, it is boiled for at least sixty minutes.",
         "Wort is boiled for sixty minutes."],
    )
    assert part["kind"] == "partial", part
    assert part["score"] == 0.5, part

    # An answer that omits every fact is a miss, not a partial.
    assert answer_relevancy("Unrelated.", ["Alpha", "Beta"])["kind"] == "missed"

    # The reworded-but-correct case: strict misses it, substance catches it.
    # That gap is why both numbers are reported.
    reworded = answer_relevancy(
        "A surge occurs when melt-water reaches the base, reducing friction and "
        "triggering a short-lived but dramatic acceleration of flow [1].",
        ["A surge is a short episode of dramatic acceleration."],
    )
    assert reworded["kind"] == "missed", reworded
    assert reworded["substance"] >= 0.7, reworded

    partial = answer_relevancy("There is a rate limit.", ["600 rpm"])
    assert partial["kind"] == "missed" and partial["missing"] == ["600 rpm"], partial

    # Hyphenated numbers are tens-plus-units, not two numbers.
    assert _content("twenty-five") == ["25"], _content("twenty-five")
    assert _content("sixty six") == ["60", "6"], _content("sixty six")

    # A refusal is a distinct case, not a zero to average in.
    empty = answer_relevancy("", ["600 rpm"])
    assert empty["kind"] == "refused", empty

    # The verdict comes from the server, not from matching the wording, so
    # rewording the refusal cannot silently turn it into a hallucination.
    out = score_answer("I could not find anything relevant.", passages, ["30 days"],
                       verdict="no_match")
    assert out["refused"] is True, out
    assert out["faithfulness"]["cited_claims"] == 0, out

    # Multiple claims are scored individually, so one bad sentence is visible
    # instead of diluted by three good ones.
    mixed = faithfulness(
        "Returns are accepted within 30 days [1]. Refunds take 90 days [2].", passages
    )
    assert mixed["cited_claims"] == 2, mixed
    assert mixed["grounded"] == 1, mixed

    print("generation metrics: all self-checks passed")


if __name__ == "__main__":
    _self_check()