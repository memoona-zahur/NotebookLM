"""Finding the exact text behind a citation.

The model is asked to quote the shortest passage text that supports each claim.
This module decides whether to believe it, and where in the chunk (and the
original document) that text is. It has three jobs:

1. `normalise` - one text-folding function used on *both* sides of every
   comparison, with a map from folded offsets back to raw ones. Ligatures,
   soft hyphens, line-break hyphenation and whitespace differ between what a
   PDF extractor stores and what a model types back; folding both sides the
   same way makes a verbatim quote match, and the map makes the match land on
   the real characters.
2. `validate_quotes` - a quote is accepted only if it is a verbatim substring of
   the chunk after folding. A quote that is not is dropped, never "repaired".
3. `match_claim` - the fallback when no quote validates: the one sentence of the
   chunk that best matches the one sentence of the answer carrying the marker.
   Weak matches return nothing. No highlight beats a wrong one.

Offsets handed to the browser are UTF-16 code units (`to_utf16_spans`), because
Python counts code points and a JavaScript string counts units: one emoji before
the span shifts every later offset by one.
"""

import re
import unicodedata

# -- offsets ---------------------------------------------------------------


def to_utf16(text: str, index: int) -> int:
    """`index` (code points into `text`) as UTF-16 code units."""
    index = max(0, min(index, len(text)))
    return index + sum(1 for ch in text[:index] if ord(ch) > 0xFFFF)


def to_utf16_spans(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Convert many `(start, end)` code-point spans over one text in a single pass."""
    if not spans:
        return []
    cuts = sorted({edge for span in spans for edge in span})
    converted: dict[int, int] = {}
    extra = 0
    previous = 0
    for cut in cuts:
        cut = max(0, min(cut, len(text)))
        extra += sum(1 for ch in text[previous:cut] if ord(ch) > 0xFFFF)
        converted[cut] = cut + extra
        previous = cut
    return [(converted[max(0, min(s, len(text)))], converted[max(0, min(e, len(text)))])
            for s, e in spans]


# -- folding ---------------------------------------------------------------

_CURLY = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
}


def normalise(text: str) -> tuple[str, list[int]]:
    """Fold `text` for comparison; return it with `raw_index[i]` for each char.

    What is folded, identically on every call:

    - ligatures and compatibility forms (NFKC, per character: "ﬁ" -> "fi");
    - typographic quotes to ASCII;
    - soft hyphens are deleted;
    - a hyphen between two letters is deleted, and so is a hyphen at a line
      break between two letters or digits ("well-\\nknown", "well-known" and
      "wellknown" all fold to "wellknown") - an extractor cannot tell a real
      hyphen from a line-break one, so neither side is allowed to;
    - every run of whitespace becomes one space.

    `raw_index[i]` is the offset in `text` of the character that produced folded
    character `i`. Several folded characters can share one raw index (a
    ligature), and deleted characters appear in no entry, which is why a span is
    mapped back with `raw_span` rather than by subtracting.
    """
    folded: list[str] = []
    raw_index: list[int] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch == "­":
            i += 1
            continue
        if ch == "-":
            before = text[i - 1] if i else ""
            # Hyphen at a line break: skip the break and the indent after it.
            j = i + 1
            while j < n and text[j] in " \t":
                j += 1
            if j < n and text[j] in "\r\n" and before.isalnum():
                while j < n and text[j] in " \t\r\n":
                    j += 1
                if j < n and text[j].isalnum():
                    i = j
                    continue
            # A hyphen inside a word.
            if before.isalpha() and i + 1 < n and text[i + 1].isalpha():
                i += 1
                continue
        if ch.isspace():
            if folded and folded[-1] != " ":
                folded.append(" ")
                raw_index.append(i)
            i += 1
            continue
        out = _CURLY.get(ch) or unicodedata.normalize("NFKC", ch)
        for piece in out:
            if piece.isspace():
                if folded and folded[-1] != " ":
                    folded.append(" ")
                    raw_index.append(i)
            else:
                folded.append(piece)
                raw_index.append(i)
        i += 1
    return "".join(folded), raw_index


def raw_span(raw_index: list[int], start: int, end: int) -> tuple[int, int]:
    """The raw `[start, end)` that a folded `[start, end)` came from.

    It begins at the first kept character and ends just after the last, so
    characters deleted *inside* the span (a soft hyphen, "-\\n") are covered and
    none outside it are.
    """
    return raw_index[start], raw_index[end - 1] + 1


def _strip_wrapping(quote: str) -> str:
    quote = quote.strip()
    while len(quote) >= 2 and quote[0] in "\"'“‘«" and quote[-1] in "\"'”’»":
        quote = quote[1:-1].strip()
    return quote


_ROLE_UNDERSCORE = re.compile(r"^(\s*)(system|assistant|user|human)_", re.IGNORECASE | re.MULTILINE)


def _variants(quote: str) -> list[str]:
    """The quote as typed, and as the document would have it before prompt fencing.

    The prompt shows passages with `user:` rewritten to `user_` so a document
    cannot fake a turn (`llm._fence`). A model quoting that line quotes the
    rewritten form, which is not in the stored text.
    """
    variants = [quote]
    restored = _ROLE_UNDERSCORE.sub(r"\1\2: ", quote)
    if restored != quote:
        variants.append(restored)
    return variants


MIN_QUOTE_WORDS = 3
MIN_QUOTE_CHARS = 12
MAX_QUOTE_CHARS = 300
MAX_QUOTE_SENTENCES = 2
MAX_QUOTES_PER_PASSAGE = 3


def find_quote(quote: str, text: str) -> tuple[int, int] | None:
    """Raw `(start, end)` of `quote` in `text` if it is verbatim after folding."""
    quote = _strip_wrapping(quote)
    if not quote:
        return None
    haystack, raw_index = normalise(text)
    for variant in _variants(quote):
        needle, _ = normalise(variant)
        needle = needle.strip()
        if len(needle.split()) < MIN_QUOTE_WORDS and len(needle) < MIN_QUOTE_CHARS:
            continue
        at = haystack.find(needle)
        if at >= 0:
            return raw_span(raw_index, at, at + len(needle))
    return None


# -- sentences -------------------------------------------------------------

_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "sec", "secs", "fig", "figs",
    "no", "nos", "vs", "etc", "approx", "inc", "ltd", "co", "corp", "dept", "eg",
    "ie", "cf", "al", "vol", "vols", "pp", "p", "ch", "eq", "ref", "refs", "est",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "gen", "gov", "col", "lt", "capt", "sgt", "rev", "hon", "mt", "ave", "blvd",
}
_DOTTED_TOKEN = re.compile(r"(?:^|[\s(\[])(?:[A-Za-z]\.){1,}[A-Za-z]?$")
_CLOSERS = "\"'”’)]}"
_OPENERS = "\"'“‘([{"


def _is_boundary(text: str, dot: int) -> bool:
    """Whether the `.`/`!`/`?` at `dot` really ends a sentence."""
    n = len(text)
    end = dot + 1
    while end < n and text[end] in _CLOSERS:
        end += 1
    if end >= n:
        return True
    if not text[end].isspace():
        return False  # "3.5", "example.com", "e.g.x"
    after = end
    while after < n and text[after] in " \t":
        after += 1
    if after >= n or text[after] in "\r\n":
        return True
    follower = text[after]
    if text[dot] == ".":
        word = re.search(r"([A-Za-z]+)$", text[:dot])
        token = word.group(1) if word else ""
        if token.lower() in _ABBREVIATIONS:
            return False
        if len(token) == 1 and token.isupper():
            return False  # an initial: "J. Smith"
        if _DOTTED_TOKEN.search(text[: dot + 1]):
            return False  # "e.g.", "i.e.", "U.S."
    return follower.isupper() or follower.isdigit() or follower in _OPENERS


_TERMINAL = ".!?:;"
_LIST_ITEM = re.compile(r"\s*(?:[-*\u2022]\s|\d{1,3}[.)]\s)")


def _continues(text: str, break_at: int, resume: int) -> bool:
    """Whether a single line break at `break_at` is a hard wrap inside a sentence.

    PDF and plain-text extraction break lines at the page width, mid-sentence, so
    treating every newline as a sentence end would cut "the wide-" from
    "reaching rules". Lines of a table, log or code are different: each is a
    unit. A break is a wrap only when the line before it looks like prose that
    simply ran out of room - several words, nothing that ends a sentence or a
    clause, no cell separator, and the next line does not open a list item.
    """
    line_start = text.rfind("\n", 0, break_at) + 1
    before = text[line_start:break_at].rstrip()
    line_end = text.find("\n", resume)
    after = text[resume : line_end if line_end >= 0 else len(text)]
    if not before or not after.strip():
        return False
    if before[-1] in _TERMINAL or before.rstrip(_CLOSERS)[-1:] in tuple(_TERMINAL):
        return False
    if " | " in before or " | " in after:
        return False
    if len(before.split()) < 3 and not before.endswith("-"):
        return False
    return not _LIST_ITEM.match(after)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """`(start, end)` of each sentence-like unit, whitespace trimmed.

    Blank lines always end a unit, and so does a single line break unless it is
    a hard wrap inside prose (`_continues`), so lines of a table, a log or code
    are units of their own. Full stops inside decimals, abbreviations, initials
    and dotted acronyms do not end one.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)

    def push(a: int, b: int) -> None:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            spans.append((a, b))

    i = 0
    while i < n:
        ch = text[i]
        if ch in "\r\n":
            run = i
            while run < n and text[run] in "\r\n \t":
                run += 1
            breaks = text.count("\n", i, run) + (
                1 if "\n" not in text[i:run] and "\r" in text[i:run] else 0
            )
            if breaks >= 2 or not _continues(text, i, run):
                push(start, i)
                start = run
            i = run
            continue
        if ch in ".!?" and _is_boundary(text, i):
            end = i + 1
            while end < n and text[end] in _CLOSERS:
                end += 1
            push(start, end)
            start = end
            i = end
            continue
        i += 1
    push(start, n)
    return spans


# -- the fallback matcher --------------------------------------------------

_STOPWORDS = frozenset(
    "a an the and or but of in on at to for from by with as is are was were be been "
    "being it its this that these those which who whom what when where how not no "
    "do does did has have had will would can could should may might than then so "
    "such there their they them he she his her we our you your i also into over".split()
)
_TOKEN = re.compile(r"\w+", re.UNICODE)

MATCH_THRESHOLD = 0.35
MIN_SHARED_TOKENS = 2
# A single unit this long that is the whole chunk is a blob (a CSV row, a minified
# line), not a sentence: marking it marks everything.
BLOB_CHARS = 240


def _content_tokens(text: str) -> set[str]:
    return {
        t for t in (m.lower() for m in _TOKEN.findall(text))
        if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())
    }


def _f1(claim: set[str], unit: set[str]) -> float:
    if not claim or not unit:
        return 0.0
    shared = len(claim & unit)
    needed = MIN_SHARED_TOKENS if len(claim) > MIN_SHARED_TOKENS else 1
    if shared < needed:
        return 0.0
    precision, recall = shared / len(unit), shared / len(claim)
    return 2 * precision * recall / (precision + recall)


def _units(text: str) -> list[tuple[int, int]]:
    """Candidate spans in a chunk: sentences, or cells when a line is a table row."""
    units = sentence_spans(text)
    if len(units) == 1 and units[0][1] - units[0][0] >= len(text.strip()) and " | " in text:
        cells, at = [], 0
        for cell in text.split(" | "):
            start = text.index(cell, at)
            at = start + len(cell)
            if cell.strip():
                cells.append((start, start + len(cell)))
        return cells or units
    return units


def best_unit(claim: str, text: str) -> tuple[int, int] | None:
    """The sentence of `text` that best matches `claim`, or None if none is good."""
    claim_tokens = _content_tokens(claim)
    units = _units(text)
    if not claim_tokens or not units:
        return None
    best, best_score = None, 0.0
    for start, end in units:
        score = _f1(claim_tokens, _content_tokens(text[start:end]))
        if score > best_score:
            best, best_score = (start, end), score
    if best is None or best_score < MATCH_THRESHOLD:
        return None
    if len(units) == 1 and (best[1] - best[0]) > BLOB_CHARS:
        return None
    if len(units) > 1 and best == (units[0][0], units[-1][1]):
        return None
    return best


_MARKER = re.compile(r"[\[［【]\s*[0-9０-９]{1,3}\s*[\]］】]")


def claim_for(answer: str, n: int) -> str:
    """The one answer sentence that carries citation `[n]`, markers removed."""
    marker = re.search(rf"\[{n}\]", answer)
    if marker is None:
        return ""
    sentences = sentence_spans(answer)
    for index, (start, end) in enumerate(sentences):
        if start <= marker.start() < end:
            clean = _MARKER.sub("", answer[start:end]).strip()
            # "...end of claim. [1]" - the marker opened its own unit; the claim
            # is the sentence before it.
            if not _content_tokens(clean) and index:
                prev_start, prev_end = sentences[index - 1]
                return _MARKER.sub("", answer[prev_start:prev_end]).strip()
            return clean
    return ""


# -- quotes ----------------------------------------------------------------


def fit_quote(quote: str, claim: str) -> str:
    """Trim an over-long quote to the one sentence that best matches the claim.

    The prompt asks for the shortest span that answers; a model that pastes a
    paragraph has not said which part answers, and highlighting a paragraph
    defeats the purpose. Over the limit, one sentence is kept - or the quote is
    left as is, and fails validation if it is still not a verbatim whole.
    """
    quote = _strip_wrapping(quote)
    sentences = sentence_spans(quote)
    if len(quote) <= MAX_QUOTE_CHARS and len(sentences) <= MAX_QUOTE_SENTENCES:
        return quote
    if not sentences:
        return ""
    claim_tokens = _content_tokens(claim)
    scored = sorted(
        ((_f1(claim_tokens, _content_tokens(quote[a:b])), a, b) for a, b in sentences),
        reverse=True,
    )
    _, a, b = scored[0]
    picked = quote[a:b]
    return picked if len(picked) <= MAX_QUOTE_CHARS else ""


def validate_quotes(
    quotes: list[str], chunk_text: str, claim: str
) -> list[tuple[str, int, int]]:
    """Quotes that are verbatim in the chunk: `(quote, start, end)` in raw offsets.

    Overlapping or duplicate spans are merged away; at most
    `MAX_QUOTES_PER_PASSAGE` are kept, in document order.
    """
    found: list[tuple[str, int, int]] = []
    for quote in quotes[:MAX_QUOTES_PER_PASSAGE]:
        fitted = fit_quote(quote, claim)
        if not fitted:
            continue
        span = find_quote(fitted, chunk_text)
        if span is None:
            continue
        if any(span[0] < e and s < span[1] for _, s, e in found):
            continue
        found.append((chunk_text[span[0]:span[1]], span[0], span[1]))
    return sorted(found, key=lambda item: item[1])


def resolve(
    answer: str, n: int, chunk_text: str, model_quotes: list[str]
) -> dict | None:
    """What to highlight in passage `n`'s chunk, and why we believe it.

    Returns `{"quotes": [...], "spans": [(start, end), ...], "source": ...}` with
    spans in code points of `chunk_text`, or None for no highlight.
    """
    claim = claim_for(answer, n)
    valid = validate_quotes(model_quotes, chunk_text, claim)
    if valid:
        return {
            "quotes": [q for q, _, _ in valid],
            "spans": [(s, e) for _, s, e in valid],
            "source": "model",
        }
    if not claim:
        return None
    span = best_unit(claim, chunk_text)
    if span is None:
        return None
    return {"quotes": [chunk_text[span[0]:span[1]]], "spans": [span], "source": "matcher"}
