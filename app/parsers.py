"""Format-aware text extraction.

Every parser returns a list of Block. The point of parsing per format is that
the same bytes mean different things: a CSV row is meaningless as bare numbers
("1234, 56.7") but highly searchable once rendered as
("revenue: 1234, margin: 56.7"). Likewise a code function is unretrievable
without its symbol name, and a heading is what makes a citation in a long
document say something more useful than "page 7".
"""

import ast
import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

CODE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".h",
    ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh",
    ".sql",
}


class UnreadableDocument(ValueError):
    """The file parsed cleanly but contains nothing retrievable.

    Distinct from a malformed file: a rejected PDF is not a broken PDF, it is one
    the app genuinely cannot read. Callers surface the message as-is, because it
    says what to do next ("re-save as text") rather than "could not read file".
    """


def _page_list(pages: list[int]) -> str:
    """'1, 2 and 7' - so the message names the pages, not just their count."""
    if len(pages) == 1:
        return str(pages[0])
    if len(pages) == 2:
        return f"{pages[0]} and {pages[1]}"
    return f"{', '.join(str(p) for p in pages[:-1])} and {pages[-1]}"


def _warn_unreadable_pages(pages: list[int], total: int, reason: str) -> None:
    import logging

    logging.getLogger(__name__).warning(
        "Skipped %d of %d pages that held only images (pages %s): %s",
        len(pages),
        total,
        _page_list(pages),
        reason,
    )


TABLE_SUFFIXES = {".csv", ".tsv"}
STRUCTURED_SUFFIXES = {".json"}
CONFIG_SUFFIXES = {".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".properties"}
LOG_SUFFIXES = {".log"}
PROSE_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}

SUPPORTED = (
    {".pdf", ".docx", ".html", ".htm"}
    | TABLE_SUFFIXES
    | STRUCTURED_SUFFIXES
    | CONFIG_SUFFIXES
    | LOG_SUFFIXES
    | PROSE_SUFFIXES
    | CODE_SUFFIXES
)

def default_chunk_size() -> int:
    from . import config

    return config.CHUNK_SIZE


def default_chunk_overlap() -> int:
    from . import config

    return config.CHUNK_OVERLAP


@dataclass
class IngestStats:
    """What chunking did to one document, reported rather than guessed.

    Filled in by `parse` when a caller passes one. It exists because two
    different ceilings apply to the same chunk - `CHUNK_SIZE` in characters,
    `EMBED_MAX_TOKENS` in wordpieces - and neither number tells you what
    happened, while this one says: how many chunks were produced, how many had
    to be cut again to fit the model's window, and what the worst chunk tokenised
    to. That is the difference between "indexed" and "indexed without silently
    losing the tail of some chunks".
    """

    chunks: int = 0
    numeric: int = 0
    # Extra cuts made purely to satisfy EMBED_MAX_TOKENS. Zero on a document
    # whose chunks were already inside the window.
    fit_splits: int = 0
    # Largest wordpiece count of any chunk after fitting, so a caller can show
    # "341 / 512" against whatever the window is.
    token_max: int = 0
    max_chars: int = 0
    total_chars: int = 0

    @property
    def avg_chars(self) -> int:
        return int(self.total_chars / self.chunks) if self.chunks else 0


def _token_limit() -> int:
    from . import config

    return config.EMBED_MAX_TOKENS


def _count(text: str) -> int:
    """Wordpieces of `text`, using the real tokenizer when it is loadable.

    Imported lazily: `sentence_transformers` is a heavy import and parsing a CSV
    should not be the thing that starts a model download. When the tokenizer is
    unavailable the documented 4-chars-per-token approximation stands in - it is
    coarse, and it is still far better than assuming 900 characters fit.
    """
    try:
        from .embeddings import token_count

        return token_count(text)
    except Exception:  # noqa: BLE001
        return max(1, (len(text) + 3) // 4)


def _fit_to_tokens(text: str, limit: int) -> tuple[list[str], bool]:
    """Cut one piece down until every fragment fits `limit` wordpieces.

    Returns the fragments and whether any cut was needed. The starting budget is
    derived from this text's own characters-per-wordpiece ratio rather than a
    guess, so the normal case is one pass; from there the budget is halved until
    every piece fits, which terminates because `chunk_text` with a budget of 1
    emits one character per chunk. Halving rather than binary search because
    `chunk_text`'s boundary rules are not monotonic in the budget, and a search
    over a non-monotonic predicate can land on a budget that does not fit.
    """
    if limit <= 0 or not text:
        return [text] if text else [], False

    measured = _count(text)
    if measured <= limit:
        return [text], False

    budget = max(1, (len(text) * limit) // measured)
    while budget > 1:
        # A little overlap so a fragment does not end mid-sentence with its
        # continuation starting the next one. Scaled to the budget rather than
        # taken from CHUNK_OVERLAP, which is sized for a 900-char window and
        # would be a third of a fragment this small.
        pieces = chunk_text(text, budget, budget // 8)
        if pieces and all(_count(piece) <= limit for piece in pieces):
            return pieces, True
        budget = max(1, budget // 2)
    return [text], True


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_MD_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MD_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|\d{2}:\d{2}:\d{2})"
)
_DIGIT_CHARS = set("0123456789.,-+eE")
_CELL_BUDGET = 900


@dataclass
class Block:
    """One indexable unit of a source."""

    text: str
    page: int = 0
    heading: str = ""


def digit_ratio(text: str) -> float:
    """Share of non-space characters that are digits or numeric punctuation."""
    body = re.sub(r"\s+", "", text)
    if not body:
        return 0.0
    return sum(1 for char in body if char in _DIGIT_CHARS) / len(body)


def _line_is_numeric(line: str) -> bool:
    body = re.sub(r"\s+", "", line)
    if len(body) < 8:
        return False
    if not any(char.isalpha() for char in body):
        return True
    return digit_ratio(body) >= 0.5


def is_numeric_heavy(text: str) -> bool:
    """True when a block reads as a data table, log stream or metrics dump.

    Decided per line rather than over the whole block. A single prose sentence
    appended to a table should not make the block count as prose, and a real
    paragraph sitting inside an otherwise numeric section should not make it
    count as a table either. Such a block is not garbage in general - it may be
    the entire point of the document - so it is flagged, never dropped.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    numeric_lines = sum(1 for line in lines if _line_is_numeric(line))
    if not numeric_lines:
        return False
    return numeric_lines * 2 >= len(lines)


def _decode(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _clean(text: str) -> str:
    text = text.lstrip("\ufeff").replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pair_columns(pairs: list[tuple[str, str]]) -> str:
    cells = [f"{key.strip()}: {value.strip()}" for key, value in pairs if key.strip()]
    return " | ".join(cells)


def _cell_blocks(pairs: list[tuple[str, str]], heading: str, size: int) -> list[Block]:
    """Render key/value pairs into indexable text, respecting the chunk budget."""
    blocks: list[Block] = []
    current: list[str] = []
    length = 0
    for key, value in pairs:
        if not key.strip():
            continue
        piece = f"{key.strip()}: {value.strip()}"
        if current and length + len(piece) > size:
            blocks.append(Block(text="\n".join(current), heading=heading))
            current, length = [], 0
        current.append(piece)
        length += len(piece) + 1
    if current:
        blocks.append(Block(text="\n".join(current), heading=heading))
    return blocks


# --------------------------------------------------------------------------
# Prose
# --------------------------------------------------------------------------


def _parse_markdown(text: str) -> list[Block]:
    lines = text.splitlines()
    blocks: list[Block] = []
    stack: list[str] = []
    buffer: list[str] = []
    index = 0

    def path() -> str:
        return " > ".join(stack)

    def flush() -> None:
        if buffer:
            body = _clean("\n".join(buffer))
            if body:
                blocks.append(Block(text=body, heading=path()))
            buffer.clear()

    while index < len(lines):
        line = lines[index]
        heading_match = _HEADING_RE.match(line)

        if heading_match:
            flush()
            level = len(heading_match.group(1))
            del stack[level - 1 :]
            stack.append(heading_match.group(2).strip())
            index += 1
            continue

        if _MD_ROW_RE.match(line) and index + 1 < len(lines) and _MD_SEP_RE.match(lines[index + 1]):
            flush()
            header = [cell.strip() for cell in line.strip().strip("|").split("|")]
            index += 2
            while index < len(lines) and _MD_ROW_RE.match(lines[index]):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                pairs = [
                    (header[position] if position < len(header) else f"col{position}", cell)
                    for position, cell in enumerate(cells)
                ]
                rendered = _pair_columns(pairs)
                if rendered:
                    blocks.append(Block(text=rendered, heading=path() or "table"))
                index += 1
            continue

        buffer.append(line)
        index += 1

    flush()
    return blocks


def _unwrap(text: str) -> list[str]:
    """Rejoin hard-wrapped lines, the way a text editor displays them.

    Plain text, reStructuredText and hand-written Markdown are often wrapped at a
    fixed column, which splits phrases mid-sentence: "to avoid red man" /
    "syndrome, which is...". Neither chunk then contains the term, so the phrase
    becomes unsearchable. A line is joined to the next when the current one does
    not end a sentence and the next does not start a new one.
    """
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.rstrip()
        if not out:
            out.append(stripped)
            continue
        previous = out[-1]
        if not previous or not stripped:
            out.append(stripped)
            continue
        ends_sentence = previous[-1] in ".!?:;\"')"
        starts_new = stripped[0].isupper() or stripped[0].isdigit()
        hyphenated = previous.endswith("-") and not previous.endswith("--")
        if hyphenated:
            out[-1] = previous[:-1] + stripped.lstrip()
        elif not ends_sentence and not starts_new:
            out[-1] = f"{previous} {stripped.lstrip()}"
        else:
            out.append(stripped)
    return out


def _parse_text(text: str) -> list[Block]:
    cleaned = _clean(text)
    if not cleaned:
        return []
    first = cleaned.splitlines()[0] if cleaned.splitlines() else ""
    if _HEADING_RE.match(first) or any(
        _HEADING_RE.match(line) for line in cleaned.splitlines()
    ):
        return _parse_markdown(cleaned)
    return [Block(text=line) for line in _unwrap(cleaned) if line.strip()]


# --------------------------------------------------------------------------
# Structured data
# --------------------------------------------------------------------------


def _parse_table(text: str, size: int) -> list[Block]:
    delimiter = ","
    try:
        delimiter = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
    except csv.Error:
        first = text.splitlines()[0] if text.strip() else ""
        if "\t" in first:
            delimiter = "\t"

    rows = [row for row in csv.reader(io.StringIO(text), delimiter=delimiter) if any(c.strip() for c in row)]
    if not rows:
        return []

    header = [cell.strip() or f"col{position}" for position, cell in enumerate(rows[0])]
    heading = f"columns: {', '.join(header[:6])}"
    body = rows[1:] if len(rows) > 1 else []

    pairs: list[tuple[str, str]] = []
    for row in body:
        for position, cell in enumerate(row):
            key = header[position] if position < len(header) else f"col{position}"
            pairs.append((key, cell))
    if not pairs:
        return [Block(text=heading, heading=heading)]
    return _cell_blocks(pairs, heading, size)


def _flatten(node, prefix: str = "$") -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            pairs.extend(_flatten(value, f"{prefix}.{key}"))
    elif isinstance(node, list):
        for position, value in enumerate(node[:200]):
            pairs.extend(_flatten(value, f"{prefix}[{position}]"))
    else:
        pairs.append((prefix, "" if node is None else str(node)))
    return pairs


def _parse_json(text: str, size: int) -> list[Block]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _parse_text(text)

    pairs = _flatten(data)
    if not pairs:
        return []
    roots = sorted({key.split(".")[1].split("[")[0] for key, _ in pairs if key.count(".") >= 1})
    return _cell_blocks(pairs, f"json: {', '.join(roots[:6])}" if roots else "json", size)


def _parse_config(text: str, suffix: str, size: int) -> list[Block]:
    """YAML / TOML / INI rendered as flattened key paths.

    Same reasoning as the JSON parser: "pool_size: 20" is searchable, a blob of
    indented lines is not. Parsed as data rather than as source code.
    """
    data = None
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml

            data = yaml.safe_load(text)
        except Exception:  # noqa: BLE001 - malformed config is still worth indexing
            data = None
    elif suffix == ".toml":
        try:
            try:
                import tomllib
            except ModuleNotFoundError:
                import tomli as tomllib

            data = tomllib.loads(text)
        except Exception:  # noqa: BLE001
            data = None
    else:
        import configparser

        parser = configparser.ConfigParser(strict=False, allow_no_value=True)
        try:
            parser.read_string(text)
            data = {
                section: dict(parser[section]) for section in parser.sections()
            }
        except Exception:  # noqa: BLE001
            data = None

    if isinstance(data, dict) and data:
        return _cell_blocks(_flatten(data), "config", size)
    if isinstance(data, list) and data:
        return _cell_blocks(_flatten(data, "$"), "config", size)
    return _parse_text(text)


def _parse_log(text: str, size: int) -> list[Block]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    blocks: list[Block] = []
    buffer: list[str] = []
    length = 0
    heading = ""
    for line in lines:
        stamp = _TIMESTAMP_RE.match(line)
        if buffer and (length + len(line) > size or (stamp and buffer)):
            blocks.append(Block(text="\n".join(buffer), heading=heading))
            buffer, length = [], 0
        if stamp and not buffer:
            heading = stamp.group(1)
        buffer.append(line.strip())
        length += len(line) + 1
    if buffer:
        blocks.append(Block(text="\n".join(buffer), heading=heading))
    return blocks


# --------------------------------------------------------------------------
# Code
# --------------------------------------------------------------------------

_DEF_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:"
    r"(?:async\s+)?def\s+(?P<fn>\w+)"
    r"|class\s+(?P<cls>\w+)"
    r"|(?:export\s+)?(?:public\s+|private\s+|protected\s+|static\s+)*"
    r"(?:function|func|fn|struct|interface|impl|type|enum|const|let|var)\s+(?P<other>\w+)"
    r")"
)
_ENTRY_RE = re.compile(
    r"^(?:CREATE|INSERT|SELECT|UPDATE|DELETE|ALTER|DROP|WITH)\b", re.IGNORECASE
)


def _symbol_at(lines: list[str], start: int) -> tuple[str, int, int] | None:
    """Name, header end line, and body end line for a top-level definition."""
    line = lines[start]
    if _ENTRY_RE.match(line):
        end = start + 1
        while end < len(lines) and lines[end].strip().endswith((";", ",")):
            end += 1
        return "sql statement", start, end

    match = _DEF_RE.match(line)
    if not match:
        return None

    indent = len(match.group("indent").expandtabs(4))
    if indent:
        return None

    name = match.group("fn") or match.group("cls") or match.group("other") or "definition"
    header_end = start
    if "(" in line and ")" not in line.split("(", 1)[1]:
        while header_end + 1 < len(lines) and ")" not in lines[header_end]:
            header_end += 1
    header_end = min(header_end + 1, len(lines) - 1)

    end = header_end + 1
    base_indent = None
    while end < len(lines):
        current = lines[end]
        if not current.strip():
            end += 1
            continue
        current_indent = len(current) - len(current.lstrip())
        if base_indent is None:
            base_indent = current_indent
        elif current_indent <= base_indent and _DEF_RE.match(current):
            break
        end += 1
    return name, header_end, end


def _parse_python(text: str) -> list[Block]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _parse_code_generic(text)

    lines = text.splitlines()
    blocks: list[Block] = []

    def visit(node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = child.name
                path = f"{prefix}{name}"
                start = child.lineno - 1
                end = getattr(child, "end_lineno", start + 1)
                body = "\n".join(lines[start:end]).rstrip()
                if body:
                    blocks.append(Block(text=f"{path}\n{body}", heading=path))
                visit(child, f"{path}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    if not blocks:
        return _parse_code_generic(text)
    return blocks


def _parse_code_generic(text: str) -> list[Block]:
    lines = text.splitlines()
    blocks: list[Block] = []
    preamble: list[str] = []
    index = 0
    while index < len(lines):
        found = _symbol_at(lines, index)
        if not found:
            preamble.append(lines[index])
            index += 1
            continue
        name, header_end, end = found
        body = "\n".join(lines[index:end]).rstrip()
        if body:
            blocks.append(Block(text=f"{name}\n{body}", heading=name))
        index = max(end, index + 1)

    if preamble and not blocks:
        cleaned = _clean("\n".join(preamble))
        if cleaned:
            return [Block(text=cleaned)]
    if preamble:
        cleaned = _clean("\n".join(preamble))
        if cleaned:
            blocks.insert(0, Block(text=cleaned, heading="(module header)"))
    return blocks


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


def _ocr_page(page, tessdata: str) -> str:
    """Read one image-only page. Empty string when nothing legible is found."""
    from . import config

    textpage = page.get_textpage_ocr(
        language=config.OCR_LANGUAGES or "eng",
        dpi=config.OCR_DPI,
        tessdata=tessdata,
    )
    return _clean(page.get_text("text", textpage=textpage))


def _ocr_ready() -> tuple[bool, str]:
    """Whether OCR can run, and why not when it cannot.

    Separate from the call itself because the failure has two very different
    causes - missing language data, which the user can fix in one command, and
    no Tesseract at all, which they cannot - and the message has to name the
    right one.

    The language file is checked here rather than left to the first page, which
    would otherwise fail one page at a time and report "OCR found no text"
    instead of "OCR is not installed". `pymupdf.get_tessdata` cannot be trusted
    for this: it echoes back whatever folder it is given without checking that
    anything is in it, so a configured-but-empty TESSDATA_DIR looks ready and
    only fails later, inside the library, as a code=3 with a Tesseract usage
    message attached.
    """
    import pymupdf

    from . import ocr

    wanted = _tessdata_languages()
    if not wanted:
        return False, "no OCR language is configured (set OCR_LANGUAGES)"

    folder = _tessdata_dir()
    absent = [code for code in wanted if not (folder / f"{code}.traineddata").is_file()]
    if absent:
        fix = f"run: python -m app.ocr --install {' '.join(absent)}"
        found = ocr.available_languages(folder)
        if found:
            return False, (
                f"OCR language data for {', '.join(absent)} is missing from "
                f"{folder} (available there: {', '.join(found)}); {fix}"
            )
        return False, (
            f"no Tesseract language data for {', '.join(absent)} in {folder}; {fix}"
        )

    try:
        return True, pymupdf.get_tessdata(str(folder))
    except Exception:  # noqa: BLE001 - the library raises bare RuntimeError
        return False, (
            "Tesseract language data is present but PyMuPDF could not load it; "
            "check TESSDATA_PREFIX"
        )


def _tessdata_dir() -> Path:
    from . import config

    return config.TESSDATA_DIR


def _tessdata_languages() -> list[str]:
    from . import config

    return [c for c in config.OCR_LANGUAGES.replace("+", " ").split() if c]


def _merge_by_page(typed: list[Block], recovered: list[Block]) -> list[Block]:
    """Interleave OCR'd pages with text pages in page order.

    OCR runs after the text pass, so appending its results would put page 40 of
    a mixed document before page 2. A chunker that merges neighbouring blocks
    would then join the end of the document to its own beginning, and the
    citation would point at text that reads as nonsense.
    """
    if not recovered:
        return typed
    if not typed:
        return recovered
    merged = {block.page: block for block in recovered}
    for block in typed:
        merged.setdefault(block.page, block)
    return [merged[page] for page in sorted(merged)]


def _pdf_lines(page) -> list[tuple[float, str, bool]]:
    """`(size, text, in_margin)` for every line of a page's text layer.

    PDF has no heading semantics - a heading is only ever bigger text - so font
    size is the only signal there is. The margin flag exists because a running
    header is set at heading size too, and a header repeated on every page would
    otherwise become the document's heading.
    """
    out: list[tuple[float, str, bool]] = []
    try:
        info = page.get_text("dict")
        height = float(page.rect.height) or 1.0
        for block in info.get("blocks", ()):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", ()):
                spans = line.get("spans", ())
                if not spans:
                    continue
                text = "".join(s.get("text", "") for s in spans)
                text = " ".join(text.split())
                if not text:
                    continue
                # The size of the span carrying the most characters: a line
                # typeset with one large word and a small remainder belongs to
                # whichever size actually sets it.
                size = float(
                    max(spans, key=lambda s: len(s.get("text", ""))).get("size", 0.0)
                )
                bbox = line.get("bbox") or (0.0, 0.0, 0.0, 0.0)
                # Tight bands on purpose. A running header usually sits in the
                # top half-inch, but a real section title sits at roughly one
                # inch, and treating the latter as a header would lose headings
                # from almost every report ever written.
                in_margin = bbox[1] < height * 0.07 or bbox[3] > height * 0.92
                out.append((size, text, in_margin))
    except Exception:
        # Optional enrichment. A page whose dict cannot be read still has its
        # text; losing the heading is a smaller failure than losing the page.
        return []
    return out


def _looks_like_heading(text: str) -> bool:
    """Reject body text that happens to be set large.

    Deliberately permissive about case and numbering - `METHODS`, `4.2 Results`
    and `Why this matters` are all headings - and strict about the things body
    text does: full stops, low letter density, and sentence length.
    """
    if not (1 <= len(text) <= 90):
        return False
    words = text.split()
    if not (1 <= len(words) <= 12):
        return False
    if text[-1] in ".,;":
        return False
    letters = sum(1 for c in text if c.isalpha())
    return letters >= 4 and letters / len(text) >= 0.45


def _pdf_headings(
    lines: dict[int, list[tuple[float, str, bool]]], total: int
) -> dict[int, str]:
    """Heading in force on each page, or nothing when none can be found.

    Only the most prominent qualifying line per page is taken: a page's own
    sub-headings do not describe the section the page belongs to. The stack then
    behaves like Markdown's heading path - a larger size opens a section, an
    equal size is a sibling and replaces the current one.

    Returning `{}` when no candidate exists is the point. Guessing is worse than
    the status quo, because a wrong heading reaches the model as context and
    turns into a wrong citation, which is harder to spot than a missing one.
    """
    if not lines:
        return {}

    # Body size is the size carrying the most characters, not the most common
    # size: a document whose cover is set at 48pt would otherwise read every
    # other page as body text and every heading as ordinary text.
    weight: dict[float, int] = {}
    for page_lines in lines.values():
        for size, text, _ in page_lines:
            if size > 0:
                weight[size] = weight.get(size, 0) + len(text)
    if not weight:
        return {}
    body = max(weight, key=lambda s: weight[s])
    if body <= 0:
        return {}

    # A section heading is typeset once, at the start of its section. A running
    # header is typeset on every page it covers. Repetition therefore separates
    # them far more reliably than position does, and it is the rule that stops a
    # document titled "ACME CORP CONFIDENTIAL" from having that on page one as
    # its only heading.
    seen: dict[str, int] = {}
    for page_lines in lines.values():
        for _, text, _ in page_lines:
            seen[text] = seen.get(text, 0) + 1
    repeated = max(2, int(total * 0.3))

    candidates: dict[int, list[tuple[float, str]]] = {}
    for number, page_lines in lines.items():
        # Largest first, so a page's own heading opens the section and the
        # lines below it nest underneath rather than competing with it.
        found = sorted(
            {
                (size, text)
                for size, text, margin in page_lines
                if not margin
                and size >= body * 1.15
                and seen.get(text, 0) < repeated
                and _looks_like_heading(text)
            },
            key=lambda s: -s[0],
        )
        if found:
            candidates[number] = found

    if not candidates:
        return {}

    stack: list[tuple[float, str]] = []
    out: dict[int, str] = {}
    for number in range(1, total + 1):
        for size, text in candidates.get(number, ()):
            # A size at or below the new heading's level belongs to it: equal
            # size is a sibling (replace), smaller size is a descendant (close).
            # Everything larger is an ancestor and stays. Getting this backwards
            # nests sections under their own children.
            while stack and stack[-1][0] <= size:
                stack.pop()
            if not stack or stack[-1][1] != text:
                stack.append((size, text))
        if stack:
            # Pages before the first heading are left out, so they read as ""
            # rather than inheriting a section they predate.
            out[number] = " > ".join(text for _, text in stack)
    return out


def _parse_pdf(path: Path) -> list[Block]:
    import pymupdf

    from . import config

    doc = pymupdf.open(path)
    blocks: list[Block] = []
    # Pages that yielded no text but do hold images. A phone photo of a receipt
    # and a genuinely blank page look identical from the text alone, so the
    # distinction has to be made from the page's images.
    image_pages: list[int] = []
    # Text-layer lines per page, kept so headings can be inferred from font
    # size. Captured during the same pass rather than a second one, because
    # parsing a PDF twice to read it once is the kind of cost nobody notices
    # until a large upload.
    lines: dict[int, list[tuple[float, str, bool]]] = {}
    total_pages = doc.page_count
    try:
        for number, page in enumerate(doc, start=1):
            content = _clean(page.get_text())
            if content:
                blocks.append(Block(text=content, page=number))
                lines[number] = _pdf_lines(page)
            elif page.get_images(full=True):
                image_pages.append(number)
        try:
            headings = _pdf_headings(lines, total_pages)
        except Exception:
            # Never let a heading heuristic block ingestion of a readable file.
            headings = {}
    finally:
        doc.close()

    if image_pages:
        # The whole document is images. That is a scan, and the text is
        # recoverable, so OCR runs whether or not OCR_ENABLED is set - refusing
        # a readable document is worse than spending a second per page on one
        # nobody expected to be a scan.
        scanned_document = not blocks
        reason = ""
        if scanned_document or config.OCR_ENABLED:
            recovered, unresolved, reason = _ocr_image_pages(
                path, image_pages, scanned_document
            )
            blocks = _merge_by_page(blocks, recovered)
            # An OCR'd page has no font sizes, but it is still inside whatever
            # section the surrounding pages established, so it inherits.
            image_pages = unresolved
        elif image_pages:
            reason = "OCR is disabled (OCR_ENABLED=0)"

        # Still nothing. Say what happened and what to do, rather than indexing
        # an empty source: an empty source appears in Sources and the assistant
        # later claims the file is not there, which is a wrong answer with
        # nothing to point at.
        #
        # The reason from the OCR attempt is preferred over the generic status:
        # a document that hit the page cap and one where OCR found nothing both
        # end up here, and they need different sentences.
        if not blocks:
            pages = "page" if len(image_pages) == 1 else "pages"
            verb = "contains" if len(image_pages) == 1 else "contain"
            detail = reason or _ocr_status_for_user()
            raise UnreadableDocument(
                f"No text found on any page, but {_page_list(image_pages)} "
                f"{pages} {verb} only images. This looks like a scanned document "
                f"or a photo, and its text could not be recovered. {detail} "
                f"Otherwise, re-save the file as a text-based PDF or a .txt/.md file."
            )
        # Mixed document: the text pages indexed normally. Recorded so the
        # message can say what was missed rather than implying full coverage.
        # The page count is read before close, since len() on a closed document
        # raises.
        if image_pages:
            _warn_unreadable_pages(
                image_pages, total_pages, reason or _ocr_status_for_user()
            )

    for block in blocks:
        block.heading = headings.get(block.page, "")

    return blocks


def _ocr_image_pages(
    path: Path, image_pages: list[int], whole_document: bool
) -> tuple[list[Block], list[int], str]:
    """OCR the given pages. Returns (blocks, pages still unread, reason)."""
    import logging

    import pymupdf

    from . import config

    log = logging.getLogger(__name__)
    ready, detail = _ocr_ready()
    if not ready:
        reason = f"OCR is unavailable ({detail})"
        if whole_document:
            log.warning("%s", reason)
        return [], list(image_pages), reason

    budget = config.OCR_MAX_PAGES
    if len(image_pages) > budget:
        # Indexing the first 50 pages of a 400-page scan and reporting success is
        # the exact failure mode this whole path exists to remove: the source
        # looks complete and is not, and nothing in the response says so. So the
        # document is refused instead, and the message names the number to raise.
        log.warning(
            "OCR skipped: %d image pages exceeds the %d-page cap",
            len(image_pages),
            budget,
        )
        return [], list(image_pages), (
            f"OCR_MAX_PAGES is {budget} and this document has {len(image_pages)} "
            f"image pages, so none of them were read. Raise OCR_MAX_PAGES or "
            f"split the document, or re-save it as a text-based PDF"
        )

    blocks: list[Block] = []
    unresolved: list[int] = []
    doc = pymupdf.open(path)
    tessdata = detail
    try:
        for number in image_pages:
            try:
                content = _ocr_page(doc[number - 1], tessdata)
            except Exception as exc:  # noqa: BLE001 - one bad page is not fatal
                log.warning("OCR failed on page %d of %s: %s", number, path.name, exc)
                unresolved.append(number)
                continue
            if content:
                blocks.append(Block(text=content, page=number))
            else:
                # A blank or near-blank scan page. Not an error: it genuinely
                # may have no text on it.
                unresolved.append(number)
    finally:
        doc.close()

    reason = ""
    if unresolved:
        pages = "page" if len(unresolved) == 1 else "pages"
        reason = f"OCR found no text on {len(unresolved)} {pages}"
    if blocks:
        log.info(
            "Recovered text from %d scanned page(s) in %s via OCR",
            len(blocks),
            path.name,
        )
    return blocks, unresolved, reason


def _ocr_status_for_user() -> str:
    """The next action, in the user's terms rather than the library's."""
    from .ocr import missing_languages

    absent = missing_languages(_tessdata_dir())
    if absent:
        return (
            f"To enable OCR, run: python -m app.ocr --install {' '.join(absent)}"
        )
    ready, detail = _ocr_ready()
    if ready:
        return (
            "OCR is configured but returned no text; the scan may be blank or "
            "too low-contrast"
        )
    return f"OCR is unavailable ({detail}); run: python -m app.ocr --install"


def _parse_docx(path: Path, size: int) -> list[Block]:
    from docx import Document

    doc = Document(path)
    blocks: list[Block] = []
    stack: list[str] = []
    buffer: list[str] = []

    def heading() -> str:
        return " > ".join(stack)

    def flush() -> None:
        if buffer:
            body = _clean("\n".join(buffer))
            if body:
                blocks.append(Block(text=body, heading=heading()))
            buffer.clear()

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name or "").lower()
        if style.startswith("heading"):
            flush()
            level = int("".join(ch for ch in style if ch.isdigit()) or 1)
            del stack[max(0, level - 1) :]
            stack.append(text)
            continue
        buffer.append(paragraph.text)
    flush()

    for table in doc.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        rows = [row for row in rows if any(row)]
        if not rows:
            continue
        header = [cell or f"col{position}" for position, cell in enumerate(rows[0])]
        pairs: list[tuple[str, str]] = []
        for row in rows[1:]:
            for position, cell in enumerate(row):
                pairs.append((header[position] if position < len(header) else f"col{position}", cell))
        if not pairs:
            continue
        blocks.extend(_cell_blocks(pairs, "table", size))
    return blocks


def _parse_html(path: Path) -> list[Block]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_decode(path), "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    blocks: list[Block] = []
    buffer: list[str] = []
    stack: list[str] = []

    def heading() -> str:
        return " > ".join(stack)

    def flush() -> None:
        if buffer:
            body = _clean("\n".join(buffer))
            if body:
                blocks.append(Block(text=body, heading=heading()))
            buffer.clear()

    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "td", "th"]):
        name = element.name
        if name.startswith("h") and len(name) == 2 and name[1].isdigit():
            flush()
            level = int(name[1])
            del stack[level - 1 :]
            stack.append(element.get_text(" ", strip=True))
            continue
        text = element.get_text(" ", strip=True)
        if text:
            buffer.append(text)
    flush()
    return blocks or ([Block(text=_clean(soup.get_text("\n")))] if soup.get_text(strip=True) else [])


def parse(
    path: Path,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    stats: IngestStats | None = None,
) -> list[Block]:
    """Extract indexable blocks from any supported file.

    `stats`, when given, is filled with what chunking actually did. Callers that
    only need the blocks pass nothing and pay no extra tokenization.
    """
    size = chunk_size or default_chunk_size()
    overlap = chunk_overlap if chunk_overlap is not None else default_chunk_overlap()
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _split_long(_parse_pdf(path), size, overlap, stats)
    if suffix == ".docx":
        blocks = _parse_docx(path, size)
    elif suffix in {".html", ".htm"}:
        blocks = _parse_html(path)
    elif suffix in TABLE_SUFFIXES:
        blocks = _parse_table(_decode(path), size)
    elif suffix in STRUCTURED_SUFFIXES:
        blocks = _parse_json(_decode(path), size)
    elif suffix in CONFIG_SUFFIXES:
        blocks = _parse_config(_decode(path), suffix, size)
    elif suffix in LOG_SUFFIXES:
        blocks = _parse_log(_decode(path), size)
    elif suffix in CODE_SUFFIXES:
        blocks = _parse_python(_decode(path)) if suffix == ".py" else _parse_code_generic(_decode(path))
    elif suffix in PROSE_SUFFIXES:
        blocks = _parse_text(_decode(path))
    else:
        raise ValueError(f"Unsupported file type: {suffix or path.name}")

    return _split_long(blocks, size, overlap, stats)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph, then sentence, then word, with overlap.

    Table-like text is split on line boundaries instead. Cutting a numeric
    block on a word boundary slices rows in half, which produces fragments like
    "003.52309 11.5 52 0" that match nothing and read as gibberish when cited.
    """
    # Both values come from config, so they are clamped here rather than trusted.
    # Overlap at or above size walks `start` backwards below and the loop never
    # ends: a user who sets CHUNK_SIZE without also lowering CHUNK_OVERLAP would
    # get a server that hangs on the next upload. Capped at half the window
    # rather than size-1, because overlap larger than half re-reads more than it
    # advances and produces one chunk per character. The default (150 of 900) is
    # far below this. Size has to be at least 1 for the same reason.
    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size // 2))

    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    if is_numeric_heavy(text):
        lines = text.splitlines()
        buffer: list[str] = []
        length = 0
        for line in lines:
            if not line.strip():
                continue
            if buffer and length + len(line) > size:
                chunks.append("\n".join(buffer))
                buffer, length = [], 0
            buffer.append(line)
            length += len(line) + 1
        if buffer:
            chunks.append("\n".join(buffer))
        return chunks

    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            split = max(window.rfind("\n\n"), window.rfind(". "), window.rfind(" "))
            if split > size * 0.5:
                end = start + split + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        # Monotonic, not merely "minus overlap". A boundary choice can move
        # `end` only slightly past `start`, and subtracting an overlap larger
        # than that step walks the cursor *backwards* - so the loop never
        # terminates and appends forever. Taking the max keeps overlap where it
        # is useful and guarantees progress where it is not.
        start = max(start + 1, end - overlap)
    return chunks


def _split_long(
    blocks: list[Block],
    size: int,
    overlap: int,
    stats: IngestStats | None = None,
) -> list[Block]:
    """Apply both ceilings - characters, then wordpieces - to every block.

    Order matters. A block is cut to `CHUNK_SIZE` characters first, which is the
    cheap check and the one that keeps chunks readable; only then is each
    fragment checked against the model's wordpiece window, because the token
    count of a block is not a simple function of its length (a table of part
    numbers and the same number of English words tokenize very differently).
    """
    limit = _token_limit()
    out: list[Block] = []
    for block in blocks:
        # The common case: inside both ceilings. Keeps the original Block
        # (page, heading and all) rather than rebuilding an equivalent one.
        if len(block.text) <= size:
            tokens = _count(block.text) if limit > 0 else 0
            if limit <= 0 or tokens <= limit:
                out.append(block)
                if stats is not None:
                    _tally(stats, block.text, tokens)
                continue

        pieces = (
            [block.text]
            if len(block.text) <= size
            else chunk_text(block.text, size, overlap)
        )
        for piece in pieces:
            fitted, split = _fit_to_tokens(piece, limit)
            if stats is not None and split:
                stats.fit_splits += len(fitted) - 1
            for fragment in fitted:
                if not fragment.strip():
                    continue
                out.append(Block(text=fragment, page=block.page, heading=block.heading))
                if stats is not None:
                    _tally(stats, fragment, _count(fragment) if limit > 0 else 0)
    return out


def _tally(stats: IngestStats, text: str, tokens: int) -> None:
    stats.chunks += 1
    stats.total_chars += len(text)
    stats.max_chars = max(stats.max_chars, len(text))
    if is_numeric_heavy(text):
        stats.numeric += 1
    stats.token_max = max(stats.token_max, tokens)


