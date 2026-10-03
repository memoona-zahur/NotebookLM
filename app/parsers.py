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


def _parse_pdf(path: Path) -> list[Block]:
    import pymupdf

    doc = pymupdf.open(path)
    blocks: list[Block] = []
    try:
        for number, page in enumerate(doc, start=1):
            content = _clean(page.get_text())
            if content:
                blocks.append(Block(text=content, page=number))
    finally:
        doc.close()
    return blocks


def _parse_docx(path: Path) -> list[Block]:
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
) -> list[Block]:
    """Extract indexable blocks from any supported file."""
    size = chunk_size or default_chunk_size()
    overlap = chunk_overlap if chunk_overlap is not None else default_chunk_overlap()
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _split_long(_parse_pdf(path), size, overlap)
    if suffix == ".docx":
        blocks = _parse_docx(path)
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

    return _split_long(blocks, size, overlap)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph, then sentence, then word, with overlap.

    Table-like text is split on line boundaries instead. Cutting a numeric
    block on a word boundary slices rows in half, which produces fragments like
    "003.52309 11.5 52 0" that match nothing and read as gibberish when cited.
    """
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
        start = end - overlap
    return chunks


def _split_long(blocks: list[Block], size: int, overlap: int) -> list[Block]:
    out: list[Block] = []
    for block in blocks:
        if len(block.text) <= size:
            out.append(block)
            continue
        for piece in chunk_text(block.text, size, overlap):
            out.append(Block(text=piece, page=block.page, heading=block.heading))
    return out
