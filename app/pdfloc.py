"""Where on a PDF page a quote sits, as rectangles to highlight.

Rectangles come from the page's own characters (`rawdict`), not from a text
search, so the match is made on the same folded text `citing` validated the
quote against and lands on exactly the characters that were quoted - a
ligature, a soft hyphen or a line-break hyphen in the page does not move it.

The chunk's position is used for one thing only: choosing between several
occurrences of the same phrase on a page, by relative position. A quote that
occurs once is located without consulting it, because the stored offsets and
the page's character stream are folded differently and trusting their absolute
values would turn a correct unique match into a wrong one.

An OCR'd page has no text layer, so there is nothing to find a rectangle on:
the result is None and the caller shows the parsed text instead.
"""

from pathlib import Path

from .citing import normalise, raw_span

Rect = list[float]


def _page_stream(page) -> tuple[list[str], list[Rect | None], list[int]]:
    """The page's characters in reading order, with a box and a line id each.

    Line and block boundaries are inserted as "\\n" with no box, so folding sees
    the same whitespace and hyphen-at-line-break shapes it sees in the text the
    chunk was cut from.
    """
    chars: list[str] = []
    boxes: list[Rect | None] = []
    lines: list[int] = []
    line_id = 0
    for block in page.get_text("rawdict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_id += 1
            for span in line.get("spans", []):
                for char in span.get("chars", []):
                    chars.append(char["c"])
                    boxes.append(list(char["bbox"]))
                    lines.append(line_id)
            chars.append("\n")
            boxes.append(None)
            lines.append(line_id)
    return chars, boxes, lines


def _occurrences(haystack: str, needle: str) -> list[int]:
    found, at = [], haystack.find(needle)
    while at >= 0:
        found.append(at)
        at = haystack.find(needle, at + 1)
    return found


def rects_for_quotes(
    path: Path, page_number: int, quotes: list[tuple[str, float | None]]
) -> dict | None:
    """Rectangles for each quote on one page, or None if the page cannot say.

    `quotes` are `(text, relative_position)`; the position is where the chunk
    that carried the quote starts on the page, as a share of the page (0..1), or
    None when unknown. Returns `{"page", "page_width", "page_height", "rects"}`
    with one rectangle per line of each located quote, in page units (points).
    A quote that cannot be found on the page contributes nothing; if none can
    be, the result is None.
    """
    import pymupdf

    try:
        doc = pymupdf.open(path)
    except Exception:  # noqa: BLE001 - an unreadable file simply has no location
        return None
    try:
        if not 1 <= page_number <= doc.page_count:
            return None
        page = doc[page_number - 1]
        chars, boxes, lines = _page_stream(page)
        if not any(box for box in boxes):
            return None
        folded, raw_index = normalise("".join(chars))
        rects: list[Rect] = []
        for quote, relative in quotes:
            needle, _ = normalise(quote)
            needle = needle.strip()
            if not needle:
                continue
            hits = _occurrences(folded, needle)
            if not hits:
                continue
            if len(hits) == 1 or relative is None:
                start = hits[0]
            else:
                start = min(hits, key=lambda at: abs(at / max(1, len(folded)) - relative))
            first, last = raw_span(raw_index, start, start + len(needle))
            rects.extend(_line_rects(boxes[first:last], lines[first:last]))
        if not rects:
            return None
        return {
            "page": page_number,
            "page_width": round(page.rect.width, 2),
            "page_height": round(page.rect.height, 2),
            "rects": rects,
        }
    finally:
        doc.close()


def _line_rects(boxes: list[Rect | None], lines: list[int]) -> list[Rect]:
    """Merge character boxes into one rectangle per text line."""
    merged: dict[int, Rect] = {}
    for box, line in zip(boxes, lines):
        if box is None:
            continue
        if line in merged:
            r = merged[line]
            r[0], r[1] = min(r[0], box[0]), min(r[1], box[1])
            r[2], r[3] = max(r[2], box[2]), max(r[3], box[3])
        else:
            merged[line] = list(box)
    return [[round(v, 2) for v in rect] for _, rect in sorted(merged.items())]
