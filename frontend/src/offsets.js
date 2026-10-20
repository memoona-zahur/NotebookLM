/**
 * Offsets between Python and the browser.
 *
 * Python counts code points; a JavaScript string is indexed in UTF-16 code
 * units, so one emoji before a span shifts every later offset by one. The server
 * sends citation highlights in UTF-16 (`highlight_unit: "utf16"`). A citation
 * persisted before that was recorded has code-point offsets and no unit, and is
 * converted here rather than shown misaligned.
 */

/** UTF-16 index in `text` of the `cp`-th code point. Clamped to the string. */
export function cpToUtf16(text, cp) {
  let units = 0;
  let seen = 0;
  for (const ch of text) {
    if (seen >= cp) break;
    units += ch.length;
    seen += 1;
  }
  return units;
}

function valid(span) {
  return span && Number.isInteger(span.start) && Number.isInteger(span.end);
}

/**
 * The spans of `citation.text` that answered, as non-overlapping UTF-16 ranges
 * inside the string, in order. Anything out of range is dropped: a highlight
 * that disagrees with the text is worse than none.
 */
export function highlightsOf(citation) {
  const text = typeof citation?.text === "string" ? citation.text : "";
  const raw = Array.isArray(citation?.highlights)
    ? citation.highlights
    : citation?.highlight
      ? [citation.highlight]
      : [];
  const inUnits = citation?.highlight_unit === "utf16";
  const spans = raw
    .filter(valid)
    .map((s) =>
      inUnits ? { start: s.start, end: s.end } : { start: cpToUtf16(text, s.start), end: cpToUtf16(text, s.end) },
    )
    .map((s) => ({ start: Math.max(0, s.start), end: Math.min(text.length, s.end) }))
    .filter((s) => s.start < s.end)
    .sort((a, b) => a.start - b.start);
  return spans.filter((s, i) => i === 0 || s.start >= spans[i - 1].end);
}
