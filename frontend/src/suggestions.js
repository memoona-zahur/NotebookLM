/**
 * Starter questions for the empty state.
 *
 * The reference product opens a notebook with tappable questions instead of a
 * paragraph, so this replaces a block of explanatory prose that was more honest
 * about grounding and read as a terms-of-service notice on first load.
 *
 * These are derived from source *names* plus each source's numeric-density flag.
 * That is a deliberate limit and worth being explicit about: the reference
 * product reads the notebook to write its suggestions, which means a model call
 * on every load, costing money and latency before the user has typed anything.
 * Names and the flag are already in memory from `/api/status`, so this costs
 * nothing and cannot fail. If the suggestions turn out to be too generic to be
 * worth having, the fix is an endpoint that samples a few chunks and asks for
 * questions - not a longer list of hardcoded strings.
 */

/** Spreadsheet kinds, where a figures question is apt whatever the chunk stats say. */
const TABLE_KINDS = new Set(["csv", "xls", "xlsx", "tsv"]);

/** Extensions and separators that make a filename a bad question subject. */
const NOISE = /\.(pdf|docx?|txt|md|csv|pptx?|xlsx?|html?|epub)$/i;

/**
 * A human-readable subject from a filename: "q3-returns-policy_v2.pdf" becomes
 * "returns policy". Deliberately lossy - it only has to read well in a
 * sentence, and the retriever resolves the real topic from the query.
 */
export function subjectFromName(name) {
  if (!name) return "";
  return name
    .replace(NOISE, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Questions to offer for a notebook, best-first.
 *
 * `sources` is the `/api/status` source list. Returns an empty array when
 * there is nothing to ask about yet, because suggesting questions for an empty
 * notebook produces four refusals - the app refuses when retrieval finds
 * nothing, and that is the first impression we would be giving.
 */
export function suggestionsFor(sources) {
  if (!sources || !sources.length) return [];

  const names = sources.map((s) => subjectFromName(s.name)).filter(Boolean);
  // A question about a specific document beats a generic one when the names are
  // short enough to stay readable; very long filenames make for absurd prose.
  const usable = names.filter((n) => n.length > 2 && n.length <= 48);
  const subject = usable[0] || "";
  // Two signals, because neither alone is right.
  //
  // `numeric` is the count of numeric-dense chunks, which catches logs and bare
  // data dumps but almost never a spreadsheet: the CSV parser reshapes a table
  // into `Label: Value` lines, and "Revenue: 12000" is 5 digits in 13
  // characters, well under the 0.5 density `parsers.is_numeric_heavy` asks for.
  // A 40-row revenue table and a one-paragraph memo both report 0.
  //
  // The source kind catches what the flag misses. A spreadsheet is figures by
  // construction, so the question is worth asking there without measuring.
  //
  // Prose that merely mentions "30 days" triggers neither, which is the intended
  // conservatism: a numbers question that cannot be answered teaches the user
  // that this app invents figures.
  const hasNumbers =
    sources.some((s) => (s.numeric || 0) > 0) ||
    sources.some((s) => TABLE_KINDS.has(String(s.kind || "").toLowerCase()));

  const out = [];
  if (subject) {
    out.push(`What are the key points about ${subject}?`);
    out.push(`Summarise ${subject} in a few sentences`);
  } else {
    // No usable filename, so nothing can be named without producing a
    // nonsensical question ("the key points about r2?"). Fall back to questions
    // that are generic by wording rather than by accident, and that still do the
    // job - a summary of the notebook is the one thing always worth offering.
    out.push("Summarise these sources in a few sentences");
    out.push("What are the key points across these sources?");
  }
  // Only offered when the corpus actually contains figures. A numbers question
  // against a prose-only corpus teaches the user that this app invents figures.
  if (hasNumbers) {
    out.push("What figures and dates does this mention?");
  }
  if (usable.length > 1) {
    out.push(`How does ${usable[1]} relate to ${usable[0]}?`);
  }
  out.push("What questions should I be asking about this?");
  return out;
}