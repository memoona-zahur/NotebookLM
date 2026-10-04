/**
 * Recognises "highlight <word>" in the chat composer.
 *
 * This is a shortcut to the highlight box in the Sources drawer, not a second
 * way of answering. Anything it does not recognise is passed to the model
 * untouched, so the only thing to get right here is being conservative: a
 * genuine question that gets swallowed and silently searched for instead of
 * answered is a much worse failure than a command that has to be typed into the
 * box instead.
 *
 * Hence the strict shapes below. "Where is the capital of France" is not a
 * highlight request, and "highlight the difference between the two papers" is
 * not a request to find the phrase "difference between the two papers". Both
 * fall through to the model.
 */

/** Politeness and assistant-speak that carry no intent of their own. */
const LEAD_IN = /^(?:please\s+|hey\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|i\s+want\s+you\s+to\s+|i'?d\s+like\s+you\s+to\s+)+/i;

/** Trailing "in my documents", "in the sources", and full stops. */
const TAIL = /[\s.!?]+$/;

const TAIL_SCOPE = /\s+(?:in|from|across|throughout|within)\s+(?:the|my|our|these|this|your)\s+[\w\s]+$/i;

/**
 * "all of", "every mention of", "the word", "the phrase" — the words people put
 * between the verb and the word they actually mean.
 *
 * Every part is optional because each of these arrives in the wild on its own:
 * "all velocity" is as reasonable as "every mention of velocity". What is not
 * optional is that something has to be stripped, which is why callers that
 * cannot afford a false positive insist on seeing filler first.
 */
const FILLER = new RegExp(
  "^(?:" +
    "(?:all|every|each)\\s+" +
      "(?:the\\s+)?" +
      "(?:occurrences?|instances?|mentions?|appearances?|uses?|hits?|times?|references?)?\\s*" +
      "(?:of\\s+)?" +
    "|" +
    "(?:the\\s+)?(?:word|term|phrase|text|string)\\s+" +
  ")",
  "i",
);

/** Quoting, so `"escape velocity"` and `'escape velocity'` work. */
const QUOTED = /^["'“”‘’](.+?)["'“”‘’]$/;

/**
 * A term has to look like something a person would search a document for:
 * letters, digits and single spaces, short enough to be a word or a phrase.
 */
const TERM = /^[A-Za-z0-9][A-Za-z0-9 '\-_]*$/;

/** Longest term accepted, matching the server's own ceiling. */
const MAX_TERM = 60;

/** A term longer than this is a sentence that slipped through, not a phrase. */
const MAX_WORDS = 4;

/**
 * Ordered most specific first: "find all occurrences of" must be tried before
 * anything that would otherwise swallow the same message.
 *
 * All case-insensitive, because people do not retype the verb in lower case.
 */
const SHAPES = [
  // "find all occurrences of X". The "of" is required, so "find the section on
  // X" cannot be mistaken for a request to search for that section.
  /^(?:find|locate|search\s+for|look\s+up)\s+(?:all|every|each)\s+(?:the\s+)?(?:occurrences?|instances?|mentions?|appearances?|uses?|hits?|times?|references?)?\s*of\s+(.+)$/i,
  // "where does X appear", "where do X occur"
  /^(?:show\s+me\s+)?where\s+(?:does|do|did|is|are|was|were)\s+(.+?)\s+(?:appear|appears|occur|occurs|show\s+up|shows\s+up|used|used\s+up|mentioned|found|appearing|occurring)$/i,
  // "where X appears", "show me where X shows up"
  /^(?:show\s+me\s+)?where\s+(.+?)\s+(?:appears|occurs|shows\s+up|turns\s+up|comes\s+up)$/i,
  // The plain ones.
  /^(?:highlight|mark|circle|underline|emphasise|emphasize|show)\s+(.+)$/i,
];

/** Not intent, just a mention of the feature inside a real question. */
const REFUSALS = [
  /\bdifference\b/i,
  /\bbetween\b/i,
  /\bwhy\b/i,
  /\bhow\b/i,
  /\bwhat\b/i,
  /\bcompare\b/i,
  /\bexplain\b/i,
];

function plausible(term) {
  const trimmed = term.trim().replace(QUOTED, "$1").trim();
  if (!trimmed || trimmed.length > MAX_TERM) return null;
  if (!TERM.test(trimmed)) return null;
  // At least one letter, or it is just a number standing in for something.
  if (!/[A-Za-z]/.test(trimmed)) return null;
  if (trimmed.split(/\s+/).length > MAX_WORDS) return null;
  if (REFUSALS.some((re) => re.test(trimmed))) return null;
  return trimmed;
}

/**
 * The word the user asked to see, or null if this message is a real question.
 *
 * Returns the user's own casing: the search is case-insensitive, but echoing
 * back "Velocity" when they typed "Velocity" reads better than normalising it.
 */
export function highlightTerm(message) {
  if (typeof message !== "string") return null;

  let text = message.trim().replace(TAIL, "");
  if (!text) return null;

  text = text.replace(LEAD_IN, "").replace(TAIL, "");
  text = text.replace(TAIL_SCOPE, "").replace(TAIL, "");
  if (!text) return null;

  for (const shape of SHAPES) {
    const found = text.match(shape);
    if (!found) continue;

    let remainder = found[1].replace(FILLER, "").replace(TAIL, "");
    // "all occurrences of" already consumed the of; a leading "of" can survive.
    remainder = remainder.replace(/^of\s+/i, "").trim();

    // A quoted term is taken whole: "highlight the word 'escape velocity'" must
    // not be cut down to "velocity".
    const quoted = remainder.match(QUOTED);
    if (quoted) return plausible(quoted[1]);

    const term = plausible(remainder);
    // Keep looking: a later shape may fit where this one produced something
    // implausible. A refusal means the user asked a question, so stop.
    if (term) return term;
    if (REFUSALS.some((re) => re.test(remainder))) return null;
  }

  return null;
}