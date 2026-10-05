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
 * The word a single, standalone highlight command asks for, or null.
 *
 * This is the original parser and the rules that keep it conservative: it only
 * accepts a message that is *entirely* a highlight request, so "what is escape
 * velocity" and "highlight the difference between the two papers" both fall
 * through to the model. `readIntent` below layers the two-instruction case on
 * top of this, and deliberately reuses these rules rather than loosening them.
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

/**
 * Split a message into the part that asks a question and the part that names
 * words to highlight.
 *
 * Two instructions in one message are common - "what is the late penalty?
 * highlight 30 days" - and before this existed the whole string went to the
 * model, so retrieval was searching for the words "highlight 30 days" as well
 * as the question. The instruction text dilutes the dense query and adds noise
 * to BM25, which is the one thing the retrieval work cannot recover from.
 *
 * Returns `{question, term}`:
 *   - both a question and a term, when the message asks and highlights
 *   - `question: null`, for a bare highlight command: nothing to answer, so the
 *     model is not called at all
 *   - `term: null`, for a question with no highlight in it, which is unchanged
 *     behaviour and by far the common case
 *
 * Split on punctuation and connectives rather than an LLM call. A model asked
 * to do this adds latency to every question and can still be talked out of it;
 * the shapes people actually type are closed. Anything ambiguous is treated as
 * a plain question, because a question silently downgraded to a word search is
 * far worse than a highlight that has to be typed into the box.
 */
export function readIntent(message) {
  if (typeof message !== "string") return { question: null, term: null };

  const whole = message.trim();
  if (!whole) return { question: null, term: null };

  // The whole message being a highlight command: no question to ask.
  const bare = highlightTerm(whole);
  if (bare) return { question: null, term: bare };

  const found = trailingHighlight(whole);
  if (!found) return { question: whole, term: null };

  const question = whole
    .slice(0, found.start)
    .replace(/[\s,;:.!?-]+$/, "")
    .trim();
  // Nothing usable left to ask. The message is a lookup with a preamble, not a
  // request for an answer, so do not call the model on the preamble.
  if (!question || question.split(/\s+/).length < 2) {
    return { question: null, term: whole };
  }
  return { question, term: found.term };
}

/**
 * A highlight instruction in the last clause of a longer message, or null.
 *
 * Two rules keep this from eating real questions, and both are load-bearing:
 *
 *  1. The verb must open a clause - after a sentence break, a comma, or a
 *     conjunction. Otherwise "what does the contract say about highlighting the
 *     deadline" becomes a search for "the deadline", which is a wrong answer
 *     rather than a missing feature.
 *  2. The verb must be the imperative form. "how do I highlight a word" is a
 *     question about the feature; only "highlight a word" is a command. The
 *     giveaway is the subject before it: a pronoun or nothing at all is a
 *     command, "I" or "we" is a person asking how.
 *
 * Anything else returns null and the message goes to the model whole.
 */
function trailingHighlight(message) {
  const CLAUSE =
    /(?:^(?<lead>)|(?<=[.!?;])\s+|\s*(?<punct>[,:;])\s*|\s+(?<conj>and|then|also|plus)\s+)(?<clause>(?:please\s+|can\s+you\s+|could\s+you\s+)?(?:highlight|mark|circle|underline|emphasise|emphasize)\b[^]*)$/i;

  const match = message.match(CLAUSE);
  if (!match || !match.groups) return null;

  const clause = match.groups.clause;

  // A subject in front of the verb, with nothing separating them, means the
  // sentence is *about* the command rather than issuing it: "how do I highlight a
  // word". With a separator it is a new instruction and the text before it is a
  // question: "what is the fee, highlight 30 days".
  //
  // Only a separator-free clause is checked, and only the words immediately in
  // front of the verb, so "what is the fee? highlight 30 days" still splits.
  const separated = Boolean(match.groups.punct || match.groups.conj);
  if (!separated) {
    const before = message.slice(0, match.index);
    const subject = before.split(/\s+/).slice(-3).join(" ");
    if (/\b(?:i|we|you|how|why|what|when|where|which|who)\b/i.test(subject)) return null;
  }

  // Reuse the strict single-message parser on the clause, so a trailing
  // instruction is accepted under exactly the same rules as a standalone one.
  const term = highlightTerm(clause);
  if (!term) return null;

  return { term, start: match.index };
}