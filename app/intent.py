"""Deciding what kind of message this is, before retrieval runs.

Every question in this app is answered from documents or not at all. That rule is
the product, and it is also why "Hi" came back as a refusal:

    No relevant passage found - best match 0%, below the 25% floor.

Technically correct, and bad. A greeting is not a failed retrieval, it is not a
question about the corpus at all, and answering it with a relevance score
reports a diagnostic the user did not ask for. Worse, the score was `0.00`
because there was nothing to score against, not because anything was checked and
found wanting.

So the two cases are separated here, before any embedding is computed, and only
one of them is a retrieval question.

There is a third case, and it is the one that looks most like a bug: "who r u",
"what can you do", "how can you help me today". These are questions about the
assistant rather than about the corpus, so retrieval is the wrong tool for them
twice over. In an empty notebook they were told to add a PDF, and in a notebook
with sources they got the relevance floor - "best match 0%, below the 25%
floor" in answer to "who are you". Neither is a failed retrieval; both are a
question the app never tried to answer. They are separated here for the same
reason greetings are, and matched the same way: against the whole message, so
"who is the CEO named in this filing" stays a retrieval question.

The temptation is to detect greetings with a keyword list - "hi", "hello", "hey",
"thanks". That is fragile in a way this file should not be: "thanks for the
refund policy, now what about the 30-day window?" is a real question that starts
with a social word, and a keyword list would silently drop it. So a message only
counts as small talk when it is *entirely* small talk: a greeting and nothing
else, checked with a full-string match after normalisation. Anything longer, or
containing a letter or digit that is not part of the greeting, is treated as a
real question and goes to retrieval, where the floor will judge it honestly.

This costs no model call and no latency, which is the other reason to do it here
rather than by prompting the model to be conversational: a prompt-level
distinction would have to spend a token on every real question to avoid spending
one on a greeting.
"""

import re

# Normalised forms of a greeting, matched against the *whole* message.
GREETINGS = frozenset(
    {
        "hi",
        "hii",
        "hey",
        "hello",
        "hallo",
        "yo",
        "hiya",
        "howdy",
        "heya",
        "sup",
        "hiya there",
        "hello there",
        "hey there",
        "good morning",
        "good evening",
        "good afternoon",
        "good day",
        "how are you",
        "how are you doing",
        "how's it going",
        "hows it going",
        "how are things",
        "whats up",
        "what is up",
        "what's up",
        "nice to meet you",
    }
)

# Gratitude and sign-off on their own. Not a greeting, but the same category:
# there is no corpus question in "thanks", and the floor should not have to
# explain itself.
SMALL_TALK = frozenset(
    {
        "thanks",
        "thank you",
        "thanks a lot",
        "thank you so much",
        "cheers",
        "bye",
        "goodbye",
        "see you",
        "good night",
        "goodnight",
        "ok",
        "okay",
        "cool",
        "nice",
        "great",
        "got it",
        "understood",
        "perfect",
        "lol",
    }
)

# Questions addressed to the assistant itself: identity, capabilities, how to
# use it. Same rule as the two sets above - the *whole* message must be one of
# these, so "who is the CEO named in this filing" and "what can you do with a
# CSV like this" stay retrieval questions. Every form below is about the
# assistant, so none of them can carry a document noun, which is what makes a
# whole-string match safe here where a keyword scan would not be.
ASSISTANT = frozenset(
    {
        "who are you",
        "who r u",
        "who are u",
        "who r you",
        "what are you",
        "what r you",
        "what is you",
        "whats your name",
        "what's your name",
        "what is your name",
        "tell me about yourself",
        "tell me about you",
        "introduce yourself",
        "who am i talking to",
        "what can you do",
        "what can you do for me",
        "what do you do",
        "how can you help me today",
        "how you can help me today",
        "how can you help me",
        "how you can help me",
        "how can you help",
        "how you can help",
        "how can i use you",
        "what can i ask you",
        "what should i ask",
        "how do you work",
        "what is this app",
        "how can i help you",
        "are you human",
        "are you a human",
        "are you a bot",
        "are you ai",
        "are you a robot",
        "are you chatgpt",
        "help me",
        "help",
    }
)

# Variants of the same questions with an optional tail ("... today", "... really").
# Normalisation has already lowercased and stripped punctuation, so this only has
# to allow word order and a trailing qualifier - and it stays anchored at both
# ends, which is the whole point of the pattern.
_ASSISTANT_VARIANTS = re.compile(
    r"(?:who (?:are|r|is) (?:you|u)"
    r"|what (?:are|r|is) (?:you|u)"
    r"|what can (?:you|u) do"
    r"|what can (?:you|u) help(?: me)?(?: with)?"
    r"|how (?:can (?:you|u)|(?:you|u) can) help(?: me)?(?: with)?"
    r"|what do (?:you|u) do"
    r"|tell me about (?:yourself|you)"
    r"|how do (?:you|u) work"
    r"|how can i use (?:you|u))"
    r"(?: today| please| really| anyway| at all)?"
)


# Punctuation and whitespace carry no meaning for this decision, and a trailing
# full stop should not decide whether a message is a greeting.
_NOISE = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(_NOISE.sub(" ", (text or "").lower()).split())


def classify(question: str) -> str:
    """One of ``"small_talk"``, ``"greeting"``, ``"assistant"`` or ``"question"``.

    `greeting` and `small_talk` are separated because they get different
    replies: a greeting is met with a greeting, and "thanks" is not.
    `assistant` is separated from both because it gets an answer about the
    assistant - identity and capabilities - rather than a greeting back.
    """
    cleaned = normalize(question)
    if not cleaned:
        return "question"
    if cleaned in GREETINGS:
        return "greeting"
    if cleaned in SMALL_TALK:
        return "small_talk"
    if cleaned in ASSISTANT or _ASSISTANT_VARIANTS.fullmatch(cleaned):
        return "assistant"
    # The default is a question. Every doubt resolves towards retrieval, because
    # the failure modes are not symmetric: a wrongly-greeted real question gets a
    # useless reply, while a wrongly-queried "hi" costs one embedding and gets
    # today's honest refusal.
    return "question"


# Replies. Deliberately short, and deliberately *not* mentioning the relevance
# floor: a score of 0.00 against an empty corpus is not a finding about the
# user's documents, and leading with it is what made the old reply feel like an
# error message rather than a reply.
GREETING_REPLY = (
    "Hello. Ask me anything about this notebook's sources and I will answer from "
    "the passages, with the exact text cited. Nothing I write here comes from "
    "outside your documents."
)

SMALL_TALK_REPLY = (
    "Any time. Ask me a question about these sources whenever you are ready."
)

EMPTY_NOTEBOOK_REPLY = (
    "This notebook has no sources yet, so there is nothing to answer from. Add a "
    "PDF, DOCX, Markdown or CSV file and I can answer questions about it."
)

# Used when the model is unreachable and the question was about the assistant.
# It has to be true whether or not the notebook has anything in it, because this
# reply does not know: "the sources indexed here" would promise documents an
# empty notebook does not have.
ASSISTANT_REPLY = (
    "I am the research assistant for this notebook. I answer from the documents "
    "indexed here rather than from outside knowledge, and I cite the passage "
    "behind each claim - add a document and ask me about it."
)


def reply_for(kind: str) -> str:
    """The text to show for a message that was never a retrieval question."""
    if kind == "greeting":
        return GREETING_REPLY
    if kind == "small_talk":
        return SMALL_TALK_REPLY
    if kind == "assistant":
        return ASSISTANT_REPLY
    return EMPTY_NOTEBOOK_REPLY