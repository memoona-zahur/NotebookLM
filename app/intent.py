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

# Punctuation and whitespace carry no meaning for this decision, and a trailing
# full stop should not decide whether a message is a greeting.
_NOISE = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(_NOISE.sub(" ", (text or "").lower()).split())


def classify(question: str) -> str:
    """One of ``"small_talk"``, ``"greeting"`` or ``"question"``.

    `greeting` and `small_talk` are separated because they get different
    replies: a greeting is met with a greeting, and "thanks" is not.
    """
    cleaned = normalize(question)
    if not cleaned:
        return "question"
    if cleaned in GREETINGS:
        return "greeting"
    if cleaned in SMALL_TALK:
        return "small_talk"
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


def reply_for(kind: str) -> str:
    """The text to show for a message that was never a retrieval question."""
    if kind == "greeting":
        return GREETING_REPLY
    if kind == "small_talk":
        return SMALL_TALK_REPLY
    return EMPTY_NOTEBOOK_REPLY