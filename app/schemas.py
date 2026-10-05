"""Request bodies, and nothing else.

Kept apart from the route handlers so that "what can a client send" is one
readable file rather than something you discover by scrolling. Every limit here
is enforced by the type, not by a check inside a handler: Pydantic rejects an
oversized body before the handler body runs, so there is no window where a 10 MB
question string has already been allocated.
"""

from pydantic import BaseModel, Field

# A question longer than this is not a question, and the prompt budget is finite
# anyway. Set on the model so the rejection happens at the edge.
MAX_QUESTION_CHARS = 4000
MAX_INSTRUCTION_CHARS = 2000
MAX_SESSION_NAME_CHARS = 120


class AskRequest(BaseModel):
    """A question plus the session it belongs to.

    There is deliberately no `history` field. The transcript is read from the
    database on the server, so a client cannot inject turns it never asked -
    including a forged `system` turn aimed at the grounding instructions.
    """

    question: str = Field(max_length=MAX_QUESTION_CHARS)


class SummarizeRequest(BaseModel):
    instruction: str = Field(default="", max_length=MAX_INSTRUCTION_CHARS)


class SessionRequest(BaseModel):
    name: str = Field(default="", max_length=MAX_SESSION_NAME_CHARS)