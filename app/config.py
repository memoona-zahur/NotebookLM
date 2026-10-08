import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
# Vite writes the compiled bundle to static/assets; the HTML references it by
# absolute path, so the directory has to exist before the app serves anything.
ASSET_DIR = STATIC_DIR / "assets"
DATA_DIR = BASE_DIR / "data"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


_load_dotenv(BASE_DIR / ".env")

# Uploads default next to the app but honour UPLOAD_DIR, so the container can
# point them at the mounted volume instead of its own filesystem layer.
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR") or (DATA_DIR / "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Ceiling on a single upload. Without it a large file is read into memory whole
# before anything can object, and one 2 GB PDF could take the process down.
# Generous for prose, low enough that "I uploaded a huge scan" is answered rather
# than fatal.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "100")) * 1024 * 1024

# OCR for scanned documents. A scan is a real document that happens to be an
# image, so refusing it is the wrong answer where the text can be recovered.
# PyMuPDF embeds Tesseract, so this needs no external binary - only the language
# data below.
#
# On by default, because a scan that gets refused is a document the user cannot
# search. The cost is roughly a second per page, which is fine for a receipt and
# wrong to impose on a 400-page upload nobody expected to be a scan - hence
# OCR_MAX_PAGES below. Set to 0 to keep only the text pages of a mixed document.
#
# A document that is entirely images gets OCR regardless of this flag; the
# alternative is refusing it outright.
OCR_ENABLED = os.getenv("OCR_ENABLED", "1") not in {"0", "false", "no"}

# Folder holding <lang>.traineddata files. PyMuPDF looks here, then at
# TESSDATA_PREFIX, then at a Tesseract install. Downloaded into data/tessdata by
# `python -m app.ocr --install`.
TESSDATA_DIR = Path(os.getenv("TESSDATA_DIR") or (DATA_DIR / "tessdata"))

# Tesseract language codes. "eng" is right for most documents; add e.g. "deu"
# for German or "fra" for French. The files must exist in TESSDATA_DIR.
OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "eng").strip()

# Resolution the page is rasterised at before recognition. 300 is the usual
# "good enough to be accurate" figure for printed text and is what Tesseract's
# own docs recommend; higher mostly costs time.
OCR_DPI = int(os.getenv("OCR_DPI", "300"))

# Pages per document OCR will touch. A cap rather than a limit on quality: a
# thousand-page scan costs real time and the app has no progress bar to show it.
# A document over the cap is refused rather than partly indexed, because a
# partial index that reports success is the failure this replaced.
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "50"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384
# The embedding model's own window, in its own wordpieces. Sentence-transformers
# truncates anything longer, silently: a 900-character chunk that tokenises to
# 379 pieces loses 123 of them before the vector is computed, and the store has
# no idea it happened. CHUNK_SIZE is a character ceiling and stays one - it is
# the knob the docs describe - but a character ceiling is not a token ceiling,
# and only the second one the model enforces. Chunks are split to fit this at
# ingest so the vector is of the whole chunk. Set 0 to disable the check.
EMBED_MAX_TOKENS = int(os.getenv("EMBED_MAX_TOKENS", "256"))

# Second-stage re-ranking. The dense/BM25/RRF pass above is cheap and recall
# oriented; a cross-encoder reads the query and the chunk together and is far
# more accurate, and far more expensive, which is why it runs on the shortlist
# rather than the corpus. Off by default until it is measured - a re-ranker that
# moves metrics in the wrong direction is worse than no re-ranker.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") not in {"0", "false", "no"}
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# How many fused candidates are handed to the cross-encoder. Above TOP_K so the
# re-ranker has something to choose between, well below "everything", because it
# scores one pair at a time.
RERANK_POOL = int(os.getenv("RERANK_POOL", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))

# How many latency samples /api/status reports percentiles over. A ring buffer
# rather than an accumulation: P95 over the process's whole life answers "was
# this slow once", P95 over the last N answers "is it slow now".
LATENCY_WINDOW = int(os.getenv("LATENCY_WINDOW", "200"))

# Postgres holds everything persistent: sources, chunk text, chunk vectors
# (pgvector), sessions and chat history. The compose default points at the
# `db` service, so the app reaches the database by service name.
DEFAULT_DATABASE_URL = os.getenv(
    "DEFAULT_DATABASE_URL",
    "postgresql://notebooklm:notebooklm@localhost:5432/notebooklm",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))

# A chunk scoring below MIN_SCORE is treated as "not in the sources". Measured
# on all-MiniLM-L6-v2: relevant chunks land around 0.30-0.45, irrelevant ones
# around 0.05-0.15, so 0.25 separates them without rejecting correct answers.
# Raise it to make the assistant more cautious, lower it to make it chattier.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))

# Kept separately so code can tell "the configured default" from "a caller
# deliberately raised the floor for this query".
DEFAULT_MIN_SCORE = MIN_SCORE

# Cap on chunks taken from any one source, so a single long document cannot
# crowd out the rest of the notebook. Set >= TOP_K to disable.
MAX_PER_SOURCE = int(os.getenv("MAX_PER_SOURCE", "3"))

# Retrieval also requires MIN_RATIO of the best hit. This is a second, per-query
# floor: it prunes the weak tail that a single strong match drags along, without
# ever accepting something the absolute MIN_SCORE rejected. Set to 0 to disable.
MIN_RATIO = float(os.getenv("MIN_RATIO", "0.5"))

# A bare table of numbers ("412300 388100 521700") is semantically flat, so
# MiniLM scores it near the top for almost any query. Rather than discarding
# such chunks at ingest - which destroys CSVs, JSON, logs and metric dumps
# outright - they are flagged and their score is damped during retrieval.
# Damping is skipped when numeric_heavy chunks already make up
# NUMERIC_PENALTY_SHARE of the candidates, so a fully numeric corpus keeps
# working: there the penalty is uniform and relative ranking is unchanged.
NUMERIC_DAMPING = float(os.getenv("NUMERIC_DAMPING", "0.5"))
NUMERIC_PENALTY_SHARE = float(os.getenv("NUMERIC_PENALTY_SHARE", "0.8"))

# Hybrid retrieval. Dense and lexical ranking are fused with reciprocal rank
# fusion, but the relevance decision still uses the dense score, because that is
# the only scale MIN_SCORE was ever calibrated against.
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "1") not in {"0", "false", "no"}
RRF_K = int(os.getenv("RRF_K", "60"))

# A weak dense match with no term overlap at all is nearly always a false
# positive: the eval corpus scored an unrelated question at 0.252 against a
# physics handbook purely on vague topical similarity. Requiring either real
# lexical support or a clearly strong dense score removes that class of leak
# without touching confident answers.
DENSE_STRONG = float(os.getenv("DENSE_STRONG", "0.40"))

# Conversely, terse keyword queries score poorly on embeddings even when the
# chunk plainly answers them (measured 0.174). A chunk is
# admitted on lexical evidence alone when enough of the query's distinctive
# terms are present, and the dense score is only a sanity floor.
COVERAGE_MIN = float(os.getenv("COVERAGE_MIN", "0.5"))
BM25_RESCUE_MIN = float(os.getenv("BM25_RESCUE_MIN", "0.12"))

# Prior conversation turns included for follow-up questions. Keep the window
# short and deterministic; when the session grows, older turns are compacted into
# a short summary instead of being sent verbatim.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))
HISTORY_SUMMARY_CHARS = int(os.getenv("HISTORY_SUMMARY_CHARS", "600"))

# Public origin the browser uses to reach this API, e.g.
# http://localhost:8000. Left empty it is derived from the request, which works
# behind a reverse proxy. Set it explicitly when the app is mounted somewhere
# the browser cannot infer.
BASE_URL = os.getenv("BASE_URL", "").rstrip("/")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

# -- Web search ------------------------------------------------------------
#
# Off unless GROQ_API_KEY is present, because it is the one feature here that
# sends text off the machine. Everything else in this app keeps the documents
# local and only sends the question and the retrieved passages to the
# configured provider; this sends the *search query* to a third-party search
# engine and fetches whatever it finds. That is a different promise, so it is
# opt-out (WEB_SEARCH=0) rather than opt-in, and the UI states it.
#
# The search model is separate from the generation model on purpose. Finding
# pages is not a reasoning task, and the browser tool injects whole pages into
# the prompt: one search measured ~72,500 input tokens, which is roughly three
# orders of magnitude more than answering a question from this notebook. So the
# cheap model does the looking, and the configured model still does the
# answering.
WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH", "1") not in {"0", "false", "no"}
WEB_SEARCH_MODEL = os.getenv("WEB_SEARCH_MODEL", "openai/gpt-oss-20b")

# Pages ingested per search. Each one is a real source in the notebook, so this
# bounds both the cost of the search and how much the corpus grows per click.
WEB_SEARCH_MAX_PAGES = int(os.getenv("WEB_SEARCH_MAX_PAGES", "5"))

# Wall-clock ceilings. Search is a reasoning model with a browser tool, so it
# is genuinely slow - 8-15s is normal - and a fetch that has not answered by
# this point is not going to.
WEB_SEARCH_TIMEOUT = int(os.getenv("WEB_SEARCH_TIMEOUT", "60"))
WEB_FETCH_TIMEOUT = int(os.getenv("WEB_FETCH_TIMEOUT", "20"))

# Ceiling on one fetched page, applied while reading rather than after, so an
# endless response body cannot be buffered whole before anything objects.
WEB_FETCH_MAX_BYTES = int(os.getenv("WEB_FETCH_MAX_MB", "8")) * 1024 * 1024

# Redirect hops followed. Zero would be safest but breaks any site that moves
# http:// to https://, and every hop is re-checked against the same address
# rules as the first one.
WEB_FETCH_MAX_REDIRECTS = int(os.getenv("WEB_FETCH_MAX_REDIRECTS", "3"))


def web_search_available() -> tuple[bool, str]:
    """Whether web search can run, and the sentence explaining it when it cannot.

    Returned as a pair rather than a bare bool because the UI has to say *why*
    the card is disabled, and a greyed-out control with no reason is the thing
    this project keeps trying not to ship. Depends on Groq specifically: the
    browser tool is a Groq built-in, so an OpenAI or Ollama configuration has no
    way to search even when a key is set.
    """
    if not WEB_SEARCH_ENABLED:
        return False, "Disabled with WEB_SEARCH=0."
    if resolved_provider() != "groq":
        return False, (
            "Web search needs the Groq provider, but LLM_PROVIDER is "
            f"{resolved_provider()!r}. Browser search is a Groq built-in tool."
        )
    if not GROQ_API_KEY:
        return False, "GROQ_API_KEY is not set."
    return True, ""


def resolved_provider() -> str:
    if LLM_PROVIDER:
        return LLM_PROVIDER
    if GROQ_API_KEY:
        return "groq"
    if OPENAI_API_KEY:
        return "openai"
    return "ollama"


def resolved_model() -> str:
    return {
        "groq": GROQ_MODEL,
        "openai": OPENAI_MODEL,
        "ollama": OLLAMA_MODEL,
    }.get(resolved_provider(), "unknown")


# Money, and how much of it is a guess.
#
# Rates are read at request time and always travel with their as-of date,
# because a price that is not dated is a price that has silently gone stale.
# `None` from `model_price`/`estimate_cost_usd` means "no rate on file", which
# is a different statement from 0.0 ("measured free") - reporting the first as
# the second is how an unpriced model ends up looking free in a total.
MODEL_PRICE_AS_OF = "2026-10-07"
MODEL_PRICE_SOURCE = (
    "Groq Cloud docs (console.groq.com/docs/models) and OpenAI pricing, "
    "checked 2026-10-07"
)
# Not included in any total: Groq charges for the browser_search tool by the
# request as well as by the token, and that fee is deliberately left out rather
# than guessed at - see the note the API returns beside the totals.
MODEL_PRICE_NOTES = [
    "Estimate at published list prices; web search tool fees are not included.",
]

MODEL_PRICE_TABLE: dict[str, dict] = {
    # Groq, self-serve, USD per 1M tokens.
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
    # OpenAI.
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    # Ollama runs these on this machine. There is no metered rate, so they are
    # priced at zero and flagged `local` - zero because it is measured free,
    # not because the rate is unknown.
    "llama3.1": {"input": 0.0, "output": 0.0, "local": True},
    "llama3.2": {"input": 0.0, "output": 0.0, "local": True},
}


def model_price(model: str) -> dict | None:
    """USD per 1M tokens for `model`, or None when no rate is on file.

    Matched exactly - the whole name first, then the part after the last slash -
    rather than by substring. Substring matching is order-dependent in a way a
    price table must not be: `gpt-4o` is a substring of `gpt-4o-mini`, so
    whichever appeared first in the table would decide the price for both.
    """
    name = (model or "").strip().lower()
    if not name:
        return None
    if name in MODEL_PRICE_TABLE:
        return MODEL_PRICE_TABLE[name]
    return MODEL_PRICE_TABLE.get(name.rsplit("/", 1)[-1])


def estimate_cost_usd(
    model: str, prompt_tokens: int = 0, completion_tokens: int = 0
) -> float | None:
    """What the call would have cost at the rate on file, or None if unknown.

    None is load-bearing. Callers must show it as "no rate on file" rather than
    folding it into a total as zero.
    """
    price = model_price(model)
    if price is None:
        return None
    input_cost = prompt_tokens * price["input"] / 1_000_000.0
    output_cost = completion_tokens * price["output"] / 1_000_000.0
    return round(input_cost + output_cost, 6)


def price_provenance() -> dict:
    """The as-of date and source that the totals computed from them must carry."""
    return {
        "currency": "USD",
        "as_of": MODEL_PRICE_AS_OF,
        "source": MODEL_PRICE_SOURCE,
        "notes": list(MODEL_PRICE_NOTES),
    }
