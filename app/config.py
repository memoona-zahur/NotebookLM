import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"

for directory in (DATA_DIR, UPLOAD_DIR):
    directory.mkdir(parents=True, exist_ok=True)


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

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384
COLLECTION = "notebook"

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

# Prior conversation turns included for follow-up questions.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "6"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


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
