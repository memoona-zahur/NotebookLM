import threading

from sentence_transformers import SentenceTransformer

from . import config

# Indexing now runs in worker threads while questions are being answered, so two
# things that were safe on one thread need a guard:
#
# - loading the model. `lru_cache` does not stop two threads that miss together
#   from both loading it (twice the memory, a slow first request each);
# - using it. The fast tokenizer inside the model is shared state, and calling it
#   from two threads at once is a known source of "Already borrowed" errors;
#   `token_count` (used while chunking) and `encode` (used to embed) both reach
#   it. Inference is CPU-bound, so serialising it costs little.
_load_lock = threading.Lock()
_use_lock = threading.RLock()
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    with _use_lock:
        vectors = model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]


def token_count(text: str) -> int:
    """Wordpieces of `text` under the embedding model's own tokenizer.

    The real count, special tokens included, because that is what the model
    actually sees and what it silently truncates at. An approximation cannot do
    this job: on the eval corpus the 4-chars-per-token rule estimates 225 pieces
    for a chunk that tokenises to 379, so it would report "fits" for a chunk
    that loses a third of itself.

    Counting an over-long input is exactly what this function is for, and
    transformers' "longer than the maximum sequence length" warning would be
    logging the problem instead of fixing it. `verbose=False` turns that
    warning off, which is safe here and only here: no other caller of this
    tokenizer wants an over-long input.
    """
    model = get_model()
    with _use_lock:
        return len(model.tokenizer(text, truncation=False, verbose=False)["input_ids"])


def score_against_query(query_vector: list[float], texts: list[str]) -> list[float]:
    """Cosine similarity of already-normalised query against new texts.

    Needed when a candidate is surfaced by lexical search but sits outside the
    dense candidate pool. Its dense score is still required, because the
    relevance decision is made on the dense scale.
    """
    if not texts:
        return []
    vectors = embed_texts(texts)
    return [float(sum(a * b for a, b in zip(query_vector, vector))) for vector in vectors]
