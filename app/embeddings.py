from functools import lru_cache

from sentence_transformers import SentenceTransformer

from . import config


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBED_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = get_model().encode(
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
    return len(get_model().tokenizer(text, truncation=False, verbose=False)["input_ids"])


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
