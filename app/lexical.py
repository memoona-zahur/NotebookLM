"""Lexical retrieval, used to complement dense embeddings.

Dense retrieval and this module fail in opposite directions, which is why both
are needed. Measured on the eval corpus: an unrelated question still scores
0.252 against a 47-page physics handbook, because a dense vector finds vague
topical similarity. Meanwhile a terse keyword query scores only 0.174 against a
document that states the answer explicitly, because a sentence embedding handles
terse queries poorly.

Term overlap separates those cases cleanly. An unrelated question shares no
distinctive term with the document; the terse query shares several.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass, field

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if",
    "in", "into", "is", "it", "its", "me", "of", "on", "or", "our", "out", "so",
    "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "up", "was", "we", "were", "what", "when", "where", "which",
    "who", "whom", "why", "will", "with", "would", "you", "your", "about",
    "after", "all", "also", "any", "because", "before", "between", "both",
    "could", "did", "do", "does", "each", "give", "just", "like", "make",
    "many", "more", "most", "much", "must", "no", "not", "only", "other",
    "over", "own", "same", "should", "some", "such", "take", "than", "them",
    "very", "want", "was", "well", "were", "what", "when", "which", "while",
    "will", "with", "would", "you",
}

_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)+|\d+")

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase tokens, plus identifier subtokens.

    Code is a first-class source, so "retry_on_timeout" and "retryOnTimeout"
    both yield the whole identifier and its parts. Matching the whole form
    keeps exact-identifier queries sharp; matching the parts lets "timeout"
    still reach "retry_on_timeout".
    """
    tokens: list[str] = []
    for match in _IDENT_RE.finditer(text):
        word = match.group(0).lower()
        if len(word) > 1 or word.isdigit():
            tokens.append(word)
        for part in _CAMEL_RE.findall(match.group(0)):
            part = part.lower()
            if len(part) > 1 and part != word:
                tokens.append(part)
    for match in _NUM_RE.finditer(text):
        tokens.append(match.group(0).replace(",", ""))
    return tokens


def content_terms(text: str) -> list[str]:
    """Query terms with stopwords removed, for coverage scoring."""
    seen: list[str] = []
    for token in tokenize(text):
        if token in STOPWORDS or len(token) < 2:
            continue
        if token not in seen:
            seen.append(token)
    return seen


@dataclass
class BM25:
    """Standard Okapi BM25 over an in-memory chunk list."""

    k1: float = K1
    b: float = B
    ids: list[str] = field(default_factory=list)
    _tokens: list[list[str]] = field(default_factory=list, repr=False)
    _sets: list[set[str]] = field(default_factory=list, repr=False)
    _freqs: list[Counter] = field(default_factory=list, repr=False)
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _avg_len: float = 0.0

    def fit(self, ids: list[str], texts: list[str]) -> None:
        self.ids = list(ids)
        self._tokens = [tokenize(text) for text in texts]
        self._sets = [set(tokens) for tokens in self._tokens]
        self._freqs = [Counter(tokens) for tokens in self._tokens]
        self._avg_len = (sum(len(t) for t in self._tokens) / len(self._tokens)) if self._tokens else 0.0

        document_count = len(self._tokens)
        frequencies: Counter = Counter()
        for tokens in self._sets:
            frequencies.update(tokens)
        self._idf = {
            term: math.log(1 + (document_count - freq + 0.5) / (freq + 0.5))
            for term, freq in frequencies.items()
        }

    def score(self, query: str) -> list[float]:
        terms = tokenize(query)
        scores = [0.0] * len(self._tokens)
        if not terms or not self._tokens:
            return scores
        for index, tokens in enumerate(self._tokens):
            if not tokens:
                continue
            length = len(tokens)
            total = 0.0
            for term in terms:
                freq = self._freqs[index].get(term)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * length / (self._avg_len or 1)
                )
                total += idf * (freq * (self.k1 + 1)) / denominator
            scores[index] = total
        return scores

    def coverage(self, query: str) -> list[float]:
        """IDF-weighted share of the query's content terms present in a chunk.

        Bounded to 0..1 and independent of document length, so it answers "does
        this chunk actually talk about what was asked" rather than "how long is
        this chunk". A single shared common word scores near zero; three of
        three distinctive terms scores 1.0.
        """
        terms = content_terms(query)
        result = [0.0] * len(self._tokens)
        if not terms:
            return result
        weights = [(term, self._idf.get(term, 0.0)) for term in terms]
        total = sum(weight for _, weight in weights) or 1.0
        for index, tokens in enumerate(self._sets):
            hit = sum(weight for term, weight in weights if term in tokens)
            result[index] = hit / total
        return result

    def top(
        self,
        query: str,
        limit: int,
        coverage_min: float = 0.0,
    ) -> list[tuple[str, float, float]]:
        """Best chunks as (id, bm25_score, coverage), best first."""
        scores = self.score(query)
        covers = self.coverage(query)
        ranked = [
            (self.ids[i], scores[i], covers[i])
            for i in range(len(self.ids))
            if covers[i] >= coverage_min
        ]
        ranked.sort(key=lambda item: -item[1])
        return ranked[:limit]


def fuse(
    dense: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal rank fusion.

    Chosen over a weighted score sum because the two retrievers are not on a
    comparable scale: dense cosine sits near 0.3-0.7 while BM25 is unbounded.
    Any mixing weight would be a tuned constant that breaks when the corpus
    changes. RRF only depends on each retriever's own ordering.
    """
    fused: dict[str, float] = {}
    for ranked in (dense, lexical):
        for rank, (chunk_id, _) in enumerate(ranked):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return fused
