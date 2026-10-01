"""Measure whether a cross-encoder reranker separates relevant from trap queries
better than MiniLM cosine alone. Run from the project root with .venv/bin/python.

Not part of the app - this is the experiment that decides whether reranking
belongs in the retrieval pipeline.
"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, ".")

from sentence_transformers import CrossEncoder

from app.store import store

RELEVANT = [
    "What is the Kepler telescope?",
    "How are time stamps generated in Kepler data?",
    "What historical events impacted the data?",
    "What is in the Data Release Notes Supplement?",
    "What causes the data to be less precise?",
    "Summarise the key facts in this document.",
    "What are the acronyms used in the handbook?",
]
TRAPS = [
    "What is the boiling point of water in Fahrenheit?",
    "Who won the 1998 FIFA World Cup?",
    "How do I bake sourdough bread with rye flour?",
    "What is the capital of France?",
    "Write me a Python function to reverse a string.",
    "What is the population of Tokyo?",
    "How do I change a flat car tyre?",
]

OVERFETCH = 20


def newest_real_pdf() -> Path:
    for path in sorted(glob.glob("data/uploads/*.pdf"), key=lambda p: -Path(p).stat().st_mtime):
        if b"MuPDF" not in Path(path).read_bytes()[:80]:
            return Path(path)
    raise SystemExit("no user pdf found in data/uploads")


def main() -> None:
    source = store.add(newest_real_pdf())
    print(f"indexed {source.name}: {source.pages} pages, {source.chunks} chunks\n")

    reranker = CrossEncoder("BAAI/bge-reranker-base")

    rows = []
    for label, questions in (("RELEVANT", RELEVANT), ("TRAP", TRAPS)):
        for question in questions:
            # min_score=0 so nothing is filtered out: we want the raw top-20.
            candidates = store.search_detailed(question, top_k=OVERFETCH, min_score=0.0).hits
            if not candidates:
                continue
            pairs = [(question, hit["text"]) for hit in candidates]
            scores = reranker.predict(pairs, batch_size=16)
            order = sorted(range(len(candidates)), key=lambda i: -scores[i])
            best = order[0]
            rows.append(
                {
                    "label": label,
                    "q": question,
                    "dense": candidates[0]["score"],
                    "dense_top": candidates[0]["text"][:55].replace("\n", " "),
                    "rr": float(scores[best]),
                    "rr_top": candidates[best]["text"][:55].replace("\n", " "),
                }
            )

    hdr = f"{'query':<46} {'dense':>7} {'rerank':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label in ("RELEVANT", "TRAP"):
        print(f"[{label}]")
        for r in [x for x in rows if x["label"] == label]:
            print(f"  {r['q'][:44]:<44} {r['dense']:>7.3f} {r['rr']:>8.3f}")
        print()

    rel = [r for r in rows if r["label"] == "RELEVANT"]
    trap = [r for r in rows if r["label"] == "TRAP"]

    rel_dense = min(r["dense"] for r in rel)
    trap_dense = max(r["dense"] for r in trap)
    rel_rr = min(r["rr"] for r in rel)
    trap_rr = max(r["rr"] for r in trap)

    def verdict(lo_rel, hi_trap, name):
        if lo_rel > hi_trap:
            print(f"  {name:<10} SEPARABLE   relevant >= {lo_rel:.3f} > traps <= {hi_trap:.3f}"
                  f"   (margin {lo_rel - hi_trap:+.3f})")
        else:
            print(f"  {name:<10} OVERLAPPING relevant >= {lo_rel:.3f}, traps <= {hi_trap:.3f}"
                  f"   (overlap {hi_trap - lo_rel:+.3f})")

    print("=" * 72)
    print("SEPARATION BETWEEN RELEVANT AND TRAP QUERIES")
    verdict(rel_dense, trap_dense, "dense:")
    verdict(rel_rr, trap_rr, "rerank:")

    print()
    print("The boiling-point query is the one that leaked in production:")
    for r in rows:
        if "boiling" in r["q"]:
            print(f"  dense  top-1 = {r['dense']:.3f}  -> "
                  f"{'PASSES 0.25 floor (leak)' if r['dense'] >= 0.25 else 'blocked'}")
            print(f"  rerank top-1 = {r['rr']:>7.3f}")
            print(f"  best reranked passage: {r['rr_top']}...")

    print()
    print("Full text of the top reranked passage for the boiling-point query:")
    for r in rows:
        if "boiling" in r["q"]:
            print(f"  {r['rr_top']}")


if __name__ == "__main__":
    main()
