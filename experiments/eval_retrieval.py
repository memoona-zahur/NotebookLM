"""Retrieval evaluation across a broad, format- and topic-diverse corpus.

    .venv/bin/python experiments/eval_retrieval.py

Builds 20 synthetic documents spanning 15 file formats and 19 unrelated
subjects, plus any real PDF already in data/uploads, then reports:

  * per-document relevant recall and trap rejection
  * per-format and per-topic aggregates, so a weakness cannot hide in one type
  * a threshold sweep showing the accuracy/safety trade-off
  * sensitivity of every tuning constant, so none can be quietly overfit to
    one document
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db, parsers  # noqa: E402
from app.store import VectorStore  # noqa: E402
from experiments import corpus  # noqa: E402

CORPUS = Path(__file__).resolve().parent / "corpus"

# One uploaded PDF is included as a real-world document, labelled by its format
# rather than named, so the report stays about formats and not about one file.
REAL_PDF = next(
    (p for p in sorted(config.UPLOAD_DIR.glob("*.pdf"), key=lambda p: -p.stat().st_size)),
    None,
)

QUESTION_SETS = dict(corpus.QUESTIONS)


def _agronomy_questions():
    return {
        "relevant": [
            "How many replicates are used per treatment?",
            "What is the target recovery time after harvest?",
            "Who signs off the randomisation seed?",
            "nitrogen_kg_ha high N",
        ],
        "trap": ["How do I bake bread?", "What is the capital of Peru?"],
    }


QUESTION_SETS["agronomy.docx"] = _agronomy_questions()


def query_store(store, questions, session_id):
    """Run every question against an already-built index.

    Separate from indexing so a config sweep can vary the retrieval settings and
    re-query, instead of replaying cached scores.
    """
    results = []
    for kind in ("relevant", "trap"):
        for question in questions[kind]:
            outcome = store.search_detailed(question, session_id=session_id, top_k=5)
            results.append(
                {
                    "kind": kind,
                    "q": question,
                    "best": outcome.best_score,
                    "answered": bool(outcome.hits),
                    "mode": outcome.mode,
                }
            )
    return results


def build_stores():
    """Index the whole corpus once, keyed by document name."""
    corpus.build(CORPUS)
    stores = {}
    for name, questions in QUESTION_SETS.items():
        path = CORPUS / name
        if not path.exists():
            print(f"  missing corpus file: {name}")
            continue
        store = VectorStore()
        # One session per document: each document is retrieved in isolation, so
        # a hit can only come from the document under test.
        session = db.create_session(f"eval: {name}")
        source = store.add(path, display_name=name, session_id=str(session.id))
        stores[name] = {
            "store": store,
            "session_id": str(session.id),
            "display": name,
            "format": path.suffix.lower().lstrip("."),
            "chunks": source.chunks,
            "numeric": source.numeric,
            "questions": questions,
        }
    if REAL_PDF is not None:
        store = VectorStore()
        session = db.create_session("eval: uploaded PDF")
        source = store.add(REAL_PDF, display_name="uploaded PDF", session_id=str(session.id))
        stores["uploaded PDF"] = {
            "store": store,
            "session_id": str(session.id),
            "display": "uploaded PDF",
            "format": "pdf",
            "chunks": source.chunks,
            "numeric": source.numeric,
            "questions": corpus.REAL_WORLD["uploaded PDF"],
        }
    return stores


def collect(stores):
    rows = []
    for name, info in stores.items():
        rows.append(
            {
                "name": name,
                "display": info["display"],
                "format": info["format"],
                "chunks": info["chunks"],
                "numeric": info["numeric"],
                "results": query_store(info["store"], info["questions"], info["session_id"]),
            }
        )
    return rows


_ENTITY_CACHE = None


def _entity_traps():
    """Entity-overlap traps, indexed once and reused by the sensitivity sweep."""
    global _ENTITY_CACHE
    if _ENTITY_CACHE is not None:
        return _ENTITY_CACHE
    found = []
    for doc, (label, pairs) in corpus.ENTITY_TRAPS.items():
        path = CORPUS / doc
        if not path.exists():
            continue
        store = VectorStore()
        session = db.create_session(f"eval-trap: {doc}")
        store.add(path, display_name=doc, session_id=str(session.id))
        for q in pairs:
            outcome = store.search_detailed(q, session_id=str(session.id), top_k=5)
            found.append(
                {
                    "doc": doc,
                    "topic": label,
                    "q": q,
                    "best": outcome.best_score,
                    "answered": bool(outcome.hits),
                }
            )
    _ENTITY_CACHE = found
    return found


def evaluate_from(stores, verbose=True):
    rows = collect(stores)

    if verbose:
        print("=" * 92)
        print(f"{'document':20}{'format':6}{'chunks':>7}{'num':>5}   per-question verdicts")
        print("=" * 92)
        for row in rows:
            rel = [r for r in row["results"] if r["kind"] == "relevant"]
            trap = [r for r in row["results"] if r["kind"] == "trap"]
            mark = "ok " if all(r["answered"] for r in rel) else "MISS"
            print(
                f"{row['name']:20}{row['format']:6}{row['chunks']:>7}{row['numeric']:>5}"
                f"   rel {sum(r['answered'] for r in rel)}/{len(rel)} {mark}"
                f"  trap {sum(not r['answered'] for r in trap)}/{len(trap)}"
                f" {'ok' if all(not r['answered'] for r in trap) else 'LEAK'}"
            )
            if not all(r["answered"] for r in rel) or not all(
                not r["answered"] for r in trap
            ):
                for r in row["results"]:
                    bad = (r["kind"] == "trap" and r["answered"]) or (
                        r["kind"] == "relevant" and not r["answered"]
                    )
                    if bad:
                        tag = "LEAK" if r["kind"] == "trap" else "MISS"
                        print(f"       {tag} {r['best']:.3f}  {r['q']}")
            print()

    return rows, report(rows, verbose)


def report(rows, verbose=True):
    rel = [(r["name"], r["format"], x) for r in rows for x in r["results"] if x["kind"] == "relevant"]
    trap = [(r["name"], r["format"], x) for r in rows for x in r["results"] if x["kind"] == "trap"]

    rel_ok = sum(1 for _, _, x in rel if x["answered"])
    trap_ok = sum(1 for _, _, x in trap if not x["answered"])

    by_format = defaultdict(lambda: [0, 0, 0, 0])
    for name, fmt, x in rel:
        b = by_format[fmt]
        b[0] += x["answered"]
        b[1] += 1
    for name, fmt, x in trap:
        b = by_format[fmt]
        b[2] += not x["answered"]
        b[3] += 1

    topics = _entity_traps() if verbose else None

    if verbose:
        print("=" * 92)
        print("BY FORMAT")
        print("=" * 92)
        print(f"{'format':10}{'relevant answered':>22}{'traps rejected':>18}")
        for fmt in sorted(by_format):
            hit, tot_rel, rej, tot_trap = by_format[fmt]
            rel_s = f"{hit}/{tot_rel}" + ("" if tot_rel == 0 else f" ({hit / tot_rel:.0%})")
            trap_s = f"{rej}/{tot_trap}" if tot_trap else "-"
            print(f"{fmt:10}{rel_s:>22}{trap_s:>18}")

        if topics:
            print()
            print("=" * 92)
            print("ENTITY-OVERLAP TRAPS  (query shares a real entity with the document)")
            print("=" * 92)
            for t in topics:
                tag = "LEAK" if t["answered"] else "ok  "
                print(f"  {tag} {t['best']:.3f}  [{t['doc']}] {t['q']}")
            caught = sum(1 for t in topics if not t["answered"])
            print(f"  caught {caught}/{len(topics)}")

        print()
        print("=" * 92)
        print("OVERALL")
        print("=" * 92)
        print(f"documents: {len(rows)}   formats: {len(by_format)}")
        print(f"relevant answered: {rel_ok}/{len(rel)} ({rel_ok / len(rel):.0%})")
        print(f"traps rejected:    {trap_ok}/{len(trap)} ({trap_ok / len(trap):.0%})")

    rel_scores = sorted(x["best"] for _, _, x in rel)
    trap_scores = sorted(x["best"] for _, _, x in trap)
    summary = {
        "rel_ok": rel_ok,
        "rel_total": len(rel),
        "trap_ok": trap_ok,
        "trap_total": len(trap),
        "rel_scores": rel_scores,
        "trap_scores": trap_scores,
        "by_format": by_format,
    }
    if not verbose:
        return summary

    print(f"relevant min {rel_scores[0]:.3f}   traps max {trap_scores[-1]:.3f}")

    print()
    print("=" * 92)
    print(f"{'threshold':>10}{'relevant':>12}{'traps':>10}{'errors':>9}")
    print("=" * 92)
    best = None
    for threshold in (0.10, 0.15, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50):
        r = sum(1 for s in rel_scores if s >= threshold)
        t = sum(1 for s in trap_scores if s >= threshold)
        errors = (len(rel) - r) + t
        mark = "  <- MIN_SCORE" if abs(threshold - config.MIN_SCORE) < 1e-9 else ""
        print(f"{threshold:>10.2f}{f'{r}/{len(rel)}':>12}{f'{len(trap) - t}/{len(trap)}':>10}{errors:>9}{mark}")
        if best is None or errors < best[1]:
            best = (threshold, errors)
    print(f"\nlowest-error threshold {best[0]:.2f}; configured {config.MIN_SCORE}")
    return summary


def sensitivity(stores):
    """Vary each constant and re-run every query, reporting real movement.

    A constant that only helps one document is a tuned constant, not a default.
    Flat columns mean the setting is not load-bearing; a column that swings means
    the default is a guess that happens to fit this corpus.
    """
    print()
    print("=" * 92)
    print("CONFIG SENSITIVITY   relevant answered / traps rejected")
    print("=" * 92)

    grid = {
        "DENSE_STRONG": [0.25, 0.30, 0.40, 0.50, 0.60],
        "COVERAGE_MIN": [0.2, 0.34, 0.5, 0.66, 0.8],
        "BM25_RESCUE_MIN": [0.0, 0.06, 0.12, 0.20, 0.30],
        "MIN_RATIO": [0.0, 0.25, 0.5, 0.75, 1.0],
        "NUMERIC_DAMPING": [0.25, 0.5, 0.75, 1.0],
        "NUMERIC_PENALTY_SHARE": [0.0, 0.5, 0.8, 1.0],
    }
    for key, values in grid.items():
        original = getattr(config, key)
        cells = []
        for value in values:
            setattr(config, key, value)
            summary = report(collect(stores), verbose=False)
            cells.append(
                f"{value:<4g} {summary['rel_ok']:>3}/{summary['rel_total']}"
                f" {summary['trap_ok']:>3}/{summary['trap_total']}"
            )
        setattr(config, key, original)
        marker = " *" if original in values else ""
        print(f"{key:24}{str(original):<7}" + "  ".join(cells) + marker)


if __name__ == "__main__":
    db.migrate()
    stores = build_stores()
    evaluate_from(stores)
    sensitivity(stores)
    print()
    print(f"supported formats: {len(parsers.SUPPORTED)}")
