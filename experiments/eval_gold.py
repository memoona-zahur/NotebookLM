"""Score the human-verified gold set against the live retriever.

    .venv/bin/python -m experiments.eval_gold
    .venv/bin/python -m experiments.eval_gold --mode shared --k 8
    .venv/bin/python -m experiments.eval_gold --allow-unverified   # dry run only

Writes experiments/results/gold_metrics.json, and refuses to write it at all
until a human has verified the gold items it scores.

Why this is not eval_retrieval.py again: that script asks whether *a* document
got answered, so every hit counts as success and only the score floor
discriminates. It cannot say how much of the right evidence was retrieved, or
in what order. This one has gold positions and required facts, so it can
report recall, precision, hit rate, nDCG and MRG at a stated cut-off, plus
whether the facts needed to answer were literally present in the context the
retriever handed back.

Two indexing modes, because they answer different questions:

  isolated  one session per document. Every chunk belongs to the document
            under test, so a wrong chunk is unambiguously wrong and the
            numbers compare chunkers without cross-document noise.
  shared    all documents in one session. This is the real product condition,
            and the only mode in which an unanswerable question means
            something: a trap has to defeat the entire corpus at once.

Answerable questions are scored in both. Traps are only meaningful in the
shared session, so they are reported there as a leak rate rather than fed to
the ranking metrics, which raise on an empty gold set by design.

Nothing unverified is ever averaged in. An unlabelled question is excluded and
counted, because scoring it 0.0 would be indistinguishable from a genuine
retrieval failure and would hide the labelling gap instead of reporting it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db  # noqa: E402
from app.store import VectorStore  # noqa: E402
from experiments import corpus, metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
GOLD_PATH = HERE / "gold" / "gold_set.json"
RESULTS_PATH = HERE / "results" / "gold_metrics.json"
CORPUS = HERE / "corpus"

REQUIRED_KEYS = ("id", "question", "source", "answerable", "required_facts", "verified_by_human")


def load_gold(path: Path = GOLD_PATH) -> list[dict]:
    """Read the gold items, refusing anything the scorer would have to guess at."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["items"]
    if not items:
        raise ValueError(f"{path} has no items")
    for item in items:
        missing = [key for key in REQUIRED_KEYS if key not in item]
        if missing:
            raise ValueError(f"{item.get('id', '?')} is missing {missing}")
        if item["answerable"] and not item["required_facts"]:
            raise ValueError(f"{item['id']} claims to be answerable but names no required fact")
    return items


def partition(items: list[dict], include_unverified: bool = False) -> tuple[list[dict], list[tuple[str, str]]]:
    """Split into what may be scored and what may not, with a reason each.

    include_unverified exists so the harness can be exercised end to end
    before anyone has read the labels. It makes the run a dry run: the results
    file is withheld and the report is stamped as unverified, so a dry run can
    never be mistaken for a result.
    """
    scorable: list[dict] = []
    excluded: list[tuple[str, str]] = []
    for item in items:
        verified = bool(item["verified_by_human"])
        if item["answerable"]:
            if verified or include_unverified:
                scorable.append(item)
            else:
                excluded.append((item["id"], "awaiting human verification"))
        elif item["required_facts"]:
            excluded.append((item["id"], "unanswerable but names facts"))
        elif verified:
            excluded.append((item["id"], "trap, checked separately"))
        else:
            excluded.append((item["id"], "unverified trap, checked separately"))
    return scorable, excluded


def index_corpus(store: VectorStore, names: list[str], tag: str) -> dict[str, dict]:
    """Ingest each document into its own session, keyed by position.

    store.add keeps the parser block index as chunks.position, so a gold
    position resolves to a chunk id with a lookup and no schema change.
    """
    sessions: dict[str, dict] = {}
    for name in names:
        path = CORPUS / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is referenced by the gold set but missing")
        session = db.create_session(f"eval-gold:{tag}: {name}")
        source = store.add(path, display_name=name, session_id=str(session.id))
        with db.connection() as conn:
            rows = conn.execute(
                "SELECT position, id::text FROM chunks WHERE source_id = %s", (source.id,)
            ).fetchall()
        sessions[name] = {
            "session_id": str(session.id),
            "by_position": {int(position): chunk_id for position, chunk_id in rows},
            "format": path.suffix.lower().lstrip("."),
        }
    return sessions


def index_shared(store: VectorStore, names: list[str], tag: str) -> str:
    """One session holding every document, for the cross-document condition."""
    session = db.create_session(f"eval-gold:{tag}: all documents")
    for name in names:
        path = CORPUS / name
        store.add(path, display_name=name, session_id=str(session.id))
    return str(session.id)


def _context(hits: list[dict]) -> str:
    return "\n".join(hit["text"] for hit in hits)


def score_item(store: VectorStore, item: dict, session_id: str, positions: dict[int, str], k: int) -> dict:
    """Rank metrics and literal fact coverage for one answerable question."""
    expected = [positions[p] for p in item["expected_positions"] if p in positions]
    labelled = len(expected) == len(item["expected_positions"])
    outcome = store.search_detailed(item["question"], session_id=session_id, top_k=k)
    retrieved = [hit["chunk_id"] for hit in outcome.hits]
    if not expected:
        # A gold position with no stored chunk means the document parsed into
        # fewer blocks at index time than at label time. Scoring it 0.0 would
        # report a chunker change as a retrieval failure.
        return {
            "id": item["id"],
            "question": item["question"],
            "source": item["source"],
            "metrics": None,
            "facts": None,
            "note": "gold positions absent from the index",
        }
    facts = metrics.fact_coverage(item["required_facts"], _context(outcome.hits))
    return {
        "id": item["id"],
        "question": item["question"],
        "source": item["source"],
        "retrieved": retrieved,
        "metrics": metrics.score_query(expected, retrieved, k),
        "facts": facts,
        "all_positions_stored": labelled,
        "best_score": outcome.best_score,
    }


def leak_rate(store: VectorStore, traps: list[dict], session_id: str, k: int) -> list[dict]:
    """A trap must retrieve nothing at all; anything returned is a leak."""
    out = []
    for item in traps:
        outcome = store.search_detailed(item["question"], session_id=session_id, top_k=k)
        out.append(
            {
                "id": item["id"],
                "question": item["question"],
                "hits": len(outcome.hits),
                "best_score": outcome.best_score,
                "leaked": bool(outcome.hits),
            }
        )
    return out


def _means(rows: list[dict]) -> dict[str, object]:
    summary = metrics.aggregate(rows)
    fact_rows = [r for r in rows if r.get("facts")]
    covered = sum(r["facts"]["covered"] for r in fact_rows)
    total = sum(r["facts"]["total"] for r in fact_rows)
    summary["fact_coverage"] = covered / total if total else 0.0
    summary["facts_covered"] = covered
    summary["facts_total"] = total
    return summary


def run(k: int, mode: str, allow_unverified: bool, verbose: bool = True) -> dict:
    items = load_gold()
    scorable, excluded = partition(items, include_unverified=allow_unverified)
    traps = [
        item
        for item in items
        if item["verified_by_human"] and not item["answerable"] and not item["required_facts"]
    ]
    unverified = [item_id for item_id, reason in excluded if reason == "awaiting human verification"]

    print("=" * 92)
    print("GOLD SET ACCOUNTING")
    print("=" * 92)
    print(f"items in file:            {len(items)}")
    print(f"answerable and verified:  {len(scorable)}")
    print(f"traps, verified:          {len(traps)}")
    print(f"excluded, unverified:     {len(unverified)}")
    for item_id, reason in excluded:
        if reason != "awaiting human verification":
            print(f"  excluded {item_id}: {reason}")

    if not scorable:
        print()
        print("No verified answerable items, so there is nothing to score.")
        print("That is the guard working, not a failure: an unlabelled question")
        print("scored 0.0 would look identical to a retrieval failure.")
        return {"scored": 0, "skipped": len(excluded), "means": {}, "rows": [], "leaks": []}

    if unverified:
        pending = [item_id for item_id, reason in excluded if reason == "awaiting human verification"]
        print()
        print(f"{len(pending)} items are still unverified and will be excluded:")
        print("  " + ", ".join(pending))
        if allow_unverified:
            print("DRY RUN: scoring unverified labels. These numbers are not evidence.")
        else:
            print("Pass --allow-unverified to see a dry run; it writes no results file.")

    corpus.build(CORPUS)
    store = VectorStore()
    names = sorted({item["source"] for item in scorable} | {item["source"] for item in traps})

    rows: list[dict] = []
    leaks: list[dict] = []
    if mode in ("isolated", "both"):
        print()
        print("=" * 92)
        print("ISOLATED  one session per document")
        print("=" * 92)
        sessions = index_corpus(store, names, "isolated")
        for item in scorable:
            info = sessions[item["source"]]
            rows.append(score_item(store, item, info["session_id"], info["by_position"], k))
    if mode in ("shared", "both"):
        shared = index_shared(store, names, "shared")
        if mode == "shared":
            # Ranking quality under cross-document competition. Gold positions
            # still resolve, because the shared index stores every document.
            shared_positions = _shared_positions(store, names, shared)
            print()
            print("=" * 92)
            print("SHARED  all documents in one session")
            print("=" * 92)
            for item in scorable:
                rows.append(score_item(store, item, shared, shared_positions[item["source"]], k))
        print()
        print("=" * 92)
        print(f"TRAPS  {len(traps)} unanswerable questions against all {len(names)} documents")
        print("=" * 92)
        leaks = leak_rate(store, traps, shared, k)

    summary = _means(rows)
    if verbose:
        _print_detail(rows, leaks, k)

    scored_verified = all(bool(item["verified_by_human"]) for item in scorable)
    report = {
        "k": k,
        "mode": mode,
        "corpus_documents": len(names),
        # Asked whether the labels actually scored here, not whether some item
        # was left out. Checking the excluded list instead let a dry run
        # through, because a dry run excludes nothing.
        "human_verified": scored_verified,
        "accounting": {
            "items": len(items),
            "answerable_verified": len(scorable),
            "traps_verified": len(traps),
            "excluded_unverified": len(unverified),
        },
        "scored": summary["scored"],
        "skipped": summary["skipped"],
        "means": summary["means"],
        "fact_coverage": summary["fact_coverage"],
        "facts": {"covered": summary["facts_covered"], "total": summary["facts_total"]},
        # Gold labels cover the blocks that answer each question, not every
        # block that happens to be on topic, so a retrieved chunk with no
        # label counts against precision@5 as if it were wrong. Treat it as a
        # lower bound; recall@5 and fact_coverage are the honest measures of
        # how much of the needed evidence came back.
        "precision_is_a_lower_bound": True,
        "by_format": _by_format(rows, names),
        "leaks": leaks,
        "trap_leak_rate": (
            sum(1 for leak in leaks if leak["leaked"]) / len(leaks) if leaks else None
        ),
        "config": {
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.TOP_K,
            "min_score": config.MIN_SCORE,
            "max_per_source": config.MAX_PER_SOURCE,
            "hybrid_enabled": config.HYBRID_ENABLED,
        },
        "rows": rows,
    }
    return report


def _shared_positions(store: VectorStore, names: list[str], session_id: str) -> dict[str, dict[int, str]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT s.name, c.position, c.id::text FROM chunks c "
            "JOIN sources s ON s.id = c.source_id WHERE s.session_id = %s",
            (session_id,),
        ).fetchall()
    out: dict[str, dict[int, str]] = defaultdict(dict)
    for name, position, chunk_id in rows:
        out[name][int(position)] = chunk_id
    return out


def _by_format(rows: list[dict], names: list[str]) -> dict[str, dict]:
    fmt_of = {name: Path(name).suffix.lower().lstrip(".") for name in names}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("metrics"):
            grouped[fmt_of.get(row["source"], "?")].append(row)
    out: dict[str, dict] = {}
    for fmt, group in sorted(grouped.items()):
        summary = _means(group)
        out[fmt] = {"scored": summary["scored"], "means": summary["means"],
                    "fact_coverage": summary["fact_coverage"]}
    return out


def _print_detail(rows: list[dict], leaks: list[dict], k: int) -> None:
    print()
    print("=" * 92)
    print(f"PER QUESTION  k={k}")
    print("=" * 92)
    for row in rows:
        if not row.get("metrics"):
            print(f"  {row['id']}  SKIPPED  {row.get('note', '')}")
            continue
        got = row["metrics"]
        facts = row["facts"]
        flag = "ok " if got[f"recall@{k}"] == 1.0 else "MISS"
        print(
            f"  {row['id']} {flag} {row['source'][:22]:22}"
            f" recall {got[f'recall@{k}']:.2f}  prec {got[f'precision@{k}']:.2f}"
            f"  ndcg {got[f'ndcg@{k}']:.2f}  facts {facts['covered']}/{facts['total']}"
        )
        if facts["missing"]:
            for missing in facts["missing"]:
                print(f"        missing fact: {missing[:88]}")
    if leaks:
        print()
        print("=" * 92)
        print("TRAP LEAKS")
        print("=" * 92)
        for leak in leaks:
            tag = "LEAK" if leak["leaked"] else "ok  "
            print(f"  {tag} {leak['best_score']:.3f} {leak['hits']:>2} hits  {leak['question'][:60]}")
        caught = sum(1 for leak in leaks if not leak["leaked"])
        print(f"  caught {caught}/{len(leaks)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=("isolated", "shared", "both"), default="both")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="score unverified labels as a dry run; writes no results file",
    )
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    db.migrate()
    report = run(args.k, args.mode, args.allow_unverified)

    if not report["rows"] and not report.get("leaks"):
        return 0

    print()
    print("=" * 92)
    print("SUMMARY")
    print("=" * 92)
    for name, value in sorted(report["means"].items()):
        print(f"  {name:16} {value:.3f}")
    print(f"  {'fact_coverage':16} {report['fact_coverage']:.3f}")
    print("  precision@5 is a lower bound: unlabelled chunks count against it")
    if report["trap_leak_rate"] is not None:
        print(f"  {'trap_leak_rate':16} {report['trap_leak_rate']:.3f}")

    if not report["human_verified"]:
        print()
        print("Not writing results: the scored labels are not human-verified.")
        print("Verify the gold items, or pass --allow-unverified for a dry run.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())