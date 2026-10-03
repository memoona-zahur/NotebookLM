"""Compare chunking settings on the same mechanically checked labels.

    .venv/bin/python -m experiments.chunker_bakeoff
    .venv/bin/python -m experiments.chunker_bakeoff --sizes 400 900 1500 --k 5

Chunking happens while parsing, so changing the chunk size changes which
blocks exist - and `chunks.position` is the parser's block index. A gold
position therefore cannot be reused across settings: at a different size,
position 7 is a different block. So gold chunks are resolved here by text. An
item's required fact is a verbatim block, and a setting is only credited for
it if some stored chunk contains that text whole.

That distinction is the point of the script. A small chunk size can score
perfectly on the questions it can still answer while destroying the records
that no longer fit in one chunk. Those items are not scored as failures,
because the label stopped existing rather than the retriever failing; they are
counted as `gold_intact`, and a setting that shreds records is visible in that
column instead of hiding inside an average.

Read `fact_coverage` as the honest headline: the fraction of questions whose
whole evidence came back in the retrieved context. It survives a fact spanning
two chunks, so it does not reward a chunker for never splitting anything.

The labels are machine-checked, not human-reviewed. See
experiments/build_cloze_gold.py for what that does and does not establish.
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
from experiments import corpus, eval_gold, metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
CLOZE_PATH = HERE / "gold" / "cloze_set.json"
RESULTS_PATH = HERE / "results" / "chunker_bakeoff.json"

# Overlap is derived from size rather than listed separately: a table of
# (size, overlap) pairs invites the two to drift apart, and the question worth
# asking is what the ratio buys, not what one hand-picked pair does.
DEFAULT_SIZES = [300, 450, 600, 900, 1200, 1600]


def stored_chunks(session_id: str) -> list[tuple[str, str]]:
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT c.id::text, c.text FROM chunks c "
            "JOIN sources s ON s.id = c.source_id WHERE s.session_id = %s",
            (session_id,),
        ).fetchall()
    return [(row[0], row[1]) for row in rows]


def resolve_gold(item: dict, chunks: list[tuple[str, str]]) -> list[str]:
    """Stored chunks that contain every required fact, whole.

    Matching on text rather than on position is what lets one label set score
    several chunking settings at all.
    """
    ids = [chunk_id for chunk_id, _ in chunks]
    out: list[str] = []
    for fact in item["required_facts"]:
        needle = metrics.normalize(fact)
        holders = [chunk_id for chunk_id, text in chunks if needle in metrics.normalize(text)]
        if not holders:
            return []  # the fact no longer fits in any chunk at this setting
        out.extend(holders)
    return sorted(set(out))


def index_at(
    size: int, overlap: int, names: list[str], mode: str
) -> tuple[dict[str, str], int]:
    """Index the corpus at one setting. Returns per-document session ids.

    In shared mode there is a single session holding every document, which is
    the product condition. In isolated mode each document gets its own, so a
    wrong chunk is unambiguously wrong and the numbers compare chunkers without
    cross-document competition.
    """
    config.CHUNK_SIZE = size
    config.CHUNK_OVERLAP = overlap
    store = VectorStore()
    tag = f"{size}-{overlap}-{mode}"
    sessions: dict[str, str] = {}
    if mode == "shared":
        session = db.create_session(f"bakeoff:{tag}: all documents")
        shared_id = str(session.id)
        for name in names:
            store.add(eval_gold.CORPUS / name, display_name=name, session_id=shared_id)
            sessions[name] = shared_id
    else:
        for name in names:
            session = db.create_session(f"bakeoff:{tag}: {name}")
            session_id = str(session.id)
            store.add(eval_gold.CORPUS / name, display_name=name, session_id=session_id)
            sessions[name] = session_id
    total = 0
    for session_id in set(sessions.values()):
        total += len(stored_chunks(session_id))
    return sessions, total


def score_setting(
    store: VectorStore,
    sessions: dict[str, str],
    items: list[dict],
    k: int,
) -> dict:
    rows: list[dict] = []
    intact = 0
    cache: dict[str, list[tuple[str, str]]] = {}
    for item in items:
        session_id = sessions[item["source"]]
        if session_id not in cache:
            cache[session_id] = stored_chunks(session_id)
        gold = resolve_gold(item, cache[session_id])
        outcome = store.search_detailed(item["question"], session_id=session_id, top_k=k)
        retrieved = [hit["chunk_id"] for hit in outcome.hits]
        context = "\n".join(hit["text"] for hit in outcome.hits)
        facts = metrics.fact_coverage(item["required_facts"], context)
        if not gold:
            # The label is intact in the corpus but not in any single chunk.
            # Scoring this 0.0 would blame the retriever for the chunker.
            rows.append(
                {
                    "id": item["id"],
                    "source": item["source"],
                    "metrics": None,
                    "facts": facts,
                    "note": "no single chunk holds the whole fact",
                }
            )
            continue
        intact += 1
        rows.append(
            {
                "id": item["id"],
                "source": item["source"],
                "metrics": metrics.score_query(gold, retrieved, k),
                "facts": facts,
            }
        )
    summary = metrics.aggregate(rows)
    fact_rows = [r for r in rows if r.get("facts")]
    covered = sum(r["facts"]["covered"] for r in fact_rows)
    total = sum(r["facts"]["total"] for r in fact_rows)
    return {
        "scored": summary["scored"],
        "unresolvable": len(rows) - summary["scored"],
        "gold_intact": intact / len(items) if items else 0.0,
        "means": summary["means"],
        "fact_coverage": covered / total if total else 0.0,
        "rows": rows,
    }


def run(
    sizes: list[int], overlap_ratio: float, k: int, mode: str, gold_path: Path = CLOZE_PATH
) -> dict:
    items = eval_gold.load_gold(gold_path)
    stale = eval_gold.machine_failures(items)
    if stale:
        raise SystemExit(f"refusing to compare settings on stale labels: {sorted(stale)}")
    machine_ids = {item["id"] for item in items}
    scorable, _ = eval_gold.partition(items, machine_ids=machine_ids)
    names = sorted({item["source"] for item in scorable})
    print(f"{len(scorable)} answerable items over {len(names)} documents, k={k}")

    baseline = (config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    settings = [(size, int(size * overlap_ratio)) for size in sizes]
    results: list[dict] = []
    for size, overlap in settings:
        sessions, chunk_total = index_at(size, overlap, names, mode)
        store = VectorStore()
        summary = score_setting(store, sessions, scorable, k)
        entry = {
            "chunk_size": size,
            "chunk_overlap": overlap,
            "chunks_stored": chunk_total,
            **{key: summary[key] for key in ("scored", "unresolvable", "gold_intact", "means", "fact_coverage")},
        }
        results.append(entry)
        print(
            f"  size {size:>5} overlap {overlap:>4}  chunks {chunk_total:>5}"
            f"  recall@5 {summary['means'].get(f'recall@{k}', 0.0):.3f}"
            f"  mrr@5 {summary['means'].get(f'mrr@{k}', 0.0):.3f}"
            f"  facts {summary['fact_coverage']:.3f}"
            f"  intact {summary['gold_intact']:.3f}"
            f"  (unscored {summary['unresolvable']})"
        )

    config.CHUNK_SIZE, config.CHUNK_OVERLAP = baseline
    return {
        "k": k,
        "mode": mode,
        "gold_file": str(gold_path),
        "label_basis": "machine-checked",
        "human_verified": False,
        "items": len(scorable),
        "documents": len(names),
        "settings": results,
        "baseline_config": {"chunk_size": baseline[0], "chunk_overlap": baseline[1]},
        "note": (
            "precision@k is a lower bound: labels cover the blocks that answer each "
            "question, not every block on topic. fact_coverage and gold_intact are "
            "the measures to compare settings on."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="*", default=None)
    parser.add_argument("--overlap-ratio", type=float, default=1 / 6)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=("isolated", "shared"), default="shared")
    parser.add_argument("--gold", type=Path, default=CLOZE_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    sizes = args.sizes or DEFAULT_SIZES
    db.migrate()
    report = run(sizes, args.overlap_ratio, args.k, args.mode, gold_path=args.gold)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print()
    print(f"wrote {args.out}")
    print("  human_verified: false - machine-checked labels, a regression baseline only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())