"""Retrieval on three kinds of question, side by side, with n on every number.

    .venv/bin/python -m experiments.eval_sets

The cloze set is easy by construction: its question is the document line with a
blank in it, so the answer shares almost every word with the question. This runs
the same measurement on that set, on the paraphrased questions, and on the
entity-overlap traps, in one shared notebook holding the whole corpus (the
product's real condition), so the three numbers can be read against each other:

  cloze        question = the document line, blanked          (inflated by design)
  paraphrase   question = how a reader would ask it           (the honest number)
  traps        reuse a real entity, ask for a fact that is not there

One yardstick for the two answerable sets, so they are comparable: an item is
*found at rank r* when the r-th retrieved chunk is from the right source and
contains the item's verbatim span (the filled line for a cloze item, the quoted
span for a paraphrase). Recall@5 is "found at any rank <= 5", MRR@5 is the mean
of 1/r. This is stricter and simpler than the position-based scoring in
`eval_gold.py`, and it reads the retrieved text itself rather than trusting a
position index.

The labels are machine-checked and model-drafted, not human-reviewed; the output
says so and carries `human_verified: false`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db, parsers  # noqa: E402
from app.store import VectorStore  # noqa: E402
from experiments.build_paraphrase_set import overlap  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
GOLD = HERE / "gold"
RESULTS = HERE / "results" / "retrieval_sets_metrics.json"
K = 5


def _norm(text: str) -> str:
    return " ".join(text.split())


def _load(name: str) -> list[dict]:
    return json.loads((GOLD / name).read_text(encoding="utf-8"))["items"]


def index_corpus(store: VectorStore) -> str:
    session = db.create_session("eval-sets: whole corpus")
    sid = str(session.id)
    for path in sorted(CORPUS.iterdir()):
        if path.suffix.lower() in parsers.SUPPORTED:
            store.add(path, display_name=path.name, session_id=sid)
    return sid


def _span(item: dict) -> str:
    """The verbatim text a correct chunk must contain."""
    if item.get("kind") == "paraphrase":
        return item["required_facts"][0]
    return item["windows"][0]


def score_answerable(store: VectorStore, sid: str, items: list[dict]) -> dict:
    rows = []
    for item in items:
        hits = store.search_detailed(item["question"], session_id=sid, top_k=K).hits
        span = _norm(_span(item))
        rank = next(
            (r for r, hit in enumerate(hits, start=1)
             if hit["source"] == item["source"] and span in _norm(hit["text"])),
            None,
        )
        rows.append({
            "id": item["id"], "source": item["source"], "answered": bool(hits), "rank": rank,
            "overlap": round(overlap(item["question"], _span(item)), 2),
        })
    n = len(rows)
    found = [r for r in rows if r["rank"] is not None]
    refused = [r for r in rows if not r["answered"]]
    return {
        "n": n,
        "recall_at_5": round(len(found) / n, 3),
        "hit_at_1": round(sum(1 for r in found if r["rank"] == 1) / n, 3),
        "mrr_at_5": round(sum(1 / r["rank"] for r in found) / n, 3),
        "refused_by_floor": len(refused),
        "mean_question_overlap_with_span": round(sum(r["overlap"] for r in rows) / n, 3),
        "missed": [r["id"] for r in rows if r["rank"] is None],
        "by_overlap": _bucket(rows),
    }


def _bucket(rows: list[dict]) -> dict:
    """Recall split by how much the question shares with the answer."""
    out = {}
    for name, low, high in (("low (<0.15)", 0.0, 0.15), ("mid (0.15-0.4)", 0.15, 0.4), ("high (>=0.4)", 0.4, 1.01)):
        group = [r for r in rows if low <= r["overlap"] < high]
        out[name] = {
            "n": len(group),
            "recall_at_5": round(sum(1 for r in group if r["rank"] is not None) / len(group), 3) if group else None,
        }
    return out


def score_traps(store: VectorStore, sid: str, items: list[dict]) -> dict:
    leaked = []
    for item in items:
        outcome = store.search_detailed(item["question"], session_id=sid, top_k=K)
        if outcome.hits:
            leaked.append({
                "id": item["id"], "question": item["question"],
                "best_score": round(outcome.best_score, 3),
                "top_source": outcome.hits[0]["source"], "mode": outcome.mode,
            })
    n = len(items)
    return {
        "n": n,
        "refused_by_floor": n - len(leaked),
        "leaked_past_floor": len(leaked),
        "leak_rate": round(len(leaked) / n, 3),
        "leaks": leaked,
    }


def run() -> dict:
    db.migrate()
    store = VectorStore()
    sid = index_corpus(store)
    cloze_all = _load("cloze_set.json")
    cloze = [i for i in cloze_all if i["answerable"]]
    cloze_traps = [i for i in cloze_all if not i["answerable"]]
    paraphrase = _load("paraphrase_set.json")
    entity_traps = _load("trap_set.json")

    report = {
        "what": "retrieval on three question styles, one shared notebook holding the whole corpus",
        "human_verified": False,
        "labels": "model-drafted and machine-checked; cloze labels machine-generated; none read by a person",
        "k": K,
        "settings": {
            "min_score": config.MIN_SCORE, "top_k_returned": K, "rerank": config.RERANK_ENABLED,
            "hybrid": config.HYBRID_ENABLED, "embed_model": config.EMBED_MODEL,
            "corpus_files": len([p for p in CORPUS.iterdir() if p.suffix.lower() in parsers.SUPPORTED]),
        },
        "cloze": score_answerable(store, sid, cloze),
        "paraphrase": score_answerable(store, sid, paraphrase),
        "traps": {
            "unrelated_topic (cloze set)": score_traps(store, sid, cloze_traps),
            "entity_overlap (trap set)": score_traps(store, sid, entity_traps),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    report = run()

    print("=" * 78)
    for name in ("cloze", "paraphrase"):
        r = report[name]
        print(f"{name:11} n={r['n']:<4} recall@5 {r['recall_at_5']:.3f}  hit@1 {r['hit_at_1']:.3f}  "
              f"mrr@5 {r['mrr_at_5']:.3f}  question/answer word overlap {r['mean_question_overlap_with_span']:.2f}")
        for bucket, v in r["by_overlap"].items():
            print(f"{'':13}overlap {bucket:15} n={v['n']:<3} recall@5 {v['recall_at_5']}")
        if r["missed"]:
            print(f"{'':13}missed: {', '.join(r['missed'][:12])}")
    for name, r in report["traps"].items():
        print(f"traps: {name:30} n={r['n']:<3} leaked past the floor {r['leaked_past_floor']}/{r['n']} "
              f"(rate {r['leak_rate']:.3f})")
    print("labels: model-drafted / machine-generated, human_verified = false")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
