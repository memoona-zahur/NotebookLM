"""Does the answer stay faithful to the passages it was given?

    .venv/bin/python -m experiments.eval_generation --k 5
    .venv/bin/python -m experiments.eval_generation --threshold-sweep

`eval_gold` answers whether the evidence reached the prompt. This answers what
came back out, using the same labels: `required_facts` is what a correct answer
must contain, so it grades generation the same way it grades retrieval instead
of needing a second annotation set that would drift from the first.

The three numbers reported are defined in `experiments/generation.py`, which is
also where the reasons for not using a judge model are written down. The short
version: an LLM judge costs money, is non-deterministic, and is from the same
model family as the system under test, so it fails for the same reasons the
system does. A metric that cannot detect the failure it is meant to detect is
not evidence.

Two things this harness deliberately does not do:

  * It does not score retrieval as good because the answer was good. Retrieval
    is measured separately, by `eval_gold`, with ranked metrics. Blurring the
    two hides which one broke.
  * It does not average a refusal into faithfulness. A correct refusal has no
    claims, and scoring it 0.0 would make the safest behaviour look like the
    worst. Refusals are reported as their own count and their own accuracy.

This harness calls a real model, so it costs tokens and it is not deterministic.
The metrics are deterministic; the answers are not. Treat a run as a
regression alarm with a threshold, not as a benchmark to quote.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db, llm  # noqa: E402
from app.store import VectorStore  # noqa: E402
from experiments import generation  # noqa: E402
from experiments.eval_gold import (  # noqa: E402
    CORPUS,
    load_gold,
    machine_failures,
    partition,
)

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results" / "generation_metrics.json"


def index_all(store: VectorStore, names: list[str], tag: str) -> str:
    """One session holding every document.

    The shared condition rather than one session per document, because this
    harness measures the answer and a model handed six documents behaves
    differently from one handed one. That is the condition the product runs in.
    """
    session = db.create_session(f"eval-gen:{tag}: all documents")
    for name in names:
        path = CORPUS / name
        if not path.exists():
            raise FileNotFoundError(f"{path} is referenced by the gold set but missing")
        store.add(path, display_name=name, session_id=str(session.id))
    return str(session.id)


def answer_one(store: VectorStore, item: dict, session_id: str, k: int) -> dict:
    """Ask one gold question and score the answer against its own labels."""
    search = store.search_detailed(item["question"], session_id=session_id, top_k=k)
    passages = [h["text"] for h in search.hits]

    if not search.relevant:
        # The relevance floor held, so the model was never asked. The correct
        # outcome for an answerable question is a miss, not a refusal success.
        return {
            "id": item["id"],
            "kind": item.get("kind") or "unspecified",
            "source": item["source"],
            "question": item["question"],
            "verdict": "no_match",
            "metrics": generation.score_answer(
                llm.NO_MATCH, passages, item["required_facts"], verdict="no_match"
            ),
            "retrieval_best_score": search.best_score,
            "context_chars": sum(len(p) for p in passages),
            "cost": llm.last_usage().as_dict(),
            "latency_ms": 0.0,
        }

    started = time.perf_counter()
    try:
        result = llm.answer(item["question"], search.hits, [])
    except llm.LLMUnavailable as exc:
        return {
            "id": item["id"],
            "kind": item.get("kind") or "unspecified",
            "source": item["source"],
            "question": item["question"],
            "verdict": "unavailable",
            "metrics": None,
            "error": str(exc),
        }
    elapsed = (time.perf_counter() - started) * 1000

    scored = generation.score_answer(
        result.text, passages, item["required_facts"], verdict="answered"
    )
    return {
        "id": item["id"],
        "kind": item.get("kind") or "unspecified",
        "source": item["source"],
        "question": item["question"],
        "verdict": "answered",
        "answer": result.text,
        "cited": result.cited,
        "invalid": result.invalid,
        "ungrounded": result.ungrounded,
        "metrics": scored,
        "retrieval_best_score": search.best_score,
        "context_chars": sum(len(p) for p in passages),
        "cost": llm.last_usage().as_dict(),
        "latency_ms": round(elapsed, 1),
    }


def trap_one(store: VectorStore, item: dict, session_id: str, k: int) -> dict:
    """An unanswerable question, where a refusal is the pass condition."""
    search = store.search_detailed(item["question"], session_id=session_id, top_k=k)
    if not search.relevant:
        return {
            "id": item["id"],
            "question": item["question"],
            "refused": True,
            "reason": "relevance floor",
            "retrieved": len(search.hits),
        }
    try:
        result = llm.answer(item["question"], search.hits, [])
    except llm.LLMUnavailable as exc:
        return {"id": item["id"], "question": item["question"], "refused": None, "error": str(exc)}

    # A trap that produced no claims and no answer text refused. A trap that
    # answered anything at all did not, whether or not the text was correct -
    # the failure here is that it spoke at all.
    spoke = bool(result.text.strip()) and bool(result.cited)
    return {
        "id": item["id"],
        "question": item["question"],
        "refused": not spoke,
        "reason": "answered" if spoke else "declined to answer",
        "answer": result.text[:400],
        "retrieved": len(search.hits),
        "cost": llm.last_usage().as_dict(),
    }


def summarise(rows: list[dict], traps: list[dict]) -> dict:
    """Aggregate without letting a refusal drag faithfulness down."""
    answered = [r for r in rows if r.get("metrics") and r["verdict"] == "answered"]
    refused = [r for r in rows if r["verdict"] == "no_match"]
    unavailable = [r for r in rows if r["verdict"] == "unavailable"]

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    faith = [r["metrics"]["faithfulness"]["score"] for r in answered]
    prec = [r["metrics"]["citations"]["score"] for r in answered]
    relev = [r["metrics"]["relevancy"]["score"] for r in answered]
    substance = [r["metrics"]["relevancy"]["substance"] for r in answered]
    verbatim = [r["metrics"]["relevancy"]["verbatim"] for r in answered]
    by_kind: dict[str, list[float]] = defaultdict(list)
    for r in answered:
        by_kind[r["kind"]].append(r["metrics"]["relevancy"]["substance"])

    # Completeness is reported as a distribution, not a mean. A mean over
    # "complete / partial / missed" hides which failure dominates, and partial
    # is the one with a different fix from missed.
    kinds: dict[str, int] = defaultdict(int)
    for r in answered:
        kinds[r["metrics"]["relevancy"]["kind"]] += 1

    latencies = [r["latency_ms"] for r in answered]
    tokens = [r["cost"]["total_tokens"] for r in answered if r.get("cost")]

    trap_refused = [t for t in traps if t.get("refused") is True]
    return {
        "answered": len(answered),
        "refused_by_floor": len(refused),
        "unavailable": len(unavailable),
        "faithfulness": mean(faith),
        "citation_precision": mean(prec),
        "relevancy": mean(relev),
        "substance": mean(substance),
        "verbatim": mean(verbatim),
        "relevancy_by_kind": {k: mean(v) for k, v in sorted(by_kind.items())},
        "relevancy_kind_counts": dict(sorted(kinds.items())),
        "uncited_claims": sum(r["metrics"]["uncited_claims"] for r in answered),
        "unsupported_claims": sum(
            len(r["metrics"]["faithfulness"]["unsupported"]) for r in answered
        ),
        "latency_ms_mean": mean(latencies),
        "tokens_mean": mean([float(t) for t in tokens]),
        "trap_refusal_rate": (
            len(trap_refused) / len(traps) if traps and all(t.get("refused") is not None for t in traps) else None
        ),
        "traps": len(traps),
    }


def _print_rows(rows: list[dict], limit: int = 12) -> None:
    """Show the worst answers, not the average."""
    bad = [
        r for r in rows
        if r.get("metrics")
        and (
            r["metrics"]["faithfulness"]["score"] < 1.0
            or r["metrics"]["relevancy"]["score"] < 1.0
            or r["metrics"]["relevancy"]["substance"] < 0.8
            or r["metrics"]["uncited_claims"]
        )
    ]
    bad.sort(
        key=lambda r: (
            r["metrics"]["faithfulness"]["score"],
            r["metrics"]["relevancy"]["score"],
        )
    )
    if not bad:
        print()
        print("Every answered item was fully faithful and complete.")
        return
    print()
    print(f"ITEMS NEEDING ATTENTION ({len(bad)} total, showing {min(limit, len(bad))})")
    for row in bad[:limit]:
        m = row["metrics"]
        print(f"  {row['id']:28} faith {m['faithfulness']['score']:.2f}"
              f"  subst {m['relevancy']['substance']:.2f}"
              f"  strict {m['relevancy']['score']:.2f} ({m['relevancy']['kind']})"
              f"  uncited {m['uncited_claims']}")
        for claim in m["faithfulness"]["unsupported"][:2]:
            print(f"      unsupported: {claim[:110]}")
        for miss in m["relevancy"]["missing"][:2]:
            print(f"      missing fact: {miss[:110]}")


def sweep(rows: list[dict]) -> None:
    """Re-score every answer at several support thresholds.

    `SUPPORT_THRESHOLD` is a judgement call, and a threshold nobody has tested
    is a constant that will be "tuned" by whoever next feels like changing it.
    This shows how much the headline number moves across a plausible range, and
    whether any item sits near the boundary - those are the ones whose score is
    an artefact of the threshold rather than of the answer.

    The support ratio per claim is threshold-independent; only the pass/fail
    against it is not. So the sweep re-reads the stored ratios instead of the
    passages, which means it does not need the whole corpus kept in the results
    file.
    """
    print()
    print("=" * 92)
    print("SUPPORT THRESHOLD SENSITIVITY")
    print("=" * 92)
    print("A row that barely moves means the number is not load-bearing. A")
    print("swinging row means it is a guess and should not be quoted alone.")
    print()

    answered = [r for r in rows if r.get("metrics") and r["verdict"] == "answered"]
    if not answered:
        print("  no answered items")
        return

    ratios = [
        float(c["support"])
        for r in answered
        for c in r["metrics"]["faithfulness"]["claims"]
    ]
    print(f"  {'threshold':>9} {'faithfulness':>13} {'claims':>8} {'near boundary':>14}")
    for threshold in (0.20, 0.25, 0.30, 0.34, 0.40, 0.50, 0.60):
        passing = sum(1 for s in ratios if s >= threshold)
        near = sum(1 for s in ratios if threshold - 0.06 <= s <= threshold + 0.06)
        score = passing / len(ratios) if ratios else None
        print(f"  {threshold:>9.2f} {score:>13.3f} {passing:>5}/{len(ratios):<3} {near:>14}")
    print()
    print(f"  In use: {generation.SUPPORT_THRESHOLD}")


def run(
    k: int,
    allow_machine: bool,
    allow_unverified: bool,
    limit: int | None,
    gold_path: Path,
) -> dict:
    items = load_gold(gold_path)
    stale = machine_failures(items) if allow_machine else {}
    machine_ids = {i["id"] for i in items} - set(stale) if allow_machine else set()
    scorable, _excluded = partition(
        items, include_unverified=allow_unverified, machine_ids=machine_ids
    )
    traps = [
        i for i in items
        if (i["verified_by_human"] or i["id"] in machine_ids)
        and i["answerable"] is False
        and not i["required_facts"]
    ]
    scorable = scorable[:limit] if limit else scorable
    traps = traps[:limit] if limit else traps

    print("=" * 92)
    print("GENERATION EVALUATION")
    print("=" * 92)
    print(f"provider/model:            {config.resolved_provider()} / {config.resolved_model()}")
    print(f"answerable items:          {len(scorable)}")
    print(f"trap questions:            {len(traps)}")
    print(f"support threshold:         {generation.SUPPORT_THRESHOLD}")
    print()
    print("This harness calls a real model. The metrics are deterministic; the")
    print("answers are not. Read it as a regression alarm, not a benchmark.")
    print()

    store = VectorStore()
    session_id = index_all(store, sorted({i["source"] for i in items}), "run")

    rows: list[dict] = []
    for n, item in enumerate(scorable, start=1):
        row = answer_one(store, item, session_id, k)
        rows.append(row)
        mark = {"answered": "ok", "no_match": "REFUSED", "unavailable": "SKIP"}.get(row["verdict"], "?")
        m = row.get("metrics") or {}
        faith = m.get("faithfulness", {}).get("score")
        rel = m.get("relevancy", {}).get("substance")
        print(f"  [{n:>3}/{len(scorable)}] {mark:8} {row['id']:26} "
              f"faith {('%.2f' % faith) if faith is not None else '  - '} "
              f"subst {('%.2f' % rel) if rel is not None else '  - '} "
              f"{(row.get('cost') or {}).get('total_tokens', 0):>6} tok")

    trap_rows: list[dict] = []
    for item in traps:
        trap_rows.append(trap_one(store, item, session_id, k))
    leaked = [t for t in trap_rows if t.get("refused") is False]
    print()
    print(f"traps refused {len(trap_rows) - len(leaked)}/{len(trap_rows)}")
    for t in leaked:
        print(f"  ANSWERED A TRAP {t['id']}: {t.get('answer', '')[:120]}")

    report = summarise(rows, trap_rows)
    sweep(rows)

    print()
    print("=" * 92)
    print("SUMMARY")
    print("=" * 92)
    print(f"  answered             {report['answered']}")
    print(f"  refused by floor     {report['refused_by_floor']}")
    for name in ("faithfulness", "citation_precision", "substance", "relevancy",
                 "verbatim", "latency_ms_mean", "tokens_mean"):
        value = report[name]
        print(f"  {name:20} {value:.3f}" if value is not None else f"  {name:20} (none)")
    print()
    print("  substance / relevancy / verbatim bracket the same answers from")
    print("  three strictnesses. substance is paraphrase-tolerant, relevancy")
    print("  requires every content word of a fact, verbatim is exact substring.")
    print("  A high substance with a low relevancy means the answer reworded the")
    print("  evidence correctly. A high relevancy with a low substance means the")
    print("  labels are too short to distinguish anything.")
    print()
    print(f"  relevancy kinds      {report['relevancy_kind_counts']}")
    print(f"  unsupported claims   {report['unsupported_claims']}")
    print(f"  uncited claims       {report['uncited_claims']}")
    if report["trap_refusal_rate"] is not None:
        print(f"  {'trap refusal rate':20} {report['trap_refusal_rate']:.3f}")
    _print_rows(rows)

    return {
        "model": config.resolved_model(),
        "provider": config.resolved_provider(),
        "k": k,
        "support_threshold": generation.SUPPORT_THRESHOLD,
        "human_verified": any(i["verified_by_human"] for i in scorable),
        "gold_file": gold_path.name,
        "summary": report,
        "rows": rows,
        "traps": trap_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generation-side evaluation against the gold set.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gold", type=Path, default=HERE / "gold" / "gold_set.json")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N items")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-machine", action="store_true")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    args = parser.parse_args()

    db.migrate()
    report = run(args.k, args.allow_machine, args.allow_unverified, args.limit, args.gold)

    if args.dry_run or not args.allow_machine and not args.allow_unverified:
        print()
        print("Dry run: writing no results file.")
        return 0

    out = args.out or RESULTS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())