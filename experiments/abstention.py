"""Abstention quality: does the accept/reject decision separate answerable
questions from ones the sources cannot answer?

    .venv/bin/python experiments/abstention.py

This is the metric the standard RAG metric list does not contain, and for this
system it is the one that matters most. Every other metric scores the passages
that were already retrieved; this scores the decision about whether to answer
at all. A system that ranks well but answers everything is useless, and a
system that refuses everything is worse.

The corpus gives both classes for free: each document has questions its own
text answers (`relevant`) and questions about entities it deliberately does not
mention (`trap`). That is a labelled set, so no manual annotation is needed.

Definitions, stated because these names have several valid implementations:

    Abstained      = best score < threshold  (system said "not in the sources")
    Abstention recall    = traps correctly abstained on / all traps
    Abstention precision = traps correctly abstained on / all abstentions

Reading them: precision is "when it refuses, how often is it right" - the cost
of a false refusal is a legitimate question going unanswered. Recall is "of the
questions it must refuse, how many did it" - the cost of a miss is a trap
getting answered from unrelated text, which is the hallucination this whole
system exists to prevent.

AUC is the threshold-free summary: the probability that a random trap scores
below a random relevant question. 0.5 means the score carries no information
about answerability; 1.0 means perfect separation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments import eval_retrieval  # noqa: E402


def _sweep(rel_scores, trap_scores, steps=200):
    """Every distinct cut point, not a hand-picked grid.

    Sweeping arbitrary thresholds can skip the only value that matters; using
    the observed scores as cut points makes the curve exact for this corpus.
    """
    cuts = sorted({round(s, 6) for s in rel_scores + trap_scores})
    if len(cuts) > steps:
        # Keep the extremes and thin the middle uniformly.
        keep = [cuts[round(i * (len(cuts) - 1) / (steps - 1))] for i in range(steps)]
        cuts = sorted(set(keep))

    rows = []
    for threshold in cuts:
        abstained_rel = sum(1 for s in rel_scores if s < threshold)
        abstained_trap = sum(1 for s in trap_scores if s < threshold)
        # A refusal on an answerable question is a false refusal.
        precision = (
            abstained_trap / (abstained_trap + abstained_rel)
            if (abstained_trap + abstained_rel)
            else 1.0
        )
        rows.append(
            {
                "threshold": threshold,
                "answered_rel": len(rel_scores) - abstained_rel,
                "answered_trap": len(trap_scores) - abstained_trap,
                "abstention_precision": precision,
                "abstention_recall": abstained_trap / len(trap_scores) if trap_scores else 1.0,
                "false_refusals": abstained_rel,
                "false_answers": len(trap_scores) - abstained_trap,
                "errors": abstained_rel + (len(trap_scores) - abstained_trap),
            }
        )
    return rows


def _auc(rel_scores, trap_scores) -> float:
    """Mann-Whitney U. Exact, and O(n log n) rather than O(n*m) pairwise.

    AUC = P(a random trap scores below a random answerable question). High is
    good: the score should be low exactly when the sources cannot answer.
    """
    if not rel_scores or not trap_scores:
        return float("nan")
    merged = sorted([(s, 0) for s in rel_scores] + [(s, 1) for s in trap_scores])
    # Rank with ties averaged, so equal scores do not manufacture separation.
    ranks = [0.0] * len(merged)
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = average
        i = j + 1
    trap_rank_sum = sum(r for r, (_, label) in zip(ranks, merged) if label == 1)
    n_rel, n_trap = len(rel_scores), len(trap_scores)
    # U here counts pairs where a trap outranks a relevant question, which is
    # the failure direction. 1 - that is the separation.
    u_trap_above = trap_rank_sum - n_trap * (n_trap + 1) / 2
    return 1 - u_trap_above / (n_rel * n_trap)


def report(rows, verbose=True):
    rel = [x["best"] for r in rows for x in r["results"] if x["kind"] == "relevant"]
    trap = [x["best"] for r in rows for x in r["results"] if x["kind"] == "trap"]
    rel.sort()
    trap.sort()

    auc = _auc(rel, trap)
    sweep = _sweep(rel, trap)

    if verbose:
        print("=" * 78)
        print("ABSTENTION QUALITY   can the system tell 'answerable' from 'not in sources'?")
        print("=" * 78)
        print(f"relevant questions {len(rel)}   traps {len(trap)}")
        print(f"relevant score range  {rel[0]:.3f} .. {rel[-1]:.3f}")
        print(f"trap     score range  {trap[0]:.3f} .. {trap[-1]:.3f}")
        print()
        print(f"AUC {auc:.3f}  ({_verdict(auc)})")
        print()

        # The overlap is the whole story, so show it rather than summarise it.
        # Overlap is the part of the score range where the two classes are
        # interleaved. AUC can be high while a band is still contested, so this
        # is reported separately: it is the region where any threshold is a
        # trade-off rather than a free choice.
        high_trap = max(trap)
        beating = [s for s in rel if s < high_trap]
        print(
            f"{len(beating)} of {len(rel)} answerable questions score below the highest trap "
            f"({high_trap:.3f})."
        )
        print(f"  The contested band is {min(s for s in rel if s >= min(trap)):.3f} .. {high_trap:.3f}.")
        print("  Inside it, refusing and answering are both wrong for some question: that")
        print("  band is the irreducible cost of a single-score threshold, and no value of")
        print("  MIN_SCORE removes it.")
        print()

        print("=" * 78)
        print(
            f"{'threshold':>10}{'answerable':>12}{'traps':>8}{'abst.prec':>11}"
            f"{'abst.rec':>10}{'errors':>8}"
        )
        print("=" * 78)
        from app import config

        # Sample the curve rather than listing every cut point: the extremes
        # are what matter, and 200 rows hides the trade-off it exists to show.
        step = max(1, len(sweep) // 18)
        for row in sweep[::step]:
            mark = "  <- MIN_SCORE" if abs(row["threshold"] - config.MIN_SCORE) < 1e-9 else ""
            answered_rel = f"{row['answered_rel']}/{len(rel)}"
            answered_trap = f"{row['answered_trap']}/{len(trap)}"
            print(
                f"{row['threshold']:>10.3f}"
                f"{answered_rel:>12}"
                f"{answered_trap:>8}"
                f"{row['abstention_precision']:>11.3f}"
                f"{row['abstention_recall']:>10.3f}"
                f"{row['errors']:>8}{mark}"
            )
        for row in sweep:
            if abs(row["threshold"] - config.MIN_SCORE) < 1e-9:
                answered_rel = f"{row['answered_rel']}/{len(rel)}"
                answered_trap = f"{row['answered_trap']}/{len(trap)}"
                print(
                    f"{row['threshold']:>10.3f}"
                    f"{answered_rel:>12}"
                    f"{answered_trap:>8}"
                    f"{row['abstention_precision']:>11.3f}"
                    f"{row['abstention_recall']:>10.3f}"
                    f"{row['errors']:>8}  <- MIN_SCORE (configured)"
                )

        best = min(sweep, key=lambda row: row["errors"])
        print()
        print(
            f"lowest-error threshold {best['threshold']:.3f} "
            f"({best['errors']} errors: {best['false_refusals']} false refusals, "
            f"{best['false_answers']} false answers)"
        )
        _recommend(sweep, rel, trap)

    return {"auc": auc, "sweep": sweep, "rel": rel, "trap": trap}


def _verdict(auc: float) -> str:
    if auc >= 0.95:
        return "excellent separation"
    if auc >= 0.9:
        return "good separation"
    if auc >= 0.8:
        return "usable, some overlap"
    if auc >= 0.7:
        return "weak separation"
    return "score barely predicts answerability"


def _recommend(sweep, rel, trap):
    """Pick an operating point by what is actually worse, and say so.

    Choosing a threshold by lowest total error treats a false refusal and a
    hallucination as equally bad. For a source-grounded assistant they are not:
    a refusal is a recoverable annoyance, a hallucination is the failure the
    product exists to prevent.
    """
    from app import config

    # Weight a false answer heavily: the answer is presented as sourced.
    scored = [
        (row["false_refusals"] + 5 * row["false_answers"], row)
        for row in sweep
        if row["answered_rel"] >= 0.9 * len(rel)  # must still answer most questions
    ]
    if not scored:
        print("no threshold keeps 90% of answerable questions")
        return
    _, best = min(scored, key=lambda pair: pair[0])
    print()
    print("Cost-weighted (a false answer counts 5x a false refusal):")
    print(
        f"  threshold {best['threshold']:.3f} answers "
        f"{best['answered_rel']}/{len(rel)} answerable, refuses "
        f"{best['answered_trap']}/{len(trap)} traps "
        f"(abst. precision {best['abstention_precision']:.3f})"
    )
    if abs(best["threshold"] - config.MIN_SCORE) < 1e-9:
        print("  That is the configured MIN_SCORE.")
    else:
        print(
            f"  Configured MIN_SCORE is {config.MIN_SCORE:.3f} "
            f"({abs(best['threshold'] - config.MIN_SCORE):.3f} away). Raising it would cut"
        )
        print("  false answers at the cost of more false refusals.")


def main():
    rows, _ = eval_retrieval.evaluate_from(eval_retrieval.build_stores(), verbose=False)
    report(rows)


if __name__ == "__main__":
    main()
