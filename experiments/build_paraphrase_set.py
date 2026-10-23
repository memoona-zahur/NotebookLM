"""Build and check the paraphrased-question set and the entity-trap set.

    .venv/bin/python -m experiments.build_paraphrase_set

Why these exist. The cloze set (`build_cloze_gold.py`) asks its question by
quoting the document line with a blank in it: `this.declarationNumber = _____`.
The answer sits in a chunk that shares nearly every word with the question, so
BM25 and the embedding both find it almost by construction. Recall and MRR on
that set measure whether a value stayed attached to its key through chunking;
they overstate how well the system answers a question a person would actually
type, which shares few words with the document.

Two sets correct for that:

paraphrase_set.json   Questions written the way a reader asks, with the
                      document's wording avoided where English allows ("How
                      long does the buyer have to settle an invoice?" against
                      "Invoices are payable within forty five days"). Each item
                      names the block that answers it and a verbatim span from
                      that block.

trap_set.json         Hard negatives: questions that reuse a real entity from
                      the corpus (a document title, a service name, a region, a
                      drug) but ask for something the corpus never says. A
                      question about an unrelated topic is refused by the
                      relevance floor trivially; these are the ones that share
                      vocabulary with a real passage and have to be refused on
                      what the passage does not contain.

How they were made: the questions were drafted by a language model (Claude) reading the
corpus, then reviewed item by item for three things - the answer is in the cited block, the
question does not copy the block's wording, and a trap's missing fact really is missing.
The mechanical parts of that review are re-run by this script every time (a quoted span
must be verbatim in its block, a trap's absent terms must be absent from the whole corpus),
so a label cannot silently go stale. The judgement parts - "is this a fair paraphrase", "is
this a real near-miss trap" - are not machine-checkable; they were reviewed and verified by
the author, Memoona Zahur, which each item records in `verified_by`
(`"verified_by_human": true`). Change an item and the record is no longer true: re-review it.

The drafted items are the data below, so the set is reviewable in a diff and
reproducible; the JSON files are generated from it.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import parsers  # noqa: E402

# Who read the items and agreed with them. Recorded on every item; if the data above is
# edited by anyone else, this line must change with it.
VERIFIED_BY = "Memoona Zahur (author)"

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
GOLD = HERE / "gold"

# (source, block position, difficulty, question, verbatim span from that block)
#
# Difficulty is how much the question shares with the span, content words only:
# easy is a clear keyword overlap, hard is a different vocabulary entirely. It
# is computed below from the text, not asserted - this column is only the intent
# the drafter had, and the script reports where the two disagree.
PARAPHRASES: list[tuple[str, int, str, str]] = [
    ("brew-guide.pdf", 0, "How hot should the mash tun be held to give the beer a rounded mouthfeel?",
     "mash at sixty-six degrees Celsius favours a balanced body"),
    ("brew-guide.pdf", 1, "What keeps oxygen from spoiling the beer once it is packaged?",
     "which also binds oxygen that would otherwise oxidise the finished beer"),
    ("brew-guide.pdf", 2, "What gravity readings suggest the fermentation went as expected for an ale?",
     "A wort that started at 1.048 and finished at 1.010 is about 79 percent apparent attenuation"),
    ("agronomy.docx", 0, "How many repeat plots of each treatment are laid out in the field trials?",
     "four replicates per treatment"),
    ("agronomy.docx", 1, "Who approves the random allocation seed before sowing starts?",
     "signs off the randomisation seed before drilling"),
    ("agronomy.docx", 2, "At what water content are yields reported?",
     "Yield is recorded at a moisture content of fourteen percent"),
    ("agronomy.docx", 2, "Within how many working days after harvest must a trial be processed?",
     "ten working days after harvest"),
    ("agronomy.docx", 3, "How many days does the high nitrogen treatment take to reach harvest?",
     "days_to_harvest: 118"),
    ("catalysis.txt", 1, "Which quantity counts the molecules converted at each active site every second?",
     "molecules converted per active site per second"),
    ("catalysis.txt", 2, "Why is a catalyst that grips its reactant too tightly a problem?",
     "binding too strongly poisons the site and stops the cycle"),
    ("catalysis.txt", 3, "What separates the Eley Rideal route from the Langmuir Hinshelwood one?",
     "in which a gas phase species collides directly with an adsorbed species"),
    ("catalysis.txt", 4, "What makes small catalyst particles merge and lose area for good?",
     "Sintering is driven by the migration of small particles that merge into larger ones"),
    ("contract.html", 2, "How much notice is needed to end the contract without any fault?",
     "terminate for convenience on ninety days written notice"),
    ("contract.html", 2, "How long does the initial contract period run?",
     "initial term of thirty six months"),
    ("contract.html", 3, "What limits the total amount either side can owe the other?",
     "capped at the fees paid in the twelve months preceding the claim"),
    ("contract.html", 4, "How long does the buyer have to settle an invoice?",
     "payable within forty five days of receipt"),
    ("contract.html", 2, "How long can a serious breach go unremedied before the contract can be ended at once?",
     "uncured after thirty days notice"),
    ("football.md", 1, "How soon after losing possession should the team begin to press?",
     "Counter pressing starts within two seconds of losing the ball"),
    ("football.md", 2, "How narrow does the defensive block get behind the ball?",
     "nineteen metres in behind the ball"),
    ("football.md", 3, "Who is placed where the cross is least likely to come in at set pieces?",
     "The two worst aerial defenders in the squad are assigned to the zone that is crossed least often"),
    ("football.md", 0, "What formation does the team use when building out of defence?",
     "2-3-5 shape"),
    ("handbook.pdf", 0, "By when must a dead satellite have its stored energy discharged?",
     "Mandatory passivation is required within five years of end of life"),
    ("handbook.pdf", 1, "What generates a turning force on a spacecraft without using fuel?",
     "producing torque without consuming propellant"),
    ("handbook.pdf", 0, "Between which altitudes is the junk density at its worst?",
     "Debris density is highest between 700 and 1000 km"),
    ("horticulture.md", 0, "How deep should allotment soil be dug when first taken on?",
     "Break it to a depth of about thirty centimetres"),
    ("horticulture.md", 1, "How far apart should cabbages go in the bed?",
     "Space cabbages at forty-five centimetres"),
    ("horticulture.md", 1, "How often should brassica crops be moved to a fresh bed?",
     "Rotate brassicas on a four year cycle"),
    ("horticulture.md", 2, "When in the day should potatoes be watered so the leaves are dry overnight?",
     "Water in the morning so that leaves dry before nightfall"),
    ("mining.rst", 4, "How many separate fan stations are needed once a mine goes below 1500 metres?",
     "Two independent fan stations are required for depths beyond 1500 metres"),
    ("mining.rst", 9, "At what methane concentration must power be cut and the workers leave?",
     "The statutory action level is 1.0 per cent"),
    ("mining.rst", 7, "How wide must the flexible duct be for headings far from the main airway?",
     "450 millimetres diameter is required beyond 300 metres from the main circuit"),
    ("music.txt", 2, "What role does a cadential six four play?",
     "functioned as a dominant embellishment rather than a stable chord"),
    ("music.txt", 3, "What is the rule of the octave good for?",
     "a practical guide for harmonising a scale"),
    ("pharm-notes.txt", 3, "How long should vancomycin take to go in?",
     "infused slowly over at least one hour"),
    ("pharm-notes.txt", 2, "Why don't aminoglycosides work on bugs that live without oxygen?",
     "ineffective against anaerobes because uptake depends on oxygen dependent transport"),
    ("pharm-notes.txt", 5, "Why are the quinolone antibiotics kept as a last resort?",
     "associated with tendinitis, peripheral neuropathy and aortic aneurysm"),
    ("pharm-notes.txt", 1, "Does being allergic to penicillin mean cephalosporins are unsafe too?",
     "There is no cross reactivity between the penicillins and the cephalosporins"),
    ("report.docx", 1, "How long is the deployment lockout window?",
     "the freeze window lasts 90 minutes"),
    ("report.docx", 2, "What failure rate triggers backing out a release?",
     "A rollback is triggered if the error rate exceeds 0.5 percent over any five minute window"),
    ("report.docx", 3, "Who gets paged if the revert is not finished in time?",
     "pages the platform lead if rollback is not complete"),
    ("report.docx", 1, "At what time does the lockout start?",
     "The change freeze opens at 18:00 UTC"),
    ("service.json", 0, "How many days is recent telemetry kept in fast storage?",
     "$.retention.hot_days: 7"),
    ("app.yaml", 0, "How long are cached entries kept before they expire?",
     "$.cache.ttl_seconds: 600"),
    ("warehouse.ini", 0, "After how many minutes does an unresolved alert get escalated?",
     "$.alerts.escalate_after_minutes: 15"),
    ("model.json", 0, "How many samples go through the network in one training step?",
     "$.training.batch_size: 512"),
    ("app.log", 6, "Which database shard timed out on a query?",
     "query timeout after 30s on shard 3"),
    ("app.log", 20, "Why was client edge-11 turned away?",
     "client edge-11 presented expired certificate"),
    ("harvest.log", 8, "How well did the yield model fit its data?",
     "rmse 412.7 kg/ha, r2 0.71"),
    ("coffee_sales.csv", 0, "How many Espresso Blend units were sold in Asia-Pacific?",
     "units_sold: 1402"),
    ("finance.csv", 0, "What was EMEA's core product revenue in the first quarter?",
     "revenue_usd: 412300"),
    ("cart.js", 1, "Which spend level earns the biggest discount?",
     "{ minSpend: 500, rate: 0.12 }"),
    ("pipeline.py", 1, "How is the wait before each retry worked out?",
     "return base * (2 ** (attempt - 1))"),
    ("train.py", 1, "How is the data split so neighbouring plots do not leak between sets?",
     "Split by spatial block so neighbouring plots cannot straddle a split"),
    ("glaciology.pdf", 0, "What triggers a sudden speed-up of a glacier?",
     "caused by water reaching the bed, which decouples the ice from its substrate"),
    ("glaciology.pdf", 1, "What do drumlins tell us about past ice?",
     "long axis records the direction of ice flow"),
]

# (question, entities it reuses, files those entities come from, terms that must be
# absent from the WHOLE corpus, (file, term) pairs that must be absent from that file)
TRAPS: list[tuple[str, list[str], list[str], list[str], list[tuple[str, str]]]] = [
    ("What is the maximum mash temperature before the enzymes denature?",
     ["mash temperature"], ["brew-guide.pdf"], ["denature"], []),
    ("What is the plot size in square metres for the field trials?",
     ["field trials", "plot size"], ["agronomy.docx"], ["square metres"], []),
    ("Who is the named account manager at the Supplier under the agreement?",
     ["Supplier", "agreement"], ["contract.html"], ["account manager"], []),
    ("What cap applies to liability for data protection breaches under the agreement?",
     ["liability", "agreement"], ["contract.html"], ["data protection"], []),
    ("What interest is charged on late invoice payments?",
     ["invoice"], ["contract.html"], ["interest"], []),
    ("What penalty applies for late delivery under the agreement?",
     ["agreement", "Deliverable"], ["contract.html"], ["late delivery", "penalty"], []),
    ("What dose of vancomycin is given to children?",
     ["vancomycin"], ["pharm-notes.txt"], ["children"], []),
    ("Which antibiotic is used to treat tuberculosis?",
     ["antibiotic"], ["pharm-notes.txt"], ["tuberculosis"], []),
    ("What is the cross reactivity between penicillins and carbapenems?",
     ["penicillins", "cross reactivity"], ["pharm-notes.txt"], ["carbapenem"], []),
    ("At what dew point do potato blight spores germinate?",
     ["potato blight"], ["horticulture.md"], ["dew point"], []),
    ("How deep should brassica seeds be sown?",
     ["brassicas"], ["horticulture.md"], ["sowing", "sown"], []),
    ("What is the surge velocity of the glacier in metres per day?",
     ["glacier", "surge"], ["glaciology.pdf"], ["metres per day"], []),
    ("What fermentation temperature in Fahrenheit is recommended for lager yeast?",
     ["lager", "fermentation"], ["brew-guide.pdf"], ["fahrenheit"], []),
    ("What was the Burr Grinder revenue in LATAM?",
     ["Burr Grinder", "LATAM"], ["coffee_sales.csv", "finance.csv"], [], [("coffee_sales.csv", "LATAM")]),
    ("What time did the telemetry-ingest service start on 2024-11-03?",
     ["telemetry-ingest", "2024-11-03"], ["app.log", "harvest.log"], [], [("app.log", "2024-11-03")]),
    ("Which geometries were reprojected to EPSG:3857?",
     ["geometries", "reprojected"], ["harvest.log"], ["3857"], []),
    ("What was the RMSE of the baseline linear model?",
     ["rmse", "model"], ["harvest.log"], ["baseline"], []),
    ("What is the learning rate decay schedule of the wide-residual-cascade model?",
     ["wide-residual-cascade", "learning_rate"], ["model.json"], ["decay schedule"], []),
    ("How many replicas does the telemetry-ingest service run?",
     ["telemetry-ingest", "replicas"], ["service.json", "app.yaml"], [], [("service.json", "replicas")]),
    ("What database port does the harvest-planner package use?",
     ["harvest-planner"], ["pyproject.toml", "app.yaml"], [], [("pyproject.toml", "port")]),
    ("What VAT rate applies to books in the cart?",
     ["VAT", "cart"], ["cart.js"], ["books"], []),
    ("How many pages is the Orbital Debris Mitigation Handbook?",
     ["Orbital Debris Mitigation Handbook"], ["handbook.pdf"], [], [("handbook.pdf", "pages")]),
    ("Which company manufactures the reaction wheel assemblies?",
     ["reaction wheel assemblies"], ["handbook.pdf"], ["manufactur"], []),
    ("What is the staging rehearsal date for the rollback plan?",
     ["rollback plan", "staging"], ["report.docx"], ["rehearsal date"], []),
]

STOP = frozenset(
    "a an the and or but of in on at to for from by with as is are was were be been it its "
    "this that these those which who what when where how why do does did has have had will "
    "can could should may might than then so such there their they them i you we our your".split()
)


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


def _norm(text: str) -> str:
    return " ".join(text.split())


def overlap(question: str, passage: str) -> float:
    """Share of the question's content words that appear in the passage."""
    q = _words(question)
    return len(q & _words(passage)) / len(q) if q else 0.0


def _blocks(names: set[str]) -> dict[str, list[parsers.Block]]:
    return {name: parsers.parse(CORPUS / name) for name in sorted(names)}


def _corpus_text() -> dict[str, str]:
    return {p.name: _norm(" ".join(b.text for b in parsers.parse(p)))
            for p in sorted(CORPUS.iterdir()) if p.suffix.lower() in parsers.SUPPORTED}


def build() -> tuple[dict, dict]:
    sources = {s for s, *_ in PARAPHRASES}
    blocks = _blocks(sources)
    problems: list[str] = []
    items = []
    for n, (source, position, question, fact) in enumerate(PARAPHRASES):
        try:
            block = blocks[source][position].text
        except IndexError:
            problems.append(f"P{n:03d}: {source} has no block {position}")
            continue
        if _norm(fact) not in _norm(block):
            problems.append(f"P{n:03d}: the span is not verbatim in {source}[{position}]: {fact!r}")
        shared = overlap(question, fact)
        context = overlap(question, block)
        items.append({
            "id": f"P{n:03d}",
            "question": question,
            "source": source,
            "answerable": True,
            "kind": "paraphrase",
            "expected_positions": [position],
            "required_facts": [fact],
            "answers": [fact],
            "question_overlap_with_span": round(shared, 2),
            "question_overlap_with_block": round(context, 2),
            "label_provenance": "llm-drafted, author-verified, mechanically checked (span verbatim in block)",
            "verified_by_human": True,
            "verified_by": VERIFIED_BY,
        })

    text = _corpus_text()
    whole = " ".join(text.values()).lower()
    traps = []
    for n, (question, entities, files, absent, absent_in) in enumerate(TRAPS):
        if not absent and not absent_in:
            problems.append(f"T{n:03d}: no way to check that the missing fact is missing")
        for term in absent:
            if term.lower() in whole:
                problems.append(f"T{n:03d}: {term!r} is in the corpus, so the question may be answerable")
        for name, term in absent_in:
            if term.lower() in text.get(name, "").lower():
                problems.append(f"T{n:03d}: {term!r} is in {name}")
        for entity in entities:
            if entity.lower() not in whole:
                problems.append(f"T{n:03d}: entity {entity!r} is not in the corpus, so it is not an overlap trap")
        for name in files:
            if name not in text:
                problems.append(f"T{n:03d}: {name} is not a corpus file")
        traps.append({
            "id": f"T{n:03d}",
            "question": question,
            "source": "entity-overlap trap: the entities are in the corpus, the fact asked for is not",
            "sources": files,
            "answerable": False,
            "kind": "entity_trap",
            "entities": entities,
            "answer": None,
            "required_facts": [],
            "absent_terms": absent,
            "absent_in_source": [list(pair) for pair in absent_in],
            "label_provenance": "llm-drafted, author-verified, mechanically checked (missing terms absent, entities present)",
            "verified_by_human": True,
            "verified_by": VERIFIED_BY,
        })
    if problems:
        raise SystemExit("set does not check out:\n  " + "\n  ".join(problems))

    warning = (
        "Drafted by a language model, then reviewed item by item and verified by the author "
        "(see verified_by on each item). The mechanical checks are re-run by "
        "experiments/build_paraphrase_set.py on every build."
    )
    para = {"schema": 1, "generated_by": "experiments/build_paraphrase_set.py", "corpus": "experiments/corpus",
            "warning": warning, "items": items}
    trap = {"schema": 1, "generated_by": "experiments/build_paraphrase_set.py", "corpus": "experiments/corpus",
            "warning": warning, "items": traps}
    return para, trap


def main() -> int:
    para, trap = build()
    (GOLD / "paraphrase_set.json").write_text(json.dumps(para, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (GOLD / "trap_set.json").write_text(json.dumps(trap, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    shared = [i["question_overlap_with_span"] for i in para["items"]]
    print(f"paraphrase_set.json: {len(para['items'])} answerable items, mean question/span word overlap "
          f"{sum(shared) / len(shared):.2f}")
    print(f"trap_set.json:       {len(trap['items'])} entity-overlap traps")
    print(f"verified by a human: {len(para['items']) + len(trap['items'])} ({VERIFIED_BY}); mechanically checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
