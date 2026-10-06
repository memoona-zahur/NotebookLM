# Presenting this project

Everything needed to run, demo, and defend the app. Written to be read top to
bottom before a presentation, and to be followed without guessing.

- [1. What you are presenting](#1-what-you-are-presenting)
- [2. Decide who is listening](#2-decide-who-is-listening)
- [3. Run it](#3-run-it)
- [4. The demo, step by step](#4-the-demo-step-by-step)
- [5. What happens to a question, step by step](#5-what-happens-to-a-question-step-by-step)
- [6. The RAG decisions, and why each one](#6-the-rag-decisions-and-why-each-one)
- [7. Grounding, and how it fails visibly](#7-grounding-and-how-it-fails-visibly)
- [8. Evidence that the numbers are real](#8-evidence-that-the-numbers-are-real)
- [9. Privacy, tracing, and observability](#9-privacy-tracing-and-observability)
- [10. Security posture, including what is missing](#10-security-posture-including-what-is-missing)
- [11. Honest limitations](#11-honest-limitations)
- [12. Questions you will be asked](#12-questions-you-will-be-asked)

---

## 1. What you are presenting

A local, single-user web app that answers questions **only** from documents you
upload, cites the exact passage behind every claim, and tells you when it does
not know.

Three properties are the project. Present them in this order, because each one
earns the next.

1. **It does not invent.** A relevance floor decides whether the model is even
   asked. If nothing in your documents covers the question, you get an honest
   refusal, not a fluent guess.
2. **It shows its work.** Every answer carries the passages used, which ones the
   model actually cited, and what it cost. Citations survive a reload.
3. **It stays out of the way.** One command, no cloud account, no key required for
   a default install, documents never leave the machine.

The demo in [section 4](#4-the-demo-step-by-step) is built to show all three in
about four minutes.

## 2. Decide who is listening

Two audiences, two emphases. Both use the same code.

| Audience | Lead with | Why |
|---|---|---|
| **Judges / reviewers** | Grounding guarantees, evaluation numbers, security limits | They will try to break the claim that it is reliable. Show the refusal, not the best answer. |
| **Users / classmates** | The interface, the citations, the source count | They will ask "how is this better than pasting a PDF into ChatGPT?" |

If the session is short, present the two guarantees and one limitation. A project
that names its own failure mode is more credible than one that claims none.

## 3. Run it

### 3.1 Two ways to run it

**Everything in containers** (Postgres, pgvector, and the app):

```bash
docker compose up -d
```

Open <http://127.0.0.1:8000>. The compose file has a healthcheck, so the app
does not start until the database accepts connections.

**Or Postgres in Docker, app on the host** — this is the one to use when
presenting, because you can restart the server and read logs immediately:

```bash
docker compose up -d db
.venv/bin/python -m pip install -e .          # first run only
.venv/bin/python -m app.ocr --install         # once, for scanned PDFs
.venv/bin/python run.py
```

Open <http://127.0.0.1:8000>. Note the different database host: `localhost` for a
local run, `db` inside compose.

The app runs on loopback by default, because it has no authentication and
`0.0.0.0` would expose every session to the network. Set `HOST` deliberately if
you need it on the LAN. See [section 10](#10-security-posture-including-what-is-missing).

### 3.2 What must be true, and how to check it

Do this **before** the presentation, not during it.

```bash
curl -s localhost:8000/api/status | python3 -m json.tool
```

This endpoint reports every knob that changes behaviour, so it is also the
cheapest way to answer "what is it actually configured to do?" Check four things:

| Field | Expect |
|---|---|
| `provider` | `groq` if `GROQ_API_KEY` is set, otherwise `none`. Without a key the app still starts and refuses to answer, rather than failing at startup and leaving you guessing |
| `embed_model` | `all-MiniLM-L6-v2` — runs locally, no embedding API key |
| `min_score`, `top_k`, `max_per_source`, `hybrid`, `numeric_damping` | The retrieval knobs from section 6 |
| `tracing` | An object: `enabled: false` unless you are deliberately demoing it |

A 200 from this endpoint also proves the database answered, since the response
includes live source and chunk counts.

### 3.3 The configuration you may need to explain

| Variable | Default | Why it matters |
|---|---|---|
| `DATABASE_URL` | `postgresql://notebooklm:notebooklm@localhost:5432/notebooklm` | Sessions, messages, chunks, vectors all live here. `db:5432` under compose |
| `UPLOAD_DIR` | `./data/uploads` | Where uploaded files are written, and the boundary for deletion |
| `GROQ_API_KEY` | unset | Generation. Without it, retrieval still works and answering refuses |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Runs locally; no embedding API key or cost |
| `MIN_SCORE` | `0.25` | The relevance floor. Raising it makes the app more cautious |
| `TOP_K` | `6` | Passages per question |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Splitting granularity, evaluated in [section 8](#8-evidence-that-the-numbers-are-real) |

Full table: `README.md`. Every knob there is a decision someone can challenge, so
know which ones you tuned and why.

### 3.4 Before you present: three things

1. `git status` clean, on the commit you intend to present.
2. `./run_tests.sh` green. If it is not, do not present.
3. One notebook with 3+ sources already uploaded, so the demo has content and you
   are not indexing during the talk. Keep a copy under `experiments/corpus/`.

## 4. The demo, step by step

Assumes a notebook named **Handover** with three sources: a release checklist, a
support policy, and a CSV of incident counts.

**Step 1 - empty state (5 seconds).** Open a fresh notebook. Point at the heading,
the `0 sources` count, and the two onboarding cards - upload, and web search.
Say: *it tells me there is nothing to ask yet, instead of offering questions it
cannot answer - and it offers both ways of fixing that.*

**Step 2 - upload (30 seconds).** Upload all three files. Watch the source count
go to 3. Say: *PDF, DOCX, Markdown, CSV, text. OCR for scanned pages, and a document
that yields no text is refused with a reason instead of indexing empty.*

**Step 3 - a question it can answer (45 seconds).** Ask:

> What is the notice period for termination?

The answer arrives with `[1]`, `[2]`. Click a marker: the cited passage opens,
and cards the model did not use are dimmed. Say: *the dimmed ones are the point -
you can see what was retrieved and what was actually used.*

**Step 4 - the evidence strip (15 seconds).** Point at the confidence chip, the
best score, and the token cost. Say: *every answer states its own evidence and its
own cost, so I can audit the answer and the bill at the same time.*

**Step 5 - a question it cannot answer (45 seconds).** This is the most important
step. Ask:

> What is our policy on sabbatical leave?

No `[n]` markers. Best match below the floor, model not called, cost $0. Say:
*it refused. If I had sent that to a model, I would have gotten a confident answer
invented from the wrong documents - and citations on top, which is worse, because
they look verified.*

**Step 6 - the near-miss (30 seconds).** Ask something adjacent but not present,
like a policy detail that exists in a different document. Say: *the floor is the
tuning knob. Too low and it chatters; too high and it goes quiet. 0.25 is measured
to separate relevant from irrelevant on this embedding model.*

**Step 7 - provenance survives (20 seconds).** Reload the page. Reopen the
notebook. The cited answer still has working citation cards, because the
citations are stored with the message, not just in the HTTP response. Say: *this
was a real bug - the `[1]` markers survived a reload while the passages they
pointed at did not.*

**Step 8 - a greeting (15 seconds).** Type `Hi`. Say: *no embedding, no
retrieval, no fake relevance score - it distinguishes "I could not find it" from
"that was not a question". The reply itself is the model's, not a fixed line: it
has nothing to be faithful to, but it is still generated, and the cost is
reported honestly.*

**Step 9 - web sources (45 seconds, only if you want the risk).** Open a fresh
notebook, click *Search the web for sources*, and type a topic. Say: *this is the
one feature that changes the privacy story - the search term goes to Groq and this
machine fetches the pages it finds. The page is then indexed exactly like an
upload, so it is citable and deletable, rather than an uncitable paragraph the
model wrote.* Then stop, or say: *the fetch path resolves and refuses private
addresses on every redirect, because a URL from a search engine can name
`169.254.169.254`.* Only do this step with a topic you are willing to have logged
by a third party.

**Step 10 - what it does not do (10 seconds).** Say: *no audio or video overviews,
no mind maps, no multi-user accounts.* Being explicit about the gap is what makes
the rest of the list believable.

## 5. What happens to a question, step by step

This is the answer to "walk me through it". Say each stage out loud; the numbers
come from `app/answering.py`.

```
POST /api/ask {"question": "...", "session_id": "..."}
   │
   ├─ 1. Validate            schema; empty question -> 400
   ├─ 2. Classify            app/intent.py. "Hi" -> conversational, stop here.
   ├─ 3. Empty check         no sources -> "add a source", stop here
   ├─ 4. Retrieve            hybrid dense + BM25, store.search()
   ├─ 5. Filter              drop chunks below MIN_SCORE = 0.25
   │                            nothing survives -> no_match, model NOT called
   ├─ 6. Diversify           at most MAX_PER_SOURCE = 3 chunks per source
   ├─ 7. Build context       passages as SOURCES; prior turns as HISTORY
   │                            (labelled not-a-source, placed first)
   ├─ 8. Generate            Groq, temperature 0, instruction to cite [n]
   ├─ 9. Validate citations  out-of-range refs stripped and reported
   │                            numbers checked against supplied passages
   ├─ 10. Audit              ungrounded = uncited, or cited an unsupplied number
   ├─ 11. Persist            message + citations + evidence -> Postgres
   └─ 12. Return             answer, citations, evidence
```

Two properties of that flow are worth stating explicitly:

- **Steps 4-5 are the gate.** Everything after them is conditional. No passage
  above the floor means no generation, no tokens, no chance of invention.
- **Step 11 exists because of a bug.** Provenance was response-only and did not
  survive a reload. It is stored now.

## 6. The RAG decisions, and why each one

Each row is a decision someone can challenge. Know the reason, and know what you
would change if the workload were different.

### 6.1 Parsing

**Decision.** Parse per format; OCR scanned PDFs inline; refuse a document that
yields no text. Refuse rather than partially index.

**Why.** A silently empty index is worse than a refusal, because the first
question against it returns "not in your sources" for the wrong reason. Refusal
carries the reason, including the OCR-language hint when OCR is the cause.

**Trade-off.** Inline indexing blocks the event loop, and OCR costs about a
second per page, capped at 50 pages. A production version moves this to a queue.
Stated in [section 11](#11-honest-limitations).

**Where the article's ingestion checklist is met, and where it is not.** The
article lists five ingestion concerns. Being explicit about which are handled
prevents overclaiming:

| Article's concern | Status here |
|---|---|
| Extraction from multiple sources | 21 formats, per-format parsers (`app/parsers.py`) |
| Content normalisation | Unicode and whitespace cleanup; table reshaped to `Label: Value` |
| Metadata enrichment | Name, kind, page, numeric-density, chunk index — but no NER or topic tagging |
| Incremental updates | **Not implemented.** Re-uploading re-indexes the file; there is no CDC or hash-based skip |
| Deduplication | **Not implemented.** The same file uploaded twice is indexed twice |
| Ingestion observability | Parse failures and refusals carry reasons; there is no ingestion metrics dashboard |

The two gaps are the same gap the article warns about: *without observability,
retrieval degradation goes undetected.* A document that silently fails to parse
looks exactly like a document that was never added. That is why a parse failure
is loud — it refuses with a reason — but a *successful* parse that extracted the
wrong thing is not detectable today. Fixing that means asserting on extracted
content at ingest time, which is the same "validate the input, don't trust it"
pattern used on answers.

### 6.2 Chunking

**Decision.** ~900 characters, 150 overlap.

**Why.** Not guessed - it was chosen by measurement over a 110-label corpus
(`experiments/eval_gold.py`). 900 characters preserves all 110 labels
(`fact_coverage=0.890`) while keeping passages small enough that 6 of them fit
the context without crowding out each other. See [section 8](#8-evidence-that-the-numbers-are-real).

**Trade-off.** Larger chunks would fit more context but dilute retrieval; smaller
chunks retrieve precisely and break facts across boundaries. 900 is the
compromise that the measurement supported, not a round number.

### 6.3 Embeddings

**Decision.** `all-MiniLM-L6-v2`, 384 dimensions, running locally in pgvector.

**Why.** No API key, no cost, no document text leaves the machine, and 384
dimensions keep the whole vector corpus small enough that a full exact scan is
measurable in milliseconds rather than needing an approximate index. For a
single-user notebook, exact search beats a vector index: recall is not the
bottleneck.

**Trade-off.** MiniLM is weaker than a hosted embedding model on long or technical
text. The relevance floor is calibrated to its scores (~0.30-0.45 relevant vs
~0.05-0.15 irrelevant), so **changing the embedding model means re-measuring
`MIN_SCORE`**. That dependency is worth stating before anyone swaps it.

**On vector database choice.** The article frames this as build / buy / extend,
with the real decision being the ANN index. This project extends Postgres with
pgvector, and the honest answer is that **the index choice has not been forced
yet**. An HNSW index exists, but at this corpus size Postgres does not reliably
use it: it picks HNSW on a cold plan and a sequential scan once the plan is
warm, because at ~1,000 rows both cost about the same (measured: ~1.0 ms
sequential vs ~1.3 ms through the index). So "exact search is used" would be
too strong - what is true is that **the exact path is available and costs
nothing to take**. For a single-user notebook that is the right trade: recall
and legibility, not latency, are the constraint. Exact search is O(N) and
becomes infeasible at millions of vectors; forcing HNSW would be the next step,
and IVF-PQ only beyond what fits in memory. Stating this as a decision that has
not yet been made is more accurate than claiming a scale the corpus does not
require.

### 6.4 Retrieval

**Decision.** Hybrid: dense vectors plus BM25 keyword search, fused.

**Why.** The two fail differently. Dense retrieval handles paraphrase
("notice period" vs "how much warning"); BM25 handles exact terms, names,
identifiers, and numbers that vectors blur. A notebook full of policy documents
and incident CSVs needs both, and the CSV case is not hypothetical.

**Trade-off.** Fusion adds a tuning parameter and a second failure mode. It is
worth it here because keyword-exact failures produce confidently wrong answers.

### 6.5 The relevance floor

**Decision.** `MIN_SCORE = 0.25`. Below it, refuse without calling the model.

**Why.** The single most important decision in the project. Without a floor, a
vague question still produces a confident, cited answer built from irrelevant
passages - and the citations make it *look* verified, which is worse than an
uncited guess because it defeats checking.

**Trade-off.** A floor that is too high makes the app frustratingly quiet. This
one is deliberately cautious: a refusal is recoverable, a wrong answer is not.

### 6.6 Context assembly

**Decision.** `TOP_K = 6`, `MAX_CONTEXT_CHARS = 14000`, at most
`MAX_PER_SOURCE = 3` chunks per source.

**Why.** Six passages is enough to answer a multi-part question without letting
the model choose which facts to ignore. The per-source cap stops one long
document from crowding out the rest of the notebook. The character cap is a
safety net: `TOP_K` is a count, and chunks vary in size.

**Trade-off.** The per-source cap is a preference, not a limit. If the cap leaves
fewer than 6 passages - which always happens with a single source - the
remaining slots backfill from the next-best chunks. Mention this if asked,
because it is the kind of detail that looks like an oversight.

### 6.6a Two knobs that do nothing, on purpose

`MIN_RATIO` raises the floor to a fraction of the best hit, and
`NUMERIC_PENALTY_SHARE` penalises number-dense chunks. Both are implemented, both
appear in `/api/status`, and the sensitivity sweep shows **neither changes a
single decision on the current corpus**.

They are precautions, not validated wins, and the README says so. Mention them
if someone asks what is over-engineered - being able to say "this one is a
precaution, and here is the measurement showing it is not carrying its weight" is
a better answer than quietly leaving dead configuration in place.

### 6.6b Query rewriting: not implemented, deliberately

The article lists five rewriting techniques — clarification, context injection,
expansion, keyword normalisation, noise reduction — and warns about over-expansion
and semantic drift.

This project rewrites queries in exactly one case, and by rule rather than by
model: a chat message can carry two instructions, and `highlight <word>` is
stripped before retrieval (`app/highlightIntent.py`, mirrored in
`highlightIntent.js`). Without it, "what is the late penalty? highlight 30 days"
searches for the *words* "highlight 30 days" as well as the question, which
dilutes the dense query and adds noise to BM25.

**Why no model-based rewriting.** It costs an extra inference call on every
question, it can be talked out of its own instruction, and it introduces
semantic drift — the failure mode where the rewrite changes what was asked. The
article's own advice applies: measure first, then introduce, then re-measure. No
end-to-end quality gain from either has been measured here: `rerank_eval.py`
shows a cross-encoder separating relevant from trap queries on a hand-built set,
but nothing ties that separation to a better answer, and [section
8.4](#84-what-these-numbers-can-and-cannot-tell-you) is where the limits of the
measurements that do exist are stated. Adding one would be an unmeasured cost.

**The parser is conservative on purpose.** It only accepts a message that is
*entirely* a highlight request. "How do I highlight a word?" is a question about
the feature, and "highlight the difference between the two papers" is not a
request to find that phrase. A real question silently downgraded to a word
search is a wrong answer; a highlight that has to be typed into the box is a
missing feature. The asymmetry decides the design.

### 6.6c Re-ranking, hierarchical, graph and agentic RAG: none of these

The article's maturity path is chunking → embeddings → retrieval metrics →
re-ranking → multi-stage → hierarchical/graph → agentic. This project stops at
retrieval metrics.

| Technique | Status | Why |
|---|---|---|
| Re-ranking (cross-encoder) | Not implemented | Needs a second model and a latency budget; no measured headroom yet |
| Hierarchical RAG | Not implemented | Corpus is 21 short documents; there is no long-form structure to descend |
| Graph RAG | Not implemented | Would need entity extraction and a graph store; no cross-document reasoning requirement exists yet |
| Agentic loops | Not implemented | Every question here is a single lookup; an iterative loop would add latency and cost for no measurable gain |

This is the article's own argument applied honestly: *advanced techniques should
not be implemented prematurely, and must be justified by complexity.* A notebook
that answers "what is the refund window" does not need a reasoning loop, and
shipping one would be a worse answer than the question.

The one thing this project did instead is make the absence visible:
`MIN_RATIO` and `NUMERIC_PENALTY_SHARE` exist in `/api/status` and change nothing
measurable ([6.6a](#66a-two-knobs-that-do-nothing-on-purpose)).

### 6.7 History

**Decision.** Server-side history from Postgres, marked *not a source*,
placed before the sources block.

**Why.** Two reasons. Provenance: if history came after sources it could be
mistaken for evidence. Integrity: there is no `history` field on `/api/ask`, so
a client cannot submit turns it never asked, and `messages.role` is constrained
to `user | assistant`, so no `system` turn can be stored to compete with the
grounding rules.

### 6.8 Generation

**Decision.** Groq `openai/gpt-oss-120b`, temperature 0, instruction to cite
`[n]`. OpenAI and Ollama fallbacks exist.

**Why.** Temperature 0 because the task is extraction and citation, not
creativity - variance would only make failures harder to reproduce. Provider
indirection because a hosted model should not be a hard dependency for a local
app.

**Trade-off.** No retry and no automatic provider failover on a generation
failure. Both are defensible product decisions, and both would muddy a module
whose job is to make one policy legible.

## 7. Grounding, and how it fails visibly

The claim is: *this app never quietly invents a fact.* The mechanism is four
server-side rules plus an audit.

| Rule | Prevents |
|---|---|
| Relevance floor | Confident answers from irrelevant passages |
| Citation validation | References to passages that were never supplied |
| History is not evidence | Prior turns being cited as sources |
| Cross-source diversity | One document dominating the evidence |

**Citation validation.** The model is told to cite `[1]`, `[2]`. A prompt is not
a guarantee, so out-of-range references are stripped and reported in
`evidence.invalid`, and CJK bracket variants are normalised to ASCII - some models
emit `【1】`, and a strict pattern would misreport a well-cited answer as uncited.

**The audit.** `ungrounded` is true when the answer cited nothing, or cited a
number that was never supplied. That check is why the numeric-heavy corpus
matters: a confident wrong figure with a citation to a real passage is the
failure mode a user is least likely to catch.

**Failure is visible, not silent.** The UI dims citation cards the model never
used, warns when a reference was removed, and prints the score floor and cost on
every refusal. A grounding failure you can see is recoverable. One you cannot is
not.

**Three verdicts.** `answered`, `no_match`, and `conversational`. The third exists
because "Hi" used to come back as *best match 0%, below the 25% floor* - true,
and meaningless, since there was nothing to score against. A greeting is not a
failed retrieval, and an empty notebook should be told to add a source rather
than be given a score about zero documents.

The `conversational` verdict is a routing decision, not a canned reply. In a
notebook with sources, a greeting goes to the model through `llm.chat()` with the
history and no passages: there is nothing to ground and nothing to cite, but the
reply is generated, and it is billed honestly like any other call. That is what
separates it from the fixed line this replaced, and it is the same reason ChatGPT
answers "hi" instead of printing a template. Two cases stay local and free - an
empty notebook, where there is no subject to be helpful about, and a provider
that is unreachable, where a fallback beats an error page.

`app/intent.py` matches the **whole message**, and that is the load-bearing
detail: "hey, what is the refund window?" opens with a social word and is a real
question. A keyword scan would greet it and drop the query. Every doubt resolves
towards retrieval, because the errors are asymmetric - a wrongly greeted question
gets a useless reply, while a wrongly searched "hi" costs one embedding and gets
an honest refusal.

## 8. Evidence that the numbers are real

Claims without numbers are marketing. These are reproducible.

### 8.1 Chunking (`experiments/eval_gold.py`)

900 characters preserves all 110 gold labels, `fact_coverage=0.890`. Bakeoff
results: `experiments/results/chunker_bakeoff.json`.

### 8.1b Retrieval accuracy

At `MIN_SCORE=0.25`, over the committed corpus: **79/79 relevant questions
answered, 42/42 traps rejected**, across 17 formats. Relevant questions score at
worst 0.126; the worst trap scores 0.000.

Ranked metrics over the same corpus, `k=5`: **recall 0.956, MRR 0.962, nDCG
0.935, precision 0.226, hit rate 1.000, fact coverage 0.923, trap leaks 0**.
Read what bounds them before quoting them
([8.4](#84-what-these-numbers-can-and-cannot-tell-you)).

Two honest notes about this number:

- **It is 79/79, not 84/84.** An earlier run reported five failures. They were not
  retrieval failures: the harness picked its "real-world PDF" as the largest file
  in the upload directory, which by then held an unrelated test fixture, and scored
  that against questions written for a different document. The path is now pinned
  via `EVAL_REAL_PDF`, and the run prints `real-world PDF: SKIPPED` when it is
  absent. Say this if the numbers come up — a measurement of the harness, reported
  as if it were a measurement of the retriever, is exactly the mistake worth
  volunteering.
- **The corpus is deliberately small** (~130 questions, 21 documents). It varies
  format and phrasing on purpose, so it catches "tuned to one document" failures.
  It is still small, so treat these as regression alarms rather than a benchmark.

### 8.2 Generation (`experiments/eval_generation.py`)

Measured on the documented sample:

| Metric | Score | Reads as |
|---|---|---|
| Faithfulness | 0.917 | Answers stay within supplied passages |
| Citation precision | 1.000 | Every cited marker pointed at a real passage |
| Substance | 0.725 | Answers were substantive |
| Trap refusal | 1.000 | Off-topic questions were refused, not answered |
| Relevancy | 0.167 | **Low - be ready to explain this** |

**Say this before anyone asks:** these are deterministic lexical metrics, not an
LLM judge. They are regression alarms, not proof of quality. Relevancy is low
because lexical overlap is the wrong instrument for judging whether an answer
addresses the question - the trap-refusal and citation-precision numbers carry
the real signal, and a human reads 20 outputs per release. Presenting 0.167 as
quality would be the fastest way to lose the room.

### 8.3 Tests

`./run_tests.sh` — 103 backend, 166 frontend. Includes: relevance floor,
citation validation, injection defence, history injection attempts, upload
ownership, citation persistence across reload, greeting behaviour, and web-source
ingestion (a found page becomes a citable source, one unreachable page does not
discard the rest, and a redirect into private address space is refused *before*
it is requested).

That last one is worth mentioning unprompted, because it is the check that would
have caught the obvious bug: letting `httpx` follow redirects means the request
to the private address has already been sent by the time you can look at the
final URL. The test asserts on the URLs actually requested, not on the error
message.

**The ownership test exists because of a real bug.** The evaluation harness
indexed `experiments/corpus/` through the normal store, and deleting an
evaluation session unlinked committed corpus files. Uploads are now only
deletable inside `UPLOAD_DIR`. Mention this unprompted if you discuss testing:
it is a better argument for the suite than a passing count.

**Second example, same shape.** The retrieval harness chose its real-world PDF by
file size in the upload directory. It silently scored a test fixture against
another document's questions and reported five retrieval failures. The lesson is
the same as the upload bug: tests and evaluations are code, and they fail
silently when they are wrong.

### 8.4 What these numbers can and cannot tell you

Recall@K, MRR, nDCG and Precision@K — the four the article asks for — are
implemented in `experiments/metrics.py`, run by `experiments/eval_gold.py`, and
reported below. The honest caveat is not that they are missing. It is that the
labels they grade against are machine-checked rather than read by a person, and
that is what bounds the numbers.

| Metric in the article | Status here | Current value |
|---|---|---|
| Recall@K | Implemented | `0.956` @5 |
| MRR | Implemented | `0.962` @5 |
| nDCG | Implemented | `0.935` @5 |
| Precision@K | Implemented | `0.226` @5 |
| Hit rate@K | Implemented | `1.000` @5 |
| Fact coverage | Implemented | `0.923` |
| Trap leak rate | Implemented | `0.000` |
| Faithfulness / groundedness | Implemented | `0.917` — fraction of answers with no unsupported claim |
| Answer relevance | Implemented | `0.167`, explicitly reported as not fit for purpose |
| Citation precision | Implemented | `1.000` |
| Trap refusal | Implemented | `1.000` |
| Answer correctness vs a reference answer | **Not implemented** | None |
| Per-format breakdown | Partial | 17 formats, aggregated |

Source: `experiments/results/cloze_set_metrics.json` (124 items, 117 answerable,
7 traps, `k=5`, `human_verified: false`).

**What actually bounds these numbers: the labels, not the metrics.**
`experiments/gold/cloze_set.json` is built by taking a line from a corpus
document, blanking the value, and using the filled value as the answer.
Correctness is decidable without domain knowledge, which is exactly why no
expert review is required — and exactly the limitation: the gold answer is the
document's own text. These numbers prove a value survives chunking attached to
its key and can be found again. They do not prove the system understands
questions. Both committed result files say so themselves, in a field nobody
edited: `"human_verified": false`.

**Why the hand-written gold set has never been scored.**
`experiments/gold/gold_set.json` holds 48 items with hand-written questions and
proposed evidence. Every one is still `verified_by_human: false`, and
`eval_gold.py` refuses to write a results file for an unverified set. The gate
is deliberate: the evidence in that file was proposed by lexical overlap, and
quoting a number computed from it would be the same circularity called out in
[8.5](#85-why-recallk-is-hard-here-and-where-the-work-is).

**What to add first, if there is time.** Read those 48 items against their
source documents and flip `verified_by_human`. One review pass converts the
retrieval claims in [8.1b](#81b-retrieval-accuracy) from "nothing above the
floor was rejected" into "the right passage was in the top K, at this rank",
graded against labels a reader checked. Reference answers per question would
then close the last row of the table above — that one is a labelling exercise,
and it is still not done.

### 8.5 Why Recall@K is hard here, and where the work is

A reference implementation was studied while writing this section
(`ArmishRao/RAG_EVAL`). Its six retrieval metrics are correct and worth reading,
but its method is circular in a way worth understanding, because it is the
obvious way to build this and it does not work:

- Ground truth is produced by retrieving the **top 6 chunks from the very
  retriever being graded**, then asking an LLM to mark a subset of those six as
  relevant.
- The retriever is then scored on whether it retrieved that subset.

So `expected_chunks` is, by construction, a subset of what it already returned.
**A retrieval miss cannot be detected by this method, because a chunk that was
never retrieved can never appear in the expected set.** Its `Recall@6 = 0.875`
measures the LLM judge's approval rate of the retriever's own output. That is
not a criticism of the code — the metrics are implemented correctly — it is a
warning about the method.

The same repo also reports a security scorecard and a LangSmith integration in
its README, with no code behind either. Worth remembering when reading any
benchmark: check that the artefact exists.

**The rule this project follows instead:** every number in this document comes
from a committed artefact that regenerates on demand, and a metric whose
meaning cannot be stated in one sentence is either fixed or reported as
unfit. That is why relevance is `0.167` and labelled as such — a low number
honestly labelled beats a high number nobody can define.

## 9. Privacy, tracing, and observability

Default posture: **uploaded documents do not leave the machine.** Embeddings run
locally; only question text and retrieved passages go to the configured LLM
provider, which is unavoidable for generation and is stated in the UI.

**The one exception, stated rather than buried: web search.** If `WEB_SEARCH` is
on (the default) and a provider that can search is configured, the *search term*
goes to that provider and this machine fetches the pages it points at. Fetched
pages are indexed and stored locally like any upload, but the fetch itself is a
request to a site named by a search engine answering a model. `WEB_SEARCH=0`
turns it off, `/api/status` reports availability with a reason, and the UI card
greys out rather than failing when clicked. Be ready to name this unprompted -
"documents stay local" is only true of documents you uploaded.

Tracing is opt-in and split by sensitivity:

| Variable | Default | Sends |
|---|---|---|
| `LANGSMITH_TRACING` | `false` | Span names, timings, scores, verdicts, token counts |
| `LANGSMITH_TRACING_INCLUDE_TEXT` | `false` | Document text and model answers |
| `LANGSMITH_TRACING_LOCAL` | `false` | Same spans to a local JSONL file, no network |

Two switches, not one, because span metadata and document content are different
risks. `/api/status` reports what is enabled, so you can prove the setting rather
than claim it.

Retrieval and generation are separate spans, so a slow answer is attributable.
Failures close their span and tracer errors never break a request - observability
that can take the app down is not observability. A refusal still opens a span,
because "what did we refuse and why" is the question you most want answered in
production.

Verified locally with 5 spans across answered and refused questions, with no
document text present. LangSmith network delivery was not live-tested; say that
rather than implying both were.

**Ingestion observability** covers parse failures and refusals, which carry
reasons. It does not cover a parse that *succeeds and extracts the wrong thing*,
which is currently undetectable: the document looks indexed, and questions about
it simply find nothing. That is the gap the article calls "retrieval degradation
going undetected", and it is the honest limit of what the current monitoring
claims.

### Cost, and what is recorded

**Cost = tokens x price, and tokens come from the API response, not an estimate.**
`usage.py` prefers the provider's own count and only falls back to a
character-length approximation, flagged `estimated: true` when it does. An
estimate shown as a measurement is a claim about precision the system does not
have.

Two events cost money, and they are recorded in two places because they are
different shapes:

| event | where recorded | why |
|---|---|---|
| answer / greeting / summary | `messages.evidence` | belongs to the turn, reads back with the transcript |
| web search | `usage_events` (migration `0004`) | not a turn - no message appears in chat when a page is fetched |

`/api/status` returns a `usage` object that sums both. Before `0004`, search
cost was computed, returned in the response, and dropped - so any total was
short by every search ever run, and short in a way that *looked* complete.

**Tokens, not dollars.** A price table goes stale, and a dollar figure implies
tokens mean the same across Groq, OpenAI and local Ollama. Tokens plus model
name stays true when prices change. If a dollar figure is wanted, compute it
offline from the persisted tokens rather than baking it into the app.

**Verified prices (Oct 2026):** `gpt-oss-120b` $0.15 in / $0.60 out per 1M;
`gpt-oss-20b` (web search) $0.075 / $0.30; MiniLM embeddings $0 - local.

**The two numbers to report:**

```
cost per question    = total tokens spent / questions asked
cost per 100 pages   = $0   (MiniLM is local)
```

**Honest caveat to state first:** this runs on Groq's free tier (200K
tokens/day), so real spend is **$0**. Any dollar figure is a list-price
equivalent, not a bill.

## 10. Security posture, including what is missing

**Do the honest audit first.** It prevents the "does it have auth?" question from
catching you off guard.

Present these as deliberate:

| Control | Why |
|---|---|
| Loopback by default | No auth means `0.0.0.0` would expose every session |
| Prompt rule: sources are data, never instructions | The actual injection defence |
| Injection detection over retrieved passages | Makes a suspicious source visible and names it |
| `messages.role` constrained to `user | assistant` | No stored `system` turn can compete with grounding rules |
| No client-supplied history | A client cannot fabricate turns it never asked |
| Uploads streamed, capped at 100 MB | Bounded disk use |
| Provider key server-side | Never reaches the browser |
| Web fetch refuses private/loopback/link-local addresses, on every redirect hop | A URL from a search engine can name `169.254.169.254` or this machine's own database. Redirects are followed by hand precisely so the check happens *before* the next request, which `httpx` cannot do for you |
| Web fetch caps size, time, pages and redirect hops | An untrusted response should not be able to exhaust disk or a worker |

Present these as gaps:

- **No authentication.** Anyone who can reach the app owns every session in it.
  Single-user by assumption, not by enforcement. Adding auth is required before
  this is exposed to anyone else - see `README.md` for the recommended shape.
  Web search makes this sharper: an unauthenticated app that can fetch URLs is an
  SSRF endpoint with an index attached.
- **Injection detection is a regex** over retrieved passages. It surfaces a
  suspicious source; it cannot block a determined injection. The prompt rule is
  the real defence, and the regex is evidence gathering.
- **Inline indexing** means a large PDF blocks the server for as long as it takes.
- **Markdown subset** rendering, so raw HTML, images, and tables from the model
  display literally.
- **No streaming** - the answer appears at once.
- **No governance layer.** The enterprise requirements in the article — row-level
  security, chunk-level access control, data masking, tenant isolation, audit
  logging — are all absent. That follows from single-user scope rather than
  oversight: there is no notion of "who" to govern. Per-chunk metadata is stored
  but never used for access filtering.

## 11. Honest limitations

State these unprompted. Each has a reason, and naming it is stronger than being
found out.

| Limitation | Why it exists / when it would change |
|---|---|
| No audio, video, or mind-map overviews | NotebookLM's headline feature is out of scope. Deliberate: those features dominate effort while adding no grounding |
| Web search sends the search term to a third party and fetches arbitrary pages | The one feature that breaks the local-only promise. Off with `WEB_SEARCH=0`; SSRF-guarded, but DNS rebinding is not fully closed and there is no auth, so it is not for a public network |
| Combined highlight metadata is not persisted | Question plus highlight is executed client-side; the original highlight is lost on reload |
| Citations before migration `0002_message_evidence` are not clickable | The `[n]` markers are in the message text, but the provenance was never stored. Re-ask, or accept the limitation. Not recoverable - the data does not exist |
| Long or technical documents may retrieve poorly | MiniLM is a small local model. Re-measure `MIN_SCORE` if you swap it |
| Single-word queries that are also small talk | `perfect`, `nice`, `ok`, `great` are treated as small talk. A user searching for one of those terms gets a greeting |
| No deep links to the original file page | A citation scrolls to the passage inside the app only |
| Highlight intent uses a heuristic | When no passage overlaps the highlight, a query is still derived. Better than doing nothing; not a substitute for a real selector |
| Large documents block the server | Needs a job queue |
| Re-uploading a file re-indexes it; no hash-based skip | No incremental ingestion or dedup, so the same document counted twice drags a source into every answer twice |
| Answer correctness against a reference answer | Would need reference answers per question, which is a labelling exercise; see [8.4](#84-what-these-numbers-can-and-cannot-tell-you) |
| Retrieval metrics graded on machine-checked labels | The metrics are implemented; the labels have never been read by a person. `gold_set.json` is 0/48 verified |
| Ground-truth corpus is small | ~110 labels, so chunking numbers are directional for larger corpora |

## 12. Questions you will be asked

**"How is this different from pasting a PDF into a general chatbot?"**
Because it can refuse. A general model always answers; if your documents do not
cover a question, it fills the gap from pretraining and attaches citations to its
own invention. Here the model is not called at all below the relevance floor, and
you can see the score and the cost of every decision.

**"What if retrieval returns the wrong passage?"**
Then the relevance floor and the audit are the backstops: dimmed unused
citations, `evidence.invalid` for references that were never supplied, and
`ungrounded` for answers citing numbers nobody provided. Not bulletproof - the
defence in depth is what matters.

**"Why 0.25 and not 0.5?"**
Because it was measured on this embedding model: relevant chunks score
~0.30-0.45, irrelevant ~0.05-0.15. 0.25 sits in the gap. And these scores are
embedding-specific - change the model, re-measure the floor.

**"Why no auth?"**
Single-user, local, loopback-only by default. It is the honest scope for the
project rather than a hidden omission, and adding it is documented. If it were
deployed, auth would be the first change.

**"Your Recall@K is 0.956. Isn't that suspiciously high?"**
It is high, and the reason is the label set, not the retriever. The labels come
from blanking a value out of a document line, so the question literally contains
the passage. The metric is implemented correctly and the run is reproducible
([8.4](#84-what-these-numbers-can-and-cannot-tell-you)) — what it does not
measure is whether a person would find the answer useful. The check that would
settle it is the 48 hand-written questions in `gold_set.json`, none of which has
been verified yet, so I do not quote a number from them.

**"Isn't computing Recall@K from your own top-k circular?"**
Yes, and this project does not do it. The failure mode is described in
[8.5](#85-why-recallk-is-hard-here-and-where-the-work-is): if ground truth is
selected from what the retriever already returned, a miss is undetectable by
construction. Here the gold set is derived from the *documents*, independent of
what was retrieved, so a miss can and does score zero.

**"Your relevancy score is 0.167. Isn't that bad?"**
That is a lexical metric being used outside its competence, and the honest answer
is that it is not a quality measure. Faithfulness 0.917, citation precision 1.000,
and trap refusal 1.000 are the numbers that speak to grounding, and a human
reviews outputs per release. I would rather show you the weak metric and explain
it than hide it.

**"How do you know it is not just keyword matching?"**
The test at `test_app.py::test_a_real_question_that_opens_with_a_greeting_still_retrieves`
asks "hey, what is the refund window?" - which must retrieve, not be greeted.
Hybrid retrieval is the other half: dense handles paraphrase, BM25 handles exact
terms.

**"What would you do next?"**
Authentication, then a job queue for indexing so large documents stop blocking
the server, then streaming. In that order: security, then correctness under load,
then feel.