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

**Step 1 - empty state (5 seconds).** Open a fresh notebook. Point at the greeting,
the `0 sources` count, and the two onboarding cards. Say: *it tells me there is
nothing to ask yet, instead of offering questions it cannot answer.*

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

**Step 8 - a greeting (10 seconds).** Type `Hi`. Say: *no embedding, no token, no
fake relevance score. It distinguishes "I could not find it" from "that was not
a question".*

**Step 9 - what it does not do (10 seconds).** Say: *no audio or video
overviews, no mind maps, no web sources, no multi-user accounts.* Being explicit
about the gap is what makes the rest of the list believable.

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
yields no text.

**Why.** A silently empty index is worse than a refusal, because the first
question against it returns "not in your sources" for the wrong reason. Refusal
carries the reason, including the OCR-language hint when OCR is the cause.

**Trade-off.** Inline indexing blocks the event loop, and OCR costs about a
second per page, capped at 50 pages. A production version moves this to a queue.
Stated in [section 11](#11-honest-limitations).

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
dimensions keep the whole vector corpus small enough to search exactly rather
than approximately. For a single-user notebook, exact search beats a vector
index: recall is not the bottleneck.

**Trade-off.** MiniLM is weaker than a hosted embedding model on long or technical
text. The relevance floor is calibrated to its scores (~0.30-0.45 relevant vs
~0.05-0.15 irrelevant), so **changing the embedding model means re-measuring
`MIN_SCORE`**. That dependency is worth stating before anyone swaps it.

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

`./run_tests.sh` — 93 backend, 158 frontend. Includes: relevance floor,
citation validation, injection defence, history injection attempts, upload
ownership, citation persistence across reload, and greeting behaviour.

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

## 9. Privacy, tracing, and observability

Default posture: **documents do not leave the machine.** Embeddings run locally;
only question text and retrieved passages go to the configured LLM provider,
which is unavoidable for generation and is stated in the UI.

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

Present these as gaps:

- **No authentication.** Anyone who can reach the app owns every session in it.
  Single-user by assumption, not by enforcement. Adding auth is required before
  this is exposed to anyone else - see `README.md` for the recommended shape.
- **Injection detection is a regex** over retrieved passages. It surfaces a
  suspicious source; it cannot block a determined injection. The prompt rule is
  the real defence, and the regex is evidence gathering.
- **Inline indexing** means a large PDF blocks the server for as long as it takes.
- **Markdown subset** rendering, so raw HTML, images, and tables from the model
  display literally.
- **No streaming** - the answer appears at once.

## 11. Honest limitations

State these unprompted. Each has a reason, and naming it is stronger than being
found out.

| Limitation | Why it exists / when it would change |
|---|---|
| No audio, video, or mind-map overviews | NotebookLM's headline feature is out of scope. Deliberate: those features dominate effort while adding no grounding |
| Web search unimplemented, shown disabled in onboarding | A local-only build. Removing the card would read as a missing feature rather than a deliberate boundary |
| Combined highlight metadata is not persisted | Question plus highlight is executed client-side; the original highlight is lost on reload |
| Citations before migration `0002_message_evidence` are not clickable | The `[n]` markers are in the message text, but the provenance was never stored. Re-ask, or accept the limitation. Not recoverable - the data does not exist |
| Long or technical documents may retrieve poorly | MiniLM is a small local model. Re-measure `MIN_SCORE` if you swap it |
| Single-word queries that are also small talk | `perfect`, `nice`, `ok`, `great` are treated as small talk. A user searching for one of those terms gets a greeting |
| No deep links to the original file page | A citation scrolls to the passage inside the app only |
| Highlight intent uses a heuristic | When no passage overlaps the highlight, a query is still derived. Better than doing nothing; not a substitute for a real selector |
| Large documents block the server | Needs a job queue |
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