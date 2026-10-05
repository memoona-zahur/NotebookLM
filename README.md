# Local NotebookLM

A local, source-grounded research assistant. Upload documents, ask questions, get
answers with inline citations that link back to the exact source passage.

## Stack

| Piece | Choice |
|---|---|
| API | FastAPI |
| Database | PostgreSQL 16 with `pgvector` |
| Migrations | Alembic |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (runs locally) |
| LLM | Groq (`openai/gpt-oss-120b`) by default, or OpenAI / Ollama |
| Frontend | React 18 + Vite, compiled to static assets |
| OCR | Tesseract via PyMuPDF, for scanned PDFs |
| Deployment | Docker Compose |

One Postgres holds everything persistent: sessions, source records, chunk text,
chunk vectors, and chat history. No separate vector database, so there is one
store to back up and one thing to reason about.

## Run with Docker

```bash
cp .env.example .env       # paste your GROQ_API_KEY into .env
docker compose up --build
```

Open http://127.0.0.1:8000

The `db` service health-gates the app, so the app never starts against a
database that cannot accept queries. Alembic applies pending migrations on boot,
so an existing volume is upgraded in place and a fresh one needs no manual step.

The app image is a two-stage build: Node compiles the React bundle, then only
Python and the compiled assets reach the runtime, so no toolchain ships to
production.

The build also fetches Tesseract's language data so the container reads scanned
PDFs without extra setup. Add `--build-arg OCR_LANGUAGES="eng deu"` for other
languages, or `--build-arg OCR_LANGUAGES=none` to skip the download entirely
(needed behind a firewall; scans are then refused with instructions). A failed
download fails the build rather than shipping an image that silently cannot
read them.

Data survives `docker compose down`: source vectors live in the `pgdata`
volume and uploaded files in the `uploads` volume. Use
`docker compose down -v` to delete both.

## Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, proxies /api to :8000
npm run build        # writes the bundle to ../static
```

Run the API on port 8000 alongside `npm run dev` and the Vite proxy handles the
rest. For production, `npm run build` and `python run.py` is enough — the
compiled bundle is committed, so there is no Node requirement at runtime.

## Run without Docker

You need a PostgreSQL 16 server with the `vector` extension available
(`pgvector`):

```bash
docker compose up -d db      # just the database, if you prefer to run the app locally
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # then paste your GROQ_API_KEY into .env
.venv\Scripts\python run.py
```

Linux / macOS:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then paste your GROQ_API_KEY into .env
.venv/bin/python run.py
```

Point `DATABASE_URL` in `.env` at your server if it is not on
`localhost:5432` with the default credentials.

Fetch the OCR language data so scanned PDFs work too - the Docker image already
has it, but a local run does not:

```bash
.venv/bin/python -m app.ocr --install
```

Skipped, scans are refused with the same instruction in the error message rather
than indexed as empty. See [Scanned documents](#scanned-documents-are-read-not-refused).

To reach the dev server from another machine on the LAN, set `HOST` — it binds
loopback by default, since the app has no auth and `0.0.0.0` would expose
every session to the network:

```bash
HOST=0.0.0.0 .venv/bin/python run.py
```

The venv is strongly recommended: `sentence-transformers` pulls in PyTorch, and the
default wheel is the CUDA build (~6 GB). On a CPU-only machine, install the CPU
wheel instead:

```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
```

Open http://127.0.0.1:8000

## Tests

The suite needs a real Postgres: it creates a scratch database, runs against it,
and drops it, so it can never touch real data.

```bash
docker compose up -d db
.venv/bin/python test_app.py
```

To use a database elsewhere, set `TEST_DATABASE_ADMIN_URL` and
`TEST_DATABASE_NAME`.

The OCR checks skip themselves when no language data is installed, and the
check that a missing install produces the right message runs instead - so the
suite is meaningful both with and without it. `python -m app.ocr --install` makes
all of them run.

### Frontend

```bash
cd frontend
npm install
npm test           # 122 checks, jsdom, no database or API key needed
npm run test:watch # re-runs on save
```

These mock `fetch` rather than the `api` module, so URL building, `detail`
unwrapping and the 503 message are covered rather than mocked away. A
`TypeError` from `fetch` is treated as "server unreachable" and asserted
separately from an HTTP failure.

### Everything

```bash
./run_tests.sh
```

Starts a scratch Postgres if none is reachable, runs both suites plus a build,
and leaves the database volume alone.

## Migrations

The schema is owned by Alembic in `migrations/`. The app runs `alembic upgrade
head` on boot, which is a no-op when the database is already current.

```bash
alembic upgrade head                        # apply everything pending
alembic current                             # which revision this database is on
alembic history --verbose                   # the full chain
alembic downgrade -1                        # step back one revision
alembic upgrade head --sql                  # print the SQL instead of running it
```

To adopt Alembic on a database created before it existed, stamp it rather than
rebuilding:

```bash
alembic stamp head
```

Migrations are hand-written rather than autogenerated. The app uses psycopg
directly instead of an ORM, so there is no SQLAlchemy metadata for
`--autogenerate` to compare against; each revision states its change in plain
SQL, which is reviewable before it reaches a database.

`migrations/env.py` reads the DSN from `app.db.dsn()`, the same source the
application uses, so a migration cannot be pointed at a different database than
the app by accident.

## Configuration

Everything is set in `.env` (all optional):

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | – | Enables the Groq provider; auto-detected |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | |
| `OPENAI_API_KEY` | – | Fallback provider |
| `OLLAMA_MODEL` | `llama3.1` | Used when no API key is set |
| `LLM_PROVIDER` | auto | Force `groq` / `openai` / `ollama` |
| `DATABASE_URL` | `postgresql://notebooklm:notebooklm@localhost:5432/notebooklm` | Use host `db` under Compose |
| `DB_POOL_MAX` | `8` | Connection pool size |
| `UPLOAD_DIR` | `./data/uploads` | Where originals are kept for re-indexing |
| `MAX_UPLOAD_MB` | `100` | Upload ceiling; larger files get HTTP 413 |
| `BASE_URL` | derived from the request | Set when the browser cannot infer the API origin |
| `CHUNK_SIZE` | `900` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | |
| `TOP_K` | `6` | Passes retrieved to the model |
| `MAX_CONTEXT_CHARS` | `14000` | Truncation guard |
| `MIN_SCORE` | `0.25` | Relevance floor - see below |
| `MAX_PER_SOURCE` | `3` | Max chunks from one source, so answers can span sources |
| `HISTORY_TURNS` | `6` | Prior turns read for follow-ups (`0` disables) |

## Sessions

Everything lives inside a session: its own sources and its own chat history. Create
one from the sidebar, switch between them, and two sessions never see each other's
documents or transcript. History is stored server-side, so a chat survives a refresh
or a different browser.

## API

Every route below accepts an optional `session_id` query parameter and acts on that
session; without one it uses the default session.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/sessions` | List sessions |
| `POST` | `/api/sessions` | Create a session (`{name}`) |
| `GET` | `/api/sessions/{id}` | One session plus its stored transcript |
| `PATCH` | `/api/sessions/{id}` | Rename |
| `DELETE` | `/api/sessions/{id}` | Delete it with its sources and history |
| `POST` | `/api/sessions/{id}/messages/clear` | Clear the transcript, keep sources |
| `GET` | `/api/status` | Provider, model, source/chunk counts, tuning |
| `POST` | `/api/sources` | Upload a file (multipart `file`) |
| `DELETE` | `/api/sources/{id}` | Remove one source |
| `DELETE` | `/api/sources` | Clear the session's sources |
| `POST` | `/api/ask` | `{question}` → `{answer, citations, evidence}` (`evidence.cost` carries tokens and latency, see below) |
| `POST` | `/api/summarize` | `{instruction}` → `{summary, citations, evidence}` |
| `GET` | `/api/occurrences` | `{term}` → every place this session's documents use the word |

If the LLM backend is unreachable, `/api/ask` returns `503` with a message telling you
what to fix instead of a raw 500. An upload over `MAX_UPLOAD_MB` returns `413` and
the partial file is deleted, so a rejected upload leaves nothing behind.

## Scanned documents are read, not refused

A PDF with no text layer - a phone photo of a receipt, a scan - used to upload
successfully, appear in Sources, and contribute zero chunks. The assistant then
answered from the other documents as if that file were not there, which is the
worst possible outcome: a confident answer with no hint that evidence was
missing.

Refusing it was better than indexing nothing, but it was still wrong: a scan is a
real document that happens to be stored as an image, and its text is
recoverable. Scanned pages are now rasterised and run through OCR (Tesseract,
embedded in PyMuPDF), and the recovered text indexes and cites like any other
source.

OCR is on by default and costs about a second per page, so it is bounded by
`OCR_MAX_PAGES` (50). A document with more image pages than that is refused
rather than partly indexed, with the number to raise named in the message: a
partial index that reports success is the failure this replaced. A mixed
document - a typed report with a scanned cover - indexes both halves, with the
pages kept in their original order so a citation still points at the right
place.

OCR needs Tesseract's language data, which is tens of megabytes per language and
therefore not committed:

```bash
python -m app.ocr --install          # fetch eng.traineddata into data/tessdata
python -m app.ocr --install deu fra  # add languages
python -m app.ocr --status           # what is available, and where
```

The Docker image fetches it at build time, so a container handles scans out of
the box. When the data is missing, a scanned upload is refused with the exact
command that fixes it. A scan that OCR reads nothing from - a blank page, a photo
too dark or low-contrast to recognise - is refused the same way, since there is
nothing to index; the `.txt` you get by exporting it is still the better source.

Settings live in `.env`: `OCR_ENABLED`, `TESSDATA_DIR`, `OCR_LANGUAGES`,
`OCR_DPI`, `OCR_MAX_PAGES`.

## Treating your documents as data, not instructions

Everything in a source goes into the prompt, so a document can address the model
the way you can. Three layers address that:

1. The system prompt states that passages are untrusted data and that text which
   looks like a command is something to report on, never to obey.
2. A line that fakes a conversation turn (`assistant:`, `user:`, `system:`) has
   its colon rewritten, so it cannot read as a real turn. The text is altered
   rather than dropped, which keeps the citation matching the stored document.
3. `detect_injection` names the passages that try to issue instructions, and the
   numbers come back on `evidence.injection` alongside the citations.

Detection is not the defence. A regex can be worded past, and a determined
injection will reach a model that reads it. Layer 1 is what actually holds;
layer 3 exists so a suspicious source is visible rather than silent. The model
is asked to say in one clause when a document contains instructions aimed at it,
so a surprising answer can be traced back to the file that asked for it.

## How answers are rendered

Answer text is rendered as a small Markdown subset: bullet and numbered lists,
headings (shifted down two levels so an answer cannot outrank the question
above it), blockquotes, emphasis, inline code, fenced code blocks, and links.
`[1]` citation markers still become chips that scroll to their passage.

Nothing goes through `dangerouslySetInnerHTML`, and there is no Markdown
dependency. Every source document is untrusted and can end up quoted inside an
answer, so HTML in model output has to be inert - `frontend/src/markdown.jsx`
builds elements directly, which makes that structural rather than a rule someone
has to remember. Raw HTML, images and tables are unsupported and render as the
literal text the model wrote. Link URLs are restricted to `http`/`https`; a
`javascript:` or `data:` URL keeps its label but is not followed.

The prompt asks for plain text and bullets only. Asking a model for Markdown and
then stripping what it produced is how an answer ends up full of literal `**`.

## Finding a word in the documents

`GET /api/occurrences?term=velocity` returns every place the active session's
documents use a word, with the exact character spans of each hit:

```json
{
  "term": "velocity",
  "count": 412,
  "truncated": true,
  "occurrences": [
    {
      "source": "handbook.pdf",
      "kind": "pdf",
      "position": 3,
      "page": 12,
      "heading": "",
      "text": "Escape velocity at the surface is 11.2 km/s.",
      "matches": [[7, 15]],
      "count": 2
    }
  ]
}
```

The spans are the contract. They are computed once, server-side, from the same
chunk text the client renders, so the highlight can never disagree with what the
server reported. Marking them in place is therefore always safe: a span that
does not fit the text is skipped rather than slicing the string backwards.

Matching is case-insensitive but word-bounded, so `rate` does not light up
inside `generate`. Results are capped at 40 passages and `truncated` says so,
because a common word like `the` appears in nearly every chunk of a large
document, and a silently short list would read as "that is all of them".

In the UI this is reachable two ways, both running the same search: type the
word into the **Highlight a word** box in the Sources drawer, or say
`highlight velocity` (also `where does velocity appear`, `mark every mention of
…`) in the composer.

A recognised command opens the drawer with the word marked and never reaches the
model — it is a lookup, not a question. Anything the parser does not recognise
falls through to `/api/ask` untouched.

### Two instructions in one message

A message can be both a question and a highlight request, and the two halves are
handled separately:

```
what is the return window? highlight 30 days
```

This asks the question *and* marks every occurrence of `30 days`. The answer
appears in the thread while the drawer is already open on the word, because the
highlight search does not depend on the answer.

Only the question is sent to the server. `highlight 30 days` inside the message
would otherwise be embedded into the dense query and scored by BM25 as if it were
evidence the user wanted to find, which biases retrieval toward passages about
highlighting rather than about return windows — and the noise is unrecoverable
downstream, because nothing later can tell a model that half the question was an
instruction. The user still sees exactly what they typed in the thread; only the
model receives the stripped question.

Parsing is deliberately conservative, in `frontend/src/highlightIntent.js`:

- The split only happens on an explicit separator (`?`, `,`, `and`, `then`, new
  line) followed by a recognised highlight verb. A question that merely contains
  the word "highlight" — *"how do I highlight a citation?"* — is left alone, as is
  *"which section should I highlight for the summary?"*
- A question that ends in a question mark with no highlight clause is left alone.
- Only the first clause is treated as the question, and only when the remainder
  begins with a highlight verb.

The failure modes are asymmetric. A missed split means the highlight does not
happen and the full message is asked, which is recoverable by typing `highlight
…` again. An over-eager split silently discards part of a real question, so the
parser is biased towards not splitting.

One thing that is deliberately **not** handled: a highlight term that itself ends
in a conjunction, such as `highlight 30 days and 2%`. The term parser rejects a
trailing `and`, because it cannot tell that conjunction from the one the question
splitter uses, so the whole message is asked instead and nothing is highlighted.
Accepting it would mean guessing where the term ends and where the question
resumes, and guessing wrong sends the wrong half to the model. A second explicit
separator such as `also highlight` would resolve it without guessing.

## Grounding guarantees

A source-grounded assistant is only worth anything if it never quietly invents a fact.
Four rules enforce that, all server-side:

1. **Relevance floor (`MIN_SCORE`).** Retrieval drops any chunk scoring below the floor.
   If nothing clears it, the LLM is **never called** - the response is a plain
   "not in the sources" message with `verdict: "no_match"`. Without this, a vague
   question still produces a confident, cited, wrong answer from irrelevant passages.

   Measured on `all-MiniLM-L6-v2`: relevant chunks score ~0.30-0.45, irrelevant ones
   ~0.05-0.15. `0.25` separates them. Raising it makes the assistant more cautious;
   lowering it makes it chattier but reintroduces guessing.

2. **Citation validation.** The model is told to cite `[1]`, `[2]`, ... but a prompt is
   not a guarantee. Any reference outside the range of passages actually supplied is
   stripped from the answer and reported in `evidence.invalid`. CJK bracket variants
   (`【1】`, `［１］`) are recognised and rewritten to ASCII, because some models emit
   them and a strict pattern would misreport a well-cited answer as uncited.

3. **History is context, never evidence.** Prior turns are passed in a block explicitly
   labelled *not a source*, placed **before** the SOURCES block so the passages that may
   be cited are the freshest thing in the context window. The transcript is read from
   Postgres, not from the request: there is no `history` field on `/api/ask`, so a client
   cannot supply turns it never asked. The `messages` table also constrains `role` to
   `user | assistant`, so no `system` turn can be stored to compete with the grounding
   rules.

4. Cross-source diversity.** At most `MAX_PER_SOURCE` chunks come from any one source,
   so a single long document cannot crowd out the rest of the notebook. This is a
   *preference*, not a limit: if the cap leaves fewer than `TOP_K` passages (which happens
   whenever the notebook holds a single source), the remaining slots are backfilled with
   the next-best chunks regardless of origin.

Every `/api/ask` response includes an `evidence` object:

```json
{
  "verdict": "answered",
  "confidence": "high",
  "best_score": 0.5114,
  "min_score": 0.25,
  "considered": 12,
  "returned": 6,
  "passages": 6,
  "cited": [1, 3],
  "invalid": [],
  "ungrounded": false
}
```

`ungrounded` is `true` when the answer cited nothing, or cited a number that was not
supplied. The UI renders this as a confidence chip, dims citation cards the model never
used, and shows a warning when a reference had to be removed - so a grounding failure is
visible instead of silent.

## Handling tables and numbers without losing data

A PDF table page extracts as bare numbers with no headers:

```
78773 1364 55002.93597 3.5 12 0   79635 2226 55003.52309 11.5 52 0   82759 5350 ...
```

Embedded as prose this is semantically flat, so it matches almost anything. An earlier version
of this project **dropped** such chunks at ingest. That fixed the symptom on one PDF and
silently destroyed every CSV, JSON export, log file and metrics dump uploaded afterwards.

The fix is structural, not destructive. `app/parsers.py` extracts each format the way that
format is actually read, so numbers arrive with their labels attached and are no longer
semantically flat:

| format | extraction |
|---|---|
| CSV / TSV | header-aware: `revenue_usd: 412300 \| region: EMEA` |
| JSON | flattened to JSON paths: `$.limits.rpm: 600` |
| logs | grouped into events, heading is the event timestamp |
| source code | split per definition, heading is the symbol path (`Retry.attempt`) |
| Markdown | heading path tracked, pipe tables rendered as key/value rows |
| DOCX | heading styles tracked, tables rendered with their header row |
| YAML / TOML / INI | flattened to key paths: `$.database.pool_size: 40` |
| PDF | per page, with table-like pages split on line boundaries |
| hard-wrapped TXT | rejoined at the line break, so a phrase split by column wrapping stays searchable |

Chunks that are still numeric-flat (a PDF table with no extractable headers) are **flagged,
not dropped**. They are damped by `NUMERIC_DAMPING` at retrieval so they cannot outrank prose,
and the damping is skipped when `NUMERIC_PENALTY_SHARE` of the candidates are already
numeric, so a corpus that is entirely numbers keeps working with its ranking untouched.

Table-like text is also chunked on line boundaries. Splitting it on word boundaries used to
produce fragments like `003.52309 11.5 52 0`, which match nothing and read as gibberish when
cited.

## Retrieval: hybrid dense + BM25

`app/lexical.py` implements Okapi BM25 with a tokenizer built for code, and the two rankings
are combined with reciprocal rank fusion. RRF is used because the retrievers are not on a
comparable scale - dense cosine sits around 0.3-0.7 while BM25 is unbounded - so any mixing
weight would be a constant tuned to one corpus.

The relevance *decision* still runs on the dense score, because that is the only scale
`MIN_SCORE` was ever calibrated against. Three routes admit a chunk:

| route | condition | fixes |
|---|---|---|
| `dense` | dense score >= `DENSE_STRONG` | confident answers pass untouched |
| `both` | dense >= `MIN_SCORE` **and** coverage >= `COVERAGE_MIN` | rejects weak-but-related matches |
| `lexical` | coverage >= `COVERAGE_MIN` and dense >= `BM25_RESCUE_MIN` | terse keyword queries the embedding under-scores |

*Coverage* is the IDF-weighted share of the query's distinctive terms present in a chunk. It
is what makes the two directions work, measured on the eval corpus below:

- An unrelated question scores **0.252** densely against a physics handbook, purely
  on vague topical similarity, but has **0.00 coverage** - no term overlap - so it is refused.
- `acronyms expanded` scores **0.237** against a document that states it explicitly, but has
  **1.00 coverage**, so it is answered.

A caller that passes a floor stricter than `MIN_SCORE` disables the `lexical` route entirely,
and a floor of `0` disables relevance gating altogether.

## Evaluating retrieval

```powershell
.venv\Scripts\python experiments\eval_retrieval.py
```

Creates one session per document, then builds **20 documents across 15 file formats
and 19 unrelated subjects** - brewing,
pharmacology, glaciology, catalysis, contract law, music theory, horticulture, football,
mining, agriculture, retail data, IoT telemetry, ML configs - plus any real PDF in
`data/uploads`, and reports per-question verdicts, per-format aggregates, a threshold
sweep, and a sensitivity sweep over every tuning constant.

Both axes of variety are deliberate. A corpus that varies file type but keeps one subject
will happily agree with a system tuned to one document. Questions come in polite and terse
keyword forms, and every document gets its own trap questions.

Current result at `MIN_SCORE=0.25`: **84/84 relevant answered, 45/46 traps rejected**,
with 16 of 17 formats at 100% recall and 100% trap rejection.

The sensitivity sweep re-queries the whole corpus per value, so a flat row means the
constant is not load-bearing and a swinging row means the default is a guess. It shows
`MIN_RATIO` and `NUMERIC_PENALTY_SHARE` currently change nothing measurable - they are
precautions, not validated wins, and are documented as such.

### Does chunking lose the answer?

Retrieval quality is only half the question. The other half is whether a fact
survives being cut into chunks and re-found, so there is a second harness:

```powershell
.venv\Scripts\python -m experiments.build_cloze_gold --write
.venv\Scripts\python -m experiments.eval_gold --gold experiments/gold/cloze_set.json --allow-machine
```

Each item blanks a value out of a real corpus document and asks for it back, so
the gold answers are the documents' own text rather than anything invented.
The set holds **124 items across all 27 corpus documents**: 33 field, 24
record, 8 definition, 31 value, 21 multi-block, and 7 traps that must be refused.
The multi-block items need two or three separate passages, which is what makes
them able to fail - an earlier set of single-passage items could not tell a
working ranker from a lucky one.

The definition items exist because three prose documents were missing entirely.
`catalysis.txt`, `glaciology.pdf` and `music.txt` have no field names and few
quotable numbers, so every value-shaped cloze returned nothing and the set
claimed to cover the corpus while skipping three of its files. A definition is
the one shape prose does have: `Till is the unsorted, unstratified sediment
deposited directly from the ice.` The cue carries the meaning, so the subject is
recoverable and the same four mechanical checks apply. They are weaker labels
than a value cloze - they prove the defining sentence came back, not that the
term was understood - and they are capped hard: no multi-word gerund phrases
("binding too weakly" is answered by anything), nothing longer than four words,
and the cue has to open the sentence's first clause so a subordinate clause is
never mistaken for a name.

Current result: recall@5 **0.956**, fact coverage **0.923**, MRR **0.962**,
nDCG **0.935**, traps leaked **0**. Precision@5 is **0.226**, which is the
point of the traps: the alternative to retrieving a sixth passage that happens
to contain the word is retrieving five that do not.

`--allow-machine` is required, and it is a real caveat rather than a formality.
These labels are **machine-checked, not human-reviewed**: every item satisfies
the mechanical checks in `experiments/build_cloze_gold`, and `eval_gold` re-runs
them before scoring anything, which makes the set reproducible and impossible to
score stale. It does not make it expert-checked. What it proves is that a value
survives chunking attached to its key and can be retrieved again - not that the
system understands the question. `test_app.py` asserts the labels can actually
fail, so a passing suite is not the set agreeing with itself.

Three corpus documents are still uncovered: `catalysis.txt`, `glaciology.pdf`,
`music.txt`.

## Does the answer stay faithful to the passages?

Retrieval can be perfect and the answer still wrong: the model reads the right
chunk, misreads it, or blends it with something it remembered. `eval_gold`
measures whether the evidence reached the prompt. This measures what came back
out, using the same labels.

```bash
.venv/bin/python -m experiments.eval_generation --limit 12 --dry-run --allow-machine
```

Three numbers, each answering a different question:

| Metric | Question it answers |
|---|---|
| `faithfulness` | Is every cited claim traceable to the passage it cites? |
| `citation_precision` | Are the citations load-bearing, or decoration? |
| `substance` | Did the answer cover the labelled facts, tolerating rewording? |
| `relevancy` | Did it land every labelled fact, word for word? |
| `verbatim` | How much does the answer quote rather than paraphrase? |

### No judge model, on purpose

An LLM judge is the usual way to score these and it was rejected here. It costs
money per evaluation, it is non-deterministic, and it is the same family of model
as the one being graded, so its errors correlate with the errors being measured.
A metric that fails for the same reason the system fails cannot be used to detect
that failure. `experiments/metrics.py` makes the same argument for its definitions.

The cost of this choice is real: these are lexical proxies, not entailment
judgements. They can be fooled by a close paraphrase that changes the meaning and
cannot detect entailment that shares no words. They are a floor and a regression
alarm, not proof of grounding. Both metrics modules are written out in full
rather than imported from a library, because a definition you cannot read is a
definition you cannot argue with.

### Why there are three relevance numbers, and how that was found

The first version matched required facts against the answer as literal
substrings, exactly as `fact_coverage` does for chunks. It scored **0.000 on six
out of six correct answers**. The labels are whole source sentences — *"mash at
sixty-six degrees Celsius favours a balanced body"* — and the model answers
*"a single-infusion mash at 66 °C favours a balanced body"*. Correct, and scored
zero for using digits.

Literal matching is right for chunks and wrong for answers. The chunk text *is*
the chunk text; a good answer paraphrases and normalises units. So relevance
counts content words with numbers and units canonicalised, and reports three
strictnesses:

```
substance / relevancy / verbatim bracket the same answers from three levels.
A high substance with a low relevancy means the answer reworded the evidence
correctly. A high relevancy with a low substance means the labels are too
short to distinguish anything.
```

A tolerant metric alone is also wrong, in the dangerous direction. Scored on
"most content words present", the answer *"the limit is 600 requests per hour"*
passes against a fact reading *"the limit is 600 rpm"* — a confidently wrong
answer scoring 1.0. So `relevancy` requires **every** content word of a fact,
which rejects it and still accepts the digit-and-unit differences that are
formatting rather than knowledge.

### What the numbers look like

At `--limit 12` against the cloze gold set: `faithfulness 0.917`,
`citation_precision 1.000`, `substance 0.725`, `relevancy 0.167`,
`verbatim 0.000`, trap refusal `1.000`.

`verbatim 0.000` is the headline caveat, and it is why the strict number is not
the headline: **the labels in this gold set are whole source sentences, and no
generated answer reproduces one.** Scoring against them measures paraphrase, not
quality. The strict number is kept because it is the signal that catches a changed
unit, not because it is a score to quote.

The one genuine generation failure in that run was `G004`: *"These differences
are described in the source passage [1]"* — a claim with no checkable content, two
of them uncited. It scores `faithfulness 0.00` and `uncited 2`, which is exactly
what the metric exists to surface.

### Refusals are never averaged into faithfulness

A correct refusal has no claims, so scoring it 0.0 would make the safest possible
behaviour look like the worst. Refusals are counted separately, and the verdict
comes from the server rather than from matching a phrase like "I could not" —
so rewording the refusal cannot silently turn it into a hallucination.

`SUPPORT_THRESHOLD` is a judgement call, so the harness re-scores every answer at
several thresholds and prints how many claims sit near the boundary. A constant
nobody has tested is a constant that will be "tuned" by whoever next feels like
changing it.

### Known retrieval limits

- **Entity-overlap traps leak.** Asked a pharmacology document "which antibiotics treat
  tuberculosis", it scores 0.54 and answers. The document genuinely is about antibiotics,
  so the shared entity is real and both dense and BM25 score it highly. 4 of 10 such traps
  are caught. This is the one gap a cross-encoder reranker is designed for, and it is the
  only evidence that would justify adding one.
- `MIN_SCORE` is a single global value. Relevant and unanswerable scores overlap, so it is
  a conservative default rather than a guarantee; callers needing stricter behaviour pass a
  higher `min_score` per query.
- `MIN_RATIO` and `NUMERIC_PENALTY_SHARE` are unvalidated by the current corpus.

## What a question costs

Every `/api/ask` response carries a `cost` object next to `evidence`, and the UI
shows it in the same strip as the confidence chip. That placement is deliberate:
one answers *should I believe this*, the other answers *was this worth asking*.
A well-cited answer can still be a wasteful one.

```json
"cost": {
  "model": "openai/gpt-oss-120b",
  "verdict": "answered",
  "passages_sent": 1,
  "context_chars": 99,
  "retrieval_ms": 18.0,
  "generation_ms": 1179.6,
  "total_tokens": 526,
  "prompt_tokens": 461,
  "completion_tokens": 65,
  "estimated": false,
  "called": true
}
```

Four decisions here, and what they reject:

**Latency is split, never totalled.** Retrieval is local CPU work and generation is a
network call. They fail for unrelated reasons, and a single `total_ms` hides which one
was slow: a slow answer from disk needs better retrieval, a slow answer from the
network needs a different provider or a smaller context. One number cannot tell them
apart, so there are two.

**A refusal reports zero tokens as a measurement, not as an absence.** When the
relevance floor rejects a question the model is never called, and `called: false` with
zero tokens is the finding. It also means "how often do we refuse" is answerable from
the response itself. Omitting the block instead would make a free answer and a
broken measurement look identical.

**Token counts come from the provider, and an estimate says so.** `prompt_tokens` and
`completion_tokens` are read off the provider response. When a provider reports none
(older Ollama builds, some proxies) the count falls back to a character-length
approximation and flips `estimated: true`, which the UI renders as `(est.)`. An
estimate shown as a measurement is a claim about precision the system does not have,
and it is worse than no number because it looks equally trustworthy in both the good
and the bad case.

**Prompt tokens dominate, and that is the point.** In the example above the question
is about fifteen words and the prompt is 461 tokens. What you pay for is the retrieved
context plus the system prompt, not the question. This is why `context_chars` and
`passages_sent` are reported alongside the tokens: the lever that reduces cost is
sending less evidence, which is a retrieval decision, not a generation one.

`total_tokens` is reported rather than converted to currency. A dollar figure would
need a price table that goes stale, and would imply the number means the same thing
across Groq, OpenAI and a local Ollama. Tokens plus the model name stays true when
prices change.

## How grounding works

1. Files are parsed per format by `app/parsers.py` (39 suffixes, see above) into blocks that
   carry a page number and a heading, then split into overlapping chunks on paragraph,
   sentence, line or word boundaries depending on content.
2. Numeric-flat chunks are flagged `numeric_heavy`. **Nothing is discarded.**
3. Each chunk is embedded with MiniLM and inserted into `chunks` with its text, page,
   heading, and a `vector(384)` column searched through an HNSW cosine index. A BM25
   index over the same chunks is built on demand and cached in memory per session,
   rebuilt only when that session's chunk count changes.
4. A question is embedded and cosine-searched, and separately scored by BM25. The candidate
   set is the **union** of both, over-fetching 4x so the per-source cap has choices, and the
   two rankings are fused with RRF.
5. Candidates are admitted via the three routes above. Numeric-flat chunks are damped.
   Surviving chunks must clear both `MIN_SCORE` and `MIN_RATIO` of the best hit, then the
   per-source cap is applied and backfilled. The survivors are numbered and injected into a
   system prompt that forbids outside knowledge and requires `[n]` citations.
6. The model's citations are validated against the passages supplied, and the UI renders
   each retrieved chunk below the answer as a collapsible card with source, page, match
   score, and full text. An inline `[n]` in the answer jumps to its card and flashes it;
   a card the model never actually cited is dimmed rather than hidden, so a weak answer
   is visible as weak.

## Notes

- Embeddings never leave your machine. Only the retrieved chunks are sent to the LLM.
- `all-MiniLM-L6-v2` is a 384-dim, ~80MB model tuned for English. For multilingual or
  higher-quality retrieval, swap in `EMBED_MODEL` and update `EMBED_DIM` in
  `app/config.py` to match.
- `MIN_SCORE` was calibrated against MiniLM's score distribution. A different embedding
  model shifts those numbers, so re-check it before trusting the default.
- `EMBED_DIM` is duplicated: as a `vector(N)` column in the migration, and as
  `EMBED_DIM` in `app/config.py`. Changing the embedding model means changing
  both, plus a new migration, since an existing column cannot be altered in place.

## Tests

76 checks covering upload, per-format parsing (including YAML/TOML/INI and hard-wrapped
text), OCR (a scanned page recovered and searchable, pages kept in order, both halves of a
mixed PDF indexed, a blank scan refused with an actionable message, the missing-language
message naming the install command, an over-cap document refused rather than indexed
partly, and a normal PDF untouched by any of it), an oversized upload refused at 413 with
the partial file removed, chunking, hybrid retrieval,
numeric damping, citations, per-question cost reporting (a refusal recorded as
having spent nothing, and a provider's own token count preferred over an estimate),
generation metric definitions, the 503 LLM-down path, Groq routing, BOM handling, the
built frontend resolving every asset it references, migrations (schema at head,
idempotency, column-for-column agreement with the app, and the `role` check and
`ON DELETE CASCADE` surviving), sessions (CRUD, source scoping, retrieval isolation,
transcript persistence, cascading deletes, and surviving a restart), and the
grounding guarantees above (relevance floor, citation validation, history
hardening, cross-source diversity). Prompt-injection handling has its own three
checks: that a document issuing instructions is flagged rather than obeyed, that a
passage cannot fake a prompt role, and that the system prompt still carries the
rule - the last one because an instruction nothing asserts can silently be edited
away. The LLM is stubbed, so no API key is needed to run them.

A failed check no longer stops the run: every check is reported and the process exits
non-zero if any failed.

## Known limitations

- **A vacuous claim can be cited and still be unfaithful.** The metric catches
  *"These differences are described in the source passage [1]"* because there is no
  checkable content in it, but a longer sentence that mixes one real fact with one
  invented clause can clear the lexical threshold. Lexical proxies cannot tell
  which part of a sentence is wrong, and this is the honest edge of the approach.

- Migrations are hand-written. There is no ORM, so `--autogenerate` cannot
  detect drift; the tests assert the migrated columns match what the code reads,
  but a column added to `app/db.py` and forgotten in a revision still fails at
  runtime rather than at migrate time.
- Uploads are kept on disk so a source can be re-indexed without re-uploading. Each delete
  path cleans up its own files; a hard crash mid-session can still leave an orphan.
- No reranking. Fused dense + BM25 order goes straight to the prompt. See the entity-overlap
  note above for the one case that measurably needs a cross-encoder.
- PDF tables are **not reconstructed**. Every other table format is parsed with its header row
  attached (`.csv`/`.tsv`, Markdown pipe tables, DOCX), but a PDF only yields positioned text
  runs, so a table there stays numeric-flat. Those chunks are flagged rather than dropped, and
  detecting the header row across the ruled lines is still open work.
- A cross-encoder reranker was evaluated and **is not wired in**: on the test handbook its
  scores for legitimate questions (0.002-0.597) overlapped those for unanswerable ones
  (0.000-0.125), so it cannot serve as an absolute relevance threshold. It may still help
  rank passages, which needs the eval harness to confirm. Reproduce with
  `.venv/bin/python experiments/rerank_eval.py`.
- Uploads are streamed to disk in 1 MiB chunks and capped at `MAX_UPLOAD_MB` (100 MB
  by default), but indexing still runs on the event loop, so a large PDF blocks the
  server for as long as it takes.
- OCR of scanned pages costs roughly a second per page and runs inline during indexing,
  bounded by `OCR_MAX_PAGES` (50); a document over the cap is refused rather than indexed
  partially. Handwriting and very low-contrast photocopies need `tessdata_best` dropped
  into the folder, since the bundled models are the fast ones.
- A blank scan, or one OCR recognises nothing in, is still refused - with the command to
  install language data if that was the cause.
- Sessions persist in Postgres, but there is no auth: anyone who can reach the app owns
  every session in it. Single-user by assumption, not by enforcement.
- Answers render a Markdown subset, so raw HTML, images and tables from the model show
  literally. There is no streaming either - the answer appears at once.
- Prompt-injection detection is a regex over retrieved passages. It makes a suspicious
  source visible and names it, but it cannot block a determined injection; the prompt
  rule is the actual defence.
- A citation scrolls to the passage inside the app but cannot deep-link to the page of the
  original file.
