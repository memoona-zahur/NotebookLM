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

### The interface is meant to look like NotebookLM

The UI is a deliberate attempt at the real product's feel: a notebook rail with
search on the left, tappable starter questions instead of an empty pane, sans
type throughout, a single blue accent, and citations that are the accent colour so
provenance and emphasis are the same thing.

This replaced an earlier design and the reasoning is worth recording, because the
earlier version was not accidental. It used warm "paper" surfaces, a terracotta
accent, and a serif for anything the model wrote, on the argument that a document
you are reading should look like paper on a desk. That argument was sound about
reading and it was overridden about resemblance: warm paper and a serif is exactly
the set of choices that can never make something feel like the reference product,
and feeling like the product was the actual requirement. The discarded position is
not in `tokens.css` any more, since a token block is read by people editing the
app and a rejected position is only useful to whoever wonders why the obvious
choice was avoided.

**What the design gives up.** Long answers are slightly less pleasant to read than
they were in the serif version. That is a real cost, paid knowingly, and it is the
first thing to revisit if reading comfort turns out to matter more than resemblance.

### What it does not copy, on purpose

- **NotebookLM's generated artifacts.** Audio overviews, video overviews and mind
  maps are the reference product's most distinctive feature and this app has none
  of them. They are a separate model pipeline, not styling.
- **Real logo.** The rail uses a rounded square in the accent colour. A shape in
  the brand colour is not a trademark question; the logo itself would be.
- **The generated-artifact chip row** above the composer, which would only ever
  render disabled here.
- **The "Add source" wall.** The empty notebook used to be a page that told you it
  was empty and offered nothing. It now leads with what you can do: upload a file,
  or search the web for pages on a topic. The heading names the product's actual
  promise - the answer comes from the sources, whether those arrived as an upload
  or as a search.

### "Session" and "notebook"

The UI says **notebook**, matching the reference product's name for a named set of
sources plus its own conversation. Everything below the UI — the API, the database
schema, the tests — still says **session**, because "notebook" names the concept
and "session" names the isolation boundary, and the boundary is what the server
actually enforces. Renaming the plumbing would imply a semantic change that is not
being made.

### Starter questions are name-derived, not content-derived

`suggestions.js` builds the empty-state questions from source *filenames*, source
*kinds*, and each source's numeric-density flag. The reference product reads the
notebook to write its suggestions, which means a model call on every load — money
and latency before the user has typed anything. Names, kinds and the flag are
already in memory from `/api/status`, so this costs nothing and cannot fail.

The honest limitation: for a notebook called `notes.txt` the suggestions are
generic. Three guards keep them from being worse than useless:

- **No suggestions at all for an empty notebook.** Every one would be a refusal,
  and that would be the first impression.
- **No figures question unless there is reason to think there are figures** — a
  spreadsheet, or a source with numeric-dense chunks. Prose that happens to
  mention "30 days" triggers neither. Two signals rather than one, because the
  numeric flag almost never fires for spreadsheets: the CSV parser reshapes a
  table into `Label: Value` lines, and `Revenue: 12000` is 5 digits in 13
  characters, far below the density `parsers.is_numeric_heavy` asks for. A
  40-row revenue table and a one-paragraph memo both report `numeric: 0`.
- **No filename too short or too long to read in a sentence.** When every name is
  dropped, the fallback questions are generic by wording rather than by accident,
  because "the key points about r2?" is worse than no question.

If generic suggestions prove not worth having, the fix is an endpoint that samples
a few chunks and asks for questions, not a longer hardcoded list.

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
npm test           # 178 checks, jsdom, no database or API key needed
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

The chain is `0001_baseline` → `0002_message_evidence` (citations and evidence on
messages) → `0003_source_url` (where a source came from) → `0004_usage_events`
(the tokens a web search spent). `0003` exists for one
reason: once a web page is indexed, nothing in the row says whether it was
uploaded or found, so the source list could not link back to the page it came
from. `sources.url` is `NOT NULL DEFAULT ''` rather than nullable, so the
"uploaded, so no origin" case is a stated fact instead of a hole in the data.

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
| `CHUNK_SIZE` | `900` | Characters per chunk. Clamped to at least 1 |
| `CHUNK_OVERLAP` | `150` | Clamped to at most `CHUNK_SIZE / 2` - an overlap at or above the window walks the splitter's cursor backwards, which never terminates |
| `TOP_K` | `6` | Passes retrieved to the model |
| `MAX_CONTEXT_CHARS` | `14000` | Truncation guard |
| `MIN_SCORE` | `0.25` | Relevance floor - see below |
| `MAX_PER_SOURCE` | `3` | Max chunks from one source, so answers can span sources |
| `HISTORY_TURNS` | `6` | Prior turns read for follow-ups (`0` disables) |
| `WEB_SEARCH` | on | `0` turns off web-source search; see below |
| `WEB_SEARCH_MODEL` | `openai/gpt-oss-20b` | Groq model used to find URLs |
| `WEB_SEARCH_MAX_PAGES` | `5` | Ceiling on pages fetched per search |
| `WEB_FETCH_TIMEOUT` | `20` | Seconds per page fetch |
| `WEB_FETCH_MAX_MB` | `8` | Larger pages are refused, not truncated |
| `WEB_FETCH_MAX_REDIRECTS` | `3` | Redirect hops followed |
| `LANGSMITH_TRACING` | off | Send traces to LangSmith - see below |
| `LANGSMITH_TRACING_INCLUDE_TEXT` | off | Let document text leave the machine. Separate switch, deliberately |
| `LANGSMITH_TRACING_LOCAL` | – | Write the same traces to a local JSONL file, no network call |

## Tracing, and what it costs you in privacy

Once the app is running, a bad answer is a fact you have to reconstruct, and the
pieces are scattered: retrieval scores in `store.py`, the citation audit in
`llm.py`, tokens and latency in `usage.py`. Tracing puts a whole run in one place
and lets you group by a field, which is how "why was Tuesday slow" stops being an
archaeology exercise.

The reason it is two switches and not one is privacy, and it is worth being blunt
about it: **tracing a RAG system sends your retrieved passages to a third party.**
Those passages are your documents. The app's core promise is that embeddings never
leave the machine and only retrieved chunks go to the model, and a tracer that
uploaded chunks by default would quietly widen that promise to cover a company you
never agreed to.

So `LANGSMITH_TRACING=1` sends the question, the retrieval scores, the citation
audit and the cost — enough to tell a retriever failure from a generation failure —
and **no document text**. `LANGSMITH_TRACING_INCLUDE_TEXT=1` is a second, separate
decision, because "I want traces" and "my documents may leave this machine" are
different questions and the person answering the second is often not the person who
turned tracing on.

`LANGSMITH_TRACING_LOCAL=traces.jsonl` writes the identical span tree to a file
with **no network call at all**. That is the mode to use when the corpus is
confidential, and testing a RAG system's observability should not require uploading
the documents it is about.

Retrieval and generation are traced as separate spans on purpose. A bad answer
caused by bad retrieval is indistinguishable from a bad answer caused by a good
retriever and a bad model, and you cannot tell them apart unless the steps are
separable. A refusal produces a retrieval span and no generation span, so
"how often does the relevance floor hold" is answerable from the trace.

A real trace, refused question included, with text withheld:

```
retrieval   retriever     19.5ms  best_score 0.6825  relevant true   returned 1
generation  llm        1092.8ms  cited [1]  ungrounded false  prompt_tokens 461
ask         chain       1125.5ms  verdict answered  cost {...}
retrieval   retriever     19.6ms  best_score 0.0     relevant false  returned 0
ask         chain         26.8ms  verdict no_match    cost {...}
```

Two guarantees that are asserted by tests rather than promised in prose:

- **A failing step still closes its span.** An exception is the reason you are
  reading the trace, so a tracer that only records success is useless.
- **Tracing can never break a request.** Every failure in the tracer is swallowed;
  observability that can take down the app is worse than no observability.

`/api/status` reports the current tracing state, including whether document text is
included, so "is anything leaving this machine?" is a question the app answers
rather than one you answer by reading `.env`.

## Web search, and where the promise changes

Everything above keeps document text on this machine. Web search is the one feature
that breaks that, and it is worth being specific about what crosses the boundary:

- the **search term** goes to the search provider (Groq's browser tool), and
- the app then **fetches the pages it finds** and indexes them locally.

Documents you uploaded still behave as before - only their retrieved chunks go to
the model. But a page found on the web is fetched by this machine, and that is a
network request to a site chosen by a search engine answering a model, so it needs
the same care as any other fetch of untrusted input.

`WEB_SEARCH=0` turns the feature off, and the card in the empty notebook greys out
and says why rather than failing when clicked. With no Groq key, or with a
non-Groq provider, `/api/status` reports `web_search.available: false` with a
reason and the same card stays disabled.

### A web page becomes a source, not an answer

The search model is asked for a bare list of URLs and nothing else, then the pages
are downloaded, parsed, chunked and embedded through exactly the same code path as
an upload. The model never writes a sentence the user reads. That matters for the
grounding guarantee: a fetched page is a citable source that survives a reload and
can be deleted, whereas a model's prose summary of a search would be uncitable
paragraph text with no provenance - exactly what this app is built to avoid.

One search costs real tokens and real time, and the model returns both in the
response so the cost is visible rather than assumed. Measured live: about 8-13s of
search and 20k-40k prompt tokens on `openai/gpt-oss-20b`, plus fetch time. That is
the price of the feature being on, and `WEB_SEARCH=0` is the switch for it.

### Why the fetcher looks paranoid

URLs arriving here come from a search engine responding to a model, so a URL can
name anything at all - including `http://169.254.169.254/` (a cloud instance's
credentials endpoint) or this machine's own database at `http://localhost:5432/`.
The fetch path therefore:

- resolves the hostname and refuses loopback, private, link-local and reserved
  ranges **before** opening a socket;
- follows redirects **by hand** and re-checks every hop, because a public host
  that 302s to `169.254.169.254` is the standard way around a check that only
  looks at the first URL. `httpx` can follow redirects itself, but it does so
  after connecting, which is too late - the request to the private address has
  already gone out by the time the final URL could be inspected;
- caps response size while streaming, so an endless body is abandoned rather than
  buffered and then measured;
- caps fetch time, page count and redirect hops.

Residual risk, stated plainly: the address is validated and then the request goes
out by hostname, so a hostile DNS server could return a public address to
`getaddrinfo` and a private one to the connection (DNS rebinding). Closing that
completely means pinning the connection to the validated IP, which is not
implemented.

This is also why the feature is unsuitable for an unauthenticated deployment on a
public network: anyone who can reach the app can make it fetch arbitrary URLs.

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
| `POST` | `/api/sources/web` | `{query, limit}` → find pages and index them as sources |
| `POST` | `/api/sources/{id}/reindex` | Re-parse and re-embed from the copy already on disk; keeps the id |
| `GET` | `/api/sources/{id}/file` | The original upload as stored, `inline` so it opens rather than downloads; HTML/SVG/XML served as `text/plain` |
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
      "heading": "Orbital mechanics > Escape velocity",
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

### Three verdicts, not two

`verdict` is one of three values, and the third exists because "Hi" used to come back
as a relevance score:

| `verdict` | When | What it means |
|---|---|---|
| `answered` | Passages cleared the floor and the model was asked | The normal case |
| `no_match` | Retrieval ran and nothing cleared the floor | "I looked, your documents do not cover this" |
| `conversational` | The message was not a question about the corpus | "There was nothing to look up" |

`conversational` is not a softer `no_match`. A greeting is not a failed retrieval: it
has no passage to retrieve and no answer to ground, and the old reply led with
"best match 0%, below the 25% floor" about a message that was never a search. Worse,
that `0.00` was not a finding about the user's documents - it was the absence of a
comparison. An empty notebook has the same problem, so it is told to add a source
rather than given a score about zero documents.

It also is not a canned string. In a notebook that has sources, a greeting is answered
by the model through `llm.chat()`, which sends the history but no passages and records
what the call actually cost - the same reason ChatGPT answers "hi" rather than
printing a fixed line. There is nothing to be faithful to, so there is nothing to
ground and no citation to check, but the reply is still the model's. Two cases stay
local and free, because in both the model has nothing to work with: a notebook with no
sources (there is no subject to be helpful about) and a provider that is unreachable
(falls back to a fixed line rather than an error page).

`app/intent.py` makes the question/greeting distinction before anything is embedded, so a
greeting costs no embedding and no retrieval. It matches the **whole message** against a greeting list, and
that is the load-bearing detail: "hey, what is the refund window?" opens with a social
word and is a real question, and a keyword scan would greet it and drop the query. Every
doubt resolves towards retrieval, because the failures are not symmetric - a wrongly
greeted real question gets a useless reply, while a wrongly-searched "hi" costs one
embedding and gets an honest refusal.

The cost of this is one honest limit: `SMALL_TALK` contains `perfect`, `nice`, `ok` and
`great`, so typing only one of those as a document query is treated as small talk rather
than searched for. The escape hatch is length - "what does perfect mean in this
contract?" is a question - but a user searching for a single-word term that happens to be
one of those four will get a greeting. That is a known trade, not an oversight.

### Citations survive a reload

`messages` stores `role`, `content`, and - since migration `0002` - `citations` and
`evidence` as JSONB. This is worth calling out because getting it wrong was a real bug:
the `[n]` markers live in `content`, so they survived a reload while the citation cards,
which only ever existed in the HTTP response, did not. Ask a question, read a cited
answer, close the tab, come back, and the provenance was gone with the markers pointing
at nothing. Nullable because a user turn has neither field; JSONB rather than more tables
because this is only ever read with its parent row and never queried on its own.

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
| PDF | per page, with table-like pages split on line boundaries. The heading is inferred: PDF declares nothing, so a line set at least 15% larger than the body text is treated as a section title, repeated running headers are rejected, and the result nests like a Markdown heading path |
| hard-wrapped TXT | rejoined at the line break, so a phrase split by column wrapping stays searchable |

Chunks that are still numeric-flat (a PDF table with no extractable headers) are **flagged,
not dropped**. They are damped by `NUMERIC_DAMPING` at retrieval so they cannot outrank prose,
and the damping is skipped when `NUMERIC_PENALTY_SHARE` of the candidates are already
numeric, so a corpus that is entirely numbers keeps working with its ranking untouched.

Table-like text is also chunked on line boundaries. Splitting it on word boundaries used to
produce fragments like `003.52309 11.5 52 0`, which match nothing and read as gibberish when
cited.

## Ingest: the same file twice, and what it actually did

Two things about ingestion used to be invisible, and both were the kind of
invisible that looks fine until it does not.

**A file you already have.** Re-uploading a PDF created a second source, indexed
it a second time, and gave the answer a context containing the same passage
twice. Nothing could say so. `POST /api/sources` now takes a SHA-256 of the
bytes *before* parsing - the check exists precisely to skip the parse - and
returns the source you already had with `duplicate: true`. The UI says
`"<file> is already in this notebook. Nothing was re-indexed."` rather than
looking like a successful upload that quietly changed no counts.

The comparison is scoped to the session. Two notebooks may both hold the same
public document, and a global uniqueness rule would tell the second person they
cannot index their own file. Rows created before this migration have a NULL
hash, which never matches - the safe direction, since a missed duplicate costs a
little storage and a false one costs an upload.

**The second ceiling.** `CHUNK_SIZE` is a character ceiling. The embedding model
enforces a wordpiece ceiling, and `sentence-transformers` truncates silently at
it: on the eval corpus a 900-character chunk tokenises to **379** pieces against
MiniLM's **256**, and the last 123 were never embedded. Nothing said so. Chunks
are now counted with the model's own tokenizer at ingest and split again if they
do not fit, so a stored chunk is always a chunk the vector actually describes.
`EMBED_MAX_TOKENS` sets the window (0 disables the check).

What ingest reports, in `GET /api/status` as `ingest` and per source:

| field | meaning |
|---|---|
| `avg_chars` / `max_chars` | chunk length in characters - is `CHUNK_SIZE` doing what was assumed |
| `token_max` / `token_window` | worst chunk's wordpieces against what the model keeps |
| `fit_splits` | extra cuts made only to satisfy the window - not a defect, a document whose characters do not map onto wordpieces the way `CHUNK_SIZE` assumes |
| `numeric_pct` | how much of the index is bare tables, which retrieval damps |
| `unmeasured` | sources indexed before any of this was recorded, whose zeros mean "unknown" |

That last field is the honest one. A source that predates the report has
`token_max` 0, and 0 here is not "everything fit", it is "nobody checked".

### Seeing what ingestion did, to one file

The drawer lists what a notebook holds. It does not say how any of it got
there, and a count of chunks is a claim with no way to check it. Two controls,
at the bottom of the drawer:

**A chevron on each source row** opens that source's ingestion detail - the
chain of steps that actually ran for *this* file, as counts rather than
adjectives:

```
read PDF › 9 blocks › 12 chunks › 0 cut by size › 1 wordpiece cut ›
worst 210/256 wordpieces › 3 table chunks › embedded 384-d
```

Below it, the fact rows: format and file size in KB, when it was indexed, the
average and longest stored chunk, which ceilings fired, and how much of the
index is tables. One row open at a time - the drawer is already a scroll
region.

Null and 0 mean different things here, and the panel keeps them apart. `0 cut by
size` is an ordinary result for a well-behaved file and a claim about it;
`size cuts not measured` is a source indexed before the count existed. The same
applies to `token_max` 0, which is why the wordpiece rows consult it rather than
`fit_splits`: both were recorded in the same pass, so no walk means neither
number exists, and printing "no cuts" for that would be a statement about a
file nobody measured.

**"View original"** opens the stored upload in a new tab - the bytes that went
in, not a rendering of them, because checking the index against a re-encoded
copy is checking it against nothing. `GET /api/sources/{id}/file` serves it
inline so a PDF opens in the browser's viewer rather than downloading.

Two rules on that route. It is scoped exactly like every other source route:
a source in another notebook, or an id you do not own, is a 404 that does not
reveal whether it exists. And a path that resolves outside `UPLOAD_DIR` is
refused too - the path comes from the row rather than the request, so there is
nothing to traverse, but rows are written by callers and a boundary check costs
nothing next to trusting every one of them forever. Uploaded HTML, XHTML, SVG
and XML are served as `text/plain` with `nosniff`: on this origin a page served
as its own content type could call the API as whoever is looking at it, and what
the control promises is the file that was indexed, not something that runs.

**"How indexing works"**, collapsed at the foot of the drawer, is the pipeline
once rather than per file, since only the format parser differs between
documents: hash and short-circuit, read the structure, split structure-first
under `CHUNK_SIZE`, re-cut under the wordpiece window, embed locally, store.

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

### Re-ranking the shortlist

Reciprocal rank fusion scores the query and the chunk *separately* and combines
two numbers, so it cannot notice a chunk that shares every keyword with the
question while answering a different one. A cross-encoder reads them jointly and
can, at roughly two orders of magnitude the cost per pair - so it runs on the
fused shortlist rather than the corpus: recall from the cheap pass, precision
from the expensive one.

`app/rerank.py` gates it. The model (`cross-encoder/ms-marco-MiniLM-L-6-v2` by
default) downloads on first use, so every failure path - no network, missing
package, a `predict` that raises - degrades to "leave the order alone". It is
observability's rule applied to ranking: a feature that can block the request is
worse than the feature being absent. `/api/status` reports `configured` and
`loaded` separately, so "the flag is on but the model never arrived" is
distinguishable from "off".

Two things deliberately do **not** change. Each hit keeps its dense cosine in
`score`, because that is the scale `MIN_SCORE` was calibrated against and the
number the floor already used; the cross-encoder's own score sits alongside in
`rerank`. And re-ranking happens only after the floor, because the floor reads
`score` - reranking before it would cost the same and change no decision.

Measured on the cloze set, `k=5`, everything else fixed:

| | without | with |
|---|---|---|
| MRR@5 | 0.952 | **0.974** |
| nDCG@5 | 0.928 | **0.943** |
| recall@5 | 0.955 | 0.955 |
| precision@5 | 0.226 | 0.226 |
| fact coverage | 0.922 | 0.922 |
| trap leaks | 0 | 0 |

The order improved and nothing else moved. A re-ranker that had pushed recall up
by dragging traps in would have been a reason to turn it off.

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

Current result at `MIN_SCORE=0.25` over the committed corpus: **79/79 relevant
answered, 42/42 traps rejected**, across 17 formats.

An optional external PDF can be scored too, with
`EVAL_REAL_PDF=/path/to/doc.pdf`. It is pinned by path and never guessed, which
is a correction: this used to fall back to "the largest PDF in `UPLOAD_DIR`", and
since that directory accumulates every file anyone has uploaded, the slot filled
with an unrelated fixture while the questions still asked about the intended
document. The five mismatched questions were then reported as retrieval failures
— a measurement of the harness, not of the retriever. If the variable is unset the
run prints `real-world PDF: SKIPPED`, because silently different totals between
two runs are worse than smaller ones.

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
The set holds **122 items across all 27 corpus documents**: 30 field, 24
record, 8 definition, 31 value, 22 multi-block, and 7 traps that must be refused.
One mechanical check is worth naming, because it was written after the set had
been wrong: a question of the form `_____ KEY:` reads to anyone as "the value of
KEY", so the answer must be what the fact says KEY holds. Questions used to be
cut on `': '` with the value from the line above, which produced
`_____ $.runtime.cpu_limit:` labelled `2048` - memory_limit_mb's value. Every
other check passed, because `2048 $.runtime.cpu_limit:` is verbatim text of the
flattened fact, and the run scored a model wrong for answering `1500`. Seventeen
items were mislabelled that way and are now correct.
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

Current result: recall@5 **0.955**, fact coverage **0.922**, MRR **0.974**,
nDCG **0.943**, traps leaked **0**. The two rank-order numbers moved when
re-ranking was added and measured (see below); recall, coverage and the trap
leak did not move at all, which is the shape a second-stage ranker should have.
 Precision@5 is **0.226**, which is the
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

Six numbers, each answering a different question:

| Metric | Question it answers |
|---|---|
| `faithfulness` | Is every cited claim traceable to the passage it cites? |
| `citation_precision` | Are the citations load-bearing, or decoration? |
| `substance` | Did the answer cover the labelled facts, tolerating rewording? |
| `relevancy` | Did it land every labelled fact, word for word? |
| `verbatim` | How much does the answer quote rather than paraphrase? |
| `reference` | Did the answer contain the value the document actually holds? |

### The answer key, and what it is worth

Five of those compare the answer to passages the retriever chose, so they grade
the system against itself. `reference` is the exception: on the cloze set the
label is the text that was blanked out of the document, so the answer key comes
from the document and nothing the retriever or the generator produced enters it.

It is lexical containment on purpose - an embedding similarity or an LLM judge
would put the model under test inside its own grading, which is the objection
written above. The cost is that a correct paraphrase of a short value scores 0,
so a low `reference` means *did not contain the words*, which is weaker than
*did not know*. It is reported next to faithfulness rather than inside it.

It only exists where the item has an answer key (`answers` on the cloze set;
traps carry `null` deliberately) and the rate is taken over those items alone,
because averaging in the keyless ones would pull the number down for a reason
unrelated to the answers.

Producing the number costs a full run - `eval_generation` calls a real model, and
115 items is on the order of a day's free-tier quota - so it is a command you
run, not a figure in CI:

```bash
.venv/bin/python -m experiments.eval_generation \
  --gold experiments/gold/cloze_set.json --allow-machine --allow-unverified
```

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
  are caught. A cross-encoder is the standard fix for exactly this, and one was added -
  but on the evidence of MRR and nDCG above, not on this. This gap was **not** re-measured
  afterwards, so it is still open and the reranker should not be credited with closing it.
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

The same two numbers are also reported as percentiles, in `GET /api/status`:

```json
"latency": {
  "window": 200,
  "samples": {
    "retrieval_ms": {"count": 41, "p50": 6.2, "p95": 18.9, "max": 24.0},
    "generation_ms": {"count": 33, "p50": 812.4, "p95": 2140.7, "max": 3021.1},
    "total_ms": {"count": 33, "p50": 818.6, "p95": 2146.0, "max": 3027.4}
  }
}
```

A mean is the wrong shape for this - one provider hiccup moves it and says nothing
about the ordinary request - and percentiles over the whole process lifetime answer
"was it slow once in March", which is true and unactionable. So it is a ring buffer
of the last `LATENCY_WINDOW` samples, and the claim is "is it slow now". Stages that
did not run are skipped rather than recorded as 0: a refusal has no generation and a
greeting has no retrieval, and counting those would make generation look faster than
it is, which is the one direction this number must not be wrong in.

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

### Where the two kinds of spend are recorded

Cost is recorded in two places, because the two events are different shapes:

| kind | recorded in | why there |
|---|---|---|
| answer, greeting, summary | `messages.evidence` | it belongs to the turn, and reads back with the transcript |
| web search | `usage_events` (migration `0004`) | it is not a turn - nothing appears in the chat when a page is fetched |

`/api/status` returns a `usage` object summing both, so a total can never be
short by one of them. A total covering only chat turns looks complete while
quietly missing every search, which is worse than no total.

The ledger's `session_id` is nullable and uses `ON DELETE SET NULL` rather than
`CASCADE`: the spend happened whether or not the notebook still exists. A cost
record that erases itself when the session is deleted is not a record.

Recording is best-effort. An exception there is swallowed rather than failing
the request - the user's pages are already indexed, and re-fetching them to
retry a log line costs more than the line is worth.

## How grounding works

1. Files are parsed per format by `app/parsers.py` (39 suffixes, see above) into blocks that
   carry a page number and a heading, then split into overlapping chunks on paragraph,
   sentence, line or word boundaries depending on content.
2. Numeric-flat chunks are flagged `numeric_heavy`. **Nothing is discarded.**
3. Each chunk is embedded with MiniLM and inserted into `chunks` with its text, page,
   heading, and a `vector(384)` column with an HNSW cosine index beside it. At this
   corpus size the planner often prefers a sequential scan anyway (~1.0 ms exact vs
   ~1.3 ms through the index), so the index is a path that exists rather than one
   that is always taken. A BM25
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

123 checks covering upload, per-format parsing (including YAML/TOML/INI and hard-wrapped
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
transcript persistence, cascading deletes, and surviving a restart), the
original file behind a source served byte for byte only inside its own notebook
(an uploaded page served as text rather than as something a browser runs, a path
outside the upload directory refused, and a missing original reported as 404
rather than as an empty body), and the
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
- Uploads are kept on disk so a source can be re-indexed without re-uploading: `POST /api/sources/{id}/reindex` re-reads the original and rebuilds its chunks under the current `CHUNK_SIZE` and `EMBED_MODEL`. Each delete
  path cleans up its own files; a hard crash mid-session can still leave an orphan.
- Re-ranking is on by default and can be turned off with `RERANK_ENABLED=0`. It improves
  rank order (MRR 0.952 -> 0.974) and leaves recall, precision and the trap leak untouched;
  it does not fix the entity-overlap case noted above, which is a labelling problem rather
  than an ordering one. The model downloads on first use, so the first query after a fresh
  install pays for it.
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
