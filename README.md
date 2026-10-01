# Local NotebookLM

A local, source-grounded research assistant. Upload documents, ask questions, get
answers with inline citations that link back to the exact source passage.

## Stack

| Piece | Choice |
|---|---|
| API | FastAPI |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (runs locally) |
| Vector store | Qdrant, in-memory (`:memory:`) |
| LLM | Groq (`openai/gpt-oss-120b`) by default, or OpenAI / Ollama |
| UI | Vanilla HTML/CSS/JS served by FastAPI |

## Setup

Windows:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # then paste your GROQ_API_KEY into .env
```

Linux / macOS:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # then paste your GROQ_API_KEY into .env
```

The venv is strongly recommended: `sentence-transformers` pulls in PyTorch, and the
default wheel is the CUDA build (~6 GB). On a CPU-only machine, install the CPU
wheel instead:

```bash
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Run

```powershell
.venv\Scripts\python run.py
```

Open http://127.0.0.1:8000

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
| `CHUNK_SIZE` | `900` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | |
| `TOP_K` | `6` | Passes retrieved to the model |
| `MAX_CONTEXT_CHARS` | `14000` | Truncation guard |
| `MIN_SCORE` | `0.25` | Relevance floor - see below |
| `MAX_PER_SOURCE` | `3` | Max chunks from one source, so answers can span sources |
| `HISTORY_TURNS` | `6` | Prior turns sent for follow-ups (`0` disables) |

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/status` | Provider, model, source/chunk counts, tuning |
| `POST` | `/api/sources` | Upload a file (multipart `file`) |
| `DELETE` | `/api/sources/{id}` | Remove one source |
| `DELETE` | `/api/sources` | Clear the notebook |
| `POST` | `/api/ask` | `{question, history}` → `{answer, citations, evidence}` |
| `POST` | `/api/summarize` | `{instruction}` → `{summary, citations, evidence}` |

If the LLM backend is unreachable, `/api/ask` returns `503` with a message telling you
what to fix instead of a raw 500.

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
   be cited are the freshest thing in the context window. `role` is validated against
   `user | assistant` by the request schema, so a client cannot inject a `system` turn to
   override the grounding rules (that attempt returns `422`).

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

Builds **20 documents across 15 file formats and 19 unrelated subjects** - brewing,
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

## How grounding works

1. Files are parsed per format by `app/parsers.py` (36 suffixes, see above) into blocks that
   carry a page number and a heading, then split into overlapping chunks on paragraph,
   sentence, line or word boundaries depending on content.
2. Numeric-flat chunks are flagged `numeric_heavy`. **Nothing is discarded.**
3. Each chunk is embedded with MiniLM and upserted into Qdrant with payload
   `{source_id, source, page, heading, text, numeric_heavy, chunk_id}`. A BM25 index over the
   same chunks is kept in memory alongside it.
4. A question is embedded and cosine-searched, and separately scored by BM25. The candidate
   set is the **union** of both, over-fetching 4x so the per-source cap has choices, and the
   two rankings are fused with RRF.
5. Candidates are admitted via the three routes above. Numeric-flat chunks are damped.
   Surviving chunks must clear both `MIN_SCORE` and `MIN_RATIO` of the best hit, then the
   per-source cap is applied and backfilled. The survivors are numbered and injected into a
   system prompt that forbids outside knowledge and requires `[n]` citations.
6. The model's citations are validated against the passages supplied, and the UI renders
   each retrieved chunk below the answer as a clickable card with source, page, match
   score, and full text.

## Notes

- Embeddings never leave your machine. Only the retrieved chunks are sent to the LLM.
- `all-MiniLM-L6-v2` is a 384-dim, ~80MB model tuned for English. For multilingual or
  higher-quality retrieval, swap in `EMBED_MODEL` and update `EMBED_DIM` in
  `app/config.py` to match.
- `MIN_SCORE` was calibrated against MiniLM's score distribution. A different embedding
  model shifts those numbers, so re-check it before trusting the default.

## Tests

```powershell
.venv\Scripts\python test_app.py
```

38 checks covering upload, per-format parsing (including YAML/TOML/INI and hard-wrapped
text), chunking, hybrid retrieval, numeric damping, citations, the 503 LLM-down path, Groq
routing, BOM handling, and the grounding guarantees above (relevance floor, citation
validation, history hardening, cross-source diversity). The LLM is stubbed, so no API key is
needed to run them.

A failed check no longer stops the run: every check is reported and the process exits
non-zero if any failed.

## Known limitations

- The index is **in-memory**. Restarting the server clears it; uploaded files stay in
  `data/uploads` but must be re-added (and are never cleaned up automatically).
- No reranking. Fused dense + BM25 order goes straight to the prompt. See the entity-overlap
  note above for the one case that measurably needs a cross-encoder.
- Table content is **dropped rather than parsed**. Structured extraction (repeating the
  header row onto each data row) would make tables searchable instead of discarded.
- A cross-encoder reranker was evaluated and **is not wired in**: on the test handbook its
  scores for legitimate questions (0.002-0.597) overlapped those for unanswerable ones
  (0.000-0.125), so it cannot serve as an absolute relevance threshold. It may still help
  rank passages, which needs the eval harness to confirm. Reproduce with
  `.venv/bin/python experiments/rerank_eval.py`.
- Uploads are read fully into memory with no size limit, and indexing a large PDF blocks
  the event loop.
- One global notebook. There is no persistence, no multiple notebooks, and no auth.
- Answers are rendered as plain text, so `**bold**` and `- bullets` show literally.
- No Markdown, no streaming, and citations cannot deep-link to the original file.
