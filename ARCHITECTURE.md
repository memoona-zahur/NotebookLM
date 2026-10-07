# Architecture: how the whole app fits together

Every flow in this document ends at a `module.py:line` reference so the picture can
be checked against the code rather than trusted. References were verified against
commit `d6ae9b8` on 2026-10-07; they drift as the code moves, and the line number
is the thing to re-check, not the shape.

Four things happen in this app: a document is uploaded and indexed, the web is
searched and indexed, a question is answered from what was indexed, and a
summary is written. Everything else — session CRUD, file viewing, the chunk map,
status — is a read over the same state.

---

## 1. The stack, end to end

```
┌────────────────────────────────  BROWSER  (frontend/src)  ────────────────────────────────┐
│ App.jsx ── turns[] ── send() (App.jsx:273) ──► api.ask()  (api.js:119)                    │
│   ├─ IngestPanel.jsx   drop → api.upload()  ───────────────┐                             │
│   ├─ SourcesDrawer.jsx source list, footer totals, panels  │                             │
│   ├─ Answer.jsx        answer + [n] citations + evidence   │                             │
│   ├─ PipelinePanel.jsx ingest diagram   ChunkMap.jsx        │                             │
│   └─ api.js  request() (api.js:19) → fetch, ?session_id= ◄──┘                            │
└───────────────────────────────────┬──────────────────────────────────────────────────────┘
                                    │  JSON over HTTP, no auth (local app)
                ════════════════════▼═════════════════════  FastAPI  app/main.py:37
                ║  routes/ — thin: parse → delegate → status code                          ║
                ║   chat.py      POST /ask, /summarize                                     ║
                ║   sources.py   POST /sources, /sources/web, GET …/file, …/chunks,        ║
                ║                …/reindex, DELETE, GET /occurrences                       ║
                ║   sessions.py  CRUD + messages/clear                                     ║
                ║   meta.py      GET /api/status (meta.py:22)                              ║
                ╚═══════════════════╤═════════════════════╝
        ┌───────────────────────────▼────────────────────────────  DOMAIN  ──┐
        │ answering.py   policy: classify → retrieve → floor → refuse/generate│
        │ store.py       hybrid retrieval + ingest + duplicate/reindex/cache  │
        │ llm.py         SYSTEM_PROMPT, provider dispatch, citation validation│
        │ parsers.py     pdf/docx/md/json/csv/code/log/html + OCR → blocks    │
        │ intent.py      greeting vs question vs empty-notebook               │
        │ payloads.py    evidence() summary, session payload                  │
        │ usage.py       Request/Usage, latency P50/P95, record()             │
        │ tracing.py     LangSmith spans (off by default)  │ webingest.py     │
        │ lexical.py BM25   embeddings.py MiniLM + reranker   websearch.py    │
        │ config.py env → constants   db.py SQL + alembic migrations          │
        └───────┬───────────────────────────────┬───────────────────┬─────────┘
                │                               │                   │
     ┌──────────▼──────────┐        ┌───────────▼────────┐   ┌──────▼──────────────┐
     │ PostgreSQL + pgvector│        │ UPLOAD_DIR files   │   │ EXTERNAL            │
     │ sessions, sources,   │        │ original uploads   │   │ Groq / OpenAI /     │
     │ chunks(+embedding),  │        │ (uuid-hex names)   │   │  Ollama (local)     │
     │ messages(evidence),  │        └────────────────────┘   │ HF model download   │
     │ usage_events,        │                                 │ Tesseract OCR       │
     │ ingest_reports       │                                 │ web fetch (SSRF-    │
     └──────────────────────┘                                 │  guarded)           │
                                                              │ LangSmith (opt-in)  │
                                                              └─────────────────────┘
```

Two properties the layout is built around:

- `routes/` handlers do no policy work. `app/answering.py:1` says so in its
  docstring, and reading any handler confirms it: parse, delegate, map one
  domain error to an HTTP status.
- The session is the isolation boundary. Retrieval, duplicate detection,
  lexical caches and every read of `chunks` are scoped by `session_id`.

---

## 2. Flow A — uploading a document

```mermaid
sequenceDiagram
    autonumber
    participant UI as IngestPanel.jsx
    participant API as routes/sources.py
    participant ST as store.add()
    participant PR as parsers.parse()
    participant EM as embeddings.py
    participant DB as PostgreSQL + pgvector

    UI->>API: POST /api/sources (multipart file, session_id)
    API->>API: suffix in ALLOWED ? otherwise 400 (sources.py:64)
    API->>API: resolve_session() unknown id is 404 (payloads.py:33)
    API->>API: write UPLOAD_DIR / uuid4hex + suffix
    API->>ST: store.add(path, display_name, session_id) (store.py:225)
    ST->>DB: content hash, then _find_duplicate in this session (store.py:312)
    alt duplicate
        ST-->>API: existing Source flagged duplicate
        Note right of ST: stored file discarded, no parse, no embed
    else new document
        ST->>PR: parsers.parse(path, stats) (parsers.py:1096)
        PR-->>ST: Block(page, heading, text) per chunk
        ST->>ST: drop empty blocks, flag is_numeric_heavy (parsers.py:206)
        ST->>DB: INSERT sources (token_max, fit_splits, size_splits, file_bytes)
        ST->>EM: embed_texts(block.text) (embeddings.py:13)
        EM-->>ST: normalised vectors, MiniLM-L6-v2, batch 32
        ST->>DB: INSERT chunks (position, page, heading, text, numeric_heavy, embedding)
        ST->>DB: UPDATE sources SET chunk_count, numeric_count
        ST->>ST: _invalidate(session_id) bumps the BM25 revision
        ST-->>API: Source(...)
    end
    API-->>UI: { source, duplicate, ...store.stats() } (store.py:548)
    UI->>UI: refresh list, advance PipelinePanel stages
```

Parsing details worth knowing, because they are where an upload fails:

- `parsers.parse` dispatches by suffix: PDF (text layer first, Tesseract OCR for
  image-only pages, `parsers.py:609`), DOCX, Markdown, HTML, JSON/YAML/TOML,
  CSV/TSV tables, logs, Python and generic code, plain text.
- Two ceilings are applied while chunking: `CHUNK_SIZE = 900` characters
  (`config.py:109`) and `EMBED_MAX_TOKENS = 256` wordpieces counted with the
  embedding model's own tokenizer (`config.py:81`, `parsers.py:123`). A chunk
  that overflows the token limit is split, and the split is recorded as
  `fit_splits` on the source rather than silently truncated.
- `token_max` is the measured worst-case wordpiece count of any chunk — it is
  `0` only when the file was never token-walked, which is the NULL-vs-zero rule
  the rest of the app follows.

---

## 3. Flow B — "Search the web" and index the results

```mermaid
sequenceDiagram
    autonumber
    participant UI as SourcesDrawer card
    participant API as routes/sources.py
    participant WI as webingest.ingest()
    participant WS as websearch.find()/fetch()
    participant ST as store.add()
    participant DB as usage_events

    UI->>API: POST /api/sources/web {query, limit} (sources.py:110)
    API->>WI: ingest(query, session_id, limit) (webingest.py:72)
    WI->>WI: web_search_available() no key or WEB_SEARCH=0 is 400 (config.py:230)
    WI->>WS: find(query, limit) (websearch.py:98)
    WS->>WS: Groq chat completion with tools=[{type browser_search}]
    WS-->>WI: candidates after URL tidy + dedupe
    loop each candidate
        WI->>WS: fetch(candidate) (websearch.py:275)
        WS->>WS: block private/loopback hosts, cap size and time
        WS-->>WI: FetchedPage or a per-URL failure
        WI->>ST: _store_page writes a temp file, then store.add (Flow A)
    end
    WI->>DB: record_usage_event(kind=web_search, model, tokens, latency) (db.py:281)
    WI-->>API: WebIngest { added[], failed[{url, reason}], cost }
    API-->>UI: 200 even when some pages failed (sources.py:128)
```

Partial success is the design: five results, three indexed, two refused is
returned as 200 with `failed` populated. 400 means nothing could run at all
(no key, disabled, empty query); 502 means the search itself failed upstream.

The web search is **manual** — it is never run as a side effect of a question.
That is why `usage_events` needs no link to a `messages` row.

---

## 4. Flow C — asking a question

```mermaid
sequenceDiagram
    autonumber
    participant UI as App.jsx send()
    participant API as routes/chat.py
    participant AS as answering.ask()
    participant IN as intent.classify()
    participant ST as store.search_detailed()
    participant LM as llm.answer()
    participant TR as tracing (opt-in)
    participant DB as PostgreSQL

    UI->>API: POST /api/ask {question} with session_id (api.js:119)
    API->>API: empty question is 400, unknown session is 404 (chat.py:20)
    API->>AS: answering.ask(question, session) (answering.py:36)
    AS->>DB: recent_messages(HISTORY_TURNS) history comes from the DB (db.py:239)
    AS->>IN: classify(question) (intent.py:103)
    AS->>ST: stats(session_id) to learn whether sources exist (store.py:548)
    alt not a question, or notebook is empty
        AS->>DB: _conversational: fixed reply or llm.chat with no SOURCES (answering.py:143)
        AS-->>UI: verdict conversational, cost measured zero if no model call
    else a real question with sources
        AS->>TR: ask_span (tracing.py:199)
        AS->>ST: search_detailed(question, session_id) inside retrieval_span
        ST-->>AS: SearchResult(hits, best_score, mode, floor_applied)
        alt search.relevant is false (best_score below floor, store.py:169)
            AS->>DB: _refuse: usage.not_called is a measured zero (answering.py:213)
            AS-->>UI: verdict no_match, no citations, floor reported in evidence
        else passages survived the floor
            AS->>LM: answer(question, hits, history) inside generation_span (llm.py:343)
            LM->>LM: _trim_context to MAX_CONTEXT_CHARS (llm.py:216)
            LM->>LM: _build_messages: SYSTEM_PROMPT, then PRIOR, SOURCES, QUESTION (llm.py:196)
            LM->>LM: _generate dispatches to groq / openai / ollama (llm.py:322)
            LM-->>AS: Grounded(text, cited, invalid, passages)
            AS->>DB: evidence(search, audit, answered, cost) builds the summary (payloads.py:55)
            AS->>DB: add_message(user) then add_message(assistant, citations, evidence) (db.py:211)
            AS->>DB: touch_session
            AS-->>UI: { answer, citations, evidence }
        end
    end
    UI->>UI: append turn, refreshStatus(), Answer.jsx renders evidence strip
```

### Inside `search_detailed` (`store.py:664`)

```mermaid
flowchart TD
    A["embed_query(question) embeddings.py:24"] --> B["pgvector cosine distance, LIMIT pool_size = TOP_K x 4, scoped to session store.py:694"]
    B --> C["distance to similarity: 1.0 - distance (store.py:706)"]
    C --> D{"HYBRID_ENABLED and BM25 has ids? store.py:730"}
    D -- no --> E["mode = dense, all hits admitted as dense"]
    D -- yes --> F["BM25 top + re-score unseen ids with score_against_query"]
    F --> G["RRF fuse of dense and lexical orderings (lexical.py:163)"]
    G --> H{"admission test store.py:778"}
    H -->|"score >= DENSE_STRONG"| I["admitted as dense"]
    H -->|"score >= floor and term coverage"| I2["admitted as both"]
    H -->|"rescue_allowed and coverage and score >= BM25_RESCUE_MIN"| J["admitted as lexical, rescued"]
    H -->|"otherwise"| K["dropped"]
    I --> L["numeric damping: numeric-heavy score x NUMERIC_DAMPING (store.py:811)"]
    I2 --> L
    J --> L
    L --> M["effective_floor = max floor, best_score x MIN_RATIO (store.py:830)"]
    M --> N{"cross-encoder rerank the head? (store.py:858)"}
    N --> O["rerank pool of RERANK_POOL, promote RERANK_TOP_K, mode becomes hybrid+rerank"]
    N --> P["keep fused order"]
    O --> Q["MAX_PER_SOURCE cap, then backfill to TOP_K (store.py:885)"]
    P --> Q
    Q --> R["SearchResult with hits, best_score, floor_applied, numeric_share, damped, mode"]
```

Why the floor sits *before* the reranker: the floor is the dense model's
calibrated decision and reranking after it only changes order, so paying for the
cross-encoder earlier would buy nothing. Why there are three admission rules
instead of one threshold: a strong term match with a mediocre dense score is
usually a real answer the embedding under-scored, and the eval corpus shows a
single global threshold cannot separate that case from noise.

---

## 5. The rest of the API

| Endpoint | Path through the code |
|---|---|
| `POST /api/summarize` | `chat.py:31` → `answering.summarize:252` → `store.search_detailed` → floor check → `llm.summarize:387` (SYSTEM_PROMPT + `TASK:`) → evidence written **without** `cost` → two rows |
| `GET /api/sources/{id}/chunks` | `sources.py:200` → `store.source_chunks:485`, session-scoped, ordered by `position`, adds `count_tokens` per chunk (`parsers.py:123`) → `ChunkMap` bar heights |
| `GET /api/sources/{id}/file` | `sources.py:154` → `store.original_file:467` (UPLOAD_DIR boundary) → `FileResponse`; scriptable content types forced to `text/plain` so uploaded HTML cannot run in the app's origin |
| `POST /api/sources/{id}/reindex` | `sources.py:231` → `store.reindex:333` re-parses the stored file with the current chunking, keeping the same source id |
| `GET /api/sources/occurrences?term=` | `sources.py:48` → `store.find_occurrences:588` → drawer highlight |
| `DELETE /api/sources/{id}` and `/api/sources` | `sources.py:279`, `sources.py:289` → `store.remove:424` / `store.clear:439`: rows, files, caches |
| `GET /api/status` | `meta.py:22` → `store.stats` + `db.usage_totals:309` + `tracing.status:77` + provider readiness |
| `/api/sessions` CRUD | `sessions.py` → `db.*`; `session_payload:96` attaches `recent_messages` for the transcript |

Error map, in one place: provider unreachable → `LLMUnavailable` → 503;
unknown session → 404; empty question or unsupported
file type → 400; a bug in retrieval or SQL → 500. The frontend turns any non-2xx
into an assistant-shaped error turn (`App.jsx:326`), so a failure is legible in
the thread instead of only in the console.

---

## 6. Where state lives

```
PostgreSQL (alembic, migrations/versions)
  0001 baseline      sessions, sources, chunks(embedding vector), messages
  0002 message_evidence   messages.evidence JSONB   (per-turn retrieval + cost)
  0003 source_url         sources.url
  0004 usage_events       kind, model, tokens, latency, session, created_at
  0005 ingest_observability  sources.token_max, fit_splits, avg_chars, max_chars
  0006 source_view        sources.file_bytes, size_splits  (nullable, unbackfilled)

UPLOAD_DIR        original files, named uuid4hex + suffix  (reindex needs no re-upload)

Caches            lru_cache SentenceTransformer (embeddings.py:9)
                  BM25 per (session, revision), rebuilt after _invalidate (store.py:207)

In process        usage.py latency window (P50/P95 over recent requests)
                  llm.py _last_usage, cleared before every call (llm.py:324)

Opt-in            LANGSMITH_TRACING spans → network, or JSONL when
                  LANGSMITH_TRACING_LOCAL=1 (tracing.py:77 reports the state)
```

Money and tokens: the database stores **tokens**, never dollars. Dollars are
computed at read time from a dated price table so the persisted fact cannot go
stale — see `README.md` "What a question costs" (line 1161) and
`PRESENTATION.md:743` for the current rationale, and the cost dashboard plan for
the rate table and its provenance.

---

## 7. Reading order, if you are new to this repo

1. `app/answering.py:1` — the whole policy in one readable function.
2. `app/store.py:664` — retrieval, with the reasoning for each step inline.
3. `app/llm.py:7` — `SYSTEM_PROMPT`, then `_build_messages` for what the model
   actually sees.
4. `app/parsers.py:1096` — how an arbitrary file becomes `Block`s.
5. `README.md` — the long-form narrative of the same flows, plus the eval
   numbers behind the chunking and retrieval choices.
6. `PRESENTATION.md` — shorter, decision-oriented version of the above.
