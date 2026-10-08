import React, { useEffect, useRef, useState } from "react";
import { Occurrences } from "./Highlight.jsx";
import { PipelinePanel, SourceDetail } from "./IngestPanel.jsx";

/**
 * The uploaded documents for the active session, in a drawer over the thread.
 *
 * Also carries the index statistics, which are only interesting while someone
 * is thinking about whether the sources are enough — so they belong here rather
 * than permanently occupying a column.
 *
 * `onFindOccurrences` is optional. Without it there is nothing to search, so the
 * highlight control is left out rather than shown and broken.
 */
export function SourcesDrawer({
  status,
  uploading,
  onClose,
  onUpload,
  onWebSearch,
  searching = false,
  onDeleteSource,
  onReindexSource = null,
  reindexingId = null,
  onClearSources,
  onFindOccurrences,
  autoSearch = null,
  // Passed through to the per-source detail: the file endpoint needs to know
  // whose notebook it is, and the wordpiece window comes from the server's
  // config rather than a number written into the component.
  sessionId = null,
  duplicateIds = null,
}) {
  const fileInput = useRef(null);
  const closeButton = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [finding, setFinding] = useState(false);
  const [query, setQuery] = useState("");
  // Which source, if any, has its ingestion detail open. One at a time: the
  // drawer is already a scroll region, and six expanded rows would push the
  // list out of reach for the sake of information one at a time answers.
  const [openId, setOpenId] = useState(null);

  // Escape closes, and focus moves into the drawer so keyboard users are not
  // left tabbing through the thread behind the overlay.
  useEffect(() => {
    closeButton.current?.focus();
    function onKey(e) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function ingest(files) {
    if (!files || !files.length) return;
    for (const file of files) await onUpload(file);
  }

  const count = status ? status.sources.length : 0;

  return (
    <>
      <div className="scrim" onClick={onClose} aria-hidden="true" />
      <div className="drawer" role="dialog" aria-modal="true" aria-label="Sources">
        <div className="drawer-head">
          <h2>Sources</h2>
          {count > 0 ? (
            <button
              className="icon-btn"
              title="Remove all sources from this notebook"
              aria-label="Remove all sources"
              onClick={onClearSources}
            >
              &times;
            </button>
          ) : null}
          <button
            className="icon-btn"
            onClick={onClose}
            title="Close"
            aria-label="Close sources"
            ref={closeButton}
          >
            &times;
          </button>
        </div>

        <div className="drawer-body">
          <label
            className={`drop${dragging ? " over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              ingest(e.dataTransfer.files);
            }}
          >
            <input
              ref={fileInput}
              type="file"
              multiple
              hidden
              onChange={(e) => {
                ingest(e.target.files);
                e.target.value = "";
              }}
            />
            <strong>{uploading ? "Indexing…" : "Add sources"}</strong>
            <small>Drop files or click · PDF, DOCX, MD, CSV, TXT</small>
          </label>

          {/* Web search lives here as well as on the empty-state card, because
              a notebook is rarely empty for long: once two files are indexed the
              card is gone, and being unable to add a web source without clearing
              the notebook would be an odd restriction to design in.
              `disabled` while a search runs so one slow provider call cannot be
              fired twice. */}
          {onWebSearch ? (
            <form
              className="websearch"
              onSubmit={(e) => {
                e.preventDefault();
                if (!query.trim() || searching) return;
                onWebSearch(query.trim());
                setQuery("");
              }}
            >
              <strong>Search the web</strong>
              <div className="websearch-row">
                <input
                  type="search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g. vector database indexing"
                  aria-label="Web search query"
                  disabled={searching}
                />
                <button type="submit" disabled={searching || !query.trim()}>
                  {searching ? "Searching…" : "Add"}
                </button>
              </div>
              <small>
                Finds pages and indexes them like uploads. Takes 10–20 seconds.
              </small>
            </form>
          ) : null}

          <div className="sources">
            {count === 0 ? (
              <p className="sources-empty">Nothing indexed in this notebook yet.</p>
            ) : (
              status.sources.map((source) => {
                const open = openId === source.id;
                const detailId = `ingest-${source.id}`;
                const tokenWindow =
                  (status.ingest && status.ingest.token_window) || 256;
                return (
                  <div className={`source${open ? " open" : ""}`} key={source.id}>
                    {/* The row is not itself a button: the name may be a link
                        for a web source, and nesting a link in a button is not
                        valid HTML. The chevron is the one control that opens
                        the detail, and it says what it opens. */}
                    <button
                      className="source-expand"
                      type="button"
                      aria-expanded={open}
                      aria-controls={open ? detailId : undefined}
                      title={
                        open
                          ? "Hide what happened to this file"
                          : "How this file was indexed, and its original"
                      }
                      aria-label={`${open ? "Hide" : "Show"} ingestion details for ${source.name}`}
                      onClick={() => setOpenId(open ? null : source.id)}
                    >
                      <span aria-hidden="true">{open ? "▾" : "▸"}</span>
                    </button>
                    <span className="source-kind">{source.kind}</span>
                    {/* A web source links back to the page it came from. That link
                        is the only way a user can check the app indexed what they
                        think it did, which is the whole concern with pulling in
                        pages from the open internet. */}
                    <span className="source-name" title={source.name}>
                      {source.url ? (
                        <a href={source.url} target="_blank" rel="noreferrer noopener">
                          {source.name}
                        </a>
                      ) : (
                        source.name
                      )}
                    </span>
                    {source.chunks !== undefined ? (
                      <span className="source-meta">{source.chunks}</span>
                    ) : null}
                    {/* Only offered when the app can actually do it. Without
                        `onReindexSource` there is nothing to call, so the control
                        is left out rather than shown and dead. */}
                    {onReindexSource ? (
                      <button
                        className="icon-btn reindex"
                        title={`Re-index ${source.name}`}
                        aria-label={`Re-index ${source.name}`}
                        disabled={reindexingId === source.id}
                        onClick={() => onReindexSource(source)}
                      >
                        {reindexingId === source.id ? "\u2026" : "\u21BB"}
                      </button>
                    ) : null}
                    <button
                      className="icon-btn danger"
                      title={`Remove ${source.name}`}
                      aria-label={`Remove ${source.name}`}
                      onClick={() => onDeleteSource(source)}
                    >
                      &times;
                    </button>
                    {/* Only rendered while open: `aria-controls` points at it
                        only then, and nothing is in the document that the row
                        does not claim to be showing. */}
                    {open ? (
                      <div className="source-detail-wrap" id={detailId}>
                        <SourceDetail
                          source={source}
                          sessionId={sessionId}
                          tokenWindow={tokenWindow}
                          duplicate={Boolean(duplicateIds && duplicateIds.has(source.id))}
                        />
                      </div>
                    ) : null}
                  </div>
                );
              })
            )}
          </div>

          {onFindOccurrences ? (
            <div className="occ-holder">
              <button
                className="occ-toggle"
                type="button"
                aria-expanded={finding || Boolean(autoSearch)}
                onClick={() => setFinding((open) => !open)}
              >
                {finding || autoSearch ? "Close highlight" : "Highlight a word"}
              </button>
              {finding || autoSearch ? (
                <Occurrences
                  onFind={onFindOccurrences}
                  onClose={() => setFinding(false)}
                  autoSearch={autoSearch}
                />
              ) : null}
            </div>
          ) : null}

          {/* Last, closed, and separate from both the list and the stats
              footer: the pipeline is not per-source and it is not a statistic,
              so putting it anywhere else would be claiming it is one or the
              other. */}
          <PipelinePanel />
        </div>

        {status ? (
          <div className="drawer-foot">
            <dl>
              <dt>Model</dt>
              <dd title={status.model}>{status.model}</dd>
              <dt>Provider</dt>
              <dd>{status.provider}</dd>
              <dt>Embeddings</dt>
              <dd title={status.embed_model}>{status.embed_model.split("/").pop()}</dd>
              <dt>Passages</dt>
              <dd>
                {status.chunks} chunk{status.chunks === 1 ? "" : "s"}
              </dd>
              {/* The ingestion report: what chunking did, not just how much of
                  it there is. Two ceilings apply to a chunk - characters here,
                  wordpieces for the embedding model - and only the second one
                  can silently truncate. Reported so "indexed" can be checked
                  rather than assumed; `unmeasured` counts sources indexed
                  before this was recorded, whose zeros mean "unknown". */}
              {status.ingest && status.ingest.chunks ? (
                <>
                  <dt>Chunk sizes</dt>
                  <dd
                    title={`average ${status.ingest.avg_chars} characters, longest ${
                      status.ingest.max_chars
                    }; ${status.ingest.numeric_pct}% are numeric tables, which retrieval damps`}
                  >
                    {status.ingest.avg_chars} avg · {status.ingest.max_chars} max ·{" "}
                    {status.ingest.numeric_pct}% numeric
                  </dd>
                  <dt>Wordpieces</dt>
                  <dd
                    title={
                      status.ingest.token_max
                        ? `worst chunk tokenises to ${status.ingest.token_max} of the ${status.ingest.token_window} the embedding model keeps${
                            status.ingest.fit_splits
                              ? `; ${status.ingest.fit_splits} chunk(s) were cut again to fit`
                              : "; nothing was cut to fit"
                          }${status.ingest.unmeasured ? `; ${status.ingest.unmeasured} source(s) indexed before this was measured` : ""}`
                        : `${status.ingest.unmeasured} source(s) indexed before this was measured`
                    }
                  >
                    {status.ingest.token_max
                      ? `${status.ingest.token_max}/${status.ingest.token_window}`
                      : "not measured"}
                    {status.ingest.fit_splits > 0
                      ? ` · ${status.ingest.fit_splits} split to fit`
                      : ""}
                  </dd>
                </>
              ) : null}
              <dt>Relevance floor</dt>
              <dd>{Math.round(status.min_score * 100)}%</dd>
              {/* Tokens here, dollars in the cost dashboard above. A price
                  table goes stale, so the dollar figure only means something
                  with the as-of date and source it travels with - which is why
                  the panel prints both. Searches are broken out because they
                  are the spend the response used to throw away. */}
              {status.usage ? (
                <>
                  <dt>Tokens spent</dt>
                  <dd
                    title={`${status.usage.llm_prompt_tokens.toLocaleString()} in / ${status.usage.llm_completion_tokens.toLocaleString()} out from answers, ${
                      status.usage.web_prompt_tokens
                    } in / ${status.usage.web_completion_tokens} out from web search`}
                  >
                    {status.usage.prompt_tokens.toLocaleString()} in,{" "}
                    {status.usage.completion_tokens.toLocaleString()} out
                  </dd>
                  <dt>Web searches</dt>
                  <dd>
                    {status.usage.searches} recorded
                    {status.usage.web_search_ms > 0
                      ? ` · ${(status.usage.web_search_ms / 1000).toFixed(1)}s`
                      : ""}
                  </dd>
                </>
              ) : null}
            </dl>
          </div>
        ) : null}
      </div>
    </>
  );
}
