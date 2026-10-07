import React, { useEffect, useState } from "react";
import { api } from "./api.js";

/**
 * Two views of the same thing the rest of the drawer only counts.
 *
 * `SourceDetail` answers "what happened to *this* file", one row per fact and
 * only when the row is opened. `PipelinePanel` answers "what happens to any
 * file", once, because the pipeline itself does not vary by document - only the
 * parser it enters at step two does. Both are kept out of `SourcesDrawer` since
 * that file is the list, and this is commentary on it.
 */

/**
 * The pipeline as a diagram: six nodes on a rail, one short line of text each.
 *
 * Written as a diagram on purpose. The steps were paragraphs first and nobody
 * read past the second one - what a reader wants from a pipeline is the shape
 * of it and where the document currently is, which is what a rail of icons
 * gives at a glance and prose does not. Anything worth more than a line lives
 * in the source's own detail panel, where it is a number attached to a file
 * rather than a claim about all of them.
 */
const ICONS = {
  // Written under a generated name, then hashed for the duplicate check.
  upload: (
    <>
      <path d="M9 11.5V3M9 3 6.2 5.8M9 3l2.8 2.8" />
      <path d="M3 11v3a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2v-3" />
    </>
  ),
  // A document whose lines keep their order.
  document: (
    <>
      <path d="M4 2.5h5.5L14 7v8.5a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-11a1 1 0 0 1 1-1Z" />
      <path d="M9.5 2.5V7H14M5.5 10h6M5.5 12.5h4" />
    </>
  ),
  // One block, divided only where it must be.
  split: (
    <>
      <rect x="2.5" y="4.5" width="13" height="9" rx="1.5" />
      <path d="M9 2.5v13" strokeDasharray="2 2.2" />
    </>
  ),
  // Two ceilings pressing down from above.
  ceilings: (
    <>
      <path d="M2.5 4h13" />
      <path d="M6 7.5v5M6 12.5 4.6 11.1M6 12.5l1.4-1.4" />
      <path d="M12 7.5v5M12 12.5l-1.4-1.4M12 12.5l1.4-1.4" />
    </>
  ),
  // A grid of numbers, which is what a vector is.
  vector: (
    <>
      <path
        d="M4.5 4.5h.01M9 4.5h.01M13.5 4.5h.01M4.5 9h.01M9 9h.01M13.5 9h.01M4.5 13.5h.01M9 13.5h.01M13.5 13.5h.01"
        strokeWidth="2.4"
        strokeLinecap="round"
      />
    </>
  ),
  // The database cylinder.
  store: (
    <>
      <ellipse cx="9" cy="4.6" rx="5.5" ry="2.1" />
      <path d="M3.5 4.6v8.8c0 1.16 2.46 2.1 5.5 2.1s5.5-.94 5.5-2.1V4.6" />
      <path d="M3.5 9c0 1.16 2.46 2.1 5.5 2.1s5.5-.94 5.5-2.1" />
    </>
  ),
};

const STAGES = [
  { icon: "upload", title: "Upload and hash", tag: "SHA-256 · skipped if already here" },
  { icon: "document", title: "Read by format", tag: "structure read as structure" },
  { icon: "split", title: "Split by structure", tag: "whole units, cut only if too big" },
  { icon: "ceilings", title: "Fit both ceilings", tag: "characters, then wordpieces" },
  { icon: "vector", title: "Embed locally", tag: "MiniLM-L6-v2 · 384 dimensions" },
  { icon: "store", title: "Store for retrieval", tag: "Postgres + pgvector" },
];

function kilobytes(bytes) {
  if (bytes === null || bytes === undefined) return "size not recorded";
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  return `${kb >= 10 ? kb.toFixed(0) : kb.toFixed(1)} KB`;
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

/**
 * The chain of steps that actually ran for one source, as one line.
 *
 * Counts rather than adjectives: "3 blocks → 3 chunks → 0 cut by size" is
 * checkable against the file, where "indexed successfully" is not. The two
 * ceiling counts are reported separately because they fire for unrelated
 * reasons and a reader is entitled to know which one did the work.
 */
function stepsFor(source, tokenWindow) {
  // `token_max` is the flag for whether this source was ever walked through the
  // model's tokenizer. It is 0 for a source indexed before that walk was
  // recorded, and it cannot be 0 for one that was - a chunk has at least one
  // wordpiece. `fit_splits` shares the pass, so 0 there means "unmeasured" too,
  // and only reading it alongside `token_max` keeps that from being printed as
  // "no cuts", which is a different statement about the file.
  const counted = Boolean(source.token_max);
  const steps = [`read ${String(source.kind || "").toUpperCase()}`];
  steps.push(plural(source.pages, "block"));
  steps.push(plural(source.chunks, "chunk"));
  // Null before 0 for the size ceiling: "0 cuts" is an ordinary result, so 0
  // cannot carry both meanings here the way it can for `token_max`.
  steps.push(
    source.size_splits === null || source.size_splits === undefined
      ? "size cuts not measured"
      : source.size_splits === 0
        ? "no size cuts"
        : `${source.size_splits} size ${source.size_splits === 1 ? "cut" : "cuts"}`
  );
  steps.push(
    !counted
      ? "token cuts not measured"
      : source.fit_splits === 0
        ? "no wordpiece cuts"
        : `${source.fit_splits} wordpiece ${source.fit_splits === 1 ? "cut" : "cuts"}`
  );
  steps.push(counted ? `worst ${source.token_max}/${tokenWindow} wordpieces` : "wordpieces not measured");
  steps.push(source.numeric ? plural(source.numeric, "table chunk") : "no table chunks");
  steps.push("embedded 384-d");
  return steps;
}

/**
 * How this file was cut, as a picture of the cut.
 *
 * One bar per stored chunk, in document order, wrapping the way the document
 * wraps - so a run of short pieces beside one long piece is visible before
 * anything is read. Width is measured against the size ceiling rather than
 * against the longest chunk, because the question is "how close to the limit
 * is this", not "how does it compare to itself".
 *
 * Fetched when the row opens rather than with the status: this is one
 * document's worth of text, and a notebook with a dozen sources should not pay
 * for the eleven nobody opened.
 */
function ChunkMap({ source, sessionId }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(null);

  useEffect(() => {
    let alive = true;
    setData(null);
    setError(null);
    setOpen(null);
    // `Promise.resolve().then` rather than a bare call: this runs inside a
    // render effect, so a synchronously thrown failure would take the whole
    // tree down instead of landing in the message below.
    Promise.resolve()
      .then(() => api.sourceChunks(sessionId, source.id))
      .then((payload) => {
        if (alive) setData(payload);
      })
      .catch((err) => {
        if (alive) setError(err.message);
      });
    return () => {
      alive = false;
    };
  }, [sessionId, source.id]);

  if (error) return <p className="chunk-note">Could not draw the chunk map: {error}</p>;
  if (!data) return <p className="chunk-note">Measuring every chunk…</p>;

  const chunks = data.chunks || [];
  if (!chunks.length) {
    return <p className="chunk-note">No chunks are stored for this source.</p>;
  }

  const window = data.ceiling.wordpieces;
  const ceiling = data.ceiling.chars || 1;
  const lengths = chunks.map((c) => c.chars).sort((a, b) => a - b);
  const at = (q) => lengths[Math.min(lengths.length - 1, Math.floor(q * lengths.length))];
  const selected = chunks.find((c) => c.position === open) || null;
  const anyOver = window > 0 && chunks.some((c) => c.wordpieces > window);
  // Widen the bars for a short document rather than leaving three slivers in an
  // empty box: the picture should fill the space it was given either way.
  const barWidth = Math.max(5, Math.min(16, Math.floor(300 / chunks.length)));
  // Rows get shorter as the count grows, because every bar is one row of the
  // picture and a 264-chunk file at full height would fill the drawer with it.
  const rowHeight = chunks.length > 120 ? 24 : chunks.length > 40 ? 32 : 48;

  return (
    <div className="chunkmap">
      <div className="chunkmap-head">
        <span>Every chunk, in order</span>
        <span className="chunkmap-scale">
          height = size against the {ceiling}-char ceiling
        </span>
      </div>

      {/* `--row` is the pitch the ceiling lines are drawn at: one line per row
          of bars, however they wrap, so the line always means "at this ceiling". */}
      <div className="chunk-map" style={{ "--row": `${rowHeight}px` }}>
        {chunks.map((c) => {
          const fill = Math.max(3, Math.min(100, Math.round((100 * c.chars) / ceiling)));
          const over = window > 0 && c.wordpieces > window;
          return (
            <button
              key={c.position}
              type="button"
              className={`chunk-bar${c.numeric ? " numeric" : ""}${over ? " over" : ""}${
                open === c.position ? " on" : ""
              }`}
              style={{ width: `${barWidth}px`, height: `${rowHeight}px`, "--fill": `${fill}%` }}
              title={`#${c.position + 1} · ${c.chars} chars · ${c.wordpieces} wordpieces${
                c.heading ? ` · ${c.heading}` : ""
              }`}
              aria-label={`Chunk ${c.position + 1}: ${c.chars} characters, ${c.wordpieces} wordpieces`}
              aria-pressed={open === c.position}
              onClick={() => setOpen(open === c.position ? null : c.position)}
            />
          );
        })}
      </div>

      <div className="chunkmap-legend">
        <span>
          <i className="sw" /> text
        </span>
        <span>
          <i className="sw numeric" /> tables
        </span>
        {anyOver ? (
          <span>
            <i className="sw over" /> past the wordpiece window
          </span>
        ) : null}
        <span className="chunkmap-range">
          {lengths[0]} min · {at(0.5)} median · {at(0.9)} p90 · {lengths[lengths.length - 1]}{" "}
          max chars
        </span>
      </div>

      {selected ? (
        <div className="chunk-view">
          <div className="chunk-view-head">
            <strong>#{selected.position + 1}</strong>
            {selected.page ? <span>p.{selected.page}</span> : null}
            <span>{selected.chars} chars</span>
            <span>
              {selected.wordpieces}/{window || "?"} wordpieces
            </span>
            {selected.numeric ? <span className="chunk-tag">table</span> : null}
          </div>
          {selected.heading ? <p className="chunk-heading">{selected.heading}</p> : null}
          <p className="chunk-text">{selected.text}</p>
        </div>
      ) : (
        <p className="chunk-hint">Click a bar to read that chunk as it was indexed.</p>
      )}
    </div>
  );
}

export function SourceDetail({ source, sessionId, tokenWindow = 256, duplicate = false }) {
  // The original document, not a re-render of it. A fetch would mean
  // re-implementing the PDF viewer and the download prompt the browser already
  // ships with, and the point of showing it is that it is *the* file.
  const fileUrl = api.sourceFileUrl(sessionId, source.id);

  return (
    <div className="source-detail">
      <div className="detail-actions">
        <a className="detail-view" href={fileUrl} target="_blank" rel="noreferrer noopener">
          View original
        </a>
        <span className="detail-hint">opens in a new tab</span>
      </div>

      <p className="detail-steps">
        {stepsFor(source, tokenWindow).map((step, index) => (
          <React.Fragment key={step}>
            {index > 0 ? <span aria-hidden="true"> › </span> : null}
            <span>{step}</span>
          </React.Fragment>
        ))}
      </p>

      <dl className="detail-facts">
        <dt>File</dt>
        <dd title={source.name}>
          {String(source.kind || "").toUpperCase()} · {kilobytes(source.file_bytes)}
        </dd>

        <dt>Indexed</dt>
        <dd>{source.created_at ? `${source.created_at} UTC` : "before this was recorded"}</dd>

        <dt>Chunk sizes</dt>
        <dd title="average and longest stored chunk, in characters">
          {source.avg_chars} avg · {source.max_chars} max chars
        </dd>

        <dt>Ceilings</dt>
        <dd
          title={`the character ceiling is what cuts an over-long block; the ${
            tokenWindow
          }-wordpiece window is what the embedding model keeps, and only the second can truncate silently`}
        >
          {source.size_splits === null || source.size_splits === undefined
            ? "size cuts not measured"
            : `${source.size_splits} cut by size`}
          {" · "}
          {!source.token_max
            ? "token cuts not measured"
            : source.fit_splits === 0
              ? "0 re-cut for tokens"
              : `${source.fit_splits} re-cut for tokens`}
        </dd>

        <dt>Tables</dt>
        <dd title="numeric-heavy chunks; retrieval damps these so a column of numbers cannot win on digits alone">
          {source.numeric} of {source.chunks} chunks
          {source.chunks ? ` (${Math.round((100 * source.numeric) / source.chunks)}%)` : ""}
        </dd>
      </dl>

      {/* The rest of the panel is what happened to the file; this is what
          happened to its contents. Fetched here so opening one row costs one
          document, not the whole notebook. */}
      <ChunkMap source={source} sessionId={sessionId} />

      {duplicate ? (
        <p className="detail-duplicate">
          This upload matched a file already in this notebook. Nothing was re-indexed.
        </p>
      ) : null}
    </div>
  );
}

/**
 * The diagram, closed by default and opened on demand.
 *
 * Deliberately the same six stages for every document and deliberately not
 * attached to a source: the pipeline is what the app does, and a per-file view
 * of it would imply the steps differ when only the format parser does. The
 * counts live in `SourceDetail` where they are checkable against one file.
 */
export function PipelinePanel() {
  const [open, setOpen] = useState(false);

  return (
    <div className="occ-holder pipeline-holder">
      <button
        className="occ-toggle"
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        {open ? "Hide how indexing works" : "How indexing works"}
      </button>
      {open ? (
        <ol className="pipeline">
          {STAGES.map((stage) => (
            <li className="stage" key={stage.title}>
              <span className="stage-node" aria-hidden="true">
                <svg
                  viewBox="0 0 18 18"
                  width="17"
                  height="17"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  {ICONS[stage.icon]}
                </svg>
              </span>
              <span className="stage-body">
                <strong>{stage.title}</strong>
                <span className="stage-tag">{stage.tag}</span>
              </span>
            </li>
          ))}
        </ol>
      ) : null}
    </div>
  );
}
