import React, { useState } from "react";
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

const STEPS = [
  {
    title: "1 · Written and hashed",
    body:
      "The bytes are stored under a generated name and hashed (SHA-256). If this notebook already holds that hash the file is not indexed again - you get the source you already had.",
  },
  {
    title: "2 · Read by its own format",
    body:
      "PDF, DOCX, Markdown, HTML, CSV, JSON, config, logs, code and prose each have their own reader. Structure is read as structure: heading paths, page numbers, function names, column headers, timestamp groups.",
  },
  {
    title: "3 · Split on structure first",
    body:
      "A chunk is a whole structural unit whenever the document gave you one. Only a block larger than the size ceiling is cut - on paragraph, then sentence, then word - and tables on line boundaries so a row is never sliced in half.",
  },
  {
    title: "4 · Both ceilings enforced",
    body:
      "First the character ceiling, then the embedding model's wordpiece window, counted with the model's own tokenizer rather than estimated. Anything over is re-cut, so no stored chunk is longer than the model can actually see. Which of the two fired, and by how much, is per file in the row above.",
  },
  {
    title: "5 · Embedded on this machine",
    body:
      "all-MiniLM-L6-v2 turns each chunk into 384 numbers, locally. The vectors are never uploaded and never leave the machine.",
  },
  {
    title: "6 · Stored for retrieval",
    body:
      "Text, heading, page number and vector go into Postgres with pgvector, where dense cosine and BM25 are fused for every question you ask.",
  },
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

      {duplicate ? (
        <p className="detail-duplicate">
          This upload matched a file already in this notebook. Nothing was re-indexed.
        </p>
      ) : null}
    </div>
  );
}

/**
 * The explainer, closed by default and opened on demand.
 *
 * Deliberately the same six steps for every document and deliberately not
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
          {STEPS.map((step) => (
            <li key={step.title}>
              <strong>{step.title}</strong>
              <p>{step.body}</p>
            </li>
          ))}
        </ol>
      ) : null}
    </div>
  );
}
