import React, { useEffect, useRef, useState } from "react";
import { Occurrences } from "./Highlight.jsx";

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
  onDeleteSource,
  onClearSources,
  onFindOccurrences,
  autoSearch = null,
}) {
  const fileInput = useRef(null);
  const closeButton = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [finding, setFinding] = useState(false);

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
              title="Remove all sources from this session"
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

          <div className="sources">
            {count === 0 ? (
              <p className="sources-empty">Nothing indexed in this session yet.</p>
            ) : (
              status.sources.map((source) => (
                <div className="source" key={source.id}>
                  <span className="source-kind">{source.kind}</span>
                  <span className="source-name" title={source.name}>
                    {source.name}
                  </span>
                  {source.chunks !== undefined ? (
                    <span className="source-meta">{source.chunks}</span>
                  ) : null}
                  <button
                    className="icon-btn danger"
                    title={`Remove ${source.name}`}
                    aria-label={`Remove ${source.name}`}
                    onClick={() => onDeleteSource(source)}
                  >
                    &times;
                  </button>
                </div>
              ))
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
              <dt>Relevance floor</dt>
              <dd>{Math.round(status.min_score * 100)}%</dd>
            </dl>
          </div>
        ) : null}
      </div>
    </>
  );
}
