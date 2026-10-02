import React, { useRef, useState } from "react";

export function Rail({
  status,
  sessions,
  activeId,
  onSelect,
  onCreate,
  onDelete,
  onUpload,
  onDeleteSource,
  onClearSources,
  uploading,
}) {
  const fileInput = useRef(null);
  const [dragging, setDragging] = useState(false);

  async function ingest(files) {
    if (!files || !files.length) return;
    for (const file of files) await onUpload(file);
  }

  return (
    <aside className="rail">
      <div className="brand">
        NotebookLM <span>local</span>
      </div>

      <div className="rail-section">
        <div className="rail-label">
          Sessions
          <button className="icon-btn" onClick={onCreate} title="New session" aria-label="New session">
            +
          </button>
        </div>
        <div className="sessions" role="listbox" aria-label="Sessions">
          {sessions.length === 0 ? (
            <p className="rail-empty">No sessions yet.</p>
          ) : (
            sessions.map((session) => (
              <div
                key={session.id}
                className="session"
                aria-current={session.id === activeId}
                role="option"
                aria-selected={session.id === activeId}
                tabIndex={0}
                onClick={() => onSelect(session.id)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelect(session.id);
                  }
                }}
              >
                <span className="session-name" title={session.name}>
                  {session.name}
                </span>
                <button
                  className="icon-btn danger"
                  title={`Delete ${session.name}`}
                  aria-label={`Delete ${session.name}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(session);
                  }}
                >
                  &times;
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="rail-section">
        <div className="rail-label">
          Sources
          {status && status.sources.length > 0 ? (
            <button
              className="icon-btn"
              title="Remove all sources from this session"
              aria-label="Remove all sources"
              onClick={onClearSources}
            >
              &times;
            </button>
          ) : null}
        </div>

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

        <div className="sources" style={{ marginTop: 10 }}>
          {!status || status.sources.length === 0 ? (
            <p className="rail-empty">Nothing indexed in this session.</p>
          ) : (
            status.sources.map((source) => (
              <div className="source" key={source.id}>
                <span className="source-kind">{source.kind}</span>
                <span className="source-name" title={source.name}>
                  {source.name}
                </span>
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
      </div>

      {status ? (
        <div className="rail-foot">
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
            <dt>Floor</dt>
            <dd>{Math.round(status.min_score * 100)}%</dd>
          </dl>
        </div>
      ) : null}
    </aside>
  );
}