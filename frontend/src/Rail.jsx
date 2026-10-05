import React, { useState } from "react";

/**
 * Session switching, and nothing else.
 *
 * Sources and model details moved into a drawer (`SourcesDrawer`) so the
 * conversation is what occupies the screen. When you are mid-question the
 * documents are context you already have; they are something you open to check
 * something, not something you read alongside.
 */
export function Rail({ sessions, activeId, onSelect, onCreate, onDelete }) {
  const [query, setQuery] = useState("");

  // Case-insensitive substring match on the name. No fuzzy matching and no
  // ranking: with a few dozen notebooks a substring is what the reference
  // product does, and a wrong match would hide a notebook that exists.
  const needle = query.trim().toLowerCase();
  const visible = needle
    ? sessions.filter((s) => s.name.toLowerCase().includes(needle))
    : sessions;

  return (
    <aside className="rail">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true" />
        NotebookLM <span>local</span>
      </div>

      <div className="rail-section">
        <div className="rail-label">
          Notebooks
          <button
            className="icon-btn"
            onClick={onCreate}
            title="New notebook"
            aria-label="New notebook"
          >
            +
          </button>
        </div>
        <div className="rail-search">
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search notebooks"
            aria-label="Search notebooks"
          />
        </div>
        {/* What is called a session everywhere else in the system is called a
            notebook here, because that is the name the reference product uses
            for the same concept: a named set of sources plus its own
            conversation. The rename is cosmetic on purpose - the API, the
            database and the tests all still say session, because "notebook"
            describes the concept and "session" describes the isolation
            boundary, and the boundary is what the server actually enforces.
            Renaming the plumbing would imply a semantic change that is not
            being made. */}
        <div className="sessions" role="listbox" aria-label="Notebooks">
          {visible.length === 0 ? (
            <p className="rail-empty">
              {sessions.length === 0 ? "No notebooks yet." : "No notebooks match."}
            </p>
          ) : (
            visible.map((session) => (
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
    </aside>
  );
}
