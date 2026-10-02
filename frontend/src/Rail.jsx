import React from "react";

/**
 * Session switching, and nothing else.
 *
 * Sources and model details moved into a drawer (`SourcesDrawer`) so the
 * conversation is what occupies the screen. When you are mid-question the
 * documents are context you already have; they are something you open to check
 * something, not something you read alongside.
 */
export function Rail({ sessions, activeId, onSelect, onCreate, onDelete }) {
  return (
    <aside className="rail">
      <div className="brand">
        NotebookLM <span>local</span>
      </div>

      <div className="rail-section">
        <div className="rail-label">
          Sessions
          <button
            className="icon-btn"
            onClick={onCreate}
            title="New session"
            aria-label="New session"
          >
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
    </aside>
  );
}
