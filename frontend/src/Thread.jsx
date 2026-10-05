import React, { useEffect, useRef } from "react";
import { Answer, Citation, Evidence, useCitationLinks } from "./Answer.jsx";

function Thinking() {
  return (
    <div className="thinking">
      <span className="dots" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
      Reading your sources
    </div>
  );
}

/**
 * One turn in the thread.
 *
 * A live answer carries `text` plus `citations` and `evidence`. A reopened one
 * carries `content`, plus the same two fields — the server persists them, so a
 * citation chip still works after a reload. `content` is deliberately not
 * renamed to `text` at the fetch boundary: the field name the API persists is
 * the one that wins, and renaming it in the client is how the two drifted apart
 * in the first place.
 *
 * The fallbacks are for turns written before the evidence was stored. They keep
 * an old transcript readable rather than rendering a bare string.
 */
function Turn({ turn }) {
  const text = turn.text ?? turn.content ?? "";
  const citations = turn.citations || [];
  const evidence = turn.evidence || null;

  return (
    <article className={`turn ${turn.role}`}>
      <div className="who">
        <span className="dot" aria-hidden="true" />
        {turn.role === "user" ? "You" : "NotebookLM"}
      </div>
      {turn.error ? (
        <div className="answer error">{text}</div>
      ) : (
        <>
          <Answer
            text={text}
            cited={evidence?.cited}
            linkable={citations.length > 0}
          />
          <Evidence evidence={evidence} />
          {citations.length ? (
            <div className="cites">
              {citations.map((citation, index) => (
                <Citation
                  key={`${citation.source}-${citation.page}-${index}`}
                  citation={citation}
                  index={index}
                  cited={evidence?.cited}
                />
              ))}
            </div>
          ) : null}
        </>
      )}
    </article>
  );
}

export function Thread({
  turns,
  pending,
  suggestions = [],
  onSuggest,
  onUpload,
  sourceCount = 0,
  notebookName = "",
  createdAt = "",
}) {
  const ref = useRef(null);
  useCitationLinks(ref);

  useEffect(() => {
    const node = ref.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [turns, pending]);

  if (!turns.length && !pending) {
    // Modelled on the reference product's opening screen: a greeting, then a
    // count and a date, then either an onboarding choice or questions to start.
    //
    // The grounding promise is stated once, in the footer line rather than as a
    // paragraph of rules. It used to be the whole empty state, which was more
    // explicit and read as a terms-of-service notice before the user had done
    // anything. It now lives where the reference product puts it: underneath.
    const hasSources = sourceCount > 0;
    return (
      <div className="thread" ref={ref}>
        <div className="thread-inner">
          <div className="empty">
            <h2>
              {hasSources
                ? `Hello, what would you like to know about ${notebookName}?`
                : "Hello. Add a source to get started."}
            </h2>
            <p className="empty-meta">
              {sourceCount} source{sourceCount === 1 ? "" : "s"}
              {createdAt ? (
                <>
                  {" · "}
                  <time>{createdAt}</time>
                </>
              ) : null}
            </p>

            {/* Onboarding, shown only while the notebook is empty. These are
                the two things that can actually be done next, so they are the
                only things offered - a suggestion question here would be
                guaranteed to be refused. */}
            {!hasSources ? (
              <div className="onboard">
                <button type="button" className="onboard-card" onClick={onUpload}>
                  <span className="onboard-title">I want to upload my own documents</span>
                  <span className="onboard-sub">
                    PDF, DOCX, Markdown, CSV and more. Answers come only from what
                    you upload.
                  </span>
                </button>
                <div className="onboard-card disabled" aria-disabled="true">
                  <span className="onboard-title">
                    Search the web for sources
                  </span>
                  <span className="onboard-sub">
                    Not available in this local build — nothing leaves your machine.
                  </span>
                </div>
              </div>
            ) : onSuggest && suggestions.length ? (
              <ul className="suggestions">
                {suggestions.map((question) => (
                  <li key={question}>
                    <button type="button" onClick={() => onSuggest(question)}>
                      {question}
                    </button>
                  </li>
                ))}
              </ul>
            ) : null}

            <p className="empty-foot">
              Every answer is drawn from your sources with the exact passage cited,
              and nothing is invented.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="thread" ref={ref}>
      <div className="thread-inner">
        {turns.map((turn, i) => (
          <Turn key={i} turn={turn} />
        ))}
        {pending ? (
          <article className="turn assistant">
            <div className="who">
              <span className="dot" aria-hidden="true" />
              NotebookLM
            </div>
            <Thinking />
          </article>
        ) : null}
      </div>
    </div>
  );
}