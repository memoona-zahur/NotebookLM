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

export function Thread({ turns, pending, suggestions = [], onSuggest }) {
  const ref = useRef(null);
  useCitationLinks(ref);

  useEffect(() => {
    const node = ref.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [turns, pending]);

  if (!turns.length && !pending) {
    return (
      <div className="thread" ref={ref}>
        <div className="thread-inner">
          <div className="empty">
            <h2>Ask anything about your notebook</h2>
            <p>
              Answers come only from your uploaded sources, with the exact passage
              cited.
            </p>
            {/* Suggested questions, as in the reference product. This replaced a
                paragraph explaining the grounding rules, which was more honest
                and read as a terms-of-service notice on first load. The grounding
                rules are still stated - in the composer hint and in every
                answer's evidence strip - but they are no longer the first thing
                on screen. */}
            {onSuggest && suggestions.length ? (
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
            {/* With no sources the list would be empty, so the empty state says
                the one thing that can actually be done next. */}
            {!suggestions.length ? (
              <p className="empty-cta">
                Add a source to start asking questions.
              </p>
            ) : null}
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