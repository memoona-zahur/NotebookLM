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

function Turn({ turn }) {
  return (
    <article className={`turn ${turn.role}`}>
      <div className="who">
        <span className="dot" aria-hidden="true" />
        {turn.role === "user" ? "You" : "NotebookLM"}
      </div>
      {turn.error ? (
        <div className="answer error">{turn.text}</div>
      ) : (
        <>
          <Answer text={turn.text} cited={turn.evidence?.cited} />
          <Evidence evidence={turn.evidence} />
          {turn.citations && turn.citations.length ? (
            <div className="cites">
              {turn.citations.map((citation, index) => (
                <Citation
                  key={`${citation.source}-${citation.page}-${index}`}
                  citation={citation}
                  index={index}
                  cited={turn.evidence?.cited}
                />
              ))}
            </div>
          ) : null}
        </>
      )}
    </article>
  );
}

export function Thread({ turns, pending }) {
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
            <h2>Ask your sources anything</h2>
            <p>
              Answers come only from the documents in this session, with the exact passage
              cited. Nothing retrieved means no answer — the model is not asked to guess.
            </p>
            <ul>
              <li>
                <b>Upload</b> a PDF, DOCX, Markdown, CSV or source file to get started
              </li>
              <li>
                <b>Follow up</b> in the same session and it will remember the thread
              </li>
              <li>
                <b>Open a new session</b> to keep separate sets of documents apart
              </li>
            </ul>
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
          <article className="turn bot">
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