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
 * The server stores `content`; an answer made in this browser session has `text`
 * plus its `citations` and `evidence`. Reading a session the user has left and
 * reloaded therefore has to work with `content` alone — no citations to show,
 * which is why the passage list is optional rather than assumed.
 *
 * `content` is deliberately not renamed to `text` at the fetch boundary: the
 * field is persisted, so the name the API speaks is the one that wins, and
 * renaming it in the client is how the two drifted apart in the first place.
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