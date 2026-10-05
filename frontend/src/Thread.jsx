import React, { useEffect, useRef } from "react";
import { Answer, Citation, Evidence, useCitationLinks } from "./Answer.jsx";
import { EXPLORE } from "./suggestions.js";

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
  onExplore,
  onUpload,
  onWebSearch,
  webSearchReason = "Not available in this local build.",
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
    // The explore flag is a control, not a question, so it is separated out
    // before rendering rather than filtered while rendering.
    const questions = suggestions.filter((q) => q !== EXPLORE);
    return (
      <div className="thread" ref={ref}>
        <div className="thread-inner">
          <div className="empty">
            {/* The heading is deliberately a statement of what this app does,
                not an instruction. "Add a source to get started" reads as a
                scolding when you have already added three, and "Hello." is
                carried by the assistant's own reply the moment the user types
                anything - a greeting is now generated, not hardcoded, so a
                static hello here would pre-empt it. */}
            <h2>
              {hasSources
                ? `What would you like to know about ${notebookName}?`
                : "Ask anything. I'll answer from your sources."}
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
              onWebSearch ? (
                <div className="onboard">
                  <button type="button" className="onboard-card" onClick={onUpload}>
                    <span className="onboard-title">
                      I want to upload my own documents
                    </span>
                    <span className="onboard-sub">
                      PDF, DOCX, Markdown, CSV and more. Answers come only from what
                      you upload.
                    </span>
                  </button>
                  {/* Live when a provider key is configured. The old card said
                      "not available in this local build" unconditionally, which
                      was false: the app could have searched, it had just never
                      been told how. The sub-line now states what actually
                      leaves the machine, because a web search really does send
                      the query to a third-party search engine. */}
                  <button type="button" className="onboard-card" onClick={onWebSearch}>
                    <span className="onboard-title">Search the web for sources</span>
                    <span className="onboard-sub">
                      Finds pages on a topic and adds them to this notebook. Your
                      search term is sent to the search provider.
                    </span>
                  </button>
                </div>
              ) : (
                <div className="onboard">
                  <button type="button" className="onboard-card" onClick={onUpload}>
                    <span className="onboard-title">
                      I want to upload my own documents
                    </span>
                    <span className="onboard-sub">
                      PDF, DOCX, Markdown, CSV and more. Answers come only from what
                      you upload.
                    </span>
                  </button>
                  <div className="onboard-card disabled" aria-disabled="true">
                    <span className="onboard-title">Search the web for sources</span>
                    <span className="onboard-sub">{webSearchReason}</span>
                  </div>
                </div>
              )
            ) : onSuggest && questions.length ? (
              <>
                <ul className="suggestions">
                  {questions.map((question) => (
                    <li key={question}>
                      <button type="button" onClick={() => onSuggest(question)}>
                        {question}
                      </button>
                    </li>
                  ))}
                </ul>
                {/* The one control that is not a question. It used to be a pill
                    reading "What questions should I be asking about this?", and
                    clicking it sent that text to the model - which found no
                    passage about how to use the app and refused, so the app
                    answered a question it had written itself with "I could not
                    find anything relevant". It now does what it says: reveals
                    the questions above, in place, without asking anything. */}
                {onExplore ? (
                  <button type="button" className="explore" onClick={onExplore}>
                    Show me other questions
                  </button>
                ) : null}
              </>
            ) : null}

            {/* The grounding promise is qualified where it has to be. Stating it
                unconditionally would be false now that web search exists: the
                answer is still drawn from sources and still cited, but one of
                those sources may have come from the open web. */}
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