import React, { useEffect, useRef, useState } from "react";

/**
 * Renders passage text with the located occurrences of a word marked.
 *
 * `matches` are `[start, end]` character offsets into `text`, computed by the
 * server. Marking them rather than re-searching here means the highlight can
 * never disagree with what the server said it found: if the spans were wrong the
 * text simply comes out unmarked rather than marking the wrong word.
 */
export function Highlighted({ text, matches = [], className }) {
  // Spans arrive ordered and non-overlapping, but a hand-edited or future
  // response might not be, and rendering them out of order would slice the text
  // backwards and produce mojibake.
  const spans = [...matches]
    .filter(([start, end]) => Number.isInteger(start) && Number.isInteger(end) && start < end)
    .sort((a, b) => a[0] - b[0])
    .filter(([start, end], i, all) => i === 0 || start >= all[i - 1][1]);

  if (!spans.length) return <>{text}</>;

  const parts = [];
  let at = 0;
  spans.forEach(([start, end], i) => {
    // A span outside the text means the two sides disagree about the string.
    // Rendering nothing beats throwing away the passage.
    if (start < at || end > text.length) return;
    if (start > at) parts.push(text.slice(at, start));
    parts.push(
      <mark className="hit" key={`${start}-${i}`}>
        {text.slice(start, end)}
      </mark>,
    );
    at = end;
  });
  if (at < text.length) parts.push(text.slice(at));

  return <span className={className}>{parts}</span>;
}

/**
 * Name a word, see everywhere this session's documents use it.
 *
 * Deliberately a box in the sources drawer rather than a chat instruction: a
 * conversation cannot be relied on to parse "highlight the word X" apart from a
 * real question about highlights, and a highlight that depends on the model
 * misfiring is worse than one that does not.
 */
export function Occurrences({ onFind, onClose }) {
  const [term, setTerm] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const input = useRef(null);
  const request = useRef(0);

  useEffect(() => {
    input.current?.focus();
  }, []);

  async function find(event) {
    event.preventDefault();
    const wanted = term.trim();
    if (!wanted) return;
    // Responses can arrive out of order when a slow request follows a fast one;
    // only the newest is allowed to write to the screen.
    const ticket = ++request.current;
    setBusy(true);
    setError(null);
    try {
      const body = await onFind(wanted);
      if (ticket !== request.current) return;
      setResult(body);
    } catch (err) {
      if (ticket !== request.current) return;
      setResult(null);
      setError(err.message || "Could not search the sources.");
    } finally {
      if (ticket === request.current) setBusy(false);
    }
  }

  function clear() {
    request.current += 1;
    setTerm("");
    setResult(null);
    setError(null);
    setBusy(false);
  }

  const shown = result ? result.count : 0;

  return (
    <div className="occurrences">
      <form className="occ-form" onSubmit={find}>
        <input
          ref={input}
          className="occ-input"
          type="search"
          value={term}
          placeholder="Highlight a word"
          aria-label="Word to highlight"
          onChange={(e) => setTerm(e.target.value)}
        />
        {/* Deliberately not disabled while a search is in flight. The guard
            below exists to discard a slow answer that a newer search has
            already replaced, which is only reachable if the user can start a
            second search before the first returns. */}
        <button className="occ-go" type="submit" disabled={!term.trim()}>
          {busy ? "…" : "Find"}
        </button>
      </form>

      {error ? <p className="occ-error">{error}</p> : null}

      {result ? (
        <>
          <p className="occ-summary">
            {shown === 0 ? (
              <>
                “{result.term}” is not in any of these documents.
              </>
            ) : (
              <>
                <b>{shown}</b> {shown === 1 ? "place" : "places"} for{" "}
                <b>{result.term}</b>
                {result.truncated ? " — showing the first ones" : ""}
              </>
            )}
          </p>

          {result.occurrences.map((occurrence, i) => (
            <details
              className="occ-hit"
              key={`${occurrence.source}-${occurrence.position}`}
              // The first hit starts open: asking for a word and getting back a
              // list of closed rows means the highlight is still not visible
              // until you click. The rest stay closed so a common word does not
              // bury the drawer in passages.
              open={i === 0 ? true : undefined}
            >
              <summary>
                <span className="occ-src">{occurrence.source}</span>
                {occurrence.page ? <span className="page"> · p.{occurrence.page}</span> : null}
                <span className="occ-count">
                  {occurrence.count} {occurrence.count === 1 ? "time" : "times"}
                </span>
              </summary>
              <pre className="occ-text">
                <Highlighted text={occurrence.text} matches={occurrence.matches} />
              </pre>
            </details>
          ))}
        </>
      ) : null}

      <button className="occ-close" type="button" onClick={onClose}>
        Done
      </button>
    </div>
  );
}