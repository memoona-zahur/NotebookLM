import React, { useEffect, useRef, useState } from "react";
import { api } from "./api.js";

/**
 * The original document, opened at the place a citation came from.
 *
 * Only what the answer rests on is marked. A PDF is shown as an image of its
 * page with a rectangle over each quoted line; any other format is shown as the
 * text the parser produced, with each quoted span marked. Everything arrives as
 * data (an image URL, rectangles, a string and offsets) and is rendered as text
 * nodes and positioned boxes - never as markup - so nothing in an uploaded file
 * can run.
 */
export function DocumentViewer({ sessionId, citation, onClose }) {
  const location = citation?.location;
  const first = useRef(null);

  useEffect(() => {
    const onKey = (event) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!location) return null;

  return (
    <div className="viewer-backdrop" onClick={onClose}>
      <div
        className="viewer"
        role="dialog"
        aria-modal="true"
        aria-label={`${location.source} source`}
        onClick={(event) => event.stopPropagation()}
      >
        <header className="viewer-head">
          <span className="viewer-title">
            {location.source}
            {location.kind === "pdf" ? ` · page ${location.page}` : ""}
          </span>
          <button type="button" className="viewer-close" onClick={onClose} aria-label="Close document">
            ×
          </button>
        </header>
        <div className="viewer-body">
          {location.kind === "pdf" ? (
            <PdfPage sessionId={sessionId} location={location} firstMark={first} />
          ) : (
            <TextWindow sessionId={sessionId} location={location} firstMark={first} />
          )}
        </div>
      </div>
    </div>
  );
}

/** Scroll the first marked thing into the middle of the viewer, once it exists. */
function useScrollToFirst(ref, ready) {
  useEffect(() => {
    if (ready && ref.current && typeof ref.current.scrollIntoView === "function") {
      ref.current.scrollIntoView({ block: "center", behavior: "auto" });
    }
  }, [ref, ready]);
}

function PdfPage({ sessionId, location, firstMark }) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const { page_width: width, page_height: height } = location;
  useScrollToFirst(firstMark, loaded);

  // Percentages of the page, so the boxes follow the image at any size without
  // measuring anything.
  const rects = (location.rects || []).filter(
    (r) => Array.isArray(r) && r.length === 4 && r.every(Number.isFinite) && width > 0 && height > 0,
  );

  return (
    <div className="viewer-page">
      {failed ? <p className="viewer-note">The page could not be loaded.</p> : null}
      <div className="viewer-sheet">
        <img
          src={api.sourcePageUrl(sessionId, location.source_id, location.page)}
          alt={`Page ${location.page} of ${location.source}`}
          onLoad={() => setLoaded(true)}
          onError={() => setFailed(true)}
        />
        {rects.map(([x0, y0, x1, y1], i) => (
          <mark
            key={i}
            ref={i === 0 ? firstMark : undefined}
            className="viewer-rect"
            style={{
              left: `${(x0 / width) * 100}%`,
              top: `${(y0 / height) * 100}%`,
              width: `${((x1 - x0) / width) * 100}%`,
              height: `${((y1 - y0) / height) * 100}%`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

function TextWindow({ sessionId, location, firstMark }) {
  const [state, setState] = useState({ status: "loading" });
  useScrollToFirst(firstMark, state.status === "ready");

  useEffect(() => {
    let live = true;
    api
      .sourceText(sessionId, location.source_id, location.spans)
      .then((data) => live && setState({ status: "ready", data }))
      .catch((err) => live && setState({ status: "error", message: err.message }));
    return () => {
      live = false;
    };
  }, [sessionId, location]);

  if (state.status === "loading") return <p className="viewer-note">Opening the document…</p>;
  if (state.status === "error") return <p className="viewer-note">{state.message}</p>;
  const { data } = state;
  if (data.text == null) {
    return (
      <p className="viewer-note">
        This source was indexed before positions were recorded. Re-index it to open the
        document here.
      </p>
    );
  }

  // Marks are UTF-16 offsets into `data.text`, ordered and non-overlapping; a
  // mark that does not fit is skipped rather than slicing the wrong characters.
  const text = data.text;
  const marks = (data.marks || [])
    .filter((m) => Number.isInteger(m.start) && Number.isInteger(m.end) && m.start < m.end)
    .sort((a, b) => a.start - b.start);
  const parts = [];
  let at = 0;
  marks.forEach((m, i) => {
    if (m.start < at || m.end > text.length) return;
    if (m.start > at) parts.push(text.slice(at, m.start));
    parts.push(
      <mark key={i} ref={parts.some((p) => typeof p !== "string") ? undefined : firstMark} className="viewer-mark">
        {text.slice(m.start, m.end)}
      </mark>,
    );
    at = m.end;
  });
  if (at < text.length) parts.push(text.slice(at));

  return (
    <div className="viewer-text">
      {data.before ? <p className="viewer-note">… earlier text not shown</p> : null}
      <pre>{parts}</pre>
      {data.after ? <p className="viewer-note">… later text not shown</p> : null}
    </div>
  );
}
