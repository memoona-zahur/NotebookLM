import React, { useEffect } from "react";
import { Markdown } from "./markdown.jsx";

/**
 * Renders answer text, turning `[n]` markers into clickable citation chips.
 *
 * `linkable` says whether any citation cards are actually rendered below. A
 * replayed turn has markers in its text but no cards — linking them anyway
 * would produce anchors pointing at ids that are not on the page.
 */
export function Answer({ text, cited, linkable = true }) {
  // A turn can arrive with no body at all (an interrupted request persisted a
  // role and nothing else). Rendering `undefined` into the markup would throw
  // and take the whole thread down with it.
  text = typeof text === "string" ? text : "";
  return (
    <div className="answer">
      <Markdown text={text} cited={cited} linkable={linkable} />
    </div>
  );
}

/**
 * Wire up citation chips to their cards: clicking scrolls the passage into
 * view, opens it, and flashes it. The active marker is tracked here rather than
 * per-chip so only one is ever highlighted.
 */
export function useCitationLinks(threadRef) {
  useEffect(() => {
    const root = threadRef.current;
    if (!root) return undefined;

    const onClick = (event) => {
      const link = event.target.closest("a.ref");
      if (!link) return;
      event.preventDefault();

      const target = root.querySelector(`#${CSS.escape(link.getAttribute("href").slice(1))}`);
      if (!target) return;

      root.querySelectorAll("a.ref.active").forEach((el) => el.classList.remove("active"));
      link.classList.add("active");

      target.classList.add("open", "flash");
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      setTimeout(() => target.classList.remove("flash"), 1200);
    };

    root.addEventListener("click", onClick);
    return () => root.removeEventListener("click", onClick);
  }, [threadRef]);
}

/** A retrieved passage, collapsed to a header and expandable to the full text. */
export function Citation({ citation, index, cited }) {
  const wasCited = !cited || !cited.length || cited.includes(index + 1);
  return (
    <div
      className={`cite${wasCited ? "" : " uncited"}`}
      id={`cite-${index + 1}`}
    >
      <button
        type="button"
        className="cite-head"
        onClick={(e) => e.currentTarget.parentElement.classList.toggle("open")}
        aria-expanded="false"
      >
        <span className="cite-caret" aria-hidden="true">
          ▸
        </span>
        <span className="cite-src" title={citation.source}>
          {citation.source}
          {citation.page ? <span className="page"> · p.{citation.page}</span> : null}
        </span>
        <span className="cite-score">{Math.round((citation.score || 0) * 100)}%</span>
        {wasCited ? <span className="cite-cited" title="Cited by the answer" /> : null}
      </button>
      <pre className="snippet">{citation.text}</pre>
    </div>
  );
}

/** The honesty strip: how confident retrieval was, and what was pruned. */
export function Evidence({ evidence }) {
  if (!evidence) return null;

  if (evidence.verdict === "no_match") {
    return (
      <div className="evidence">
        <span className="nomatch">
          No relevant passage found — best match{" "}
          {Math.round(evidence.best_score * 100)}%, below the{" "}
          {Math.round(evidence.min_score * 100)}% floor. The model was not asked, so
          nothing was invented.
        </span>
      </div>
    );
  }

  return (
    <div className="evidence">
      <span className={`chip ${evidence.confidence}`}>{evidence.confidence} confidence</span>
      <span>
        best match {Math.round(evidence.best_score * 100)}% · {evidence.returned} of{" "}
        {evidence.considered} candidates used
      </span>
      {evidence.invalid && evidence.invalid.length ? (
        <span className="chip warn">
          removed citation {evidence.invalid.map((n) => `[${n}]`).join(", ")}
        </span>
      ) : null}
      {evidence.numeric_damped && evidence.numeric_share < 0.5 ? (
        <span className="chip warn">numeric passages damped</span>
      ) : null}
    </div>
  );
}