import React, { useEffect, useState } from "react";
import { Markdown } from "./markdown.jsx";
import { highlightsOf } from "./offsets.js";

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

      // Looked up inside the chip's own turn: every turn numbers its cards from
      // 1, so a search from the thread root would always find the first turn's.
      const scope = link.closest("article") || root;
      const target = scope.querySelector(`#${CSS.escape(link.getAttribute("href").slice(1))}`);
      if (!target) return;

      root.querySelectorAll("a.ref.active").forEach((el) => el.classList.remove("active"));
      link.classList.add("active");

      target.classList.add("open", "flash");
      const exact = target.querySelector(".source-highlight");
      if (exact) {
        exact.classList.add("flash");
      }
      target.scrollIntoView({ behavior: "smooth", block: "center" });
      setTimeout(() => target.classList.remove("flash"), 1200);
      setTimeout(() => {
        if (exact) exact.classList.remove("flash");
      }, 1200);
    };

    root.addEventListener("click", onClick);
    return () => root.removeEventListener("click", onClick);
  }, [threadRef]);
}

/**
 * The passage text with only the answering spans marked. `spans` are UTF-16
 * ranges inside `text` (see offsets.js), already clipped and ordered.
 */
function SourceHighlight({ text, spans }) {
  if (!spans.length) return <>{text}</>;
  const parts = [];
  let at = 0;
  spans.forEach((span, i) => {
    if (span.start > at) parts.push(text.slice(at, span.start));
    parts.push(
      <mark key={i} className="source-highlight">
        {text.slice(span.start, span.end)}
      </mark>,
    );
    at = span.end;
  });
  if (at < text.length) parts.push(text.slice(at));
  return <>{parts}</>;
}

/** A retrieved passage, collapsed to a header and expandable to the full text. */
export function Citation({ citation, index, cited, onOpen, onReindex }) {
  // Re-indexing fixes the source, not this answer: the place a quote sits is
  // worked out when the question is answered, so an answer given before the
  // re-index stays without one. Say so, instead of leaving the same button up.
  const [reindex, setReindex] = useState("idle");
  const wasCited = !cited || !cited.length || cited.includes(index + 1);
  // Nothing to open: the source was indexed before positions were recorded (a
  // PDF is the exception - its page is enough to find the quote again).
  const needsReindex =
    wasCited &&
    !citation.location &&
    !citation.page &&
    citation.char_start == null &&
    Boolean(citation.source_id);
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
        <span className="cite-src" title={citation.heading || citation.source}>
          {citation.source}
          {citation.page ? <span className="page"> · p.{citation.page}</span> : null}
        </span>
        {citation.heading ? (
          <span className="cite-heading">{citation.heading}</span>
        ) : null}
        <span className="cite-score">{Math.round((citation.score || 0) * 100)}%</span>
        {wasCited ? <span className="cite-cited" title="Cited by the answer" /> : null}
      </button>
      <pre className="snippet">
        <SourceHighlight
          text={citation.text}
          spans={wasCited ? highlightsOf(citation) : []}
        />
      </pre>
      {wasCited && citation.location && onOpen ? (
        <button type="button" className="cite-open" onClick={() => onOpen(citation)}>
          Open in document
        </button>
      ) : null}
      {needsReindex && onReindex && reindex !== "done" ? (
        <button
          type="button"
          className="cite-open"
          disabled={reindex === "busy"}
          onClick={async () => {
            setReindex("busy");
            const ok = await onReindex(citation);
            setReindex(ok === false ? "idle" : "done");
          }}
        >
          {reindex === "busy" ? "Re-indexing…" : "Re-index to open in document"}
        </button>
      ) : null}
      {wasCited && !citation.location && !needsReindex && onOpen ? (
        // Say why there is nothing to open, instead of a card that looks
        // unfinished: either the quote was found but the original file is gone,
        // or no sentence matched confidently enough to point at.
        <p className="cite-note">
          {citation.quotes?.length
            ? "The original file is no longer available to open. Upload it again to see this passage in the document."
            : "No exact sentence could be matched with confidence, so none is highlighted."}
        </p>
      ) : null}
      {needsReindex && reindex === "done" ? (
        <p className="cite-note">
          Re-indexed. Ask the question again and this passage will open in the document.
        </p>
      ) : null}
    </div>
  );
}

/** The honesty strip: how confident retrieval was, and what was pruned. */
export function Evidence({ evidence }) {
  if (!evidence) return null;

  if (evidence.verdict === "conversational") {
    // No retrieval ran, so there is no score to report and no floor to have
    // missed. The one fact worth stating is that no document was consulted and
    // nothing was spent, because that is the guarantee the rest of this strip
    // exists to make.
    return (
      <div className="evidence">
        <span className="nomatch">
          No documents consulted for this one — nothing was invented.
        </span>
        <Cost cost={evidence.cost} />
      </div>
    );
  }

  if (evidence.verdict === "no_match") {
    return (
      <div className="evidence">
        <span className="nomatch">
          No relevant passage found — best match{" "}
          {Math.round(evidence.best_score * 100)}%, below the{" "}
          {Math.round(evidence.min_score * 100)}% floor. The model was not asked, so
          nothing was invented.
        </span>
        {/* A refusal is the cheapest answer, and saying so is the point. The
            sentence above already explains the model was skipped; this records
            that the skip cost nothing. */}
        <Cost cost={evidence.cost} />
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
      <Cost cost={evidence.cost} />
    </div>
  );
}

/**
 * What the question cost, in the same strip as how confident retrieval was.
 *
 * Retrieval confidence answers "should I believe this"; cost answers "was this
 * worth asking". Both are shown because an answer can be well-cited and still
 * have dragged half the library into the prompt, and that is only visible if
 * the tokens are on screen.
 *
 * Latency is split rather than totalled, because retrieval and generation fail
 * for unrelated reasons and one number hides which one was slow.
 */
function Cost({ cost }) {
  if (!cost) return null;

  if (!cost.called) {
    return <span className="chip ok">no tokens spent — refused before the model</span>;
  }

  // Read defensively: a turn recorded by an older build - or restored from a
  // backup taken before a field existed - can be missing one of these, and a
  // missing number must not take the whole thread down with it.
  const total = Number(cost.total_tokens ?? 0);
  const ms = Math.round(Number(cost.generation_ms ?? 0));
  const seconds = ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;

  return (
    <span className="chip" title={cost.estimated ? "Token count estimated from text length" : "Token count reported by the provider"}>
      {total.toLocaleString()} tokens{cost.estimated ? " (est.)" : ""} ·{" "}
      {seconds} to answer · {Math.round(Number(cost.retrieval_ms ?? 0))}ms to search
    </span>
  );
}