import React from "react";

/**
 * A small Markdown renderer for answer text.
 *
 * Written by hand rather than pulled in as a dependency because the input is
 * model output that has to be rendered *safely*: every source document is
 * untrusted, and a document can end up quoted inside an answer. Nothing here
 * goes through dangerouslySetInnerHTML, so no HTML in the text can ever become
 * an element. A `marked`/`react-markdown` dependency would bring the same
 * guarantee, but this app has two runtime dependencies and the subset below is
 * all a research answer actually uses.
 *
 * Deliberately not supported: raw HTML, images, tables, nested lists. A model
 * asked to answer from sources rarely emits them, and every one of them is
 * either an injection surface or a layout liability. Unsupported syntax renders
 * as the literal text the model wrote, which is the honest outcome.
 *
 * Paragraphs emit bare text nodes rather than <p> elements. The container is
 * already `white-space: pre-wrap`, so newlines already render, and leaving
 * single-paragraph answers as plain nodes keeps the existing markup - and every
 * existing test - unchanged.
 */

const SAFE_URL = /^(?:https?:)?\/\//i;

// Inline, in one pass. Order matters: code spans win over everything, and a
// bracketed number followed by parentheses is a Markdown link rather than a
// citation marker.
const INLINE = new RegExp(
  [
    // `code`
    "`(?<code>[^`\\n]+)`",
    // [1](https://...) - a link whose label is a citation number
    "\\[(?<lnum>\\d{1,3})\\]\\((?<lurl>[^)\\s]+)\\)",
    // [label](url)
    "\\[(?<label>[^\\]\\n]+)\\]\\((?<url>[^)\\s]+)\\)",
    // [1]
    "\\[(?<num>\\d{1,3})\\]",
    // **strong** then *em*
    "\\*\\*(?<strong>[^*\\n]+?)\\*\\*",
    "\\*(?<em>[^*\\n]+?)\\*",
  ].join("|"),
  "g",
);

const HEADING = /^(#{1,6})\s+(.*)$/;
const BULLET = /^[-*+]\s+(.*)$/;
const ORDERED = /^(\d+)[.)]\s+(.*)$/;
const QUOTE = /^>\s?(.*)$/;
const FENCE = /^```(\w*)\s*$/;

/** A link, or just its label when the URL is not one we are willing to follow. */
function Anchor({ url, children }) {
  if (!SAFE_URL.test(url)) return <>{children}</>;
  return (
    <a href={url} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  );
}

/** One citation marker, kept as the same `.ref` chip the answer has always used. */
function Ref({ index, cited, linkable }) {
  // An out-of-range marker means the model cited a passage it was never given.
  // The server strips those, but rendering a dead link would be worse than
  // showing nothing, so an unexpected one is dropped rather than linked.
  if (index < 0 || (cited && cited.length && index >= cited.length)) return null;
  if (!linkable) return <>[{index + 1}]</>;
  return (
    <a
      className="ref"
      href={`#cite-${index + 1}`}
      data-cite={index + 1}
      title={`Jump to source ${index + 1}`}
    >
      [{index + 1}]
    </a>
  );
}

/** Inline Markdown within one block of text. */
export function inline(text, options = {}) {
  const nodes = [];
  let last = 0;
  let key = 0;
  let match;

  INLINE.lastIndex = 0;
  while ((match = INLINE.exec(text)) !== null) {
    const groups = match.groups || {};
    let node = null;

    if (groups.code !== undefined) {
      node = <code>{groups.code}</code>;
    } else if (groups.lnum !== undefined) {
      // A number that looks like a citation but is wrapped in a link URL is
      // just a link whose text happens to be a number.
      node = <Anchor url={groups.lurl}>{`[${groups.lnum}]`}</Anchor>;
    } else if (groups.label !== undefined) {
      node = <Anchor url={groups.url}>{groups.label}</Anchor>;
    } else if (groups.num !== undefined) {
      node = (
        <Ref
          key={`ref-${key}`}
          index={parseInt(groups.num, 10) - 1}
          cited={options.cited}
          linkable={options.linkable}
        />
      );
    } else if (groups.strong !== undefined) {
      node = <strong>{groups.strong}</strong>;
    } else if (groups.em !== undefined) {
      node = <em>{groups.em}</em>;
    }

    // A run that the renderer dropped entirely - an out-of-range citation -
    // must not reappear as literal text, so nothing is pushed for it.
    if (node === null) {
      last = match.index + match[0].length;
      continue;
    }

    if (match.index > last) nodes.push(text.slice(last, match.index));
    if (React.isValidElement(node)) node = React.cloneElement(node, { key: `i-${key}` });
    nodes.push(node);
    last = match.index + match[0].length;
    key += 1;
  }
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

/** Block-level Markdown. Returns an array of React nodes. */
export function blocks(text, options = {}) {
  const lines = String(text).split("\n");
  const out = [];
  let index = 0;
  let key = 0;

  const flushList = (ordered, items) => {
    if (!items.length) return;
    const children = items.map((item, i) => <li key={`li-${key}-${i}`}>{inline(item, options)}</li>);
    key += 1;
    out.push(ordered ? <ol key={`l-${key}`}>{children}</ol> : <ul key={`l-${key}`}>{children}</ul>);
  };

  while (index < lines.length) {
    const line = lines[index];

    // Fenced code. Everything up to the closing fence is literal: no inline
    // Markdown inside a code block, which is the entire point of a code block.
    const fence = FENCE.exec(line);
    if (fence) {
      const body = [];
      index += 1;
      while (index < lines.length && !FENCE.test(lines[index])) {
        body.push(lines[index]);
        index += 1;
      }
      index += 1; // the closing fence
      out.push(
        <pre className="code" key={`p-${key}`}>
          <code>{body.join("\n")}</code>
        </pre>,
      );
      key += 1;
      continue;
    }

    if (!line.trim()) {
      index += 1;
      continue;
    }

    const heading = HEADING.exec(line);
    if (heading) {
      // h1/h2 are the page's own headings; an answer that opens with one should
      // not outrank the question above it, so levels are shifted down.
      const level = Math.min(6, heading[1].length + 2);
      const Tag = `h${level}`;
      out.push(
        <Tag className="md-h" key={`p-${key}`}>
          {inline(heading[2], options)}
        </Tag>,
      );
      key += 1;
      index += 1;
      continue;
    }

    if (QUOTE.test(line)) {
      const body = [];
      while (index < lines.length && QUOTE.test(lines[index])) {
        body.push(QUOTE.exec(lines[index])[1]);
        index += 1;
      }
      out.push(
        <blockquote className="md-quote" key={`p-${key}`}>
          {inline(body.join(" "), options)}
        </blockquote>,
      );
      key += 1;
      continue;
    }

    const bullet = BULLET.exec(line);
    const ordered = ORDERED.exec(line);
    if (bullet || ordered) {
      const isOrdered = Boolean(ordered);
      const items = [];
      while (index < lines.length) {
        const b = BULLET.exec(lines[index]);
        const o = ORDERED.exec(lines[index]);
        if (isOrdered ? !o : !b) break;
        items.push((isOrdered ? o[2] : b[1]).trim());
        index += 1;
      }
      flushList(isOrdered, items);
      continue;
    }

    // A paragraph runs until a blank line or a line that starts a block. Joined
    // with a space: the container's pre-wrap would otherwise render the source
    // line break as a hard break mid-sentence.
    const para = [];
    while (index < lines.length) {
      const current = lines[index];
      if (
        !current.trim() ||
        FENCE.test(current) ||
        HEADING.test(current) ||
        BULLET.test(current) ||
        ORDERED.test(current) ||
        QUOTE.test(current)
      ) {
        break;
      }
      para.push(current.trim());
      index += 1;
    }
    out.push(<React.Fragment key={`p-${key}`}>{inline(para.join(" "), options)}</React.Fragment>);
    key += 1;
  }

  return out;
}

/** Answer text as Markdown, with `[n]` markers still rendered as citation chips. */
export function Markdown({ text, cited, linkable = true }) {
  return <>{blocks(text, { cited, linkable })}</>;
}