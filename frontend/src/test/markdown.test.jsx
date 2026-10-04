import React from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Answer } from "../Answer.jsx";
import { blocks, inline } from "../markdown.jsx";

const html = (node) => {
  const { container } = render(node);
  return container.innerHTML;
};

describe("inline markdown", () => {
  it("renders strong and emphasis", () => {
    render(<Answer text="**bold** and *italic*" cited={[]} />);
    expect(screen.getByText("bold").tagName).toBe("STRONG");
    expect(screen.getByText("italic").tagName).toBe("EM");
  });

  it("does not treat a lone asterisk as emphasis", () => {
    // '5 * 3' is arithmetic, not emphasis, and losing the asterisk would
    // silently change what the document said.
    render(<Answer text="5 * 3 = 15" cited={[]} />);
    expect(screen.getByText("5 * 3 = 15")).toBeInTheDocument();
  });

  it("keeps a marker inside emphasis as a citation", () => {
    render(<Answer text="*fast* [1]" cited={[1]} />);
    expect(screen.getAllByRole("link")).toHaveLength(1);
    expect(screen.getByText("fast").tagName).toBe("EM");
  });

  it("renders inline code without interpreting it", () => {
    const markup = html(<Answer text="run `pytest -k x` now" cited={[]} />);
    expect(markup).toContain("<code>pytest -k x</code>");
  });

  it("renders a citation-shaped code span as code, not a link", () => {
    const markup = html(<Answer text="the array is `[1]`" cited={[]} />);
    expect(markup).toContain("<code>[1]</code>");
    expect(markup).not.toContain('class="ref"');
  });
});

describe("links", () => {
  it("follows an http link", () => {
    render(<Answer text="see [the docs](https://example.com/a)" cited={[]} />);
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("href", "https://example.com/a");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("refuses a javascript: url but keeps the label", () => {
    const { container } = render(<Answer text="[click](javascript:alert(1))" cited={[]} />);
    // No anchor, so there is nothing to click and nothing for the URL to do.
    expect(container.querySelector("a")).toBeNull();
    expect(container.innerHTML).not.toContain("javascript:");
    // The label survives as text. The stray ')' does too: the url pattern stops
    // at the first ')', so this is not a URL we ever parsed, just text.
    expect(container.textContent).toContain("click");
  });

  it("refuses a data: url", () => {
    const markup = html(<Answer text="[x](data:text/html,<script>)" cited={[]} />);
    expect(markup).not.toContain("<a ");
  });
});

describe("html in the text is inert", () => {
  it("never becomes an element", () => {
    const { container } = render(
      <Answer text='<script>alert(1)</script> <img src=x onerror="alert(2)"> ok' cited={[]} />,
    );
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    // The text is still shown, just as characters.
    expect(container.textContent).toContain("alert(1)");
  });

  it("renders a bare tag as visible text", () => {
    render(<Answer text="<b>not bold</b>" cited={[]} />);
    expect(document.querySelector(".answer b")).toBeNull();
    expect(screen.getByText(/not bold/)).toBeInTheDocument();
  });
});

describe("block markdown", () => {
  it("renders bullets as a list", () => {
    render(<Answer text={"- first\n- second"} cited={[]} />);
    expect(screen.getAllByRole("list")).toHaveLength(1);
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });

  it("renders both dash and star bullets", () => {
    render(<Answer text={"- one\n* two\n+ three"} cited={[]} />);
    expect(screen.getAllByRole("listitem")).toHaveLength(3);
  });

  it("renders an ordered list", () => {
    render(<Answer text={"1. one\n2. two"} cited={[]} />);
    expect(screen.getByRole("list")).toHaveProperty("tagName", "OL");
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });

  it("keeps citations inside list items", () => {
    render(<Answer text={"- alpha [1]\n- beta [2]"} cited={[1, 2]} />);
    expect(screen.getAllByRole("link")).toHaveLength(2);
  });

  it("shifts heading levels down so an answer cannot outrank the question", () => {
    render(<Answer text={"# Big\ntext"} cited={[]} />);
    const heading = screen.getByRole("heading");
    expect(heading.tagName).toBe("H3");
  });

  it("renders a fenced code block literally", () => {
    const markup = html(<Answer text={"```py\nx = 1  # **not bold**\n```"} cited={[]} />);
    expect(markup).toContain("<pre");
    expect(markup).toContain("# **not bold**");
    expect(markup).not.toContain("<strong>");
  });

  it("does not linkify citations inside a code block", () => {
    const markup = html(<Answer text={"```\nvalue = [1]\n```"} cited={[1]} />);
    expect(markup).not.toContain('class="ref"');
  });

  it("leaves an unterminated fence open rather than losing the rest", () => {
    const markup = html(<Answer text={"```\nx = 1"} cited={[]} />);
    expect(markup).toContain("x = 1");
  });

  it("renders a blockquote", () => {
    render(<Answer text={"> quoted line"} cited={[]} />);
    expect(screen.getByText("quoted line").tagName).toBe("BLOCKQUOTE");
  });

  it("joins a wrapped paragraph instead of breaking mid-sentence", () => {
    render(<Answer text={"the block narrows to\nnineteen metres behind"} cited={[]} />);
    expect(screen.getByText("the block narrows to nineteen metres behind")).toBeInTheDocument();
  });

  it("keeps blank lines between paragraphs", () => {
    const markup = html(<Answer text={"first para\n\nsecond para"} cited={[]} />);
    expect(markup).toContain("first para");
    expect(markup).toContain("second para");
  });

  it("treats a dash inside a sentence as prose, not a bullet", () => {
    render(<Answer text="the delta-v - 15 - is fixed" cited={[]} />);
    expect(screen.queryAllByRole("list")).toHaveLength(0);
  });

  it("stops a paragraph at a heading", () => {
    render(<Answer text={"intro line\n## Later"} cited={[]} />);
    expect(screen.getByRole("heading")).toHaveTextContent("Later");
  });

  it("leaves an unrecognised construct as the text the model wrote", () => {
    // Unsupported Markdown must degrade to literal text, not vanish.
    const markup = html(<Answer text={"| a | b |\n| - | - |"} cited={[]} />);
    expect(markup).toContain("| a | b |");
  });

  it("does not crash on an unclosed emphasis marker", () => {
    expect(() => render(<Answer text="**never closed" cited={[]} />)).not.toThrow();
  });

  it("does not crash on only whitespace", () => {
    expect(() => render(<Answer text="   \n\n  " cited={[]} />)).not.toThrow();
  });
});

describe("markdown and citations together", () => {
  it("links a marker inside a heading", () => {
    render(<Answer text={"## Findings [1]"} cited={[1]} />);
    expect(screen.getAllByRole("link")).toHaveLength(1);
  });

  it("still drops an out-of-range marker in a list", () => {
    render(<Answer text={"- a [1]\n- b [9]"} cited={[1]} />);
    const refs = screen.getAllByRole("link");
    expect(refs).toHaveLength(1);
    expect(refs[0]).toHaveAttribute("href", "#cite-1");
  });

  it("keeps an unlinked marker literal when there are no cards", () => {
    render(<Answer text="- a [1]" cited={[1]} linkable={false} />);
    expect(screen.queryAllByRole("link")).toHaveLength(0);
    expect(screen.getByText(/\[1\]/)).toBeInTheDocument();
  });
});

describe("exports", () => {
  it("blocks() and inline() are usable directly", () => {
    expect(inline("plain")).toEqual(["plain"]);
    expect(blocks("plain")).toHaveLength(1);
  });
});