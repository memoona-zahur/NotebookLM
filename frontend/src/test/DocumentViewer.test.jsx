import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { api } from "../api.js";
import { Citation } from "../Answer.jsx";
import { DocumentViewer } from "../DocumentViewer.jsx";
import { Thread } from "../Thread.jsx";
import { cpToUtf16, highlightsOf } from "../offsets.js";

afterEach(() => vi.restoreAllMocks());

const PDF_LOCATION = {
  kind: "pdf",
  source_id: "s1",
  source: "handbook.pdf",
  page: 2,
  page_width: 600,
  page_height: 800,
  rects: [
    [60, 80, 300, 96],
    [60, 96, 180, 112],
  ],
};

describe("offsets", () => {
  it("counts an emoji as two UTF-16 units", () => {
    expect(cpToUtf16("😀 find cat here", 7)).toBe(8);
    expect(cpToUtf16("😀 find cat here", 10)).toBe(11);
    expect(cpToUtf16("abc", 99)).toBe(3);
  });

  it("converts a legacy highlight (code points, no unit) so it slices the right text", () => {
    const text = "😀 find cat here";
    const [span] = highlightsOf({ text, highlight: { start: 7, end: 10 } });
    expect(text.slice(span.start, span.end)).toBe("cat");
  });

  it("takes a UTF-16 highlight as it is", () => {
    const text = "😀 find cat here";
    const [span] = highlightsOf({
      text,
      highlights: [{ start: 8, end: 11 }],
      highlight_unit: "utf16",
    });
    expect(text.slice(span.start, span.end)).toBe("cat");
  });

  it("drops spans outside the text, empty spans and overlaps", () => {
    const spans = highlightsOf({
      text: "abcdefghij",
      highlights: [
        { start: 2, end: 5 },
        { start: 4, end: 8 },
        { start: 20, end: 30 },
        { start: 6, end: 6 },
        { start: "x", end: 3 },
      ],
      highlight_unit: "utf16",
    });
    expect(spans).toEqual([{ start: 2, end: 5 }]);
  });
});

describe("Citation card", () => {
  const base = { source: "notes.md", source_id: "s2", score: 0.5, text: "Alpha one. Beta two. Gamma three." };

  it("marks every quoted span and nothing between them", () => {
    render(
      <Citation
        citation={{
          ...base,
          highlights: [
            { start: 0, end: 10 },
            { start: 21, end: 33 },
          ],
          highlight_unit: "utf16",
        }}
        index={0}
        cited={[1]}
      />,
    );
    const marks = [...document.querySelectorAll(".source-highlight")].map((m) => m.textContent);
    expect(marks).toEqual(["Alpha one.", "Gamma three."]);
  });

  it("offers to open the document only when a location exists", async () => {
    const onOpen = vi.fn();
    const citation = { ...base, location: { kind: "text", source_id: "s2", source: "notes.md", spans: [{ start: 0, end: 5 }] } };
    const { rerender } = render(<Citation citation={citation} index={0} cited={[1]} onOpen={onOpen} />);
    await userEvent.click(screen.getByRole("button", { name: /open in document/i }));
    expect(onOpen).toHaveBeenCalledWith(citation);

    rerender(<Citation citation={{ ...base, char_start: 0 }} index={0} cited={[1]} onOpen={onOpen} />);
    expect(screen.queryByRole("button", { name: /open in document/i })).toBeNull();
  });

  it("asks for a re-index for a source indexed before positions were kept", async () => {
    const onReindex = vi.fn();
    const old = { ...base, char_start: null };
    render(<Citation citation={old} index={0} cited={[1]} onReindex={onReindex} />);
    await userEvent.click(screen.getByRole("button", { name: /re-index to open in document/i }));
    expect(onReindex).toHaveBeenCalledWith(old);
  });

  it("does not ask for a re-index when the passage was not cited, or for a PDF page", () => {
    const { rerender } = render(<Citation citation={{ ...base, char_start: null }} index={1} cited={[1]} onReindex={() => {}} />);
    expect(screen.queryByRole("button", { name: /re-index/i })).toBeNull();
    rerender(<Citation citation={{ ...base, page: 3, char_start: null }} index={0} cited={[1]} onReindex={() => {}} />);
    expect(screen.queryByRole("button", { name: /re-index/i })).toBeNull();
  });
});

describe("DocumentViewer", () => {
  it("shows the PDF page with one box per quoted line, positioned as a share of the page", () => {
    render(<DocumentViewer sessionId="sess" citation={{ location: PDF_LOCATION }} onClose={() => {}} />);
    const img = screen.getByAltText(/page 2 of handbook\.pdf/i);
    expect(img.getAttribute("src")).toContain("/api/sources/s1/page/2.png");
    expect(img.getAttribute("src")).toContain("session_id=sess");

    const boxes = document.querySelectorAll(".viewer-rect");
    expect(boxes).toHaveLength(2);
    expect(boxes[0].style.left).toBe("10%");
    expect(boxes[0].style.top).toBe("10%");
    expect(boxes[0].style.width).toBe("40%");
    expect(boxes[0].style.height).toBe("2%");
  });

  it("ignores malformed rectangles rather than drawing somewhere arbitrary", () => {
    const location = { ...PDF_LOCATION, rects: [[1, 2, 3], ["a", 1, 2, 3], [60, 80, 300, 96]] };
    render(<DocumentViewer sessionId="sess" citation={{ location }} onClose={() => {}} />);
    expect(document.querySelectorAll(".viewer-rect")).toHaveLength(1);
  });

  it("shows the parsed text with only the quoted spans marked, as inert text", async () => {
    const text = "Intro <b>bold</b> sentence. The answer is 42 days. Trailing <script>x()</script> text.";
    const a = text.indexOf("The answer is 42 days.");
    vi.spyOn(api, "sourceText").mockResolvedValue({
      text,
      marks: [{ start: a, end: a + "The answer is 42 days.".length }],
      before: false,
      after: false,
    });
    const location = { kind: "text", source_id: "s2", source: "notes.md", spans: [{ start: 100, end: 122 }] };
    render(<DocumentViewer sessionId="sess" citation={{ location }} onClose={() => {}} />);

    const mark = await screen.findByText("The answer is 42 days.");
    expect(mark.tagName).toBe("MARK");
    expect(document.querySelectorAll(".viewer-mark")).toHaveLength(1);
    // Source text is rendered as text, never parsed as markup.
    expect(document.querySelector(".viewer-text b")).toBeNull();
    expect(document.querySelector(".viewer-text script")).toBeNull();
    expect(document.querySelector(".viewer-text pre").textContent).toBe(text);
    expect(api.sourceText).toHaveBeenCalledWith("sess", "s2", location.spans);
  });

  it("marks every quoted span and ignores a mark that does not fit the text", async () => {
    vi.spyOn(api, "sourceText").mockResolvedValue({
      text: "one two three four",
      marks: [
        { start: 0, end: 3 },
        { start: 8, end: 13 },
        { start: 14, end: 99 },
      ],
    });
    const location = { kind: "text", source_id: "s2", source: "n.txt", spans: [] };
    render(<DocumentViewer sessionId="s" citation={{ location }} onClose={() => {}} />);
    await waitFor(() => expect(document.querySelectorAll(".viewer-mark")).toHaveLength(2));
    expect([...document.querySelectorAll(".viewer-mark")].map((m) => m.textContent)).toEqual(["one", "three"]);
  });

  it("explains that an old source needs a re-index", async () => {
    vi.spyOn(api, "sourceText").mockResolvedValue({ text: null, reason: "reindex", marks: [] });
    const location = { kind: "text", source_id: "s2", source: "n.txt", spans: [] };
    render(<DocumentViewer sessionId="s" citation={{ location }} onClose={() => {}} />);
    expect(await screen.findByText(/re-index it to open the document/i)).toBeInTheDocument();
  });

  it("closes on Escape and on the close button", async () => {
    const onClose = vi.fn();
    render(<DocumentViewer sessionId="s" citation={{ location: PDF_LOCATION }} onClose={onClose} />);
    await userEvent.keyboard("{Escape}");
    await userEvent.click(screen.getByRole("button", { name: /close document/i }));
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("renders nothing for a citation with no location", () => {
    const { container } = render(<DocumentViewer sessionId="s" citation={{}} onClose={() => {}} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("Thread opening the viewer", () => {
  const turn = (location) => ({
    role: "assistant",
    text: "The limit is 5.5 bar [1].",
    evidence: { verdict: "answered", confidence: "high", best_score: 0.6, considered: 3, returned: 1, cited: [1], invalid: [] },
    citations: [{ source: "handbook.pdf", source_id: "s1", page: 2, score: 0.6, text: "The limit is 5.5 bar.", location }],
  });

  it("opens the document when a chip with a location is clicked", async () => {
    render(<Thread turns={[turn(PDF_LOCATION)]} pending={false} sessionId="sess" />);
    await userEvent.click(screen.getByRole("link", { name: "[1]" }));
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(document.querySelectorAll(".viewer-rect")).toHaveLength(2);
  });

  it("still only scrolls to the card when the server found no location", async () => {
    render(<Thread turns={[turn(null)]} pending={false} sessionId="sess" />);
    await userEvent.click(screen.getByRole("link", { name: "[1]" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.querySelector(".cite")).toHaveClass("open");
  });
});
