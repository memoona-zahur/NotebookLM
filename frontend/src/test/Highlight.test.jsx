import React from "react";
import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Highlighted, Occurrences } from "../Highlight.jsx";

// The sentence the tests mark inside. Spans are derived from the text rather
// than written by hand, so a reworded fixture cannot quietly shift them.
const PASSAGE = "Escape velocity from the surface is approximately 11.2 km/s.";

function spansOf(text, ...words) {
  let from = 0;
  return words.map((word) => {
    const at = text.indexOf(word, from);
    from = at + word.length;
    return [at, at + word.length];
  });
}

const VELOCITY = spansOf(PASSAGE, "velocity");

describe("Highlighted", () => {
  it("marks the spans the server reported and nothing else", () => {
    render(<Highlighted text={PASSAGE} matches={VELOCITY} />);

    const mark = screen.getByText("velocity");
    expect(mark.tagName).toBe("MARK");
    expect(mark).toHaveClass("hit");
    // The word either side of the mark must survive intact, or the passage was
    // sliced wrong.
    expect(screen.getByText(/from the surface is/)).toBeInTheDocument();
  });

  it("marks every occurrence when the word appears more than once", () => {
    const text = "the rate is set, so the rate holds";
    render(<Highlighted text={text} matches={spansOf(text, "rate", "rate")} />);

    expect(screen.getAllByText("rate")).toHaveLength(2);
  });

  it("leaves the text alone when there is nothing to mark", () => {
    render(<Highlighted text={PASSAGE} matches={[]} />);
    expect(document.querySelector("mark")).toBeNull();
    expect(screen.getByText(PASSAGE)).toBeInTheDocument();
  });

  // Spans come from the server, but rendering them out of order would slice the
  // string backwards and produce mojibake, so they are sorted first.
  it("marks correctly when the spans arrive out of order", () => {
    const text = "alpha beta gamma";
    render(<Highlighted text={text} matches={[[11, 16], [0, 5]]} />);

    expect(screen.getByText("alpha")).toBeInTheDocument();
    expect(screen.getByText("gamma")).toBeInTheDocument();
    expect(document.querySelector("mark")).not.toBeNull();
  });

  // A span past the end of the text means the two sides disagree about the
  // string. Dropping the mark keeps the passage readable.
  it("ignores spans that fall outside the text", () => {
    render(<Highlighted text="short" matches={[[0, 99]]} />);
    expect(document.querySelector("mark")).toBeNull();
    expect(screen.getByText("short")).toBeInTheDocument();
  });

  it("ignores a reversed or empty span rather than rendering nonsense", () => {
    render(<Highlighted text="alpha beta" matches={[[6, 2], [3, 3]]} />);
    expect(document.querySelector("mark")).toBeNull();
  });

  it("does not double-mark overlapping spans", () => {
    const text = "escape velocity";
    render(<Highlighted text={text} matches={[[0, 12], [7, 16]]} />);

    // The second span starts inside the first, so only the first is marked and
    // the tail is plain text rather than a second overlapping mark.
    expect(screen.getAllByRole("mark")).toHaveLength(1);
    expect(document.body.textContent).toBe("escape velocity");
  });
});

const RESULT = {
  term: "velocity",
  count: 3,
  truncated: false,
  occurrences: [
    {
      source: "notes.pdf",
      kind: "pdf",
      position: 0,
      page: 2,
      heading: "",
      text: PASSAGE,
      matches: spansOf(PASSAGE, "velocity"),
      count: 1,
    },
    {
      source: "data.csv",
      kind: "csv",
      position: 1,
      page: 0,
      heading: "",
      text: "velocity is recorded per second",
      matches: [[0, 8]],
      count: 2,
    },
  ],
};

// The second search in the staleness test returns its own answer, so that the
// rendered word alone reveals which search won.
const ORBIT = {
  term: "orbit",
  count: 9,
  truncated: false,
  occurrences: [
    {
      source: "notes.pdf",
      kind: "pdf",
      position: 0,
      page: 1,
      heading: "",
      text: "The orbit period is 90 minutes.",
      matches: [[4, 10]],
      count: 1,
    },
  ],
};

describe("Occurrences", () => {
  function setup(onFind = vi.fn().mockResolvedValue(RESULT)) {
    const user = userEvent.setup();
    render(<Occurrences onFind={onFind} onClose={vi.fn()} />);
    return { user, onFind };
  }

  it("asks for the word that was typed and reports what it found", async () => {
    const { user, onFind } = setup();

    await user.type(screen.getByLabelText(/word to highlight/i), "velocity");
    await user.click(screen.getByRole("button", { name: /find/i }));

    await waitFor(() => expect(onFind).toHaveBeenCalledWith("velocity"));
    expect(await screen.findByText("3")).toBeInTheDocument();
    expect(screen.getByText("notes.pdf")).toBeInTheDocument();
    expect(screen.getByText("data.csv")).toBeInTheDocument();
  });

  it("marks the word inside the passage it opens, without needing a click", async () => {
    const { user } = setup();

    await user.type(screen.getByLabelText(/word to highlight/i), "velocity");
    await user.click(screen.getByRole("button", { name: /find/i }));

    // Asking for a word and receiving only closed rows means the highlight is
    // not visible yet, which is the whole point of asking.
    const first = await screen.findByText("notes.pdf");
    const open = first.closest("details");
    expect(open.open).toBe(true);

    const marks = open.querySelectorAll("mark.hit");
    expect(marks.length).toBe(1);
    expect(marks[0].textContent.toLowerCase()).toBe("velocity");

    // The rest stay closed so a common word does not bury the drawer.
    const second = screen.getByText("data.csv").closest("details");
    expect(second.open).toBe(false);
  });

  // Nothing indexed is a normal answer, not an error: the word may simply not
  // be in these documents.
  it("says so when the word is not in any document", async () => {
    const { user } = setup(
      vi.fn().mockResolvedValue({
        term: "sourdough",
        count: 0,
        truncated: false,
        occurrences: [],
      }),
    );

    await user.type(screen.getByLabelText(/word to highlight/i), "sourdough");
    await user.click(screen.getByRole("button", { name: /find/i }));

    expect(await screen.findByText(/not in any of these documents/i)).toBeInTheDocument();
    expect(document.querySelector("mark")).toBeNull();
  });

  it("shows the failure next to the box instead of leaving an empty panel", async () => {
    const { user } = setup(vi.fn().mockRejectedValue(new Error("Word too long")));

    await user.type(screen.getByLabelText(/word to highlight/i), "x".repeat(200));
    await user.click(screen.getByRole("button", { name: /find/i }));

    expect(await screen.findByText(/word too long/i)).toBeInTheDocument();
  });

  // A slow first search must not overwrite the results of the second one.
  it("keeps the newest result when an older one lands late", async () => {
    const user = userEvent.setup();
    let releaseSlow;
    const slow = new Promise((resolve) => {
      releaseSlow = resolve;
    });
    const onFind = vi.fn().mockImplementationOnce(() => slow).mockResolvedValueOnce(ORBIT);

    render(<Occurrences onFind={onFind} onClose={vi.fn()} />);
    const box = screen.getByLabelText(/word to highlight/i);
    const go = screen.getByRole("button", { name: /find/i });

    await user.type(box, "velocity");
    await user.click(go);
    await waitFor(() => expect(onFind).toHaveBeenCalledTimes(1));

    // The second search has to be launchable while the first is still in
    // flight, which is exactly what the "…" must not be blocking.
    await user.clear(box);
    await user.type(box, "orbit");
    await user.click(go);
    await waitFor(() => expect(onFind).toHaveBeenCalledTimes(2));

    // The word shows twice on success: once in the summary, once marked in the
// passage. Either is proof the newer answer is the one on screen.
    const newerWord = () => screen.getAllByText("orbit");
    expect(await screen.findByText("9")).toBeInTheDocument();
    expect(newerWord().length).toBeGreaterThan(0);

    // The first search finally lands, carrying a different word and count.
    releaseSlow(RESULT);
    await waitFor(() => expect(screen.getByText("9")).toBeInTheDocument());
    expect(screen.queryAllByText("velocity")).toHaveLength(0);
    expect(newerWord().length).toBeGreaterThan(0);
  });

  it("does not search on an empty word", async () => {
    const { user, onFind } = setup();

    await user.click(screen.getByRole("button", { name: /find/i }));

    expect(onFind).not.toHaveBeenCalled();
  });

  it("reports how many times the word occurs in each passage", async () => {
    const { user } = setup();

    await user.type(screen.getByLabelText(/word to highlight/i), "velocity{Enter}");

    const hits = await screen.findAllByText("data.csv");
    const detail = hits[0].closest("details");
    expect(within(detail).getByText("2 times")).toBeInTheDocument();
  });

  it("says when only the first occurrences are being shown", async () => {
    const user = userEvent.setup();
    render(
      <Occurrences
        onFind={vi.fn().mockResolvedValue({ ...RESULT, count: 412, truncated: true })}
        onClose={vi.fn()}
      />,
    );

    await user.type(screen.getByLabelText(/word to highlight/i), "velocity{Enter}");

    expect(await screen.findByText(/showing the first ones/i)).toBeInTheDocument();
  });
});