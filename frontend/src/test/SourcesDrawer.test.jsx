import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SourcesDrawer } from "../SourcesDrawer.jsx";

// Mirrors what `GET /api/status` actually returns for a source, including the
// ingest-report fields. A trimmed fixture would hide exactly what the expanded
// row renders, which is the part under test below.
const STATUS = {
  model: "llama-3.3-70b-versatile",
  provider: "Groq",
  embed_model: "sentence-transformers/all-MiniLM-L6-v2",
  chunks: 42,
  min_score: 0.25,
  ingest: { token_window: 256, token_max: 12, token_unmeasured: 0 },
  sources: [
    {
      id: "s1",
      name: "notes.pdf",
      kind: "pdf",
      chunks: 12,
      pages: 9,
      numeric: 3,
      token_max: 210,
      fit_splits: 1,
      size_splits: 2,
      file_bytes: 18432,
      avg_chars: 233,
      max_chars: 871,
      created_at: "2026-10-07 12:31:04",
    },
    {
      id: "s2",
      name: "data.csv",
      kind: "csv",
      chunks: 30,
      pages: 1,
      numeric: 14,
      token_max: 96,
      fit_splits: 0,
      size_splits: 0,
      file_bytes: 4096,
      avg_chars: 140,
      max_chars: 512,
      created_at: "2026-10-07 12:33:41",
    },
  ],
};

function setup(props = {}) {
  const onClose = vi.fn();
  const onUpload = vi.fn();
  const onDeleteSource = vi.fn();
  const onClearSources = vi.fn();
  const user = userEvent.setup();
  render(
    <SourcesDrawer
      status={STATUS}
      uploading={false}
      onClose={onClose}
      onUpload={onUpload}
      onDeleteSource={onDeleteSource}
      onClearSources={onClearSources}
      {...props}
    />
  );
  return {
    onClose,
    onUpload,
    onDeleteSource,
    onClearSources,
    user,
    dialog: screen.getByRole("dialog"),
  };
}

// Expanding a source now asks the server for that source's chunks. The default
// answer is an empty map, so every test above the chunk map keeps testing the
// panel it lives in instead of the network call it happens to make.
function stubChunks(payload) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      status: 200,
      text: async () => JSON.stringify(payload),
    }))
  );
}

beforeEach(() => stubChunks({ source_id: "s1", ceiling: { chars: 900, wordpieces: 256 }, chunks: [] }));
afterEach(() => vi.unstubAllGlobals());

// A map small enough that every height can be worked out by hand: the ceiling
// is 100 characters, so a full chunk is a full-height bar and a quarter chunk
// is a quarter one.
const CHUNKS = {
  source_id: "s1",
  ceiling: { chars: 100, wordpieces: 256 },
  chunks: [
    { position: 0, page: 1, heading: "Section one", text: "a".repeat(100), chars: 100, wordpieces: 96, numeric: false },
    { position: 1, page: null, heading: null, text: "b".repeat(50), chars: 50, wordpieces: 48, numeric: true },
    { position: 2, page: null, heading: null, text: "c".repeat(25), chars: 25, wordpieces: 24, numeric: false },
  ],
};

describe("SourcesDrawer", () => {
  it("is exposed as a modal dialog and lists the session's sources", () => {
    const { dialog } = setup();

    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(screen.getByText("notes.pdf")).toBeInTheDocument();
    expect(screen.getByText("data.csv")).toBeInTheDocument();
  });

  // The chat is behind the drawer, so keyboard users must land inside it.
  it("moves focus into the drawer on open", () => {
    setup();

    expect(document.activeElement).toBe(screen.getByRole("button", { name: /close sources/i }));
  });

  it("closes on Escape", async () => {
    const { onClose, user } = setup();

    await user.keyboard("{Escape}");

    expect(onClose).toHaveBeenCalledOnce();
  });

  it("closes when the scrim behind it is clicked", async () => {
    const { onClose, user } = setup();

    await user.click(document.querySelector(".scrim"));

    expect(onClose).toHaveBeenCalledOnce();
  });

  it("closes from the header button", async () => {
    const { onClose, user } = setup();

    await user.click(screen.getByRole("button", { name: /close sources/i }));

    expect(onClose).toHaveBeenCalledOnce();
  });

  it("reports the index configuration that moved out of the rail", () => {
    setup();

    expect(screen.getByText("Groq")).toBeInTheDocument();
    expect(screen.getByText("llama-3.3-70b-versatile")).toBeInTheDocument();
    expect(screen.getByText("all-MiniLM-L6-v2")).toBeInTheDocument();
    expect(screen.getByText("42 chunks")).toBeInTheDocument();
  });

  it("removes a single source and all sources through their controls", async () => {
    const { onDeleteSource, onClearSources, user } = setup();

    await user.click(screen.getByRole("button", { name: /remove notes\.pdf/i }));
    expect(onDeleteSource).toHaveBeenCalledWith(STATUS.sources[0]);

    await user.click(screen.getByRole("button", { name: /remove all sources/i }));
    expect(onClearSources).toHaveBeenCalledOnce();
  });

  // A session with nothing indexed must offer the drop target, not a bare
  // empty list, or there is no way to add a first source.
  it("offers the drop target when nothing is indexed", () => {
    setup({ status: { ...STATUS, sources: [], chunks: 0 } });

    expect(screen.getByText(/nothing indexed/i)).toBeInTheDocument();
    expect(screen.getByText(/add sources/i)).toBeInTheDocument();
  });

  it("shows progress instead of the idle label while indexing", () => {
    setup({ uploading: true });

    // Exact rather than a pattern: the footer is not the only thing on this
    // screen containing the word, and a pattern would pass on the wrong one.
    expect(screen.getByText("Indexing\u2026")).toBeInTheDocument();
    expect(screen.queryByText("Add sources")).not.toBeInTheDocument();
  });

  it("uploads every chosen file", async () => {
    const { onUpload, user } = setup();
    const input = document.querySelector('input[type="file"]');
    const files = [
      new File(["a"], "one.pdf", { type: "application/pdf" }),
      new File(["b"], "two.md", { type: "text/markdown" }),
    ];

    await user.upload(input, files);

    await waitFor(() => expect(onUpload).toHaveBeenCalledTimes(2));
    expect(onUpload).toHaveBeenNthCalledWith(1, files[0]);
    expect(onUpload).toHaveBeenNthCalledWith(2, files[1]);
  });

  it("renders before any session status has loaded", () => {
    setup({ status: null });

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText(/add sources/i)).toBeInTheDocument();
  });

  describe("ingestion detail", () => {
    it("keeps the panel closed until the source row is expanded", () => {
      setup({ sessionId: "sess-1" });

      expect(chevron("notes\\.pdf")).toHaveAttribute("aria-expanded", "false");
      expect(screen.queryByText("View original")).not.toBeInTheDocument();
      expect(screen.queryByText(/210\/256/)).not.toBeInTheDocument();
    });

    it("opens on the chevron and points at that source's original file", async () => {
      const { user } = setup({ sessionId: "sess-1" });

      await user.click(chevron("notes\\.pdf"));

      // The file, not a rendering of it. The href is what the control resolves
      // to, so it has to carry the session as well as the source id. Matched as
      // parts rather than a whole string because it is resolved against the
      // origin, which differs between the browser and the test environment.
      const href = screen.getByText("View original").getAttribute("href");
      expect(href).toContain("/api/sources/s1/file");
      expect(href).toContain("session_id=sess-1");
      expect(screen.getByText(/210\/256 wordpieces/)).toBeInTheDocument();
      expect(screen.getByText(/3 of 12 chunks/)).toBeInTheDocument();
    });

    it("shows the counts of the row that was opened, not of the drawer", async () => {
      const { user } = setup({ sessionId: "sess-1" });

      await user.click(chevron("data\\.csv"));

      expect(screen.getByText(/96\/256 wordpieces/)).toBeInTheDocument();
      expect(screen.queryByText(/210\/256/)).not.toBeInTheDocument();
    });

    it("refuses to say nothing was cut when the counts were never kept", async () => {
      const { user } = setup({
        sessionId: "sess-1",
        status: {
          ...STATUS,
          sources: [
            // Indexed before 0005/0006 were recorded: no wordpiece walk, no
            // size count, no size on disk.
            { ...STATUS.sources[0], token_max: 0, fit_splits: 0, size_splits: null, file_bytes: null, created_at: "" },
          ],
        },
      });

      await user.click(chevron("notes\\.pdf"));

      const facts = document.querySelector(".detail-facts");
      // Patterns, because a fact row joins its two ceilings into one cell, so
      // an exact match would never be the cell itself. Scoped to the fact list
      // because the trace line above repeats these phrases.
      expect(within(facts).getByText(/size cuts not measured/)).toBeInTheDocument();
      expect(within(facts).getByText(/token cuts not measured/)).toBeInTheDocument();
      expect(within(facts).getByText(/size not recorded/)).toBeInTheDocument();
      expect(within(facts).getByText(/before this was recorded/)).toBeInTheDocument();
      expect(within(facts).queryByText(/0 cut by size/)).not.toBeInTheDocument();
      expect(within(facts).queryByText(/0 re-cut for tokens/)).not.toBeInTheDocument();
    });

    it("says only when the upload matched something already indexed", async () => {
      const { user } = setup({ sessionId: "sess-1", duplicateIds: new Set(["s1"]) });

      await user.click(chevron("notes\\.pdf"));
      expect(screen.getByText(/Nothing was re-indexed/i)).toBeInTheDocument();

      await user.click(chevron("data\\.csv"));
      expect(screen.queryByText(/Nothing was re-indexed/i)).not.toBeInTheDocument();
    });
  });

  describe("the pipeline diagram", () => {
    it("is closed by default and draws all six stages once opened", async () => {
      const { user } = setup();

      const toggle = screen.getByRole("button", { name: /how indexing works/i });
      expect(toggle).toHaveAttribute("aria-expanded", "false");
      expect(screen.queryByText("Upload and hash")).not.toBeInTheDocument();

      await user.click(toggle);

      expect(toggle).toHaveAttribute("aria-expanded", "true");
      const stages = document.querySelectorAll(".pipeline .stage");
      expect(stages).toHaveLength(6);
      // One icon per stage, and one short line of text at most. A diagram with
      // no pictures in it is the prose this replaced, wearing a list.
      for (const stage of stages) {
        expect(stage.querySelector("svg")).not.toBeNull();
        const tag = stage.querySelector(".stage-tag");
        expect((tag && tag.textContent.trim()) || "").not.toBe("");
      }
      expect(screen.getByText("Upload and hash")).toBeInTheDocument();
      expect(screen.getByText("Store for retrieval")).toBeInTheDocument();
      expect(screen.getByText(/SHA-256/)).toBeInTheDocument();
    });

    it("closes again rather than staying open in the way", async () => {
      const { user } = setup();

      await user.click(screen.getByRole("button", { name: /how indexing works/i }));
      await user.click(screen.getByRole("button", { name: /hide how indexing works/i }));

      expect(screen.queryByText("Upload and hash")).not.toBeInTheDocument();
    });
  });

  describe("the chunk map", () => {
    it("draws one bar per stored chunk, sized against the ceiling", async () => {
      stubChunks(CHUNKS);
      const { user } = setup({ sessionId: "sess-1" });

      await user.click(chevron("notes\\.pdf"));

      await waitFor(() => expect(document.querySelectorAll(".chunk-bar")).toHaveLength(3));

      // Fetched for the row that was opened, and only that one: a notebook
      // with a dozen sources must not pay for the eleven nobody opened.
      const url = String(fetch.mock.calls[0][0]);
      expect(url).toContain("/api/sources/s1/chunks");
      expect(url).toContain("session_id=sess-1");

      const bars = [...document.querySelectorAll(".chunk-bar")];
      // Height, not width: the picture is a skyline under a ceiling line, so a
      // full chunk fills the box and a quarter chunk is a quarter of it.
      expect(bars.map((bar) => bar.style.height)).toEqual(["48px", "24px", "12px"]);
      // A table chunk is drawn as one because damping did something different
      // to it, and the legend says what that colour means.
      expect(bars[1]).toHaveClass("numeric");
      expect(bars[0]).not.toHaveClass("numeric");
      // Document order, not insertion order of anything else.
      expect(bars[0]).toHaveAttribute(
        "aria-label",
        "Chunk 1: 100 characters, 96 wordpieces"
      );
      expect(
        screen.getByText(/25 min · 50 median · 100 p90 · 100 max chars/)
      ).toBeInTheDocument();
    });

    it("reads a chunk exactly as it was indexed when its bar is chosen", async () => {
      stubChunks(CHUNKS);
      const { user } = setup({ sessionId: "sess-1" });
      await user.click(chevron("notes\\.pdf"));
      await waitFor(() => expect(document.querySelectorAll(".chunk-bar")).toHaveLength(3));

      await user.click(screen.getByRole("button", { name: /^Chunk 1:/ }));

      const view = document.querySelector(".chunk-view");
      expect(view).not.toBeNull();
      expect(within(view).getByText("Section one")).toBeInTheDocument();
      expect(within(view).getByText(/96\/256 wordpieces/)).toBeInTheDocument();
      // The stored text, not a preview of it: this is the piece the retrieval
      // will quote, and the whole point is to be able to check that.
      expect(within(view).getByText("a".repeat(100))).toBeInTheDocument();

      // Choosing another moves the view rather than stacking a second one.
      await user.click(screen.getByRole("button", { name: /^Chunk 2:/ }));
      expect(document.querySelectorAll(".chunk-view")).toHaveLength(1);
      expect(within(document.querySelector(".chunk-view")).getByText("table")).toBeInTheDocument();
      expect(within(document.querySelector(".chunk-view")).queryByText("Section one")).toBeNull();
    });

    it("distinguishes a source with nothing stored from a failed fetch", async () => {
      stubChunks({ ...CHUNKS, chunks: [] });
      const { user } = setup({ sessionId: "sess-1" });

      await user.click(chevron("notes\\.pdf"));

      expect(await screen.findByText(/No chunks are stored for this source/i)).toBeInTheDocument();
      expect(screen.queryByText(/Could not draw/i)).not.toBeInTheDocument();
    });

    it("says why the map is missing rather than showing an empty box", async () => {
      vi.stubGlobal(
        "fetch",
        vi.fn(async () => ({
          ok: false,
          status: 404,
          text: async () => JSON.stringify({ detail: "That source is not in this notebook." }),
        }))
      );
      const { user } = setup({ sessionId: "sess-1" });

      await user.click(chevron("notes\\.pdf"));

      expect(await screen.findByText(/Could not draw the chunk map/i)).toBeInTheDocument();
      expect(screen.getByText(/not in this notebook/i)).toBeInTheDocument();
      expect(document.querySelectorAll(".chunk-bar")).toHaveLength(0);
    });
  });
});

// The chevron is the only control that opens a source's detail, and its label
// carries the source name so two open rows cannot be confused for one.
function chevron(name) {
  return screen.getByRole("button", {
    name: new RegExp(`show ingestion details for ${name}`, "i"),
  });
}
