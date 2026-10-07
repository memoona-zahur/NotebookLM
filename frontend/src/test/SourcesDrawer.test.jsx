import React from "react";
import { describe, expect, it, vi } from "vitest";
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

  describe("the pipeline explainer", () => {
    it("is closed by default and lists every step once opened", async () => {
      const { user } = setup();

      const toggle = screen.getByRole("button", { name: /how indexing works/i });
      expect(toggle).toHaveAttribute("aria-expanded", "false");

      await user.click(toggle);

      expect(toggle).toHaveAttribute("aria-expanded", "true");
      for (const heading of ["1 \u00b7 Written and hashed", "6 \u00b7 Stored for retrieval"]) {
        expect(screen.getByText(heading)).toBeInTheDocument();
      }
      expect(screen.getByText(/SHA-256/)).toBeInTheDocument();
    });

    it("closes again rather than staying open in the way", async () => {
      const { user } = setup();

      await user.click(screen.getByRole("button", { name: /how indexing works/i }));
      await user.click(screen.getByRole("button", { name: /hide how indexing works/i }));

      expect(screen.queryByText("1 \u00b7 Written and hashed")).not.toBeInTheDocument();
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
