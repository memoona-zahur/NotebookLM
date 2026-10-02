import React from "react";
import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SourcesDrawer } from "../SourcesDrawer.jsx";

const STATUS = {
  model: "llama-3.3-70b-versatile",
  provider: "Groq",
  embed_model: "sentence-transformers/all-MiniLM-L6-v2",
  chunks: 42,
  min_score: 0.25,
  sources: [
    { id: "s1", name: "notes.pdf", kind: "pdf", chunks: 12 },
    { id: "s2", name: "data.csv", kind: "csv", chunks: 30 },
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

    expect(screen.getByText(/indexing/i)).toBeInTheDocument();
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
});
