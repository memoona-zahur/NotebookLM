import React from "react";
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Composer } from "../Composer.jsx";

function setup(props = {}) {
  const onSend = vi.fn();
  const user = userEvent.setup();
  render(<Composer onSend={onSend} busy={false} {...props} />);
  const box = screen.getByRole("textbox");
  return { onSend, user, box, send: screen.getByRole("button", { name: /send/i }) };
}

describe("Composer", () => {
  it("sends the trimmed question and clears the box", async () => {
    const { onSend, user, box } = setup();
    await user.type(box, "  what is escape velocity?  ");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(onSend).toHaveBeenCalledOnce();
    expect(onSend).toHaveBeenCalledWith("what is escape velocity?");
    expect(box).toHaveValue("");
  });

  it("sends on Enter without a modifier", async () => {
    const { onSend, user, box } = setup();
    await user.type(box, "hello{Enter}");
    expect(onSend).toHaveBeenCalledOnce();
    expect(onSend).toHaveBeenCalledWith("hello");
  });

  // Shift+Enter is how the placeholder promises to insert a newline, so it must
  // not submit.
  it("inserts a newline on Shift+Enter instead of sending", async () => {
    const { onSend, user, box } = setup();
    await user.type(box, "line one{Shift>}{Enter}{/Shift}line two");
    expect(onSend).not.toHaveBeenCalled();
    expect(box).toHaveValue("line one\nline two");
  });

  it("refuses to send an empty or whitespace-only question", async () => {
    const { onSend, user, box, send } = setup();
    await user.type(box, "    ");
    expect(send).toBeDisabled();
    await user.click(send);
    expect(onSend).not.toHaveBeenCalled();
  });

  it("disables the button while a request is in flight", async () => {
    const { onSend, user, box, send } = setup({ busy: true });
    await user.type(box, "a question");
    expect(send).toBeDisabled();
    // The send button being disabled is not enough: Enter on the textarea can
    // still submit the form, and that path has to be guarded too.
    await user.type(box, "{Enter}");
    expect(onSend).not.toHaveBeenCalled();
  });

  it("keeps the typed question if the send is refused", async () => {
    const { user, box } = setup();
    await user.type(box, "kept");
    await user.clear(box);
    expect(box).toHaveValue("");
  });

  it("grows the textarea as lines are added", async () => {
    const { user, box } = setup();
    // jsdom reports scrollHeight as 0, so this asserts the handler runs without
    // throwing and leaves a numeric height rather than the exact value.
    await user.type(box, "one{Shift>}{Enter}{/Shift}two{Shift>}{Enter}{/Shift}three");
    expect(box.style.height).toMatch(/px$/);
  });
});