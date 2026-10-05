import React, { useEffect, useRef, useState } from "react";

export function Composer({ onSend, busy }) {
  const [value, setValue] = useState("");
  const textarea = useRef(null);

  useEffect(() => {
    // Clear after a send, but keep the box focused so a follow-up can be typed
    // straight away.
    if (!busy) textarea.current?.focus();
  }, [busy]);

  function grow() {
    const node = textarea.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, 180)}px`;
  }

  function submit(event) {
    event.preventDefault();
    const question = value.trim();
    if (!question || busy) return;
    onSend(question);
    setValue("");
    requestAnimationFrame(grow);
  }

  return (
    <div className="composer-wrap">
      <form className="composer" onSubmit={submit}>
        <textarea
          ref={textarea}
          rows={1}
          value={value}
          placeholder="Ask a question about this notebook's sources…"
          onChange={(e) => {
            setValue(e.target.value);
            grow();
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit(e);
            }
          }}
        />
        {/* aria-label, not just title: the glyph alone is the accessible name
            otherwise, which a screen reader announces as "upwards arrow". */}
        <button
          className="send"
          type="submit"
          disabled={busy || !value.trim()}
          title="Send"
          aria-label="Send"
        >
          ↑
        </button>
      </form>
      <p className="hint">
        <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line
      </p>
    </div>
  );
}