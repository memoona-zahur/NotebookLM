import React from "react";
import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "../App.jsx";
import { deferred, on, requests, response, seed } from "./harness.js";

async function boot() {
  const user = userEvent.setup();
  render(<App />);
  await waitFor(() => expect(screen.getByRole("heading", { name: "Alpha" })).toBeInTheDocument());
  return user;
}

async function ask(user, question) {
  const box = screen.getByRole("textbox");
  await user.type(box, question);
  await user.click(screen.getByRole("button", { name: /send/i }));
}

const HITS = {
  term: "velocity",
  count: 1,
  truncated: false,
  occurrences: [
    {
      source: "kepler.pdf",
      kind: "pdf",
      position: 0,
      page: 1,
      heading: "",
      text: "Escape velocity is 11.2 km/s.",
      matches: [[7, 15]],
      count: 1,
    },
  ],
};

const ANSWER = {
  answer: "Escape velocity at the surface is 11.2 km/s [1].",
  evidence: {
    cited: [1],
    verdict: "answered",
    confidence: "high",
    best_score: 0.51,
    considered: 40,
    returned: 6,
    invalid: [],
    cost: {
      called: true,
      total_tokens: 8234,
      estimated: false,
      retrieval_ms: 41.2,
      generation_ms: 890.4,
    },
  },
  citations: [
    { source: "kepler.pdf", page: 3, score: 0.42, text: "The photometer has 42 CCDs." },
    { source: "kepler.pdf", page: 4, score: 0.31, text: "Orbital period was 90 minutes." },
  ],
};

const NO_MATCH = {
  answer: "No answer found in the uploaded sources.",
  evidence: {
    cited: [],
    verdict: "no_match",
    best_score: 0.11,
    min_score: 0.25,
    cost: { called: false, retrieval_ms: 12, total_tokens: 0 },
  },
  citations: [],
};

describe("first paint", () => {
  it("states the grounding rules instead of showing a blank page", async () => {
    await boot();
    expect(screen.getByText(/Nothing retrieved means no answer/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });

  it("lists sessions and marks the active one", async () => {
    await boot();
    const rail = screen.getByRole("complementary");
    const options = within(rail).getAllByRole("option");
    expect(options).toHaveLength(2);
    expect(options[0]).toHaveAttribute("aria-selected", "true");
    expect(options[0]).toHaveTextContent("Alpha");
    expect(options[1]).toHaveTextContent("Beta");
  });

  // Regression: the server returns `content`, the client read `text`, and the
  // mismatch threw on mount so every session with history rendered blank.
  it("replays a persisted transcript from the server", async () => {
    seed("alpha", [
      { role: "user", content: "how many CCDs?" },
      { role: "assistant", content: "42 CCDs [1]." },
    ]);
    await boot();
    expect(screen.getByText("how many CCDs?")).toBeInTheDocument();
    // The body is split around the [1] marker, so match on the container.
    expect(
      screen.getByText((_, el) => el?.className === "answer" && /42 CCDs/.test(el.textContent)),
    ).toBeInTheDocument();
    expect(screen.getByText("You")).toBeInTheDocument();
    expect(screen.getAllByText("NotebookLM").length).toBeGreaterThan(0);
    // Both turns rendered, in order, rather than only the first one.
    expect(screen.getAllByText("You")).toHaveLength(1);
  });

  it("renders a persisted turn that has no body rather than crashing", async () => {
    seed("alpha", [{ role: "assistant", content: "" }]);
    await boot();
    expect(screen.getByRole("heading", { name: "Alpha" })).toBeInTheDocument();
  });

  it("renders a turn from an older build that stored text instead of content", async () => {
    seed("alpha", [{ role: "assistant", text: "cached locally" }]);
    await boot();
    expect(screen.getByText("cached locally")).toBeInTheDocument();
  });

  it("shows no evidence strip for a replayed answer", async () => {
    // Replayed turns carry no citations or evidence: the server stores the
    // message body only. The UI must not imply grounding it cannot show.
    seed("alpha", [{ role: "assistant", content: "42 CCDs [1]." }]);
    await boot();
    expect(screen.queryByText(/confidence/)).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "[1]" })).not.toBeInTheDocument();
  });
});

describe("asking a question", () => {
  it("shows the question, then the answer with a working citation link", async () => {
    on("POST", "/api/ask", () => ANSWER);
    const user = await boot();

    await ask(user, "what is escape velocity?");

    expect(await screen.findByText(/Escape velocity at the surface/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "[1]" })).toHaveAttribute("href", "#cite-1");
    expect(screen.getByText("high confidence")).toBeInTheDocument();
  });

  // Cost sits next to confidence on purpose: "should I believe this" and "was
  // this worth asking" are the two questions a grounded system has to answer.
  it("shows what the question cost next to the confidence chip", async () => {
    on("POST", "/api/ask", () => ANSWER);
    const user = await boot();

    await ask(user, "what is escape velocity?");

    expect(await screen.findByText(/8,234 tokens/)).toBeInTheDocument();
    expect(screen.getByText(/890ms to answer/)).toBeInTheDocument();
  });

  it("says a refusal spent no tokens", async () => {
    on("POST", "/api/ask", () => NO_MATCH);
    const user = await boot();

    await ask(user, "unrelated question");

    expect(await screen.findByText(/no tokens spent/i)).toBeInTheDocument();
  });

  it("lists the passages behind the answer", async () => {
    on("POST", "/api/ask", () => ANSWER);
    const user = await boot();
    await ask(user, "a question");

    await screen.findByText(/Escape velocity/);
    expect(screen.getByText(/The photometer has 42 CCDs/)).toBeInTheDocument();
    // Retrieved but uncited: shown, but marked as unused.
    expect(screen.getByText(/Orbital period was 90 minutes/)).toBeInTheDocument();
  });

  it("clears the composer and accepts the next question", async () => {
    on("POST", "/api/ask", () => ANSWER);
    const user = await boot();

    await ask(user, "second question");

    await screen.findByText(/Escape velocity/);
    const box = screen.getByRole("textbox");
    expect(box).toHaveValue("");
    // The button is disabled while the box is empty by design; typing has to
    // bring it back, which only works if the busy flag was cleared.
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
    await user.type(box, "another question");
    expect(screen.getByRole("button", { name: /send/i })).toBeEnabled();
  });

  it("scopes the ask to the session it was sent from", async () => {
    on("POST", "/api/ask", () => ANSWER);
    const user = await boot();
    await ask(user, "scoped question");

    const posted = requests().find((r) => r.method === "POST" && r.path === "/api/ask");
    expect(posted.sessionId).toBe("alpha");
  });

  it("switches session and shows that session's transcript", async () => {
    seed("beta", [{ role: "user", content: "beta question" }]);
    const user = await boot();

    await user.click(screen.getByRole("option", { name: /Beta/ }));

    expect(await screen.findByText("beta question")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Beta" })).toBeInTheDocument();
    expect(screen.queryByText("alpha question")).not.toBeInTheDocument();
  });
});

describe("the session-switch race", () => {
  // The regression that motivated these tests: a slow reply for the session the
  // user had already left used to land in whatever session was on screen.
  it("drops a late answer into nothing when the user has moved on", async () => {
    const slow = deferred();
    on("POST", "/api/ask", async ({ sessionId }) => {
      if (sessionId === "alpha") await slow.promise;
      return ANSWER;
    });

    const user = await boot();
    await ask(user, "asked in Alpha");

    await user.click(screen.getByRole("option", { name: /Beta/ }));
    await screen.findByRole("heading", { name: "Beta" });

    slow.resolve();
    await new Promise((r) => setTimeout(r, 60));

    expect(screen.queryByText(/Escape velocity at the surface/)).not.toBeInTheDocument();
    expect(screen.getByText(/Nothing retrieved means no answer/i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Beta" })).toBeInTheDocument();
  });

  it("leaves the thread usable after discarding a stale reply", async () => {
    const slow = deferred();
    on("POST", "/api/ask", async ({ sessionId }) => {
      if (sessionId === "alpha") await slow.promise;
      return ANSWER;
    });

    const user = await boot();
    await ask(user, "asked in Alpha");
    await user.click(screen.getByRole("option", { name: /Beta/ }));
    await screen.findByRole("heading", { name: "Beta" });

    slow.resolve();
    await new Promise((r) => setTimeout(r, 60));

    // Not stuck showing "Reading your sources" forever.
    expect(screen.queryByText(/Reading your sources/)).not.toBeInTheDocument();

    const box = screen.getByRole("textbox");
    await user.type(box, "asking Beta instead");
    expect(screen.getByRole("button", { name: /send/i })).toBeEnabled();
  });
});

describe("failure modes", () => {
  it("reports a grounded no_match rather than an answer", async () => {
    on("POST", "/api/ask", () => NO_MATCH);
    const user = await boot();

    await ask(user, "something unrelated");

    expect(await screen.findByText(/No answer found in the uploaded sources/)).toBeInTheDocument();
    expect(screen.getByText(/best match 11%, below the 25% floor/)).toBeInTheDocument();
    expect(screen.getByText(/The model was not asked, so nothing was invented/)).toBeInTheDocument();
  });

  it("shows the server's message when the LLM is unavailable", async () => {
    on("POST", "/api/ask", () => response({ detail: "LLM unavailable" }, 503));
    const user = await boot();

    await ask(user, "anything");

    expect(await screen.findByText("LLM unavailable")).toBeInTheDocument();
  });

  it("shows the error in the thread and clears the thinking state", async () => {
    on("POST", "/api/ask", () => response({ detail: "boom" }, 500));
    const user = await boot();

    await ask(user, "will this be lost?");

    await screen.findByText("boom");
    expect(screen.queryByText(/Reading your sources/)).not.toBeInTheDocument();
    await user.type(screen.getByRole("textbox"), "retry");
    expect(screen.getByRole("button", { name: /send/i })).toBeEnabled();
  });

  it("recovers and can answer on the next attempt", async () => {
    on("POST", "/api/ask", () => response({ detail: "boom" }, 500));
    const user = await boot();
    await ask(user, "first try");
    await screen.findByText("boom");

    on("POST", "/api/ask", () => ANSWER);
    await ask(user, "second try");

    expect(await screen.findByText(/Escape velocity at the surface/)).toBeInTheDocument();
  });

  it("tells the user when the server is unreachable", async () => {
    on("POST", "/api/ask", () => ANSWER);
    const user = await boot();

    // A transport failure, not an HTTP error: fetch itself throws.
    on("POST", "/api/ask", () => {
      throw new TypeError("Failed to fetch");
    });
    await ask(user, "does the user get told?");

    expect(await screen.findByText(/Could not reach the server/)).toBeInTheDocument();
  });
});

describe("two instructions in one message", () => {
  // The regression this exists for: "what is the fee? highlight late payment"
  // used to be sent to the server whole, so retrieval searched for the words
  // "highlight late payment" too. That is noise in the dense query and in BM25,
  // and it is unrecoverable downstream.
  it("asks the question and highlights, and sends only the question to the model", async () => {
    on("POST", "/api/ask", () => ANSWER);
    const user = await boot();

    await ask(user, "what is the return window? highlight 30 days");

    expect(await screen.findByText(/Escape velocity at the surface/)).toBeInTheDocument();
    const sent = JSON.parse(requests().find((r) => r.path === "/api/ask").body);
    expect(sent.question).toBe("what is the return window");
    expect(sent.question).not.toMatch(/highlight/i);

    // And the drawer opened on the word, without waiting for the answer.
    const drawer = await screen.findByRole("dialog");
    await waitFor(() =>
      expect(within(drawer).getByPlaceholderText(/highlight a word/i)).toHaveValue("30 days")
    );
  });

  // The user typed one message, so the thread shows one message. Appending per
  // branch in send() would render it twice, which reads as a duplicate send.
  it("shows the message the user typed, once, not the stripped question", async () => {
    on("POST", "/api/ask", () => ANSWER);
    const user = await boot();

    await ask(user, "what is the return window? highlight 30 days");

    await screen.findByText(/Escape velocity at the surface/);
    expect(screen.getAllByText("what is the return window? highlight 30 days")).toHaveLength(1);
  });

  // A bare command must not become an answer, and must not be asked about.
  it("never calls the model for a bare highlight command", async () => {
    const user = await boot();

    await ask(user, "highlight late payment");

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(requests().filter((r) => r.path === "/api/ask")).toHaveLength(0);
  });
});

describe("chat-first layout", () => {
  // Sources left the rail and moved into a drawer, so the conversation is
  // what occupies the screen. This is the part that can break silently: the
  // drawer exists as a component but nothing reaches it.
  it("keeps the rail to sessions and hides sources until asked", async () => {
    const user = await boot();
    const rail = screen.getByRole("complementary");

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(within(rail).queryByText(/Add sources/i)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /sources/i }));

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    expect(within(rail).queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("reports the source count on the toggle and again in the drawer", async () => {
    // The count comes from /api/status, whose harness default is deliberately
    // empty, so a test about a count has to script the documents it counts.
    on("GET", "/api/status", () => ({
      provider: "groq",
      model: "openai/gpt-oss-120b",
      embed_model: "sentence-transformers/all-MiniLM-L6-v2",
      min_score: 0.25,
      // `id` is included because the real /api/status always sends one and the
      // drawer keys its list on it; a fixture without it makes React warn.
      sources: [
        { id: "s1", name: "kepler.pdf", kind: "pdf", chunks: 2 },
        { id: "s2", name: "notes.txt", kind: "txt", chunks: 1 },
      ],
      chunks: 3,
    }));

    const user = await boot();
    const toggle = screen.getByRole("button", { name: /sources/i });

    // The heading appears before the source list has loaded, so the count has
    // to be waited for rather than read on the first paint.
    await waitFor(() => expect(toggle).toHaveTextContent(/2 sources/));

    await user.click(toggle);

    expect(await screen.findByText("kepler.pdf")).toBeInTheDocument();
  });

  it("closes the drawer again and leaves the thread reachable", async () => {
    const user = await boot();
    await user.click(screen.getByRole("button", { name: /sources/i }));
    await screen.findByRole("dialog");

    await user.keyboard("{Escape}");

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(screen.getByRole("textbox")).toBeEnabled();
  });
});

describe("highlight commands from the chat", () => {
  it("opens the drawer with the word marked instead of asking the model", async () => {
    on("GET", "/api/occurrences", () => HITS);
    on("POST", "/api/ask", () => ANSWER);

    const user = await boot();
    await ask(user, "highlight velocity");

    // The whole point: no model call, because this is a lookup.
    expect(requests().filter((r) => r.path === "/api/ask")).toHaveLength(0);

    const drawer = await screen.findByRole("dialog");
    expect(within(drawer).getByDisplayValue("velocity")).toBeInTheDocument();
    // Twice on purpose: once named in the summary, once marked in the passage.
    await waitFor(() => expect(within(drawer).getAllByText("velocity").length).toBeGreaterThan(0));
    expect(within(drawer).getByText("kepler.pdf")).toBeInTheDocument();
  });

  it("shows the command in the thread, so the transcript still matches", async () => {
    on("GET", "/api/occurrences", () => HITS);

    const user = await boot();
    await ask(user, "highlight velocity");

    expect(await screen.findByText("highlight velocity")).toBeInTheDocument();
  });

  it("looks up the word rather than asking about where-is", async () => {
    on("GET", "/api/occurrences", () => HITS);
    on("POST", "/api/ask", () => ANSWER);

    const user = await boot();
    await ask(user, "where does velocity appear");

    expect(requests().filter((r) => r.path === "/api/ask")).toHaveLength(0);
    expect(await screen.findByRole("dialog")).toBeInTheDocument();
  });

  it("still answers a real question about a word", async () => {
    on("POST", "/api/ask", () => ANSWER);

    const user = await boot();
    await ask(user, "where is the capital of France");

    await waitFor(() => expect(requests().filter((r) => r.path === "/api/ask")).toHaveLength(1));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("still answers a question that merely contains the word highlight", async () => {
    on("POST", "/api/ask", () => ANSWER);

    const user = await boot();
    await ask(user, "highlight the difference between the two papers");

    await waitFor(() => expect(requests().filter((r) => r.path === "/api/ask")).toHaveLength(1));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  // The same word twice is two requests, not one repeat being swallowed by the
  // guard that stops a re-render re-running a command.
  it("searches again when the same word is asked for twice", async () => {
    on("GET", "/api/occurrences", () => HITS);

    const user = await boot();
    await ask(user, "highlight velocity");
    await screen.findByRole("dialog");
    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    await ask(user, "highlight velocity");
    await screen.findByRole("dialog");

    await waitFor(() =>
      expect(requests().filter((r) => r.path === "/api/occurrences")).toHaveLength(2),
    );
  });

  // Otherwise reopening the drawer silently re-runs the previous search, which
  // reads as the app having remembered a question nobody asked again.
  it("does not re-run the last search when the drawer is reopened by hand", async () => {
    on("GET", "/api/occurrences", () => HITS);

    const user = await boot();
    await ask(user, "highlight velocity");
    await screen.findByRole("dialog");
    await user.keyboard("{Escape}");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: /sources/i }));
    await screen.findByRole("dialog");

    // Still just the one search from the command.
    expect(requests().filter((r) => r.path === "/api/occurrences")).toHaveLength(1);
  });

  it("reports nothing found rather than claiming the word is absent", async () => {
    on("GET", "/api/occurrences", () => ({
      term: "velocity",
      count: 0,
      truncated: false,
      occurrences: [],
    }));

    const user = await boot();
    await ask(user, "highlight velocity");

    const drawer = await screen.findByRole("dialog");
    expect(await within(drawer).findByText(/not in any of these documents/i)).toBeInTheDocument();
  });
});