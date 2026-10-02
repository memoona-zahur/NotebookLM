import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * A scripted stand-in for the API.
 *
 * Deliberately intercepts `fetch` rather than the `api` module: the URL
 * building, the `detail` unwrapping and the 503 message are real logic worth
 * exercising, and mocking them away would test nothing.
 */

/** Pristine state. `reset` restores this, so a `seed` in one test cannot
 *  change what the next one sees. */
function freshSessions() {
  return {
    alpha: { id: "alpha", name: "Alpha", history: [], api_base: "", sources: [] },
    beta: { id: "beta", name: "Beta", history: [], api_base: "", sources: [] },
  };
}

let SESSIONS = freshSessions();

/**
 * Give a session a persisted transcript.
 *
 * Keys are `content`, not `text`: that is what `db.recent_messages` returns and
 * what the app actually receives. Seeding a friendlier shape here is what let a
 * real blank-page crash pass the suite.
 */
export function seed(id, history) {
  SESSIONS[id] = { ...SESSIONS[id], history };
}

let handlers = [];
let calls = [];

function json(body, status = 200) {
  return {
    ok: status < 400,
    status,
    text: async () => JSON.stringify(body),
  };
}

/** Register a handler for `METHOD /path`; the last registered wins. */
export function on(method, path, handler) {
  handlers.push({ method, path, handler });
}

export function reset() {
  handlers = [];
  calls = [];
  SESSIONS = freshSessions();
}

export function requests() {
  return calls;
}

function answer(method, path) {
  for (let i = handlers.length - 1; i >= 0; i -= 1) {
    const h = handlers[i];
    if (h.method === method && path.startsWith(h.path)) return h.handler;
  }
  return null;
}

beforeEach(() => {
  reset();

  vi.stubGlobal("fetch", async (input, init = {}) => {
    const url = new URL(String(input), "http://localhost");
    const method = init.method || "GET";
    const path = url.pathname;
    const sessionId = url.searchParams.get("session_id");

    calls.push({ method, path, sessionId, body: init.body });

    // A registered handler always wins, including over the defaults below —
    // otherwise a test could not script a failing /api/status.
    const handler = answer(method, path);
    if (handler) {
      const result = await handler({ sessionId, init, url });
      return result && result.__response ? result : json(result ?? {});
    }

    // Otherwise answer the boot calls with a plausible shape, so a test only
    // has to script the parts it cares about.
    if (path === "/api/sessions" && method === "GET") {
      return json({ sessions: Object.values(SESSIONS).map(({ id, name }) => ({ id, name })) });
    }
    if (path === "/api/status") {
      return json({
        provider: "groq",
        model: "openai/gpt-oss-120b",
        embed_model: "sentence-transformers/all-MiniLM-L6-v2",
        min_score: 0.25,
        sources: [],
        chunks: 0,
      });
    }
    const detail = path.match(/^\/api\/sessions\/([^/]+)$/);
    if (detail && method === "GET") {
      const stored = SESSIONS[decodeURIComponent(detail[1])];
      if (stored) return json(stored);
    }

    throw new Error(`no handler for ${method} ${path}`);
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  window.confirm = () => true;
});

/** Build a deferred that the test resolves by hand, to control ordering. */
export function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

export function response(body, status = 200) {
  return { __response: true, ...json(body, status) };
}