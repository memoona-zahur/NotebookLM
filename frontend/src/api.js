// Thin fetch wrapper. Every call returns parsed JSON or throws an Error whose
// message is the server's `detail`, so callers never have to dig through a
// response object to find out what went wrong.

let baseUrl = "";

export function setApiBase(url) {
  baseUrl = (url || "").replace(/\/+$/, "");
}

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path, { method = "GET", body, sessionId, signal } = {}) {
  const url = new URL(baseUrl + path, window.location.origin);
  if (sessionId) url.searchParams.set("session_id", sessionId);

  const init = { method, signal, headers: {} };
  if (body instanceof FormData) {
    init.body = body; // let the browser set the multipart boundary
  } else if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(url, init);
  } catch (cause) {
    if (cause.name === "AbortError") throw cause;
    throw new ApiError("Could not reach the server. Is it still running?", 0);
  }

  if (response.status === 204) return null;

  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      // A non-JSON body means something upstream answered instead of the app.
      throw new ApiError(text.slice(0, 200) || `HTTP ${response.status}`, response.status);
    }
  }

  if (!response.ok) {
    const detail =
      (payload && (payload.detail || payload.message)) ||
      `Request failed (${response.status})`;
    throw new ApiError(
      typeof detail === "string" ? detail : JSON.stringify(detail),
      response.status,
    );
  }
  return payload;
}

export const api = {
  listSessions: () => request("/api/sessions"),
  createSession: (name) => request("/api/sessions", { method: "POST", body: { name } }),
  getSession: (id) => request(`/api/sessions/${encodeURIComponent(id)}`),
  renameSession: (id, name) =>
    request(`/api/sessions/${encodeURIComponent(id)}`, { method: "PATCH", body: { name } }),
  deleteSession: (id) => request(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" }),
  clearHistory: (id) =>
    request(`/api/sessions/${encodeURIComponent(id)}/messages/clear`, { method: "POST" }),

  status: (sessionId) => request("/api/status", { sessionId }),
  upload: (sessionId, file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/sources", { method: "POST", body: form, sessionId });
  },
  // Searches the web and indexes what it finds. `limit` is capped server-side;
  // the value here is what the UI asks for, not what it is guaranteed.
  webSearch: (sessionId, query, limit) =>
    request("/api/sources/web", {
      method: "POST",
      body: { query, limit },
      sessionId,
    }),
  // Re-reads the original already on disk and rebuilds its chunks under the
  // current CHUNK_SIZE / EMBED_MODEL. The id survives, so citations in past
  // answers still point somewhere real.
  reindexSource: (sessionId, sourceId) =>
    request(`/api/sources/${encodeURIComponent(sourceId)}/reindex`, {
      method: "POST",
      sessionId,
    }),
  // A URL to open in a new tab rather than a fetch. The original document is
  // meant to be looked at, and fetching it would mean re-implementing the PDF
  // viewer, the text renderer and the download prompt the browser already has.
  sourceFileUrl: (sessionId, sourceId) => {
    const url = new URL(
      baseUrl + `/api/sources/${encodeURIComponent(sourceId)}/file`,
      window.location.origin
    );
    if (sessionId) url.searchParams.set("session_id", sessionId);
    return url.toString();
  },
  deleteSource: (sessionId, sourceId) =>
    request(`/api/sources/${encodeURIComponent(sourceId)}`, { method: "DELETE", sessionId }),
  clearSources: (sessionId) => request("/api/sources", { method: "DELETE", sessionId }),

  occurrences: (sessionId, term) =>
    request(`/api/occurrences?term=${encodeURIComponent(term || "")}`, { sessionId }),

  ask: (sessionId, question, signal) =>
    request("/api/ask", { method: "POST", body: { question }, sessionId, signal }),
  summarize: (sessionId, instruction, signal) =>
    request("/api/summarize", { method: "POST", body: { instruction }, sessionId, signal }),
};