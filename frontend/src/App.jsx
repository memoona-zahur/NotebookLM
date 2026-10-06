import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, setApiBase } from "./api.js";
import { readIntent } from "./highlightIntent.js";
import { suggestionsFor } from "./suggestions.js";

/**
 * Offered by "show me other questions".
 *
 * These are generic on purpose, unlike `suggestionsFor`, which derives its
 * questions from the source names actually in the notebook. Every one is
 * phrased so that a corpus with no answer to it produces the ordinary refusal
 * rather than something stranger - "no relevant passage" is a correct response
 * to "Are there any contradictions between these documents?", and pretending
 * otherwise would break the promise the rest of the interface makes.
 */
const MORE_QUESTIONS = [
  "What are the key points across these sources?",
  "What terminology or jargon does this define?",
  "Are there any contradictions or disagreements between these sources?",
  "What questions do these sources leave unanswered?",
  "Which of these sources would I start with?",
];
import { Rail } from "./Rail.jsx";
import { SourcesDrawer } from "./SourcesDrawer.jsx";
import { Thread } from "./Thread.jsx";
import { Composer } from "./Composer.jsx";

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [status, setStatus] = useState(null);
  const [reindexing, setReindexing] = useState(null);
  const [turns, setTurns] = useState([]);
  // Which session is waiting on an answer. Null when nothing is in flight.
  // Tracking the id rather than a boolean matters: a plain `pending` flag is
  // global, so switching sessions mid-request would either leave the new
  // session showing "Reading your sources" forever (if the stale response is
  // ignored) or flash it in the wrong thread. With the id, busy is only true
  // for the session that actually asked something.
  const [pendingFor, setPendingFor] = useState(null);
  const pending = pendingFor === activeId;
  const [uploading, setUploading] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  // The word a chat command asked to see, which pre-fills and runs the same
  // search box the Sources drawer offers by hand. Carries an id as well as the
  // word so asking for the same word twice searches twice.
  const [highlight, setHighlight] = useState(null);
  const commands = useRef(0);
  const [toasts, setToasts] = useState([]);
  // Extra questions, revealed by the "show me other questions" control in the
  // empty state. Kept out of `suggestions` so that replacing the derived list
  // with a wider one is a separate piece of state rather than a mutation of a
  // value that is recomputed on every status refresh.
  const [extraQuestions, setExtraQuestions] = useState([]);
  // Web search is the only action that waits on the open internet - the search
  // itself takes 8-15s, because it is a reasoning model with a browser tool. It
  // gets its own flag rather than reusing `uploading`, so a failed search cannot
  // clear a half-finished upload, and so the drawer can say which one it is
  // waiting for.
  const [searching, setSearching] = useState(false);

  // Guards against a slow response for a session the user has already left
  // writing itself into the thread.
  const sessionRef = useRef(null);
  sessionRef.current = activeId;

  const toast = useCallback((message, kind = "error") => {
    const id = `${Date.now()}-${Math.random()}`;
    setToasts((all) => [...all, { id, message, kind }]);
    setTimeout(() => setToasts((all) => all.filter((t) => t.id !== id)), 6000);
  }, []);

  const loadSessions = useCallback(async () => {
    const data = await api.listSessions();
    setSessions(data.sessions);
    return data.sessions;
  }, []);

  const openSession = useCallback(async (id) => {
    const data = await api.getSession(id);
    setApiBase(data.api_base);
    setActiveId(id);
    setTurns(data.history);
    setStatus(await api.status(id));
  }, []);

  // Boot: load the session list, then open one. The server always has at least
  // the default session, so there is always something to open.
  useEffect(() => {
    (async () => {
      try {
        const list = await loadSessions();
        if (list.length) await openSession(list[0].id);
        else {
          const created = await api.createSession("My session");
          await loadSessions();
          await openSession(created.id);
        }
      } catch (err) {
        toast(err.message);
      }
    })();
  }, [loadSessions, openSession, toast]);

  async function refreshStatus(id) {
    const current = id || sessionRef.current;
    if (!current) return;
    try {
      setStatus(await api.status(current));
    } catch (err) {
      toast(err.message);
    }
  }

  async function selectSession(id) {
    try {
      await openSession(id);
    } catch (err) {
      toast(err.message);
    }
  }

  async function createSession() {
    const name = window.prompt("Name this session", "New session");
    if (name === null) return;
    try {
      const created = await api.createSession(name.trim() || "Untitled session");
      await loadSessions();
      await openSession(created.id);
    } catch (err) {
      toast(err.message);
    }
  }

  async function deleteSession(session) {
    if (
      !window.confirm(
        `Delete "${session.name}"? Its sources and chat history go with it.`,
      )
    )
      return;
    try {
      await api.deleteSession(session.id);
      const list = await loadSessions();
      if (list.length) await openSession(list[0].id);
      else {
        const created = await api.createSession("My session");
        await loadSessions();
        await openSession(created.id);
      }
    } catch (err) {
      toast(err.message);
    }
  }

  async function upload(file) {
    const id = sessionRef.current;
    if (!id) return;
    setUploading(true);
    try {
      await api.upload(id, file);
      await refreshStatus(id);
    } catch (err) {
      toast(`${file.name}: ${err.message}`);
    } finally {
      setUploading(false);
    }
  }

  async function webSearch(query) {
    const id = sessionRef.current;
    if (!id) return;
    const trimmed = (query || "").trim();
    if (!trimmed) return;
    setSearching(true);
    setSourcesOpen(true);
    try {
      const data = await api.webSearch(id, trimmed, status?.web_search?.max_pages);
      await refreshStatus(id);
      const { added_count: added, failed } = data.web_search;
      if (added) {
        toast(
          `Added ${added} source${added === 1 ? "" : "s"} from the web.`,
          "ok",
        );
      }
      // Reported even when something was added. A search that indexed three
      // pages and could not read two is worth telling the user about, and
      // swallowing the reasons would leave them thinking it found nothing.
      for (const miss of failed || []) {
        toast(`${miss.url}: ${miss.reason}`, "error");
      }
      if (!added && !(failed || []).length) {
        toast(`No pages found for "${trimmed}". Try different words.`, "error");
      }
    } catch (err) {
      toast(err.message);
    } finally {
      setSearching(false);
    }
  }

  async function deleteSource(source) {
    const id = sessionRef.current;
    if (!id || !window.confirm(`Remove "${source.name}"?`)) return;
    try {
      await api.deleteSource(id, source.id);
      await refreshStatus(id);
    } catch (err) {
      toast(err.message);
    }
  }

  // Re-indexing is not destructive and needs no confirmation: it rebuilds the
  // same source in place. It is only slow, so the button is disabled per source
  // rather than globally.
  async function reindexSource(source) {
    const id = sessionRef.current;
    if (!id) return;
    setReindexing(source.id);
    try {
      await api.reindexSource(id, source.id);
      await refreshStatus(id);
    } catch (err) {
      toast(err.message);
    } finally {
      setReindexing(null);
    }
  }

  async function clearSources() {
    const id = sessionRef.current;
    if (!id || !window.confirm("Remove every source from this session?")) return;
    try {
      await api.clearSources(id);
      await refreshStatus(id);
    } catch (err) {
      toast(err.message);
    }
  }

  // Where a word appears in this session's sources. Errors are thrown rather
  // than toasted so the highlight box can show the message next to the word
  // that could not be found, which is where the user is looking.
  async function findOccurrences(term) {
    const id = sessionRef.current;
    if (!id) throw new Error("No session is open.");
    return api.occurrences(id, term);
  }

  async function send(question) {
    const id = sessionRef.current;
    if (!id) return;

    // A message can carry two instructions: what to ask, and which words to
    // highlight. Only the question is retrieved on or shown to the model, so
    // "highlight 30 days" does not dilute the search.
    const { question: asked, term } = readIntent(question);

    // The highlight is a lookup over the index, so it runs whether or not there
    // is a question to answer, and before the model call so the passages are on
    // screen while it is still thinking.
    if (term) {
      commands.current += 1;
      setHighlight({ term, id: commands.current });
      setSourcesOpen(true);
    }

    // Nothing to answer: a bare "highlight velocity" is a lookup, and calling the
    // model on it produces prose about the word instead of showing where it is.
    // The turn is still recorded, because the user did type it.
    if (!asked) {
      setTurns((all) => [...all, { role: "user", text: question }]);
      return;
    }

    // Show the question the user typed, not the stripped version, so the thread
    // matches what they sent. Only the model receives the stripped question.
    // Added once whether or not there was also a highlight; appending per branch
    // would show the message twice.
    setTurns((all) => [...all, { role: "user", text: question }]);
    setPendingFor(id);

    try {
      const answer = await api.ask(id, asked);
      if (sessionRef.current !== id) return; // the user switched sessions
      setTurns((all) => [
        ...all,
        {
          role: "assistant",
          text: answer.answer,
          citations: answer.citations,
          evidence: answer.evidence,
        },
      ]);
      refreshStatus(id);
    } catch (err) {
      if (sessionRef.current !== id) return;
      setTurns((all) => [...all, { role: "assistant", text: err.message, error: true }]);
    } finally {
      // Clear only our own flag: a newer request in this same session may have
      // already started, and clearing unconditionally would unsuspend the
      // composer under it.
      setPendingFor((current) => (current === id ? null : current));
    }
  }

  const active = sessions.find((s) => s.id === activeId);

  // "5 Oct 2026", matching how the reference product dates a notebook. Guarded
  // because an unparseable date should cost the date, not the empty state.
  const createdAt = (() => {
    if (!active?.created_at) return "";
    const when = new Date(active.created_at);
    return Number.isNaN(when.getTime())
      ? ""
      : when.toLocaleDateString(undefined, {
          day: "numeric",
          month: "short",
          year: "numeric",
        });
  })();

  return (
    <div className="app">
      <Rail
        sessions={sessions}
        activeId={activeId}
        onSelect={selectSession}
        onCreate={createSession}
        onDelete={deleteSession}
      />

      <main className="main">
        <header className="topbar">
          <h1>{active ? active.name : "NotebookLM"}</h1>
          <div className="spacer" />
          <button
            className="sources-toggle"
            onClick={() => setSourcesOpen(true)}
            aria-haspopup="dialog"
          >
            <span>
              {status ? status.sources.length : 0} source
              {status && status.sources.length === 1 ? "" : "s"}
            </span>
          </button>
          {status ? (
            <div className="topbar-meta">
              <span className="pill live">{status.provider}</span>
            </div>
          ) : null}
        </header>

        {/* Starter questions when the thread is empty. `send` is passed
            directly rather than wrapped: a suggestion is a question, and it must
            take the same path as a typed one - same intent parsing, same
            highlight handling - or the empty state would answer differently
            from the composer. */}
        <Thread
          turns={turns}
          pending={pending}
          onSuggest={status ? send : null}
          suggestions={
            status
              ? [...suggestionsFor(status.sources), ...extraQuestions]
              : []
          }
          // Hidden once revealed, so it cannot be clicked into an empty list.
          onExplore={
            status &&
            suggestionsFor(status.sources).length &&
            !extraQuestions.length
              ? () => setExtraQuestions(MORE_QUESTIONS)
              : null
          }
          onUpload={() => setSourcesOpen(true)}
          onWebSearch={
            status?.web_search?.available
              ? () => {
                  const query = window.prompt(
                    "What should the web pages be about?",
                    "",
                  );
                  if (query) webSearch(query);
                }
              : null
          }
          webSearchReason={status?.web_search?.reason || "Not available in this local build."}
          sourceCount={status ? status.sources.length : 0}
          notebookName={active ? active.name : ""}
          createdAt={createdAt}
        />
        <Composer onSend={send} busy={pending} />
      </main>

      {sourcesOpen ? (
        <SourcesDrawer
          status={status}
          uploading={uploading}
          onClose={() => {
            setSourcesOpen(false);
            // Disarm the command, or reopening the drawer would re-run the last
            // search without being asked.
            setHighlight(null);
          }}
          onUpload={upload}
          onWebSearch={status?.web_search?.available ? webSearch : null}
          searching={searching}
          onDeleteSource={deleteSource}
          onReindexSource={reindexSource}
          reindexingId={reindexing}
          onClearSources={clearSources}
          onFindOccurrences={findOccurrences}
          autoSearch={highlight}
        />
      ) : null}

      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind}`}>
            {t.message}
          </div>
        ))}
      </div>
    </div>
  );
}