import React, { useCallback, useEffect, useRef, useState } from "react";
import { api, setApiBase } from "./api.js";
import { Rail } from "./Rail.jsx";
import { Thread } from "./Thread.jsx";
import { Composer } from "./Composer.jsx";

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [status, setStatus] = useState(null);
  const [turns, setTurns] = useState([]);
  const [pending, setPending] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [toasts, setToasts] = useState([]);

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

  async function send(question) {
    const id = sessionRef.current;
    if (!id) return;

    // Show the question immediately; the server keeps the real transcript, so
    // this local copy is purely optimistic and gets replaced on the next load.
    setTurns((all) => [...all, { role: "user", text: question }]);
    setPending(true);

    try {
      const answer = await api.ask(id, question);
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
      if (sessionRef.current === id) setPending(false);
    }
  }

  const active = sessions.find((s) => s.id === activeId);

  return (
    <div className="app">
      <Rail
        status={status}
        sessions={sessions}
        activeId={activeId}
        onSelect={selectSession}
        onCreate={createSession}
        onDelete={deleteSession}
        onUpload={upload}
        onDeleteSource={deleteSource}
        onClearSources={clearSources}
        uploading={uploading}
      />

      <main className="main">
        <header className="topbar">
          <h1>{active ? active.name : "NotebookLM"}</h1>
          <div className="spacer" />
          {status ? (
            <div className="topbar-meta">
              <span className="pill">
                {status.sources.length} source{status.sources.length === 1 ? "" : "s"}
              </span>
              <span className="pill live">{status.provider}</span>
            </div>
          ) : null}
        </header>

        <Thread turns={turns} pending={pending} />
        <Composer onSend={send} busy={pending} />
      </main>

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