const $ = (id) => document.getElementById(id);
const MAX_TRANSCRIPT_ROWS = 200;

const pct = (score) => `${Math.round((score || 0) * 100)}%`;

// The active session is server state, not browser state. Every call carries it,
// so switching sessions in another tab cannot make this tab ask the wrong
// question of the wrong documents.
let sessionId = null;
let apiBase = "";

function url(path) {
  // `apiBase` is derived from the page origin unless the server overrides it,
  // which keeps the app working when it is served from a subpath.
  return `${apiBase || ""}${path}${path.includes("?") ? "&" : "?"}session_id=${encodeURIComponent(sessionId)}`;
}

async function loadSessions() {
  const res = await fetch(`${apiBase}/api/sessions`);
  const data = await res.json();
  const select = $("session-select");
  select.innerHTML = "";
  data.sessions.forEach((s) => {
    const option = document.createElement("option");
    option.value = s.id;
    option.textContent = s.name;
    select.appendChild(option);
  });
  return data.sessions;
}

async function openSession(id) {
  sessionId = id;
  const res = await fetch(`${apiBase}/api/sessions/${encodeURIComponent(id)}`);
  const data = await res.json();
  apiBase = data.api_base || "";
  $("session-select").value = id;

  // Replay the stored transcript instead of keeping it in this tab, so a
  // refresh or a different browser shows the same conversation.
  const box = $("messages");
  box.innerHTML = "";
  (data.history || []).forEach((turn) => {
    addMessage(turn.role === "assistant" ? "bot" : "user", turn.content);
  });
  await refresh();
}

async function refresh() {
  const data = await (await fetch(url("/api/status"))).json();
  const list = $("sources");
  list.innerHTML = "";
  if (!data.sources.length) {
    list.innerHTML = '<li class="empty">No sources yet</li>';
  }
  data.sources.forEach((s) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="name" title="${s.name}">${s.name}</span><button class="x">&times;</button>`;
    li.querySelector(".x").onclick = async () => {
      await fetch(url(`/api/sources/${s.id}`), { method: "DELETE" });
      refresh();
    };
    list.appendChild(li);
  });
  const numeric = data.sources.reduce((n, s) => n + (s.numeric || 0), 0);
  $("meta").textContent =
    `${data.sources.length} sources · ${data.chunks} chunks\n` +
    `LLM: ${data.model} (${data.provider})\n` +
    `Embeddings: ${data.embed_model}\n` +
    `Relevance floor: ${pct(data.min_score)}` +
    (numeric ? `\n${numeric} numeric chunks (damped in mixed queries)` : "");
  $("meta").style.whiteSpace = "pre-line";
}

function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  if (role === "bot") wrap.innerHTML = '<div class="who">Assistant</div>';
  else wrap.innerHTML = '<div class="who">You</div>';
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;
  wrap.appendChild(body);
  $("messages").appendChild(wrap);
  $("messages").scrollTop = $("messages").scrollHeight;
  return wrap;
}

function addCitations(container, citations, cited) {
  if (!citations || !citations.length) return null;
  const box = document.createElement("div");
  box.className = "cites";
  const cards = [];
  const used = new Set(cited || []);

  citations.forEach((c, i) => {
    const div = document.createElement("div");
    div.className = "cite";
    div.id = `cite-${Date.now()}-${i}`;
    // Retrieved but never actually cited by the model - keep it visible, but
    // make clear it did not back the answer.
    if (used.size && !used.has(i + 1)) div.classList.add("uncited");
    div.dataset.index = String(i + 1);

    const head = document.createElement("div");
    head.className = "cite-head";

    const caret = document.createElement("span");
    caret.className = "caret";
    caret.textContent = "▸";

    const label = document.createElement("span");
    label.className = "src";
    label.textContent = `[${i + 1}] ${c.source}${c.page ? " p." + c.page : ""}`;

    const score = document.createElement("span");
    score.className = "score";
    score.textContent = pct(c.score) + " match";

    head.append(caret, label, score);

    const snippet = document.createElement("pre");
    snippet.className = "snippet";
    snippet.textContent = c.text;

    div.append(head, snippet);
    div.addEventListener("click", () => div.classList.toggle("open"));

    box.appendChild(div);
    cards.push(div);
  });

  container.appendChild(box);
  return cards;
}

function addEvidence(wrap, evidence) {
  if (!evidence) return;
  const bar = document.createElement("div");
  bar.className = "evidence";

  if (evidence.verdict === "no_match") {
    bar.classList.add("nomatch");
    bar.textContent =
      `No relevant passage found - best match ${pct(evidence.best_score)}, ` +
      `below the ${pct(evidence.min_score)} floor. The model was not asked, so nothing was invented.`;
    wrap.appendChild(bar);
    return;
  }

  const chip = document.createElement("span");
  chip.className = "chip conf-" + evidence.confidence;
  chip.textContent = evidence.confidence + " confidence";
  bar.appendChild(chip);

  const detail = document.createElement("span");
  detail.className = "detail";
  detail.textContent =
    `best match ${pct(evidence.best_score)} · ` +
    `${evidence.returned} of ${evidence.considered} candidates used`;
  bar.appendChild(detail);

  if (evidence.invalid && evidence.invalid.length) {
    const warn = document.createElement("span");
    warn.className = "chip warn";
    warn.textContent =
      "removed citation " + evidence.invalid.map((n) => `[${n}]`).join(", ");
    bar.appendChild(warn);
  }

  wrap.appendChild(bar);
}

function linkifyCitations(bodyEl, cards) {
  if (!cards || !cards.length) return;
  const raw = bodyEl.textContent;
  const re = /\[(\d{1,3})\]/g;
  let last = 0;
  let match;
  bodyEl.textContent = "";

  while ((match = re.exec(raw)) !== null) {
    const index = parseInt(match[1], 10) - 1;
    if (index < 0 || index >= cards.length) continue;
    bodyEl.appendChild(document.createTextNode(raw.slice(last, match.index)));

    const link = document.createElement("a");
    link.className = "ref";
    link.textContent = match[0];
    link.href = "#" + cards[index].id;
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const card = cards[index];
      card.classList.add("open");
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      card.classList.remove("flash");
      void card.offsetWidth;
      card.classList.add("flash");
    });
    bodyEl.appendChild(link);
    last = match.index + match[0].length;
  }
  bodyEl.appendChild(document.createTextNode(raw.slice(last)));
}

$("file").onchange = async (e) => {
  const files = e.target.files;
  if (!files.length) return;
  addMessage("bot", `Indexing ${files.length} file(s)...`);
  for (const file of files) {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch(url("/api/sources"), { method: "POST", body: fd });
    const out = await res.json();
    if (!res.ok) addMessage("bot", `Failed: ${out.detail}`);
  }
  refresh();
  e.target.value = "";
};

const drop = $("drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  const files = e.dataTransfer.files;
  if (files.length) { $("file").files = files; $("file").onchange({ target: $("file") }); }
};

$("clear").onclick = async () => {
  await fetch(url("/api/sources"), { method: "DELETE" });
  refresh();
};

$("session-select").onchange = async (e) => {
  await openSession(e.target.value);
};

$("session-new").onclick = async () => {
  const name = prompt("Name this session", "New session");
  if (name === null) return;
  const res = await fetch(`${apiBase}/api/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  const data = await res.json();
  await loadSessions();
  await openSession(data.id);
};

$("session-delete").onclick = async () => {
  if (!confirm("Delete this session, its sources and its chat history?")) return;
  await fetch(`${apiBase}/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
  const sessions = await loadSessions();
  if (!sessions.length) {
    addMessage("bot", "All sessions deleted.");
    return;
  }
  await openSession(sessions[0].id);
};

$("form").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (!q) return;
  $("q").value = "";
  $("q").style.height = "auto";
  addMessage("user", q);
  const pending = addMessage("bot", "Thinking...");

  const res = await fetch(url("/api/ask"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // No history in the body: the server owns the transcript, so a question
    // cannot be answered against a conversation this tab made up.
    body: JSON.stringify({ question: q }),
  });
  const out = await res.json();
  pending.innerHTML = '<div class="who">Assistant</div>';
  const body = document.createElement("div");
  body.className = "body" + (res.ok ? "" : " err");
  body.textContent = res.ok ? out.answer : out.detail;
  pending.appendChild(body);
  if (res.ok) {
    addEvidence(pending, out.evidence);
    const cards = addCitations(pending, out.citations, out.evidence && out.evidence.cited);
    linkifyCitations(body, cards);
  }
  $("messages").scrollTop = $("messages").scrollHeight;
};

$("q").addEventListener("input", (e) => {
  e.target.style.height = "auto";
  e.target.style.height = Math.min(e.target.scrollHeight, 180) + "px";
});
$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("form").requestSubmit(); }
});

(async function boot() {
  const sessions = await loadSessions();
  if (sessions.length) {
    await openSession(sessions[0].id);
  } else {
    const res = await fetch(`${apiBase}/api/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: "My session" }),
    });
    const data = await res.json();
    await loadSessions();
    await openSession(data.id);
  }
})();