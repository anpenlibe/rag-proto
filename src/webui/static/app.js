"use strict";
/* rag-proto console — vanilla-JS trace inspector over runs/.
   No framework, no build. Corpus text and model answers are rendered as text nodes
   (never innerHTML) — the only untrusted strings in the payload. */

// -- tiny DOM factory (text-safe) -----------------------------------------------------
const $ = (s, r = document) => r.querySelector(s);
function h(tag, props, ...kids) {
  const e = document.createElement(tag);
  if (props) for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v;
    else if (k === "dataset") Object.assign(e.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2).toLowerCase(), v);
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return e;
}
const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); };

// -- formatters -----------------------------------------------------------------------
const fmtMs = (ms) => ms == null ? "—" : ms < 1 ? "0ms" : ms < 1000 ? `${Math.round(ms)}ms`
  : `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)}s`;
const fmtTok = (n) => (n || 0).toLocaleString("en-US");
const pct = (x) => (x == null || Number.isNaN(x)) ? "—" : `${(x * 100).toFixed(1)}%`;
const num = (x, d = 3) => (x == null) ? "—" : Number(x).toFixed(d);
const shortDate = (iso) => (iso || "").replace("T", " ").slice(0, 16);
function pageHue(id) { let x = 0; for (const c of String(id)) x = (x * 31 + c.charCodeAt(0)) % 360; return x; }
const pageColor = (id) => `hsl(${pageHue(id)}, 52%, 55%)`;

// -- state ----------------------------------------------------------------------------
const state = { status: null, runs: [], run: null, evalData: null, view: "welcome",
                activeRun: null, activeTrace: null };

async function api(path, opts) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({ error: `${r.status} ${r.statusText}` }));
  if (!r.ok) throw Object.assign(new Error(data.error || r.statusText), { status: r.status, data });
  return data;
}

// ====================================================================================
// boot + chrome
// ====================================================================================
async function boot() {
  initTheme();
  $("#btn-theme").addEventListener("click", toggleTheme);
  $("#btn-refresh").addEventListener("click", () => loadRuns());
  $("#btn-ask").addEventListener("click", showAsk);
  $("#btn-eval").addEventListener("click", showEvalModal);
  try {
    state.status = await api("/api/status");
  } catch (e) { state.status = { error: String(e.message) }; }
  renderChips();
  await loadRuns();
  renderWelcome();
}

function renderChips() {
  const box = $("#chips");
  clear(box);
  const s = state.status || {};
  const q = s.qdrant || {};
  const spent = s.budget?.spent || 0;
  const dayBudget = s.budget?.day_budget_estimate || 400000;
  const budgetCls = spent >= dayBudget * 0.8 ? "warn" : "";
  box.append(
    chip(q.mode === "server" ? (q.reachable ? "ok" : "warn") : "",
         "qdrant", q.mode === "server" ? (q.reachable ? "server" : "down") : "local"),
    chip(s.has_keys ? "ok" : "warn", "keys", s.has_keys ? "ready" : "none"),
    chip(budgetCls, "tokens today", `${fmtTok(spent)} / ~${Math.round(dayBudget / 1000)}k`),
  );
}
const chip = (cls, label, val) =>
  h("span", { class: `chip ${cls}` }, h("span", { class: "dot" }), `${label} `, h("b", null, val));

function initTheme() {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.dataset.theme = saved;
}
function toggleTheme() {
  const cur = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("theme", next);
}

// ====================================================================================
// runs sidebar
// ====================================================================================
async function loadRuns() {
  try {
    const data = await api("/api/runs");
    state.runs = data.runs || [];
  } catch (e) { state.runs = []; }
  renderRunList();
}

function renderRunList() {
  const box = $("#run-list");
  clear(box);
  if (!state.runs.length) { box.append(h("div", { class: "empty" }, "No runs yet.")); return; }
  for (const r of state.runs) box.append(runRow(r));
}

// An ad-hoc run gets a hover-revealed delete ✕. It's a *sibling* of the card, not a child:
// .run-card is itself a <button>, and a button nested in a button is invalid HTML.
function runRow(r) {
  const card = runCard(r);
  if (r.kind !== "adhoc") return card;
  const del = h("button", {
    class: "rc-del", title: "Delete this ad-hoc run", "aria-label": `Delete run ${r.run_id.slice(-9)}`,
    onClick: (e) => { e.stopPropagation(); deleteRun(r.run_id); },
  }, "✕");
  return h("div", { class: "run-row" }, card, del);
}

function runCard(r) {
  const cons = r.eval_score?.consistency;
  const card = h("button", {
    class: `run-card ${state.activeRun === r.run_id ? "active" : ""}`,
    onClick: () => selectRun(r.run_id),
  },
    h("div", { class: "rc-top" },
      h("span", { class: "rc-id" }, r.run_id.slice(-9)),
      h("span", { class: `badge ${r.kind || "batch"}` }, r.kind || "run")),
    r.label && h("div", { class: "rc-label" }, r.label),
    h("div", { class: "rc-meta" },
      h("span", null, h("b", null, r.n_queries), " q"),
      r.total_tokens ? h("span", null, h("b", null, fmtTok(r.total_tokens)), " tok") : null,
      cons != null ? h("span", null, "cons ", h("b", null, pct(cons))) : null,
      r.status === "open" ? h("span", { class: "badge status-open" }, "open") : null),
  );
  return card;
}

// ====================================================================================
// welcome
// ====================================================================================
function renderWelcome() {
  state.view = "welcome"; state.activeRun = null; state.activeTrace = null;
  renderRunList();
  const v = $("#view");
  clear(v);
  const cfg = state.status?.config || {};
  v.append(
    h("div", { class: "section-title" },
      h("h1", null, "Trace console"),
      h("span", { class: "hash" }, "config ", h("b", null, cfg.config_hash || "—"))),
    h("p", { class: "sub", style: "max-width:66ch;margin-bottom:22px" },
      "Every hop of the RAG flow is logged to disk. Pick a run to replay its queries " +
      "hop-by-hop — retrieved pool → selected → exact prompt → answer → resolved citations — " +
      "or ask a live question. Replaying needs nothing but the filesystem."),
    h("div", { class: "row" },
      infoCard("1 · Traceability", "Open any run, then any query. The pipeline rail walks you " +
        "through each stage with its latency; expand a stage for the full retrieved pool, the " +
        "exact prompt messages, and the raw model response."),
      infoCard("2 · Paraphrase robustness", "On a scored run, the divergence strip shows which " +
        "pages each rephrasing selected — the headline problem (consistency) drawn directly."),
      infoCard("3 · Citations", "Answers carry inline [n] markers that resolve to numbered " +
        "sources with their page, heading path and URL.")),
    h("div", { class: "notice", style: "margin-top:20px" },
      h("b", null, "Baseline E0 · "), `model ${cfg.model || "?"} · retrieve ${cfg.candidate_k}`,
      ` → select ${cfg.top_k} · `,
      state.runs.length ? h("a", { href: "#", onClick: (e) => { e.preventDefault(); selectRun(state.runs.find(r => r.total_tokens)?.run_id || state.runs[0].run_id); } },
        "open the demo run →") : "no runs on disk yet"),
  );
}
const infoCard = (title, body) => h("div", { class: "panel" },
  h("h2", null, title), h("div", { class: "sub" }, body));

// ====================================================================================
// run overview
// ====================================================================================
async function selectRun(runId) {
  state.activeRun = runId; state.activeTrace = null; state.view = "run";
  renderRunList();
  const v = $("#view");
  clear(v); v.append(h("div", { class: "empty" }, h("span", { class: "spin" }), " loading run…"));
  try {
    state.run = await api(`/api/runs/${runId}`);
    state.evalData = state.run.has_eval ? await api(`/api/runs/${runId}/eval`) : null;
  } catch (e) { clear(v); v.append(errNotice(e)); return; }
  renderRunOverview();
}

function renderRunOverview() {
  const v = $("#view"); clear(v);
  const m = state.run.manifest, cfg = m.config || {};
  v.append(
    h("div", { class: "section-title" },
      h("h1", null, "Run ", h("span", { class: "mono" }, m.run_id.slice(-9))),
      h("span", { class: `badge ${m.kind}` }, m.kind),
      m.kind === "adhoc" ? h("button", { class: "btn danger", style: "margin-left:auto;align-self:center",
        onClick: () => deleteRun(m.run_id) }, "Delete run") : null),
    h("div", { class: "crumbs" },
      h("span", { class: "hash" }, "config ", h("b", null, m.config_hash)),
      h("span", { class: "sep" }, "·"),
      h("span", { class: "hash" }, "index ", h("b", null, m.index_hash)),
      m.eval_hash ? h("span", { class: "sep" }, "·") : null,
      m.eval_hash ? h("span", { class: "hash" }, "eval ", h("b", null, m.eval_hash)) : null,
      h("span", { class: "sep" }, "·"), shortDate(m.created_at)),
    m.label && h("p", { class: "sub", style: "margin:-8px 0 18px" }, m.label),
    m.note ? h("div", { class: "notice" }, m.note) : null,
    runAggregates(m),
    state.evalData ? evalPanel(state.evalData) : null,
    state.evalData ? divergencePanel(state.evalData) : null,
    queryListPanel(state.run.queries),
  );
}

function runAggregates(m) {
  return h("div", { class: "panel" },
    h("h2", null, "Pipeline"),
    h("dl", { class: "kv", style: "margin-top:10px" },
      dt("model"), dd(m.config?.model || "—"),
      dt("retrieve → select"), dd(`${m.config?.candidate_k} → ${m.config?.top_k}`),
      dt("queries"), dd(String(m.n_queries)),
      dt("tokens"), dd(m.total_tokens ? `${fmtTok(m.total_tokens)} (${fmtTok(m.prompt_tokens)} in · ${fmtTok(m.completion_tokens)} out)` : "0 (retrieval-only)"),
      dt("avg latency"), dd(fmtMs(m.avg_latency_ms)),
    ));
}
const dt = (t) => h("dt", null, t);
const dd = (t) => h("dd", null, t);

function evalPanel(ev) {
  const p = ev.scores.panel, meta = ev.scores.meta || {};
  const tile = (k, val, opts = {}) => {
    const t = h("div", { class: `tile ${opts.head ? "head" : ""} ${opts.dim ? "dim" : ""}` },
      h("div", { class: "k" }, k), h("div", { class: "v" }, val));
    if (opts.bar != null) t.append(h("div", { class: "bar" }, h("i", { style: `width:${Math.max(0, Math.min(1, opts.bar)) * 100}%` })));
    return t;
  };
  const judged = meta.n_judged ? `${meta.n_judged}/${meta.n_judgeable}` : null;
  return h("div", { class: "panel" },
    h("h2", null, "Eval panel"),
    h("div", { class: "sub" }, `gold ${ev.scores.gold_set} · ${meta.n_groups_scored ?? "?"} groups · ${meta.n_retrieval_scored ?? "?"} retrieval-scored` +
      (judged ? ` · ${judged} judged (${ev.scores.judge_model})` : " · not judged")),
    h("div", { class: "tiles", style: "margin-top:14px" },
      tile("consistency", pct(p.consistency), { head: true, bar: p.consistency }),
      tile("recall@k", pct(p["recall@k"]), { bar: p["recall@k"] }),
      tile("recall@cand", pct(p["recall@cand"]), { bar: p["recall@cand"] }),
      tile("mrr", num(p.mrr), { bar: p.mrr }),
      p.answer_agreement != null ? tile("answer agree", pct(p.answer_agreement), { bar: p.answer_agreement }) : null,
      p.faithfulness != null ? tile("faithfulness", pct(p.faithfulness), { bar: p.faithfulness }) : null,
      p.citation_acc != null ? tile("citation acc", pct(p.citation_acc), { bar: p.citation_acc }) : null,
    ),
  );
}

// -- divergence strip (signature #2) --------------------------------------------------
function divergencePanel(ev) {
  const byGroup = new Map();
  for (const r of ev.per_query) {
    if (!byGroup.has(r.group_id)) byGroup.set(r.group_id, []);
    byGroup.get(r.group_id).push(r);
  }
  const consByGroup = new Map((ev.per_group || []).map((g) => [g.group_id, g.consistency]));
  const rows = [];
  for (const [gid, list] of [...byGroup.entries()].sort()) {
    list.sort((a, b) => (a.query_id || "").localeCompare(b.query_id || ""));
    rows.push(h("div", { class: "dv-row", title: `${gid} — how the 6 phrasings' selected page-sets compare` },
      h("div", { class: "dv-label" },
        h("span", null, gid),
        h("span", { class: "dv-c" }, pct(consByGroup.get(gid)))),
      h("div", { class: "dv-cells" }, list.map((r) => phrasingCell(r)))));
  }
  return h("details", { class: "panel", open: true },
    h("summary", { style: "cursor:pointer;list-style:none" },
      h("h2", { style: "display:inline" }, "Paraphrase divergence"),
      h("span", { class: "sub", style: "margin-left:10px" }, "each block = one phrasing; each square = a selected page. Same colour = same page.")),
    h("div", { class: "divergence", style: "margin-top:12px" }, rows),
    h("div", { class: "dv-legend" },
      h("span", null, h("span", { class: "dv-dot expected", style: "background:var(--faint)" }), "expected page reached prompt"),
      h("span", null, h("span", { class: "dv-dot miss" }), "expected page missing"),
      h("span", null, "column drift across phrasings = the churn behind consistency")),
  );
}

function phrasingCell(r) {
  const expected = new Set(r.expected_page_ids || []);
  const selected = r.selected_pages || [];
  const dots = selected.map((pid) =>
    h("span", {
      class: `dv-dot ${expected.has(pid) ? "expected" : ""}`,
      style: `background:${pageColor(pid)}`,
      title: pid + (expected.has(pid) ? " (expected)" : ""),
    }));
  for (const pid of expected) if (!selected.includes(pid))
    dots.push(h("span", { class: "dv-dot miss", title: `${pid} (expected, not selected)` }));
  const cell = h("button", { class: "dv-phr", title: `${r.query_id} · recall@k ${r["recall@k"]}`,
    onClick: () => r.trace_id && selectTrace(state.activeRun, r.trace_id) },
    ...dots);
  return cell;
}

// -- query list -----------------------------------------------------------------------
function queryListPanel(queries) {
  const wrap = h("div", { class: "panel" }, h("h2", null, `Queries (${queries.length})`));
  // Group by gold group id if the compact rows carry one; else flat.
  const evByTrace = new Map((state.evalData?.per_query || []).map((r) => [r.trace_id, r]));
  const groups = new Map();
  for (const q of queries) {
    const ev = evByTrace.get(q.trace_id);
    const gid = ev?.group_id || "—";
    if (!groups.has(gid)) groups.set(gid, []);
    groups.get(gid).push({ q, ev });
  }
  for (const [gid, items] of groups) {
    if (gid !== "—")
      wrap.append(h("div", { class: "qgroup-head" }, gid,
        h("span", { class: "muted" }, items[0].ev?.expected_page_ids?.length
          ? `expects ${items[0].ev.expected_page_ids.map((p) => p.slice(0, 6)).join(", ")}` : "")));
    for (const { q, ev } of items) wrap.append(queryRow(q, ev));
  }
  return wrap;
}

function queryRow(q, ev) {
  return h("button", { class: "qrow", onClick: () => selectTrace(state.activeRun, q.trace_id) },
    h("div", null,
      h("div", { class: "q-q" },
        ev ? h("span", { class: `qkind ${ev.kind}` }, ev.kind) : null, " ", q.query),
      q.answer ? h("div", { class: "q-a" }, q.answer) : h("div", { class: "q-a" }, "· retrieval-only")),
    h("div", { class: "q-meta" },
      ev ? h("div", null, `r@k ${ev["recall@k"]} · rr ${num(ev.rr, 2)}`) : null,
      h("div", null, `${q.n_citations || 0} cite · ${fmtMs(q.total_latency_ms)}`),
      q.total_tokens ? h("div", null, `${fmtTok(q.total_tokens)} tok`) : null),
  );
}

// ====================================================================================
// trace replay (signature #1: the pipeline rail)
// ====================================================================================
async function selectTrace(runId, traceId) {
  state.activeTrace = traceId; state.view = "trace";
  const v = $("#view"); clear(v);
  v.append(h("div", { class: "empty" }, h("span", { class: "spin" }), " loading trace…"));
  try {
    if (!state.run || state.run.manifest.run_id !== runId) {
      state.run = await api(`/api/runs/${runId}`);
      state.evalData = state.run.has_eval ? await api(`/api/runs/${runId}/eval`) : null;
    }
    const trace = await api(`/api/runs/${runId}/trace/${traceId}`);
    renderTrace(trace, { runId });
  } catch (e) { clear(v); v.append(errNotice(e)); }
}

function renderTrace(t, { runId, live } = {}) {
  const v = $("#view"); clear(v);
  const evRow = (state.evalData?.per_query || []).find((r) => r.trace_id === t.trace_id);
  const judge = evRow && (state.evalData?.judge_answers || []).find((r) => r.query_id === evRow.query_id);

  v.append(
    h("div", { class: "crumbs" },
      live ? h("span", null, "live result") : h("a", { href: "#", onClick: (e) => { e.preventDefault(); selectRun(runId); } }, `run ${String(runId).slice(-9)}`),
      h("span", { class: "sep" }, "›"),
      h("span", { class: "mono" }, t.trace_id),
      t.query_id ? h("span", { class: "sep" }, "·") : null,
      t.query_id ? h("span", { class: "hash" }, t.query_id) : null,
      h("span", { class: "sep" }, "·"),
      statusTag(t.status)),
    h("div", { class: "qtext" }, t.query),
    pipelineRail(t),
    t.status === "error" ? h("div", { class: "notice err" }, h("b", null, `${t.error_type}: `), t.error) : null,
    answerPanel(t),
    evRow ? goldPanel(evRow, judge) : null,
    stageDetails(t, evRow),
  );
}

const statusTag = (s) => h("span", {
  class: "hash", style: `color:var(--${s === "ok" ? "green" : s === "error" ? "red" : "muted"})`,
}, s);

function pipelineRail(t) {
  const st = t.stages || {};
  const nq = st.query_transform?.queries?.length ?? 1;
  const nret = st.retrieve?.retrieved?.length ?? 0;
  const nrr = st.rerank?.reranked?.length ?? nret;
  const nsel = st.select?.selected?.length ?? 0;
  const nsrc = st.assemble?.sources ? Object.keys(st.assemble.sources).length : nsel;
  const gen = st.generate;
  const stages = [
    { key: "query_transform", cls: "", name: "transform", count: nq, sub: nq === 1 ? "identity" : "multi-query", lat: st.query_transform?.latency_ms },
    { key: "retrieve", cls: "retrieve", name: "retrieve", count: nret, sub: "dense · bge", lat: st.retrieve?.latency_ms },
    { key: "rerank", cls: "", name: "rerank", count: nrr, sub: st.rerank?.enabled ? (st.rerank.method || "rerank") : "passthrough", lat: st.rerank?.latency_ms },
    { key: "select", cls: "", name: "select", count: nsel, sub: `top-${nsel}`, lat: st.select?.latency_ms },
    { key: "assemble", cls: "", name: "assemble", count: nsrc, sub: `${st.assemble?.context_chars ?? "?"} chars`, lat: st.assemble?.latency_ms },
    gen
      ? { key: "generate", cls: "gen", name: "generate", count: gen.usage?.completion_tokens ?? "✓", sub: gen.finish_reason || "", lat: gen.latency_ms, tokens: true }
      : { key: "generate", cls: "", name: "generate", count: "—", sub: "skipped", skipped: true },
  ];
  const rail = h("div", { class: "rail" });
  for (const s of stages) {
    rail.append(h("button", {
      class: `rail-stage ${s.cls} ${s.skipped ? "skipped" : ""}`,
      onClick: () => openStage(s.key),
    },
      s.lat != null && h("span", { class: "rs-lat" }, fmtMs(s.lat)),
      h("div", { class: "rs-name" }, s.name),
      h("div", { class: "rs-count" }, String(s.count), s.tokens ? h("small", null, " tok") : null),
      h("div", { class: "rs-sub" }, s.sub)));
  }
  return rail;
}

function openStage(key) {
  const el = document.getElementById(`stage-${key}`);
  if (!el) return;
  el.open = true;
  el.scrollIntoView({ behavior: "smooth", block: "start" });
  el.classList.add("active");
  setTimeout(() => el.classList.remove("active"), 1200);
}

// -- answer + citations ---------------------------------------------------------------
function answerPanel(t) {
  if (t.status === "retrieval_only")
    return h("div", { class: "panel" }, h("h2", null, "Answer"),
      h("div", { class: "sub" }, "Retrieval-only run — no answer was generated (0 tokens)."));
  const sources = t.stages?.assemble?.sources || {};
  const cited = new Set((t.citations || []).map((c) => c.n));
  const invalid = new Set(t.invalid_citations || []);
  const panel = h("div", { class: "panel" },
    h("h2", null, "Answer"),
    h("div", { class: "answer", style: "margin:8px 0 16px" }, ...renderAnswer(t.answer || "", cited, invalid)),
    h("div", { class: "eyebrow", style: "margin-bottom:8px" },
      `Sources (${Object.keys(sources).length} selected · ${cited.size} cited)`),
    h("div", { class: "sources" },
      Object.keys(sources).sort((a, b) => a - b).map((k) => sourceRow(sources[k], cited.has(+k)))),
  );
  return panel;
}

function renderAnswer(text, cited, invalid) {
  const parts = [];
  const re = /\[(\d+)\]/g; let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const n = +m[1];
    const bad = invalid.has(n) || !cited.has(n);
    parts.push(h("span", {
      class: `cite ${bad ? "bad" : ""}`, dataset: { n },
      title: bad ? "cites a source not in the resolved citation set" : `source [${n}]`,
      onClick: () => highlightSource(n),
    }, `[${n}]`));
    last = re.lastIndex;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

function highlightSource(n) {
  const el = document.getElementById(`src-${n}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("hl");
  setTimeout(() => el.classList.remove("hl"), 1400);
}

function sourceRow(s, isCited) {
  const path = (s.heading_path || []).join(" › ");
  return h("div", { class: `source ${isCited ? "hl" : ""}`, id: `src-${s.n}` },
    h("div", { class: "n" }, `[${s.n}]`),
    h("div", null,
      h("div", { class: "s-title" }, s.title || "(untitled)"),
      path ? h("div", { class: "s-path" }, path) : null,
      h("a", { class: "s-url", href: s.url, target: "_blank", rel: "noopener" }, s.url)),
    h("div", { class: "s-score" }, s.section || ""),
  );
}

// -- gold / judge panel (when the trace is part of a scored run) ----------------------
function goldPanel(evRow, judge) {
  return h("div", { class: "panel" },
    h("h2", null, "Gold & judge"),
    h("dl", { class: "kv", style: "margin-top:10px" },
      dt("expected page(s)"), dd((evRow.expected_page_ids || []).join(", ") || "—"),
      dt("selected page(s)"), dd((evRow.selected_pages || []).join(", ") || "—"),
      dt("recall@k · rr"), dd(`${evRow["recall@k"]} · ${num(evRow.rr, 3)}`),
      judge ? dt("faithfulness · citation") : null,
      judge ? dd(`${judge.faithfulness} · ${judge.citation_acc}`) : null,
    ),
    judge?.reason ? h("div", { class: "notice", style: "margin-top:10px" },
      h("b", null, "judge: "), judge.reason) : null,
  );
}

// -- stage detail sections ------------------------------------------------------------
function stageDetails(t, evRow) {
  const st = t.stages || {};
  const expected = new Set(evRow?.expected_page_ids || []);
  const selectedIds = new Set((st.select?.selected || []).map((c) => c.id));
  const wrap = h("div", null);

  wrap.append(stageBlock("retrieve", `retrieve · ${(st.retrieve?.retrieved || []).length} candidates`,
    (st.retrieve?.retrieved || []).map((c, i) => chunkCard(c, i, { selectedIds, expected })), st.retrieve?.latency_ms));

  if (st.rerank && st.rerank.enabled)
    wrap.append(stageBlock("rerank", `rerank · ${st.rerank.method}`,
      (st.rerank.reranked || []).map((c, i) => chunkCard(c, i, { selectedIds, expected })), st.rerank.latency_ms));

  wrap.append(stageBlock("select", `select · top-${(st.select?.selected || []).length}`,
    (st.select?.selected || []).map((c, i) => chunkCard(c, i, { selectedIds, expected, showText: true })), st.select?.latency_ms));

  wrap.append(stageBlock("assemble", `assemble · ${st.assemble?.context_chars ?? "?"} chars`,
    [h("div", { class: "eyebrow", style: "margin-bottom:6px" }, "context sent to the model"),
     h("pre", { class: "code" }, st.assemble?.context || "")], st.assemble?.latency_ms));

  if (st.generate) {
    const g = st.generate;
    const kids = [];
    for (const msg of g.prompt_messages || []) {
      kids.push(h("div", { class: "msg-role" }, msg.role));
      kids.push(h("pre", { class: "code" }, msg.content));
    }
    kids.push(h("div", { class: "msg-role" }, "raw response"));
    kids.push(h("pre", { class: "code" }, g.raw_response || ""));
    kids.push(h("dl", { class: "kv", style: "margin-top:10px" },
      dt("model"), dd(g.model), dt("temperature"), dd(String(g.temperature)),
      dt("key_id"), dd(String(g.key_id)), dt("finish_reason"), dd(g.finish_reason || "—"),
      dt("usage"), dd(`${fmtTok(g.usage?.prompt_tokens)} in · ${fmtTok(g.usage?.completion_tokens)} out · ${fmtTok(g.usage?.total_tokens)} total`),
      (g.rate_limit_events || []).length ? dt("rate-limit rotations") : null,
      (g.rate_limit_events || []).length ? dd(JSON.stringify(g.rate_limit_events)) : null,
    ));
    wrap.append(stageBlock("generate", `generate · ${g.model}`, kids, g.latency_ms));
  }
  return wrap;
}

function stageBlock(key, title, bodyNodes, lat) {
  return h("details", { class: "stage-detail", id: `stage-${key}` },
    h("summary", null,
      h("span", { class: "st-name" }, title),
      h("span", { class: "st-meta" }, fmtMs(lat))),
    h("div", { class: "stage-body" }, bodyNodes));
}

function chunkCard(c, i, { selectedIds, expected, showText } = {}) {
  const isSel = selectedIds?.has(c.id);
  const isExp = expected?.has(c.page_id);
  const card = h("div", { class: `chunk ${isSel ? "selected" : ""} ${isExp ? "expected" : ""}` },
    h("div", { class: "c-head" },
      h("span", { class: "c-rank" }, `#${i + 1}`),
      h("span", { class: "c-title" }, c.title || "(untitled)"),
      isExp ? h("span", { class: "hash", style: "color:var(--green)" }, "gold") : null,
      c.score != null ? h("span", { class: "c-score" }, num(c.score, 4)) : null),
    h("div", { class: "c-id" }, `${c.id} · ${c.section || ""}`),
  );
  const text = c.text || "";
  if (showText || text.length) {
    const body = h("div", { class: `c-text ${showText ? "open" : ""}` }, text);
    card.append(body);
    if (!showText && text.length > 240)
      card.append(h("button", { class: "c-more", onClick: (e) => { body.classList.toggle("open"); e.target.textContent = body.classList.contains("open") ? "show less" : "show more"; } }, "show more"));
  }
  return card;
}

// ====================================================================================
// ask (live ad-hoc query)
// ====================================================================================
function showAsk() {
  state.activeRun = null; state.activeTrace = null; state.view = "ask";
  renderRunList();
  const v = $("#view"); clear(v);
  const noKeys = !state.status?.has_keys;
  const ta = h("textarea", { placeholder: "Ask a question about studying at the University of Vienna…", "aria-label": "question" });
  const out = h("div", { style: "margin-top:20px" });
  const submit = h("button", { class: "btn primary", onClick: run }, "Ask →");

  async function run() {
    const query = ta.value.trim();
    if (!query) { ta.focus(); return; }
    submit.disabled = true;
    clear(out); out.append(h("div", { class: "empty" }, h("span", { class: "spin" }), " running the pipeline — retrieve → generate…"));
    try {
      const res = await api("/api/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query }) });
      state.run = null; state.evalData = null;
      await loadRuns();
      renderTrace(res.trace, { runId: res.run_id, live: true });
    } catch (e) {
      clear(out);
      const trace = e.data?.trace;
      out.append(h("div", { class: "notice err" }, h("b", null, "Query failed: "), e.message));
      if (trace) { out.append(pipelineRail(trace)); out.append(stageDetails(trace, null)); }
      submit.disabled = false;
    }
  }

  v.append(
    h("div", { class: "section-title" }, h("h1", null, "Ask a question")),
    noKeys ? h("div", { class: "notice err" }, "No Groq keys configured — live generation is disabled. Replay existing runs instead.") : null,
    h("div", { class: "panel ask-form" },
      ta,
      h("div", { class: "ask-row" }, submit,
        h("span", { class: "hint" }, "≈2k tokens · saved as an ad-hoc run")),
      h("div", { class: "chips-inline" },
        ...["How much is the tuition fee?", "What is the STEOP?", "How do I register for exams?", "When is the application deadline for a master programme?"]
          .map((ex) => h("button", { class: "ex", onClick: () => { ta.value = ex; ta.focus(); } }, ex)))),
    out,
  );
  if (noKeys) submit.disabled = true;
  ta.focus();
}

// ====================================================================================
// run eval (modal + job polling)
// ====================================================================================
function showEvalModal() {
  let mode = "retrieve_only";
  let goldSet = null;                     // chosen set name (filled once /api/gold-sets loads)
  let goldSets = [];                      // [{name, n_groups}]
  const root = $("#modal-root");
  const status = h("div", { class: "sub", style: "margin-top:6px;min-height:20px" });
  const startBtn = h("button", { class: "btn primary", onClick: start }, "Start");
  const optsBox = h("div", { class: "opts" });
  const setSelect = h("select", { class: "gold-select", "aria-label": "gold set",
    onChange: (e) => { goldSet = e.target.value; renderOpts(); } });

  const nGroups = () => (goldSets.find((s) => s.name === goldSet) || {}).n_groups || 10;

  function optRow(val, title, desc) {
    const input = h("input", { type: "radio", name: "evalmode", value: val, checked: val === mode });
    const el = h("label", { class: `opt ${val === mode ? "sel" : ""}` }, input,
      h("div", null, h("div", { class: "o-t" }, title), h("div", { class: "o-d" }, desc)));
    input.addEventListener("change", () => {
      mode = val;
      for (const o of optsBox.children) o.classList.toggle("sel", o.querySelector("input").checked);
    });
    return el;
  }
  function renderOpts() {
    const n = nGroups();
    clear(optsBox);
    optsBox.append(
      optRow("retrieve_only", "Retrieval panel",
        `${n}×6 = ${n * 6} queries · consistency / recall / mrr · 0 tokens. Safe.`),
      optRow("smoke", "Smoke (3 groups)",
        `Generates answers for 3 groups of ${goldSet || "the set"} · ~50k tokens · costs Groq budget.`));
  }

  async function start() {
    startBtn.disabled = true;
    const body = mode === "smoke" ? { mode, limit: 3, gold_set: goldSet } : { mode, gold_set: goldSet };
    clear(status); status.append(h("span", { class: "spin" }), " starting…");
    try {
      const res = await api("/api/eval", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      pollJob(res.job_id, status, startBtn);
    } catch (e) { clear(status); status.append(h("span", { style: "color:var(--red)" }, e.message)); startBtn.disabled = false; }
  }

  const modal = h("div", { class: "modal-back", onClick: (e) => { if (e.target === modal) close(); } },
    h("div", { class: "modal" },
      h("h3", null, "Run the eval set"),
      h("div", { class: "sub" }, "Score a gold set against the current pipeline."),
      h("label", { class: "gold-row" }, h("span", { class: "eyebrow" }, "gold set"), setSelect),
      optsBox, status,
      h("div", { class: "modal-actions" },
        h("button", { class: "btn ghost", onClick: close }, "Close"), startBtn)));
  function close() { clear(root); }
  clear(root); root.append(modal);
  renderOpts();

  api("/api/gold-sets").then((d) => {
    goldSets = d.gold_sets || [];
    goldSet = (goldSets.find((s) => s.name === "gold_v1_small") || goldSets[0] || {}).name || null;
    clear(setSelect);
    for (const s of goldSets)
      setSelect.append(h("option", { value: s.name, selected: s.name === goldSet }, `${s.name} · ${s.n_groups} groups`));
    renderOpts();
  }).catch(() => {});
}

function pollJob(jobId, statusEl, startBtn) {
  const tick = async () => {
    let job;
    try { job = await api(`/api/eval/jobs/${jobId}`); }
    catch (e) { clear(statusEl); statusEl.append("lost job: " + e.message); return; }
    if (job.status === "running") {
      clear(statusEl); statusEl.append(h("span", { class: "spin" }), " running… (does not block replay)");
      setTimeout(tick, 1500); return;
    }
    clear(statusEl);
    const newIds = job.result?.new_run_ids || [];
    if (job.status === "done" && newIds.length) {
      statusEl.append("done — ", h("a", { href: "#", onClick: (e) => { e.preventDefault(); clear($("#modal-root")); loadRuns().then(() => selectRun(newIds[newIds.length - 1])); } }, `open run ${newIds[newIds.length - 1].slice(-9)} →`));
      loadRuns();
    } else {
      statusEl.append(h("span", { style: "color:var(--red)" }, "failed: "),
        h("pre", { class: "code" }, (job.result?.stderr_tail || job.result?.error || "unknown error")));
    }
    startBtn.disabled = false;
  };
  tick();
}

// ====================================================================================
// delete an ad-hoc run (confirm modal + DELETE)
// ====================================================================================
function deleteRun(runId) {
  const root = $("#modal-root");
  const status = h("div", { class: "sub", style: "margin-top:8px;min-height:18px" });
  const delBtn = h("button", { class: "btn danger", onClick: confirmDelete }, "Delete");

  function close() { clear(root); }

  async function confirmDelete() {
    delBtn.disabled = true;
    clear(status); status.append(h("span", { class: "spin" }), " deleting…");
    try {
      await api(`/api/runs/${runId}`, { method: "DELETE" });
      const wasActive = state.activeRun === runId;
      close();
      await loadRuns();                                   // refresh the sidebar from disk
      if (wasActive) { state.run = null; state.evalData = null; renderWelcome(); }
    } catch (e) {
      delBtn.disabled = false;
      clear(status); status.append(h("span", { style: "color:var(--red)" }, e.message || String(e)));
    }
  }

  const modal = h("div", { class: "modal-back", onClick: (e) => { if (e.target === modal) close(); } },
    h("div", { class: "modal" },
      h("h3", null, "Delete ad-hoc run"),
      h("div", { class: "sub" }, "Remove ", h("span", { class: "mono" }, runId.slice(-9)),
        " and its trace folder from disk. This cannot be undone."),
      status,
      h("div", { class: "modal-actions" },
        h("button", { class: "btn ghost", onClick: close }, "Cancel"), delBtn)));
  clear(root); root.append(modal);
  delBtn.focus();
}

// -- misc -----------------------------------------------------------------------------
const errNotice = (e) => h("div", { class: "notice err" }, h("b", null, "Error: "), e.message || String(e));

boot();
