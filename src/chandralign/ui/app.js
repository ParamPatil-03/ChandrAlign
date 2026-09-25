"use strict";
/**
 * CHANDRALIGN CONSOLE — app.js
 *
 * Every number, image and label this page shows comes from the backend (api.py): the curated
 * pairs and their committed evidence (GET /pairs), the failure-mode register
 * (GET /failure-modes), and a run's own files (GET /runs/{id}, /runs/{id}/assets/...).
 * Nothing is simulated and nothing falls back to canned results: when something fails, the
 * page says what failed. (Audit 2026-09-26 C-05: the previous version showed hard-coded tiers
 * and metrics under invented product IDs, and swapped them in on any error.)
 */

const S = {
  api: window.location.origin,
  up: false,
  pairs: [],
  modes: {},            // failure-mode id -> {name, mitigation}
  pair: null,
  tab: "cur",
  running: false,
  runId: null,
  windows: [],          // [{dir, confidence_tier, ...}]
  detail: null,         // the selected window's own result.json
  base: "",             // asset path prefix of the selected window ("" or "window_01/")
  viz: "swipe",
  swipeX: 0.5,
  dragging: false,
};

const TIER = {
  HIGH:     { chip: "tier-high",     ribbon: "ribbon-high",     icon: "●" },
  MEDIUM:   { chip: "tier-medium",   ribbon: "ribbon-medium",   icon: "●" },
  LOW:      { chip: "tier-low",      ribbon: "ribbon-low",      icon: "▲" },
  REJECTED: { chip: "tier-rejected", ribbon: "ribbon-rejected", icon: "✕" },
};

const $ = id => document.getElementById(id);
const esc = v => String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ─────────────────────────────────────────────────────────────────── init
document.addEventListener("DOMContentLoaded", async () => {
  initSwipeDrag();
  setViz("swipe");
  await connect();
});

async function connect() {
  try {
    const r = await fetch(S.api + "/health", { signal: AbortSignal.timeout(3000) });
    S.up = r.ok;
  } catch { S.up = false; }
  setStatus(S.up);
  if (!S.up) {
    $("pair-list").innerHTML = `<div class="pair-note" style="padding:10px;">The backend is not reachable at
      ${esc(S.api)}. Start it with <code>python -m chandralign.api</code> and reload. This console shows
      only real results, so there is nothing to show without it.</div>`;
    $("btn-run").disabled = true;
    $("pairs-badge").textContent = "API OFFLINE";
    log("err", "Backend not reachable: " + S.api);
    return;
  }
  try {
    const [pairs, modes] = await Promise.all([getJSON("/pairs"), getJSON("/failure-modes")]);
    S.pairs = pairs.pairs || [];
    for (const m of modes.failure_modes || []) S.modes[m.id] = m;
    buildPairList();
    log("info", `Connected to ${S.api}: ${S.pairs.length} curated pairs.`);
  } catch (e) {
    log("err", "Could not load pairs: " + e.message);
  }
}

async function getJSON(path) {
  const r = await fetch(S.api + path);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
  return body;
}

function setStatus(up) {
  $("sys-dot").style.background = up ? "#10b981" : "#ef4444";
  $("sys-dot").style.boxShadow  = up ? "0 0 6px #10b981" : "0 0 6px #ef4444";
  $("sys-txt").textContent = up ? "API CONNECTED" : "API OFFLINE";
}

// ─────────────────────────────────────────────────────────────────── pairs
function evidenceText(ev) {
  if (!ev || !ev.available) return "no committed evidence found";
  const tiers = ev.tiers ? " (" + Object.entries(ev.tiers).map(([t, n]) => `${n} ${t}`).join(", ") + ")" : "";
  const verdict = ev.verdict ? ` — ${ev.verdict}` : "";
  return `committed: ${ev.accepted}/${ev.windows} windows accepted${tiers}${verdict}`;
}

function buildPairList() {
  $("pairs-badge").textContent = `${S.pairs.filter(p => p.runnable).length}/${S.pairs.length} RUNNABLE`;
  $("pair-list").innerHTML = S.pairs.map(p => `
    <div class="pair-item ${p.runnable ? "" : "disabled"}" id="pi-${esc(p.pair_id)}" onclick="selectPair('${esc(p.pair_id)}')"
         title="${esc(p.runnable ? "" : "Cannot run here: " + p.why_not)}">
      <div class="pair-radio"></div>
      <div class="pair-info">
        <div class="pair-name">${esc(p.label)}</div>
        <div class="pair-meta">${esc(evidenceText(p.evidence))}</div>
        <div class="pair-note">${esc(p.note)}</div>
      </div>
      <div class="tier-chip ${p.runnable ? "tier-medium" : "tier-none"}">${p.runnable ? "RUNNABLE" : "EVIDENCE ONLY"}</div>
    </div>`).join("");
  const first = S.pairs.find(p => p.runnable) || S.pairs[0];
  if (first) selectPair(first.pair_id);
}

function selectPair(id) {
  if (S.running) return;
  S.pair = S.pairs.find(p => p.pair_id === id);
  document.querySelectorAll(".pair-item").forEach(el => el.classList.remove("selected"));
  const el = $("pi-" + id);
  if (el) el.classList.add("selected");
  $("ribbon-pair").textContent = `${S.pair.src_product} → ${S.pair.ref_product}`;
  if (!S.pair.runnable) log("warn", `${S.pair.label}: cannot run here (${S.pair.why_not}). Its committed evidence is shown.`);
}

function setTab(t) {
  S.tab = t;
  ["cur", "up"].forEach(x => $("tab-" + x).classList.toggle("active", x === t));
  $("panel-cur").style.display = t === "cur" ? "block" : "none";
  $("panel-up").style.display  = t === "up"  ? "block" : "none";
}

// ─────────────────────────────────────────────────────────────────── run
function requestBody() {
  const windows = Math.max(1, Math.min(5, parseInt($("in-windows").value, 10) || 1));
  const mock = $("tog-mock").checked;
  if (S.tab === "up") {
    if (mock) return { mock: true };
    const src = $("in-src").value.trim(), ref = $("in-ref").value.trim();
    if (!src || !ref) throw new Error("give both server label paths");
    return { src, ref, windows };
  }
  if (!S.pair) throw new Error("select a pair");
  return mock ? { pair_id: S.pair.pair_id, mock: true } : { pair_id: S.pair.pair_id, windows };
}

async function startRegistration() {
  if (S.running || !S.up) return;
  let body;
  try { body = requestBody(); } catch (e) { log("err", e.message); return; }
  S.running = true;
  setRunButton(true);
  resetResult();
  setStages(1, false);
  log("info", "POST /register " + JSON.stringify(body));
  try {
    const r = await fetch(S.api + "/register", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const reply = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(typeof reply.detail === "string" ? reply.detail : JSON.stringify(reply.detail || r.status));
    S.runId = reply.run_id;
    $("run-id-lbl").textContent = "RUN: " + S.runId.slice(0, 8).toUpperCase();
    await poll(S.runId);
  } catch (e) {
    showFailure(e.message);
  } finally {
    S.running = false;
    setRunButton(false);
  }
}

async function poll(runId) {
  let seen = 0;
  const deadline = Date.now() + 30 * 60 * 1000;          // a multi-window real run takes minutes
  while (Date.now() < deadline) {
    await sleep(1000);
    const rec = await getJSON(`/runs/${runId}`);
    const lines = rec.progress_log || [];
    for (let i = seen; i < lines.length; i++) { log("info", lines[i]); trackStage(lines[i]); }
    seen = lines.length;
    if (rec.status === "DONE") { setStages(6, true); await showRun(rec); return; }
    if (rec.status === "FAILED") throw new Error(rec.error || "the run failed");
  }
  throw new Error("still running after 30 minutes; the run continues on the server: GET /runs/" + runId);
}

// Stage pills follow the real progress log of workflows.products / the API worker.
function trackStage(line) {
  const m = /window (\d+)\/(\d+)/.exec(line);
  if (/overlap/.test(line)) setStages(2, false);
  if (m) {
    const k = +m[1], n = +m[2];
    setStages(3, false);
    setProgress(Math.round(100 * (k - 1) / n));
  }
  if (/Writing the run folder|Done/.test(line)) setStages(5, false);
}

function setStages(active, finished) {
  for (let i = 1; i <= 5; i++) {
    $("sp" + i).className = "stage-pill " + (finished || i < active ? "done" : i === active ? "active" : "");
  }
  if (finished) setProgress(100);
}

function setProgress(pct) {
  $("prog-fill").style.width = pct + "%";
  $("prog-pct").textContent = pct + "%";
}

function setRunButton(running) {
  $("btn-run").disabled = running;
  $("btn-run").classList.toggle("running", running);
  $("btn-icon").textContent = running ? "⏳" : "▶";
  $("btn-txt").textContent = running ? "RUNNING…" : "RUN REGISTRATION";
}

// ─────────────────────────────────────────────────────────────────── results
async function showRun(rec) {
  const res = rec.result || {};
  if (Array.isArray(res.window_results)) {               // a real product run
    S.windows = res.window_results;
    $("t-cascade").textContent = res.workflow || "—";
    log("ok", `Done: ${res.accepted}/${res.windows} windows accepted ${JSON.stringify(res.tiers)}`);
  } else {                                               // a synthetic (mock) run: one folder
    S.windows = [{ dir: "", confidence_tier: res.confidence_tier }];
    $("t-cascade").textContent = "synthetic pair (pipeline.register_bundle)";
    log("warn", `Done (SYNTHETIC): tier ${res.confidence_tier}`);
  }
  const sel = $("sel-window");
  sel.innerHTML = S.windows.map((w, i) =>
    `<option>${S.windows.length > 1 || w.dir ? `Window ${i + 1}` : "Run"} — ${esc(w.confidence_tier)}</option>`).join("");
  sel.disabled = S.windows.length < 2;
  await selectWindow(0);
}

async function selectWindow(i) {
  const w = S.windows[i];
  S.base = w.dir ? w.dir + "/" : "";
  try {
    S.detail = await getJSON(`/runs/${S.runId}/assets/${S.base}result.json`);
  } catch (e) {
    showFailure("could not read the window's result.json: " + e.message);
    return;
  }
  showDetail(S.detail);
}

function showDetail(d) {
  const tier = d.confidence_tier;
  const t = TIER[tier] || { ribbon: "ribbon-none", icon: "○" };
  const m = d.metrics || {};
  const synthetic = m.source === "synthetic";
  $("ribbon").className = "confidence-ribbon " + t.ribbon;
  $("ribbon-icon").textContent = t.icon;
  $("ribbon-txt").textContent = `CONFIDENCE TIER: ${tier}` + (synthetic ? " — SYNTHETIC DATA" : "");
  $("ribbon-pair").textContent = [d.pairing, windowPlace(d.window)].filter(Boolean).join(" · ") || "synthetic pair";

  // Rejection (rule H3): the reason and the canonical failure modes, by name
  const rej = tier === "REJECTED";
  $("rej-box").classList.toggle("show", rej);
  $("rej-reason").textContent = "";
  $("rej-modes").innerHTML = "";
  if (rej) {
    $("rej-reason").textContent = d.status || d.provenance?.limiting_signal || "see failure modes";
    $("rej-modes").innerHTML = (d.failure_modes || []).map(id => {
      const fm = S.modes[id];
      return `<span class="rej-pill" title="${esc(fm ? fm.mitigation : "")}">#${esc(id)} ${esc(fm ? fm.name : "unknown mode")}</span>`;
    }).join("");
  }

  const fmt = (v, dgt = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(dgt);
  const gap = d.stages?.uniformity?.max_delaunay_gap_px;
  $("m-rmse").textContent = fmt(m.rmse_px);
  $("m-rmse-m").textContent = "model fit residual";
  $("m-inliers").textContent = m.inlier_count ?? "—";
  $("m-ratio").textContent = "Ratio: " + (m.inlier_ratio != null ? fmt(100 * m.inlier_ratio, 1) + "%" : "—");
  $("m-cov").textContent = m.spatial_coverage != null ? fmt(100 * m.spatial_coverage, 1) : "—";
  $("m-gap").textContent = gap != null ? fmt(gap, 1) : "—";
  $("m-rt").textContent = fmt(m.runtime_s, 1);
  $("m-matcher").textContent = d.matcher || "—";
  document.querySelectorAll(".metric-src").forEach(el => {
    el.textContent = m.source ? m.source.toUpperCase() : "—";
    el.classList.toggle("synthetic", synthetic);
  });
  $("t-matcher").textContent = d.matcher || "—";
  $("t-regime").textContent = d.regime || "—";
  $("tb-matches").textContent = d.provenance?.evidence_matches ?? "—";
  $("tb-inliers").textContent = m.inlier_count ?? "—";
  $("vp-idle").style.display = "none";
  setViz(S.viz);
}

function windowPlace(w) {
  if (!w) return "";
  if (w.tmc_row != null) return `TMC-2 row ${w.tmc_row}`;
  if (w.ohrc_row != null) return `OHRC row ${w.ohrc_row} → ${w.nac || ""}`;
  return "";
}

function resetResult() {
  S.detail = null; S.windows = [];
  $("ribbon").className = "confidence-ribbon ribbon-none";
  $("ribbon-icon").textContent = "○";
  $("ribbon-txt").textContent = "RUNNING…";
  $("rej-box").classList.remove("show");
  ["m-rmse", "m-inliers", "m-cov", "m-gap", "m-rt", "m-matcher"].forEach(id => $(id).textContent = "—");
  $("sel-window").innerHTML = "<option>running…</option>";
  $("sel-window").disabled = true;
  setProgress(0);
  setViz(S.viz);
}

function showFailure(message) {
  $("ribbon").className = "confidence-ribbon ribbon-failed";
  $("ribbon-icon").textContent = "✕";
  $("ribbon-txt").textContent = "RUN FAILED — NO RESULT";
  $("ribbon-pair").textContent = message;
  setStages(0, false);
  log("err", message);
}

// ─────────────────────────────────────────────────────────────────── viewer (real run images)
const VIEW_ASSET = { checker: "checkerboard.png", matches: "matches.png", coverage: "coverage.png", side: "side-by-side.png" };

function setViz(mode) {
  S.viz = mode;
  ["swipe", "checker", "matches", "coverage", "side"].forEach(v => $("vt-" + v).classList.toggle("active", v === mode));
  const d = S.detail;
  const url = name => `${S.api}/runs/${S.runId}/assets/${S.base}${name}`;
  const has = name => d && (d.exports || []).includes(name);
  const why = name => (d && d.exports_skipped && (d.exports_skipped[name] || d.exports_skipped["*"])) || "not produced";
  const swipe = mode === "swipe";
  $("swipe-div").style.display = swipe && d ? "block" : "none";
  $("vp-lbl-l").style.display = $("vp-lbl-r").style.display = swipe && d ? "block" : "none";
  $("tb-match-info").style.display = mode === "matches" ? "inline" : "none";
  const missing = $("viz-missing");
  missing.style.display = "none";
  ["viz-img", "swipe-ref", "swipe-top"].forEach(id => $(id).style.display = "none");
  if (!d) return;

  if (swipe) {
    if (has("registered.png") && has("reference.png")) {
      $("swipe-ref").src = url("reference.png");
      $("swipe-top").src = url("registered.png");
      $("swipe-ref").style.display = $("swipe-top").style.display = "block";
      updateSwipe();
    } else {
      missing.textContent = "No registered product to compare: " + why("registered.png");
      missing.style.display = "flex";
      $("swipe-div").style.display = "none";
      $("vp-lbl-l").style.display = $("vp-lbl-r").style.display = "none";
    }
    return;
  }
  const name = VIEW_ASSET[mode];
  if (has(name)) {
    $("viz-img").src = url(name);
    $("viz-img").style.display = "block";
  } else {
    missing.textContent = `${name}: ${why(name)}`;
    missing.style.display = "flex";
  }
}

function updateSwipe() {
  const pct = S.swipeX * 100;
  $("swipe-top").style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
  $("swipe-div").style.left = pct + "%";
  $("tb-swipe-pct").textContent = Math.round(pct) + "%";
}

function initSwipeDrag() {
  const div = $("swipe-div"), vp = $("viewport");
  const move = e => {
    if (!S.dragging) return;
    const r = vp.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    S.swipeX = Math.max(0.01, Math.min(0.99, x / r.width));
    if (S.viz === "swipe") updateSwipe();
  };
  div.addEventListener("mousedown", () => { S.dragging = true; });
  window.addEventListener("mouseup", () => { S.dragging = false; });
  window.addEventListener("mousemove", move);
  div.addEventListener("touchstart", () => { S.dragging = true; }, { passive: true });
  window.addEventListener("touchend", () => { S.dragging = false; });
  window.addEventListener("touchmove", move, { passive: true });
}

// ─────────────────────────────────────────────────────────────────── downloads (real assets only)
function dlAsset(name) {
  const d = S.detail;
  if (!d) { log("warn", "No run yet: nothing to download."); return; }
  if (!(d.exports || []).includes(name)) {
    const why = (d.exports_skipped && (d.exports_skipped[name] || d.exports_skipped["*"])) || "not produced by this run";
    log("warn", `${name}: ${why}`);
    return;
  }
  window.open(`${S.api}/runs/${S.runId}/assets/${S.base}${name}`, "_blank");
}

// ─────────────────────────────────────────────────────────────────── log
function log(type, msg) {
  const t = $("terminal");
  const ts = new Date().toTimeString().split(" ")[0];
  const div = document.createElement("div");
  div.className = "log-line log-" + (["info", "ok", "warn"].includes(type) ? type : "err");
  const stamp = document.createElement("span");
  stamp.className = "log-ts";
  stamp.textContent = `[${ts}]`;
  div.appendChild(stamp);
  div.appendChild(document.createTextNode(msg));
  t.appendChild(div);
  t.scrollTop = t.scrollHeight;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
