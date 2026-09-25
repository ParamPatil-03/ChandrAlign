"use strict";
/**
 * CHANDRALIGN MISSION CONTROL — app.js
 * Handles: pair selection, registration simulation, canvas visualizer,
 *          live metric display, swipe interaction, download stubs.
 */

// ─────────────────────────────────────────────────────────────────────────────
// 1.  BENCHMARK PAIR DATA
// ─────────────────────────────────────────────────────────────────────────────
const PAIRS = [
  {
    id: "ohrc_nac",
    name: "CH-2 OHRC  ⇄  LRO NAC",
    desc: "South Pole, 0.25 m GSD · Δ sun 28.5°",
    tier: "GOLD", tierCls: "tier-gold",
    pairLabel: "CH2_OHRC_20200214 ⇄ LRO_NAC_M119283",
    regime: "DIFFERENT-MODAL",
    cascadeStr: "Phase Congruency → Feature",
    matcherStr: "SIFT-Normal / Phase Direct",
    metrics: { rmse_px:0.78, rmse_m:0.20, inliers:512, ratio:86.4, cov:89.1, cells:57, gap:38.4, rt:1.34, matcher:"Phase+SIFT" },
    failure_modes: []
  },
  {
    id: "tmc2_selene",
    name: "TMC-2  ⇄  SELENE TC",
    desc: "Boguslawsky crater · Δ sun 42°",
    tier: "SILVER", tierCls: "tier-silver",
    pairLabel: "TMC2_20210308 ⇄ SELENE_TC_MN00001",
    regime: "CROSS-MISSION",
    cascadeStr: "Phase Congruency Direct",
    matcherStr: "Phase Congruency / SIFT-Fallback",
    metrics: { rmse_px:1.12, rmse_m:5.60, inliers:324, ratio:74.2, cov:76.5, cells:49, gap:54.2, rt:1.82, matcher:"Phase Direct" },
    failure_modes: []
  },
  {
    id: "psr_crater",
    name: "PSR Crater Rim (Extreme Shadow)",
    desc: "Lunar south pole · Δ sun 58° · LoFTR dense",
    tier: "HARD", tierCls: "tier-hard",
    pairLabel: "OHRC_PSR_RIM_01 ⇄ NAC_SHADOW_ZONE",
    regime: "EXTREME-ILLUM",
    cascadeStr: "LoFTR Dense → RANSAC",
    matcherStr: "LoFTR Transformer Dense",
    metrics: { rmse_px:1.45, rmse_m:0.72, inliers:188, ratio:61.8, cov:65.6, cells:42, gap:72.0, rt:2.15, matcher:"LoFTR Dense" },
    failure_modes: []
  },
  {
    id: "low_texture",
    name: "Featureless Regolith (Honesty Test)",
    desc: "Smooth mare · Triggers Rule H3 rejection",
    tier: "REJECTED", tierCls: "tier-rejected",
    pairLabel: "OHRC_PLAIN_07 ⇄ NAC_MARE_FLAT",
    regime: "LOW-TEXTURE",
    cascadeStr: "Safety Gate → REFUSED",
    matcherStr: "None (Gating Refusal)",
    metrics: { rmse_px:null, rmse_m:null, inliers:8, ratio:9.5, cov:12.5, cells:8, gap:240, rt:0.88, matcher:"None (refused)" },
    failure_modes: ["#01 FEW_INLIERS", "#02 LOW_TEXTURE_VARIANCE", "#12 DELAUNAY_LARGE_GAP"]
  }
];

// ─────────────────────────────────────────────────────────────────────────────
// 2.  APPLICATION STATE
// ─────────────────────────────────────────────────────────────────────────────
const S = {
  pair: PAIRS[0],
  vizMode: "swipe",
  isMock: true,
  running: false,
  swipeX: 0.5,
  dragging: false,
  checkerN: 8,
  apiBase: window.location.origin,
  runId: null,
  surfaces: null   // { ref, src, warp, pts }
};

// ─────────────────────────────────────────────────────────────────────────────
// 3.  INIT
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  buildPairList();
  buildLunarSurfaces();
  initSwipeDrag();
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
  checkBackend();
  // Show idle overlay until user runs
  document.getElementById("vp-idle").style.display = "flex";
});

// ─────────────────────────────────────────────────────────────────────────────
// 4.  BACKEND HEALTH
// ─────────────────────────────────────────────────────────────────────────────
async function checkBackend() {
  try {
    const r = await fetch(S.apiBase + "/health", { signal: AbortSignal.timeout(1500) });
    if (r.ok) {
      setStatus(true);
      S.isMock = false;
      document.getElementById("tog-mock").checked = false;
      log("info", `Live API connected at ${S.apiBase}`);
    }
  } catch { setStatus(false); }
}

function setStatus(live) {
  const dot = document.getElementById("sys-dot");
  const txt = document.getElementById("sys-txt");
  dot.style.background = live ? "#10b981" : "#38bdf8";
  dot.style.boxShadow  = live ? "0 0 6px #10b981" : "0 0 6px #38bdf8";
  txt.textContent = live ? "API CONNECTED" : "OFFLINE READY";
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  PAIR LIST
// ─────────────────────────────────────────────────────────────────────────────
function buildPairList() {
  const el = document.getElementById("pair-list");
  el.innerHTML = PAIRS.map((p, i) => `
    <div class="pair-item ${i === 0 ? "selected" : ""}" id="pi-${p.id}" onclick="selectPair('${p.id}')">
      <div class="pair-radio"></div>
      <div class="pair-info">
        <div class="pair-name">${p.name}</div>
        <div class="pair-meta">${p.desc}</div>
      </div>
      <div class="tier-chip ${p.tierCls}">${p.tier}</div>
    </div>
  `).join("");
}

function selectPair(id) {
  if (S.running) return;
  S.pair = PAIRS.find(p => p.id === id);
  document.querySelectorAll(".pair-item").forEach(el => el.classList.remove("selected"));
  document.getElementById("pi-" + id).classList.add("selected");
  // Update telemetry labels
  document.getElementById("t-cascade").textContent = S.pair.cascadeStr;
  document.getElementById("t-matcher").textContent  = S.pair.matcherStr;
  document.getElementById("t-regime").textContent   = S.pair.regime;
  buildLunarSurfaces();
  updateMetrics(S.pair.metrics, S.pair);
  render();
  log("info", `Loaded pair: ${S.pair.name}`);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6.  TAB / VIEW SWITCHING
// ─────────────────────────────────────────────────────────────────────────────
function setTab(t) {
  ["cur","up"].forEach(x => document.getElementById("tab-"+x).classList.toggle("active", x===t));
  document.getElementById("panel-cur").style.display = t==="cur" ? "block" : "none";
  document.getElementById("panel-up").style.display  = t==="up"  ? "block" : "none";
}

function setViz(mode) {
  S.vizMode = mode;
  ["swipe","checker","matches","coverage","side"].forEach(m => {
    const b = document.getElementById("vt-"+m);
    if (b) b.classList.toggle("active", m === mode);
  });
  // Show/hide swipe divider + labels
  const div     = document.getElementById("swipe-div");
  const lblL    = document.getElementById("vp-lbl-l");
  const lblR    = document.getElementById("vp-lbl-r");
  const tbChk   = document.getElementById("tb-checker-wrap");
  const tbMatch = document.getElementById("tb-match-info");

  div.style.display = mode === "swipe" ? "block" : "none";
  lblL.style.display = (mode === "swipe" || mode === "side") ? "block" : "none";
  lblR.style.display = (mode === "swipe" || mode === "side") ? "block" : "none";
  tbChk.style.display = mode === "checker" ? "inline-flex" : "none";
  tbMatch.style.display = mode === "matches" ? "inline" : "none";

  if (mode === "swipe") {
    lblL.textContent = "Source (Before Warp)";
    lblR.textContent = "Registered Product";
  } else if (mode === "side") {
    lblL.textContent = "Source (CH-2)";
    lblR.textContent = "Reference (LRO)";
  }
  render();
}

function updateChecker(v) {
  S.checkerN = parseInt(v);
  document.getElementById("tb-grid").textContent = `${v}×${v}`;
  render();
}

function onMockToggle() {
  S.isMock = document.getElementById("tog-mock").checked;
  log("info", "Mode: " + (S.isMock ? "Offline simulation" : "Live FastAPI"));
}

function fileSelected(type, e) {
  const f = e.target.files[0];
  if (!f) return;
  document.getElementById("lbl-" + type).textContent = `✓ ${f.name}`;
  log("ok", `Attached ${type.toUpperCase()}: ${f.name}`);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7.  LUNAR CANVAS SURFACE GENERATOR
// ─────────────────────────────────────────────────────────────────────────────
function buildLunarSurfaces() {
  const W = 1200, H = 600;
  const rejected = S.pair.tier === "REJECTED";

  // Seed craters deterministically
  const craters = [
    { x:220, y:160, r:70 }, { x:630, y:290, r:90 }, { x:450, y:200, r:44 },
    { x:110, y:330, r:36 }, { x:790, y:120, r:55 }, { x:320, y:370, r:30 },
    { x:550, y:90,  r:24 }, { x:750, y:380, r:32 }, { x:980, y:260, r:48 }
  ];

  const ref  = mkCanvas(W, H);
  const src  = mkCanvas(W, H);
  const warp = mkCanvas(W, H);

  drawLunar(ref.ctx,  W, H, craters, -Math.PI/4,    rejected, 88);
  drawLunar(src.ctx,  W, H, craters, -Math.PI*0.65, rejected, 72, 18, -14);
  drawLunar(warp.ctx, W, H, craters, -Math.PI*0.65, rejected, 72);

  // Match points
  const pts = [];
  craters.forEach((c, i) => {
    if (rejected && i > 2) return;
    const n = Math.max(2, Math.floor(c.r / 4));
    for (let k = 0; k < n; k++) {
      const a = (k/n) * Math.PI*2 + (Math.random()-.5)*.3;
      const rad = c.r * (.35 + Math.random()*.55);
      const rx = c.x + Math.cos(a)*rad, ry = c.y + Math.sin(a)*rad;
      const sx = rx + (rejected ? (Math.random()-.5)*70 : 18);
      const sy = ry + (rejected ? (Math.random()-.5)*70 : -14);
      pts.push({ rx, ry, sx, sy, inlier: rejected ? (k%4===0) : (Math.random()>.13) });
    }
  });

  S.surfaces = { ref: ref.el, src: src.el, warp: warp.el, pts };
  render();
}

function mkCanvas(w, h) {
  const el = document.createElement("canvas");
  el.width = w; el.height = h;
  return { el, ctx: el.getContext("2d") };
}

function drawLunar(ctx, W, H, craters, sunAngle, rejected, base=85, dx=0, dy=0) {
  // Base regolith
  ctx.fillStyle = `rgb(${base},${base},${base+3})`;
  ctx.fillRect(0,0,W,H);
  // Noise
  const id = ctx.getImageData(0,0,W,H);
  const d = id.data;
  for (let i=0; i<d.length; i+=4) {
    const n = (Math.random()-.5)*22;
    d[i]=clamp(d[i]+n); d[i+1]=clamp(d[i+1]+n); d[i+2]=clamp(d[i+2]+n);
  }
  ctx.putImageData(id,0,0);

  // Craters
  craters.forEach(c => {
    if (rejected && c.r > 45) return;
    const cx = c.x + dx, cy = c.y + dy;
    const sx = Math.cos(sunAngle), sy = Math.sin(sunAngle);

    // Rim highlight
    const g = ctx.createRadialGradient(cx+sx*c.r*.4, cy+sy*c.r*.4, c.r*.05, cx, cy, c.r);
    g.addColorStop(0, "rgba(235,240,250,.9)");
    g.addColorStop(.35,"rgba(175,180,190,.65)");
    g.addColorStop(.65,"rgba(38,42,50,.8)");
    g.addColorStop(1, "rgba(8,10,14,.95)");
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(cx,cy,c.r,0,Math.PI*2); ctx.fill();

    // Floor shadow
    const s = ctx.createRadialGradient(cx-sx*c.r*.35, cy-sy*c.r*.35, c.r*.05, cx,cy,c.r*.76);
    s.addColorStop(0,"rgba(4,5,8,.95)"); s.addColorStop(.6,"rgba(28,32,38,.4)"); s.addColorStop(1,"transparent");
    ctx.fillStyle = s;
    ctx.beginPath(); ctx.arc(cx,cy,c.r*.76,0,Math.PI*2); ctx.fill();

    // Central peak
    if (c.r > 55) {
      ctx.fillStyle = "rgba(215,225,238,.75)";
      ctx.beginPath(); ctx.arc(cx+sx*4, cy+sy*4, c.r*.11, 0, Math.PI*2); ctx.fill();
    }
  });
}

function clamp(v) { return Math.max(0, Math.min(255, Math.round(v))); }

// ─────────────────────────────────────────────────────────────────────────────
// 8.  CANVAS RENDER
// ─────────────────────────────────────────────────────────────────────────────
function resizeCanvas() {
  const vp = document.getElementById("viewport");
  const cv = document.getElementById("viz-canvas");
  cv.width  = vp.clientWidth;
  cv.height = vp.clientHeight;
  render();
}

function render() {
  const cv  = document.getElementById("viz-canvas");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  if (!S.surfaces) return;
  const { ref, src, warp, pts } = S.surfaces;

  ctx.clearRect(0,0,W,H);

  if (S.vizMode === "swipe") {
    ctx.drawImage(warp, 0,0,W,H);
    const sp = S.swipeX * W;
    ctx.save(); ctx.beginPath(); ctx.rect(0,0,sp,H); ctx.clip();
    ctx.drawImage(src, 0,0,W,H);
    ctx.restore();
    document.getElementById("swipe-div").style.left = (S.swipeX*100)+"%";
    document.getElementById("tb-swipe-pct").textContent = Math.round(S.swipeX*100)+"%";

  } else if (S.vizMode === "checker") {
    const n = S.checkerN, tw = W/n, th = H/n;
    for (let r=0; r<n; r++) {
      for (let c=0; c<n; c++) {
        const img = (r+c)%2===0 ? ref : warp;
        ctx.drawImage(img, c*tw,r*th,tw,th, c*tw,r*th,tw,th);
        ctx.strokeStyle="rgba(56,189,248,.12)";
        ctx.strokeRect(c*tw,r*th,tw,th);
      }
    }

  } else if (S.vizMode === "matches") {
    ctx.drawImage(ref, 0,0,W,H);
    const inliers = pts.filter(p=>p.inlier), outliers = pts.filter(p=>!p.inlier);
    const scaleX = W/ref.width, scaleY = H/ref.height;

    outliers.forEach(p => {
      ctx.beginPath();
      ctx.moveTo(p.rx*scaleX, p.ry*scaleY);
      ctx.lineTo(p.sx*scaleX, p.sy*scaleY);
      ctx.strokeStyle="rgba(239,68,68,.6)"; ctx.lineWidth=1.2; ctx.stroke();
      ctx.beginPath(); ctx.arc(p.rx*scaleX, p.ry*scaleY, 2, 0, Math.PI*2);
      ctx.fillStyle="#ef4444"; ctx.fill();
    });
    inliers.forEach(p => {
      ctx.beginPath();
      ctx.moveTo(p.rx*scaleX, p.ry*scaleY);
      ctx.lineTo(p.sx*scaleX, p.sy*scaleY);
      ctx.strokeStyle="rgba(16,185,129,.75)"; ctx.lineWidth=1.6; ctx.stroke();
      ctx.beginPath(); ctx.arc(p.rx*scaleX, p.ry*scaleY, 2.5, 0, Math.PI*2);
      ctx.fillStyle="#10b981"; ctx.fill();
    });
    document.getElementById("tb-inliers").textContent  = inliers.length;
    document.getElementById("tb-outliers").textContent = outliers.length;

  } else if (S.vizMode === "coverage") {
    ctx.drawImage(ref, 0,0,W,H);
    const G = 8, cw = W/G, ch = H/G;
    const sx = W/ref.width, sy = H/ref.height;
    for (let r=0; r<G; r++) {
      for (let c=0; c<G; c++) {
        const cnt = pts.filter(p => p.inlier
          && p.rx*sx >= c*cw && p.rx*sx < (c+1)*cw
          && p.ry*sy >= r*ch && p.ry*sy < (r+1)*ch
        ).length;
        ctx.fillStyle = cnt > 0
          ? `rgba(16,185,129,${Math.min(.65,.12+cnt*.09)})`
          : "rgba(239,68,68,.22)";
        ctx.fillRect(c*cw, r*ch, cw, ch);
        ctx.strokeStyle="rgba(255,255,255,.08)";
        ctx.strokeRect(c*cw, r*ch, cw, ch);
        ctx.fillStyle="rgba(255,255,255,.7)";
        ctx.font="10px Courier New";
        ctx.fillText(cnt, c*cw+5, r*ch+14);
      }
    }

  } else if (S.vizMode === "side") {
    const hw = W/2;
    ctx.drawImage(src, 0,0,hw,H);
    ctx.drawImage(ref, hw,0,hw,H);
    ctx.strokeStyle="rgba(56,189,248,.35)"; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(hw,0); ctx.lineTo(hw,H); ctx.stroke();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9.  SWIPE DRAG
// ─────────────────────────────────────────────────────────────────────────────
function initSwipeDrag() {
  const div = document.getElementById("swipe-div");
  const vp  = document.getElementById("viewport");

  const move = e => {
    if (!S.dragging) return;
    const r = vp.getBoundingClientRect();
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    S.swipeX = Math.max(.01, Math.min(.99, cx/r.width));
    if (S.vizMode === "swipe") render();
  };

  div.addEventListener("mousedown",  () => { S.dragging = true; });
  window.addEventListener("mouseup", () => { S.dragging = false; });
  window.addEventListener("mousemove", move);
  div.addEventListener("touchstart",  () => { S.dragging = true; }, {passive:true});
  window.addEventListener("touchend", () => { S.dragging = false; });
  window.addEventListener("touchmove", move, {passive:true});
}

// ─────────────────────────────────────────────────────────────────────────────
// 10.  METRICS DISPLAY
// ─────────────────────────────────────────────────────────────────────────────
function updateMetrics(m, pair) {
  const fmt = (v, d=2) => v !== null && v !== undefined ? v.toFixed(d) : "—";

  document.getElementById("m-rmse").textContent  = fmt(m.rmse_px);
  document.getElementById("m-rmse-m").textContent = m.rmse_m !== null && m.rmse_m !== undefined ? `≈ ${m.rmse_m.toFixed(2)} m` : "Not measured (H1)";
  document.getElementById("m-inliers").textContent = m.inliers ?? "—";
  document.getElementById("m-ratio").textContent   = `Ratio: ${fmt(m.ratio,1)}%`;
  document.getElementById("m-cov").textContent      = fmt(m.cov, 1);
  document.getElementById("m-cells").textContent    = m.cells ? `${m.cells}/64 cells` : "—";
  document.getElementById("m-gap").textContent      = fmt(m.gap, 1);
  document.getElementById("m-rt").textContent       = fmt(m.rt, 2);
  document.getElementById("m-matcher").textContent  = m.matcher ?? "—";

  // Ribbon
  const ribbon = document.getElementById("ribbon");
  const icons  = { GOLD:"⭐", SILVER:"🔵", BRONZE:"🟠", HARD:"🟡", REJECTED:"🔴" };
  const cls    = { GOLD:"ribbon-gold", SILVER:"ribbon-silver", HARD:"ribbon-bronze", BRONZE:"ribbon-bronze", REJECTED:"ribbon-rejected" };

  ribbon.className = "confidence-ribbon " + (cls[pair.tier] || "ribbon-silver");
  document.getElementById("ribbon-icon").textContent = icons[pair.tier] || "●";
  document.getElementById("ribbon-txt").textContent  = `CONFIDENCE TIER: ${pair.tier}`;
  document.getElementById("ribbon-pair").textContent = pair.pairLabel;

  // Rejection box (Rule H3)
  const rejBox = document.getElementById("rej-box");
  if (pair.tier === "REJECTED") {
    rejBox.classList.add("show");
    document.getElementById("rej-modes").innerHTML =
      (pair.failure_modes || []).map(f => `<span class="rej-pill">${f}</span>`).join("");
  } else {
    rejBox.classList.remove("show");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 11.  PIPELINE STAGE TRACKER
// ─────────────────────────────────────────────────────────────────────────────
function setStage(active, pct) {
  for (let i=1; i<=5; i++) {
    const el = document.getElementById("sp"+i);
    if (!el) continue;
    el.className = "stage-pill " + (i < active ? "done" : i === active ? "active" : "");
  }
  document.getElementById("prog-fill").style.width = pct + "%";
  document.getElementById("prog-pct").textContent  = pct + "%";
}

// ─────────────────────────────────────────────────────────────────────────────
// 12.  REGISTRATION TRIGGER
// ─────────────────────────────────────────────────────────────────────────────
async function startRegistration() {
  if (S.running) return;
  S.running = true;
  const btn = document.getElementById("btn-run");
  btn.disabled = true;
  btn.classList.add("running");
  document.getElementById("btn-icon").textContent = "⏳";
  document.getElementById("btn-txt").textContent  = "PIPELINE RUNNING…";

  // Hide idle overlay, show canvas
  document.getElementById("vp-idle").style.display = "none";

  setStage(1, 0);
  log("info", `Dispatching ${S.pair.name} (${S.pair.regime})`);
  log("info", `Cascade: ${S.pair.cascadeStr}`);

  if (S.isMock) {
    await runSimulation();
  } else {
    await runLive();
  }

  btn.disabled = false;
  btn.classList.remove("running");
  document.getElementById("btn-icon").textContent = "▶";
  document.getElementById("btn-txt").textContent  = "RUN REGISTRATION";
  S.running = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// 13.  SIMULATION MODE
// ─────────────────────────────────────────────────────────────────────────────
async function runSimulation() {
  const steps = [
    { stage:1, pct:18,  delay:280, msg:"Stage 1/5: Safety gating & geometry check…" },
    { stage:2, pct:40,  delay:380, msg:"Stage 2/5: Phase congruency moment extraction (PREP-02)…" },
    { stage:3, pct:65,  delay:420, msg:`Stage 3/5: ${S.pair.cascadeStr} — feature matching…` },
    { stage:4, pct:84,  delay:340, msg:"Stage 4/5: RANSAC estimation + sub-pixel local refinement…" },
    { stage:5, pct:100, delay:240, msg:"Stage 5/5: GeoTIFF warp + provenance manifest (OUT-02/03)…" }
  ];

  for (const s of steps) {
    setStage(s.stage, s.pct);
    log("info", s.msg);
    await sleep(s.delay);
  }

  buildLunarSurfaces();
  updateMetrics(S.pair.metrics, S.pair);
  render();
  document.getElementById("run-id-lbl").textContent = "RUN: DEMO-" + Math.random().toString(36).slice(2,8).toUpperCase();
  log("ok", `Done. Tier: ${S.pair.tier} · RMSE: ${S.pair.metrics.rmse_px ?? "N/A"} px`);
  if (S.pair.tier === "REJECTED") {
    log("warn", `Rejection active — failure modes: ${S.pair.failure_modes.join(", ")}`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 14.  LIVE BACKEND MODE
// ─────────────────────────────────────────────────────────────────────────────
async function runLive() {
  try {
    const matcher = document.getElementById("sel-matcher").value;
    log("info", `POST /register  matcher=${matcher}  mock=true`);

    const res = await fetch(S.apiBase + "/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mock: true, matcher })
    });

    if (!res.ok) throw new Error("Server " + res.status);
    const { run_id, poll_url } = await res.json();
    S.runId = run_id;
    document.getElementById("run-id-lbl").textContent = "RUN: " + run_id.slice(0,8).toUpperCase();
    log("info", `Job queued: ${run_id}`);

    // Poll
    let prevLogs = 0, polls = 0;
    while (polls < 90) {
      await sleep(500);
      polls++;
      const r = await fetch(`${S.apiBase}/runs/${run_id}`);
      if (!r.ok) continue;
      const rec = await r.json();

      // Stream new log lines
      const allLogs = rec.progress_log || [];
      for (let i = prevLogs; i < allLogs.length; i++) {
        log("info", allLogs[i]);
      }
      prevLogs = allLogs.length;

      const pct = Math.min(95, polls * 12);
      setStage(Math.min(5, Math.ceil(pct/20)), pct);

      if (rec.status === "DONE") {
        setStage(5, 100);
        const res = rec.result || {};
        const m = res.metrics || {};
        updateMetrics({
          rmse_px: m.rmse_px, rmse_m: m.rmse_m,
          inliers: m.inlier_count, ratio: m.inlier_ratio != null ? m.inlier_ratio*100 : null,
          cov: m.spatial_coverage != null ? m.spatial_coverage*100 : null,
          cells: null, gap: m.max_delaunay_gap_px, rt: m.runtime_s,
          matcher: document.getElementById("sel-matcher").value
        }, { tier: res.confidence_tier || "SILVER", pairLabel: S.pair.pairLabel, failure_modes: res.failure_modes || [] });
        buildLunarSurfaces();
        render();
        log("ok", `Job done. Tier: ${res.confidence_tier}`);
        return;
      }

      if (rec.status === "FAILED") {
        throw new Error(rec.error || "Pipeline failure");
      }
    }
    throw new Error("Timeout waiting for backend");
  } catch (e) {
    log("err", "Backend error: " + e.message + " — falling back to simulation");
    document.getElementById("tog-mock").checked = true;
    S.isMock = true;
    await runSimulation();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 15.  LOG HELPER
// ─────────────────────────────────────────────────────────────────────────────
function log(type, msg) {
  const t  = document.getElementById("terminal");
  const ts = new Date().toTimeString().split(" ")[0];
  const div = document.createElement("div");
  div.className = "log-line log-" + (type === "info" ? "info" : type === "ok" ? "ok" : type === "warn" ? "warn" : "err");
  div.innerHTML = `<span class="log-ts">[${ts}]</span>${msg}`;
  t.appendChild(div);
  t.scrollTop = t.scrollHeight;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─────────────────────────────────────────────────────────────────────────────
// 16.  ASSET DOWNLOAD STUBS
// ─────────────────────────────────────────────────────────────────────────────
function dlAsset(name) {
  if (S.runId && !S.isMock) {
    window.open(`${S.apiBase}/runs/${S.runId}/assets/${name}`, "_blank");
    return;
  }
  // Generate demo artifact
  let content = "", mime = "text/plain";
  if (name.endsWith(".json")) {
    content = JSON.stringify({ project:"Chandralign", pair:S.pair.name, metrics:S.pair.metrics, ts:new Date().toISOString() }, null, 2);
    mime = "application/json";
  } else if (name.endsWith(".csv")) {
    content = "src_x,src_y,ref_x,ref_y,inlier\n" +
      (S.surfaces?.pts||[]).map(p=>`${p.sx.toFixed(1)},${p.sy.toFixed(1)},${p.rx.toFixed(1)},${p.ry.toFixed(1)},${p.inlier?1:0}`).join("\n");
    mime = "text/csv";
  } else {
    content = `Chandralign demo export: ${name}\nPair: ${S.pair.name}\nTier: ${S.pair.tier}\nTimestamp: ${new Date().toISOString()}`;
  }
  const blob = new Blob([content], {type: mime});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
  log("ok", `Exported: ${name}`);
}
