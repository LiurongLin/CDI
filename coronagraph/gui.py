from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HELP_TEXT = """Coronagraph Web GUI Parameter Guide

Feature
- ROI phase simulation only.

Mask and Sampling
- Phase Mask: roddier or vortex.
- Roddier Radius / Roddier Phase: mask parameters when Phase Mask=roddier.
- Vortex Charge: mask charge when Phase Mask=vortex.
- Pupil SS: entrance-pupil supersampling factor.

Local/ROI Controls
- Sweep Mode: regional or global.
- Local Region Radius: circular ROI radius (lambda/D).
- Region Shape: currently only circle.
- FOV Count: number of FOVs phase-shifted simultaneously per step.
- Number of FOV Centers: total FOV centers explored sequentially.
- Single Ring Radius: optional orbit/expansion radius.

Phase Controls
- Phase Step: number of phase steps for each ROI.
- Phase Cycles / FOV: number of phase cycles allocated to each sequential FOV-center group.
- Planet Flux Ratio.
- Planet Offset X/Y.
- Secondary Ratio (Local).

ROI Size Sweep
- Enable ROI Size Sweep.
- ROI Size Min / Max / Step.
- When enabled, Local Region Radius is ignored.
- Output is only SNR vs theta and stored in a dedicated folder.

Optics/Pupil
- Spider Width.
- Spider Angles.

Flags
- Disable Ghost.
- Disable Interference.
- Disable Companion Ghost.
- Build Map per FOV.
"""


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>CDI Explorer</title>
  <style>
    :root {
      --bg0: #f7f7e0;
      --bg1: #b4ccd2;
      --card: rgba(12, 20, 44, 0.82);
      --card-border: rgba(121, 163, 206, 0.36);
      --text: #f7f7e0;
      --muted: #b4ccd2;
      --accent: #79a3ce;
      --accent2: #b4ccd2;
      --warm: #f0dc84;
      --ok: #f0dc84;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(1200px 500px at -10% -20%, rgba(121,163,206,0.40), transparent 60%),
        radial-gradient(1200px 500px at 110% -15%, rgba(240,220,132,0.34), transparent 60%),
        linear-gradient(160deg, var(--bg0), var(--bg1));
      padding: 28px;
    }
    body[data-theme="presentation"] {
      --card: rgba(12, 20, 44, 0.82);
      --card-border: rgba(121, 163, 206, 0.36);
    }
    body[data-theme="contrast"] {
      --card: rgba(8, 12, 30, 0.92);
      --card-border: rgba(240, 220, 132, 0.65);
    }
    .container {
      max-width: 1120px;
      margin: 0 auto;
      background: var(--card);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      box-shadow: 0 24px 70px rgba(0, 0, 0, 0.50), 0 0 0 1px rgba(121,163,206,0.22) inset;
      padding: 24px;
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }
    .theme-select-wrap {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 700;
    }
    .theme-select-wrap select {
      min-width: 150px;
      padding: 8px 10px;
    }
    h2 {
      margin: 0 0 8px 0;
      font-size: 28px;
      letter-spacing: 0.2px;
    }
    p.subtitle { margin: 0 0 18px 0; color: var(--muted); }
    code {
      color: #f7f7e0;
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid rgba(255, 255, 255, 0.14);
      border-radius: 8px;
      padding: 2px 8px;
    }
    .grid {
      column-count: 2;
      column-gap: 22px;
    }
    .field {
      break-inside: avoid;
      display: grid;
      grid-template-columns: minmax(170px, 220px) minmax(140px, 1fr);
      gap: 8px 12px;
      align-items: center;
      margin: 0 0 12px 0;
    }
    .field input, .field select {
      width: 12ch;
      max-width: 100%;
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .field select {
      text-align: left;
      width: 16ch;
    }
    label {
      color: #f7f7e0;
      font-weight: 800;
      font-size: 14px;
      letter-spacing: 0.2px;
      text-shadow: 0 1px 0 rgba(0,0,0,0.35);
    }
    .unit {
      display: inline-block;
      margin-left: 6px;
      padding: 1px 7px;
      border-radius: 999px;
      font-size: 11px;
      color: #0f1d3b;
      background: linear-gradient(140deg, #f0dc84, #b4ccd2);
      border: 1px solid rgba(255,255,255,0.55);
      vertical-align: middle;
      text-shadow: none;
    }
    .tip {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      margin-left: 6px;
      font-size: 11px;
      color: #0f1d3b;
      background: #f0dc84;
      border: 1px solid rgba(255,255,255,0.6);
      cursor: help;
      text-shadow: none;
    }
    input, select, button, textarea { font: inherit; }
    input, select, textarea {
      background: rgba(5, 12, 30, 0.86);
      color: var(--text);
      border: 1px solid rgba(121, 163, 206, 0.42);
      border-radius: 10px;
      padding: 10px 12px;
      outline: none;
      transition: border-color .2s ease, box-shadow .2s ease, background .2s ease;
    }
    input:focus, select:focus, textarea:focus {
      border-color: rgba(121, 163, 206, 0.95);
      box-shadow: 0 0 0 3px rgba(121, 163, 206, 0.24);
      background: rgba(8, 18, 42, 0.9);
    }
    input:disabled, select:disabled {
      opacity: 0.45;
      border-color: rgba(180,204,210,0.35);
      background: rgba(12, 20, 38, 0.65);
      cursor: not-allowed;
    }
    .row { margin-top: 26px; }
    .checks { display: flex; gap: 12px; flex-wrap: wrap; margin: 14px 0; color: #f7f7e0; }
    .checks label {
      font-weight: 700;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid rgba(255,255,255,0.20);
      background: rgba(255,255,255,0.07);
      border-radius: 999px;
      padding: 8px 12px;
      cursor: pointer;
      user-select: none;
      transition: all .15s ease;
    }
    .checks label:hover {
      border-color: rgba(121,163,206,0.75);
      box-shadow: 0 0 0 2px rgba(121,163,206,0.18);
    }
    .checks input[type="checkbox"] {
      appearance: none;
      -webkit-appearance: none;
      width: 14px;
      height: 14px;
      margin: 0;
      border: 2px solid #b4ccd2;
      border-radius: 4px;
      background: transparent;
      position: relative;
    }
    .checks input[type="checkbox"]:checked {
      border-color: #f0dc84;
      background: #f0dc84;
    }
    .checks label.is-on {
      background: rgba(240,220,132,0.22);
      border-color: rgba(240,220,132,0.62);
      color: #f7f7e0;
    }
    .toggle-inline {
      font-weight: 700;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid rgba(255,255,255,0.20);
      background: rgba(255,255,255,0.07);
      border-radius: 999px;
      padding: 8px 12px;
      cursor: pointer;
      user-select: none;
      transition: all .15s ease;
      width: fit-content;
    }
    .toggle-inline:hover {
      border-color: rgba(121,163,206,0.75);
      box-shadow: 0 0 0 2px rgba(121,163,206,0.18);
    }
    .toggle-inline input[type="checkbox"] {
      appearance: none;
      -webkit-appearance: none;
      width: 14px;
      height: 14px;
      margin: 0;
      border: 2px solid #b4ccd2;
      border-radius: 4px;
      background: transparent;
      position: relative;
    }
    .toggle-inline input[type="checkbox"]:checked {
      border-color: #f0dc84;
      background: #f0dc84;
    }
    .toggle-inline.is-on {
      background: rgba(240,220,132,0.22);
      border-color: rgba(240,220,132,0.62);
      color: #f7f7e0;
    }
    textarea {
      width: 100%;
      height: 360px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.45;
      background: #060b1a;
      border-color: rgba(121,163,206,0.45);
      color: #d9e6ff;
    }
    fieldset {
      margin-top: 28px;
      border: 1px solid rgba(121,163,206,0.38);
      border-radius: 14px;
      padding: 14px;
      background: rgba(4, 10, 24, 0.55);
    }
    .section-card {
      margin-top: 28px;
      border: 1px solid rgba(121,163,206,0.38);
      border-radius: 14px;
      background: rgba(4, 10, 24, 0.55);
      padding: 14px;
    }
    .section-title {
      color: #f7f7e0;
      font-weight: 800;
      letter-spacing: 0.3px;
      margin: 0 0 10px 0;
      font-size: 15px;
      text-transform: uppercase;
    }
    legend {
      color: #b4ccd2;
      font-weight: 650;
      padding: 0 8px;
    }
    .roi-sweep-wrap {
      background: linear-gradient(160deg, rgba(121,163,206,0.16), rgba(240,220,132,0.10));
      border: 1px solid rgba(240,220,132,0.45);
      border-radius: 12px;
      padding: 10px 12px;
      box-shadow: 0 0 0 1px rgba(240,220,132,0.18) inset;
    }
    .roi-sweep-toggle-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }
    .roi-sweep-toggle-row .left {
      color: #f0dc84;
      font-weight: 700;
      font-size: 15px;
      letter-spacing: 0.2px;
      text-transform: uppercase;
    }
    .roi-sweep-fields {
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px dashed rgba(240,220,132,0.42);
      transition: all .18s ease;
    }
    .roi-sweep-fields.is-hidden {
      display: none;
    }
    .actions {
      position: sticky;
      bottom: 12px;
      z-index: 20;
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 30px;
      padding: 10px 12px;
      background: rgba(7, 13, 30, 0.92);
      border: 1px solid rgba(121,163,206,0.45);
      border-radius: 12px;
      box-shadow: 0 10px 28px rgba(0,0,0,0.35);
    }
    button {
      border: 1px solid transparent;
      border-radius: 10px;
      padding: 10px 14px;
      color: #071122;
      background: linear-gradient(140deg, var(--accent), var(--accent2));
      font-weight: 650;
      cursor: pointer;
      transition: transform .12s ease, box-shadow .2s ease, filter .2s ease;
      box-shadow: 0 8px 22px rgba(121,163,206,0.45);
    }
    button:hover { transform: translateY(-1px); filter: brightness(1.04); }
    button:active { transform: translateY(0); }
    button:nth-child(2) { background: linear-gradient(140deg, #f0dc84, #b4ccd2); box-shadow: 0 8px 20px rgba(240,220,132,0.42); }
    button:nth-child(3), button:nth-child(4) {
      background: rgba(255,255,255,0.08);
      color: #f7f7e0;
      border-color: rgba(255,255,255,0.18);
      box-shadow: none;
    }
    #status {
      margin-left: 8px;
      color: var(--muted);
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.22);
      background: rgba(255,255,255,0.06);
    }
    .status-running { color: var(--ok) !important; }
    .status-idle { color: #b4ccd2 !important; }
    @media (max-width: 980px) {
      .grid {
        column-count: 1;
      }
      .field input, .field select { width: 100%; }
    }
    #toast_wrap {
      position: fixed;
      right: 18px;
      top: 18px;
      z-index: 99;
      display: flex;
      flex-direction: column;
      gap: 8px;
      pointer-events: none;
    }
    .toast {
      pointer-events: auto;
      min-width: 220px;
      max-width: 360px;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.22);
      box-shadow: 0 12px 30px rgba(0,0,0,0.35);
      font-size: 13px;
      font-weight: 700;
      color: #0f1d3b;
      background: linear-gradient(140deg, #f0dc84, #b4ccd2);
      opacity: 0.98;
    }
    .toast.error { background: linear-gradient(140deg, #ff9aa2, #ffd1a8); }
    .toast.ok { background: linear-gradient(140deg, #b4ffd7, #f0dc84); }
  </style>
</head>
<body>
  <div class="container">
  <div class="topbar">
    <div>
      <h2>CDI Explorer</h2>
      <p class="subtitle">Mode: <code>coc-planet-phase</code></p>
    </div>
    <div class="theme-select-wrap">
      <label for="theme_preset">Theme</label>
      <select id="theme_preset">
        <option value="presentation">Presentation</option>
        <option value="contrast">High Contrast</option>
      </select>
    </div>
  </div>

  <div class="section-card">
  <h3 class="section-title">Simulation Parameters</h3>
  <div class="grid">
    <div class="field"><label>Phase Mask</label><select id="phase_mask_type"><option>roddier</option><option>vortex</option></select></div>
    <div class="field"><label id="label_roddier_mask_radius">Roddier Radius <span class="unit">λ/D</span></label><input id="roddier_mask_radius" value="0.53" /></div>
    <div class="field"><label id="label_roddier_mask_phase">Roddier Phase <span class="unit">rad</span></label><input id="roddier_mask_phase" value="3.1415926535" /></div>
    <div class="field"><label id="label_vortex_charge">Vortex Charge</label><input id="vortex_charge" value="2" /></div>
    <div class="field"><label>Spider Width <span class="unit">px</span></label><input id="spider_width" value="0.25" /></div>
    <div class="field"><label>Spider Angles <span class="tip" title="Space-separated angles in degrees, e.g. 0 90">?</span></label><input id="spider_angles" value="0 90" /></div>
    <div class="field"><label>Pupil SS <span class="tip" title="Entrance pupil supersampling factor per axis">?</span></label><input id="pupil_ss" value="8" /></div>
    <div class="field"><label>Sweep Mode</label><select id="phase_sweep_mode"><option>regional</option><option>global</option></select></div>
    <div class="field"><label>Local Region Radius <span class="unit">λ/D</span></label><input id="local_region_radius" value="2.0" /></div>
    <div class="field"><label>Region Shape</label><select id="region_shape"><option>circle</option></select></div>
    <div class="field"><label>FOV Count <span class="tip" title="How many FOVs are phase-shifted simultaneously">?</span></label><input id="fov_count" value="1" /></div>
    <div class="field"><label>Number of FOV Centers <span class="tip" title="Total FOV centers visited sequentially">?</span></label><input id="fov_centers_count" value="1" /></div>
    <div class="field"><label>Single Ring Radius <span class="unit">λ/D</span></label><input id="single_region_ring_radius" value="" /></div>
    <div class="field"><label>Phase Step</label><input id="phase_step" value="61" /></div>
    <div class="field"><label>Phase Cycles / FOV</label><input id="phase_cycles" value="1.0" /></div>
    <div class="field"><label>Planet Offset X <span class="unit">λ/D</span></label><input id="planet_offset_x_local" value="0.0" /></div>
    <div class="field"><label>Planet Offset Y <span class="unit">λ/D</span></label><input id="planet_offset_y_local" value="0.0" /></div>
    <div class="field"><label>Secondary Ratio (Local)</label><input id="secondary_ratio_local" value="0.25" /></div>
    <div class="field"><label>Planet Flux Ratio</label><input id="planet_flux_ratio_local" value="0.01" /></div>
  </div>
  </div>

  <fieldset>
    <legend>ROI Size Sweep</legend>
    <div class="roi-sweep-wrap">
      <div class="roi-sweep-toggle-row">
        <div class="left">Enable ROI Size Sweep</div>
        <label class="toggle-inline" id="label_roi_size_sweep"><input type="checkbox" id="roi_size_sweep" /> On</label>
      </div>
      <div id="roi_sweep_fields" class="roi-sweep-fields">
        <div class="grid">
          <div class="field"><label>ROI Size Min</label><input id="roi_size_min" value="0.5" /></div>
          <div class="field"><label>ROI Size Max</label><input id="roi_size_max" value="3.0" /></div>
          <div class="field"><label>ROI Size Step</label><input id="roi_size_step" value="0.25" /></div>
        </div>
      </div>
    </div>
  </fieldset>

  <div class="section-card">
  <h3 class="section-title">Run Options</h3>
  <div class="checks">
    <label><input type="checkbox" id="disable_ghost" /> Disable Ghost</label>
    <label><input type="checkbox" id="disable_interference" /> Disable Interference</label>
    <label><input type="checkbox" id="disable_companion_ghost" /> Disable Companion Ghost</label>
    <label><input type="checkbox" id="build_map_per_fov" /> Build Map per FOV</label>
  </div>
  </div>

  <div class="row actions">
    <button onclick="runCmd()">Run</button>
    <button onclick="stopCmd()">Stop</button>
    <button onclick="clearLog()">Clear Log</button>
    <button onclick="showHelp()">Help</button>
    <span id="status"></span>
  </div>

  <div class="section-card">
  <h3 class="section-title">Execution Log</h3>
  <div class="row"><textarea id="log" readonly></textarea></div>
  </div>
  </div>
  <div id="toast_wrap"></div>

<script>
const fields = [
  "phase_mask_type","roddier_mask_radius","roddier_mask_phase","vortex_charge","spider_width",
  "spider_angles","pupil_ss","phase_sweep_mode","local_region_radius","region_shape","fov_count","fov_centers_count",
  "single_region_ring_radius","phase_step",
  "phase_cycles","planet_offset_x_local","planet_offset_y_local","secondary_ratio_local",
  "planet_flux_ratio_local","roi_size_sweep","roi_size_min","roi_size_max","roi_size_step",
  "disable_ghost","disable_interference",
  "disable_companion_ghost","build_map_per_fov"
];

function collect() {
  const d = {};
  for (const k of fields) {
    const el = document.getElementById(k);
    d[k] = (el.type === "checkbox") ? el.checked : el.value;
  }
  return d;
}

function showToast(msg, kind="ok") {
  const wrap = document.getElementById("toast_wrap");
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  wrap.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 250);
  }, 2400);
}

function setMaskVisibility() {
  const isRoddier = document.getElementById("phase_mask_type").value === "roddier";
  const roddierIds = ["label_roddier_mask_radius", "roddier_mask_radius", "label_roddier_mask_phase", "roddier_mask_phase"];
  const vortexIds = ["label_vortex_charge", "vortex_charge"];
  for (const id of roddierIds) document.getElementById(id).style.display = isRoddier ? "" : "none";
  for (const id of vortexIds) document.getElementById(id).style.display = isRoddier ? "none" : "";
}

function updateChoiceButtons() {
  const labels = document.querySelectorAll(".checks label");
  for (const lab of labels) {
    const cb = lab.querySelector('input[type="checkbox"]');
    if (!cb) continue;
    if (cb.checked) lab.classList.add("is-on");
    else lab.classList.remove("is-on");
  }
  const roiLab = document.getElementById("label_roi_size_sweep");
  const roiCb = document.getElementById("roi_size_sweep");
  if (roiLab && roiCb) {
    if (roiCb.checked) roiLab.classList.add("is-on");
    else roiLab.classList.remove("is-on");
  }
}

async function runCmd() {
  const r = await fetch("/api/run", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(collect())
  });
  const j = await r.json();
  if (!j.ok) {
    showToast(j.error || "Run failed", "error");
  } else {
    showToast("Run started", "ok");
  }
}

async function stopCmd() {
  await fetch("/api/stop", {method:"POST"});
  showToast("Termination requested", "ok");
}
function clearLog() { document.getElementById("log").value = ""; }
async function showHelp() {
  const r = await fetch("/api/help");
  const j = await r.json();
  alert(j.help);
}

function saveUiState() {
  try {
    localStorage.setItem("cdi_explorer_ui_state", JSON.stringify(collect()));
  } catch (_e) {}
}

function restoreUiState() {
  try {
    const raw = localStorage.getItem("cdi_explorer_ui_state");
    if (!raw) return;
    const state = JSON.parse(raw);
    for (const k of fields) {
      if (!(k in state)) continue;
      const el = document.getElementById(k);
      if (!el) continue;
      if (el.type === "checkbox") el.checked = !!state[k];
      else el.value = String(state[k]);
    }
  } catch (_e) {}
}

async function tick() {
  const r = await fetch("/api/status");
  const j = await r.json();
  const statusEl = document.getElementById("status");
  statusEl.textContent = j.running ? "Running..." : "Idle";
  statusEl.className = j.running ? "status-running" : "status-idle";
  const log = document.getElementById("log");
  log.value = j.log || "";
  log.scrollTop = log.scrollHeight;

  const roiSweepEnabled = document.getElementById("roi_size_sweep").checked;
  document.getElementById("local_region_radius").disabled = roiSweepEnabled;
  const roiFields = document.getElementById("roi_sweep_fields");
  if (roiSweepEnabled) roiFields.classList.remove("is-hidden");
  else roiFields.classList.add("is-hidden");
  setMaskVisibility();
  updateChoiceButtons();
}
setInterval(tick, 500);
document.getElementById("phase_mask_type").addEventListener("change", setMaskVisibility);
document.getElementById("roi_size_sweep").addEventListener("change", tick);
for (const cb of document.querySelectorAll('.checks input[type="checkbox"]')) {
  cb.addEventListener("change", updateChoiceButtons);
}
for (const k of fields) {
  const el = document.getElementById(k);
  if (!el) continue;
  el.addEventListener("change", saveUiState);
  el.addEventListener("input", saveUiState);
}
const themeEl = document.getElementById("theme_preset");
themeEl.addEventListener("change", () => {
  document.body.setAttribute("data-theme", themeEl.value);
  try { localStorage.setItem("cdi_explorer_theme", themeEl.value); } catch (_e) {}
});
try {
  const theme = localStorage.getItem("cdi_explorer_theme") || "presentation";
  themeEl.value = theme;
  document.body.setAttribute("data-theme", theme);
} catch (_e) {
  document.body.setAttribute("data-theme", "presentation");
}
restoreUiState();
window.addEventListener("pagehide", () => {
  try {
    navigator.sendBeacon("/api/shutdown", new Blob([], {type: "application/octet-stream"}));
  } catch (_e) {}
});
tick();
</script>
</body>
</html>
"""


class Runner:
    def __init__(self) -> None:
        self.proc: subprocess.Popen[str] | None = None
        self.log: deque[str] = deque(maxlen=12000)
        self.lock = threading.Lock()

    def _append(self, text: str) -> None:
        with self.lock:
            self.log.append(text)

    def current_log(self) -> str:
        with self.lock:
            return "".join(self.log)

    def build_cmd(self, payload: dict) -> list[str]:
        cmd = [sys.executable, "-m", "coronagraph.cli", "--feature", "coc-planet-phase"]
        args_map = {
            "phase_mask_type": "--phase-mask-type",
            "roddier_mask_radius": "--roddier-mask-radius",
            "roddier_mask_phase": "--roddier-mask-phase",
            "vortex_charge": "--vortex-charge",
            "spider_width": "--spider-width",
            "pupil_ss": "--pupil-ss",
            "local_region_radius": "--local-region-radius",
            "phase_sweep_mode": "--phase-sweep-mode",
            "region_shape": "--region-shape",
            "fov_count": "--fov-count",
            "fov_centers_count": "--fov-centers-count",
            "phase_step": "--phase-step",
            "phase_cycles": "--phase-cycles",
            "planet_offset_x_local": "--planet-offset-x-local",
            "planet_offset_y_local": "--planet-offset-y-local",
            "secondary_ratio_local": "--secondary-ratio-local",
            "planet_flux_ratio_local": "--planet-flux-ratio-local",
            "roi_size_min": "--roi-size-min",
            "roi_size_max": "--roi-size-max",
            "roi_size_step": "--roi-size-step",
        }
        for key, flag in args_map.items():
            cmd.extend([flag, str(payload.get(key, "")).strip()])

        if bool(payload.get("roi_size_sweep", False)):
            cmd.append("--roi-size-sweep")

        spider_angles = str(payload.get("spider_angles", "")).strip().split()
        if spider_angles:
            cmd.extend(["--spider-angles", *spider_angles])

        ring_radius = str(payload.get("single_region_ring_radius", "")).strip()
        if ring_radius:
            cmd.extend(["--single-region-ring-radius", ring_radius])

        for key, flag in [
            ("disable_ghost", "--disable-ghost"),
            ("disable_interference", "--disable-interference"),
            ("disable_companion_ghost", "--disable-companion-ghost"),
            ("build_map_per_fov", "--build-map-per-fov"),
        ]:
            if bool(payload.get(key, False)):
                cmd.append(flag)
        return cmd

    def start(self, payload: dict) -> tuple[bool, str | None]:
        if self.proc is not None:
            return False, "A run is already in progress."
        try:
            cmd = self.build_cmd(payload)
            self._append(f"$ {' '.join(cmd)}\n")
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            self.proc = None
            return False, str(exc)
        threading.Thread(target=self._stream_output, daemon=True).start()
        return True, None

    def _stream_output(self) -> None:
        proc = self.proc
        if proc is None:
            return
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._append(line)
            rc = proc.wait()
            self._append(f"\nProcess finished with code {rc}\n")
        except Exception as exc:
            self._append(f"\nFailed to run: {exc}\n")
        finally:
            self.proc = None

    def stop(self) -> None:
        if self.proc is None:
            self._append("No running process.\n")
            return
        self.proc.terminate()
        self._append("Termination requested.\n")

    def status(self) -> dict:
        return {"running": self.proc is not None, "log": self.current_log()}


def _json_response(handler: BaseHTTPRequestHandler, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, html: str) -> None:
    body = html.encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def main() -> None:
    runner = Runner()
    shutdown_event = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                _html_response(self, HTML)
                return
            if self.path == "/api/status":
                _json_response(self, runner.status())
                return
            if self.path == "/api/help":
                _json_response(self, {"help": HELP_TEXT})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path == "/api/run":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                except Exception as exc:
                    _json_response(self, {"ok": False, "error": f"Invalid payload: {exc}"}, HTTPStatus.BAD_REQUEST)
                    return
                ok, err = runner.start(payload)
                _json_response(self, {"ok": ok, "error": err})
                return
            if self.path == "/api/stop":
                runner.stop()
                _json_response(self, {"ok": True})
                return
            if self.path == "/api/shutdown":
                runner.stop()
                _json_response(self, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                shutdown_event.set()
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, _format: str, *_args) -> None:
            return

    host, port = "127.0.0.1", 8765
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Coronagraph HTML GUI running at {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if not shutdown_event.is_set():
            # Give pending unload beacon a short chance to arrive.
            time.sleep(0.2)
        runner.stop()
        server.server_close()


if __name__ == "__main__":
    main()
