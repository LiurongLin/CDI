from __future__ import annotations

import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .interactive_slm_gui import SLMLyotResponseCalculator, save_slm_lyot_result


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Interactive SLM Lyot Diagnostic</title>
  <style>
    :root {
      --bg: #f5f6f8;
      --panel: #ffffff;
      --line: #d7dce2;
      --text: #18202a;
      --muted: #5b6673;
      --red: #d7191c;
      --blue: #1d70b8;
      --yellow: #c28a00;
      --cyan: #007f91;
      --magenta: #9a3aa5;
      --green: #168245;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .app {
      display: grid;
      grid-template-columns: minmax(340px, 380px) minmax(420px, 0.95fr) minmax(540px, 1.15fr);
      gap: 14px;
      min-height: 100vh;
      padding: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    .controls {
      max-height: calc(100vh - 28px);
      overflow-y: auto;
    }
    h1, h2 {
      margin: 0 0 12px;
      line-height: 1.15;
    }
    h1 { font-size: 20px; }
    h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
    label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 5px; }
    .field { margin-bottom: 10px; }
    .control-group {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 12px;
      margin-bottom: 12px;
    }
    .run-group {
      position: sticky;
      top: 0;
      z-index: 2;
      box-shadow: 0 3px 10px rgba(24, 32, 42, 0.08);
    }
    .control-group h2 { margin-bottom: 10px; }
    .control-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .control-grid .full { grid-column: 1 / -1; }
    .compact-row {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .compact-row > * { flex: 1 1 120px; }
    .checkrow {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
      font-size: 14px;
    }
    .inline-checks {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
      margin-bottom: 10px;
    }
    .inline-checks .checkrow { margin-bottom: 0; }
    input[type="number"], select {
      width: 100%;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 5px 8px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    button {
      min-height: 36px;
      border: 1px solid #b9c0c9;
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
      cursor: pointer;
      padding: 6px 10px;
      line-height: 1.15;
    }
    button.primary {
      background: var(--blue);
      border-color: var(--blue);
      color: #fff;
    }
    button:disabled { opacity: 0.55; cursor: default; }
    button.icon-button {
      width: 34px;
      padding: 0;
      line-height: 1;
    }
    .button-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .button-grid.three {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
    .button-grid .wide {
      grid-column: 1 / -1;
      height: 40px;
      font-weight: 650;
    }
    .button-grid .danger {
      border-color: #b84a4a;
      color: #9b1c1c;
    }
    .param-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
      padding: 10px;
      margin-bottom: 12px;
    }
    .param-card-title {
      font-size: 13px;
      font-weight: 650;
      margin-bottom: 8px;
      color: var(--text);
    }
    .param-row {
      display: grid;
      grid-template-columns: 118px minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }
    .param-row label {
      margin: 0;
      color: var(--text);
      font-size: 12px;
    }
    .param-row input {
      height: 30px;
    }
    .source-geometry {
      border-top: 1px solid var(--line);
      padding-top: 10px;
      margin-top: 10px;
    }
    .speckle-row {
      display: grid;
      grid-template-columns: 24px 1fr 1fr 34px;
      gap: 6px;
      align-items: center;
      margin-bottom: 6px;
    }
    .speckle-row span {
      font-size: 12px;
      color: var(--muted);
    }
    .shape-control[hidden] {
      display: none;
    }
    .canvas-card {
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 8px;
      min-height: 0;
    }
    .canvas-wrap {
      position: relative;
      width: 100%;
      aspect-ratio: 1 / 1;
      min-height: 360px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f4f4f4;
      overflow: hidden;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
    }
    .legend {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px 12px;
      font-size: 13px;
      color: var(--muted);
    }
    .swatch {
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      margin-right: 6px;
      vertical-align: -1px;
    }
    .map-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .map {
      min-width: 0;
    }
    .map-title {
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 5px;
    }
    .map-canvas {
      aspect-ratio: 1 / 1;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #111820;
      overflow: hidden;
    }
    .plot-canvas {
      height: 260px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
      margin-top: 12px;
    }
    .plot-canvas.short { height: 210px; }
    .plots-panel {
      display: grid;
      gap: 14px;
      align-content: start;
    }
    .comparison-summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }
    .comparison-summary .metric {
      background: #fff;
      border-color: #c7d7e8;
    }
    .comparison-summary .metric strong {
      font-size: 16px;
    }
    .results {
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-width: 0;
      background: #fbfcfd;
    }
    .metric span { display: block; font-size: 12px; color: var(--muted); }
    .metric strong { display: block; font-size: 14px; overflow-wrap: anywhere; }
    .status {
      margin-top: 10px;
      min-height: 34px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      font-size: 13px;
      color: var(--muted);
    }
    .axis-note {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.35;
    }
    details.advanced {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px 12px;
      margin-bottom: 12px;
    }
    details.advanced summary {
      cursor: pointer;
      font-size: 13px;
      font-weight: 650;
      color: var(--text);
    }
    details.advanced .advanced-body {
      margin-top: 12px;
    }
    @media (max-width: 1200px) {
      .app { grid-template-columns: 360px minmax(420px, 1fr); }
      .right { grid-column: 1 / -1; }
      .controls { max-height: none; }
    }
    @media (max-width: 760px) {
      .app { grid-template-columns: 1fr; }
      .canvas-wrap { min-height: 280px; }
      .map-grid { grid-template-columns: 1fr; }
      .control-grid, .comparison-summary, .inline-checks { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="panel controls">
      <h1>SLM Lyot Diagnostic</h1>

      <div class="control-group run-group">
        <h2>Run</h2>
        <div class="button-grid">
          <button class="primary wide" id="runComparisonBtn">Run comparison</button>
          <button id="runSpeckleSweepBtn">Run speckle sweep</button>
          <button id="runPlanetSweepBtn">Run planet sweep</button>
          <button class="danger" id="stopBtn">Stop</button>
          <button id="resetPhaseBtn">Reset phase</button>
        </div>
        <div id="status" class="status">Ready.</div>
      </div>

      <div class="control-group">
        <h2>Sources</h2>
        <div class="inline-checks">
          <label class="checkrow"><input id="starEnabled" type="checkbox" checked /> Star</label>
          <label class="checkrow"><input id="planetEnabled" type="checkbox" checked /> Planet</label>
          <label class="checkrow"><input id="specklesEnabled" type="checkbox" /> Speckles</label>
        </div>
        <div class="control-grid">
          <div class="field">
            <label for="planetX">Planet X [lambda/D]</label>
            <input id="planetX" type="number" step="0.1" value="4.0" />
          </div>
          <div class="field">
            <label for="planetY">Planet Y [lambda/D]</label>
            <input id="planetY" type="number" step="0.1" value="0.0" />
          </div>
          <div class="field">
            <label for="targetSpeckle">Target speckle</label>
            <select id="targetSpeckle"></select>
          </div>
          <div class="field">
            <label for="starPlanetRatio">Star / planet</label>
            <input id="starPlanetRatio" type="number" min="1e-12" step="1" value="500" />
          </div>
          <div class="field">
            <label for="starSpeckleRatio">Star / speckle</label>
            <input id="starSpeckleRatio" type="number" min="1e-12" step="1" value="100" />
          </div>
          <label class="checkrow full"><input id="matchPhaseZeroPower" type="checkbox" /> Match phase-0 coherent and incoherent power</label>
          <div class="field">
            <label>Speckles used</label>
            <div id="speckleSelector" class="axis-note">Speckle list loads with the SLM grid.</div>
          </div>
        </div>
        <div class="source-geometry">
          <div class="param-card-title">Edit speckle positions</div>
          <div id="speckleEditor"></div>
          <div class="button-grid">
            <button id="addSpeckleBtn">Add Speckle</button>
            <button id="updateSourcesBtn">Update Sources</button>
          </div>
        </div>
      </div>

      <div class="control-group">
        <h2>Mask</h2>
        <div class="field">
          <label for="tool">Shape</label>
          <select id="tool">
            <option value="freehand">Freehand pixels</option>
            <option value="circle">Circle</option>
            <option value="ring">Ring / Annulus</option>
          </select>
        </div>
        <div class="param-card" id="shapeParams">
          <div class="param-card-title">Shape parameters</div>
          <div class="param-row shape-control shape-center">
            <label for="centerX">Center X [px]</label>
            <input id="centerX" type="number" min="0" step="1" value="0" />
          </div>
          <div class="param-row shape-control shape-center">
            <label for="centerY">Center Y [px]</label>
            <input id="centerY" type="number" min="0" step="1" value="0" />
          </div>
          <div class="param-row shape-control shape-circle">
            <label for="radiusPx">Radius [px]</label>
            <input id="radiusPx" type="number" min="0" step="1" value="30" />
          </div>
          <div class="param-row shape-control shape-ring">
            <label for="innerRadiusPx">Inner [px]</label>
            <input id="innerRadiusPx" type="number" min="0" step="1" value="20" />
          </div>
          <div class="param-row shape-control shape-ring">
            <label for="outerRadiusPx">Outer [px]</label>
            <input id="outerRadiusPx" type="number" min="1" step="1" value="40" />
          </div>
          <div class="button-grid shape-control shape-center">
            <button id="addRegionBtn">Add Region</button>
            <button id="replaceRegionBtn">Replace Mask</button>
          </div>
          <div class="axis-note" id="shapeHelp"></div>
        </div>
        <div class="button-grid three">
          <button id="clearBtn">Clear</button>
          <button id="undoBtn">Undo</button>
          <button id="loadMaskBtn">Load Mask</button>
          <button id="saveMaskBtn">Save Mask</button>
        </div>
      </div>

      <div class="control-group">
        <h2>Phase sweep</h2>
        <div class="control-grid">
          <div class="field">
            <label for="phaseSteps">Steps</label>
            <input id="phaseSteps" type="number" min="2" step="1" value="4" />
          </div>
          <div class="field">
            <label for="phaseStartRad">Start [rad]</label>
            <input id="phaseStartRad" type="number" step="0.01" value="0" />
          </div>
          <div class="field full">
            <label for="phaseSpanRad">Span [rad]</label>
            <input id="phaseSpanRad" type="number" step="0.01" value="6.283185307179586" />
            <div class="axis-note">Full cycle. Endpoint is not duplicated.</div>
          </div>
        </div>
      </div>

      <div class="control-group">
        <h2>Optimization</h2>
        <div class="control-grid">
          <div class="field">
            <label for="searchRadiusPx">Search radius [px]</label>
            <input id="searchRadiusPx" type="number" min="1" step="1" value="24" />
          </div>
          <div class="field">
            <label for="optIterations">Candidates</label>
            <input id="optIterations" type="number" min="1" step="1" value="12" />
          </div>
          <div class="field">
            <label for="groupSizePx">Pixel group [px]</label>
            <input id="groupSizePx" type="number" min="1" step="1" value="4" />
          </div>
          <div class="field">
            <label for="planetPenalty">Penalty</label>
            <input id="planetPenalty" type="number" min="0" step="0.1" value="1" />
          </div>
        </div>
        <button class="primary" id="optimizeMaskBtn" style="width:100%;">Start optimization</button>
        <div class="axis-note">Circle masks optimize radius. Other masks use pixel groups. Ratio score selects the best candidate.</div>
      </div>

      <div class="control-group">
        <h2>Results</h2>
        <div class="button-grid">
          <button id="saveResultBtn">Save results</button>
          <button id="saveLogBtn">Save log</button>
        </div>
      </div>

      <details class="advanced">
        <summary>Advanced settings</summary>
        <div class="advanced-body">
          <div class="field">
            <label for="phaseModulationRad">Single phase [rad]</label>
            <input id="phaseModulationRad" type="number" step="0.01" value="3.141592653589793" />
          </div>
          <div class="field">
            <label for="referenceSubtractPercent">Reference subtraction [%]</label>
            <input id="referenceSubtractPercent" type="number" min="0" step="1" value="100" />
          </div>
          <div class="field">
            <label for="subtractionMode">Subtraction model</label>
            <select id="subtractionMode">
              <option value="field">Electric field subtraction</option>
              <option value="intensity">Intensity subtraction</option>
            </select>
          </div>
          <div class="field">
            <label for="displayMode">Lyot map quantity</label>
            <select id="displayMode">
              <option value="abs">abs(Delta E_L)</option>
              <option value="phase">phase(Delta E_L)</option>
              <option value="real">real(Delta E_L)</option>
              <option value="imag">imag(Delta E_L)</option>
            </select>
          </div>
          <div class="button-grid">
            <button id="applyBtn">Apply single phase</button>
            <button id="evaluateModulationBtn">Evaluate current mask</button>
          </div>
        </div>
      </details>
      <input id="maskFile" type="file" accept=".json" hidden />
    </section>

    <section class="panel canvas-card">
      <h2>Focal-plane SLM mask</h2>
      <div class="canvas-wrap">
        <canvas id="slmCanvas"></canvas>
      </div>
      <div class="legend">
        <div><span class="swatch" style="background:#d7191c"></span>pi SLM pixels</div>
        <div><span class="swatch" style="background:#168245"></span>speckle placement</div>
        <div><span class="swatch" style="background:#1d70b8"></span>planet placement</div>
        <div><span class="swatch" style="background:#c28a00"></span>star</div>
        <div><span class="swatch" style="background:#00a5b8"></span>planet</div>
        <div><span class="swatch" style="background:#b24bbd"></span>coherent speckles</div>
      </div>
      <div class="axis-note">SLM coordinates are focal-plane lambda/D. The source markers show where each source PSF is centered on this focal grid.</div>
    </section>

    <section class="panel right plots-panel">
      <h2>Results</h2>
      <div id="comparisonSummary" class="comparison-summary">
        <div class="metric"><span>Speckle peak-to-peak modulation</span><strong>--</strong></div>
        <div class="metric"><span>Planet peak-to-peak modulation</span><strong>--</strong></div>
        <div class="metric"><span>Modulation ratio</span><strong>--</strong></div>
      </div>
      <h2>Lyot-plane response</h2>
      <div class="map-grid">
        <div class="map"><div class="map-title">Star</div><div class="map-canvas"><canvas id="starMap"></canvas></div></div>
        <div class="map"><div class="map-title">Planet</div><div class="map-canvas"><canvas id="planetMap"></canvas></div></div>
        <div class="map"><div class="map-title">Speckles</div><div class="map-canvas"><canvas id="speckleMap"></canvas></div></div>
        <div class="map"><div class="map-title">Coherent star + speckles</div><div class="map-canvas"><canvas id="coherentMap"></canvas></div></div>
        <div class="map"><div class="map-title">Incoherent star + planet amplitude</div><div class="map-canvas"><canvas id="incoherentMap"></canvas></div></div>
      </div>
      <div class="map-title">Current mask response, integrated Lyot-stop power</div>
      <div class="plot-canvas"><canvas id="phasePlot"></canvas></div>
      <div class="map-grid">
        <div>
          <div class="map-title">Speckle-position sweep: power vs phase [rad]</div>
          <div class="plot-canvas short"><canvas id="speckleSweepPlot"></canvas></div>
        </div>
        <div>
          <div class="map-title">Planet-position sweep: power vs phase [rad]</div>
          <div class="plot-canvas short"><canvas id="planetSweepPlot"></canvas></div>
        </div>
      </div>
      <div class="map-title">Optimization trade-off</div>
      <div class="plot-canvas short"><canvas id="tradeoffPlot"></canvas></div>
      <div id="metrics" class="results"></div>
    </section>
  </main>

  <script>
    const state = {
      config: null,
      mask: null,
      baseMask: null,
      previewMask: null,
      undo: [],
      dragging: false,
      circleStart: null,
      lastPixel: null,
      lastResult: null,
      lastSweep: null,
      lastComparison: null,
      lastComparisonSignature: null,
      lastOptimization: null,
      placedMasks: null,
      pollTimer: null,
    };

    const els = {
      slm: document.getElementById("slmCanvas"),
      status: document.getElementById("status"),
      tool: document.getElementById("tool"),
      displayMode: document.getElementById("displayMode"),
      applyBtn: document.getElementById("applyBtn"),
      evaluateModulationBtn: document.getElementById("evaluateModulationBtn"),
      runSpeckleSweepBtn: document.getElementById("runSpeckleSweepBtn"),
      runPlanetSweepBtn: document.getElementById("runPlanetSweepBtn"),
      runComparisonBtn: document.getElementById("runComparisonBtn"),
      optimizeMaskBtn: document.getElementById("optimizeMaskBtn"),
      stopBtn: document.getElementById("stopBtn"),
      resetPhaseBtn: document.getElementById("resetPhaseBtn"),
      clearBtn: document.getElementById("clearBtn"),
      undoBtn: document.getElementById("undoBtn"),
      saveMaskBtn: document.getElementById("saveMaskBtn"),
      loadMaskBtn: document.getElementById("loadMaskBtn"),
      saveResultBtn: document.getElementById("saveResultBtn"),
      saveLogBtn: document.getElementById("saveLogBtn"),
      addRegionBtn: document.getElementById("addRegionBtn"),
      replaceRegionBtn: document.getElementById("replaceRegionBtn"),
      planetX: document.getElementById("planetX"),
      planetY: document.getElementById("planetY"),
      speckleEditor: document.getElementById("speckleEditor"),
      addSpeckleBtn: document.getElementById("addSpeckleBtn"),
      updateSourcesBtn: document.getElementById("updateSourcesBtn"),
      phaseModulationRad: document.getElementById("phaseModulationRad"),
      referenceSubtractPercent: document.getElementById("referenceSubtractPercent"),
      phaseSteps: document.getElementById("phaseSteps"),
      phaseStartRad: document.getElementById("phaseStartRad"),
      phaseSpanRad: document.getElementById("phaseSpanRad"),
      subtractionMode: document.getElementById("subtractionMode"),
      speckleSelector: document.getElementById("speckleSelector"),
      targetSpeckle: document.getElementById("targetSpeckle"),
      searchRadiusPx: document.getElementById("searchRadiusPx"),
      groupSizePx: document.getElementById("groupSizePx"),
      optIterations: document.getElementById("optIterations"),
      planetPenalty: document.getElementById("planetPenalty"),
      matchPhaseZeroPower: document.getElementById("matchPhaseZeroPower"),
      maskFile: document.getElementById("maskFile"),
      shapeParams: document.getElementById("shapeParams"),
      shapeHelp: document.getElementById("shapeHelp"),
      centerX: document.getElementById("centerX"),
      centerY: document.getElementById("centerY"),
      radiusPx: document.getElementById("radiusPx"),
      innerRadiusPx: document.getElementById("innerRadiusPx"),
      outerRadiusPx: document.getElementById("outerRadiusPx"),
      metrics: document.getElementById("metrics"),
      comparisonSummary: document.getElementById("comparisonSummary"),
      phasePlot: document.getElementById("phasePlot"),
      speckleSweepPlot: document.getElementById("speckleSweepPlot"),
      planetSweepPlot: document.getElementById("planetSweepPlot"),
      tradeoffPlot: document.getElementById("tradeoffPlot"),
      maps: {
        star: document.getElementById("starMap"),
        planet: document.getElementById("planetMap"),
        speckle: document.getElementById("speckleMap"),
        coherent: document.getElementById("coherentMap"),
        incoherent: document.getElementById("incoherentMap"),
      }
    };

    function setStatus(text) { els.status.textContent = text || ""; }

    function resizeCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(1, Math.round(rect.width * dpr));
      const h = Math.max(1, Math.round(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
    }

    function slmPixelToCanvas(x, y) {
      const cfg = state.config;
      const minX = cfg.slm_extent[0], maxX = cfg.slm_extent[1];
      const minY = cfg.slm_extent[2], maxY = cfg.slm_extent[3];
      const lamX = (x - (cfg.n_fft - 1) / 2) / cfg.focal_sampling;
      const lamY = (y - (cfg.n_fft - 1) / 2) / cfg.focal_sampling;
      return [
        (lamX - minX) / (maxX - minX) * els.slm.width,
        els.slm.height - (lamY - minY) / (maxY - minY) * els.slm.height
      ];
    }

    function canvasToSlmPixel(event) {
      const rect = els.slm.getBoundingClientRect();
      const cfg = state.config;
      const u = (event.clientX - rect.left) / rect.width;
      const v = (event.clientY - rect.top) / rect.height;
      const lamX = cfg.slm_extent[0] + u * (cfg.slm_extent[1] - cfg.slm_extent[0]);
      const lamY = cfg.slm_extent[3] - v * (cfg.slm_extent[3] - cfg.slm_extent[2]);
      const x = Math.round(lamX * cfg.focal_sampling + (cfg.n_fft - 1) / 2);
      const y = Math.round(lamY * cfg.focal_sampling + (cfg.n_fft - 1) / 2);
      if (x < 0 || y < 0 || x >= cfg.n_fft || y >= cfg.n_fft) return null;
      return [x, y];
    }

    function drawLinePixels(a, b) {
      const [x0, y0] = a, [x1, y1] = b;
      const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
      const n = state.config.n_fft;
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const x = Math.round(x0 + (x1 - x0) * t);
        const y = Math.round(y0 + (y1 - y0) * t);
        if (x >= 0 && y >= 0 && x < n && y < n) state.mask[y * n + x] = 1;
      }
    }

    function commitCurrentMaskAsBase() {
      state.baseMask = state.mask ? state.mask.slice() : new Uint8Array(state.config.n_fft * state.config.n_fft);
      state.previewMask = null;
      markResultsOutdated();
    }

    function drawCirclePixels(start, end) {
      const n = state.config.n_fft;
      const [x0, y0] = start, [x1, y1] = end;
      const r = Math.round(Math.hypot(x1 - x0, y1 - y0));
      setShapeFields(x0, y0, r, null, null);
      applyParameterizedShape();
    }

    function clampInt(value, minValue, maxValue) {
      const parsed = Number(value);
      if (!Number.isFinite(parsed)) return null;
      return Math.max(minValue, Math.min(maxValue, Math.round(parsed)));
    }

    function setShapeFields(cx, cy, radius, inner, outer) {
      const n = state.config.n_fft;
      if (cx !== null) els.centerX.value = String(Math.max(0, Math.min(n - 1, Math.round(cx))));
      if (cy !== null) els.centerY.value = String(Math.max(0, Math.min(n - 1, Math.round(cy))));
      if (radius !== null) els.radiusPx.value = String(Math.max(0, Math.round(radius)));
      if (inner !== null) els.innerRadiusPx.value = String(Math.max(0, Math.round(inner)));
      if (outer !== null) els.outerRadiusPx.value = String(Math.max(0, Math.round(outer)));
    }

    function updateShapeParameterVisibility() {
      const tool = els.tool.value;
      const showShape = tool === "circle" || tool === "ring";
      els.shapeParams.hidden = false;
      for (const node of document.querySelectorAll(".shape-control")) node.hidden = true;
      if (tool === "freehand") {
        els.shapeHelp.textContent = "Freehand uses the painted mask directly. Shape center/radius controls are inactive in this mode.";
        return;
      }
      for (const node of document.querySelectorAll(".shape-center")) node.hidden = false;
      if (tool === "circle") {
        for (const node of document.querySelectorAll(".shape-circle")) node.hidden = false;
        els.shapeHelp.textContent = "Circle mask: pixels with distance from center <= radius are set to pi.";
      } else if (tool === "ring") {
        for (const node of document.querySelectorAll(".shape-ring")) node.hidden = false;
        els.shapeHelp.textContent = "Ring mask: pixels with inner radius <= distance from center <= outer radius are set to pi.";
      }
      if (!showShape) els.shapeHelp.textContent = "";
    }

    function shapeParams() {
      const n = state.config.n_fft;
      const cx = clampInt(els.centerX.value, 0, n - 1);
      const cy = clampInt(els.centerY.value, 0, n - 1);
      const radius = clampInt(els.radiusPx.value, 0, n);
      const inner = clampInt(els.innerRadiusPx.value, 0, n);
      const outer = clampInt(els.outerRadiusPx.value, 0, n);
      if (cx === null || cy === null || radius === null || inner === null || outer === null) {
        throw new Error("Shape parameters must be numeric.");
      }
      if (els.tool.value === "ring" && outer <= inner) {
        throw new Error("Ring outer radius must be greater than inner radius.");
      }
      return {cx, cy, radius, inner, outer};
    }

    function buildParameterizedShapeMask() {
      const n = state.config.n_fft;
      const p = shapeParams();
      const next = new Uint8Array(n * n);
      const isRing = els.tool.value === "ring";
      const rMin2 = p.inner * p.inner;
      const rMax = isRing ? p.outer : p.radius;
      const rMax2 = rMax * rMax;
      const xmin = Math.max(0, p.cx - rMax), xmax = Math.min(n - 1, p.cx + rMax);
      const ymin = Math.max(0, p.cy - rMax), ymax = Math.min(n - 1, p.cy + rMax);
      for (let y = ymin; y <= ymax; y++) {
        for (let x = xmin; x <= xmax; x++) {
          const d2 = (x - p.cx) * (x - p.cx) + (y - p.cy) * (y - p.cy);
          if (isRing ? (d2 >= rMin2 && d2 <= rMax2) : (d2 <= rMax2)) next[y * n + x] = 1;
        }
      }
      return next;
    }

    function refreshMaskFromBaseAndPreview() {
      const n = state.config.n_fft * state.config.n_fft;
      const composed = state.baseMask ? state.baseMask.slice() : new Uint8Array(n);
      if (state.previewMask) {
        for (let i = 0; i < n; i++) {
          if (state.previewMask[i]) composed[i] = 1;
        }
      }
      state.mask = composed;
      redrawSlm();
      markResultsOutdated();
    }

    function applyParameterizedShape() {
      if (els.tool.value === "freehand") {
        setStatus("Freehand mode: draw directly on the SLM panel. Numeric shape parameters are inactive.");
        return;
      }
      try {
        if (!state.baseMask) state.baseMask = state.mask ? state.mask.slice() : null;
        state.previewMask = buildParameterizedShapeMask();
        refreshMaskFromBaseAndPreview();
        setStatus(`Shape preview updated: ${maskCount()} pixels selected. Press Apply to propagate.`);
      } catch (err) {
        setStatus(String(err.message || err));
      }
    }

    function commitPreviewRegion() {
      if (!state.previewMask) {
        applyParameterizedShape();
      }
      if (!state.previewMask) return;
      state.undo.push(state.baseMask ? state.baseMask.slice() : state.mask.slice());
      if (!state.baseMask) state.baseMask = new Uint8Array(state.config.n_fft * state.config.n_fft);
      for (let i = 0; i < state.baseMask.length; i++) {
        if (state.previewMask[i]) state.baseMask[i] = 1;
      }
      state.previewMask = null;
      state.mask = state.baseMask.slice();
      redrawSlm();
      setStatus(`Region added: ${maskCount()} pixels selected. Press Apply to propagate.`);
    }

    function replaceMaskWithPreview() {
      if (!state.previewMask) {
        applyParameterizedShape();
      }
      if (!state.previewMask) return;
      state.undo.push(state.mask.slice());
      state.baseMask = state.previewMask.slice();
      state.previewMask = null;
      state.mask = state.baseMask.slice();
      redrawSlm();
      setStatus(`Mask replaced: ${maskCount()} pixels selected. Press Apply to propagate.`);
    }

    function maskCount() {
      let count = 0;
      for (const v of state.mask) count += v ? 1 : 0;
      return count;
    }

    function drawMarker(ctx, marker) {
      const cfg = state.config;
      const [minX, maxX, minY, maxY] = cfg.slm_extent;
      const x = (marker.x - minX) / (maxX - minX) * els.slm.width;
      const y = els.slm.height - (marker.y - minY) / (maxY - minY) * els.slm.height;
      if (x < 0 || y < 0 || x > els.slm.width || y > els.slm.height) return;
      ctx.save();
      ctx.strokeStyle = marker.color;
      ctx.fillStyle = marker.color;
      ctx.lineWidth = Math.max(2, els.slm.width / 280);
      ctx.beginPath();
      ctx.arc(x, y, Math.max(5, els.slm.width / 95), 0, 2 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x - 8, y);
      ctx.lineTo(x + 8, y);
      ctx.moveTo(x, y - 8);
      ctx.lineTo(x, y + 8);
      ctx.stroke();
      ctx.font = `${Math.max(11, Math.round(els.slm.width / 46))}px system-ui`;
      ctx.fillText(marker.label, x + 10, y - 10);
      ctx.restore();
    }

    function speckleMarkers() {
      return (state.config.source_markers || []).filter(marker => marker.kind === "speckle");
    }

    function renderSpeckleSelector() {
      const markers = speckleMarkers();
      els.speckleSelector.innerHTML = "";
      els.targetSpeckle.innerHTML = "";
      if (!markers.length) {
        els.speckleSelector.textContent = "No coherent speckles are configured.";
        const option = document.createElement("option");
        option.value = "0";
        option.textContent = "No speckles";
        els.targetSpeckle.appendChild(option);
        return;
      }
      for (const marker of markers) {
        const row = document.createElement("label");
        row.className = "checkrow";
        row.style.marginBottom = "4px";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.className = "speckle-choice";
        checkbox.value = String(marker.index);
        checkbox.checked = true;
        const text = document.createElement("span");
        text.textContent = `${marker.label}: (${marker.x.toFixed(2)}, ${marker.y.toFixed(2)}) lambda/D`;
        row.appendChild(checkbox);
        row.appendChild(text);
        els.speckleSelector.appendChild(row);
        const option = document.createElement("option");
        option.value = String(marker.index);
        option.textContent = `${marker.label}: (${marker.x.toFixed(2)}, ${marker.y.toFixed(2)})`;
        els.targetSpeckle.appendChild(option);
      }
    }

    function renderSourceGeometryControls() {
      const planet = (state.config.source_markers || []).find(marker => marker.kind === "planet");
      if (planet) {
        els.planetX.value = String(planet.x);
        els.planetY.value = String(planet.y);
      }
      els.speckleEditor.innerHTML = "";
      const markers = speckleMarkers();
      for (const marker of markers) {
        addSpeckleEditorRow(marker.x, marker.y);
      }
      updateSpeckleRowLabels();
    }

    function updateSpeckleRowLabels() {
      const rows = Array.from(els.speckleEditor.querySelectorAll(".speckle-row"));
      rows.forEach((row, index) => {
        const label = `s${index + 1}`;
        row.querySelector(".speckle-label").textContent = label;
        row.querySelector(".speckle-x").setAttribute("aria-label", `Speckle ${index + 1} X`);
        row.querySelector(".speckle-y").setAttribute("aria-label", `Speckle ${index + 1} Y`);
        const removeButton = row.querySelector(".remove-speckle");
        removeButton.setAttribute("aria-label", `Remove speckle ${index + 1}`);
        removeButton.title = `Remove ${label}`;
      });
    }

    function addSpeckleEditorRow(x = 0, y = 0) {
      const idx = els.speckleEditor.querySelectorAll(".speckle-row").length + 1;
      const row = document.createElement("div");
      row.className = "speckle-row";
      row.innerHTML = `
        <span class="speckle-label">s${idx}</span>
        <input class="speckle-x" type="number" step="0.1" value="${Number(x).toFixed(3)}" aria-label="Speckle ${idx} X" />
        <input class="speckle-y" type="number" step="0.1" value="${Number(y).toFixed(3)}" aria-label="Speckle ${idx} Y" />
        <button class="icon-button remove-speckle" type="button" aria-label="Remove speckle ${idx}" title="Remove s${idx}">x</button>
      `;
      row.querySelector(".remove-speckle").addEventListener("click", () => {
        row.remove();
        updateSpeckleRowLabels();
      });
      els.speckleEditor.appendChild(row);
      updateSpeckleRowLabels();
    }

    function readSpeckleGeometry() {
      const rows = Array.from(els.speckleEditor.querySelectorAll(".speckle-row"));
      return rows.map(row => [
        Number(row.querySelector(".speckle-x").value),
        Number(row.querySelector(".speckle-y").value),
      ]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
    }

    async function updateSourceGeometry() {
      try {
        const response = await fetch("/api/source-geometry", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            planet_x: Number(els.planetX.value),
            planet_y: Number(els.planetY.value),
            speckle_offsets: readSpeckleGeometry(),
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Source update failed");
        state.config = data;
        renderSpeckleSelector();
        renderSourceGeometryControls();
        redrawSlm();
        state.lastResult = null;
        state.lastComparison = null;
        state.lastComparisonSignature = null;
        state.lastOptimization = null;
        state.placedMasks = null;
        renderComparisonSummary(null, "Settings changed. Run comparison again.");
        setStatus("Source locations updated. Press Apply to propagate.");
      } catch (err) {
        setStatus(String(err.message || err));
      }
    }

    function selectedSpeckleIndices() {
      return Array.from(document.querySelectorAll(".speckle-choice"))
        .filter(input => input.checked)
        .map(input => Number(input.value));
    }

    function niceTicks(minValue, maxValue, targetCount) {
      const span = Math.max(Math.abs(maxValue - minValue), 1e-12);
      const rawStep = span / Math.max(1, targetCount);
      const pow10 = Math.pow(10, Math.floor(Math.log10(rawStep)));
      const scaled = rawStep / pow10;
      let nice = 1;
      if (scaled > 5) nice = 10;
      else if (scaled > 2) nice = 5;
      else if (scaled > 1) nice = 2;
      const step = nice * pow10;
      const first = Math.ceil(minValue / step) * step;
      const ticks = [];
      for (let v = first; v <= maxValue + 0.5 * step; v += step) {
        if (v >= minValue - 1e-9 && v <= maxValue + 1e-9) ticks.push(Math.abs(v) < 1e-10 ? 0 : v);
      }
      return ticks;
    }

    function formatTick(value) {
      const abs = Math.abs(value);
      if (abs >= 10 || abs === 0) return String(Math.round(value));
      if (abs >= 1) return value.toFixed(1).replace(/\.0$/, "");
      return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
    }

    function drawAxes(ctx, width, height, extent, options) {
      const cfg = Object.assign({
        color: "rgba(24,32,42,0.82)",
        gridColor: "rgba(24,32,42,0.13)",
        label: "",
        targetTicks: 5,
        labelBackground: "rgba(255,255,255,0.72)"
      }, options || {});
      const [xmin, xmax, ymin, ymax] = extent;
      const xTicks = niceTicks(xmin, xmax, cfg.targetTicks);
      const yTicks = niceTicks(ymin, ymax, cfg.targetTicks);
      ctx.save();
      ctx.lineWidth = 1;
      ctx.font = `${Math.max(10, Math.round(width / 34))}px system-ui`;
      ctx.textBaseline = "top";

      ctx.strokeStyle = cfg.gridColor;
      for (const tx of xTicks) {
        const x = (tx - xmin) / (xmax - xmin) * width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (const ty of yTicks) {
        const y = height - (ty - ymin) / (ymax - ymin) * height;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      ctx.strokeStyle = cfg.color;
      ctx.fillStyle = cfg.color;
      for (const tx of xTicks) {
        const x = (tx - xmin) / (xmax - xmin) * width;
        ctx.beginPath();
        ctx.moveTo(x, height - 7);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.fillText(formatTick(tx), Math.min(width - 28, Math.max(2, x + 3)), height - 20);
      }
      ctx.textBaseline = "middle";
      for (const ty of yTicks) {
        const y = height - (ty - ymin) / (ymax - ymin) * height;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(7, y);
        ctx.stroke();
        ctx.fillText(formatTick(ty), 9, Math.min(height - 10, Math.max(10, y)));
      }
      if (cfg.label) {
        const labelWidth = ctx.measureText(cfg.label).width + 12;
        ctx.fillStyle = cfg.labelBackground;
        ctx.fillRect(8, 8, labelWidth, 20);
        ctx.fillStyle = cfg.color;
        ctx.textBaseline = "top";
        ctx.fillText(cfg.label, 14, 11);
      }
      ctx.restore();
    }

    function redrawSlm() {
      if (!state.config) return;
      resizeCanvas(els.slm);
      const ctx = els.slm.getContext("2d");
      ctx.clearRect(0, 0, els.slm.width, els.slm.height);
      ctx.fillStyle = "#f4f4f4";
      ctx.fillRect(0, 0, els.slm.width, els.slm.height);

      const n = state.config.n_fft;
      const cellW = els.slm.width / n;
      const cellH = els.slm.height / n;
      ctx.fillStyle = "rgba(215,25,28,0.92)";
      for (let y = 0; y < n; y++) {
        const row = y * n;
        for (let x = 0; x < n; x++) {
          if (state.mask[row + x]) {
            const px = x * cellW;
            const py = els.slm.height - (y + 1) * cellH;
            ctx.fillRect(px, py, Math.max(1, cellW), Math.max(1, cellH));
          }
        }
      }

      ctx.strokeStyle = "#9aa3ad";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, els.slm.height / 2);
      ctx.lineTo(els.slm.width, els.slm.height / 2);
      ctx.moveTo(els.slm.width / 2, 0);
      ctx.lineTo(els.slm.width / 2, els.slm.height);
      ctx.stroke();

      drawAxes(ctx, els.slm.width, els.slm.height, state.config.slm_extent, {
        label: "lambda/D",
        targetTicks: 6
      });

      for (const marker of state.config.source_markers) drawMarker(ctx, marker);

      if (state.placedMasks) {
        const overlays = [
          {key: "speckle_position", color: "rgba(22,130,69,0.42)"},
          {key: "planet_position", color: "rgba(29,112,184,0.36)"},
        ];
        for (const overlay of overlays) {
          const mask = state.placedMasks[overlay.key];
          if (!mask) continue;
          ctx.fillStyle = overlay.color;
          for (let y = 0; y < n; y++) {
            const row = y * n;
            for (let x = 0; x < n; x++) {
              if (mask[row + x]) {
                const px = x * cellW;
                const py = els.slm.height - (y + 1) * cellH;
                ctx.fillRect(px, py, Math.max(1, cellW), Math.max(1, cellH));
              }
            }
          }
        }
      }

      ctx.fillStyle = "#18202a";
      ctx.font = `${Math.max(12, Math.round(els.slm.width / 48))}px system-ui`;
      const selectedText = `selected: ${maskCount()} px`;
      const selectedWidth = ctx.measureText(selectedText).width + 12;
      ctx.fillStyle = "rgba(255,255,255,0.76)";
      ctx.fillRect(8, els.slm.height - 32, selectedWidth, 22);
      ctx.fillStyle = "#18202a";
      ctx.fillText(selectedText, 14, els.slm.height - 16);
    }

    function packMask(mask) {
      let s = "";
      const chunk = 32768;
      for (let i = 0; i < mask.length; i += chunk) {
        s += String.fromCharCode(...mask.subarray(i, i + chunk));
      }
      return btoa(s);
    }

    function unpackBase64Mask() {
      return packMask(state.mask);
    }

    function decodeMaskB64(maskB64) {
      const raw = atob(maskB64 || "");
      const next = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) next[i] = raw.charCodeAt(i);
      return next;
    }

    function valueToColor(value, vmin, vmax, mode) {
      if (!Number.isFinite(value)) return [0, 0, 0];
      let t;
      if (mode === "phase") {
        t = (value + Math.PI) / (2 * Math.PI);
      } else {
        t = vmax === vmin ? 0.5 : (value - vmin) / (vmax - vmin);
      }
      t = Math.max(0, Math.min(1, t));
      const grey = Math.round(255 * t);
      return [grey, grey, grey];
    }

    function drawMap(canvas, payload, sharedScale) {
      resizeCanvas(canvas);
      const ctx = canvas.getContext("2d");
      const [h, w] = payload.shape;
      const mode = document.getElementById("displayMode").value;
      const data = payload.modes ? payload.modes[mode] : payload.values;
      let vmin = sharedScale ? sharedScale.vmin : Infinity;
      let vmax = sharedScale ? sharedScale.vmax : -Infinity;
      if (!sharedScale) {
        for (const v of data) {
          if (Number.isFinite(v)) {
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
          }
        }
      }
      if (!Number.isFinite(vmin) || !Number.isFinite(vmax)) { vmin = 0; vmax = 1; }
      const image = ctx.createImageData(w, h);
      for (let i = 0; i < data.length; i++) {
        const [r, g, b] = valueToColor(data[i], vmin, vmax, mode);
        const p = 4 * i;
        image.data[p] = r; image.data[p + 1] = g; image.data[p + 2] = b; image.data[p + 3] = 255;
      }
      const off = document.createElement("canvas");
      off.width = w; off.height = h;
      off.getContext("2d").putImageData(image, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, 0, 0, canvas.width, canvas.height);

      if (payload.lyot_stop) {
        ctx.fillStyle = "rgba(255,255,255,0.16)";
        const sx = canvas.width / w, sy = canvas.height / h;
        for (let y = 1; y < h - 1; y++) {
          for (let x = 1; x < w - 1; x++) {
            const idx = y * w + x;
            if (!payload.lyot_stop[idx]) continue;
            if (!payload.lyot_stop[idx - 1] || !payload.lyot_stop[idx + 1] || !payload.lyot_stop[idx - w] || !payload.lyot_stop[idx + w]) {
              ctx.fillRect(x * sx, y * sy, Math.max(1, sx), Math.max(1, sy));
            }
          }
        }
      }
      drawAxes(ctx, canvas.width, canvas.height, payload.extent, {
        color: "rgba(255,255,255,0.9)",
        gridColor: "rgba(255,255,255,0.16)",
        label: "pupil D",
        targetTicks: 4,
        labelBackground: "rgba(0,0,0,0.35)"
      });
    }

    function redrawMaps() {
      if (!state.lastResult) return;
      const mode = document.getElementById("displayMode").value;
      let vmin = Infinity, vmax = -Infinity;
      for (const key of Object.keys(els.maps)) {
        const payload = state.lastResult.maps[key];
        const data = payload.modes ? payload.modes[mode] : payload.values;
        if (mode === "phase") {
          vmin = -Math.PI;
          vmax = Math.PI;
          break;
        }
        for (const v of data) {
          if (Number.isFinite(v)) {
            if (v < vmin) vmin = v;
            if (v > vmax) vmax = v;
          }
        }
      }
      const sharedScale = {vmin, vmax};
      for (const key of Object.keys(els.maps)) {
        drawMap(els.maps[key], state.lastResult.maps[key], sharedScale);
      }
    }

    function drawPhasePlot(sweep) {
      resizeCanvas(els.phasePlot);
      const canvas = els.phasePlot;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (!sweep) return;

      const phases = sweep.phases;
      const series = [
        {key: "coherent", label: "coherent star + speckles", color: "#005ea8", width: 3.2, dash: [], marker: "circle", alpha: 1},
        {key: "incoherent", label: "incoherent star + planet", color: "#c51b29", width: 3.2, dash: [10, 5], marker: "square", alpha: 1},
        {key: "star", label: "star", color: "#9f6b00", width: 1.5, dash: [3, 4], marker: "triangle", alpha: 0.72},
        {key: "speckle", label: "speckles", color: "#7a3db8", width: 1.5, dash: [8, 4, 2, 4], marker: "diamond", alpha: 0.72},
        {key: "planet", label: "planet", color: "#007f91", width: 1.5, dash: [2, 3], marker: "cross", alpha: 0.72},
      ].filter(item => sweep.powers && sweep.powers[item.key]);
      let ymin = Infinity, ymax = -Infinity;
      const primary = series.filter(item => item.key === "coherent" || item.key === "incoherent");
      const scaleSeries = primary.length ? primary : series;
      for (const item of scaleSeries) {
        for (const value of sweep.powers[item.key]) {
          if (!Number.isFinite(value)) continue;
          ymin = Math.min(ymin, value);
          ymax = Math.max(ymax, value);
        }
      }
      if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) {
        ymin = 0;
        ymax = 1;
      }
      if (ymin === ymax) {
        ymin -= 0.5;
        ymax += 0.5;
      }
      const meanScale = Math.max(Math.abs(ymin), Math.abs(ymax), 1);
      if ((ymax - ymin) < meanScale * 1e-6) {
        const center = 0.5 * (ymin + ymax);
        const halfSpan = 0.5 * meanScale * 1e-6;
        ymin = center - halfSpan;
        ymax = center + halfSpan;
      }
      const pad = 0.08 * (ymax - ymin);
      ymin -= pad;
      ymax += pad;
      const margin = {
        left: Math.max(48, Math.round(canvas.width * 0.09)),
        right: 18,
        top: 18,
        bottom: 42
      };
      const plotW = canvas.width - margin.left - margin.right;
      const plotH = canvas.height - margin.top - margin.bottom;
      const xFor = phase => margin.left + (phase / (2 * Math.PI)) * plotW;
      const yFor = value => margin.top + (1 - (value - ymin) / (ymax - ymin)) * plotH;

      ctx.save();
      ctx.strokeStyle = "rgba(24,32,42,0.16)";
      ctx.fillStyle = "#5b6673";
      ctx.font = `${Math.max(11, Math.round(canvas.width / 62))}px system-ui`;
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const x = margin.left + (i / 4) * plotW;
        ctx.beginPath();
        ctx.moveTo(x, margin.top);
        ctx.lineTo(x, margin.top + plotH);
        ctx.stroke();
        const labels = ["0", "pi/2", "pi", "3pi/2", "2pi"];
        ctx.fillText(labels[i], x - 10, margin.top + plotH + 18);
      }
      for (let i = 0; i <= 4; i++) {
        const y = margin.top + (i / 4) * plotH;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left + plotW, y);
        ctx.stroke();
        const value = ymax - (i / 4) * (ymax - ymin);
        ctx.fillText(value.toExponential(2), 4, y - 6);
      }
      ctx.strokeStyle = "#18202a";
      ctx.beginPath();
      ctx.moveTo(margin.left, margin.top);
      ctx.lineTo(margin.left, margin.top + plotH);
      ctx.lineTo(margin.left + plotW, margin.top + plotH);
      ctx.stroke();

      function markerPath(x, y, marker, size) {
        ctx.beginPath();
        if (marker === "square") {
          ctx.rect(x - size, y - size, 2 * size, 2 * size);
        } else if (marker === "triangle") {
          ctx.moveTo(x, y - size);
          ctx.lineTo(x + size, y + size);
          ctx.lineTo(x - size, y + size);
          ctx.closePath();
        } else if (marker === "diamond") {
          ctx.moveTo(x, y - size);
          ctx.lineTo(x + size, y);
          ctx.lineTo(x, y + size);
          ctx.lineTo(x - size, y);
          ctx.closePath();
        } else if (marker === "cross") {
          ctx.moveTo(x - size, y - size);
          ctx.lineTo(x + size, y + size);
          ctx.moveTo(x + size, y - size);
          ctx.lineTo(x - size, y + size);
        } else {
          ctx.arc(x, y, size, 0, 2 * Math.PI);
        }
      }

      ctx.save();
      ctx.beginPath();
      ctx.rect(margin.left, margin.top, plotW, plotH);
      ctx.clip();
      for (const item of series) {
        const values = sweep.powers[item.key];
        ctx.strokeStyle = item.color;
        ctx.lineWidth = item.width;
        ctx.setLineDash(item.dash);
        ctx.globalAlpha = item.alpha;
        ctx.beginPath();
        values.forEach((value, idx) => {
          const x = xFor(phases[idx]);
          const y = yFor(value);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.lineTo(xFor(2 * Math.PI), yFor(values[0]));
        ctx.stroke();
        ctx.setLineDash([]);
        values.forEach((value, idx) => {
          const x = xFor(phases[idx]);
          const y = yFor(value);
          markerPath(x, y, item.marker, item.width > 2 ? 4.5 : 3.5);
          ctx.fillStyle = "#ffffff";
          ctx.fill();
          ctx.strokeStyle = item.color;
          ctx.lineWidth = item.width > 2 ? 1.8 : 1.2;
          ctx.stroke();
        });
        ctx.globalAlpha = 1;
      }
      ctx.restore();

      let legendX = margin.left + 8;
      let legendY = margin.top + 6;
      const legendWidth = Math.min(plotW - 16, 245);
      const legendHeight = series.length * 18 + 8;
      ctx.fillStyle = "rgba(255,255,255,0.86)";
      ctx.fillRect(legendX - 6, legendY - 11, legendWidth, legendHeight);
      ctx.strokeStyle = "rgba(24,32,42,0.18)";
      ctx.strokeRect(legendX - 6, legendY - 11, legendWidth, legendHeight);
      for (const item of series) {
        ctx.strokeStyle = item.color;
        ctx.lineWidth = item.width;
        ctx.setLineDash(item.dash);
        ctx.beginPath();
        ctx.moveTo(legendX, legendY);
        ctx.lineTo(legendX + 22, legendY);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#18202a";
        ctx.fillText(item.label, legendX + 29, legendY - 6);
        legendY += 18;
      }
      ctx.fillStyle = "#18202a";
      ctx.fillText("phase [rad]", margin.left + plotW / 2 - 28, canvas.height - 12);
      ctx.restore();
    }

    function renderMetrics(metrics) {
      els.metrics.innerHTML = "";
      const order = ["A_star", "A_speckle", "A_coherent", "A_planet", "A_incoherent", "R", "R1"];
      for (const key of order) {
        if (!(key in metrics)) continue;
        const div = document.createElement("div");
        div.className = "metric";
        div.innerHTML = `<span>${key}</span><strong>${Number(metrics[key]).toExponential(6)}</strong>`;
        els.metrics.appendChild(div);
      }
    }

    function formatRatio(value) {
      if (value === undefined) return "--";
      return value === null ? "undefined" : Number(value).toExponential(6);
    }

    function renderSweepMetrics(sweep) {
      els.metrics.innerHTML = "";
      const ratios = sweep.metrics.ratios || {};
      for (const [key, label] of [["R_response_harmonic", "R_response harmonic"], ["R_response_pp", "R_response peak-to-peak"]]) {
        if (!(key in ratios)) continue;
        const div = document.createElement("div");
        div.className = "metric";
        div.innerHTML = `<span>${label}</span><strong>${formatRatio(ratios[key])}</strong>`;
        els.metrics.appendChild(div);
      }
      for (const name of ["speckle_response", "planet_response", "coherent", "incoherent", "star", "speckle", "planet"]) {
        const item = sweep.metrics[name];
        if (!item) continue;
        const div = document.createElement("div");
        div.className = "metric";
        div.innerHTML = `<span>${name}</span><strong>M_pp ${Number(item.M_pp).toExponential(3)}<br>M_rms ${Number(item.M_rms).toExponential(3)}<br>M_harmonic ${Number(item.M_harmonic).toExponential(3)}<br>phase ${Number(item.phase_response).toFixed(3)}</strong>`;
        els.metrics.appendChild(div);
      }
    }

    function renderComparisonSummary(comparison, note) {
      const m = comparison && comparison.metrics ? comparison.metrics : {};
      const rows = [
        ["Speckle peak-to-peak modulation", m.main_speckle_pp],
        ["Planet peak-to-peak modulation", m.main_planet_pp],
        ["Modulation ratio", m.ratio_pp],
      ];
      els.comparisonSummary.innerHTML = "";
      for (const [label, value] of rows) {
        const div = document.createElement("div");
        div.className = "metric";
        const nearZero = label === "Modulation ratio" && m.ratio_denominator_near_zero;
        const suffix = nearZero ? "<br>denominator near zero" : "";
        div.innerHTML = `<span>${label}</span><strong>${formatRatio(value)}${suffix}</strong>`;
        els.comparisonSummary.appendChild(div);
      }
      if (note) {
        const div = document.createElement("div");
        div.className = "metric";
        div.style.gridColumn = "1 / -1";
        div.innerHTML = `<span>Status</span><strong>${note}</strong>`;
        els.comparisonSummary.appendChild(div);
      }
    }

    function renderComparisonMetrics(comparison) {
      els.metrics.innerHTML = "";
      const complete = comparison.metrics && comparison.metrics.complete;
      const sig = currentComparisonSignature();
      const outdated = state.lastComparisonSignature && sig !== state.lastComparisonSignature;
      renderComparisonSummary(
        comparison,
        !complete ? "Incomplete result" : (outdated ? "Settings changed after this result" : "")
      );
      const m = comparison.metrics || {};
      const rows = [
        ["speckle_run_planet_pp", "Planet pp, speckle mask"],
        ["planet_run_speckle_pp", "Speckle pp, planet mask"],
        ["main_planet_mean", "Mean planet power"],
      ];
      for (const [key, label] of rows) {
        if (!(key in m)) continue;
        const div = document.createElement("div");
        div.className = "metric";
        const suffix = key === "ratio_pp" && m.ratio_denominator_near_zero ? "<br>denominator near zero" : "";
        div.innerHTML = `<span>${label}</span><strong>${formatRatio(m[key])}${suffix}</strong>`;
        els.metrics.appendChild(div);
      }
      for (const [runKey, label] of [["speckle_position", "Speckle-position run"], ["planet_position", "Planet-position run"]]) {
        const sweep = comparison.sweeps && comparison.sweeps[runKey];
        if (!sweep) continue;
        for (const responseName of ["speckle_response", "planet_response"]) {
          const item = sweep.metrics[responseName];
          const div = document.createElement("div");
          div.className = "metric";
          div.innerHTML = `<span>${label}: ${responseName}</span><strong>mean ${Number(item.P_mean).toExponential(3)}<br>pp ${Number(item.M_pp).toExponential(3)}<br>harm ${Number(item.M_harmonic).toExponential(3)}</strong>`;
          els.metrics.appendChild(div);
        }
      }
      if (comparison.phase_zero_match) {
        const div = document.createElement("div");
        div.className = "metric";
        div.innerHTML = `<span>Phase-0 match star / speckle</span><strong>${Number(comparison.phase_zero_match.star_speckle_ratio).toExponential(6)}</strong>`;
        els.metrics.appendChild(div);
      }
    }

    function currentComparisonSignature() {
      if (!state.mask || !state.config) return "";
      const payload = commonSweepPayload();
      return JSON.stringify(payload);
    }

    function markResultsOutdated() {
      if (!state.lastComparison) return;
      renderComparisonMetrics(state.lastComparison);
      setStatus("Settings changed. Existing comparison may be outdated.");
    }

    function drawTradeoffPlot(optimization) {
      resizeCanvas(els.tradeoffPlot);
      const canvas = els.tradeoffPlot;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (!optimization || !optimization.history || !optimization.history.length) return;
      const points = optimization.history;
      let xmax = 0, ymax = 0;
      for (const p of points) {
        xmax = Math.max(xmax, Number(p.planet_pp) || 0);
        ymax = Math.max(ymax, Number(p.speckle_pp) || 0);
      }
      xmax = xmax || 1;
      ymax = ymax || 1;
      const margin = {left: 56, right: 16, top: 18, bottom: 38};
      const w = canvas.width - margin.left - margin.right;
      const h = canvas.height - margin.top - margin.bottom;
      ctx.strokeStyle = "rgba(24,32,42,0.18)";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const x = margin.left + (i / 4) * w;
        const y = margin.top + (i / 4) * h;
        ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(margin.left + w, y); ctx.stroke();
      }
      ctx.strokeStyle = "#18202a";
      ctx.beginPath();
      ctx.moveTo(margin.left, margin.top);
      ctx.lineTo(margin.left, margin.top + h);
      ctx.lineTo(margin.left + w, margin.top + h);
      ctx.stroke();
      const pareto = new Set(optimization.pareto_indices || []);
      points.forEach((p, idx) => {
        const x = margin.left + ((Number(p.planet_pp) || 0) / xmax) * w;
        const y = margin.top + (1 - ((Number(p.speckle_pp) || 0) / ymax)) * h;
        ctx.fillStyle = idx === optimization.best_index ? "#d7191c" : (pareto.has(idx) ? "#168245" : "#1d70b8");
        ctx.beginPath();
        ctx.arc(x, y, idx === optimization.best_index ? 6 : 4, 0, 2 * Math.PI);
        ctx.fill();
      });
      ctx.fillStyle = "#5b6673";
      ctx.font = `${Math.max(11, Math.round(canvas.width / 62))}px system-ui`;
      ctx.fillText("planet pp", margin.left + w / 2 - 28, canvas.height - 11);
      ctx.save();
      ctx.translate(14, margin.top + h / 2 + 34);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("speckle pp", 0, 0);
      ctx.restore();
    }

    async function applyMask() {
      setStatus("Propagating...");
      els.applyBtn.disabled = true;
      try {
        const response = await fetch("/api/propagate", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            mask_b64: unpackBase64Mask(),
            include_star: document.getElementById("starEnabled").checked,
            include_planet: document.getElementById("planetEnabled").checked,
            include_speckles: document.getElementById("specklesEnabled").checked,
            selected_speckle_indices: selectedSpeckleIndices(),
            star_planet_ratio: Number(document.getElementById("starPlanetRatio").value),
            star_speckle_ratio: Number(document.getElementById("starSpeckleRatio").value),
            phase_modulation_rad: Number(els.phaseModulationRad.value),
            lyot_reference_scale: Number(els.referenceSubtractPercent.value) / 100,
            display_mode: document.getElementById("displayMode").value
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Propagation failed");
        state.lastResult = data;
        state.lastSweep = null;
        redrawMaps();
        drawPhasePlot(null);
        renderMetrics(data.metrics);
        setStatus("Propagation complete.");
      } catch (err) {
        setStatus(String(err.message || err));
      } finally {
        els.applyBtn.disabled = false;
      }
    }

    async function evaluateModulation() {
      setStatus("Evaluating phase sweep...");
      els.evaluateModulationBtn.disabled = true;
      try {
        const response = await fetch("/api/phase-sweep", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            mask_b64: unpackBase64Mask(),
            phase_steps: Number(els.phaseSteps.value),
            phase_start_rad: Number(els.phaseStartRad.value),
            phase_span_rad: Number(els.phaseSpanRad.value),
            subtraction_mode: els.subtractionMode.value,
            include_star: document.getElementById("starEnabled").checked,
            include_planet: document.getElementById("planetEnabled").checked,
            include_speckles: document.getElementById("specklesEnabled").checked,
            selected_speckle_indices: selectedSpeckleIndices(),
            star_planet_ratio: Number(document.getElementById("starPlanetRatio").value),
            star_speckle_ratio: Number(document.getElementById("starSpeckleRatio").value),
            lyot_reference_scale: Number(els.referenceSubtractPercent.value) / 100
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Phase sweep failed");
        state.lastSweep = data;
        drawPhasePlot(data);
        renderSweepMetrics(data);
        if (data.mask_fraction > 0.99 && data.subtraction_mode === "field") {
          setStatus("Modulation phase sweep complete. Full-mask phase is global; the reference tracks the same SLM mask, so global-phase response is removed.");
        } else if (data.mask_fraction > 0.99 && data.subtraction_mode === "intensity") {
          setStatus("Modulation phase sweep complete. Full-mask phase is global, so individual source powers can remain constant.");
        } else {
          setStatus("Modulation phase sweep complete.");
        }
      } catch (err) {
        setStatus(String(err.message || err));
      } finally {
        els.evaluateModulationBtn.disabled = false;
      }
    }

    function commonSweepPayload() {
      return {
        mask_b64: unpackBase64Mask(),
        selected_speckle_index: Number(els.targetSpeckle.value || 0),
        phase_steps: Number(els.phaseSteps.value),
        phase_start_rad: Number(els.phaseStartRad.value),
        phase_span_rad: Number(els.phaseSpanRad.value),
        subtraction_mode: els.subtractionMode.value,
        include_star: document.getElementById("starEnabled").checked,
        include_planet: document.getElementById("planetEnabled").checked,
        include_speckles: document.getElementById("specklesEnabled").checked,
        star_planet_ratio: Number(document.getElementById("starPlanetRatio").value),
        star_speckle_ratio: Number(document.getElementById("starSpeckleRatio").value),
        match_phase_zero_power: els.matchPhaseZeroPower.checked,
        lyot_reference_scale: Number(els.referenceSubtractPercent.value) / 100
      };
    }

    async function runComparison(mode) {
      const labels = {speckle: "Running speckle-position sweep...", planet: "Running planet-position sweep...", both: "Running comparison..."};
      setStatus(labels[mode] || "Running comparison...");
      setBusy(true);
      try {
        const signature = currentComparisonSignature();
        const response = await fetch("/api/comparison", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(Object.assign(commonSweepPayload(), {mode}))
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Comparison failed");
        state.lastComparison = data;
        state.lastComparisonSignature = signature;
        state.lastSweep = null;
        state.placedMasks = {
          speckle_position: decodeMaskB64(data.placed_masks.speckle_position),
          planet_position: decodeMaskB64(data.placed_masks.planet_position),
        };
        redrawSlm();
        drawPhasePlot(null);
        drawPhasePlotOn(els.speckleSweepPlot, data.sweeps.speckle_position || null);
        drawPhasePlotOn(els.planetSweepPlot, data.sweeps.planet_position || null);
        renderComparisonMetrics(data);
        const complete = data.metrics && data.metrics.complete;
        setStatus(complete ? "Comparison complete. Main metric is peak-to-peak modulation." : "Run stopped. Completed results are kept.");
      } catch (err) {
        setStatus(String(err.message || err));
      } finally {
        setBusy(false);
      }
    }

    function drawPhasePlotOn(canvas, sweep) {
      const previous = els.phasePlot;
      els.phasePlot = canvas;
      try {
        drawPhasePlot(sweep);
      } finally {
        els.phasePlot = previous;
      }
    }

    async function optimizeMask() {
      setStatus("Optimizing mask...");
      setBusy(true);
      try {
        const response = await fetch("/api/optimize", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(Object.assign(commonSweepPayload(), {
            optimization_mode: els.tool.value === "circle" ? "circle_size" : "pixel_groups",
            circle_center_x: Number(els.centerX.value),
            circle_center_y: Number(els.centerY.value),
            circle_radius_px: Number(els.radiusPx.value),
            search_radius_px: Number(els.searchRadiusPx.value),
            group_size_px: Number(els.groupSizePx.value),
            iterations: Number(els.optIterations.value),
            planet_penalty: Number(els.planetPenalty.value)
          }))
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Optimization failed");
        state.lastOptimization = data;
        drawTradeoffPlot(data);
        if (data.best_mask_b64) {
          state.undo.push(state.mask.slice());
          state.mask = decodeMaskB64(data.best_mask_b64);
          commitCurrentMaskAsBase();
          redrawSlm();
        }
        renderOptimizationMetrics(data);
        setStatus(data.stopped ? "Optimization stopped. Completed candidates are kept." : "Optimization complete. Best mask was applied to the editor.");
      } catch (err) {
        setStatus(String(err.message || err));
      } finally {
        setBusy(false);
      }
    }

    function renderOptimizationMetrics(opt) {
      els.metrics.innerHTML = "";
      const best = opt.best_index === null || opt.best_index === undefined ? null : opt.history[opt.best_index];
      const rows = best ? [
        ["Mode", opt.optimization_mode || "pixel_groups"],
        ["Best candidate", best.name],
        ["Best score", Number(best.score).toExponential(6)],
        ["Speckle pp", Number(best.speckle_pp).toExponential(6)],
        ["Planet pp", Number(best.planet_pp).toExponential(6)],
        ["Ratio pp", formatRatio(best.ratio_pp)],
        ["Pareto candidates", String((opt.pareto_indices || []).length)],
      ] : [["Best candidate", "none"]];
      if (opt.phase_zero_match) {
        rows.push(["Matched star / speckle", Number(opt.phase_zero_match.star_speckle_ratio).toExponential(6)]);
      }
      for (const [label, value] of rows) {
        const div = document.createElement("div");
        div.className = "metric";
        div.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
        els.metrics.appendChild(div);
      }
      for (const item of opt.history || []) {
        const div = document.createElement("div");
        div.className = "metric";
        div.innerHTML = `<span>${item.name}</span><strong>score ${Number(item.score).toExponential(2)}<br>speckle ${Number(item.speckle_pp).toExponential(2)}<br>planet ${Number(item.planet_pp).toExponential(2)}</strong>`;
        els.metrics.appendChild(div);
      }
    }

    function setBusy(isBusy) {
      for (const button of [
        els.applyBtn, els.evaluateModulationBtn, els.runSpeckleSweepBtn,
        els.runPlanetSweepBtn, els.runComparisonBtn, els.optimizeMaskBtn,
        els.clearBtn, els.undoBtn, els.saveMaskBtn, els.loadMaskBtn,
        els.saveResultBtn, els.saveLogBtn, els.addRegionBtn, els.replaceRegionBtn
      ]) {
        button.disabled = Boolean(isBusy);
      }
      els.stopBtn.disabled = !isBusy;
      if (isBusy) startStatusPolling();
      else stopStatusPolling();
    }

    function startStatusPolling() {
      stopStatusPolling();
      state.pollTimer = window.setInterval(async () => {
        try {
          const response = await fetch("/api/status", {method: "POST"});
          const data = await response.json();
          if (!response.ok) return;
          const active = data.active_run || "running";
          const parts = [`Active run: ${active}`];
          if (data.phase_index && data.phase_count) {
            parts.push(`phase ${data.phase_index}/${data.phase_count}`);
          }
          if (data.candidate && data.candidate_count) {
            parts.push(`candidate ${data.candidate}/${data.candidate_count}`);
          }
          if (data.phase_rad !== undefined) {
            parts.push(`phase ${Number(data.phase_rad).toFixed(3)} rad`);
          }
          setStatus(parts.join(", "));
        } catch (_err) {
          return;
        }
      }, 1000);
    }

    function stopStatusPolling() {
      if (state.pollTimer !== null) {
        window.clearInterval(state.pollTimer);
        state.pollTimer = null;
      }
    }

    async function stopActiveRun() {
      const response = await fetch("/api/stop", {method: "POST"});
      const data = await response.json();
      setStatus(data.status || "Stop requested.");
    }

    function resetPhase() {
      els.phaseModulationRad.value = "0";
      setStatus("Phase set to zero for the next single propagation.");
    }

    function downloadText(filename, text) {
      const blob = new Blob([text], {type: "application/json"});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    }

    function saveMask() {
      downloadText("interactive_slm_mask.json", JSON.stringify({
        n_fft: state.config.n_fft,
        mask_b64: unpackBase64Mask()
      }));
    }

    function currentLogPayload() {
      return {
        saved_at: new Date().toISOString(),
        gui: "html_slm_gui",
        n_fft: state.config ? state.config.n_fft : null,
        mask_pixels: state.mask ? maskCount() : 0,
        mask_fraction: state.mask && state.config ? maskCount() / (state.config.n_fft * state.config.n_fft) : 0,
        controls: {
          include_star: document.getElementById("starEnabled").checked,
          include_planet: document.getElementById("planetEnabled").checked,
          include_speckles: document.getElementById("specklesEnabled").checked,
          selected_speckle_indices: selectedSpeckleIndices(),
          star_planet_ratio: Number(document.getElementById("starPlanetRatio").value),
          star_speckle_ratio: Number(document.getElementById("starSpeckleRatio").value),
          match_phase_zero_power: els.matchPhaseZeroPower.checked,
          science_phase_modulation_rad: Number(els.phaseModulationRad.value),
          reference_subtraction_percent: Number(els.referenceSubtractPercent.value),
          phase_steps: Number(els.phaseSteps.value),
          phase_start_rad: Number(els.phaseStartRad.value),
          phase_span_rad: Number(els.phaseSpanRad.value),
          subtraction_mode: els.subtractionMode.value,
        },
        source_markers: state.config ? state.config.source_markers : [],
        single_phase_metrics: state.lastResult ? state.lastResult.metrics : null,
        phase_sweep: state.lastSweep,
        comparison: state.lastComparison,
        optimization: state.lastOptimization,
      };
    }

    function saveLog() {
      const payload = currentLogPayload();
      downloadText("interactive_slm_lyot_log.json", JSON.stringify(payload, null, 2));
      setStatus("Saved log download.");
    }

    function loadMaskFile(file) {
      const reader = new FileReader();
      reader.onload = () => {
        const payload = JSON.parse(reader.result);
        if (payload.n_fft !== state.config.n_fft) {
          setStatus(`Mask grid mismatch: file ${payload.n_fft}, current ${state.config.n_fft}`);
          return;
        }
        const raw = atob(payload.mask_b64);
        const next = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) next[i] = raw.charCodeAt(i);
        state.undo.push(state.mask.slice());
        state.mask = next;
        commitCurrentMaskAsBase();
        redrawSlm();
        setStatus("Mask loaded.");
      };
      reader.readAsText(file);
    }

    async function saveResult() {
      const payload = currentLogPayload();
      payload.mask_b64 = unpackBase64Mask();
      payload.placed_masks_b64 = state.placedMasks ? {
        speckle_position: packMask(state.placedMasks.speckle_position),
        planet_position: packMask(state.placedMasks.planet_position),
      } : null;
      downloadText("interactive_slm_optimization_results.json", JSON.stringify(payload, null, 2));
      if (state.lastResult) {
        const response = await fetch("/api/save-result", {method: "POST"});
        const data = await response.json();
        setStatus(data.path ? `Downloaded JSON and saved ${data.path}` : "Downloaded JSON results.");
      } else {
        setStatus("Downloaded JSON results.");
      }
    }

    function bindEvents() {
      els.slm.addEventListener("pointerdown", event => {
        const pix = canvasToSlmPixel(event);
        if (!pix) return;
        state.undo.push(state.mask.slice());
        state.dragging = true;
        els.slm.setPointerCapture(event.pointerId);
        if (els.tool.value === "circle" || els.tool.value === "ring") {
          if (!state.baseMask) state.baseMask = state.mask.slice();
          state.circleStart = pix;
          setShapeFields(pix[0], pix[1], null, null, null);
          applyParameterizedShape();
        } else {
          commitCurrentMaskAsBase();
          state.lastPixel = pix;
          drawLinePixels(pix, pix);
          redrawSlm();
        }
      });
      els.slm.addEventListener("pointermove", event => {
        const pix = canvasToSlmPixel(event);
        if (!pix) return;
        if (!state.dragging) return;
        if (els.tool.value === "freehand") {
          drawLinePixels(state.lastPixel, pix);
          state.lastPixel = pix;
          redrawSlm();
        } else if (els.tool.value === "circle" && state.circleStart) {
          const r = Math.hypot(pix[0] - state.circleStart[0], pix[1] - state.circleStart[1]);
          setShapeFields(state.circleStart[0], state.circleStart[1], r, null, null);
          applyParameterizedShape();
        } else if (els.tool.value === "ring" && state.circleStart) {
          const r = Math.hypot(pix[0] - state.circleStart[0], pix[1] - state.circleStart[1]);
          setShapeFields(state.circleStart[0], state.circleStart[1], null, null, r);
          applyParameterizedShape();
        }
      });
      els.slm.addEventListener("pointerup", event => {
        const pix = canvasToSlmPixel(event);
        if (state.dragging && els.tool.value === "circle" && state.circleStart && pix) {
          drawCirclePixels(state.circleStart, pix);
          redrawSlm();
        } else if (state.dragging && els.tool.value === "ring" && state.circleStart && pix) {
          const r = Math.hypot(pix[0] - state.circleStart[0], pix[1] - state.circleStart[1]);
          setShapeFields(state.circleStart[0], state.circleStart[1], null, null, r);
          applyParameterizedShape();
        }
        state.dragging = false;
        if (els.tool.value === "freehand") {
          commitCurrentMaskAsBase();
        }
        state.circleStart = null;
        state.lastPixel = null;
      });
      els.applyBtn.addEventListener("click", applyMask);
      els.evaluateModulationBtn.addEventListener("click", evaluateModulation);
      els.runSpeckleSweepBtn.addEventListener("click", () => runComparison("speckle"));
      els.runPlanetSweepBtn.addEventListener("click", () => runComparison("planet"));
      els.runComparisonBtn.addEventListener("click", () => runComparison("both"));
      els.optimizeMaskBtn.addEventListener("click", optimizeMask);
      els.stopBtn.addEventListener("click", stopActiveRun);
      els.resetPhaseBtn.addEventListener("click", resetPhase);
      els.clearBtn.addEventListener("click", () => {
        state.undo.push(state.mask.slice());
        state.mask.fill(0);
        commitCurrentMaskAsBase();
        redrawSlm();
      });
      els.undoBtn.addEventListener("click", () => {
        const prev = state.undo.pop();
        if (!prev) return;
        state.mask = prev;
        commitCurrentMaskAsBase();
        redrawSlm();
      });
      els.saveMaskBtn.addEventListener("click", saveMask);
      els.loadMaskBtn.addEventListener("click", () => els.maskFile.click());
      els.maskFile.addEventListener("change", event => {
        const file = event.target.files && event.target.files[0];
        if (file) loadMaskFile(file);
      });
      els.saveResultBtn.addEventListener("click", saveResult);
      els.saveLogBtn.addEventListener("click", saveLog);
      els.addSpeckleBtn.addEventListener("click", () => addSpeckleEditorRow(0, 0));
      els.updateSourcesBtn.addEventListener("click", updateSourceGeometry);
      els.addRegionBtn.addEventListener("click", commitPreviewRegion);
      els.replaceRegionBtn.addEventListener("click", replaceMaskWithPreview);
      els.tool.addEventListener("change", () => {
        updateShapeParameterVisibility();
        applyParameterizedShape();
      });
      for (const input of [els.centerX, els.centerY, els.radiusPx, els.innerRadiusPx, els.outerRadiusPx]) {
        input.addEventListener("input", applyParameterizedShape);
        input.addEventListener("change", applyParameterizedShape);
        input.addEventListener("keyup", applyParameterizedShape);
      }
      els.displayMode.addEventListener("change", redrawMaps);
      for (const input of [
        document.getElementById("starEnabled"),
        document.getElementById("planetEnabled"),
        document.getElementById("specklesEnabled"),
        document.getElementById("starPlanetRatio"),
        document.getElementById("starSpeckleRatio"),
        els.matchPhaseZeroPower,
        els.targetSpeckle,
        els.phaseSteps,
        els.phaseStartRad,
        els.phaseSpanRad,
        els.referenceSubtractPercent,
        els.subtractionMode,
      ]) {
        input.addEventListener("input", markResultsOutdated);
        input.addEventListener("change", markResultsOutdated);
      }
      window.addEventListener("resize", () => {
        redrawSlm();
        redrawMaps();
        drawPhasePlot(state.lastSweep);
        if (state.lastComparison) {
          drawPhasePlotOn(els.speckleSweepPlot, state.lastComparison.sweeps.speckle_position || null);
          drawPhasePlotOn(els.planetSweepPlot, state.lastComparison.sweeps.planet_position || null);
        }
        drawTradeoffPlot(state.lastOptimization);
      });
    }

    async function init() {
      const response = await fetch("/api/config");
      state.config = await response.json();
      state.mask = new Uint8Array(state.config.n_fft * state.config.n_fft);
      state.baseMask = state.mask.slice();
      renderSpeckleSelector();
      renderSourceGeometryControls();
      const c = Math.round((state.config.n_fft - 1) / 2);
      setShapeFields(c, c, null, null, null);
      bindEvents();
      updateShapeParameterVisibility();
      redrawSlm();
      drawPhasePlot(null);
      drawPhasePlotOn(els.speckleSweepPlot, null);
      drawPhasePlotOn(els.planetSweepPlot, null);
      drawTradeoffPlot(null);
      renderComparisonSummary(null, "");
      setBusy(false);
      setStatus(`Ready. SLM grid ${state.config.n_fft} x ${state.config.n_fft}.`);
    }

    init();
  </script>
</body>
</html>
"""


def _component_for_display(field: np.ndarray, mode: str) -> np.ndarray:
    if mode == "phase":
        return np.angle(field)
    if mode == "real":
        return np.real(field)
    if mode == "imag":
        return np.imag(field)
    return np.abs(field)


class _SLMHtmlServer:
    def __init__(self):
        self.calculator = SLMLyotResponseCalculator()
        self.last_result: dict | None = None
        self.last_comparison: dict | None = None
        self.last_optimization: dict | None = None
        self.last_save_path = Path("interactive_slm_lyot_response_html.npz")
        self.stop_event = threading.Event()
        self.progress: dict = {"active_run": "idle"}
        center = self.calculator.n_fft // 2
        half_width = max(2, int(round(0.75 * float(self.calculator.config.pupil_pixels))))
        row_slice = slice(max(0, center - half_width), min(self.calculator.n_fft, center + half_width + 1))
        col_slice = slice(max(0, center - half_width), min(self.calculator.n_fft, center + half_width + 1))
        self.lyot_view = np.s_[row_slice, col_slice]

    def source_markers(self) -> list[dict]:
        markers = [{"label": "star", "x": 0.0, "y": 0.0, "color": "#c28a00", "kind": "star"}]
        px, py = self.calculator.config.companion_offset_lamD
        markers.append({"label": "planet", "x": float(px), "y": float(py), "color": "#00a5b8", "kind": "planet"})
        for idx, (sx, sy) in enumerate(self.calculator._speckle_offsets()):
            markers.append(
                {
                    "label": f"s{idx + 1}",
                    "index": int(idx),
                    "x": float(sx),
                    "y": float(sy),
                    "color": "#b24bbd",
                    "kind": "speckle",
                }
            )
        return markers

    def update_source_geometry(
        self,
        *,
        planet_x: float,
        planet_y: float,
        speckle_offsets: object,
    ) -> dict:
        planet = (float(planet_x), float(planet_y))
        if not np.all(np.isfinite(planet)):
            raise ValueError("Planet coordinates must be finite numbers.")

        if not isinstance(speckle_offsets, list):
            raise ValueError("speckle_offsets must be a list of [x, y] pairs.")
        parsed_speckles: list[tuple[float, float]] = []
        for idx, offset in enumerate(speckle_offsets, start=1):
            if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                raise ValueError(f"Speckle {idx} must be an [x, y] pair.")
            sx, sy = float(offset[0]), float(offset[1])
            if not np.isfinite(sx) or not np.isfinite(sy):
                raise ValueError(f"Speckle {idx} coordinates must be finite numbers.")
            parsed_speckles.append((sx, sy))

        self.calculator.config.companion_offset_lamD = planet
        self.calculator.config.custom_speckle_offsets_lamD = tuple(parsed_speckles)
        self.calculator._unit_unmodulated_cache.clear()
        self.last_result = None
        return self.config_payload()

    def config_payload(self) -> dict:
        calc = self.calculator
        x = calc.x_lamD
        y = calc.y_lamD
        lyot_x = calc._template._x / float(calc.config.pupil_pixels)
        lyot_y = calc._template._y / float(calc.config.pupil_pixels)
        return {
            "n_fft": calc.n_fft,
            "focal_sampling": calc.focal_sampling,
            "slm_extent": [
                float(np.min(x)),
                float(np.max(x)),
                float(np.min(y)),
                float(np.max(y)),
            ],
            "lyot_extent": [
                float(np.min(lyot_x[self.lyot_view])),
                float(np.max(lyot_x[self.lyot_view])),
                float(np.min(lyot_y[self.lyot_view])),
                float(np.max(lyot_y[self.lyot_view])),
            ],
            "source_markers": self.source_markers(),
        }

    def decode_mask(self, mask_b64: str) -> np.ndarray:
        raw = base64.b64decode(mask_b64.encode("ascii"), validate=True)
        expected = self.calculator.n_fft * self.calculator.n_fft
        if len(raw) != expected:
            raise ValueError(f"Mask byte length must be {expected}, got {len(raw)}.")
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.calculator.n_fft, self.calculator.n_fft)
        ).astype(bool)

    def encode_mask(self, mask: np.ndarray) -> str:
        return base64.b64encode(np.asarray(mask, dtype=np.uint8).ravel().tobytes()).decode("ascii")

    def _progress_callback(self, progress: dict) -> None:
        self.progress = {str(key): value for key, value in progress.items()}

    def _stop_requested(self) -> bool:
        return self.stop_event.is_set()

    def calibrated_star_speckle_ratio(self, payload: dict, selected_indices: object) -> tuple[float, dict | None]:
        base_ratio = float(payload.get("star_speckle_ratio", 100.0))
        if not bool(payload.get("match_phase_zero_power", False)):
            return base_ratio, None
        calibration = self.calculator.calibrate_star_speckle_ratio_for_phase_zero_match(
            selected_speckle_indices=selected_indices,
            star_planet_ratio=float(payload.get("star_planet_ratio", 500.0)),
            lyot_reference_scale=float(payload.get("lyot_reference_scale", 1.0)),
            subtraction_mode=str(payload.get("subtraction_mode", "field")),
        )
        return float(calibration["star_speckle_ratio"]), calibration

    def map_payload(self, result: dict, mode: str) -> dict:
        maps = {}
        extent = self.config_payload()["lyot_extent"]
        stop = np.asarray(result["lyot_stop"], dtype=bool)[self.lyot_view]
        for name in ("star", "planet", "speckle", "coherent", "incoherent"):
            delta = result["fields"][name]["delta"]
            if name == "incoherent":
                values = np.asarray(delta, dtype=float)[self.lyot_view]
                mode_values = {
                    "abs": values,
                    "phase": np.zeros_like(values),
                    "real": values,
                    "imag": np.zeros_like(values),
                }
            else:
                values = _component_for_display(delta, mode)[self.lyot_view]
                mode_values = {
                    display_mode: np.asarray(
                        _component_for_display(delta, display_mode)[self.lyot_view],
                        dtype=float,
                    )
                    for display_mode in ("abs", "phase", "real", "imag")
                }
            maps[name] = {
                "shape": list(values.shape),
                "values": np.asarray(values, dtype=float).ravel().tolist(),
                "modes": {
                    display_mode: np.asarray(mode_value, dtype=float).ravel().tolist()
                    for display_mode, mode_value in mode_values.items()
                },
                "lyot_stop": stop.astype(int).ravel().tolist(),
                "extent": extent,
            }
        return maps

    def phase_sweep_payload(self, sweep: dict) -> dict:
        metrics = {}
        for name, values in sweep["metrics"].items():
            if isinstance(values, dict):
                metrics[name] = {
                    key: None if value is None else float(value)
                    for key, value in values.items()
                }
            else:
                metrics[name] = None if values is None else float(values)
        return {
            "phase_steps": int(sweep["phase_steps"]),
            "phase_start_rad": float(sweep.get("phase_start_rad", 0.0)),
            "phase_span_rad": float(sweep.get("phase_span_rad", 2.0 * np.pi)),
            "subtraction_mode": str(sweep["subtraction_mode"]),
            "lyot_reference_scale": float(sweep["lyot_reference_scale"]),
            "phases": np.asarray(sweep["phases"], dtype=float).tolist(),
            "mask_fraction": float(np.mean(np.asarray(sweep["slm_region_mask"], dtype=bool))),
            "powers": {
                name: np.asarray(values, dtype=float).tolist()
                for name, values in sweep["powers"].items()
            },
            "responses": {
                name: np.asarray(values, dtype=float).tolist()
                for name, values in sweep.get("responses", {}).items()
            },
            "metrics": metrics,
            "phase_zero_match": sweep.get("phase_zero_match"),
        }

    def comparison_payload(
        self,
        comparison: dict,
        *,
        speckle_mask: np.ndarray,
        planet_mask: np.ndarray,
    ) -> dict:
        return {
            "measurement_plane": comparison["measurement_plane"],
            "primary_metric": comparison["primary_metric"],
            "metrics": comparison["metrics"],
            "placements": comparison["placements"],
            "phase_zero_match": comparison.get("phase_zero_match"),
            "placed_masks": {
                "speckle_position": self.encode_mask(speckle_mask),
                "planet_position": self.encode_mask(planet_mask),
            },
            "sweeps": {
                name: self.phase_sweep_payload(sweep)
                for name, sweep in comparison["sweeps"].items()
            },
        }

    def optimization_payload(self, optimization: dict) -> dict:
        history = []
        for item in optimization["history"]:
            copied = {
                key: value
                for key, value in item.items()
                if key not in {"mask", "comparison"}
            }
            history.append(copied)
        best_index = optimization["best_index"]
        best_mask_b64 = None
        if best_index is not None and 0 <= best_index < len(optimization["history"]):
            best_mask_b64 = self.encode_mask(optimization["history"][best_index]["mask"])
        return {
            "primary_metric": optimization["primary_metric"],
            "optimization_mode": optimization.get("optimization_mode", "pixel_groups"),
            "search_radius_px": optimization["search_radius_px"],
            "group_size_px": optimization["group_size_px"],
            "iterations": optimization["iterations"],
            "planet_penalty": optimization["planet_penalty"],
            "random_seed": optimization["random_seed"],
            "phase_zero_match": optimization.get("phase_zero_match"),
            "history": history,
            "best_index": best_index,
            "best_mask_b64": best_mask_b64,
            "pareto_indices": optimization["pareto_indices"],
            "stopped": optimization["stopped"],
        }


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(app: _SLMHtmlServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}")

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/api/config":
                _json_response(self, 200, app.config_payload())
                return
            _json_response(self, 404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                if path == "/api/propagate":
                    mask = app.decode_mask(str(payload["mask_b64"]))
                    result = app.calculator.propagate(
                        mask,
                        include_star=bool(payload.get("include_star", True)),
                        include_planet=bool(payload.get("include_planet", True)),
                        include_speckles=bool(payload.get("include_speckles", False)),
                        selected_speckle_indices=payload.get("selected_speckle_indices"),
                        star_planet_ratio=float(payload.get("star_planet_ratio", 500.0)),
                        star_speckle_ratio=float(payload.get("star_speckle_ratio", 100.0)),
                        phase_modulation_rad=float(payload.get("phase_modulation_rad", np.pi)),
                        lyot_reference_scale=float(payload.get("lyot_reference_scale", 1.0)),
                    )
                    app.last_result = result
                    mode = str(payload.get("display_mode", "abs"))
                    _json_response(
                        self,
                        200,
                        {
                            "metrics": result["metrics"],
                            "maps": app.map_payload(result, mode),
                        },
                    )
                    return
                if path == "/api/phase-sweep":
                    mask = app.decode_mask(str(payload["mask_b64"]))
                    selected_indices = payload.get("selected_speckle_indices")
                    star_speckle_ratio, phase_zero_match = app.calibrated_star_speckle_ratio(
                        payload,
                        selected_indices,
                    )
                    sweep = app.calculator.phase_sweep(
                        mask,
                        phase_steps=int(payload.get("phase_steps", 4)),
                        phase_start_rad=float(payload.get("phase_start_rad", 0.0)),
                        phase_span_rad=float(payload.get("phase_span_rad", 2.0 * np.pi)),
                        subtraction_mode=str(payload.get("subtraction_mode", "field")),
                        include_star=bool(payload.get("include_star", True)),
                        include_planet=bool(payload.get("include_planet", True)),
                        include_speckles=bool(payload.get("include_speckles", False)),
                        selected_speckle_indices=selected_indices,
                        star_planet_ratio=float(payload.get("star_planet_ratio", 500.0)),
                        star_speckle_ratio=star_speckle_ratio,
                        lyot_reference_scale=float(payload.get("lyot_reference_scale", 1.0)),
                        progress_callback=lambda update: app._progress_callback(
                            {"active_run": "phase_sweep", **update}
                        ),
                    )
                    sweep["phase_zero_match"] = phase_zero_match
                    _json_response(self, 200, app.phase_sweep_payload(sweep))
                    return
                if path == "/api/comparison":
                    app.stop_event.clear()
                    template = app.decode_mask(str(payload["mask_b64"]))
                    selected_index = int(payload.get("selected_speckle_index", 0))
                    speckle_offsets = app.calculator._speckle_offsets()
                    if not speckle_offsets:
                        raise ValueError("At least one speckle is required.")
                    star_speckle_ratio, phase_zero_match = app.calibrated_star_speckle_ratio(
                        payload,
                        (selected_index,),
                    )
                    speckle_placement = app.calculator.place_mask_template(
                        template,
                        speckle_offsets[selected_index],
                    )
                    planet_placement = app.calculator.place_mask_template(
                        template,
                        app.calculator.config.companion_offset_lamD,
                    )
                    mode = str(payload.get("mode", "both"))
                    sweep_kwargs = dict(
                        phase_steps=int(payload.get("phase_steps", 4)),
                        phase_start_rad=float(payload.get("phase_start_rad", 0.0)),
                        phase_span_rad=float(payload.get("phase_span_rad", 2.0 * np.pi)),
                        subtraction_mode=str(payload.get("subtraction_mode", "field")),
                        include_star=bool(payload.get("include_star", True)),
                        include_planet=bool(payload.get("include_planet", True)),
                        include_speckles=bool(payload.get("include_speckles", True)),
                        selected_speckle_indices=(selected_index,),
                        star_planet_ratio=float(payload.get("star_planet_ratio", 500.0)),
                        star_speckle_ratio=star_speckle_ratio,
                        lyot_reference_scale=float(payload.get("lyot_reference_scale", 1.0)),
                    )
                    sweeps = {}
                    app._progress_callback({"active_run": "speckle", "completed_runs": 0, "total_runs": 2})
                    if mode in {"speckle", "both"} and not app._stop_requested():
                        sweeps["speckle_position"] = app.calculator.phase_sweep(
                            speckle_placement["mask"],
                            **sweep_kwargs,
                            progress_callback=lambda update: app._progress_callback(
                                {"active_run": "speckle", "completed_runs": 0, "total_runs": 2, **update}
                            ),
                        )
                    app._progress_callback({"active_run": "planet", "completed_runs": len(sweeps), "total_runs": 2})
                    if mode in {"planet", "both"} and not app._stop_requested():
                        sweeps["planet_position"] = app.calculator.phase_sweep(
                            planet_placement["mask"],
                            **sweep_kwargs,
                            progress_callback=lambda update: app._progress_callback(
                                {"active_run": "planet", "completed_runs": len(sweeps), "total_runs": 2, **update}
                            ),
                        )
                    app._progress_callback({"active_run": "idle", "completed_runs": len(sweeps), "total_runs": 2})
                    comparison = app.calculator._comparison_result(
                        sweeps,
                        speckle_placement,
                        planet_placement,
                    )
                    comparison["phase_zero_match"] = phase_zero_match
                    app.last_comparison = comparison
                    _json_response(
                        self,
                        200,
                        app.comparison_payload(
                            comparison,
                            speckle_mask=speckle_placement["mask"],
                            planet_mask=planet_placement["mask"],
                        ),
                    )
                    return
                if path == "/api/optimize":
                    app.stop_event.clear()
                    template = app.decode_mask(str(payload["mask_b64"]))
                    selected_index = int(payload.get("selected_speckle_index", 0))
                    star_speckle_ratio, phase_zero_match = app.calibrated_star_speckle_ratio(
                        payload,
                        (selected_index,),
                    )
                    optimization = app.calculator.optimize_common_mask(
                        template,
                        optimization_mode=str(payload.get("optimization_mode", "pixel_groups")),
                        circle_center_pixel=(
                            int(payload.get("circle_center_x", app.calculator.n_fft // 2)),
                            int(payload.get("circle_center_y", app.calculator.n_fft // 2)),
                        ),
                        circle_radius_min_px=max(
                            1,
                            int(round(float(payload.get("circle_radius_px", 1.0))))
                            - int(payload.get("search_radius_px", 24)),
                        ),
                        circle_radius_max_px=(
                            int(round(float(payload.get("circle_radius_px", 1.0))))
                            + int(payload.get("search_radius_px", 24))
                        ),
                        selected_speckle_index=selected_index,
                        phase_steps=int(payload.get("phase_steps", 4)),
                        phase_start_rad=float(payload.get("phase_start_rad", 0.0)),
                        phase_span_rad=float(payload.get("phase_span_rad", 2.0 * np.pi)),
                        subtraction_mode=str(payload.get("subtraction_mode", "field")),
                        include_star=bool(payload.get("include_star", True)),
                        include_planet=bool(payload.get("include_planet", True)),
                        include_speckles=bool(payload.get("include_speckles", True)),
                        star_planet_ratio=float(payload.get("star_planet_ratio", 500.0)),
                        star_speckle_ratio=star_speckle_ratio,
                        lyot_reference_scale=float(payload.get("lyot_reference_scale", 1.0)),
                        search_radius_px=int(payload.get("search_radius_px", 24)),
                        group_size_px=int(payload.get("group_size_px", 4)),
                        iterations=int(payload.get("iterations", 12)),
                        planet_penalty=float(payload.get("planet_penalty", 1.0)),
                        progress_callback=app._progress_callback,
                        stop_requested=app._stop_requested,
                    )
                    optimization["phase_zero_match"] = phase_zero_match
                    app.last_optimization = optimization
                    _json_response(self, 200, app.optimization_payload(optimization))
                    return
                if path == "/api/stop":
                    app.stop_event.set()
                    _json_response(self, 200, {"status": "Stop requested. Completed results will be kept."})
                    return
                if path == "/api/status":
                    _json_response(self, 200, app.progress)
                    return
                if path == "/api/save-result":
                    if app.last_result is None:
                        _json_response(self, 400, {"error": "No propagation result to save."})
                        return
                    save_slm_lyot_result(app.last_save_path, app.last_result)
                    _json_response(self, 200, {"path": str(app.last_save_path)})
                    return
                if path == "/api/source-geometry":
                    config = app.update_source_geometry(
                        planet_x=payload.get("planet_x"),
                        planet_y=payload.get("planet_y"),
                        speckle_offsets=payload.get("speckle_offsets", []),
                    )
                    _json_response(self, 200, config)
                    return
            except Exception as exc:
                _json_response(self, 400, {"error": str(exc)})
                return
            _json_response(self, 404, {"error": "not found"})

    return Handler


def launch_html_gui(host: str = "127.0.0.1", port: int = 8765) -> None:
    app = _SLMHtmlServer()
    server = ThreadingHTTPServer((host, int(port)), make_handler(app))
    url = f"http://{host}:{int(port)}/"
    print(f"HTML SLM GUI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping HTML SLM GUI.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HTML SLM/Lyot diagnostic GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    launch_html_gui(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
