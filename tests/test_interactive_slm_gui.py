from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime
from pathlib import Path

import numpy as np

from coronagraph.interactive_slm_gui import (
    SLMLyotConfig,
    SLMLyotResponseCalculator,
    default_slm_gui_config,
    save_slm_lyot_result,
    launch_gui,
)
from coronagraph.html_slm_gui import HTML, _SLMHtmlServer
from coronagraph.masks import NoPhaseMask


class InteractiveSLMGuiTests(unittest.TestCase):
    def test_desktop_science_window_scan_and_exports(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.widgets import Button, TextBox, RadioButtons

        calc = SLMLyotResponseCalculator(SLMLyotConfig(
            pupil_pixels=12, focal_sampling=3.0, phase_mask=NoPhaseMask(),
            companion_offset_lamD=(2.0, 1.0),
            custom_speckle_offsets_lamD=((-2.0, 0.0), (0.0, -2.0)),
            include_ghost=False, ghost_fraction=0.0))
        buttons, boxes, radios = {}, {}, {}
        def button(*args, **kwargs):
            widget = Button(*args, **kwargs)
            buttons[widget.label.get_text()] = widget
            return widget
        def textbox(*args, **kwargs):
            widget = TextBox(*args, **kwargs)
            boxes[widget.label.get_text().strip()] = widget
            return widget
        def radio(*args, **kwargs):
            widget = RadioButtons(*args, **kwargs)
            radios[tuple(label.get_text() for label in widget.labels)] = widget
            return widget
        def click(name):
            buttons[name]._observers.process("clicked", None)

        previous_directory = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory, patch(
                "coronagraph.interactive_slm_gui.SLMLyotResponseCalculator", return_value=calc
            ), patch("matplotlib.widgets.Button", side_effect=button), patch(
                "matplotlib.widgets.TextBox", side_effect=textbox
            ), patch("matplotlib.widgets.RadioButtons", side_effect=radio), patch.object(plt, "show"), patch.object(Figure, "show"):
                os.chdir(directory)
                launch_gui()
                main = plt.gcf()
                boxes["Cx px"].set_val("12")
                boxes["Cy px"].set_val("18")
                boxes["R px"].set_val("2")
                boxes["Steps"].set_val("5")
                boxes["Star/Planet"].set_val("700")
                original_mask = main.axes[0].images[0].get_array().copy()
                click("Intensity")
                science = plt.gcf()
                self.assertIsNot(main, science)
                boxes["ROI r [lambda/D]"].set_val("0.7")
                boxes["Star/Speckle"].set_val("80")
                click("Scan Intensity")
                image_axes, curve_axes = science.axes[0], science.axes[2]
                self.assertEqual(len(image_axes.patches), 3)
                self.assertEqual(image_axes.patches[0].center, (-2.0, 0.0))
                self.assertEqual(image_axes.patches[1].center, (0.0, -2.0))
                self.assertTrue(all(circle.radius == 0.7 for circle in image_axes.patches))
                self.assertEqual([line.get_label() for line in curve_axes.lines], ["speckle_1", "speckle_2", "planet"])
                self.assertTrue(all(len(line.get_ydata()) == 5 for line in curve_axes.lines))
                self.assertGreater(np.max(image_axes.images[0].get_array()), 0)
                np.testing.assert_array_equal(main.axes[0].images[0].get_array(), original_mask)
                for name in ("Save Result", "Save Log", "Save Fig"):
                    click(name)
                artifacts = list(Path("interactive_slm_logs").glob("*/*science_intensity*"))
                self.assertEqual(len(artifacts), 3)
                saved_log = json.loads(next(p for p in artifacts if p.suffix == ".json").read_text())
                measurement = saved_log["science_intensity"]
                self.assertEqual(measurement["source_state"]["star_planet_ratio"], 700)
                self.assertEqual(measurement["source_state"]["star_speckle_ratio"], 80)
                self.assertNotIn("preview_intensity", measurement)
                with np.load(next(p for p in artifacts if p.suffix == ".npz")) as saved:
                    self.assertEqual(saved["speckle_2_radius_lamD"], 0.7)
                    np.testing.assert_array_equal(saved["speckle_2_center_lamD"], [0, -2])
                    np.testing.assert_allclose(saved["planet_mean_intensity"], curve_axes.lines[2].get_ydata())
                    np.testing.assert_array_equal(saved["slm_region_mask"], original_mask.astype(bool))
                radios[("Lyot plane", "Final science")].set_active(0)
                click("Eval Mod")
                lyot_axes = next(ax for ax in main.axes if ax.get_title() == "Integrated Lyot-stop power vs phase")
                self.assertEqual(len(lyot_axes.lines), 5)
                click("Save Log")
                lyot_log = next(Path("interactive_slm_logs").glob("*/*phase_sweep_log.json"))
                lyot_payload = json.loads(lyot_log.read_text())
                self.assertEqual(lyot_payload['controls']['measurement_plane'], "Lyot plane")
                self.assertEqual(len(lyot_payload['phase_sweep']['phases']), 5)
                radios[("Lyot plane", "Final science")].set_active(1)
                # Invalid controls leave the previous valid measurement intact.
                boxes["ROI r [lambda/D]"].set_val("-1")
                click("Scan Intensity")
                self.assertIn("Intensity scan failed", science.texts[1].get_text())
                self.assertEqual(len(image_axes.patches), 3)
                self.assertGreater(next(p for p in artifacts if p.suffix == ".png").stat().st_size, 1000)
        finally:
            os.chdir(previous_directory)
            plt.close("all")

    @unittest.skipUnless(shutil.which("node"), "Node.js is needed for GUI behavior checks")
    def test_html_result_dialog_and_selective_image_saving(self) -> None:
        script = HTML.split("<script>", 1)[1].split("</script>", 1)[0]
        # Run the actual GUI functions with a minimal DOM/canvas adapter.
        harness = r"""
const assert = require("node:assert/strict");
const elements = new Map();
const labels = [];
const textPositions = [];
const speckleChoices = [];
const rectangles = [];
const context = new Proxy({
  measureText: text => ({width: text.length * 7}),
  fillText: (text, x, y) => { labels.push(text); textPositions.push({text, x, y}); },
  fillRect: (...args) => rectangles.push({color: context.fillStyle, args}),
}, {get: (target, key) => target[key] || (() => {})});
function element(id = "") {
  return {id, checked: true, disabled: false, open: false, children: [], dataset: {},
    parentElement: {after() {}},
    closest() { return this.parentElement; },
    append(...children) { this.children.push(...children); },
    appendChild(child) { this.children.push(child); },
    replaceChildren() { this.children = []; },
    after() {},
    setAttribute() {}, addEventListener() {},
    getBoundingClientRect: () => ({left: 0, top: 0, width: 900, height: 380}),
    getContext: () => context,
    toDataURL: () => "data:image/png;base64," + id,
    showModal() { this.open = true; this.opens = (this.opens || 0) + 1; },
    close() { this.open = false; },
  };
}
global.document = {
  getElementById(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); },
  createElement: tag => element(tag),
  querySelector(selector) { return this.getElementById(selector); },
  querySelectorAll: selector => selector === ".speckle-choice" ? speckleChoices : [],
};
global.window = {devicePixelRatio: 2};
"""
        checks = r"""
setupFigureControls();
assert.equal(els.resultsDialog.open, false);
showResults();
assert.equal(els.resultsDialog.open, false, "No dialog before results");
state.config = {};
assert.deepEqual(Object.keys(figurePayload()), ["slm_mask"], "Empty plots are excluded");
state.lastResult = {maps: {planet: {}}, metrics: {A_planet: 1}};
updateResultVisibility();
assert.equal(document.getElementById("mapResults").hidden, false);
assert.equal(els.maps.star.parentElement.hidden, true);
assert.equal(els.maps.planet.parentElement.hidden, false);
for (const id of ["currentPhaseResults", "comparisonResults", "comparisonSummary", "optimizationResults"]) {
  assert.equal(document.getElementById(id).hidden, true, `${id} has no content`);
}
state.lastSweep = {};
updateResultVisibility();
assert.equal(document.getElementById("currentPhaseResults").hidden, false);
state.lastSweep = null;
state.lastComparison = {sweeps: {speckle_position: {}}};
updateResultVisibility();
assert.equal(document.getElementById("comparisonResults").hidden, false);
assert.equal(document.getElementById("comparisonSummary").hidden, false);
assert.equal(document.getElementById("speckleSweepResults").hidden, false);
assert.equal(document.getElementById("planetSweepResults").hidden, true);
state.lastComparison = {sweeps: {planet_position: {}}};
updateResultVisibility();
assert.equal(document.getElementById("speckleSweepResults").hidden, true);
assert.equal(document.getElementById("planetSweepResults").hidden, false);
state.lastComparison = {comparison_type: "shared_mask", sweeps: {shared_mask: {}}};
updateResultVisibility();
assert.equal(document.getElementById("speckleSweepResults").hidden, false);
assert.equal(document.getElementById("planetSweepResults").hidden, true);
assert(document.getElementById("comparisonSweepTitle").textContent.includes("Same-mask"));
assert("speckle_sweep_plot" in figurePayload(), "Shared sweep can be saved");
state.lastComparison = null;
state.lastOptimization = {history: [{}]};
updateResultVisibility();
assert.equal(document.getElementById("optimizationResults").hidden, false);
state.lastOptimization = null;
document.querySelector('input[data-figure="planet_map"]').checked = false;
assert.equal("planet_map" in figurePayload(), false);
assert.deepEqual(Object.keys(figurePayload("planet_map")), ["planet_map"], "Individual save ignores bulk selection");
redrawResults = () => {};
showResults();
assert.equal(els.resultsDialog.open, true);
assert.equal(els.showResultsBtn.disabled, false);
showResults();
assert.equal(els.resultsDialog.opens, 1, "Updating an open dialog does not reopen it");
els.resultsDialog.close();
showResults();
assert.equal(els.resultsDialog.opens, 2);
const optimization = {history: [
  {name: "small_circle", planet_pp: 0.01, speckle_pp: 0.4},
  {name: "large_circle", planet_pp: 0.01, speckle_pp: 0.4},
  {name: "zero_mask", planet_pp: 0, speckle_pp: 0},
], best_index: 1, pareto_indices: [0, 1]};
drawTradeoffPlot(optimization);
assert(labels.includes("C1") && labels.includes("C2") && labels.includes("C3"));
assert(labels.includes("Best score") && labels.includes("Pareto frontier"));
assert(labels.some(text => text.startsWith("Planet peak-to-peak")));
assert(labels.some(text => text.startsWith("Speckle peak-to-peak")));
assert(labels.some(text => /e[-+]/.test(text)), "Axes have numeric ticks");
assert.deepEqual(circleRadiusPoints(optimization), [], "Other mask types have no radius plot");
const circleOptimization = {best_index: 0, history: [
  {name: "circle_radius_3px", radius_px: 3, planet_pp: 0.2, speckle_pp: 0.6, ratio_pp: 3},
  {name: "circle_radius_1px", radius_px: 1, planet_pp: 0.1, speckle_pp: 0.3},
  {name: "circle_radius_2px", radius_px: 2, planet_pp: 0.15, speckle_pp: 0.4},
]};
assert.deepEqual(circleRadiusPoints(circleOptimization).map(p => p.radius_px), [1, 2, 3]);
drawRadiusPlot(circleOptimization);
for (const label of ["Planet PP", "Speckle PP", "Circle radius [px]", "Peak-to-peak Lyot-stop power", "Best radius: 3 px"]) {
  assert(labels.includes(label), `Radius plot includes ${label}`);
}
assert(labels.includes("R = Speckle PP / Planet PP"));
assert(labels.includes("R=3.00e+0"), "Radius plot labels the reported ratio");
assert(labels.includes("R=2.67e+0"), "Radius plot can derive ratios from PP values");
drawRadiusPlot({history: [{radius_px: 1, planet_pp: 0, speckle_pp: 0.3, ratio_pp: null}]});
assert(labels.includes("R=undefined"), "Zero planet PP does not produce a misleading finite ratio");
state.lastOptimization = circleOptimization;
updateResultVisibility();
assert.equal(document.getElementById("radiusResults").hidden, false);
assert("optimization_radius_plot" in figurePayload());
drawRadiusPlot({best_index: 0, history: [circleOptimization.history[0]]});
state.lastOptimization = optimization;
updateResultVisibility();
assert.equal(document.getElementById("radiusResults").hidden, true);
assert.equal("optimization_radius_plot" in figurePayload(), false);
const ringOptimization = {best_index: 1, ring_grid: {inner_radii_px: [1, 2], outer_radii_px: [3, 4]}, history: [
  {name: "ring1", inner_radius_px: 1, outer_radius_px: 3, planet_pp: 1, speckle_pp: 2, ratio_pp: 2},
  {name: "ring2", inner_radius_px: 2, outer_radius_px: 4, planet_pp: 1, speckle_pp: 5, ratio_pp: 5},
  {name: "ring3", inner_radius_px: 2, outer_radius_px: 3, planet_pp: 0, speckle_pp: 1, ratio_pp: null},
]};
textPositions.length = 0;
drawRingPlot(ringOptimization);
for (const label of ["Inner radius [px]", "Outer radius [px]", "2.00e+0", "5.00e+0", "undefined", "Best ring: inner 2 px, outer 4 px"]) {
  assert(labels.includes(label), `Ring heatmap includes ${label}`);
}
const lowCell = textPositions.find(p => p.text === "2.00e+0");
const highCell = textPositions.find(p => p.text === "5.00e+0");
assert(highCell.x > lowCell.x, "Inner radius increases to the right");
assert(highCell.y < lowCell.y, "Outer radius increases upward");
state.lastOptimization = ringOptimization;
updateResultVisibility();
assert.equal(document.getElementById("ringResults").hidden, false);
assert.equal(document.getElementById("radiusResults").hidden, true);
assert("optimization_ring_plot" in figurePayload(), "Heatmap can be saved");
drawRingPlot({history: [ringOptimization.history[2]], best_index: null});
state.lastOptimization = optimization;
updateResultVisibility();
assert.equal(document.getElementById("ringResults").hidden, true);
const rows = els.tradeoffCandidates.children[0].children[0].children;
assert.equal(rows[1].children[1].textContent, "large_circle");
assert.equal(rows[1].children[4].textContent, "Best score, Pareto frontier");
{
  state.config = {n_fft: 32, focal_sampling: 2, source_markers: [
    {kind: "star", label: "star", x: 0, y: 0},
    {kind: "planet", label: "planet", x: 2, y: 0},
    {kind: "speckle", label: "s1", index: 0, x: 0, y: 2},
    {kind: "speckle", label: "s2", index: 1, x: 0, y: -2},
  ]};
  speckleChoices.push({checked: true, value: "0"}, {checked: false, value: "1"});
  state.mask = new Uint8Array(32 * 32);
  state.mask[5 * 32 + 24] = 1;
  state.mask[8 * 32 + 26] = 1;
  const originalMask = state.mask.slice();
  redrawSlm();
  assert(labels.includes("star") && labels.includes("planet") && labels.includes("s1"));
  assert(!labels.includes("s2"), "Deselected speckles are hidden");
  assert.deepEqual(slmViewExtent(), [-8, 8, -8, 8], "All sources remain in the full SLM view");
  const [cx, cy] = slmPixelToCanvas(24, 5);
  assert.deepEqual(canvasToSlmPixel({clientX: cx / 2, clientY: cy / 2}), [24, 5]);
  document.getElementById("starEnabled").checked = false;
  document.getElementById("specklesEnabled").checked = false;
  labels.length = 0;
  redrawSlm();
  assert(labels.includes("planet") && !labels.includes("star") && !labels.includes("s1"));
  const signature = currentComparisonSignature;
  currentComparisonSignature = () => "comparison-settings";
  state.lastComparisonSignature = "comparison-settings";
  state.lastProcess = "comparison";
  state.lastComparison = {sweeps: {planet_position: {}}};
  const placed = new Uint8Array(32 * 32);
  placed[10 * 32 + 20] = 1;
  state.placedMasks = {planet_position: placed, speckle_position: originalMask};
  rectangles.length = 0;
  redrawSlm();
  const painted = rectangles.filter(item => item.color === "rgba(29,112,184,0.65)");
  assert.equal(painted.length, 1, "Only the executed placement is shown");
  assert.deepEqual(painted[0].args, [20 * els.slm.width / 32, 21 * els.slm.height / 32, els.slm.width / 32, els.slm.height / 32]);
  assert(!rectangles.some(item => item.color === "rgba(215,25,28,0.92)"), "The unused template is hidden after a comparison");
  state.lastComparison.sweeps.speckle_position = {};
  assert.equal(displayedSlmMasks().length, 2, "Both executed placements are shown for a comparison");
  state.lastComparisonSignature = "old-settings";
  assert.equal(displayedSlmMasks()[0].mask, state.mask, "Editing restores the current modulation mask");
  state.lastProcess = "single";
  assert.equal(displayedSlmMasks()[0].mask, state.mask, "Single propagation shows the mask actually sent");
  assert.deepEqual(state.mask, originalMask, "Displaying placements preserves the editor mask");
  currentComparisonSignature = signature;
  document.getElementById("starEnabled").checked = true;
  document.getElementById("specklesEnabled").checked = true;
}

(async () => {
  const requests = [];
  currentLogPayload = () => ({});
  global.fetch = async (url, options) => {
    requests.push({url, body: JSON.parse(options.body)});
    return {ok: true, json: async () => ({paths: ["test.png"], directory: "test"})};
  };
  await saveFigures("planet_map");
  assert.equal(requests[0].url, "/api/save-figures");
  assert.deepEqual(Object.keys(requests[0].body.figures), ["planet_map"]);
  for (const [key] of figureDefinitions) document.querySelector(`input[data-figure="${key}"]`).checked = false;
  await saveFigures();
  assert.equal(requests.length, 1, "Empty selection does not save anything");
  setBusy = () => {};
  els.autoSaveFigures.checked = false;
  const winner = new Uint8Array(32 * 32); winner[16 * 32 + 16] = 1;
  const planetMask = new Uint8Array(32 * 32); planetMask[16 * 32 + 24] = 1;
  const speckleMask = new Uint8Array(32 * 32); speckleMask[24 * 32 + 16] = 1;
  const optimizationResult = {
    best_index: 0, best_mask_b64: packMask(winner),
    history: [{name: "winning-mask", score: 2, speckle_pp: 2, planet_pp: 1, ratio_pp: 2}],
    best_comparison: {sweeps: {speckle_position: {}, planet_position: {}},
      placed_masks: {speckle_position: packMask(speckleMask), planet_position: packMask(planetMask)}},
  };
  els.tool.value = "circle";
  document.getElementById("circleRadiusMin").value = "2";
  document.getElementById("circleRadiusMax").value = "6";
  let circleRequest;
  global.fetch = async (url, options) => {
    circleRequest = JSON.parse(options.body);
    return {ok: true, json: async () => optimizationResult};
  };
  await optimizeMask();
  assert.equal(circleRequest.circle_radius_min_px, 2);
  assert.equal(circleRequest.circle_radius_max_px, 6);
  assert.equal("search_radius_px" in circleRequest, false);
  assert.equal("group_size_px" in circleRequest, false);
  assert.deepEqual(state.mask, winner, "The editor receives the winning template");
  assert.deepEqual(state.baseMask, winner);
  assert.equal(state.previewMask, null);
  assert.equal(state.lastProcess, "optimization");
  assert.deepEqual(displayedSlmMasks().map(item => item.mask), [speckleMask, planetMask], "Optimization displays the winning applied placements");
  assert.equal(state.lastComparison, optimizationResult.best_comparison);
  assert.equal(state.lastComparisonSignature, currentComparisonSignature());
  els.comparisonType.value = "shared_mask";
  els.tool.value = "ring";
  for (const [id, value] of [["ringInnerMin", 1], ["ringInnerMax", 3], ["ringOuterMin", 2], ["ringOuterMax", 4], ["ringInnerStep", 1], ["ringOuterStep", 2]]) {
    document.getElementById(id).value = String(value);
  }
  const sharedResult = {
    ...optimizationResult,
    comparison_type: "shared_mask",
    best_comparison: {comparison_type: "shared_mask", sweeps: {shared_mask: {}},
      placed_masks: {shared_mask: packMask(winner)}},
  };
  let optimizationRequest;
  global.fetch = async (url, options) => {
    optimizationRequest = JSON.parse(options.body);
    return {ok: true, json: async () => sharedResult};
  };
  await optimizeMask();
  assert.equal(optimizationRequest.comparison_type, "shared_mask", "Optimization sends the selected comparison mode");
  assert.equal(optimizationRequest.optimization_mode, "ring_radii");
  assert.equal(optimizationRequest.ring_inner_radius_min_px, 1);
  assert.equal(optimizationRequest.ring_inner_radius_max_px, 3);
  assert.equal(optimizationRequest.ring_outer_radius_min_px, 2);
  assert.equal(optimizationRequest.ring_outer_radius_max_px, 4);
  assert.equal(optimizationRequest.ring_inner_radius_step_px, 1);
  assert.equal(optimizationRequest.ring_outer_radius_step_px, 2);
  assert.deepEqual(displayedSlmMasks(), [{mask: winner, color: "rgba(215,25,28,0.92)"}],
    "Same-mask optimization shows only the winning mask at its chosen pixels");
  assert.equal(state.lastComparison.comparison_type, "shared_mask");
  const beforeNoWinner = state.mask.slice();
  global.fetch = async () => ({ok: true, json: async () => ({best_mask_b64: null, best_index: null, history: [], stopped: true})});
  await optimizeMask();
  assert.deepEqual(state.mask, beforeNoWinner);
  assert.equal(state.placedMasks, null, "A run without a winner clears previous placements");
  assert(els.status.textContent.includes("No valid winning mask"));
})().catch(err => { console.error(err); process.exitCode = 1; });
"""
        result = subprocess.run(
            [shutil.which("node"), "-"],
            input=harness + script.replace("    init();", "") + checks,
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_planet_intensity_ratio_uses_square_root_field_scaling(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(0.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[calc.n_fft // 2 :, :] = True

        result = calc.propagate(
            mask,
            include_star=True,
            include_planet=True,
            include_speckles=False,
            star_planet_ratio=400.0,
            reference_phase_modulation_rad=0.0,
        )

        self.assertGreater(result["metrics"]["A_star"], 0.0)
        self.assertAlmostEqual(
            result["metrics"]["A_planet"],
            result["metrics"]["A_star"] / np.sqrt(400.0),
            places=10,
        )

    def test_phase_modulation_value_controls_selected_pixels(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(0.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[calc.n_fft // 2 :, :] = True

        result = calc.propagate(
            mask,
            include_star=False,
            include_planet=True,
            phase_modulation_rad=0.25,
        )

        self.assertEqual(result["phase_modulation_rad"], 0.25)
        self.assertTrue(np.all(result["phase_map_rad"][mask] == 0.25))
        self.assertTrue(np.all(result["phase_map_rad"][~mask] == 0.0))

    def test_limited_mask_preserves_each_sources_self_interference(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=10,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(2.0, 0.0),
                custom_speckle_offsets_lamD=((0.0, 2.0),),
                lyot_reference_scale=0.8,
                include_ghost=False,
                include_interference=False,
            )
        )
        yy, xx = np.indices((calc.n_fft, calc.n_fft), dtype=float)
        x = xx - (calc.n_fft - 1) / 2.0
        y = yy - (calc.n_fft - 1) / 2.0
        stop = calc.lyot_stop.astype(bool)

        def to_lyot(field):
            return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))

        sources = (
            ("star", (0.0, 0.0), 1.0),
            ("planet", (2.0, 0.0), 1.0 / np.sqrt(500.0)),
            ("speckle", (0.0, 2.0), 1.0 / np.sqrt(100.0)),
        )
        for source, offset, amplitude in sources:
            cx, cy = calc.source_pixel(offset)
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 2.0**2
            # Independent Fourier split: E(phi) = E_out + exp(-i phi) E_in.
            dx = calc.config.focal_shift_pixels[0] - offset[0] * calc.focal_sampling
            dy = calc.config.focal_shift_pixels[1] - offset[1] * calc.focal_sampling
            pupil = amplitude * calc._template.entrance_pupil() * np.exp(
                -2j * np.pi * (dx * x + dy * y) / calc.n_fft
            )
            focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
            inside = to_lyot(focal * mask)
            outside = to_lyot(focal * ~mask)
            # The tracked perfect reference removes 80% of the ideal star field.
            residual_scale = 0.2 if source == "star" else 1.0
            flags = dict(
                include_star=source == "star",
                include_planet=source == "planet",
                include_speckles=source == "speckle",
            )
            for mode in ("field", "intensity"):
                with self.subTest(source=source, subtraction_mode=mode):
                    sweep = calc.phase_sweep(mask, phase_steps=8, subtraction_mode=mode, **flags)
                    # Intensity subtraction removes |0.8 E|^2, rather than fields.
                    power_scale = residual_scale**2 if mode == "field" else (
                        1.0 - 0.8**2 if source == "star" else 1.0
                    )
                    baseline = power_scale * np.sum(
                        np.abs(outside[stop]) ** 2 + np.abs(inside[stop]) ** 2
                    )
                    overlap = np.sum(inside[stop] * np.conj(outside[stop]))
                    cross_term = 2.0 * power_scale * np.real(
                        np.exp(-1j * sweep["phases"]) * overlap
                    )
                    self.assertGreater(np.max(np.abs(cross_term)), 1e-8)
                    np.testing.assert_allclose(
                        sweep["powers"][source], baseline + cross_term,
                        rtol=1e-11, atol=1e-12,
                    )
                    self.assertGreater(sweep["metrics"][source]["M_pp"], 1e-8)
                    for phase in (0.0, 0.37, np.pi):
                        result = calc.propagate(mask, phase_modulation_rad=phase, **flags)
                        np.testing.assert_allclose(
                            result["fields"][source]["modulated"],
                            residual_scale * (outside + np.exp(-1j * phase) * inside),
                            rtol=1e-11, atol=1e-12,
                        )

    def test_empty_and_full_masks_do_not_modulate_individual_source_power(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=10,
                focal_sampling=2.0,
                companion_offset_lamD=(2.0, 0.0),
                custom_speckle_offsets_lamD=((0.0, 2.0),),
                lyot_reference_scale=0.8,
            )
        )
        for selected in (False, True):
            mask = np.full((calc.n_fft, calc.n_fft), selected, dtype=bool)
            for source in ("star", "planet", "speckle"):
                with self.subTest(full_mask=selected, source=source):
                    sweep = calc.phase_sweep(
                        mask, phase_steps=8,
                        include_star=source == "star",
                        include_planet=source == "planet",
                        include_speckles=source == "speckle",
                    )
                    power = sweep["powers"][source]
                    self.assertGreater(power[0], 0.0)
                    np.testing.assert_allclose(power, power[0], rtol=1e-12, atol=1e-12)

    def test_phase_sweep_reports_first_harmonic_modulation_metrics(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(0.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[calc.n_fft // 2 :, :] = True

        sweep = calc.phase_sweep(
            mask,
            phase_steps=4,
            include_star=True,
            include_planet=True,
            include_speckles=False,
            star_planet_ratio=400.0,
        )
        coherent_power = sweep["powers"]["coherent"]
        phases = sweep["phases"]
        c1 = np.mean(coherent_power * np.exp(-1j * phases))
        speckle_response = sweep["powers"]["coherent"] - sweep["powers"]["star"]
        planet_response = sweep["powers"]["incoherent"] - sweep["powers"]["star"]

        np.testing.assert_allclose(phases, [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
        np.testing.assert_allclose(sweep["responses"]["speckle_response"], speckle_response)
        np.testing.assert_allclose(sweep["responses"]["planet_response"], planet_response)
        self.assertAlmostEqual(
            sweep["metrics"]["coherent"]["M_harmonic"],
            float(abs(c1)),
            places=10,
        )
        self.assertAlmostEqual(
            sweep["metrics"]["coherent"]["phase_response"],
            float(np.angle(c1)),
            places=10,
        )
        self.assertIn("speckle_response", sweep["metrics"])
        self.assertIn("planet_response", sweep["metrics"])
        self.assertIn("R_response_harmonic", sweep["metrics"]["ratios"])
        self.assertIn("R_response_pp", sweep["metrics"]["ratios"])
        self.assertNotIn("R_mod_harmonic", sweep["metrics"]["ratios"])
        self.assertNotIn("R_mod_pp", sweep["metrics"]["ratios"])
        self.assertAlmostEqual(
            sweep["metrics"]["ratios"]["R_response_pp"],
            sweep["metrics"]["speckle_response"]["M_pp"]
            / sweep["metrics"]["planet_response"]["M_pp"],
            places=10,
        )

    def test_phase_sweep_incoherent_power_sums_star_and_planet_intensities(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(2.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[:, calc.n_fft // 2 :] = True

        sweep = calc.phase_sweep(
            mask,
            phase_steps=4,
            include_star=True,
            include_planet=True,
            include_speckles=False,
            star_planet_ratio=400.0,
        )

        np.testing.assert_allclose(
            sweep["powers"]["incoherent"],
            sweep["powers"]["star"] + sweep["powers"]["planet"],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_phase_sweep_reference_tracks_full_mask_global_phase_response(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(2.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.ones((calc.n_fft, calc.n_fft), dtype=bool)

        field_sweep = calc.phase_sweep(
            mask,
            phase_steps=4,
            subtraction_mode="field",
            include_star=True,
            include_planet=False,
            include_speckles=False,
        )
        intensity_sweep = calc.phase_sweep(
            mask,
            phase_steps=4,
            subtraction_mode="intensity",
            include_star=True,
            include_planet=False,
            include_speckles=False,
        )

        self.assertAlmostEqual(
            field_sweep["metrics"]["star"]["M_harmonic"],
            0.0,
            places=10,
        )
        self.assertEqual(intensity_sweep["subtraction_mode"], "intensity")
        self.assertAlmostEqual(
            intensity_sweep["metrics"]["star"]["M_harmonic"],
            0.0,
            places=10,
        )

    def test_reference_subtraction_scale_controls_tracked_reference_field_response(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(2.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[:, calc.n_fft // 2 :] = True

        full_reference = calc.phase_sweep(
            mask,
            phase_steps=4,
            subtraction_mode="field",
            lyot_reference_scale=1.0,
            include_star=True,
            include_planet=False,
            include_speckles=False,
        )
        no_reference = calc.phase_sweep(
            mask,
            phase_steps=4,
            subtraction_mode="field",
            lyot_reference_scale=0.0,
            include_star=True,
            include_planet=False,
            include_speckles=False,
        )

        self.assertAlmostEqual(full_reference["metrics"]["star"]["M_harmonic"], 0.0, places=10)
        self.assertIsNone(full_reference["metrics"]["ratios"]["R_response_harmonic"])
        self.assertIsNone(full_reference["metrics"]["ratios"]["R_response_pp"])
        self.assertGreater(no_reference["metrics"]["star"]["M_harmonic"], 0.0)
        self.assertEqual(no_reference["lyot_reference_scale"], 0.0)

    def test_coherent_metric_is_norm_of_complex_sum_not_sum_of_norms(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=10,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(3.0, 0.0),
                coherent_ring_speckle_count=2,
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[:, calc.n_fft // 2 :] = True

        result = calc.propagate(
            mask,
            include_star=True,
            include_planet=False,
            include_speckles=True,
            star_speckle_ratio=100.0,
            lyot_reference_scale=0.0,
        )
        stop = result["lyot_stop"].astype(bool)
        expected = np.sqrt(
            np.sum(np.abs(result["fields"]["coherent"]["delta"][stop]) ** 2)
        )

        self.assertAlmostEqual(result["metrics"]["A_coherent"], float(expected), places=10)
        self.assertNotAlmostEqual(
            result["metrics"]["A_coherent"],
            result["metrics"]["A_star"] + result["metrics"]["A_speckle"],
            places=10,
        )

    def test_can_select_subset_of_coherent_speckles(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=10,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(3.0, 0.0),
                coherent_ring_speckle_count=3,
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[:, calc.n_fft // 2 :] = True

        one = calc.propagate(
            mask,
            include_star=False,
            include_planet=False,
            include_speckles=True,
            selected_speckle_indices=[1],
            star_speckle_ratio=100.0,
        )
        all_speckles = calc.propagate(
            mask,
            include_star=False,
            include_planet=False,
            include_speckles=True,
            star_speckle_ratio=100.0,
        )

        self.assertEqual(one["selected_speckle_indices"], (1,))
        self.assertEqual(len(one["speckle_offsets_lamD"]), 1)
        self.assertEqual(len(all_speckles["speckle_offsets_lamD"]), 3)
        self.assertFalse(
            np.allclose(
                one["fields"]["speckle"]["delta"],
                all_speckles["fields"]["speckle"]["delta"],
            )
        )

    def test_speckle_intensity_ratio_uses_square_root_field_scaling(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=10,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(3.0, 0.0),
                custom_speckle_offsets_lamD=((3.0, 0.0),),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[:, calc.n_fft // 2 :] = True

        result = calc.propagate(
            mask,
            include_star=False,
            include_planet=True,
            include_speckles=True,
            star_planet_ratio=400.0,
            star_speckle_ratio=100.0,
        )

        self.assertAlmostEqual(
            result["metrics"]["A_speckle"],
            result["metrics"]["A_planet"] * np.sqrt(400.0 / 100.0),
            places=10,
        )

    def test_mask_template_placement_translates_without_clipping(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=10,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(2.0, 0.0),
            )
        )
        template = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        center = int(round((calc.n_fft - 1) / 2.0))
        template[center, center] = True
        template[center, center + 1] = True

        placed = calc.place_mask_template(template, (2.0, 0.0))
        target_x, target_y = placed["target_pixel"]

        self.assertTrue(placed["mask"][target_y, target_x])
        self.assertTrue(placed["mask"][target_y, target_x + 1])
        self.assertEqual(placed["selected_pixels"], 2)

    def test_comparison_uses_same_phase_sequence_and_separate_placements(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(2.0, 0.0),
                custom_speckle_offsets_lamD=((0.0, 2.0),),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        template = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        center = int(round((calc.n_fft - 1) / 2.0))
        template[center, center] = True

        comparison = calc.comparison_sweeps(
            template,
            selected_speckle_index=0,
            phase_steps=4,
            include_star=True,
            include_planet=True,
            include_speckles=True,
        )

        np.testing.assert_allclose(
            comparison["sweeps"]["speckle_position"]["phases"],
            comparison["sweeps"]["planet_position"]["phases"],
        )
        self.assertNotEqual(
            comparison["placements"]["speckle_position"]["target_pixel"],
            comparison["placements"]["planet_position"]["target_pixel"],
        )
        self.assertIn("main_speckle_pp", comparison["metrics"])
        self.assertIn("main_planet_pp", comparison["metrics"])
        self.assertIn("ratio_pp", comparison["metrics"])

    def test_shared_mask_comparison_preserves_user_position_and_uses_one_sweep(self) -> None:
        calc = SLMLyotResponseCalculator(SLMLyotConfig(
            pupil_pixels=8, focal_sampling=2.0, phase_mask=NoPhaseMask(),
            companion_offset_lamD=(2.0, 0.0),
            custom_speckle_offsets_lamD=((0.0, 2.0),),
        ))
        yy, xx = np.indices((calc.n_fft, calc.n_fft))
        cx, cy = 5, 6  # Deliberately away from both sources and the grid center.
        mask = (xx - cx)**2 + (yy - cy)**2 <= 4
        original = mask.copy()
        kwargs = dict(phase_steps=8, phase_start_rad=0.3, phase_span_rad=np.pi,
                      selected_speckle_indices=(0,), include_speckles=True)
        expected = calc.phase_sweep(mask, **kwargs)
        with patch.object(calc, "place_mask_template", side_effect=AssertionError("Mask moved")), \
             patch.object(calc, "phase_sweep", wraps=calc.phase_sweep) as sweep_call:
            result = calc.comparison_sweeps(
                mask, comparison_type="shared_mask", phase_steps=8,
                phase_start_rad=0.3, phase_span_rad=np.pi,
            )
        self.assertEqual(sweep_call.call_count, 1)
        self.assertEqual(set(result["sweeps"]), {"shared_mask"})
        sweep = result["sweeps"]["shared_mask"]
        np.testing.assert_array_equal(mask, original)
        np.testing.assert_array_equal(sweep["slm_region_mask"], original)
        np.testing.assert_allclose(sweep["phases"], expected["phases"])
        for response in ("planet_response", "speckle_response"):
            np.testing.assert_allclose(sweep["responses"][response], expected["responses"][response])
        self.assertTrue(result["metrics"]["complete"])
        self.assertEqual(result["metrics"]["main_planet_pp"], sweep["metrics"]["planet_response"]["M_pp"])
        self.assertEqual(result["metrics"]["main_speckle_pp"], sweep["metrics"]["speckle_response"]["M_pp"])
        server = _SLMHtmlServer.__new__(_SLMHtmlServer)
        server.calculator = calc
        payload = server.comparison_payload(result, speckle_mask=mask, planet_mask=mask)
        self.assertEqual(payload["comparison_type"], "shared_mask")
        self.assertEqual(set(payload["placed_masks"]), {"shared_mask"})
        np.testing.assert_array_equal(server.decode_mask(payload["placed_masks"]["shared_mask"]), mask)

    def test_phase_zero_matching_calibrates_speckle_ratio(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(2.0, 0.0),
                custom_speckle_offsets_lamD=((0.0, 2.0),),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )

        calibration = calc.calibrate_star_speckle_ratio_for_phase_zero_match(
            selected_speckle_indices=(0,),
            star_planet_ratio=400.0,
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        sweep = calc.phase_sweep(
            mask,
            phase_steps=2,
            include_star=True,
            include_planet=True,
            include_speckles=True,
            selected_speckle_indices=(0,),
            star_planet_ratio=400.0,
            star_speckle_ratio=calibration["star_speckle_ratio"],
        )

        self.assertGreater(calibration["star_speckle_ratio"], 0.0)
        self.assertAlmostEqual(
            sweep["powers"]["coherent"][0],
            sweep["powers"]["incoherent"][0],
            places=10,
        )

    def test_html_phase_zero_match_payload_is_serialized(self) -> None:
        server = _SLMHtmlServer()
        calibration = server.calculator.calibrate_star_speckle_ratio_for_phase_zero_match(
            selected_speckle_indices=(0,),
            star_planet_ratio=500.0,
        )
        mask = np.zeros((server.calculator.n_fft, server.calculator.n_fft), dtype=bool)
        sweep = server.calculator.phase_sweep(
            mask,
            phase_steps=2,
            include_star=True,
            include_planet=True,
            include_speckles=True,
            selected_speckle_indices=(0,),
            star_planet_ratio=500.0,
            star_speckle_ratio=calibration["star_speckle_ratio"],
        )
        sweep["phase_zero_match"] = calibration

        payload = server.phase_sweep_payload(sweep)

        self.assertIn("phase_zero_match", payload)
        self.assertAlmostEqual(
            payload["phase_zero_match"]["star_speckle_ratio"],
            calibration["star_speckle_ratio"],
            places=10,
        )

    def test_circle_optimization_varies_only_radius(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(2.0, 0.0),
                custom_speckle_offsets_lamD=((0.0, 2.0),),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        center = int(round((calc.n_fft - 1) / 2.0))
        yy, xx = np.indices((calc.n_fft, calc.n_fft))
        template = (xx - center) ** 2 + (yy - center) ** 2 <= 2**2

        optimization = calc.optimize_common_mask(
            template,
            optimization_mode="circle_size",
            circle_center_pixel=(center, center),
            circle_radius_min_px=1,
            circle_radius_max_px=3,
            selected_speckle_index=0,
            phase_steps=2,
            iterations=3,
            include_star=True,
            include_planet=True,
            include_speckles=True,
        )

        self.assertEqual(optimization["optimization_mode"], "circle_size")
        self.assertEqual(
            [item["name"] for item in optimization["history"]],
            ["circle_radius_1px", "circle_radius_2px", "circle_radius_3px"],
        )
        for item in optimization["history"]:
            if item["ratio_pp"] is None:
                self.assertEqual(item["score"], -np.inf)
            else:
                self.assertEqual(item["score"], item["ratio_pp"])
        for item in optimization["history"]:
            radius = int(item["name"].split("_")[-1].removesuffix("px"))
            self.assertEqual(item["radius_px"], radius)
            expected = (xx - center) ** 2 + (yy - center) ** 2 <= radius**2
            np.testing.assert_array_equal(item["mask"], expected)

        server = _SLMHtmlServer()
        server.calculator = calc
        self.assertIsNotNone(optimization["best_index"])
        payload = server.optimization_payload(optimization)
        best = optimization["history"][optimization["best_index"]]
        np.testing.assert_array_equal(server.decode_mask(payload["best_mask_b64"]), best["mask"])
        for name, sweep in best["comparison"]["sweeps"].items():
            np.testing.assert_array_equal(
                server.decode_mask(payload["best_comparison"]["placed_masks"][name]),
                sweep["slm_region_mask"],
            )
            self.assertNotEqual(
                payload["best_comparison"]["placed_masks"][name], payload["best_mask_b64"]
            )
        json.dumps(payload)
        no_winner = server.optimization_payload({**optimization, "best_index": None})
        self.assertIsNone(no_winner["best_mask_b64"])
        self.assertIsNone(no_winner["best_comparison"])

    def test_shared_mask_optimization_preserves_selected_circle_position(self) -> None:
        calc = SLMLyotResponseCalculator(SLMLyotConfig(
            pupil_pixels=8, focal_sampling=2.0, phase_mask=NoPhaseMask(),
            companion_offset_lamD=(2.0, 0.0),
            custom_speckle_offsets_lamD=((0.0, 2.0),),
            ghost_fraction=0.0, include_ghost=False,
        ))
        cx, cy = calc.n_fft // 2 + 2, calc.n_fft // 2 - 1
        yy, xx = np.indices((calc.n_fft, calc.n_fft))
        template = (xx - cx)**2 + (yy - cy)**2 <= 2**2
        with patch.object(calc, "place_mask_template", side_effect=AssertionError("Mask moved")):
            optimization = calc.optimize_common_mask(
                template, optimization_mode="circle_size", comparison_type="shared_mask",
                circle_center_pixel=(cx, cy), circle_radius_min_px=1,
                circle_radius_max_px=3, iterations=3, phase_steps=4,
            )
        self.assertEqual(optimization["comparison_type"], "shared_mask")
        for item in optimization["history"]:
            expected = (xx - cx)**2 + (yy - cy)**2 <= item["radius_px"]**2
            np.testing.assert_array_equal(item["mask"], expected)
            comparison = item["comparison"]
            self.assertEqual(comparison["comparison_type"], "shared_mask")
            self.assertEqual(set(comparison["sweeps"]), {"shared_mask"})
            np.testing.assert_array_equal(comparison["sweeps"]["shared_mask"]["slm_region_mask"], expected)
        self.assertIsNotNone(optimization["best_index"])
        server = _SLMHtmlServer()
        server.calculator = calc
        payload = server.optimization_payload(optimization)
        self.assertEqual(payload["comparison_type"], "shared_mask")
        self.assertEqual(payload["best_comparison"]["comparison_type"], "shared_mask")
        self.assertEqual(payload["best_comparison"]["placed_masks"],
                         {"shared_mask": payload["best_mask_b64"]})

    def test_ring_optimization_searches_both_radii(self) -> None:
        calc = SLMLyotResponseCalculator(SLMLyotConfig(
            pupil_pixels=12, focal_sampling=2.0, phase_mask=NoPhaseMask(),
            companion_offset_lamD=(2.0, 0.0),
            custom_speckle_offsets_lamD=((0.0, 2.0),),
            ghost_fraction=0.0, include_ghost=False,
        ))
        cx, cy = calc.n_fft // 2 + 2, calc.n_fft // 2 - 1
        yy, xx = np.indices((calc.n_fft, calc.n_fft))
        distance2 = (xx - cx)**2 + (yy - cy)**2
        template = (distance2 >= 1) & (distance2 <= 4)
        options = dict(optimization_mode="ring_radii", circle_center_pixel=(cx, cy),
                       ring_inner_radius_min_px=1, ring_inner_radius_max_px=3,
                       ring_outer_radius_min_px=2, ring_outer_radius_max_px=4,
                       ring_inner_radius_step_px=1, ring_outer_radius_step_px=2, phase_steps=4)
        server = _SLMHtmlServer()
        server.calculator = calc
        for comparison_type in ("shared_mask", "separate"):
            with self.subTest(comparison_type=comparison_type):
                optimization = calc.optimize_common_mask(template, comparison_type=comparison_type, **options)
                pairs = {(item["inner_radius_px"], item["outer_radius_px"]) for item in optimization["history"]}
                self.assertEqual(pairs, {(1, 2), (1, 4), (2, 4), (3, 4)})
                for item in optimization["history"]:
                    expected = (distance2 >= item["inner_radius_px"]**2) & (distance2 <= item["outer_radius_px"]**2)
                    np.testing.assert_array_equal(item["mask"], expected)
                    self.assertEqual(item["comparison"]["comparison_type"], comparison_type)
                    if comparison_type == "shared_mask":
                        np.testing.assert_array_equal(item["comparison"]["sweeps"]["shared_mask"]["slm_region_mask"], expected)
                self.assertIsNotNone(optimization["best_index"])
                payload = server.optimization_payload(optimization)
                self.assertEqual(payload["ring_grid"], {"inner_radii_px": [1, 2, 3], "outer_radii_px": [2, 4]})
                self.assertEqual(len(payload["history"]), 4)
                json.dumps(payload)
        for step_key in ("ring_inner_radius_step_px", "ring_outer_radius_step_px"):
            with self.assertRaisesRegex(ValueError, "positive step"):
                calc.optimize_common_mask(template, **{**options, step_key: 0})
        with self.assertRaisesRegex(ValueError, "at least one inner"):
            calc.optimize_common_mask(template, **{**options, "ring_inner_radius_min_px": 4, "ring_inner_radius_max_px": 5})
        with self.assertRaisesRegex(ValueError, "boundary"):
            calc.optimize_common_mask(template, **{**options, "ring_outer_radius_max_px": calc.n_fft})
        stopped = calc.optimize_common_mask(template, **options, stop_requested=lambda: True)
        self.assertEqual(stopped["history"], [])
        self.assertIsNone(stopped["best_index"])

    def test_incoherent_map_combines_star_and_planet_without_speckles(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=10,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(3.0, 0.0),
                coherent_ring_speckle_count=1,
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[:, calc.n_fft // 2 :] = True

        result = calc.propagate(
            mask,
            include_star=True,
            include_planet=True,
            include_speckles=True,
            star_planet_ratio=400.0,
            star_speckle_ratio=100.0,
        )
        expected_map = np.sqrt(
            np.abs(result["fields"]["star"]["delta"]) ** 2
            + np.abs(result["fields"]["planet"]["delta"]) ** 2
        )
        stop = result["lyot_stop"].astype(bool)
        expected_a = float(np.sqrt(np.sum(expected_map[stop] ** 2)))

        np.testing.assert_allclose(result["fields"]["incoherent"]["delta"], expected_map)
        self.assertFalse(
            np.allclose(
                result["fields"]["incoherent"]["delta"],
                np.sqrt(
                    np.abs(result["fields"]["coherent"]["delta"]) ** 2
                    + np.abs(result["fields"]["planet"]["delta"]) ** 2
                ),
            )
        )
        self.assertAlmostEqual(result["metrics"]["A_incoherent"], expected_a, places=10)
        self.assertAlmostEqual(
            result["metrics"]["R1"],
            result["metrics"]["A_coherent"] / (result["metrics"]["A_incoherent"] + 1e-30),
            places=10,
        )

    def test_disabling_star_skips_lyot_reference_subtraction(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(2.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )

        def fail_reference(_sim):
            raise AssertionError("No Lyot reference should be built when the star is disabled.")

        calc._reference_field = fail_reference
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[calc.n_fft // 2 :, :] = True

        result = calc.propagate(
            mask,
            include_star=False,
            include_planet=True,
            include_speckles=False,
            star_planet_ratio=400.0,
        )

        stop = result["lyot_stop"].astype(bool)
        expected_a = float(
            np.sqrt(np.sum(np.abs(result["fields"]["planet"]["delta"][stop]) ** 2))
        )
        self.assertNotIn("A_star", result["metrics"])
        self.assertNotIn("A_coherent", result["metrics"])
        self.assertNotIn("R1", result["metrics"])
        self.assertAlmostEqual(result["metrics"]["A_incoherent"], expected_a, places=10)

    def test_save_result_writes_complex_fields_and_source_state(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(0.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[calc.n_fft // 2 :, :] = True
        result = calc.propagate(mask, include_star=True, include_planet=True)

        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            save_slm_lyot_result(tmp.name, result)
            data = np.load(tmp.name)
            self.assertIn("delta_E_L_star", data.files)
            self.assertIn("delta_E_L_planet", data.files)
            self.assertIn("delta_E_L_coherent", data.files)
            self.assertTrue(bool(data["include_star"]))
            self.assertTrue(np.iscomplexobj(data["delta_E_L_star"]))
            self.assertIn("phase_modulation_rad", data.files)

    def test_html_config_includes_slm_source_markers(self) -> None:
        server = _SLMHtmlServer()
        payload = server.config_payload()
        labels = {marker["label"] for marker in payload["source_markers"]}
        speckles = [marker for marker in payload["source_markers"] if marker.get("kind") == "speckle"]

        self.assertIn("star", labels)
        self.assertIn("planet", labels)
        self.assertEqual(len(speckles), default_slm_gui_config().coherent_ring_speckle_count)
        self.assertEqual(len(speckles), 1)
        self.assertTrue(
            all("index" in marker for marker in payload["source_markers"] if marker.get("kind") == "speckle")
        )
        self.assertEqual(payload["n_fft"], server.calculator.n_fft)

    def test_html_source_geometry_can_move_planet_and_speckles(self) -> None:
        server = _SLMHtmlServer()
        payload = server.update_source_geometry(
            planet_x=2.5,
            planet_y=-1.25,
            speckle_offsets=[[1.0, 2.0], [-3.5, 0.5]],
        )
        markers = payload["source_markers"]
        planet = next(marker for marker in markers if marker["kind"] == "planet")
        speckles = [marker for marker in markers if marker["kind"] == "speckle"]

        self.assertEqual((planet["x"], planet["y"]), (2.5, -1.25))
        self.assertEqual(
            [(marker["x"], marker["y"]) for marker in speckles],
            [(1.0, 2.0), (-3.5, 0.5)],
        )
        self.assertEqual(server.calculator.config.companion_offset_lamD, (2.5, -1.25))
        self.assertEqual(
            server.calculator.config.custom_speckle_offsets_lamD,
            ((1.0, 2.0), (-3.5, 0.5)),
        )

    def test_html_source_geometry_can_remove_all_speckles(self) -> None:
        server = _SLMHtmlServer()
        payload = server.update_source_geometry(
            planet_x=4.0,
            planet_y=0.0,
            speckle_offsets=[],
        )
        self.assertFalse(
            any(marker["kind"] == "speckle" for marker in payload["source_markers"])
        )

        mask = np.zeros((server.calculator.n_fft, server.calculator.n_fft), dtype=bool)
        result = server.calculator.propagate(
            mask,
            include_star=False,
            include_planet=False,
            include_speckles=True,
        )

        self.assertEqual(result["speckle_offsets_lamD"], ())
        self.assertEqual(result["selected_speckle_indices"], ())
        self.assertNotIn("A_speckle", result["metrics"])

    def test_html_log_path_uses_date_time_planet_and_process(self) -> None:
        server = _SLMHtmlServer()
        payload = {
            "process_kind": "comparison",
            "source_markers": [
                {"kind": "planet", "x": 2.5, "y": -1.25},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            server.log_root = Path(tmpdir)
            path = server.save_log_payload(
                payload,
                now=datetime(2026, 10, 9, 14, 5, 6),
            )

            self.assertEqual(path.parent.name, "2026-10-09")
            self.assertEqual(
                path.name,
                "140506_planet_x_p2p500_y_m1p250_comparison_log.json",
            )
            self.assertTrue(path.exists())

    def test_html_log_payload_is_concise_but_keeps_physics_summary(self) -> None:
        server = _SLMHtmlServer()
        payload = {
            "saved_at": "2026-10-09T14:05:06.000Z",
            "process_kind": "optimization",
            "n_fft": 16,
            "mask_pixels": 12,
            "mask_fraction": 12 / 256,
            "controls": {
                "include_star": True,
                "include_planet": True,
                "include_speckles": True,
                "star_planet_ratio": 500.0,
                "star_speckle_ratio": 100.0,
                "reference_subtraction_percent": 100.0,
                "subtraction_mode": "field",
                "phase_steps": 4,
            },
            "source_markers": [
                {"kind": "planet", "x": 2.5, "y": -1.25},
                {"kind": "speckle", "index": 0, "x": 0.0, "y": 2.0},
            ],
            "phase_sweep": {
                "phase_steps": 4,
                "phases": [0.0, 1.0, 2.0, 3.0],
                "powers": {"coherent": [1.0, 2.0, 1.0, 0.0]},
                "responses": {"speckle_response": [0.0, 1.0, 0.0, -1.0]},
                "metrics": {"ratios": {"R_response_pp": 2.0}},
            },
            "comparison": {
                "measurement_plane": "lyot",
                "primary_metric": "ratio_pp",
                "metrics": {"ratio_pp": 3.0},
                "placed_masks": {"speckle_position": "large-mask"},
                "sweeps": {
                    "speckle_position": {
                        "phase_steps": 2,
                        "phases": [0.0, 3.141592653589793],
                        "powers": {"coherent": [1.0, 0.5]},
                        "responses": {},
                        "metrics": {},
                    },
                },
            },
            "optimization": {
                "primary_metric": "ratio_pp",
                "optimization_mode": "pixel_groups",
                "search_radius_px": 24,
                "group_size_px": 4,
                "iterations": 2,
                "best_index": 1,
                "best_mask_b64": "large-best-mask",
                "history": [
                    {"name": "candidate_0", "score": 1.0, "mask": [[1]], "comparison": {"huge": True}},
                    {"name": "candidate_1", "score": 2.0, "ratio_pp": 4.0, "mask": [[1]]},
                ],
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            server.log_root = Path(tmpdir)
            path = server.save_log_payload(payload, now=datetime(2026, 10, 9, 14, 5, 6))
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["log_format"], "concise_slm_physics_v1")
        self.assertIn("physics_context", saved)
        self.assertEqual(saved["setup"]["controls"]["star_planet_ratio"], 500.0)
        self.assertEqual(saved["phase_sweep"]["phases_rad"], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(
            saved["comparison"]["sweeps"]["speckle_position"]["powers"]["coherent"],
            [1.0, 0.5],
        )
        self.assertEqual(saved["optimization"]["best_candidate"]["name"], "candidate_1")
        saved_text = json.dumps(saved)
        self.assertNotIn("best_mask_b64", saved_text)
        self.assertNotIn("placed_masks", saved_text)
        self.assertNotIn("\"mask\"", saved_text)
        self.assertNotIn("\"comparison\"", json.dumps(saved["optimization"]["history"]))

    def test_html_saves_figure_payloads_to_dated_artifact_directory(self) -> None:
        server = _SLMHtmlServer()
        payload = {
            "process_kind": "optimization",
            "source_markers": [
                {"kind": "planet", "x": 4.0, "y": 0.0},
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            server.log_root = Path(tmpdir)
            paths = server.save_figure_payloads(
                payload,
                {"slm_mask": "data:image/png;base64,aGVsbG8="},
                now=datetime(2026, 10, 9, 14, 5, 6),
            )

            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].parent.name, "2026-10-09")
            self.assertEqual(
                paths[0].name,
                "140506_planet_x_p4p000_y_p0p000_optimization_slm_mask.png",
            )
            self.assertEqual(paths[0].read_bytes(), b"hello")

    def test_html_phase_sweep_payload_includes_response_arrays(self) -> None:
        server = _SLMHtmlServer()
        mask = np.zeros((server.calculator.n_fft, server.calculator.n_fft), dtype=bool)
        mask[server.calculator.n_fft // 2 :, :] = True
        sweep = server.calculator.phase_sweep(mask, phase_steps=4)

        payload = server.phase_sweep_payload(sweep)

        self.assertIn("responses", payload)
        self.assertIn("speckle_response", payload["responses"])
        self.assertIn("planet_response", payload["responses"])
        self.assertEqual(len(payload["responses"]["speckle_response"]), 4)
        self.assertEqual(len(payload["responses"]["planet_response"]), 4)
        self.assertIn("R_response_harmonic", payload["metrics"]["ratios"])
        self.assertIn("R_response_pp", payload["metrics"]["ratios"])
        self.assertNotIn("R_mod_harmonic", payload["metrics"]["ratios"])
        self.assertNotIn("R_mod_pp", payload["metrics"]["ratios"])


if __name__ == "__main__":
    unittest.main()
