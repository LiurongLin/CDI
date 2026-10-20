# Script Layout

Standalone utility scripts are grouped by function:

- `scripts/fits/`: FITS cube assembly, deduplication, and frame-axis reduction.
- `scripts/frequency_maps/`: FFT/windowed map builders and band-limited reconstructions.
- `scripts/frequency_maps/ring_planet_data_processing_pipeline.py`: end-to-end pipeline for combining ring-planet cubes, extracting aperture time series, plotting FFTs, and building coherence/incoherence maps.
- `scripts/regions/`: region definitions, aperture-sum extraction, and annulus-region analysis.
- `scripts/visualization/`: plotting, contact-sheet rendering, GIF export, and FFT visualization helpers.
- `scripts/coronagraph/`: direct coronagraph entrypoints and thin app-level launchers.

There are no root-level standalone `*.py` scripts anymore; use the grouped paths under `scripts/`.

## Direct science-plane intensity in the SLM GUI

Launch from the repository root:

```bash
.venv/bin/python scripts/coronagraph/slm_gui.py
```

Open the printed browser URL (default `http://127.0.0.1:8765`).

1. Under **Sources**, set the planet coordinates and speckle coordinates, then
   click **Update Sources**. Choose **Target speckle**, the enabled sources and
   speckles, and the star/planet and star/speckle ratios.
2. At the top of **Run**, choose **Lyot plane** or **Final science plane** under
   **Modulation measurement plane**. Draw or load the SLM modulation mask with the existing mask tools. Commit a
   parameterized shape using **Add Region** or **Replace Mask** as usual.
3. Under **Final science measurement regions**, set **Measurement radius [lambda/D]**.
   This common radius controls fixed circles around every configured speckle and
   the planet. It does not change the SLM mask.
4. Under **Phase sweep**, set **Steps**, **Start [rad]**, and **Span [rad]**.
   Set **Reference subtraction [%]** under **Advanced settings**. Click
   **Measure modulation** at the top of Run to use the selected plane.
   **Measure science intensity** always runs the final-science scan.
   The scan samples `start + span * k / count`, excluding the endpoint. Source
   geometry, circle radius, and the SLM mask are captured once per scan.
5. The results dialog shows the physical final science focal image at the
   **first sampled phase**, labeled circles, both mean pixel intensity curves,
   and each curve's sampled maximum, minimum, and peak-to-peak difference.
   **Open results** reopens the dialog. The log image display uses one color
   scale across the scan. Means use linear intensity divided by one unit-star
   direct-PSF peak, even if the star is disabled. Increasing the phase count
   resolves extrema more finely.
6. **Save Log** writes the phases, curves, region centers/radius/pixel counts,
   normalization, source settings, and modulation results to
   `interactive_slm_logs/YYYY-MM-DD/`, with the usual time/planet/process filename.
   **Save results** downloads JSON and saves a dated science `*_result.npz` with
   the sampled phases, curves, geometry, extrema, normalization, first-phase
   image, and captured SLM mask. **Save image** or **Save figures** exports the
   science image and curves using the existing figure selection controls.

The science measurement includes all enabled light in both circles. Star and
selected speckles interfere coherently; the planet contributes in intensity.
Existing Lyot reference subtraction and configured ghost coherence remain in
force. The diagnostic field/intensity subtraction selector does not alter this
measurement. Optional phase-zero matching calibrates one fixed source amplitude
as described below. No star-only, companion-background, or
phase-zero image is subtracted. Circles must fit within the science image and
contain at least one pixel center. Changing controls after a scan affects the
next scan; saved measurement geometry and source settings describe the completed
scan.

The results dialog shows only the selected measurement plane. Switching the
plane selector immediately hides the other plane's plots, tables, and metrics.
If the selected plane has no saved calculation yet, the dialog asks you to run
that measurement. Previous results remain available when switching back. Image
exports and result/log payloads also use only the selected plane.

### Match phase-0 levels in the selected plane

Enable **Match phase-0 levels in selected measurement plane** under Sources,
then use **Measure modulation** or **Run comparison**.

- **Lyot plane**: retains the existing phase-zero coherent/incoherent integrated
  power matching, including for Measure modulation.
- **Final science plane**: calibrates the common **Star/Speckle** ratio until the
  **total mean physical intensity** in the **Target speckle** circle equals the
  total mean intensity in the planet circle at zero applied SLM phase. The
  calibration includes stellar light, all enabled speckles, planet light, and
  enabled ghosts. No background is removed from the measured curves.

The calibrated ratio is held fixed for every phase and every mask placement;
normalization also stays fixed. The result note displays the matched baseline
intensities and effective ratio. Calibration details are included in JSON and
NPZ exports. Unchecking the option uses the entered source ratio directly.
Matching is always evaluated at phase zero, even if the requested scan starts at
another phase. With several speckles, this matches the selected target and the
planet; other speckle regions are measured but not independently rescaled.
The target speckle and planet must be enabled; settings without a positive
physical matching solution return an explanatory error.

### Separate-mask comparison in the final science plane

In the browser GUI, choose **Final science plane** under **Modulation measurement
plane**, then **Separate masks centered on each source** under **Comparison type**.
Draw or load the SLM mask template, set **Measurement radius [lambda/D]** and the
phase settings, and click **Run comparison** or **Measure modulation**.

The same mask shape is translated to each configured speckle and the planet, one
source at a time. Its selected-pixel centroid is used as the template anchor, so
an off-axis drawn template is still centered on each requested source. The mask
then stays fixed for that placement's full phase scan. The input template is
preserved. All enabled sources remain present and every science measurement
circle stays at its original source coordinates throughout every scan.

The science results contain a summary of each source region's maximum, minimum,
and peak-to-peak intensity **when the mask is centered on that source**. The
**Each source with its own centered mask** overview overlays each source's
own-region curve from that source's mask placement: the planet curve comes from
the planet-centered scan, while each speckle curve comes from its own scan.
These are separate scans plotted together, using the common physical
normalization. The overview has its own Save image control.
Use **Inspect one scan: SLM mask placement** to select a detailed run: its image
and all region curves show both the intended response and the effects in other
companion regions. This detailed view initially selects the first speckle; the
planet's curve there is its response to the speckle-centered mask. The
SLM canvas displays the selected run's actual mask. All placements share one
physical intensity normalization and one image color scale. No star-only or
companion-background image is subtracted. Optional phase-zero matching uses the
science-region intensities and keeps one calibrated source ratio for all runs.

**Same mask at user-selected position** instead makes a single science scan at
the original mask pixels and compares every region from that same scan.
**Run speckle sweep** and **Run planet sweep** use the selected plane too; in
science mode they scan only the target speckle placement or planet placement,
while still measuring all regions and keeping all enabled light present.

**Save Log** and **Save results** retain every placement's phase values, all
region curves, region geometry, normalization, extrema, and the cross-region
peak-to-peak modulation matrix. The NPZ also stores the original template and
every placed mask; arrays are prefixed with their placement, for example
`planet__planet_mean_intensity` and `speckle_1__planet_mean_intensity`.
**Save image / Save figures** export the currently inspected placement's image
and curves, with the placement included in the filenames. Restart the Python
server and reload the browser page to load these controls after updating code.

### Optimize final-science modulation: s1 versus planet

In the browser GUI:

1. Select **Final science plane** and the desired **Comparison type**.
2. Enable **s1** (the first configured speckle) and **Planet**. Other enabled
   sources still propagate and contribute their light to the measured regions.
3. Set **Measurement radius [lambda/D]**, **Steps**, **Start**, and **Span**.
   The measurement radius is held fixed throughout optimization.
4. Choose the SLM **Shape**. Set the circle radius range, ring inner/outer ranges,
   or pixel-group search settings under **Optimization**. These sizes are SLM
   mask pixels, independent of the measurement-circle radius in lambda/D.
5. Optionally enable **Match phase-0 levels in selected measurement plane**.
   Science optimization calibrates **s1 versus planet** once before the search,
   regardless of the Target speckle selected for manual measurements. The
   calibrated source ratio stays fixed across every candidate and phase.
6. Click **Start optimization**.

The score is `s1 intensity PP / planet intensity PP`, where PP is the maximum
minus minimum of each region's mean total physical science intensity. Higher
scores are preferred. Only these two regions are scored and shown in the winning
comparison; other enabled sources are retained in the optical calculation.
No star-only or companion-background subtraction is applied, and normalization
is common to all phases and candidates. A zero planet PP has an undefined ratio
and is not selected as a winning finite-score candidate.

**Separate masks** evaluates each candidate centered on s1 and on the planet in
two scans, then compares each source's own-region PP. **Same mask** evaluates one
candidate at its chosen pixels and compares both regions in that same scan.
The results dialog opens at the top with **Final science optimization — s1 vs
planet**, the winning candidate, score, and both modulation levels. Radius/ring
and trade-off plots follow; the winning science image and phase curves are
below. The plane selector is locked while the calculation runs, and the
completed result is displayed in the plane that was requested. The winner is
also applied to the editor.
**Stop** finishes the current science candidate and keeps completed candidates.

**Save Log / Save results** export the science scoring plane, s1/planet metrics,
candidate score history, fixed region geometry, source calibration, and the
winner's phase curves and masks. NPZ includes `optimization_json` alongside the
winning science-comparison arrays. Plot exports use the selected plane. Choosing
**Lyot plane** continues to use the existing Lyot optimizer.

### Desktop Matplotlib GUI

For the desktop GUI defined in `coronagraph/interactive_slm_gui.py`, launch:

```bash
.venv/bin/python -m coronagraph.interactive_slm_gui
```

Set the SLM mask, source checkboxes, **Star/Planet**, **Ref %**, and **Steps** in
the main window. Under **Intensity measurement plane**, choose:

- **Lyot plane**: the existing integrated Lyot-stop power modulation scan.
- **Final science**: physical intensity after propagation through the Lyot stop,
  including the existing coronagraph reference treatment and enabled ghost light.

Click **Eval Mod** or **Intensity** to scan the selected plane. In the measurement
window, set **ROI r [lambda/D]** and **Star/Speckle**, then click **Scan Intensity**
to repeat. Each scan covers 0 to 2 pi with the endpoint excluded, using the main
window's Steps setting. Every configured speckle (`speckle_1`, `speckle_2`, ...)
and the planet receive a fixed circular measurement region of the common radius.
There is no need to select a single target speckle. Source toggles control the
light propagated; each region measures all enabled light falling inside it.

The window shows the first sampled phase's physical intensity image, labeled
circles, a separate mean-intensity curve per companion, and a table of every
region's sampled maximum, minimum, and peak-to-peak modulation. All phases use
one unit-star direct-PSF normalization. No background is subtracted. Measurement controls never edit the SLM modulation mask.

After a scan, the main window's **Save Result**, **Save Log**, and **Save Fig**
export the NPZ, JSON log, and figure to the existing dated `interactive_slm_logs`
directory. Filenames identify `phase_sweep` (Lyot) or `science_intensity`; science exports
contain the measured plane, every companion's curve and modulation statistics,
and the fixed measurement geometry. **Apply** still displays the existing Lyot
field diagnostics. The field/intensity display and subtraction selectors do not
alter the physical intensity measurement.
