from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from PIL import Image, ImageOps, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from coronagraph.cli import default_sim_kwargs
from coronagraph.masks import VortexPhaseMask
from coronagraph.region_shapes import annulus_radii_from_width
from coronagraph.simulator import CoronagraphSimulator, resolve_phase_screen_path


OUTPUT_DIR = Path(__file__).resolve().parent
DPI = 180

PLANET_CENTER_LAMD = (4.5, -4.5)
PLANET_FLUX_RATIO = 1e-3
PLANET_EVAL_RADIUS_LAMD = 0.5
RING_WIDTH_LAMD = 2.0
PHASE_CYCLES = 2.0
MODULATION_STEPS_PER_CYCLE = 10
REPRESENTATIVE_PHASES = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=float)
SPECKLE_COUNT = 3
SPECKLE_RADIUS_MATCH_TOL_LAMD = 0.35
SPECKLE_APERTURE_RADIUS_LAMD = 0.35
SPECKLE_MIN_SEPARATION_LAMD = 1.2

PHASE_MASK_NAME = "coc_phase_mask_modulation_vortex_charge_2_ring_width_2p0_4phase_quadrature_square.png"
FINAL_PSF_NAME = "coc_final_psf_overlay_planet_circle_eval_radius_0p5_from_ring_width_2p0_with_companion_quadrature_square.png"
TIME_SERIES_NAME = "coc_circle_intensity_sum_timeseries_vortex_charge_2_ring_width_2p0_eval_radius_0p5_cycles_2p0_with_companion_square.png"
FFT_NAME = "coc_circle_intensity_sum_timeseries_fft_vortex_charge_2_ring_width_2p0_eval_radius_0p5_cycles_2p0_with_companion.png"
WORKFLOW_STRIP_NAME = "workflow.png"


def _build_sim_kwargs() -> dict:
    sim_kwargs = default_sim_kwargs()
    sim_kwargs.update(
        phase_mask=VortexPhaseMask(charge=2),
        secondary_diameter_ratio=0.25,
        spider_width_pixels=0.25,
        spider_angles_deg=(0.0, 90.0),
        pupil_supersample=8,
        include_ghost=True,
        include_interference=True,
        include_companion_ghost=True,
        companion_flux_ratio=PLANET_FLUX_RATIO,
        companion_offset_lamD=PLANET_CENTER_LAMD,
        e_final_phase_offset=0.0,
        phase_screen_path=resolve_phase_screen_path("10"),
        phase_screen_index=0,
    )
    ring_radius_lamD = float(math.hypot(*PLANET_CENTER_LAMD))
    ring_rmin_lamD, ring_rmax_lamD = annulus_radii_from_width(
        mid_radius_lamD=ring_radius_lamD,
        width_lamD=RING_WIDTH_LAMD,
    )
    sim_kwargs.update(
        focal_local_phase_shape="ring",
        focal_local_phase_centers_lamD=(),
        focal_local_phase_radius_lamD=0.0,
        focal_local_phase_inner_radius_lamD=float(ring_rmin_lamD),
        focal_local_phase_outer_radius_lamD=float(ring_rmax_lamD),
        focal_local_phase_offset=0.0,
    )
    return sim_kwargs


def _build_grids(result: dict) -> tuple[np.ndarray, np.ndarray]:
    n_fft = int(result["n_fft"])
    focal_sampling = float(result["focal_sampling"])
    pix = np.arange(n_fft, dtype=float)
    center = (float(n_fft) - 1.0) / 2.0
    axis_lamd = (pix - center) / focal_sampling
    return np.meshgrid(axis_lamd, axis_lamd)


def _crop_slice(result: dict, crop_radius_lamD: float) -> tuple[slice, float]:
    n_fft = int(result["n_fft"])
    focal_sampling = float(result["focal_sampling"])
    half = int(crop_radius_lamD * focal_sampling)
    center = n_fft // 2
    return slice(center - half, center + half), float(crop_radius_lamD)


def _planet_eval_mask(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    return (
        (xx - float(PLANET_CENTER_LAMD[0])) ** 2
        + (yy - float(PLANET_CENTER_LAMD[1])) ** 2
        <= float(PLANET_EVAL_RADIUS_LAMD) ** 2
    )


def _aperture_mask(xx: np.ndarray, yy: np.ndarray, center: tuple[float, float], radius_lamD: float) -> np.ndarray:
    return (
        (xx - float(center[0])) ** 2 + (yy - float(center[1])) ** 2
        <= float(radius_lamD) ** 2
    )


def _save_phase_mask_panels(sim_kwargs: dict, base_result: dict) -> None:
    sim = CoronagraphSimulator(**sim_kwargs)
    sl, crop_lamD = _crop_slice(base_result, crop_radius_lamD=9.0)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.0), constrained_layout=True)
    for ax, phase in zip(axes.ravel(), REPRESENTATIVE_PHASES):
        sim.focal_local_phase_offset = float(phase)
        local_phase_map = sim._local_focal_phase_map()  # noqa: SLF001
        panel = np.angle(base_result["mask"] * np.exp(-1j * local_phase_map))[sl, sl]
        im = ax.imshow(
            panel,
            origin="lower",
            cmap="twilight",
            extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD],
            vmin=-np.pi,
            vmax=np.pi,
        )
        ax.contour(
            local_phase_map[sl, sl] > 0.0,
            levels=[0.5],
            colors=["white"],
            linewidths=0.8,
            origin="lower",
            extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD],
        )
        ax.set_title(f"{phase / np.pi:.1f}pi", fontsize=12)
        ax.set_xlabel("x [lambda/D]")
        ax.set_ylabel("y [lambda/D]")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Wrapped phase of vortex mask + modulation [rad]")
    fig.savefig(OUTPUT_DIR / PHASE_MASK_NAME, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _select_speckles(image: np.ndarray, xx: np.ndarray, yy: np.ndarray) -> list[tuple[float, float]]:
    arr = np.asarray(image, dtype=float)
    rr = np.sqrt(xx**2 + yy**2)
    planet_radius_lamD = float(math.hypot(*PLANET_CENTER_LAMD))
    planet_aperture_mask = _aperture_mask(xx, yy, PLANET_CENTER_LAMD, PLANET_EVAL_RADIUS_LAMD)
    candidate_mask = (
        (np.abs(rr - planet_radius_lamD) <= SPECKLE_RADIUS_MATCH_TOL_LAMD)
        & (~planet_aperture_mask)
    )
    candidate_indices = np.argsort(arr[candidate_mask])[::-1]
    candidate_coords = np.argwhere(candidate_mask)
    selected: list[tuple[float, float]] = []
    for idx in candidate_indices:
        y_idx, x_idx = candidate_coords[int(idx)]
        x_lamd = float(xx[y_idx, x_idx])
        y_lamd = float(yy[y_idx, x_idx])
        if all(math.hypot(x_lamd - sx, y_lamd - sy) >= SPECKLE_MIN_SEPARATION_LAMD for sx, sy in selected):
            selected.append((x_lamd, y_lamd))
        if len(selected) >= SPECKLE_COUNT:
            break
    return selected


def _speckle_aperture_series(
    phase_offsets: np.ndarray,
    sim_kwargs: dict,
    xx: np.ndarray,
    yy: np.ndarray,
    aperture_centers: list[tuple[float, float]],
    aperture_radius_lamD: float,
) -> np.ndarray:
    traces = np.zeros((len(aperture_centers), phase_offsets.size), dtype=float)
    aperture_masks = [_aperture_mask(xx, yy, center, aperture_radius_lamD) for center in aperture_centers]
    for idx, phase in enumerate(phase_offsets):
        local_kwargs = dict(sim_kwargs)
        local_kwargs["focal_local_phase_offset"] = float(phase)
        result = CoronagraphSimulator(**local_kwargs).run()
        image = np.asarray(result["final_psf_with_ghost"], dtype=float)
        for j, aperture_mask in enumerate(aperture_masks):
            traces[j, idx] = float(np.sum(image[aperture_mask]))
    return traces


def _save_final_psf_panels(sim_kwargs: dict, base_result: dict, speckle_centers: list[tuple[float, float]]) -> None:
    sl, crop_lamD = _crop_slice(base_result, crop_radius_lamD=9.0)
    ring_radius_lamD = float(math.hypot(*PLANET_CENTER_LAMD))
    ring_rmin_lamD, ring_rmax_lamD = annulus_radii_from_width(
        mid_radius_lamD=ring_radius_lamD,
        width_lamD=RING_WIDTH_LAMD,
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 8.0), constrained_layout=True)
    for ax, phase in zip(axes.ravel(), REPRESENTATIVE_PHASES):
        local_kwargs = dict(sim_kwargs)
        local_kwargs["focal_local_phase_offset"] = float(phase)
        result = CoronagraphSimulator(**local_kwargs).run()
        panel = result["final_psf_with_ghost"][sl, sl]
        im = ax.imshow(
            np.log10(panel + 1e-12),
            origin="lower",
            cmap="inferno",
            extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD],
            vmin=-8.0,
            vmax=0.0,
        )
        ax.add_patch(Circle((0.0, 0.0), ring_rmin_lamD, fill=False, edgecolor="cyan", linewidth=1.0))
        ax.add_patch(Circle((0.0, 0.0), ring_rmax_lamD, fill=False, edgecolor="lime", linewidth=1.0))
        # Keep the image clean by labeling targets from outside the axes.
        ax.annotate(
            "Planet",
            xy=PLANET_CENTER_LAMD,
            xytext=(crop_lamD - 0.3, PLANET_CENTER_LAMD[1] + 1.0),
            color="white",
            fontsize=8,
            ha="right",
            va="bottom",
            arrowprops=dict(arrowstyle="->", color="white", lw=1.0),
        )
        callout_y = min(crop_lamD - 0.8, max(sy for _, sy in speckle_centers) + 1.2)
        for speckle_idx, (sx, sy) in enumerate(speckle_centers, start=1):
            ax.annotate(
                f"S{speckle_idx}",
                xy=(sx, sy),
                xytext=(-crop_lamD + 0.4, callout_y - 1.0 * (speckle_idx - 1)),
                color="#ffd966",
                fontsize=8,
                ha="left",
                va="center",
                arrowprops=dict(arrowstyle="->", color="#ffd966", lw=0.9),
            )
        ax.set_title(f"{phase / np.pi:.1f}pi", fontsize=12)
        ax.set_xlabel("x [lambda/D]")
        ax.set_ylabel("y [lambda/D]")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("log10 intensity")
    fig.savefig(OUTPUT_DIR / FINAL_PSF_NAME, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _phase_sweep_series() -> np.ndarray:
    n_phase_samples = int(PHASE_CYCLES * MODULATION_STEPS_PER_CYCLE) + 1
    return np.linspace(0.0, 2.0 * np.pi * PHASE_CYCLES, n_phase_samples, endpoint=True)


def _save_time_series(phase_offsets: np.ndarray, aperture_traces: np.ndarray, labels: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.1), constrained_layout=True)
    colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for idx in range(aperture_traces.shape[0]):
        color = colors[idx % len(colors)]
        is_planet = idx == 0
        ax.plot(
            phase_offsets / np.pi,
            aperture_traces[idx],
            color=color,
            linewidth=3.2 if is_planet else 2.0,
            linestyle="-" if is_planet else "-",
            label=labels[idx],
            zorder=4 if is_planet else 2,
        )
        ax.scatter(
            phase_offsets / np.pi,
            aperture_traces[idx],
            color=color,
            s=34 if is_planet else 18,
            marker="D" if is_planet else "o",
            edgecolors="black" if is_planet else "none",
            linewidths=0.6 if is_planet else 0.0,
            zorder=5 if is_planet else 3,
        )
    ax.set_xlabel("Local phase offset [pi rad]")
    ax.set_ylabel("Aperture sum")
    ax.set_title("Planet and Same-Radius Speckle Sums vs Local Phase Offset")
    ax.set_xlim(float(phase_offsets[0] / np.pi), float(phase_offsets[-1] / np.pi))
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(OUTPUT_DIR / TIME_SERIES_NAME, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _save_fft(phase_offsets: np.ndarray, aperture_traces: np.ndarray, labels: list[str]) -> None:
    phase_fft = np.asarray(phase_offsets, dtype=float)
    if phase_fft.size > 2 and np.isclose(phase_fft[0], 0.0) and np.isclose(phase_fft[-1], 2.0 * np.pi * PHASE_CYCLES):
        phase_fft = phase_fft[:-1]
        aperture_traces = aperture_traces[:, :-1]
    dphi = float(np.mean(np.diff(phase_fft)))
    freqs = np.fft.fftfreq(aperture_traces.shape[1], d=dphi)
    pos = freqs >= 0.0
    fft_width = 0.03

    fig, ax = plt.subplots(figsize=(6.6, 6.1), constrained_layout=True)
    colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    positive_amps: list[np.ndarray] = []
    for idx in range(aperture_traces.shape[0]):
        amp = np.abs(np.fft.fft(aperture_traces[idx])) / max(aperture_traces.shape[1], 1)
        positive_amps.append(amp[pos])
        color = colors[idx % len(colors)]
        is_planet = idx == 0
        ax.plot(
            freqs[pos],
            amp[pos],
            color=color,
            linewidth=3.2 if is_planet else 2.0,
            label=labels[idx],
            zorder=4 if is_planet else 2,
        )
        ax.scatter(
            freqs[pos],
            amp[pos],
            color=color,
            s=34 if is_planet else 18,
            marker="D" if is_planet else "o",
            edgecolors="black" if is_planet else "none",
            linewidths=0.6 if is_planet else 0.0,
            zorder=5 if is_planet else 3,
        )

    pos_freqs = freqs[pos]
    if pos_freqs.size >= 2 and positive_amps:
        mean_amp = np.mean(np.vstack(positive_amps), axis=0)
        dc_freq = float(pos_freqs[0])
        ac_idx = 1 + int(np.argmax(mean_amp[1:])) if mean_amp.size > 1 else 0
        ac_freq = float(pos_freqs[ac_idx])
        dc_left = dc_freq - 0.5 * fft_width
        dc_right = dc_freq + 0.5 * fft_width
        ac_left = ac_freq - 0.5 * fft_width
        ac_right = ac_freq + 0.5 * fft_width
        ax.axvspan(dc_left, dc_right, color="#ffef99", alpha=0.35, zorder=0)
        ax.axvspan(ac_left, ac_right, color="#9fd5ff", alpha=0.30, zorder=0)
        ymax = float(np.max(mean_amp)) if np.isfinite(np.max(mean_amp)) else 1.0
        ax.text(
            dc_freq + 0.004,
            ymax * 0.92,
            "DC channel",
            color="#7a5d00",
            fontsize=11,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff7cc", edgecolor="none", alpha=0.85),
        )
        ax.text(
            ac_freq + 0.004,
            ymax * 0.78,
            "AC channel",
            color="#0b4f8a",
            fontsize=11,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#dff1ff", edgecolor="none", alpha=0.85),
        )
    ax.set_xlabel("Frequency [cycles/rad]")
    ax.set_ylabel("Amplitude")
    ax.set_title("FFT in Time of Planet and Same-Radius Speckle Traces")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(OUTPUT_DIR / FFT_NAME, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _fit_panel(image: Image.Image, panel_width: int, panel_height: int, pad: int, bg: str) -> Image.Image:
    usable_w = max(1, panel_width - 2 * pad)
    usable_h = max(1, panel_height - 2 * pad)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    fitted = ImageOps.contain(image, (usable_w, usable_h), method=resample)
    panel = Image.new("RGB", (panel_width, panel_height), bg)
    x0 = (panel_width - fitted.width) // 2
    y0 = (panel_height - fitted.height) // 2
    panel.paste(fitted, (x0, y0))
    return panel


def _save_workflow_strip() -> None:
    bg = "#050B14"
    accent = "#F39B5B"
    text = "#FFF4E3"
    mute = "#9FE7FF"
    strip_w = 3600
    strip_h = 980
    margin_x = 60
    panel_w = 760
    panel_h = 760
    top_y = 80
    arrow_w = 120
    pad = 20
    titles = ["1. Phase Mask", "2. Coronagraphic PSF", "3. Time Series", "4. FFT in Time"]
    filenames = [PHASE_MASK_NAME, FINAL_PSF_NAME, TIME_SERIES_NAME, FFT_NAME]

    canvas = Image.new("RGB", (strip_w, strip_h), bg)
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
        arrow_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 66)
        footer_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
    except Exception:
        title_font = ImageFont.load_default()
        arrow_font = ImageFont.load_default()
        footer_font = ImageFont.load_default()

    x = margin_x
    for idx, (title, filename) in enumerate(zip(titles, filenames)):
        draw.text((x + 120, 18), title, fill=text, font=title_font)
        panel = Image.open(OUTPUT_DIR / filename).convert("RGB")
        panel = _fit_panel(panel, panel_w, panel_h, pad, bg)
        canvas.paste(panel, (x, top_y))
        if idx < len(titles) - 1:
            arrow_x = x + panel_w + 18
            draw.text((arrow_x, top_y + panel_h // 2 - 28), ">", fill=accent, font=arrow_font)
            x += panel_w + arrow_w

    footer = "Phase modulation creates a temporal signal that separates incoherent planet light from coherent stellar speckles."
    footer_box = draw.textbbox((0, 0), footer, font=footer_font)
    footer_x = (strip_w - (footer_box[2] - footer_box[0])) // 2
    draw.text((footer_x, 885), footer, fill=mute, font=footer_font)
    canvas.save(OUTPUT_DIR / WORKFLOW_STRIP_NAME)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sim_kwargs = _build_sim_kwargs()
    base_result = CoronagraphSimulator(**sim_kwargs).run()
    xx, yy = _build_grids(base_result)
    speckle_centers = _select_speckles(np.asarray(base_result["final_psf_with_ghost"], dtype=float), xx, yy)
    trace_centers = [PLANET_CENTER_LAMD, *speckle_centers]
    trace_labels = ["Planet", *[f"S{i}" for i in range(1, len(speckle_centers) + 1)]]
    phase_offsets = _phase_sweep_series()
    aperture_traces = _speckle_aperture_series(
        phase_offsets=phase_offsets,
        sim_kwargs=sim_kwargs,
        xx=xx,
        yy=yy,
        aperture_centers=trace_centers,
        aperture_radius_lamD=PLANET_EVAL_RADIUS_LAMD,
    )
    _save_phase_mask_panels(sim_kwargs=sim_kwargs, base_result=base_result)
    _save_final_psf_panels(sim_kwargs=sim_kwargs, base_result=base_result, speckle_centers=speckle_centers)
    _save_time_series(phase_offsets=phase_offsets, aperture_traces=aperture_traces, labels=trace_labels)
    _save_fft(phase_offsets=phase_offsets, aperture_traces=aperture_traces, labels=trace_labels)
    _save_workflow_strip()
    print(f"Saved {PHASE_MASK_NAME}")
    print(f"Saved {FINAL_PSF_NAME}")
    print(f"Saved {TIME_SERIES_NAME}")
    print(f"Saved {FFT_NAME}")
    print(f"Saved {WORKFLOW_STRIP_NAME}")


if __name__ == "__main__":
    main()
