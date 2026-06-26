from __future__ import annotations

import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from .plotting import (
    _coc_build_incoherence_maps,
    _coc_frequency_selection_spectrum,
    plot_coc_planet_phase_outputs,
)
from .region_shapes import (
    annulus_radii_from_width,
    build_touching_circle_ring,
    normalize_region_shape,
)
from .simulator import CoronagraphSimulator

SNR_APERTURE_RADIUS_LAMD = 1.0
SNR_ANNULUS_HALF_WIDTH_LAMD = 1.0


def _theta_back_and_forth(n: int, max_abs: float = np.pi) -> np.ndarray:
    if n <= 1:
        return np.array([0.0], dtype=float)
    # Unique ring angles, ordered as back-and-forth around zero.
    raw = np.arange(n, dtype=float) * (2.0 * np.pi / float(n))
    wrapped = ((raw + np.pi) % (2.0 * np.pi)) - np.pi
    order = sorted(
        range(n),
        key=lambda i: (abs(float(wrapped[i])), 0 if float(wrapped[i]) <= 0.0 else 1),
    )
    arr = wrapped[np.asarray(order, dtype=int)]
    arr[np.abs(arr) < 1e-14] = 0.0
    return arr


def _write_rgb_gif(rgb8_frames: np.ndarray, gif_path: str, duration_ms: int = 500) -> None:
    saved_gif = False
    try:
        import imageio.v2 as imageio

        imageio.mimsave(gif_path, list(rgb8_frames), duration=duration_ms / 1000.0, loop=0)
        saved_gif = True
    except Exception:
        try:
            from PIL import Image

            frames = [Image.fromarray(frame, mode="RGB") for frame in rgb8_frames]
            if len(frames) > 0:
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=duration_ms,
                    loop=0,
                )
                saved_gif = True
        except Exception:
            saved_gif = False

    if not saved_gif:
        raise RuntimeError("GIF export requires either 'imageio' or 'Pillow' (PIL) to be installed.")


def _local_phase_region_kwargs(
    region_shape_name: str,
    region_width_or_radius_lamD: float,
    orbit_radius_lamD: float,
    centers_lamD: list[tuple[float, float]],
    ring_center_lamD: tuple[float, float] = (0.0, 0.0),
) -> dict:
    if region_shape_name == "ring":
        inner_radius_lamD, outer_radius_lamD = annulus_radii_from_width(
            mid_radius_lamD=orbit_radius_lamD,
            width_lamD=region_width_or_radius_lamD,
        )
        return {
            "focal_local_phase_shape": "ring",
            "focal_local_phase_centers_lamD": (),
            "focal_local_phase_radius_lamD": 0.0,
            "focal_local_phase_ring_center_lamD": (
                float(ring_center_lamD[0]),
                float(ring_center_lamD[1]),
            ),
            "focal_local_phase_inner_radius_lamD": float(inner_radius_lamD),
            "focal_local_phase_outer_radius_lamD": float(outer_radius_lamD),
        }
    return {
        "focal_local_phase_shape": "circle",
        "focal_local_phase_centers_lamD": tuple((float(cx), float(cy)) for cx, cy in centers_lamD),
        "focal_local_phase_radius_lamD": float(region_width_or_radius_lamD),
        "focal_local_phase_ring_center_lamD": (0.0, 0.0),
        "focal_local_phase_inner_radius_lamD": 0.0,
        "focal_local_phase_outer_radius_lamD": 0.0,
    }


def _inclusive_float_range(start: float, stop: float, step: float) -> np.ndarray:
    start_f = float(start)
    stop_f = float(stop)
    step_f = float(step)
    if np.isclose(start_f, stop_f):
        if step_f < 0.0:
            raise ValueError("step must be >= 0 for a single-value sweep.")
        return np.array([start_f], dtype=float)
    if step_f <= 0.0:
        raise ValueError("step must be > 0.")
    return np.arange(start_f, stop_f + 0.5 * step_f, step_f, dtype=float)


def _polar_to_cartesian_lamD(radius_lamD: float, theta_deg: float) -> tuple[float, float]:
    theta_rad = np.deg2rad(float(theta_deg))
    radius = float(radius_lamD)
    return (
        float(radius * np.cos(theta_rad)),
        float(radius * np.sin(theta_rad)),
    )


def _phase_screen_folder_tag(args: argparse.Namespace, sim_kwargs: dict) -> str:
    jitter_choice = getattr(args, "phase_screen_jitter", None)
    if jitter_choice is not None:
        token = str(jitter_choice).strip().lower()
        if token in {"", "none"}:
            return "_phase_screen_off"
        return f"_phase_screen_on_{token}"
    if sim_kwargs.get("phase_screen_path") is None:
        return "_phase_screen_off"
    return "_phase_screen_on"


def _focal_shift_lamD(sim_kwargs: dict) -> tuple[float, float]:
    focal_shift_pixels = tuple(float(v) for v in sim_kwargs.get("focal_shift_pixels", (0.0, 0.0)))
    focal_sampling = float(sim_kwargs.get("focal_sampling", 1.0))
    if np.isclose(focal_sampling, 0.0):
        return (0.0, 0.0)
    return (
        float(focal_shift_pixels[0] / focal_sampling),
        float(focal_shift_pixels[1] / focal_sampling),
    )


def _compute_incoherence_map_info(
    stack: np.ndarray,
    phase_offsets: np.ndarray,
    mode: str = "fft_band",
    planet_region_mask: np.ndarray | None = None,
) -> dict[str, object] | None:
    phase_series = np.asarray(phase_offsets, dtype=float)
    stack_arr = np.asarray(stack, dtype=float)
    if (
        phase_series.size > 2
        and np.isclose(phase_series[0], 0.0)
        and np.isclose(phase_series[-1], float(phase_series.max()))
    ):
        phase_series = phase_series[:-1]
        stack_arr = stack_arr[:-1]
    if phase_series.size < 2 or stack_arr.shape[0] < 2:
        return None
    dphi = float(np.mean(np.diff(phase_series)))
    freq = np.fft.fftfreq(stack_arr.shape[0], d=dphi)
    fft_cube = np.fft.fft(stack_arr, axis=0)
    map_info = _coc_build_incoherence_maps(
        freq_bins=freq,
        fft_cube=fft_cube,
        central_stack_fft=stack_arr,
        mode=mode,
        planet_region_mask=planet_region_mask,
    )
    return {
        **map_info,
        "analysis_stack": np.asarray(stack_arr, dtype=float),
        "analysis_phase_series": np.asarray(phase_series, dtype=float),
        "freq_bins": np.asarray(freq, dtype=float),
    }


def _compute_incoherence_map(
    stack: np.ndarray,
    phase_offsets: np.ndarray,
    mode: str = "fft_band",
    planet_region_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    map_info = _compute_incoherence_map_info(
        stack=stack,
        phase_offsets=phase_offsets,
        mode=mode,
        planet_region_mask=planet_region_mask,
    )
    if map_info is None:
        return None
    return np.asarray(map_info["incoherence_map"], dtype=float)


def _planet_region_snr(
    incoh: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    planet_center_lamD: tuple[float, float],
    orbit_radius_lamD: float,
    eval_radius_lamD: float = SNR_APERTURE_RADIUS_LAMD,
    annulus_half_width_lamD: float = SNR_ANNULUS_HALF_WIDTH_LAMD,
    snr_eps: float = 1e-12,
) -> tuple[float, float, float]:
    noise_centers = _noise_aperture_centers_lamD(
        planet_center_lamD=planet_center_lamD,
        orbit_radius_lamD=orbit_radius_lamD,
        eval_radius_lamD=eval_radius_lamD,
        annulus_half_width_lamD=annulus_half_width_lamD,
    )
    eval_radius = float(eval_radius_lamD)
    orbit_radius = float(orbit_radius_lamD)
    planet_x = float(planet_center_lamD[0])
    planet_y = float(planet_center_lamD[1])

    planet_mask = ((xx - planet_x) ** 2 + (yy - planet_y) ** 2) <= eval_radius ** 2
    signal_mean = float(np.mean(incoh[planet_mask])) if np.any(planet_mask) else float("nan")

    rr = np.sqrt(xx**2 + yy**2)
    annulus_mask = (rr >= (orbit_radius - eval_radius)) & (rr <= (orbit_radius + eval_radius))
    aperture_means: list[float] = []
    for cx, cy in noise_centers:
        aperture_mask = ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) <= eval_radius ** 2
        if not np.any(aperture_mask):
            continue
        if not np.all(annulus_mask[aperture_mask]):
            continue
        aperture_means.append(float(np.mean(incoh[aperture_mask])))
    background_mean = (
        float(np.mean(np.asarray(aperture_means, dtype=float)))
        if len(aperture_means) > 0
        else float("nan")
    )
    background_std = float(np.std(np.asarray(aperture_means, dtype=float))) if len(aperture_means) > 0 else float("nan")

    if np.isfinite(signal_mean) and np.isfinite(background_mean) and np.isfinite(background_std):
        noise_term = float(background_std)
        noise_safe = (
            noise_term
            if abs(noise_term) > float(snr_eps)
            else (float(snr_eps) if noise_term >= 0.0 else -float(snr_eps))
        )
        snr = float((signal_mean - background_mean) / noise_safe)
    else:
        snr = float("nan")
    return signal_mean, background_std, snr


def _planet_region_centered_snr(
    incoh: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    planet_center_lamD: tuple[float, float],
    orbit_radius_lamD: float,
    eval_radius_lamD: float = SNR_APERTURE_RADIUS_LAMD,
    annulus_half_width_lamD: float = SNR_ANNULUS_HALF_WIDTH_LAMD,
    snr_eps: float = 1e-12,
) -> tuple[float, float, float, float]:
    noise_centers = _noise_aperture_centers_lamD(
        planet_center_lamD=planet_center_lamD,
        orbit_radius_lamD=orbit_radius_lamD,
        eval_radius_lamD=eval_radius_lamD,
        annulus_half_width_lamD=annulus_half_width_lamD,
    )
    eval_radius = float(eval_radius_lamD)
    orbit_radius = float(orbit_radius_lamD)
    planet_x = float(planet_center_lamD[0])
    planet_y = float(planet_center_lamD[1])

    planet_mask = ((xx - planet_x) ** 2 + (yy - planet_y) ** 2) <= eval_radius ** 2
    signal_mean = float(np.mean(incoh[planet_mask])) if np.any(planet_mask) else float("nan")

    rr = np.sqrt(xx**2 + yy**2)
    annulus_mask = (rr >= (orbit_radius - eval_radius)) & (rr <= (orbit_radius + eval_radius))
    background_aperture_means: list[float] = []
    for cx, cy in noise_centers:
        aperture_mask = ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) <= eval_radius ** 2
        if not np.any(aperture_mask):
            continue
        if not np.all(annulus_mask[aperture_mask]):
            continue
        background_aperture_means.append(float(np.mean(incoh[aperture_mask])))
    background_mean = (
        float(np.mean(np.asarray(background_aperture_means, dtype=float)))
        if len(background_aperture_means) > 0
        else float("nan")
    )
    background_std = (
        float(np.std(np.asarray(background_aperture_means, dtype=float)))
        if len(background_aperture_means) > 0
        else float("nan")
    )

    if np.isfinite(signal_mean) and np.isfinite(background_std):
        noise_term = float(background_std)
        noise_safe = (
            noise_term
            if abs(noise_term) > float(snr_eps)
            else (float(snr_eps) if noise_term >= 0.0 else -float(snr_eps))
        )
        snr = float((signal_mean - background_mean) / noise_safe)
    else:
        snr = float("nan")
    return signal_mean, background_mean, background_std, snr


def _planet_region_snr_from_coherence(
    coherence_map: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    planet_center_lamD: tuple[float, float],
    orbit_radius_lamD: float,
    eval_radius_lamD: float = SNR_APERTURE_RADIUS_LAMD,
    annulus_half_width_lamD: float = SNR_ANNULUS_HALF_WIDTH_LAMD,
    snr_eps: float = 1e-12,
) -> tuple[float, float, float]:
    noise_centers = _noise_aperture_centers_lamD(
        planet_center_lamD=planet_center_lamD,
        orbit_radius_lamD=orbit_radius_lamD,
        eval_radius_lamD=eval_radius_lamD,
        annulus_half_width_lamD=annulus_half_width_lamD,
    )
    coh_arr = np.asarray(coherence_map, dtype=float)
    xx_arr = np.asarray(xx, dtype=float)
    yy_arr = np.asarray(yy, dtype=float)
    eval_radius = float(eval_radius_lamD)
    orbit_radius = float(orbit_radius_lamD)
    annulus_half_width = float(annulus_half_width_lamD)
    planet_x = float(planet_center_lamD[0])
    planet_y = float(planet_center_lamD[1])

    planet_mask = ((xx_arr - planet_x) ** 2 + (yy_arr - planet_y) ** 2) <= eval_radius ** 2
    planet_stat = float(np.mean(coh_arr[planet_mask])) if np.any(planet_mask) else float("nan")

    rr = np.sqrt(xx_arr**2 + yy_arr**2)
    annulus_mask = (rr >= (orbit_radius - annulus_half_width)) & (
        rr <= (orbit_radius + annulus_half_width)
    )

    aperture_stats: list[float] = []
    for cx, cy in noise_centers:
        aperture_mask = ((xx_arr - float(cx)) ** 2 + (yy_arr - float(cy)) ** 2) <= eval_radius ** 2
        if not np.any(aperture_mask):
            continue
        if not np.all(annulus_mask[aperture_mask]):
            continue
        aperture_stats.append(float(np.mean(coh_arr[aperture_mask])))

    background_mean = (
        float(np.mean(np.asarray(aperture_stats, dtype=float)))
        if len(aperture_stats) > 0
        else float("nan")
    )
    noise_std = (
        float(np.std(np.asarray(aperture_stats, dtype=float)))
        if len(aperture_stats) > 0
        else float("nan")
    )
    if np.isfinite(planet_stat) and np.isfinite(background_mean) and np.isfinite(noise_std):
        noise_safe = (
            noise_std
            if abs(noise_std) > float(snr_eps)
            else (float(snr_eps) if noise_std >= 0.0 else -float(snr_eps))
        )
        snr = float((background_mean - planet_stat) / noise_safe)
    else:
        snr = float("nan")
    return planet_stat, background_mean, snr


def _noise_aperture_centers_lamD(
    planet_center_lamD: tuple[float, float],
    orbit_radius_lamD: float,
    eval_radius_lamD: float = SNR_APERTURE_RADIUS_LAMD,
    annulus_half_width_lamD: float = SNR_ANNULUS_HALF_WIDTH_LAMD,
) -> list[tuple[float, float]]:
    eval_radius = float(eval_radius_lamD)
    orbit_radius = float(orbit_radius_lamD)
    annulus_half_width = float(annulus_half_width_lamD)
    if eval_radius <= 0.0 or orbit_radius <= 0.0 or annulus_half_width < eval_radius:
        return []

    anchor_angle = 0.0
    ring = build_touching_circle_ring(
        requested_region_radius_lamD=eval_radius,
        orbit_radius_lamD=orbit_radius,
        anchor_angle_rad=anchor_angle,
        rotation_fraction=0.0,
    )
    return [(float(cx), float(cy)) for cx, cy in ring["centers_lamD"]]


def _select_reference_speckle_centers_lamD(
    planet_center_lamD: tuple[float, float],
    orbit_radius_lamD: float,
    n_select: int = 3,
    eval_radius_lamD: float = SNR_APERTURE_RADIUS_LAMD,
    annulus_half_width_lamD: float = SNR_ANNULUS_HALF_WIDTH_LAMD,
) -> list[tuple[float, float]]:
    noise_centers = _noise_aperture_centers_lamD(
        planet_center_lamD=planet_center_lamD,
        orbit_radius_lamD=orbit_radius_lamD,
        eval_radius_lamD=eval_radius_lamD,
        annulus_half_width_lamD=annulus_half_width_lamD,
    )
    if len(noise_centers) <= n_select:
        return noise_centers
    idx = np.linspace(0, len(noise_centers) - 1, int(n_select), dtype=int)
    idx = np.unique(idx)
    return [noise_centers[int(i)] for i in idx]


def _resolve_roi_configuration(
    region_shape_name: str,
    requested_roi_size_lamD: float,
    orbit_radius_lamD: float,
    initial_angle_rad: float,
    planet_center_lamD: tuple[float, float],
) -> tuple[list[tuple[float, float]], float]:
    if region_shape_name == "ring_of_circle":
        ring = build_touching_circle_ring(
            requested_region_radius_lamD=float(requested_roi_size_lamD),
            orbit_radius_lamD=float(orbit_radius_lamD),
            anchor_angle_rad=float(initial_angle_rad),
            rotation_fraction=0.0,
        )
        return ([(float(cx), float(cy)) for cx, cy in ring["centers_lamD"]], float(ring["resolved_radius_lamD"]))
    if region_shape_name == "ring":
        return ([(float(planet_center_lamD[0]), float(planet_center_lamD[1]))], float(requested_roi_size_lamD))
    return ([(float(planet_center_lamD[0]), float(planet_center_lamD[1]))], float(requested_roi_size_lamD))


def _evaluate_best_roi_for_planet_center(
    *,
    planet_center: tuple[float, float],
    roi_sizes: np.ndarray,
    region_shape_name: str,
    sim_local: dict,
    phase_offsets: np.ndarray,
    sl16: slice,
    half16: int,
    xx16: np.ndarray,
    yy16: np.ndarray,
    incoherence_map_mode: str = "fft_band",
    collect_panels: bool = False,
) -> tuple[list[dict[str, float | int]], dict[str, float | int] | None, list[dict[str, object]]]:
    orbit_radius_lamD = float(np.hypot(*planet_center))
    if orbit_radius_lamD <= 0.0:
        return [], None, []
    initial_angle_rad = float(np.arctan2(planet_center[1], planet_center[0]))
    rows: list[dict[str, float | int]] = []
    best_entry: dict[str, float | int] | None = None
    panels: list[dict[str, object]] = []
    for roi_size in roi_sizes:
        roi_centers, roi_radius_eff = _resolve_roi_configuration(
            region_shape_name=region_shape_name,
            requested_roi_size_lamD=float(roi_size),
            orbit_radius_lamD=orbit_radius_lamD,
            initial_angle_rad=initial_angle_rad,
            planet_center_lamD=planet_center,
        )
        stack = np.zeros((phase_offsets.size, 2 * half16, 2 * half16), dtype=float)
        for i, ph in enumerate(phase_offsets):
            phase_sim = CoronagraphSimulator(
                **{
                    **sim_local,
                    "companion_offset_lamD": planet_center,
                    "e_final_phase_offset": 0.0,
                    "focal_local_phase_offset": float(ph),
                    **_local_phase_region_kwargs(
                        region_shape_name=region_shape_name,
                        region_width_or_radius_lamD=float(roi_radius_eff),
                        orbit_radius_lamD=orbit_radius_lamD,
                        centers_lamD=roi_centers,
                        ring_center_lamD=_focal_shift_lamD(sim_local),
                    ),
                }
            )
            stack[i] = phase_sim.run()["final_psf_with_ghost"][sl16, sl16]
        map_info = _compute_incoherence_map_info(
            stack=stack,
            phase_offsets=phase_offsets,
            mode=str(incoherence_map_mode),
            planet_region_mask=(
                (xx16 - float(planet_center[0])) ** 2
                + (yy16 - float(planet_center[1])) ** 2
                <= SNR_APERTURE_RADIUS_LAMD ** 2
            ),
        )
        if map_info is None:
            continue
        incoh = np.asarray(map_info["incoherence_map"], dtype=float)
        coherence_map = np.asarray(map_info["coherence_map"], dtype=float)
        max_minus_coherence_map = np.nanmax(coherence_map) - coherence_map
        planet_mask = (
            (xx16 - float(planet_center[0])) ** 2
            + (yy16 - float(planet_center[1])) ** 2
            <= SNR_APERTURE_RADIUS_LAMD ** 2
        )
        planet_std = float(np.std(np.asarray(incoh[planet_mask], dtype=float))) if np.any(planet_mask) else float("nan")
        peak, med, snr = _planet_region_snr(
            incoh=incoh,
            xx=xx16,
            yy=yy16,
            planet_center_lamD=planet_center,
            orbit_radius_lamD=orbit_radius_lamD,
        )
        centered_peak, centered_comparison_mean, centered_comparison_std, centered_snr = _planet_region_centered_snr(
            incoh=incoh,
            xx=xx16,
            yy=yy16,
            planet_center_lamD=planet_center,
            orbit_radius_lamD=orbit_radius_lamD,
        )
        row = {
            "planet_x_lamD": float(planet_center[0]),
            "planet_y_lamD": float(planet_center[1]),
            "orbit_radius_lamD": float(orbit_radius_lamD),
            "planet_theta_rad": float(initial_angle_rad),
            "requested_roi_size_lamD": float(roi_size),
            "resolved_roi_size_lamD": float(roi_radius_eff),
            "n_circles": int(len(roi_centers)),
            "planet_peak": float(peak),
            "planet_std": float(planet_std),
            "background_aperture_std": float(med),
            "raw_snr": float(snr),
            "background_aperture_mean": float(centered_comparison_mean),
            "background_aperture_std_centered": float(centered_comparison_std),
            "snr": float(centered_snr),
        }
        rows.append(row)
        if best_entry is None or (
            np.isfinite(float(snr))
            and (not np.isfinite(float(best_entry["snr"])) or float(snr) > float(best_entry["snr"]))
        ):
            best_entry = row
        if collect_panels:
            reference_spectra: list[dict[str, object]] = []
            freq_bins = np.asarray(map_info.get("freq_bins", np.array([], dtype=float)), dtype=float)
            analysis_stack = np.asarray(
                map_info.get("analysis_stack", np.array([], dtype=float)),
                dtype=float,
            )
            if freq_bins.size > 0 and analysis_stack.ndim == 3 and analysis_stack.shape[0] >= 2:
                for iref, (ref_x, ref_y) in enumerate(
                    _select_reference_speckle_centers_lamD(
                        planet_center_lamD=planet_center,
                        orbit_radius_lamD=orbit_radius_lamD,
                    ),
                    start=1,
                ):
                    ref_mask = (
                        (xx16 - float(ref_x)) ** 2
                        + (yy16 - float(ref_y)) ** 2
                        <= SNR_APERTURE_RADIUS_LAMD ** 2
                    )
                    spec = _coc_frequency_selection_spectrum(
                        freq_bins=freq_bins,
                        central_stack_fft=analysis_stack,
                        planet_region_mask=ref_mask,
                    )
                    reference_spectra.append(
                        {
                            "label": f"Speckle {iref}",
                            "center_lamD": (float(ref_x), float(ref_y)),
                            "nonnegative_freqs": np.asarray(
                                spec.get("nonnegative_freqs", np.array([], dtype=float)),
                                dtype=float,
                            ),
                            "nonnegative_mag": np.asarray(
                                spec.get("nonnegative_mag", np.array([], dtype=float)),
                                dtype=float,
                            ),
                        }
                    )
            panels.append(
                {
                    "requested_roi_size_lamD": float(roi_size),
                    "resolved_roi_size_lamD": float(roi_radius_eff),
                    "n_circles": int(len(roi_centers)),
                    "planet_peak": float(peak),
                    "planet_std": float(planet_std),
                    "background_aperture_std": float(med),
                    "raw_snr": float(snr),
                    "background_aperture_mean": float(centered_comparison_mean),
                    "background_aperture_std_centered": float(centered_comparison_std),
                    "snr": float(centered_snr),
                    "planet_peak_centered_snr": float(centered_peak),
                    "orbit_radius_lamD": float(orbit_radius_lamD),
                    "planet_center_lamD": (float(planet_center[0]), float(planet_center[1])),
                    "roi_centers_lamD": [(float(cx), float(cy)) for cx, cy in roi_centers],
                    "incoherence_map": np.array(incoh, dtype=float),
                    "coherence_map": coherence_map,
                    "max_minus_coherence_map": max_minus_coherence_map,
                    "incoherence_map_mode": str(incoherence_map_mode),
                    "selected_target_freq": float(map_info["selected_target_freq"]),
                    "selection_nonnegative_freqs": np.asarray(map_info.get("selection_nonnegative_freqs", np.array([], dtype=float)), dtype=float),
                    "selection_nonnegative_mag": np.asarray(map_info.get("selection_nonnegative_mag", np.array([], dtype=float)), dtype=float),
                    "selection_positive_freqs": np.asarray(map_info.get("selection_positive_freqs", np.array([], dtype=float)), dtype=float),
                    "selection_positive_mag": np.asarray(map_info.get("selection_positive_mag", np.array([], dtype=float)), dtype=float),
                    "reference_spectra": reference_spectra,
                }
            )
    return rows, best_entry, panels


def _save_roi_size_incoherence_pdf_for_planet_location(
    *,
    output_path: str,
    region_shape_name: str,
    panels: list[dict[str, object]],
    extent: list[float],
) -> None:
    _save_roi_size_map_pdf_for_planet_location(
        output_path=output_path,
        region_shape_name=region_shape_name,
        panels=panels,
        extent=extent,
        map_key="incoherence_map",
        title_prefix="Incoherence",
        cmap="viridis",
        snr_key="snr",
        snr_label="SNR",
        curve_snr_key="snr",
        curve_snr_label="SNR",
        vmax_percentile=90.0,
        include_mean_map_page=True,
    )


def _save_roi_size_coherence_pdf_for_planet_location(
    *,
    output_path: str,
    region_shape_name: str,
    panels: list[dict[str, object]],
    extent: list[float],
) -> None:
    _save_roi_size_map_pdf_for_planet_location(
        output_path=output_path,
        region_shape_name=region_shape_name,
        panels=panels,
        extent=extent,
        map_key="coherence_map",
        title_prefix="Coherence",
        cmap="viridis",
        snr_key="snr",
        snr_label="SNR",
        curve_snr_key="snr",
        curve_snr_label="SNR",
        vmax_percentile=98.0,
    )


def _save_roi_size_max_minus_coherence_pdf_for_planet_location(
    *,
    output_path: str,
    region_shape_name: str,
    panels: list[dict[str, object]],
    extent: list[float],
) -> None:
    _save_roi_size_map_pdf_for_planet_location(
        output_path=output_path,
        region_shape_name=region_shape_name,
        panels=panels,
        extent=extent,
        map_key="max_minus_coherence_map",
        title_prefix="max(Coherence) - Coherence",
        cmap="viridis",
        snr_key="snr",
        snr_label="SNR",
        curve_snr_key="snr",
        curve_snr_label="SNR",
        vmax_percentile=99.0,
    )


def _save_roi_size_map_pdf_for_planet_location(
    *,
    output_path: str,
    region_shape_name: str,
    panels: list[dict[str, object]],
    extent: list[float],
    map_key: str,
    title_prefix: str,
    cmap: str = "viridis",
    snr_key: str | None = None,
    snr_label: str | None = None,
    curve_snr_key: str = "snr",
    curve_snr_label: str = "SNR",
    vmin_percentile: float | None = None,
    vmax_percentile: float = 98.0,
    include_mean_map_page: bool = False,
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    if len(panels) == 0:
        return

    panels_sorted = sorted(panels, key=lambda p: float(p["requested_roi_size_lamD"]))
    curve_snr_keys = (
        "snr",
    )
    curve_snr_all = np.asarray(
        [
            float(panel[key])
            for panel in panels_sorted
            for key in curve_snr_keys
            if key in panel and np.isfinite(float(panel[key]))
        ],
        dtype=float,
    )
    if curve_snr_all.size > 0:
        curve_snr_ymin = float(np.min(curve_snr_all))
        curve_snr_ymax = float(np.max(curve_snr_all))
        if curve_snr_ymax <= curve_snr_ymin:
            pad = max(abs(curve_snr_ymax) * 0.05, 1e-6)
        else:
            pad = 0.05 * (curve_snr_ymax - curve_snr_ymin)
        curve_snr_ylim = (curve_snr_ymin - pad, curve_snr_ymax + pad)
    else:
        curve_snr_ylim = None
    n_panels = len(panels_sorted)
    ncols = min(4, max(2, int(np.ceil(np.sqrt(n_panels)))))
    nrows_maps = int(np.ceil(float(n_panels) / float(ncols)))
    height_ratios = [1.0] * nrows_maps + [1.25]

    with PdfPages(output_path) as pdf:
        fig = plt.figure(
            figsize=(4.4 * ncols, 3.6 * nrows_maps + 3.8),
            constrained_layout=True,
        )
        gs = fig.add_gridspec(nrows_maps + 1, ncols, height_ratios=height_ratios)

        for idx, panel in enumerate(panels_sorted):
            ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
            map_arr = np.asarray(panel[map_key], dtype=float)
            finite_vals = map_arr[np.isfinite(map_arr)]
            if finite_vals.size > 0:
                vmax = float(np.percentile(finite_vals, float(vmax_percentile)))
                if vmin_percentile is None:
                    vmin = 0.0
                else:
                    vmin = float(np.percentile(finite_vals, float(vmin_percentile)))
            else:
                vmin = 0.0
                vmax = 1.0
            vmax = max(vmax, 1e-20)
            if vmax <= vmin:
                vmax = vmin + 1e-20
            im = ax.imshow(
                map_arr,
                origin="lower",
                cmap=cmap,
                extent=extent,
                vmin=vmin,
                vmax=vmax,
            )
            planet_center = panel["planet_center_lamD"]
            orbit_radius_lamD = float(panel["orbit_radius_lamD"])
            roi_centers = panel["roi_centers_lamD"]
            roi_size = float(panel["resolved_roi_size_lamD"])
            planet_x = float(planet_center[0])
            planet_y = float(planet_center[1])
            x_text = extent[1] - 0.8 if planet_x >= 0.0 else extent[0] + 0.8
            x_align = "right" if planet_x >= 0.0 else "left"
            ax.annotate(
                "Planet",
                xy=(planet_x, planet_y),
                xytext=(x_text, planet_y + 0.9),
                color="white",
                fontsize=8,
                ha=x_align,
                va="bottom",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.0),
            )
            if region_shape_name == "ring":
                ring_rmin_lamD, ring_rmax_lamD = annulus_radii_from_width(
                    mid_radius_lamD=orbit_radius_lamD,
                    width_lamD=roi_size,
                )
                ax.add_patch(plt.Circle((0.0, 0.0), float(ring_rmin_lamD), fill=False, edgecolor="lime", linewidth=1.4))
                ax.add_patch(plt.Circle((0.0, 0.0), float(ring_rmax_lamD), fill=False, edgecolor="cyan", linewidth=1.4))
            else:
                for j, (cx, cy) in enumerate(roi_centers):
                    edge = "lime" if j == 0 else "cyan"
                    ax.add_patch(plt.Circle((float(cx), float(cy)), roi_size, fill=False, edgecolor=edge, linewidth=1.4))
            if snr_key is not None and snr_label is not None and snr_key in panel:
                title_text = (
                    f"{title_prefix} | ROI {float(panel['requested_roi_size_lamD']):.2f} -> {roi_size:.2f}\n"
                    f"{str(snr_label)} {float(panel[snr_key]):.3e}"
                )
            else:
                title_text = (
                    f"{title_prefix} | ROI {float(panel['requested_roi_size_lamD']):.2f} -> {roi_size:.2f}"
                )
            ax.set_title(title_text, fontsize=8, pad=6)
            ax.set_xlabel("x [λ/D]")
            ax.set_ylabel("y [λ/D]")
        for idx in range(n_panels, nrows_maps * ncols):
            ax_unused = fig.add_subplot(gs[idx // ncols, idx % ncols])
            ax_unused.axis("off")

        bottom_gs = gs[nrows_maps, :].subgridspec(1, 1)
        ax_curve = fig.add_subplot(bottom_gs[0, 0])
        requested_roi = np.asarray([float(panel["requested_roi_size_lamD"]) for panel in panels_sorted], dtype=float)
        resolved_roi = np.asarray([float(panel["resolved_roi_size_lamD"]) for panel in panels_sorted], dtype=float)
        snr_vals = np.asarray([float(panel[curve_snr_key]) for panel in panels_sorted], dtype=float)
        ax_curve.plot(requested_roi, snr_vals, "-o", lw=1.8, ms=4.8, color="tab:blue", label=str(curve_snr_label))
        ax_curve.set_xlabel("requested ROI size [λ/D]")
        ax_curve.set_ylabel(str(curve_snr_label), color="tab:blue")
        ax_curve.tick_params(axis="y", labelcolor="tab:blue")
        ax_curve.grid(alpha=0.3)
        if curve_snr_ylim is not None:
            ax_curve.set_ylim(*curve_snr_ylim)
        if np.any(np.abs(resolved_roi - requested_roi) > 1e-12):
            ax_resolved = ax_curve.twinx()
            ax_resolved.plot(requested_roi, resolved_roi, "--s", lw=1.4, ms=4.0, color="tab:orange", label="Resolved ROI")
            ax_resolved.set_ylabel("resolved ROI size [λ/D]", color="tab:orange")
            ax_resolved.tick_params(axis="y", labelcolor="tab:orange")
        title_bits = []
        first_panel = panels_sorted[0]
        if "planet_radius_lamD" in first_panel and "planet_theta_deg" in first_panel:
            title_bits.append(
                f"Planet location: r={float(first_panel['planet_radius_lamD']):.3f} λ/D, "
                f"theta={float(first_panel['planet_theta_deg']):.1f} deg"
            )
        title_bits.append(
            f"xy=({float(first_panel['planet_center_lamD'][0]):+.3f}, {float(first_panel['planet_center_lamD'][1]):+.3f})"
        )
        ax_curve.set_title(f"{curve_snr_label} vs ROI size | " + " | ".join(title_bits))

        pdf.savefig(fig)
        plt.close(fig)

        if include_mean_map_page:
            mean_stack = np.stack(
                [np.asarray(panel[map_key], dtype=float) for panel in panels_sorted],
                axis=0,
            )
            mean_map = np.mean(mean_stack, axis=0)
            finite_vals = mean_map[np.isfinite(mean_map)]
            if finite_vals.size > 0:
                mean_vmax = float(np.percentile(finite_vals, float(vmax_percentile)))
                if vmin_percentile is None:
                    mean_vmin = 0.0
                else:
                    mean_vmin = float(np.percentile(finite_vals, float(vmin_percentile)))
            else:
                mean_vmin = 0.0
                mean_vmax = 1.0
            mean_vmax = max(mean_vmax, 1e-20)
            if mean_vmax <= mean_vmin:
                mean_vmax = mean_vmin + 1e-20

            fig_mean, ax_mean = plt.subplots(1, 1, figsize=(8.0, 7.2), constrained_layout=True)
            im_mean = ax_mean.imshow(
                mean_map,
                origin="lower",
                cmap=cmap,
                extent=extent,
                vmin=mean_vmin,
                vmax=mean_vmax,
            )
            fig_mean.colorbar(im_mean, ax=ax_mean, fraction=0.046, pad=0.04)
            first_panel = panels_sorted[0]
            planet_center = first_panel["planet_center_lamD"]
            planet_x = float(planet_center[0])
            planet_y = float(planet_center[1])
            x_text = extent[1] - 0.8 if planet_x >= 0.0 else extent[0] + 0.8
            x_align = "right" if planet_x >= 0.0 else "left"
            ax_mean.annotate(
                "Planet",
                xy=(planet_x, planet_y),
                xytext=(x_text, planet_y + 0.9),
                color="white",
                fontsize=9,
                ha=x_align,
                va="bottom",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.1),
            )
            requested_roi = np.asarray(
                [float(panel["requested_roi_size_lamD"]) for panel in panels_sorted],
                dtype=float,
            )
            title_lines = [
                f"Mean {title_prefix.lower()} map across ROI sizes",
                "ROI {:.2f} to {:.2f} λ/D ({:d} maps)".format(
                    float(np.min(requested_roi)),
                    float(np.max(requested_roi)),
                    int(requested_roi.size),
                ),
            ]
            ax_mean.set_title("\n".join(title_lines), fontsize=10, pad=8)
            ax_mean.set_xlabel("x [λ/D]")
            ax_mean.set_ylabel("y [λ/D]")
            pdf.savefig(fig_mean)
            plt.close(fig_mean)


def _save_roi_size_fft_spectra_pdf_for_planet_location(
    *,
    output_path: str,
    panels: list[dict[str, object]],
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    if len(panels) == 0:
        return

    panels_sorted = sorted(panels, key=lambda p: float(p["requested_roi_size_lamD"]))
    with PdfPages(output_path) as pdf:
        for panel in panels_sorted:
            spec_freq = np.asarray(panel.get("selection_nonnegative_freqs", np.array([], dtype=float)), dtype=float)
            spec_mag = np.asarray(panel.get("selection_nonnegative_mag", np.array([], dtype=float)), dtype=float)
            if spec_freq.size == 0 or spec_mag.size == 0:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2), constrained_layout=True)
            ax.plot(spec_freq, spec_mag, color="tab:orange", lw=2.0, label="Planet aperture")
            ref_spectra = list(panel.get("reference_spectra", []))
            ref_colors = plt.cm.Blues(np.linspace(0.45, 0.85, max(len(ref_spectra), 1)))
            for color, ref_spec in zip(ref_colors, ref_spectra):
                ref_freq = np.asarray(ref_spec.get("nonnegative_freqs", np.array([], dtype=float)), dtype=float)
                ref_mag = np.asarray(ref_spec.get("nonnegative_mag", np.array([], dtype=float)), dtype=float)
                if ref_freq.size == 0 or ref_mag.size == 0:
                    continue
                center = ref_spec.get("center_lamD", (float("nan"), float("nan")))
                ax.plot(
                    ref_freq,
                    ref_mag,
                    color=color,
                    lw=1.2,
                    alpha=0.95,
                    label=(
                        f"{ref_spec.get('label', 'Speckle')} "
                        f"({float(center[0]):+.2f}, {float(center[1]):+.2f})"
                    ),
                )
            ax.axvline(float(panel["selected_target_freq"]), color="crimson", lw=1.3, ls="--")
            ax.set_xlabel("frequency [cycles/rad]")
            ax.set_ylabel("|FFT|")
            ax.grid(alpha=0.3)
            ax.set_title(
                "Planet-Aperture FFT Spectrum | ROI {:.2f} -> {:.2f} | f={:.4f}".format(
                    float(panel["requested_roi_size_lamD"]),
                    float(panel["resolved_roi_size_lamD"]),
                    float(panel["selected_target_freq"]),
                )
            )
            ax.legend(fontsize=8, loc="best")
            pdf.savefig(fig)
            plt.close(fig)


def _save_planet_locations_on_mean_final_psf(
    *,
    output_path: str,
    sim_local: dict,
    planet_centers_lamD: list[tuple[float, float]],
    crop_lamD: float = 12.0,
) -> None:
    if len(planet_centers_lamD) == 0:
        return

    psf_stack: list[np.ndarray] = []
    n_fft: int | None = None
    samp: float | None = None
    for ctr in planet_centers_lamD:
        result = CoronagraphSimulator(
            **{
                **sim_local,
                "companion_offset_lamD": (float(ctr[0]), float(ctr[1])),
                "e_final_phase_offset": 0.0,
                "focal_local_phase_offset": 0.0,
            }
        ).run()
        psf_stack.append(np.asarray(result["final_psf_with_ghost"], dtype=float))
        if n_fft is None:
            n_fft = int(result["n_fft"])
            samp = float(result["focal_sampling"])

    if n_fft is None or samp is None or len(psf_stack) == 0:
        return

    mean_psf = np.mean(np.stack(psf_stack, axis=0), axis=0)
    half = int(float(crop_lamD) * float(samp))
    cc = int(n_fft // 2)
    sl = slice(cc - half, cc + half)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 7.0), constrained_layout=True)
    im = ax.imshow(
        np.log10(mean_psf[sl, sl] + 1e-12),
        origin="lower",
        cmap="inferno",
        vmin=-8,
        vmax=0,
        extent=[-float(crop_lamD), float(crop_lamD), -float(crop_lamD), float(crop_lamD)],
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(planet_centers_lamD), 1), endpoint=False))
    for idx, ctr in enumerate(planet_centers_lamD):
        color = palette[idx % len(palette)]
        ax.plot([float(ctr[0])], [float(ctr[1])], marker="o", markersize=6.5, color=color, linestyle="None")
        ax.text(
            float(ctr[0]),
            float(ctr[1]),
            str(idx + 1),
            color="white",
            fontsize=9,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="black", alpha=0.45, edgecolor="none"),
        )
    ax.set_title("Planet Locations on Mean Final PSF")
    ax.set_xlabel("x [λ/D]")
    ax.set_ylabel("y [λ/D]")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_planet_position_snr_summary_pdf(
    *,
    output_path: str,
    location_panels: list[dict[str, object]],
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    if len(location_panels) == 0:
        return

    per_page = 6
    ncols = 2
    nrows = 3
    summary_snr_keys = (
        "snr",
    )
    summary_snr_all = np.asarray(
        [
            float(panel[key])
            for item in location_panels
            for panel in item["panels"]
            for key in summary_snr_keys
            if key in panel and np.isfinite(float(panel[key]))
        ],
        dtype=float,
    )
    if summary_snr_all.size > 0:
        summary_ymin = float(np.min(summary_snr_all))
        summary_ymax = float(np.max(summary_snr_all))
        if summary_ymax <= summary_ymin:
            summary_pad = max(abs(summary_ymax) * 0.05, 1e-6)
        else:
            summary_pad = 0.05 * (summary_ymax - summary_ymin)
        summary_ylim = (summary_ymin - summary_pad, summary_ymax + summary_pad)
    else:
        summary_ylim = None
    with PdfPages(output_path) as pdf:
        for start in range(0, len(location_panels), per_page):
            chunk = location_panels[start:start + per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 13.0), constrained_layout=True)
            axes_flat = np.asarray(axes).ravel()
            for ax, item in zip(axes_flat, chunk):
                panels_sorted = sorted(
                    list(item["panels"]),
                    key=lambda p: float(p["requested_roi_size_lamD"]),
                )
                requested_roi = np.asarray(
                    [float(panel["requested_roi_size_lamD"]) for panel in panels_sorted],
                    dtype=float,
                )
                snr_series = [
                    ("SNR", "snr", "tab:blue"),
                ]
                resolved_roi = np.asarray(
                    [float(panel["resolved_roi_size_lamD"]) for panel in panels_sorted],
                    dtype=float,
                )
                for label, key, color in snr_series:
                    if not all(key in panel for panel in panels_sorted):
                        continue
                    vals = np.asarray([float(panel[key]) for panel in panels_sorted], dtype=float)
                    ax.plot(requested_roi, vals, "-o", lw=1.6, ms=4.0, color=color, label=label)
                ax.set_xlabel("requested ROI size [λ/D]")
                ax.set_ylabel("SNR")
                ax.grid(alpha=0.3)
                if summary_ylim is not None:
                    ax.set_ylim(*summary_ylim)
                title = (
                    f"r={float(item['planet_radius_lamD']):.3f} λ/D, "
                    f"theta={float(item['planet_theta_deg']):.1f} deg\n"
                    f"xy=({float(item['planet_center_lamD'][0]):+.3f}, {float(item['planet_center_lamD'][1]):+.3f})"
                )
                ax.set_title(title)
                ax.legend(fontsize=7, loc="best")
                if np.any(np.abs(resolved_roi - requested_roi) > 1e-12):
                    ax2 = ax.twinx()
                    ax2.plot(requested_roi, resolved_roi, "--s", lw=1.2, ms=3.5, color="tab:orange")
                    ax2.set_ylabel("resolved ROI [λ/D]", color="tab:orange")
                    ax2.tick_params(axis="y", labelcolor="tab:orange")
            for ax in axes_flat[len(chunk):]:
                ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

        # Add aggregated summary pages: mean SNR across theta for each sampled radius.
        radius_groups: dict[float, list[dict[str, object]]] = {}
        for item in location_panels:
            radius_key = float(item["planet_radius_lamD"])
            radius_groups.setdefault(radius_key, []).append(item)

        grouped_radii = sorted(radius_groups.keys())
        if len(grouped_radii) > 0:
            for start in range(0, len(grouped_radii), per_page):
                radius_chunk = grouped_radii[start:start + per_page]
                fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 13.0), constrained_layout=True)
                axes_flat = np.asarray(axes).ravel()
                for ax, radius_key in zip(axes_flat, radius_chunk):
                    items = radius_groups[radius_key]
                    roi_to_snrs: dict[float, list[float]] = {}
                    roi_to_resolved: dict[float, list[float]] = {}
                    for item in items:
                        for panel in item["panels"]:
                            if "snr" not in panel:
                                continue
                            requested_roi = float(panel["requested_roi_size_lamD"])
                            roi_to_snrs.setdefault(requested_roi, []).append(float(panel["snr"]))
                            roi_to_resolved.setdefault(requested_roi, []).append(float(panel["resolved_roi_size_lamD"]))
                    requested_roi = np.asarray(sorted(roi_to_snrs.keys()), dtype=float)
                    mean_snr = np.asarray(
                        [float(np.mean(np.asarray(roi_to_snrs[roi], dtype=float))) for roi in requested_roi],
                        dtype=float,
                    )
                    mean_resolved = np.asarray(
                        [float(np.mean(np.asarray(roi_to_resolved[roi], dtype=float))) for roi in requested_roi],
                        dtype=float,
                    )
                    ax.plot(
                        requested_roi,
                        mean_snr,
                        "-o",
                        lw=1.8,
                        ms=4.4,
                        color="tab:purple",
                        label="Mean SNR over theta",
                    )
                    ax.set_xlabel("requested ROI size [λ/D]")
                    ax.set_ylabel("mean SNR over theta")
                    ax.grid(alpha=0.3)
                    if summary_ylim is not None:
                        ax.set_ylim(*summary_ylim)
                    theta_vals = sorted({float(item["planet_theta_deg"]) for item in items})
                    ax.set_title(
                        f"r={radius_key:.3f} λ/D\n"
                        f"mean over theta ({len(theta_vals)} samples)"
                    )
                    ax.legend(fontsize=7, loc="best")
                    if np.any(np.abs(mean_resolved - requested_roi) > 1e-12):
                        ax2 = ax.twinx()
                        ax2.plot(requested_roi, mean_resolved, "--s", lw=1.2, ms=3.5, color="tab:orange")
                        ax2.set_ylabel("mean resolved ROI [λ/D]", color="tab:orange")
                        ax2.tick_params(axis="y", labelcolor="tab:orange")
                for ax in axes_flat[len(radius_chunk):]:
                    ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)


def _run_planet_position_roi_size_sweep(
    args: argparse.Namespace,
    sim_local: dict,
    incoherence_map_mode: str,
    sweep_output_dir: str,
    mask_output_tag: str,
    phase_cycles_tag: str,
    phase_sweep_mode_tag: str,
    single_region_tag: str,
    ghost_suffix: str,
) -> None:
    os.makedirs(sweep_output_dir, exist_ok=True)
    region_shape_name = normalize_region_shape(args.region_shape)
    radius_vals = _inclusive_float_range(
        args.planet_position_radius_min,
        args.planet_position_radius_max,
        args.planet_position_radius_step,
    )
    theta_deg_vals = _inclusive_float_range(
        args.planet_position_theta_min_deg,
        args.planet_position_theta_max_deg,
        args.planet_position_theta_step_deg,
    )
    roi_sizes = _inclusive_float_range(args.roi_size_min, args.roi_size_max, args.roi_size_step)

    base = CoronagraphSimulator(**sim_local).run()
    n_fft = int(base["n_fft"])
    samp = float(base["focal_sampling"])
    central_box_lamD = 24.0
    half16 = int(0.5 * central_box_lamD * samp)
    cc16 = n_fft // 2
    sl16 = slice(cc16 - half16, cc16 + half16)
    x16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, 2 * half16, endpoint=False)
    y16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, 2 * half16, endpoint=False)
    xx16, yy16 = np.meshgrid(x16, y16)
    extent = [-0.5 * central_box_lamD, 0.5 * central_box_lamD, -0.5 * central_box_lamD, 0.5 * central_box_lamD]
    radius_tag = (
        f"_r_{float(args.planet_position_radius_min):.3f}_{float(args.planet_position_radius_max):.3f}_{float(args.planet_position_radius_step):.3f}"
        .replace(".", "p")
    )
    theta_tag = (
        f"_theta_{float(args.planet_position_theta_min_deg):.3f}_{float(args.planet_position_theta_max_deg):.3f}_{float(args.planet_position_theta_step_deg):.3f}"
        .replace(".", "p")
    )
    roi_tag = (
        f"_rmin_{float(args.roi_size_min):.3f}_rmax_{float(args.roi_size_max):.3f}_rstep_{float(args.roi_size_step):.3f}"
        .replace(".", "p")
    )

    phase_cycles = float(args.phase_cycles)
    phase_offsets = np.linspace(0.0, 2.0 * np.pi * phase_cycles, int(args.phase_step), endpoint=True)
    rows: list[dict[str, float | int]] = []
    location_panels: list[dict[str, object]] = []

    for itheta, theta_deg in enumerate(theta_deg_vals):
        for iradius, radius_lamD in enumerate(radius_vals):
            if float(radius_lamD) <= 0.0:
                print(f"[planet-roi-polar] skipping radius {float(radius_lamD):+.3f} because orbit radius is zero")
                continue
            planet_center = _polar_to_cartesian_lamD(
                radius_lamD=float(radius_lamD),
                theta_deg=float(theta_deg),
            )
            sample_rows, best_entry, panels = _evaluate_best_roi_for_planet_center(
                planet_center=planet_center,
                roi_sizes=roi_sizes,
                region_shape_name=region_shape_name,
                sim_local=sim_local,
                phase_offsets=phase_offsets,
                sl16=sl16,
                half16=half16,
                xx16=xx16,
                yy16=yy16,
                incoherence_map_mode=incoherence_map_mode,
                collect_panels=True,
            )
            rows.extend(sample_rows)
            if len(panels) > 0:
                location_tag = f"planet_r_{float(radius_lamD):+.3f}_theta_{float(theta_deg):+.3f}".replace(".", "p").replace("+", "")
                location_dir = os.path.join(sweep_output_dir, location_tag)
                os.makedirs(location_dir, exist_ok=True)
                for panel in panels:
                    panel["planet_radius_lamD"] = float(radius_lamD)
                    panel["planet_theta_deg"] = float(theta_deg)
                location_panels.append(
                    {
                        "planet_radius_lamD": float(radius_lamD),
                        "planet_theta_deg": float(theta_deg),
                        "planet_center_lamD": (float(planet_center[0]), float(planet_center[1])),
                        "panels": list(panels),
                    }
                )
                location_pdf = os.path.join(
                    location_dir,
                    "planet_position_roi_size_sweep_incoherence_maps_with_snr_24lamD_"
                    f"{location_tag}",
                )
                location_pdf = f"{location_pdf}_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_tag}{ghost_suffix}.pdf"
                _save_roi_size_incoherence_pdf_for_planet_location(
                    output_path=location_pdf,
                    region_shape_name=region_shape_name,
                    panels=panels,
                    extent=extent,
                )
                location_coh_pdf = os.path.join(
                    location_dir,
                    "planet_position_roi_size_sweep_coherence_maps_with_snr_24lamD_"
                    f"{location_tag}",
                )
                location_coh_pdf = f"{location_coh_pdf}_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_tag}{ghost_suffix}.pdf"
                _save_roi_size_coherence_pdf_for_planet_location(
                    output_path=location_coh_pdf,
                    region_shape_name=region_shape_name,
                    panels=panels,
                    extent=extent,
                )
                location_max_minus_coh_pdf = os.path.join(
                    location_dir,
                    "planet_position_roi_size_sweep_max_minus_coherence_maps_24lamD_"
                    f"{location_tag}",
                )
                location_max_minus_coh_pdf = f"{location_max_minus_coh_pdf}_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_tag}{ghost_suffix}.pdf"
                _save_roi_size_max_minus_coherence_pdf_for_planet_location(
                    output_path=location_max_minus_coh_pdf,
                    region_shape_name=region_shape_name,
                    panels=panels,
                    extent=extent,
                )
                if str(incoherence_map_mode).strip().lower() == "lab_fft_ratio":
                    location_fft_pdf = os.path.join(
                        location_dir,
                        "planet_position_roi_size_sweep_frequency_selection_spectra_"
                        f"{location_tag}_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_tag}{ghost_suffix}.pdf",
                    )
                    _save_roi_size_fft_spectra_pdf_for_planet_location(
                        output_path=location_fft_pdf,
                        panels=panels,
                    )
                location_csv = os.path.join(
                    location_dir,
                    "planet_position_roi_size_sweep_table_"
                    f"{location_tag}_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_tag}{ghost_suffix}.csv",
                )
                with open(location_csv, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(
                        fh,
                        fieldnames=[
                            "planet_x_lamD",
                            "planet_y_lamD",
                            "orbit_radius_lamD",
                            "planet_theta_rad",
                            "requested_roi_size_lamD",
                            "resolved_roi_size_lamD",
                            "n_circles",
                            "planet_peak",
                            "planet_std",
                            "background_aperture_std",
                            "raw_snr",
                            "snr",
                            "background_aperture_mean",
                            "background_aperture_std_centered",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(sample_rows)
            if best_entry is not None:
                print(
                    "[planet-roi-polar] "
                    f"planet=(r={float(radius_lamD):+.3f}, theta={float(theta_deg):+.3f} deg) "
                    f"xy=({planet_center[0]:+.3f}, {planet_center[1]:+.3f}) "
                    f"best_snr={float(best_entry['snr']):.6e} "
                    f"best_roi={float(best_entry['resolved_roi_size_lamD']):.3f} λ/D"
                )

    out_csv = os.path.join(
        sweep_output_dir,
        "planet_position_roi_size_sweep_table_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{ghost_suffix}.csv",
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "planet_x_lamD",
                "planet_y_lamD",
                "orbit_radius_lamD",
                "planet_theta_rad",
                "requested_roi_size_lamD",
                "resolved_roi_size_lamD",
                "n_circles",
                "planet_peak",
                "planet_std",
                "background_aperture_std",
                "raw_snr",
                "snr",
                "background_aperture_mean",
                "background_aperture_std_centered",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    out_snr_summary_pdf = os.path.join(
        sweep_output_dir,
        "planet_position_roi_size_sweep_snr_summary_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{ghost_suffix}.pdf",
    )
    _save_planet_position_snr_summary_pdf(
        output_path=out_snr_summary_pdf,
        location_panels=location_panels,
    )

    print(f"Saved planet-position/ROI-size sweep table: {out_csv}")
    print(f"Saved planet-position/ROI-size sweep SNR summary PDF: {out_snr_summary_pdf}")


def _run_planet_diagonal_roi_size_sweep(
    args: argparse.Namespace,
    sim_local: dict,
    incoherence_map_mode: str,
    sweep_output_dir: str,
    mask_output_tag: str,
    phase_cycles_tag: str,
    phase_sweep_mode_tag: str,
    single_region_tag: str,
    ghost_suffix: str,
) -> None:
    os.makedirs(sweep_output_dir, exist_ok=True)
    region_shape_name = normalize_region_shape(args.region_shape)
    diag_vals = _inclusive_float_range(args.planet_diagonal_t_min, args.planet_diagonal_t_max, args.planet_diagonal_t_step)
    roi_sizes = _inclusive_float_range(args.roi_size_min, args.roi_size_max, args.roi_size_step)
    y_sign = -1.0 if str(getattr(args, "planet_diagonal_mode", "anti")).strip().lower() == "anti" else 1.0

    base = CoronagraphSimulator(**sim_local).run()
    n_fft = int(base["n_fft"])
    samp = float(base["focal_sampling"])
    central_box_lamD = 24.0
    half16 = int(0.5 * central_box_lamD * samp)
    cc16 = n_fft // 2
    sl16 = slice(cc16 - half16, cc16 + half16)
    x16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, 2 * half16, endpoint=False)
    y16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, 2 * half16, endpoint=False)
    xx16, yy16 = np.meshgrid(x16, y16)
    extent = [-0.5 * central_box_lamD, 0.5 * central_box_lamD, -0.5 * central_box_lamD, 0.5 * central_box_lamD]
    phase_offsets = np.linspace(0.0, 2.0 * np.pi * float(args.phase_cycles), int(args.phase_step), endpoint=True)

    rows: list[dict[str, float | int]] = []
    best_rows: list[dict[str, float | int]] = []
    sampled_planet_centers: list[tuple[float, float]] = []
    for t_val in diag_vals:
        planet_center = (float(t_val), float(y_sign * float(t_val)))
        if float(np.hypot(*planet_center)) <= 0.0:
            print(f"[planet-roi-diag] skipping t={float(t_val):+.3f} because orbit radius is zero")
            continue
        sampled_planet_centers.append(planet_center)
        sample_rows, best_entry, panels = _evaluate_best_roi_for_planet_center(
            planet_center=planet_center,
            roi_sizes=roi_sizes,
            region_shape_name=region_shape_name,
            sim_local=sim_local,
            phase_offsets=phase_offsets,
            sl16=sl16,
            half16=half16,
            xx16=xx16,
            yy16=yy16,
            incoherence_map_mode=incoherence_map_mode,
            collect_panels=True,
        )
        for row in sample_rows:
            row["diagonal_t_lamD"] = float(t_val)
            row["diagonal_mode"] = str(getattr(args, "planet_diagonal_mode", "anti")).strip().lower()
        rows.extend(sample_rows)
        if len(panels) > 0:
            location_pdf = os.path.join(
                sweep_output_dir,
                "planet_diagonal_roi_size_sweep_incoherence_maps_with_snr_"
                f"t_{float(t_val):+.3f}_planet_x_{planet_center[0]:+.3f}_planet_y_{planet_center[1]:+.3f}".replace(".", "p").replace("+", ""),
            )
            location_pdf = f"{location_pdf}_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.pdf"
            _save_roi_size_incoherence_pdf_for_planet_location(
                output_path=location_pdf,
                region_shape_name=region_shape_name,
                panels=panels,
                extent=extent,
            )
        if best_entry is not None:
            best_entry = dict(best_entry)
            best_entry["diagonal_t_lamD"] = float(t_val)
            best_entry["diagonal_mode"] = str(getattr(args, "planet_diagonal_mode", "anti")).strip().lower()
            best_rows.append(best_entry)
            print(
                "[planet-roi-diag] "
                f"t={float(t_val):+.3f} planet=({planet_center[0]:+.3f}, {planet_center[1]:+.3f}) "
                f"best_snr={float(best_entry['snr']):.6e} best_roi={float(best_entry['resolved_roi_size_lamD']):.3f} λ/D"
            )

    diag_mode = str(getattr(args, "planet_diagonal_mode", "anti")).strip().lower()
    diag_tag = (
        f"_{diag_mode}_t_{float(args.planet_diagonal_t_min):.3f}_{float(args.planet_diagonal_t_max):.3f}_{float(args.planet_diagonal_t_step):.3f}"
        .replace(".", "p")
    )
    roi_tag = (
        f"_rmin_{float(args.roi_size_min):.3f}_rmax_{float(args.roi_size_max):.3f}_rstep_{float(args.roi_size_step):.3f}"
        .replace(".", "p")
    )
    out_csv = os.path.join(
        sweep_output_dir,
        "planet_diagonal_roi_size_sweep_table_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{diag_tag}{roi_tag}{ghost_suffix}.csv",
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "diagonal_t_lamD",
                "diagonal_mode",
                "planet_x_lamD",
                "planet_y_lamD",
                "orbit_radius_lamD",
                "planet_theta_rad",
                "requested_roi_size_lamD",
                "resolved_roi_size_lamD",
                "n_circles",
                "planet_peak",
                "background_aperture_std",
                "snr",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if len(best_rows) > 0:
        t_arr = np.asarray([float(row["diagonal_t_lamD"]) for row in best_rows], dtype=float)
        snr_arr = np.asarray([float(row["snr"]) for row in best_rows], dtype=float)
        roi_arr = np.asarray([float(row["resolved_roi_size_lamD"]) for row in best_rows], dtype=float)
        peak_arr = np.asarray([float(row["planet_peak"]) for row in best_rows], dtype=float)
        med_arr = np.asarray([float(row["background_aperture_std"]) for row in best_rows], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True)
        axes[0].plot(t_arr, snr_arr, "-o", lw=1.8, ms=4.5, color="tab:blue")
        axes[0].set_title("Best SNR Along Diagonal")
        axes[0].set_xlabel("diagonal t [λ/D]")
        axes[0].set_ylabel("SNR")
        axes[0].grid(alpha=0.3)
        axes[1].plot(t_arr, roi_arr, "-o", lw=1.8, ms=4.5, color="tab:orange")
        axes[1].set_title("Best ROI Size Along Diagonal")
        axes[1].set_xlabel("diagonal t [λ/D]")
        axes[1].set_ylabel("resolved ROI size [λ/D]")
        axes[1].grid(alpha=0.3)
        out_summary = os.path.join(
            sweep_output_dir,
            "planet_diagonal_roi_size_sweep_summary_"
            f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{diag_tag}{roi_tag}{ghost_suffix}.png",
        )
        fig.savefig(out_summary, dpi=170, bbox_inches="tight")
        plt.close(fig)

        fig_aux, axes_aux = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True)
        axes_aux[0].plot(t_arr, peak_arr, "-o", lw=1.8, ms=4.5, color="tab:red")
        axes_aux[0].set_title("Peak Incoherence at Best ROI")
        axes_aux[0].set_xlabel("diagonal t [λ/D]")
        axes_aux[0].set_ylabel("peak incoherence")
        axes_aux[0].grid(alpha=0.3)
        axes_aux[1].plot(t_arr, med_arr, "-o", lw=1.8, ms=4.5, color="tab:green")
        axes_aux[1].set_title("Annulus Median at Best ROI")
        axes_aux[1].set_xlabel("diagonal t [λ/D]")
        axes_aux[1].set_ylabel("background ring std")
        axes_aux[1].grid(alpha=0.3)
        out_aux = os.path.join(
            sweep_output_dir,
            "planet_diagonal_roi_size_sweep_aux_"
            f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{diag_tag}{roi_tag}{ghost_suffix}.png",
        )
        fig_aux.savefig(out_aux, dpi=170, bbox_inches="tight")
        plt.close(fig_aux)
        print(f"Saved planet-diagonal/ROI-size sweep summary plot: {out_summary}")
        print(f"Saved planet-diagonal/ROI-size sweep auxiliary plot: {out_aux}")
    out_mean_psf = os.path.join(
        sweep_output_dir,
        "planet_diagonal_mean_final_psf_with_locations_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{diag_tag}{roi_tag}{ghost_suffix}.png",
    )
    _save_planet_locations_on_mean_final_psf(
        output_path=out_mean_psf,
        sim_local=sim_local,
        planet_centers_lamD=sampled_planet_centers,
    )
    print(f"Saved planet locations on mean final PSF: {out_mean_psf}")
    print(f"Saved planet-diagonal/ROI-size sweep table: {out_csv}")


def _build_ring_probe_pixels(
    n_fft: int,
    focal_sampling: float,
    orbit_radius_lamD: float,
    base_angle_rad: float,
    angular_step_rad: float,
    n_probes: int = 10,
) -> list[dict[str, float | int]]:
    c = (float(n_fft) - 1.0) / 2.0
    probes: list[dict[str, float | int]] = []
    seen_pixels: set[tuple[int, int]] = set()
    max_attempts = max(int(n_probes), 1) * 8
    step_idx = 0
    while len(probes) < int(n_probes) and step_idx < max_attempts:
        theta = float(base_angle_rad + float(step_idx) * angular_step_rad)
        x_lamD = float(orbit_radius_lamD * np.cos(theta))
        y_lamD = float(orbit_radius_lamD * np.sin(theta))
        x_idx = int(np.clip(np.round(c + x_lamD * focal_sampling), 0, int(n_fft) - 1))
        y_idx = int(np.clip(np.round(c + y_lamD * focal_sampling), 0, int(n_fft) - 1))
        pixel_key = (y_idx, x_idx)
        if pixel_key not in seen_pixels:
            seen_pixels.add(pixel_key)
            probes.append(
                {
                    "probe_index": int(len(probes)),
                    "step_index": int(step_idx),
                    "theta_rad": theta,
                    "x_lamD": x_lamD,
                    "y_lamD": y_lamD,
                    "x_idx": x_idx,
                    "y_idx": y_idx,
                }
            )
        step_idx += 1
    return probes


def _save_ring_rotation_probe_fft_page(
    pdf,
    img_last: np.ndarray,
    stack: np.ndarray,
    phase_series: np.ndarray,
    probes: list[dict[str, float | int]],
    rotation_fraction: float,
    applied_rotation_rad: float,
    region_centers: list[tuple[float, float]],
    region_radius_lamD: float,
    planet_center_lamD: tuple[float, float],
    sl_crop: slice,
    psf_crop_lamD: float,
    n_fft: int,
    focal_sampling: float,
) -> None:
    if img_last is None or stack.size == 0 or phase_series.size < 2 or len(probes) == 0:
        return

    dphi = float(np.mean(np.diff(phase_series)))
    freq = np.fft.fftfreq(int(phase_series.size), d=dphi)
    pos = freq >= 0.0
    crop_center = int(n_fft // 2)
    crop_half = int((sl_crop.stop - sl_crop.start) // 2)
    c_full = (float(n_fft) - 1.0) / 2.0

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), constrained_layout=True)
    ax_ts, ax_fft, ax_img = axes
    palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(probes), 1), endpoint=False))
    fft_xticks: list[float] = []

    for j, probe in enumerate(probes):
        color = palette[j % len(palette)]
        x_idx = int(probe["x_idx"])
        y_idx = int(probe["y_idx"])
        x_local = x_idx - (crop_center - crop_half)
        y_local = y_idx - (crop_center - crop_half)
        if y_local < 0 or y_local >= stack.shape[1] or x_local < 0 or x_local >= stack.shape[2]:
            continue
        trace = stack[:, y_local, x_local]
        fft_trace = np.fft.fft(trace)
        amp = np.abs(fft_trace) / max(trace.size, 1)
        label = "planet center" if j == 0 else f"probe {j}"
        freq_pos = freq[pos]
        amp_pos = amp[pos]
        fft_xticks = [float(v) for v in freq_pos]
        ax_ts.plot(
            phase_series,
            trace,
            lw=2.0 if j == 0 else 1.2,
            alpha=1.0 if j == 0 else 0.85,
            color=color,
            label=label,
        )
        ax_fft.plot(
            freq_pos,
            amp_pos,
            lw=2.0 if j == 0 else 1.2,
            alpha=1.0 if j == 0 else 0.85,
            color=color,
            marker="o",
            markersize=3.5 if j == 0 else 2.5,
            label=label,
        )
        x_lamD = (float(x_idx) - c_full) / float(focal_sampling)
        y_lamD = (float(y_idx) - c_full) / float(focal_sampling)
        ax_img.plot(
            [x_lamD],
            [y_lamD],
            marker="o",
            markersize=4.0 if j == 0 else 3.0,
            color=color,
            linestyle="None",
        )
        ax_img.text(x_lamD, y_lamD, str(j), color="white", fontsize=10, ha="left", va="bottom")

    phase_mid = 0.5 * (float(phase_series[0]) + float(phase_series[-1]))
    ax_ts.set_title("Fixed Probe-Pixel Time Series")
    ax_ts.set_xlabel("Local phase shift [rad]")
    ax_ts.set_ylabel("Intensity")
    ax_ts.set_xticks([float(phase_series[0]), phase_mid, float(phase_series[-1])])
    ax_ts.set_xticklabels([f"{phase_series[0]/np.pi:.1f}π", f"{phase_mid/np.pi:.1f}π", f"{phase_series[-1]/np.pi:.1f}π"])
    ax_ts.grid(alpha=0.3)
    ax_ts.legend(fontsize=10, ncol=1, loc="best")

    ax_fft.set_title("Fixed Probe-Pixel FFT Magnitude")
    ax_fft.set_xlabel("Frequency [cycles/rad]")
    ax_fft.set_ylabel("Amplitude")
    if fft_xticks:
        ax_fft.set_xticks(fft_xticks)
        ax_fft.set_xticklabels([f"{tick:.3f}" for tick in fft_xticks], rotation=90, fontsize=6)
    ax_fft.grid(alpha=0.3)
    ax_fft.legend(fontsize=10, ncol=1, loc="best")

    im = ax_img.imshow(
        np.log10(img_last[sl_crop, sl_crop] + 1e-12),
        origin="lower",
        cmap="inferno",
        vmin=-8,
        vmax=0,
        extent=[-psf_crop_lamD, psf_crop_lamD, -psf_crop_lamD, psf_crop_lamD],
    )
    for j, (cx, cy) in enumerate(region_centers):
        edge = "lime" if j == 0 else "cyan"
        ax_img.add_patch(plt.Circle((cx, cy), region_radius_lamD, fill=False, edgecolor=edge, linewidth=1.2))
    ax_img.plot([planet_center_lamD[0]], [planet_center_lamD[1]], marker="+", color="white", markersize=9, linestyle="None")
    ax_img.set_title(
        f"Probe Pixels on Rotated Ring\nu={rotation_fraction:.2f}, rot={applied_rotation_rad:.3f} rad"
    )
    ax_img.set_xlabel("x [λ/D]")
    ax_img.set_ylabel("y [λ/D]")
    fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
    pdf.savefig(fig)
    plt.close(fig)


def _save_ring_of_circle_rotation_gif(
    gif_path: str,
    requested_region_radius_lamD: float,
    orbit_radius_lamD: float,
    anchor_angle_rad: float,
    fixed_center_lamD: tuple[float, float],
    resolved_region_radius_lamD: float,
    n_circles: int,
    n_frames: int = 21,
) -> None:
    frames: list[np.ndarray] = []
    max_extent = max(orbit_radius_lamD + 2.5 * resolved_region_radius_lamD, orbit_radius_lamD + 1.5, 2.0)
    u_values = np.linspace(0.0, 1.0, max(2, int(n_frames)))
    for u in u_values:
        ring = build_touching_circle_ring(
            requested_region_radius_lamD=requested_region_radius_lamD,
            orbit_radius_lamD=orbit_radius_lamD,
            anchor_angle_rad=anchor_angle_rad,
            rotation_fraction=float(u),
            min_circles=int(n_circles),
        )
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2), constrained_layout=True)
        ax.set_aspect("equal")
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        ax.grid(alpha=0.2)
        ax.add_patch(plt.Circle((0.0, 0.0), orbit_radius_lamD, fill=False, edgecolor="0.65", linestyle="--", linewidth=1.2))
        ax.plot(0.0, 0.0, marker="+", color="black", markersize=8, markeredgewidth=1.5)
        ax.plot(fixed_center_lamD[0], fixed_center_lamD[1], marker="o", color="tab:red", markersize=6)
        for idx, (cx, cy) in enumerate(ring["centers_lamD"]):
            edge_color = "tab:green" if idx == 0 else "tab:blue"
            line_width = 2.0 if idx == 0 else 1.4
            ax.add_patch(
                plt.Circle((cx, cy), ring["resolved_radius_lamD"], fill=False, edgecolor=edge_color, linewidth=line_width)
            )
            ax.plot(cx, cy, marker=".", color=edge_color, markersize=5)
        ax.set_title("ring_of_circle rotation")
        ax.set_xlabel("x [λ/D]")
        ax.set_ylabel("y [λ/D]")
        ax.text(
            0.02,
            0.98,
            (
                f"u={u:.2f}\n"
                f"rotation={ring['applied_rotation_rad']:.4f} rad\n"
                f"edge target={ring['edge_cut_rotation_rad']:.4f} rad\n"
                f"r={ring['resolved_radius_lamD']:.4f} λ/D, N={ring['n_circles']}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

    _write_rgb_gif(np.asarray(frames, dtype=np.uint8), gif_path, duration_ms=250)


def _run_roi_size_sweep_snr_vs_theta(
    args: argparse.Namespace,
    sim_local: dict,
    incoherence_map_mode: str,
    sweep_output_dir: str,
    mask_output_tag: str,
    phase_cycles_tag: str,
    phase_sweep_mode_tag: str,
    single_region_tag: str,
    ghost_suffix: str,
    orbit_radius_lamD: float,
    initial_angle_rad: float,
    centers_lamD: list[tuple[float, float]],
    planet_center_lamD: tuple[float, float],
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    roi_dir = sweep_output_dir
    os.makedirs(roi_dir, exist_ok=True)
    region_shape_name = normalize_region_shape(args.region_shape)

    roi_min = float(args.roi_size_min)
    roi_max = float(args.roi_size_max)
    roi_step = float(args.roi_size_step)
    roi_sizes = np.arange(roi_min, roi_max + 0.5 * roi_step, roi_step, dtype=float)
    roi_min_tag = f"{roi_min:.3f}".replace(".", "p")
    roi_max_tag = f"{roi_max:.3f}".replace(".", "p")
    roi_step_tag = f"{roi_step:.3f}".replace(".", "p")
    roi_sweep_tag = f"_rmin_{roi_min_tag}_rmax_{roi_max_tag}_rstep_{roi_step_tag}"
    fixed_planet_eval_radius_lamD = SNR_APERTURE_RADIUS_LAMD
    snr_eps = 1e-12

    phase_cycles = float(args.phase_cycles)
    phase_offsets = np.linspace(0.0, 2.0 * np.pi * phase_cycles, int(args.phase_step), endpoint=True)
    theta_samples = max(1, int(args.fov_centers_count))
    theta_rel = _theta_back_and_forth(theta_samples, max_abs=np.pi)
    if region_shape_name not in {"ring", "ring_of_circle"}:
        print(f"ROI-size sweep theta samples [rad]: {theta_rel.tolist()}")

    base = CoronagraphSimulator(**sim_local).run()
    n_fft = int(base["n_fft"])
    samp = float(base["focal_sampling"])
    central_box_lamD = 24.0
    half16 = int(0.5 * central_box_lamD * samp)
    cc16 = n_fft // 2
    sl16 = slice(cc16 - half16, cc16 + half16)
    x16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, 2 * half16, endpoint=False)
    y16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, 2 * half16, endpoint=False)
    xx16, yy16 = np.meshgrid(x16, y16)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.4), constrained_layout=True)
    out_psf_pdf = os.path.join(
        roi_dir,
        "roi_size_sweep_final_psf_with_regions_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_sweep_tag}{ghost_suffix}.pdf",
    )
    out_incoh_pdf = os.path.join(
        roi_dir,
        "roi_size_sweep_incoherence_maps_with_snr_24lamD_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_sweep_tag}{ghost_suffix}.pdf",
    )
    out_table_csv = os.path.join(
        roi_dir,
        "roi_size_sweep_snr_theta_table_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_sweep_tag}{ghost_suffix}.csv",
    )
    psf_crop_lamD = 10.0
    half_crop = int(psf_crop_lamD * samp)
    cc = n_fft // 2
    sl_crop = slice(cc - half_crop, cc + half_crop)
    table_rows: list[dict[str, float]] = []
    ring_requested_sizes: list[float] = []
    ring_resolved_sizes: list[float] = []
    ring_snrs: list[float] = []
    ring_incoh_panels: list[dict[str, object]] = []
    with PdfPages(out_incoh_pdf) as pdf_incoh:
        # Use high-contrast categorical colors for better curve distinguishability.
        base_cmap = plt.get_cmap("tab10" if len(roi_sizes) <= 10 else "tab20")
        curve_colors = [base_cmap(i % base_cmap.N) for i in range(max(len(roi_sizes), 1))]
        for roi_r in roi_sizes:
            color_idx = int(np.where(np.isclose(roi_sizes, roi_r))[0][0]) if len(roi_sizes) > 0 else 0
            curve_color = curve_colors[color_idx % len(curve_colors)]
            snr_vals: list[float] = []
            if region_shape_name == "ring_of_circle":
                ring = build_touching_circle_ring(
                    requested_region_radius_lamD=float(roi_r),
                    orbit_radius_lamD=orbit_radius_lamD,
                    anchor_angle_rad=initial_angle_rad,
                    rotation_fraction=0.0,
                )
                roi_centers = [(float(cx), float(cy)) for cx, cy in ring["centers_lamD"]]
                roi_radius_eff = float(ring["resolved_radius_lamD"])
                theta_items = [(0.0, planet_center_lamD)]
            elif region_shape_name == "ring":
                roi_centers = [(float(planet_center_lamD[0]), float(planet_center_lamD[1]))]
                roi_radius_eff = float(roi_r)
                theta_items = [(0.0, planet_center_lamD)]
            else:
                roi_centers = [(float(cx), float(cy)) for cx, cy in centers_lamD]
                roi_radius_eff = float(roi_r)
                theta_items = []
                for th_rel in theta_rel:
                    th = float(initial_angle_rad + th_rel)
                    ctr = (float(orbit_radius_lamD * np.cos(th)), float(orbit_radius_lamD * np.sin(th)))
                    theta_items.append((float(th_rel), ctr))

            for th_rel_eff, ctr in theta_items:

                stack = np.zeros((phase_offsets.size, 2 * half16, 2 * half16), dtype=float)
                for i, ph in enumerate(phase_offsets):
                    phase_sim = CoronagraphSimulator(
                        **{
                            **sim_local,
                            "e_final_phase_offset": 0.0,
                            "focal_local_phase_offset": float(ph),
                            **_local_phase_region_kwargs(
                                region_shape_name=region_shape_name,
                                region_width_or_radius_lamD=float(roi_radius_eff),
                                orbit_radius_lamD=orbit_radius_lamD,
                                centers_lamD=roi_centers if region_shape_name in {"ring", "ring_of_circle"} else [ctr],
                            ),
                        }
                    )
                    img = phase_sim.run()["final_psf_with_ghost"]
                    stack[i] = img[sl16, sl16]

                phase_series = np.asarray(phase_offsets, dtype=float)
                if phase_series.size > 2 and np.isclose(phase_series[0], 0.0) and np.isclose(phase_series[-1], float(phase_series.max())):
                    phase_series = phase_series[:-1]
                    stack = stack[:-1]
                planet_mask = (
                    (xx16 - float(planet_center_lamD[0])) ** 2
                    + (yy16 - float(planet_center_lamD[1])) ** 2
                    <= float(fixed_planet_eval_radius_lamD) ** 2
                )
                incoh = _compute_incoherence_map(
                    stack=stack,
                    phase_offsets=phase_series,
                    mode=incoherence_map_mode,
                    planet_region_mask=planet_mask,
                )
                if incoh is None:
                    continue

                # Evaluate SNR at the fixed planet region across all FOV-center locations.
                peak, med, snr = _planet_region_snr(
                    incoh=incoh,
                    xx=xx16,
                    yy=yy16,
                    planet_center_lamD=planet_center_lamD,
                    orbit_radius_lamD=orbit_radius_lamD,
                    eval_radius_lamD=fixed_planet_eval_radius_lamD,
                    annulus_half_width_lamD=SNR_ANNULUS_HALF_WIDTH_LAMD,
                    snr_eps=snr_eps,
                )
                planet_std = float(np.std(np.asarray(incoh[planet_mask], dtype=float))) if np.any(planet_mask) else float("nan")
                snr_vals.append(snr)
                table_rows.append(
                    {
                        "roi_radius_lamD": float(roi_r),
                        "resolved_roi_radius_lamD": float(roi_radius_eff),
                        "n_circles": int(len(roi_centers)),
                        "theta_rel_rad": float(th_rel_eff),
                        "active_center_x_lamD": float(ctr[0]),
                        "active_center_y_lamD": float(ctr[1]),
                        "planet_peak": float(peak),
                        "planet_std": float(planet_std),
                        "background_aperture_std": float(med),
                        "snr": float(snr),
                    }
                )

                fig_m, ax_m = plt.subplots(1, 1, figsize=(7.2, 6.2), constrained_layout=True)
                im_m = ax_m.imshow(
                    incoh,
                    origin="lower",
                    cmap="viridis",
                    extent=[-0.5 * central_box_lamD, 0.5 * central_box_lamD, -0.5 * central_box_lamD, 0.5 * central_box_lamD],
                )
                fig_m.colorbar(im_m, ax=ax_m, fraction=0.046, pad=0.04)
                ax_m.set_title(
                    f"ROI Size Sweep Incoherence Map: {roi_r:.3f} λ/D ROI"
                    f" | 24x24 λ/D crop"
                )
                ax_m.add_patch(
                    plt.Circle(
                        (float(planet_center_lamD[0]), float(planet_center_lamD[1])),
                        float(fixed_planet_eval_radius_lamD),
                        fill=False,
                        edgecolor="white",
                        linewidth=0.3,
                        linestyle="-",
                        label=f"planet region (r={fixed_planet_eval_radius_lamD:.1f} λ/D)",
                    )
                )
                ax_m.plot(
                    [float(ctr[0])],
                    [float(ctr[1])],
                    marker="x",
                    markersize=7,
                    markeredgewidth=1.4,
                    color="white",
                    linestyle="None",
                    label="POV center",
                )
                ax_m.add_patch(
                    plt.Circle(
                        (0.0, 0.0),
                        float(max(orbit_radius_lamD - SNR_ANNULUS_HALF_WIDTH_LAMD, 0.0)),
                        fill=False,
                        edgecolor="orange",
                        linewidth=1.2,
                        linestyle="--",
                    )
                )
                ax_m.add_patch(
                    plt.Circle(
                        (0.0, 0.0),
                        float(orbit_radius_lamD + SNR_ANNULUS_HALF_WIDTH_LAMD),
                        fill=False,
                        edgecolor="orange",
                        linewidth=1.2,
                        linestyle="--",
                        label="annulus",
                    )
                )
                if region_shape_name in {"ring", "ring_of_circle"}:
                    ring_incoh_panels.append(
                        {
                            "requested_roi_radius_lamD": float(roi_r),
                            "resolved_roi_radius_lamD": float(roi_radius_eff),
                            "n_circles": int(len(roi_centers)),
                            "snr": float(snr),
                            "peak": float(peak),
                            "median": float(med),
                            "centers": [(float(cx), float(cy)) for cx, cy in roi_centers],
                            "map": np.array(incoh, dtype=float),
                        }
                    )
                else:
                    ax_m.plot([ctr[0]], [ctr[1]], marker="o", markersize=4.5, color="cyan", linestyle="None", label="active FOV center")
                    th_rel_disp = 0.0 if abs(float(th_rel_eff)) < 1e-10 else float(th_rel_eff)
                    ax_m.set_title(
                        f"Incoherence Map ({incoherence_map_mode}) | ROI r={roi_r:.2f} λ/D"
                        f" -> resolved {roi_radius_eff:.2f} λ/D | theta={th_rel_disp:+.3f} rad"
                    )
                ax_m.set_xlabel("x [λ/D]")
                ax_m.set_ylabel("y [λ/D]")
                if region_shape_name not in {"ring", "ring_of_circle"}:
                    ax_m.legend(loc="upper right", fontsize=8)
                    ax_m.text(
                        0.02,
                        0.98,
                        (
                            f"SNR={snr:.6e}\nsignal mean={peak:.6e}\nbackground ring std={med:.6e}\n"
                            f"planet eval r={fixed_planet_eval_radius_lamD:.1f} λ/D"
                        ),
                        transform=ax_m.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.60, edgecolor="none"),
                    )
                    pdf_incoh.savefig(fig_m)
                plt.close(fig_m)

            if region_shape_name in {"ring", "ring_of_circle"}:
                ring_requested_sizes.append(float(roi_r))
                ring_resolved_sizes.append(float(roi_radius_eff))
                ring_snrs.append(float(snr_vals[0]) if len(snr_vals) > 0 else float("nan"))
            else:
                theta_arr = np.asarray([item[0] for item in theta_items], dtype=float)
                snr_arr = np.asarray(snr_vals, dtype=float)
                order = np.argsort(theta_arr)
                ax.plot(
                    theta_arr[order],
                    snr_arr[order],
                    "-o",
                    lw=1.3,
                    ms=4.0,
                    color=curve_color,
                    label=f"ROI r={roi_r:.2f} λ/D",
                )
        if region_shape_name in {"ring", "ring_of_circle"} and len(ring_incoh_panels) > 0:
            def _log10_map(map_data: np.ndarray) -> np.ndarray:
                arr = np.asarray(map_data, dtype=float)
                return np.where(arr > 0.0, np.log10(arr), np.nan)

            all_vals = np.concatenate([_log10_map(panel["map"]).ravel() for panel in ring_incoh_panels])
            finite_vals = all_vals[np.isfinite(all_vals)]
            if finite_vals.size > 0:
                vmin = float(np.nanpercentile(finite_vals, 5.0))
                vmax = float(np.nanpercentile(finite_vals, 99.0))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
                    vmin = float(np.nanmin(finite_vals))
                    vmax = float(np.nanmax(finite_vals))
            else:
                vmin, vmax = 0.0, 1.0
            n_panels = len(ring_incoh_panels)
            ncols = int(np.ceil(np.sqrt(float(n_panels))))
            nrows = int(np.ceil(float(n_panels) / float(ncols)))
            extent16 = [-0.5 * central_box_lamD, 0.5 * central_box_lamD, -0.5 * central_box_lamD, 0.5 * central_box_lamD]
            poster_bg = "#08111d"
            panel_bg = "#0b1626"
            panel_edge = "#32455f"
            title_color = "#f6f1df"
            label_color = "#c7d5ea"
            mint_edge = "#88f0c4"
            powder_edge = "#8fc8ff"
            copper_edge = "#d89a52"
            snr_line = "#8ef0c5"
            grid_color = "#8ca0bf"
            panel_title_fontsize = 34
            panel_text_fontsize = 24
            axis_label_fontsize = 27
            tick_label_fontsize = 23
            snr_axis_label_fontsize = 30
            snr_tick_label_fontsize = 25
            snr_title_fontsize = 36
            suptitle_fontsize = 44
            subtitle_fontsize = 24
            colorbar_label_fontsize = 26
            colorbar_tick_fontsize = 22
            panel_size = 7.8
            fig_w = max(26.0, panel_size * float(ncols) + 4.8)
            fig_h = max(24.0, panel_size * float(nrows) + 11.0)
            fig_p = plt.figure(figsize=(fig_w, fig_h))
            fig_p.patch.set_facecolor(poster_bg)
            ref_panel = min(
                ring_incoh_panels,
                key=lambda panel: abs(float(panel["requested_roi_radius_lamD"]) - float(panel["resolved_roi_radius_lamD"])),
            )
            ref_bg = fig_p.add_axes([0.0, 0.0, 1.0, 1.0], zorder=0)
            ref_bg.set_facecolor(poster_bg)
            ref_map = _log10_map(ref_panel["map"])
            ref_norm = ref_map - np.nanmin(ref_map)
            ref_scale = np.nanpercentile(ref_norm[np.isfinite(ref_norm)], 99.0) if np.any(np.isfinite(ref_norm)) else 0.0
            if np.isfinite(ref_scale) and ref_scale > 0.0:
                ref_norm = np.clip(ref_norm / ref_scale, 0.0, 1.0)
            else:
                ref_norm = np.zeros_like(ref_norm)
            ref_bg.imshow(ref_norm, origin="lower", cmap="inferno", alpha=0.12, interpolation="bilinear", aspect="auto")
            ref_bg.axis("off")
            gs = fig_p.add_gridspec(
                nrows + 1,
                ncols,
                height_ratios=[1.0] * nrows + [1.15],
                left=0.07,
                right=0.93,
                top=0.78,
                bottom=0.08,
                hspace=0.42,
                wspace=0.24,
            )
            axes_flat = []
            for row_idx in range(nrows):
                row_start = row_idx * ncols
                row_remaining = max(0, n_panels - row_start)
                row_count = min(ncols, row_remaining)
                if row_count <= 0:
                    row_count = ncols
                if row_count == ncols:
                    row_axes = [fig_p.add_subplot(gs[row_idx, col_idx], zorder=2) for col_idx in range(ncols)]
                else:
                    start_col = (ncols - row_count) // 2
                    row_axes = [
                        fig_p.add_subplot(gs[row_idx, start_col + col_idx], zorder=2)
                        for col_idx in range(row_count)
                    ]
                axes_flat.extend(row_axes)
            last_im = None
            for ax_i, panel in enumerate(ring_incoh_panels):
                ax_p = axes_flat[ax_i]
                ax_p.set_facecolor(panel_bg)
                panel_map_log10 = _log10_map(panel["map"])
                last_im = ax_p.imshow(
                    panel_map_log10,
                    origin="lower",
                    cmap="inferno",
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent16,
                    aspect="equal",
                )
                if region_shape_name == "ring":
                    ring_rmin_lamD, ring_rmax_lamD = annulus_radii_from_width(
                        mid_radius_lamD=orbit_radius_lamD,
                        width_lamD=float(panel["resolved_roi_radius_lamD"]),
                    )
                    ax_p.add_patch(plt.Circle((0.0, 0.0), float(ring_rmin_lamD), fill=False, edgecolor=mint_edge, linewidth=2.0))
                    ax_p.add_patch(plt.Circle((0.0, 0.0), float(ring_rmax_lamD), fill=False, edgecolor=powder_edge, linewidth=2.0))
                else:
                    for j, (cx, cy) in enumerate(panel["centers"]):
                        edge = mint_edge if j == 0 else powder_edge
                        ax_p.add_patch(plt.Circle((cx, cy), float(panel["resolved_roi_radius_lamD"]), fill=False, edgecolor=edge, linewidth=2.0))
                ax_p.add_patch(
                    plt.Circle(
                        (float(planet_center_lamD[0]), float(planet_center_lamD[1])),
                        float(fixed_planet_eval_radius_lamD),
                        fill=False,
                        edgecolor=title_color,
                        linewidth=2.2,
                    )
                )
                ax_p.add_patch(
                    plt.Circle(
                        (0.0, 0.0),
                        float(max(orbit_radius_lamD - SNR_ANNULUS_HALF_WIDTH_LAMD, 0.0)),
                        fill=False,
                        edgecolor=copper_edge,
                        linewidth=1.8,
                        linestyle="--",
                    )
                )
                ax_p.add_patch(
                    plt.Circle(
                        (0.0, 0.0),
                        float(orbit_radius_lamD + SNR_ANNULUS_HALF_WIDTH_LAMD),
                        fill=False,
                        edgecolor=copper_edge,
                        linewidth=1.8,
                        linestyle="--",
                    )
                )
                ax_p.set_title(
                    "Used {:.2f} λ/D\nSNR {:.2e}".format(
                        float(panel["resolved_roi_radius_lamD"]),
                        float(panel["snr"]),
                    ),
                    fontsize=panel_title_fontsize,
                    color=title_color,
                    pad=16.0,
                )
                ax_p.text(
                    0.02,
                    0.03,
                    "",
                    transform=ax_p.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=panel_text_fontsize,
                    color=title_color,
                    bbox=dict(boxstyle="round,pad=0.32", facecolor="#111a28", alpha=0.56, edgecolor="none"),
                )
                ax_p.tick_params(colors=label_color, labelsize=tick_label_fontsize)
                for spine in ax_p.spines.values():
                    spine.set_color(panel_edge)
                if ax_i % ncols == 0:
                    ax_p.set_ylabel("y [λ/D]", color=label_color, fontsize=axis_label_fontsize)
                else:
                    ax_p.set_ylabel("")
                if ax_i // ncols == nrows - 1:
                    ax_p.set_xlabel("x [λ/D]", color=label_color, fontsize=axis_label_fontsize)
                else:
                    ax_p.set_xlabel("")
            for ax_p in axes_flat[n_panels:]:
                ax_p.set_facecolor(poster_bg)
                ax_p.axis("off")
            snr_col_span = max(1, ncols)
            snr_col_start = max(0, (ncols - snr_col_span) // 2)
            ax_snr = fig_p.add_subplot(gs[nrows, snr_col_start:snr_col_start + snr_col_span], zorder=2)
            ax_snr.set_facecolor("#0a1422")
            req_arr = np.asarray(ring_requested_sizes, dtype=float)
            res_arr = np.asarray(ring_resolved_sizes, dtype=float)
            snr_arr = np.asarray(ring_snrs, dtype=float)
            ax_snr.plot(res_arr, snr_arr, "-o", lw=4.4, ms=10.0, color=snr_line, markerfacecolor="#b8ffe5", markeredgecolor=snr_line)
            ax_snr.set_xlabel("resolved circle radius [λ/D]", color=label_color, fontsize=snr_axis_label_fontsize)
            ax_snr.set_ylabel("planet-region SNR", color=label_color, fontsize=snr_axis_label_fontsize)
            ax_snr.set_title("SNR vs Resolved Circle Radius", fontsize=snr_title_fontsize, color=title_color, pad=8.0)
            ax_snr.grid(alpha=0.22, color=grid_color)
            ax_snr.tick_params(colors=label_color, labelsize=snr_tick_label_fontsize)
            for spine in ax_snr.spines.values():
                spine.set_color(panel_edge)
            fig_p.suptitle(
                "ROI Size Sweep Incoherence Maps",
                fontsize=88,
                color=title_color,
                fontweight="bold",
                y=0.975,
            )
            pdf_incoh.savefig(fig_p, facecolor=fig_p.get_facecolor())
            plt.close(fig_p)
    # Write all ROI-size PSF overlays into one PDF (one page per ROI radius).
    with PdfPages(out_psf_pdf) as pdf:
        for roi_r in roi_sizes:
            fig_case, ax_case = plt.subplots(1, 1, figsize=(7.0, 6.2), constrained_layout=True)
            # Re-run one representative final image for this ROI size at final phase of first group.
            if region_shape_name == "ring_of_circle":
                ring = build_touching_circle_ring(
                    requested_region_radius_lamD=float(roi_r),
                    orbit_radius_lamD=orbit_radius_lamD,
                    anchor_angle_rad=initial_angle_rad,
                    rotation_fraction=0.0,
                )
                case_centers = [(float(cx), float(cy)) for cx, cy in ring["centers_lamD"]]
                case_radius = float(ring["resolved_radius_lamD"])
            elif region_shape_name == "ring":
                case_centers = [(float(planet_center_lamD[0]), float(planet_center_lamD[1]))]
                case_radius = float(roi_r)
            else:
                case_centers = [(float(cx), float(cy)) for cx, cy in centers_lamD]
                case_radius = float(roi_r)
            ctr0 = case_centers[0] if len(case_centers) > 0 else (0.0, 0.0)
            phase_sim = CoronagraphSimulator(
                **{
                    **sim_local,
                    "e_final_phase_offset": 0.0,
                    "focal_local_phase_offset": float(phase_offsets[-1] if phase_offsets.size > 0 else 0.0),
                    **_local_phase_region_kwargs(
                        region_shape_name=region_shape_name,
                        region_width_or_radius_lamD=float(case_radius),
                        orbit_radius_lamD=orbit_radius_lamD,
                        centers_lamD=case_centers,
                    ),
                }
            )
            img_case = phase_sim.run()["final_psf_with_ghost"]
            im = ax_case.imshow(
                np.log10(img_case[sl_crop, sl_crop] + 1e-12),
                origin="lower",
                cmap="inferno",
                vmin=-8,
                vmax=0,
                extent=[-psf_crop_lamD, psf_crop_lamD, -psf_crop_lamD, psf_crop_lamD],
            )
            fig_case.colorbar(im, ax=ax_case, fraction=0.046, pad=0.04)
            if region_shape_name == "ring":
                ring_rmin_lamD, ring_rmax_lamD = annulus_radii_from_width(
                    mid_radius_lamD=orbit_radius_lamD,
                    width_lamD=float(case_radius),
                )
                ax_case.add_patch(plt.Circle((0.0, 0.0), float(ring_rmin_lamD), fill=False, edgecolor="lime", linewidth=1.4))
                ax_case.add_patch(plt.Circle((0.0, 0.0), float(ring_rmax_lamD), fill=False, edgecolor="cyan", linewidth=1.4))
            else:
                for j, (cx, cy) in enumerate(case_centers):
                    edge = "lime" if j == 0 else "cyan"
                    ax_case.add_patch(plt.Circle((cx, cy), float(case_radius), fill=False, edgecolor=edge, linewidth=1.4))
                    ax_case.text(cx, cy, str(j), color="white", fontsize=10, ha="center", va="center")
            ax_case.plot([ctr0[0]], [ctr0[1]], marker="+", color="white", markersize=9, linestyle="None")
            ax_case.set_title(
                "Final PSF with All Regions "
                f"(ROI r={roi_r:.2f} λ/D -> resolved {case_radius:.2f} λ/D, N={len(case_centers)})"
            )
            ax_case.set_xlabel("x [λ/D]")
            ax_case.set_ylabel("y [λ/D]")
            pdf.savefig(fig_case)
            plt.close(fig_case)

    if region_shape_name in {"ring", "ring_of_circle"}:
        if len(ring_requested_sizes) > 0:
            res_arr = np.asarray(ring_resolved_sizes, dtype=float)
            snr_arr = np.asarray(ring_snrs, dtype=float)
            ax.plot(
                res_arr,
                snr_arr,
                "-o",
                lw=1.8,
                ms=4.5,
                color="tab:orange",
                label="SNR vs ring width" if region_shape_name == "ring" else "SNR vs resolved ring radius",
            )
        ax.set_xlabel("ROI width [λ/D]" if region_shape_name == "ring" else "ROI size [λ/D]")
        ax.set_ylabel("SNR")
        ax.set_title(
            "Planet-region SNR vs ROI Width (ring)"
            if region_shape_name == "ring"
            else "Planet-region SNR vs ROI Size (ring_of_circle)"
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=1)
    else:
        ax.axvline(0.0, color="tab:red", lw=1.0, ls="--", alpha=0.8)
        ax.set_xlabel("theta relative to planet [rad] (planet = 0)")
        ax.set_ylabel("SNR")
        ax.set_title("Planet-region SNR vs theta (ROI Size Sweep)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    out_path = os.path.join(
        roi_dir,
        "roi_size_sweep_snr_vs_theta_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_sweep_tag}{ghost_suffix}.png",
    )
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    with open(out_table_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "roi_radius_lamD",
                "resolved_roi_radius_lamD",
                "n_circles",
                "theta_rel_rad",
                "active_center_x_lamD",
                "active_center_y_lamD",
                "planet_peak",
                "planet_std",
                "background_aperture_std",
                "snr",
            ],
        )
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Saved ROI-size sweep SNR-vs-theta plot: {out_path}")
    print(f"Saved ROI-size sweep final-PSF region overlays (PDF): {out_psf_pdf}")
    print(f"Saved ROI-size sweep incoherence maps with SNR (PDF): {out_incoh_pdf}")
    print(f"Saved ROI-size sweep theta/SNR table (CSV): {out_table_csv}")


def _run_ring_rotation_sweep(
    args: argparse.Namespace,
    sim_local: dict,
    incoherence_map_mode: str,
    sweep_output_dir: str,
    mask_output_tag: str,
    phase_cycles_tag: str,
    phase_sweep_mode_tag: str,
    single_region_tag: str,
    ghost_suffix: str,
    orbit_radius_lamD: float,
    initial_angle_rad: float,
    planet_center_lamD: tuple[float, float],
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    sweep_dir = sweep_output_dir
    os.makedirs(sweep_dir, exist_ok=True)

    rot_max = float(args.ring_rotation_sweep_max)
    rot_step = float(args.ring_rotation_sweep_step)
    rot_vals = np.arange(0.0, rot_max + 0.5 * rot_step, rot_step, dtype=float)
    rot_max_tag = f"{rot_max:.3f}".replace(".", "p")
    rot_step_tag = f"{rot_step:.3f}".replace(".", "p")
    rot_sweep_tag = f"_rotmax_{rot_max_tag}_rotstep_{rot_step_tag}"
    fixed_planet_eval_radius_lamD = SNR_APERTURE_RADIUS_LAMD
    snr_eps = 1e-12

    phase_cycles = float(args.phase_cycles)
    phase_offsets = np.linspace(0.0, 2.0 * np.pi * phase_cycles, int(args.phase_step), endpoint=True)

    base = CoronagraphSimulator(**sim_local).run()
    n_fft = int(base["n_fft"])
    samp = float(base["focal_sampling"])
    central_box_lamD = 12.0
    half12 = int(0.5 * central_box_lamD * samp)
    cc16 = n_fft // 2
    sl16 = slice(cc16 - half12, cc16 + half12)
    x16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, 2 * half12, endpoint=False)
    y16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, 2 * half12, endpoint=False)
    xx16, yy16 = np.meshgrid(x16, y16)
    psf_crop_lamD = 8.0
    half_crop = int(psf_crop_lamD * samp)
    sl_crop = slice(cc16 - half_crop, cc16 + half_crop)
    rr = np.sqrt(xx16**2 + yy16**2)

    out_psf_pdf = os.path.join(
        sweep_dir,
        "ring_rotation_sweep_final_psf_with_regions_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{rot_sweep_tag}{ghost_suffix}.pdf",
    )
    out_incoh_pdf = os.path.join(
        sweep_dir,
        "ring_rotation_sweep_incoherence_maps_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{rot_sweep_tag}{ghost_suffix}.pdf",
    )
    out_coh_pdf = os.path.join(
        sweep_dir,
        "ring_rotation_sweep_coherence_maps_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{rot_sweep_tag}{ghost_suffix}.pdf",
    )
    out_probe_pdf = os.path.join(
        sweep_dir,
        "ring_rotation_sweep_probe_pixel_fft_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{rot_sweep_tag}{ghost_suffix}.pdf",
    )
    out_table_csv = os.path.join(
        sweep_dir,
        "ring_rotation_sweep_table_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{rot_sweep_tag}{ghost_suffix}.csv",
    )
    out_probe_csv = os.path.join(
        sweep_dir,
        "ring_rotation_sweep_probe_pixel_locations_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{rot_sweep_tag}{ghost_suffix}.csv",
    )
    out_plot = os.path.join(
        sweep_dir,
        "ring_rotation_sweep_snr_vs_rotation_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{rot_sweep_tag}{ghost_suffix}.png",
    )

    rotations_used: list[float] = []
    snrs: list[float] = []
    peaks: list[float] = []
    medians: list[float] = []
    table_rows: list[dict[str, float]] = []
    probe_rows: list[dict[str, float | int]] = []
    incoh_panels: list[dict] = []

    with PdfPages(out_psf_pdf) as pdf_psf, PdfPages(out_incoh_pdf) as pdf_incoh, PdfPages(out_coh_pdf) as pdf_coh, PdfPages(out_probe_pdf) as pdf_probe:
        for rot_u in rot_vals:
            ring = build_touching_circle_ring(
                requested_region_radius_lamD=float(args.local_region_radius),
                orbit_radius_lamD=orbit_radius_lamD,
                anchor_angle_rad=initial_angle_rad,
                rotation_fraction=float(rot_u),
            )
            centers = list(ring["centers_lamD"])
            region_radius_lamD = float(ring["resolved_radius_lamD"])
            stack = np.zeros((phase_offsets.size, 2 * half12, 2 * half12), dtype=float)
            img_last = None
            for i, ph in enumerate(phase_offsets):
                if str(args.phase_sweep_mode).strip().lower() == "global":
                    phase_sim = CoronagraphSimulator(
                        **{
                            **sim_local,
                            "e_final_phase_offset": float(ph),
                            "focal_local_phase_offset": 0.0,
                            "focal_local_phase_centers_lamD": (),
                            "focal_local_phase_radius_lamD": 0.0,
                        }
                    )
                else:
                    phase_sim = CoronagraphSimulator(
                        **{
                            **sim_local,
                            "e_final_phase_offset": 0.0,
                            "focal_local_phase_offset": float(ph),
                            "focal_local_phase_centers_lamD": tuple((float(cx), float(cy)) for cx, cy in centers),
                            "focal_local_phase_radius_lamD": region_radius_lamD,
                        }
                    )
                img = phase_sim.run()["final_psf_with_ghost"]
                stack[i] = img[sl16, sl16]
                img_last = img

            phase_series = np.asarray(phase_offsets, dtype=float)
            if phase_series.size > 2 and np.isclose(phase_series[0], 0.0) and np.isclose(phase_series[-1], float(phase_series.max())):
                phase_series = phase_series[:-1]
                stack = stack[:-1]
            dphi = float(np.mean(np.diff(phase_series)))
            freq = np.fft.fftfreq(stack.shape[0], d=dphi)
            fft_cube = np.fft.fft(stack, axis=0)
            planet_mask = (
                (xx16 - float(planet_center_lamD[0])) ** 2
                + (yy16 - float(planet_center_lamD[1])) ** 2
                <= float(fixed_planet_eval_radius_lamD) ** 2
            )
            map_info = _coc_build_incoherence_maps(
                freq_bins=freq,
                fft_cube=fft_cube,
                central_stack_fft=stack,
                mode=incoherence_map_mode,
                planet_region_mask=planet_mask,
            )
            incoh = np.asarray(map_info["incoherence_map"], dtype=float)
            coherence_map = np.asarray(map_info["coherence_map"], dtype=float)
            inverse_coherence_map = np.asarray(map_info["inverse_coherence_map"], dtype=float)
            selected_target_freq = float(map_info["selected_target_freq"])
            if incoh is None:
                continue

            peak, med, snr = _planet_region_snr(
                incoh=incoh,
                xx=xx16,
                yy=yy16,
                planet_center_lamD=planet_center_lamD,
                orbit_radius_lamD=orbit_radius_lamD,
                eval_radius_lamD=fixed_planet_eval_radius_lamD,
                annulus_half_width_lamD=SNR_ANNULUS_HALF_WIDTH_LAMD,
                snr_eps=snr_eps,
            )
            planet_mask = (
                (xx16 - float(planet_center_lamD[0])) ** 2
                + (yy16 - float(planet_center_lamD[1])) ** 2
                <= fixed_planet_eval_radius_lamD ** 2
            )
            planet_std = float(np.std(np.asarray(incoh[planet_mask], dtype=float))) if np.any(planet_mask) else float("nan")

            rotations_used.append(float(rot_u))
            peaks.append(peak)
            medians.append(med)
            snrs.append(snr)
            table_rows.append(
                {
                    "rotation_fraction": float(rot_u),
                    "applied_rotation_rad": float(ring["applied_rotation_rad"]),
                    "edge_cut_rotation_rad": float(ring["edge_cut_rotation_rad"]),
                    "resolved_region_radius_lamD": region_radius_lamD,
                    "n_circles": int(ring["n_circles"]),
                    "planet_peak": peak,
                    "planet_std": planet_std,
                    "background_aperture_std": med,
                    "snr": snr,
                }
            )
            incoh_panels.append(
                {
                    "rotation_fraction": float(rot_u),
                    "applied_rotation_rad": float(ring["applied_rotation_rad"]),
                    "snr": float(snr),
                    "peak": float(peak),
                    "median": float(med),
                    "region_radius_lamD": float(region_radius_lamD),
                    "centers": [(float(cx), float(cy)) for cx, cy in centers],
                    "map": np.array(incoh, dtype=float),
                    "coherence_map": np.array(coherence_map, dtype=float),
                    "inverse_coherence_map": np.array(inverse_coherence_map, dtype=float),
                    "coherence_target_freq": selected_target_freq,
                    "incoherence_map_mode": incoherence_map_mode,
                }
            )

            base_angle = float(initial_angle_rad + ring["applied_rotation_rad"])
            probe_pixels = _build_ring_probe_pixels(
                n_fft=n_fft,
                focal_sampling=samp,
                orbit_radius_lamD=orbit_radius_lamD,
                base_angle_rad=base_angle,
                angular_step_rad=float(ring["center_angle_step_rad"]),
                n_probes=10,
            )
            for probe in probe_pixels:
                probe_rows.append(
                    {
                        "rotation_fraction": float(rot_u),
                        "applied_rotation_rad": float(ring["applied_rotation_rad"]),
                        "probe_index": int(probe["probe_index"]),
                        "step_index": int(probe["step_index"]),
                        "theta_rad": float(probe["theta_rad"]),
                        "x_lamD": float(probe["x_lamD"]),
                        "y_lamD": float(probe["y_lamD"]),
                        "x_idx": int(probe["x_idx"]),
                        "y_idx": int(probe["y_idx"]),
                    }
                )

            if img_last is not None:
                fig_case, ax_case = plt.subplots(1, 1, figsize=(7.0, 6.2), constrained_layout=True)
                im = ax_case.imshow(
                    np.log10(img_last[sl_crop, sl_crop] + 1e-12),
                    origin="lower",
                    cmap="inferno",
                    vmin=-8,
                    vmax=0,
                    extent=[-psf_crop_lamD, psf_crop_lamD, -psf_crop_lamD, psf_crop_lamD],
                )
                fig_case.colorbar(im, ax=ax_case, fraction=0.046, pad=0.04)
                for j, (cx, cy) in enumerate(centers):
                    edge = "lime" if j == 0 else "cyan"
                    ax_case.add_patch(plt.Circle((cx, cy), region_radius_lamD, fill=False, edgecolor=edge, linewidth=1.4))
                    ax_case.text(cx, cy, str(j), color="white", fontsize=10, ha="center", va="center")
                ax_case.plot([planet_center_lamD[0]], [planet_center_lamD[1]], marker="+", color="white", markersize=9, linestyle="None")
                ax_case.set_title(f"Final PSF with Ring-of-Circle Regions (rotation={rot_u:.2f})")
                ax_case.set_xlabel("x [λ/D]")
                ax_case.set_ylabel("y [λ/D]")
                pdf_psf.savefig(fig_case)
                plt.close(fig_case)
                _save_ring_rotation_probe_fft_page(
                    pdf=pdf_probe,
                    img_last=img_last,
                    stack=stack,
                    phase_series=phase_series,
                    probes=probe_pixels,
                    rotation_fraction=float(rot_u),
                    applied_rotation_rad=float(ring["applied_rotation_rad"]),
                    region_centers=[(float(cx), float(cy)) for cx, cy in centers],
                    region_radius_lamD=region_radius_lamD,
                    planet_center_lamD=planet_center_lamD,
                    sl_crop=sl_crop,
                    psf_crop_lamD=psf_crop_lamD,
                    n_fft=n_fft,
                    focal_sampling=samp,
                )

                fig_coh, axes_coh = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
                ax_coh, ax_inv = axes_coh
                coh_extent = [-0.5 * central_box_lamD, 0.5 * central_box_lamD, -0.5 * central_box_lamD, 0.5 * central_box_lamD]
                inverse_coherence_log_map = np.log10(np.maximum(inverse_coherence_map, 1e-20))
                inv_vals = inverse_coherence_log_map[np.isfinite(inverse_coherence_log_map)]
                if inv_vals.size > 0:
                    inv_vmin = float(np.nanpercentile(inv_vals, 5.0))
                    inv_vmax = float(np.nanpercentile(inv_vals, 95.0))
                    if (not np.isfinite(inv_vmin)) or (not np.isfinite(inv_vmax)) or np.isclose(inv_vmin, inv_vmax):
                        inv_vmin = float(np.nanmin(inv_vals))
                        inv_vmax = float(np.nanmax(inv_vals))
                else:
                    inv_vmin, inv_vmax = 0.0, 1.0
                im_coh = ax_coh.imshow(
                    coherence_map,
                    origin="lower",
                    cmap="magma",
                    extent=coh_extent,
                )
                im_inv = ax_inv.imshow(
                    inverse_coherence_log_map,
                    origin="lower",
                    cmap="cividis",
                    vmin=inv_vmin,
                    vmax=inv_vmax,
                    extent=coh_extent,
                )
                for ax_map in axes_coh:
                    for j, (cx, cy) in enumerate(centers):
                        edge = "lime" if j == 0 else "cyan"
                        ax_map.add_patch(plt.Circle((cx, cy), region_radius_lamD, fill=False, edgecolor=edge, linewidth=1.1))
                    ax_map.plot([planet_center_lamD[0]], [planet_center_lamD[1]], marker="+", color="white", markersize=8, linestyle="None")
                    ax_map.set_xlabel("x [λ/D]")
                    ax_map.set_ylabel("y [λ/D]")
                ax_coh.set_title(f"Coherence |FFT({selected_target_freq:.3f})| / |FFT(0)|\nu={rot_u:.2f}")
                ax_inv.set_title(
                    f"log10(1 / Coherence) |FFT(0)| / |FFT({selected_target_freq:.3f})|\n"
                    f"u={rot_u:.2f} | mode={incoherence_map_mode} (p5-p95)"
                )
                fig_coh.colorbar(im_coh, ax=ax_coh, fraction=0.046, pad=0.04)
                cbar_inv = fig_coh.colorbar(im_inv, ax=ax_inv, fraction=0.046, pad=0.04)
                cbar_inv.set_label("log10(1 / coherence)")
                pdf_coh.savefig(fig_coh)
                plt.close(fig_coh)

        if len(incoh_panels) > 0:
            all_vals = np.concatenate([panel["map"].ravel() for panel in incoh_panels])
            finite_vals = all_vals[np.isfinite(all_vals)]
            if finite_vals.size > 0:
                vmin = float(np.nanpercentile(finite_vals, 5.0))
                vmax = float(np.nanpercentile(finite_vals, 99.0))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
                    vmin = float(np.nanmin(finite_vals))
                    vmax = float(np.nanmax(finite_vals))
            else:
                vmin, vmax = 0.0, 1.0
            n_panels = len(incoh_panels)
            extent16 = [-0.5 * central_box_lamD, 0.5 * central_box_lamD, -0.5 * central_box_lamD, 0.5 * central_box_lamD]
            if n_panels == 11:
                # Prefer a wide poster-friendly layout: 6 panels on the first row, 5 on the second.
                panel_rows = [6, 5]
            elif n_panels > 6:
                first_row = min(6, int(np.ceil(0.5 * float(n_panels))))
                second_row = n_panels - first_row
                panel_rows = [first_row, second_row] if second_row > 0 else [first_row]
            else:
                panel_rows = [n_panels]
            nrows = len(panel_rows)
            ncols_ref = max(panel_rows)
            panel_size = 7.2
            fig_w = max(24.0, panel_size * float(ncols_ref) + 4.2)
            fig_h = max(18.0, panel_size * float(nrows) + 9.5)
            fig_p = plt.figure(figsize=(fig_w, fig_h))
            fig_p.patch.set_facecolor("#07111f")
            ref_panel = min(incoh_panels, key=lambda panel: abs(float(panel["rotation_fraction"])))
            ref_bg = fig_p.add_axes([0.0, 0.0, 1.0, 1.0], zorder=0)
            ref_bg.set_facecolor("#07111f")
            ref_map = np.array(ref_panel["map"], dtype=float)
            ref_norm = ref_map - np.nanmin(ref_map)
            ref_scale = np.nanpercentile(ref_norm[np.isfinite(ref_norm)], 99.0) if np.any(np.isfinite(ref_norm)) else 0.0
            if np.isfinite(ref_scale) and ref_scale > 0.0:
                ref_norm = np.clip(ref_norm / ref_scale, 0.0, 1.0)
            else:
                ref_norm = np.zeros_like(ref_norm)
            ref_bg.imshow(
                ref_norm,
                origin="lower",
                cmap="magma",
                alpha=0.12,
                interpolation="bilinear",
                aspect="auto",
            )
            ref_bg.axis("off")
            gs = fig_p.add_gridspec(
                nrows + 1,
                ncols_ref,
                height_ratios=[1.0] * nrows + [1.05],
                left=0.07,
                right=0.93,
                top=0.78,
                bottom=0.08,
                hspace=0.42,
                wspace=0.24,
            )
            axes_flat: list[plt.Axes] = []
            for row_idx, row_count in enumerate(panel_rows):
                if row_count == ncols_ref:
                    row_axes = [fig_p.add_subplot(gs[row_idx, col_idx], zorder=2) for col_idx in range(row_count)]
                else:
                    start_col = (ncols_ref - row_count) // 2
                    row_axes = [
                        fig_p.add_subplot(gs[row_idx, start_col + col_idx], zorder=2)
                        for col_idx in range(row_count)
                    ]
                axes_flat.extend(row_axes)
            last_im = None
            for ax_i, panel in enumerate(incoh_panels):
                ax = axes_flat[ax_i]
                ax.set_facecolor("#020814")
                last_im = ax.imshow(
                    panel["map"],
                    origin="lower",
                    cmap="inferno",
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent16,
                    aspect="equal",
                )
                for j, (cx, cy) in enumerate(panel["centers"]):
                    edge = "#7CFFB2" if j == 0 else "#79C7FF"
                    ax.add_patch(
                        plt.Circle(
                            (cx, cy),
                            panel["region_radius_lamD"],
                            fill=False,
                            edgecolor=edge,
                            linewidth=2.0,
                        )
                    )
                ax.add_patch(
                    plt.Circle(
                        (float(planet_center_lamD[0]), float(planet_center_lamD[1])),
                        float(fixed_planet_eval_radius_lamD),
                        fill=False,
                        edgecolor="#FFFFFF",
                        linewidth=1.2,
                    )
                )
                ax.add_patch(
                    plt.Circle(
                        (0.0, 0.0),
                        float(max(orbit_radius_lamD - SNR_ANNULUS_HALF_WIDTH_LAMD, 0.0)),
                        fill=False,
                        edgecolor="#FFB347",
                        linewidth=1.8,
                        linestyle="--",
                    )
                )
                ax.add_patch(
                    plt.Circle(
                        (0.0, 0.0),
                        float(orbit_radius_lamD + SNR_ANNULUS_HALF_WIDTH_LAMD),
                        fill=False,
                        edgecolor="#FFB347",
                        linewidth=1.8,
                        linestyle="--",
                    )
                )
                ax.plot([planet_center_lamD[0]], [planet_center_lamD[1]], marker="o", color="#FF5F5F", markersize=8.0)
                ax.set_title(
                    f"u={panel['rotation_fraction']:.2f} | SNR={panel['snr']:.2e} | {panel['incoherence_map_mode']}",
                    fontsize=34,
                    color="white",
                    pad=14.0,
                )
                ax.text(
                    0.02,
                    0.98,
                    (
                        f"rot={panel['applied_rotation_rad']:.3f} rad\n"
                        f"peak={panel['peak']:.2e}\n"
                        f"med={panel['median']:.2e}"
                    ),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=22,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="#000000", alpha=0.55, edgecolor="none"),
                )
                ax.tick_params(colors="#C9D7F0", labelsize=22)
                for spine in ax.spines.values():
                    spine.set_color("#3A4A66")
                row_start = 0
                row_idx = 0
                for row_len in panel_rows:
                    if ax_i < row_start + row_len:
                        break
                    row_start += row_len
                    row_idx += 1
                col_idx = ax_i - row_start
                row_len = panel_rows[row_idx]
                if col_idx == 0:
                    ax.set_ylabel("y [λ/D]", color="#C9D7F0", fontsize=27)
                else:
                    ax.set_ylabel("")
                if row_idx == nrows - 1:
                    ax.set_xlabel("x [λ/D]", color="#C9D7F0", fontsize=27)
                else:
                    ax.set_xlabel("")
            ax_snr = fig_p.add_subplot(gs[nrows, :], zorder=2)
            ax_snr.set_facecolor((2 / 255, 8 / 255, 20 / 255, 0.88))
            rot_arr = np.asarray(rotations_used, dtype=float)
            snr_arr = np.asarray(snrs, dtype=float)
            ax_snr.plot(rot_arr, snr_arr, "-o", lw=4.4, ms=10.0, color="#7CFFB2", markerfacecolor="#B8FFE5", markeredgecolor="#7CFFB2")
            ax_snr.axvline(0.0, color="#FF5F5F", lw=2.0, ls="--", alpha=0.9)
            ax_snr.set_xlabel("ring rotation fraction", color="#C9D7F0", fontsize=42)
            ax_snr.set_ylabel("planet-region SNR", color="#C9D7F0", fontsize=42)
            ax_snr.set_title("SNR vs Ring Rotation", fontsize=36, color="white", pad=8.0)
            ax_snr.grid(alpha=0.22, color="#93A4C3")
            ax_snr.tick_params(colors="#C9D7F0", labelsize=36)
            for spine in ax_snr.spines.values():
                spine.set_color("#3A4A66")
            fig_p.suptitle(
                f"Ring Rotation Sweep Incoherence Maps ({incoherence_map_mode})",
                fontsize=88,
                color="white",
                fontweight="bold",
                y=0.975,
            )
            pdf_incoh.savefig(fig_p, facecolor=fig_p.get_facecolor())
            plt.close(fig_p)

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 5.4), constrained_layout=True)
    ax.plot(np.asarray(rotations_used, dtype=float), np.asarray(snrs, dtype=float), "-o", lw=1.6, ms=4.0, color="tab:blue")
    ax.set_xlabel("ring rotation fraction")
    ax.set_ylabel("Planet-region SNR")
    ax.set_title(f"Planet-region SNR vs Ring Rotation Fraction ({incoherence_map_mode})")
    ax.grid(alpha=0.3)
    fig.savefig(out_plot, dpi=170, bbox_inches="tight")
    plt.close(fig)

    with open(out_table_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rotation_fraction",
                "applied_rotation_rad",
                "edge_cut_rotation_rad",
                "resolved_region_radius_lamD",
                "n_circles",
                "planet_peak",
                "planet_std",
                "background_aperture_std",
                "snr",
            ],
        )
        writer.writeheader()
        writer.writerows(table_rows)

    with open(out_probe_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rotation_fraction",
                "applied_rotation_rad",
                "probe_index",
                "step_index",
                "theta_rad",
                "x_lamD",
                "y_lamD",
                "x_idx",
                "y_idx",
            ],
        )
        writer.writeheader()
        writer.writerows(probe_rows)

    print(f"Saved ring rotation sweep SNR-vs-rotation plot: {out_plot}")
    print(f"Saved ring rotation sweep final-PSF overlays (PDF): {out_psf_pdf}")
    print(f"Saved ring rotation sweep incoherence maps (PDF): {out_incoh_pdf}")
    print(f"Saved ring rotation sweep coherence maps (PDF): {out_coh_pdf}")
    print(f"Saved ring rotation sweep table (CSV): {out_table_csv}")
    print(f"Saved ring rotation sweep probe-pixel FFT pages (PDF): {out_probe_pdf}")
    print(f"Saved ring rotation sweep probe-pixel locations (CSV): {out_probe_csv}")


def run_coc_planet_phase(
    args: argparse.Namespace,
    sim_kwargs: dict,
    mask_output_tag: str,
    phase_cycles_tag: str,
    phase_sweep_mode_tag: str,
    single_region_tag: str,
    ghost_suffix: str,
    print_progress_bar,
    float_filename_token,
) -> None:
    effective_args = argparse.Namespace(**vars(args))
    region_shape_name = normalize_region_shape(args.region_shape)
    if int(args.fov_count) < 1:
        raise ValueError("--fov-count must be >= 1.")
    if int(args.fov_centers_count) < 1:
        raise ValueError("--fov-centers-count must be >= 1.")
    if region_shape_name not in {"ring", "ring_of_circle"} and int(args.fov_count) > int(args.fov_centers_count):
        raise ValueError("--fov-count must be <= --fov-centers-count.")
    if float(args.local_region_radius) <= 0.0:
        raise ValueError("--local-region-radius must be > 0.")
    if int(args.phase_step) < 2:
        raise ValueError("--phase-step must be >= 2.")
    if float(args.planet_flux_ratio_local) < 0.0:
        raise ValueError("--planet-flux-ratio-local must be >= 0.")
    if sum(
        int(bool(getattr(args, name, False)))
        for name in ("roi_size_sweep", "planet_position_roi_size_sweep", "planet_diagonal_roi_size_sweep")
    ) > 1:
        raise ValueError("Enable at most one of --roi-size-sweep, --planet-position-roi-size-sweep, --planet-diagonal-roi-size-sweep.")
    if bool(getattr(args, "roi_size_sweep", False)):
        if float(args.roi_size_min) <= 0.0:
            raise ValueError("--roi-size-min must be > 0.")
        if float(args.roi_size_max) < float(args.roi_size_min):
            raise ValueError("--roi-size-max must be >= roi-size-min.")
        if float(args.roi_size_step) < 0.0 or (
            float(args.roi_size_step) == 0.0 and not np.isclose(float(args.roi_size_min), float(args.roi_size_max))
        ):
            raise ValueError("--roi-size-step must be > 0 unless --roi-size-min == --roi-size-max.")
    if bool(getattr(args, "planet_position_roi_size_sweep", False)):
        if float(args.roi_size_min) <= 0.0:
            raise ValueError("--roi-size-min must be > 0.")
        if float(args.roi_size_max) < float(args.roi_size_min):
            raise ValueError("--roi-size-max must be >= --roi-size-min.")
        if float(args.roi_size_step) < 0.0 or (
            float(args.roi_size_step) == 0.0 and not np.isclose(float(args.roi_size_min), float(args.roi_size_max))
        ):
            raise ValueError("--roi-size-step must be > 0 unless --roi-size-min == --roi-size-max.")
        if float(args.planet_position_radius_min) <= 0.0:
            raise ValueError("--planet-position-radius-min must be > 0.")
        if float(args.planet_position_radius_max) < float(args.planet_position_radius_min):
            raise ValueError("--planet-position-radius-max must be >= --planet-position-radius-min.")
        if float(args.planet_position_theta_max_deg) < float(args.planet_position_theta_min_deg):
            raise ValueError("--planet-position-theta-max-deg must be >= --planet-position-theta-min-deg.")
        if float(args.planet_position_radius_step) < 0.0 or (
            float(args.planet_position_radius_step) == 0.0
            and not np.isclose(float(args.planet_position_radius_min), float(args.planet_position_radius_max))
        ):
            raise ValueError("--planet-position-radius-step must be > 0 unless --planet-position-radius-min == --planet-position-radius-max.")
        if float(args.planet_position_theta_step_deg) < 0.0 or (
            float(args.planet_position_theta_step_deg) == 0.0
            and not np.isclose(float(args.planet_position_theta_min_deg), float(args.planet_position_theta_max_deg))
        ):
            raise ValueError("--planet-position-theta-step-deg must be > 0 unless --planet-position-theta-min-deg == --planet-position-theta-max-deg.")
    if bool(getattr(args, "planet_diagonal_roi_size_sweep", False)):
        if float(args.roi_size_min) <= 0.0:
            raise ValueError("--roi-size-min must be > 0.")
        if float(args.roi_size_max) < float(args.roi_size_min):
            raise ValueError("--roi-size-max must be >= --roi-size-min.")
        if float(args.roi_size_step) < 0.0 or (
            float(args.roi_size_step) == 0.0 and not np.isclose(float(args.roi_size_min), float(args.roi_size_max))
        ):
            raise ValueError("--roi-size-step must be > 0 unless --roi-size-min == --roi-size-max.")
        if float(args.planet_diagonal_t_max) < float(args.planet_diagonal_t_min):
            raise ValueError("--planet-diagonal-t-max must be >= --planet-diagonal-t-min.")
        if float(args.planet_diagonal_t_step) < 0.0 or (
            float(args.planet_diagonal_t_step) == 0.0
            and not np.isclose(float(args.planet_diagonal_t_min), float(args.planet_diagonal_t_max))
        ):
            raise ValueError("--planet-diagonal-t-step must be > 0 unless --planet-diagonal-t-min == --planet-diagonal-t-max.")
    if bool(getattr(args, "ring_rotation_sweep", False)):
        if region_shape_name != "ring_of_circle":
            raise ValueError("--ring-rotation-sweep requires --region-shape ring_of_circle.")
        if float(args.ring_rotation_sweep_max) < 0.0 or float(args.ring_rotation_sweep_max) > 1.0:
            raise ValueError("--ring-rotation-sweep-max must be within [0, 1].")
        if float(args.ring_rotation_sweep_step) <= 0.0:
            raise ValueError("--ring-rotation-sweep-step must be > 0.")

    local_kwargs = dict(sim_kwargs)
    coc_secondary = (
        float(args.secondary_ratio_local)
        if args.secondary_ratio_local is not None
        else float(local_kwargs.get("secondary_diameter_ratio", 0.0))
    )
    if coc_secondary <= 0.0:
        coc_secondary = 0.25
    local_kwargs["secondary_diameter_ratio"] = float(coc_secondary)

    phase_screen_folder_tag = _phase_screen_folder_tag(args, sim_kwargs)
    fixed_center = (float(args.planet_offset_x_local), float(args.planet_offset_y_local))
    ring_radius_lamD = float(np.hypot(*fixed_center))
    initial_angle_rad = float(np.arctan2(fixed_center[1], fixed_center[0]))
    if region_shape_name == "ring_of_circle":
        ring = build_touching_circle_ring(
            requested_region_radius_lamD=float(args.local_region_radius),
            orbit_radius_lamD=ring_radius_lamD,
            anchor_angle_rad=initial_angle_rad,
            rotation_fraction=float(getattr(args, "ring_rotation_fraction", 0.0)),
        )
        centers = list(ring["centers_lamD"])
        effective_args.local_region_radius = float(ring["resolved_radius_lamD"])
        effective_args.fov_count = int(ring["n_circles"])
        effective_args.fov_centers_count = int(ring["n_circles"])
        effective_args.ring_rotation_fraction = float(ring["rotation_fraction"])
        print(
            "Resolved ring region shape: "
            f"requested radius={float(args.local_region_radius):.4f} λ/D, "
            f"snapped radius={float(effective_args.local_region_radius):.4f} λ/D, "
            f"circles={int(ring['n_circles'])}, "
            f"orbit radius={ring_radius_lamD:.4f} λ/D, "
            f"rotation fraction={float(ring['rotation_fraction']):.4f}, "
            f"rotation={float(ring['applied_rotation_rad']):.6f} rad"
        )
    elif region_shape_name == "ring":
        ring_rmin_lamD, ring_rmax_lamD = annulus_radii_from_width(
            mid_radius_lamD=ring_radius_lamD,
            width_lamD=float(args.local_region_radius),
        )
        centers = [fixed_center]
        effective_args.fov_count = 1
        effective_args.fov_centers_count = 1
        print(
            "Resolved ring region shape: "
            f"width={float(args.local_region_radius):.4f} λ/D, "
            f"rmin={float(ring_rmin_lamD):.4f} λ/D, "
            f"rmax={float(ring_rmax_lamD):.4f} λ/D, "
            f"mid-radius={ring_radius_lamD:.4f} λ/D"
        )
    else:
        theta_rel = _theta_back_and_forth(max(1, int(args.fov_centers_count)), max_abs=np.pi)
        centers = [
            (
                float(ring_radius_lamD * np.cos(initial_angle_rad + th_rel)),
                float(ring_radius_lamD * np.sin(initial_angle_rad + th_rel)),
            )
            for th_rel in theta_rel
        ]
    fov_count = int(effective_args.fov_count)
    fov_centers_count = int(effective_args.fov_centers_count)

    d2 = [(cx - fixed_center[0]) ** 2 + (cy - fixed_center[1]) ** 2 for cx, cy in centers]
    planet_region_idx = int(np.argmin(d2))
    planet_center = centers[planet_region_idx]
    coc_planet_ratio_dir = (
        f"coc_planet_ratio_{float_filename_token(args.planet_flux_ratio_local, precision=6)}"
        f"_planet_x_{float_filename_token(planet_center[0], precision=3)}"
        f"_y_{float_filename_token(planet_center[1], precision=3)}"
        f"_pov_r_{float_filename_token(effective_args.local_region_radius, precision=3)}"
        f"{phase_screen_folder_tag}"
    )
    os.makedirs(coc_planet_ratio_dir, exist_ok=True)
    coc_planet_ratio_dir_no_pov = (
        f"coc_planet_ratio_{float_filename_token(args.planet_flux_ratio_local, precision=6)}"
        f"_planet_x_{float_filename_token(planet_center[0], precision=3)}"
        f"_y_{float_filename_token(planet_center[1], precision=3)}"
        f"{phase_screen_folder_tag}"
    )

    coc_phase_cycles = float(args.phase_cycles)
    n_fov_groups = int(np.ceil(float(fov_centers_count) / float(fov_count)))
    phase_steps_per_fov = int(args.phase_step)
    phase_steps_total = max(2, phase_steps_per_fov * max(n_fov_groups, 1))
    phase_offsets = np.linspace(
        0.0,
        2.0 * np.pi * coc_phase_cycles * float(n_fov_groups),
        phase_steps_total,
        endpoint=True,
    )
    print(
        "Phase sampling: "
        f"{phase_steps_per_fov} steps/FOV-block, "
        f"{n_fov_groups} blocks, "
        f"{phase_steps_total} total samples."
    )

    sim_local = dict(local_kwargs)
    sim_local["companion_flux_ratio"] = float(args.planet_flux_ratio_local)
    sim_local["companion_offset_lamD"] = (float(planet_center[0]), float(planet_center[1]))
    sim_local["e_final_phase_offset"] = 0.0
    print(f"Using phase mask for coc-planet-phase: {sim_local['phase_mask'].__class__.__name__}")
    single_fov_orbit_radius = float(ring_radius_lamD)

    if bool(getattr(args, "roi_size_sweep", False)):
        sweep_folder = os.path.join(
            coc_planet_ratio_dir_no_pov,
            "roi_size_sweep",
        )
        _run_roi_size_sweep_snr_vs_theta(
            args=effective_args,
            sim_local=sim_local,
            incoherence_map_mode=str(getattr(effective_args, "incoherence_map_mode", "fft_band")),
            sweep_output_dir=sweep_folder,
            mask_output_tag=mask_output_tag,
            phase_cycles_tag=phase_cycles_tag,
            phase_sweep_mode_tag=phase_sweep_mode_tag,
            single_region_tag=single_region_tag,
            ghost_suffix=ghost_suffix,
            orbit_radius_lamD=single_fov_orbit_radius,
            initial_angle_rad=initial_angle_rad,
            centers_lamD=centers,
            planet_center_lamD=planet_center,
        )
        print("ROI size sweep enabled: skipped standard outputs.")
        return
    if bool(getattr(args, "planet_position_roi_size_sweep", False)):
        polar_root = (
            f"coc_planet_ratio_{float_filename_token(args.planet_flux_ratio_local, precision=6)}"
            "_planet_position_polar"
            f"_r_{float_filename_token(float(args.planet_position_radius_min), precision=3)}"
            f"_{float_filename_token(float(args.planet_position_radius_max), precision=3)}"
            f"_{float_filename_token(float(args.planet_position_radius_step), precision=3)}"
            f"_theta_{float_filename_token(float(args.planet_position_theta_min_deg), precision=3)}"
            f"_{float_filename_token(float(args.planet_position_theta_max_deg), precision=3)}"
            f"_{float_filename_token(float(args.planet_position_theta_step_deg), precision=3)}"
            f"{phase_screen_folder_tag}"
        )
        sweep_folder = os.path.join(polar_root, "planet_position_roi_size_sweep")
        _run_planet_position_roi_size_sweep(
            args=effective_args,
            sim_local=sim_local,
            incoherence_map_mode=str(getattr(effective_args, "incoherence_map_mode", "fft_band")),
            sweep_output_dir=sweep_folder,
            mask_output_tag=mask_output_tag,
            phase_cycles_tag=phase_cycles_tag,
            phase_sweep_mode_tag=phase_sweep_mode_tag,
            single_region_tag=single_region_tag,
            ghost_suffix=ghost_suffix,
        )
        print("Planet-position/ROI-size 2D sweep enabled: skipped standard outputs.")
        return
    if bool(getattr(args, "planet_diagonal_roi_size_sweep", False)):
        sweep_folder = os.path.join(
            coc_planet_ratio_dir_no_pov,
            "planet_diagonal_roi_size_sweep",
        )
        _run_planet_diagonal_roi_size_sweep(
            args=effective_args,
            sim_local=sim_local,
            incoherence_map_mode=str(getattr(effective_args, "incoherence_map_mode", "fft_band")),
            sweep_output_dir=sweep_folder,
            mask_output_tag=mask_output_tag,
            phase_cycles_tag=phase_cycles_tag,
            phase_sweep_mode_tag=phase_sweep_mode_tag,
            single_region_tag=single_region_tag,
            ghost_suffix=ghost_suffix,
        )
        print("Planet-diagonal/ROI-size sweep enabled: skipped standard outputs.")
        return
    if bool(getattr(args, "ring_rotation_sweep", False)):
        sweep_folder = os.path.join(
            coc_planet_ratio_dir_no_pov,
            "ring_rotation_sweep",
        )
        _run_ring_rotation_sweep(
            args=effective_args,
            sim_local=sim_local,
            incoherence_map_mode=str(getattr(effective_args, "incoherence_map_mode", "fft_band")),
            sweep_output_dir=sweep_folder,
            mask_output_tag=mask_output_tag,
            phase_cycles_tag=phase_cycles_tag,
            phase_sweep_mode_tag=phase_sweep_mode_tag,
            single_region_tag=single_region_tag,
            ghost_suffix=ghost_suffix,
            orbit_radius_lamD=single_fov_orbit_radius,
            initial_angle_rad=initial_angle_rad,
            planet_center_lamD=fixed_center,
        )
        print("Ring rotation sweep enabled: skipped standard outputs.")
        return

    base = CoronagraphSimulator(**sim_local).run()
    n_fft = int(base["n_fft"])
    samp = float(base["focal_sampling"])
    pix = np.arange(n_fft, dtype=float)
    c = (n_fft - 1.0) / 2.0
    x_lamD = (pix - c) / samp
    y_lamD = (pix - c) / samp
    xx, yy = np.meshgrid(x_lamD, y_lamD)
    if region_shape_name == "ring":
        ring_rmin_lamD, ring_rmax_lamD = annulus_radii_from_width(
            mid_radius_lamD=ring_radius_lamD,
            width_lamD=float(effective_args.local_region_radius),
        )
        rr = np.sqrt(xx**2 + yy**2)
        roi_masks = [(rr >= ring_rmin_lamD) & (rr <= ring_rmax_lamD)]
    else:
        roi_masks = [
            (xx - xc) ** 2 + (yy - yc) ** 2 <= float(effective_args.local_region_radius) ** 2
            for xc, yc in centers
        ]

    centers_tuple = tuple((float(cx), float(cy)) for cx, cy in centers)
    group_cycle_span = 2.0 * np.pi * max(float(coc_phase_cycles), 1e-12)

    integrated_intensity = np.zeros((len(centers), phase_offsets.size), dtype=float)
    center_pixels_yx = np.zeros((len(centers), 2), dtype=int)
    center_pixel_intensity = np.zeros((len(centers), phase_offsets.size), dtype=float)
    phase_psf_cube = np.zeros((phase_offsets.size, n_fft, n_fft), dtype=np.float32)
    phase_active_region_idx = np.full(phase_offsets.size, -1, dtype=np.int16)
    central_box_lamD = 12.0
    half12 = int(0.5 * central_box_lamD * samp)
    cc16 = n_fft // 2
    sl16 = slice(cc16 - half12, cc16 + half12)
    central_phase_stack = np.zeros((phase_offsets.size, 2 * half12, 2 * half12), dtype=float)
    for j, (cx, cy) in enumerate(centers):
        x_idx = int(np.clip(np.round(c + cx * samp), 0, n_fft - 1))
        y_idx = int(np.clip(np.round(c + cy * samp), 0, n_fft - 1))
        center_pixels_yx[j] = np.array([y_idx, x_idx], dtype=int)

    start_time = time.perf_counter()
    for i, ph in enumerate(phase_offsets):
        if str(effective_args.phase_sweep_mode).strip().lower() == "global":
            phase_active_region_idx[i] = -1
            phase_sim = CoronagraphSimulator(
                **{
                    **sim_local,
                    "e_final_phase_offset": float(ph),
                    "focal_local_phase_offset": 0.0,
                    "focal_local_phase_centers_lamD": (),
                    "focal_local_phase_radius_lamD": 0.0,
                }
            )
            r = phase_sim.run()
            img = r["final_psf_with_ghost"]
            central_phase_stack[i] = img[sl16, sl16]
            for j, m in enumerate(roi_masks):
                integrated_intensity[j, i] = float(np.max(img[m])) if np.any(m) else 0.0
                y_idx, x_idx = int(center_pixels_yx[j, 0]), int(center_pixels_yx[j, 1])
                center_pixel_intensity[j, i] = float(img[y_idx, x_idx])
        else:
            group_idx = int(np.floor(float(ph) / group_cycle_span))
            group_idx = int(np.clip(group_idx, 0, n_fov_groups - 1))
            start_idx = group_idx * fov_count
            end_idx = min(start_idx + fov_count, len(centers))
            active_centers = centers[start_idx:end_idx]
            local_phase = float(ph - group_idx * group_cycle_span)
            phase_active_region_idx[i] = int(start_idx) if len(active_centers) > 0 else -1
            phase_sim = CoronagraphSimulator(
                **{
                    **sim_local,
                    "e_final_phase_offset": 0.0,
                    "focal_local_phase_offset": local_phase,
                    **_local_phase_region_kwargs(
                        region_shape_name=region_shape_name,
                        region_width_or_radius_lamD=float(effective_args.local_region_radius),
                        orbit_radius_lamD=ring_radius_lamD,
                        centers_lamD=active_centers,
                    ),
                }
            )
            r = phase_sim.run()
            img = r["final_psf_with_ghost"]
            central_phase_stack[i] = img[sl16, sl16]
            for j, m in enumerate(roi_masks):
                integrated_intensity[j, i] = float(np.max(img[m])) if np.any(m) else 0.0
                y_idx, x_idx = int(center_pixels_yx[j, 0]), int(center_pixels_yx[j, 1])
                center_pixel_intensity[j, i] = float(img[y_idx, x_idx])
        phase_psf_cube[i] = img.astype(np.float32)
        print_progress_bar(
            completed=i + 1,
            total=phase_offsets.size,
            start_time=start_time,
            prefix="coc-planet-phase",
        )

    # Intentionally skip CoC FITS cube export to keep CoC outputs minimal.

    try:
        from matplotlib import cm

        gif_name = (
            f"{coc_planet_ratio_dir}/coc_planet_final_psf_16lamD_local_{float(effective_args.local_region_radius):.3f}_"
            f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.gif"
        )
        cube16 = phase_psf_cube[:, sl16, sl16].astype(np.float64)
        log_cube16 = np.log10(np.maximum(cube16, 1e-12))
        vmin = float(np.nanpercentile(log_cube16, 1.0))
        vmax = float(np.nanpercentile(log_cube16, 99.5))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            vmin, vmax = -8.0, 0.0
        norm = np.clip((log_cube16 - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
        rgba = cm.get_cmap("inferno")(norm)
        rgb8 = (255.0 * rgba[..., :3]).astype(np.uint8)
        rgb8 = np.flip(rgb8, axis=1)
        _write_rgb_gif(rgb8, gif_name, duration_ms=500)
        print(f"Saved central 16x16 λ/D GIF: {gif_name}")
    except Exception as exc:
        print(f"Could not save central 16x16 λ/D GIF: {exc}")

    if region_shape_name == "ring_of_circle":
        try:
            ring_shift_gif = (
                f"{coc_planet_ratio_dir}/coc_planet_ring_of_circle_shift_"
                f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.gif"
            )
            _save_ring_of_circle_rotation_gif(
                gif_path=ring_shift_gif,
                requested_region_radius_lamD=float(args.local_region_radius),
                orbit_radius_lamD=ring_radius_lamD,
                anchor_angle_rad=initial_angle_rad,
                fixed_center_lamD=fixed_center,
                resolved_region_radius_lamD=float(effective_args.local_region_radius),
                n_circles=int(fov_count),
            )
            print(f"Saved ring_of_circle rotation GIF: {ring_shift_gif}")
        except Exception as exc:
            print(f"Could not save ring_of_circle rotation GIF: {exc}")

    plot_info = plot_coc_planet_phase_outputs(
        args=effective_args,
        base=base,
        centers=centers,
        planet_region_idx=planet_region_idx,
        coc_planet_ratio_dir=coc_planet_ratio_dir,
        mask_output_tag=mask_output_tag,
        phase_cycles_tag=phase_cycles_tag,
        phase_sweep_mode_tag=phase_sweep_mode_tag,
        single_region_tag=single_region_tag,
        ghost_suffix=ghost_suffix,
        phase_offsets=phase_offsets,
        roi_masks=roi_masks,
        integrated_intensity=integrated_intensity,
        central_phase_stack=central_phase_stack,
    )
    print(f"Saved circles-of-circles overlay plot: {plot_info['out_overlay']}")
    print(f"Saved combined FFT+overlay plot: {plot_info['out_fft_overlay']}")
    print(f"Saved coherence/incoherence map plot: {plot_info['out_maps']}")
    if plot_info.get("out_selection_spectrum"):
        print(f"Saved frequency-selection FFT spectrum plot: {plot_info['out_selection_spectrum']}")
    if plot_info.get("out_maps_per_fov_pdf"):
        print(f"Saved per-active-FOV incoherence-map PDF: {plot_info['out_maps_per_fov_pdf']}")
    print(f"Incoherence map mode: {plot_info['incoherence_map_mode']}")
    print(
        "Incoherence-map planet SNR (planet-aperture sum / annulus-aperture std): "
        f"{plot_info['incoherence_planet_snr']:.6e}"
    )
    print(
        "  sum(planet region) = "
        f"{plot_info['incoherence_planet_region_peak']:.6e}"
    )
    print(
        "  std(equal-area annulus apertures, r={:.3f} λ/D, width={:.3f} λ/D) = {:.6e}".format(
            float(plot_info["incoherence_annulus_radius_lamD"]),
            float(plot_info["incoherence_annulus_width_lamD"]),
            float(plot_info["incoherence_annulus_median"]),
        )
    )
    print(f"Planet region index: {planet_region_idx}")
    print(
        "Planet region center [λ/D]: "
        f"({planet_center[0]:+.3f}, {planet_center[1]:+.3f})"
    )
    print("Sampled center pixels (y, x) for CoC regions:")
    for j, yx in enumerate(center_pixels_yx):
        print(f"  region {j}: ({int(yx[0])}, {int(yx[1])})")
    print(
        "Simulation includes secondary obstruction="
        f"{local_kwargs['secondary_diameter_ratio']:.3f}, "
        f"spider width={local_kwargs['spider_width_pixels']:.3f}px, "
        f"spider angles={local_kwargs['spider_angles_deg']}"
    )
    print(f"Planet FFT peaks (filter-1 smooth-prominence) [cycles/rad]: {plot_info['f1_freqs']}")
    print(f"Planet FFT peaks (filter-2 high-pass) [cycles/rad]: {plot_info['f2_freqs']}")
    band_a_min, band_a_max = plot_info["band_a_bounds"]
    band_b_min, band_b_max = plot_info["band_b_bounds"]
    print(f"Central-field FFT incoherence band [cycles/rad]: [{band_a_min:.3f}, {band_a_max:.3f}]")
    print(f"Central-field FFT incoherence bins used [cycles/rad]: {plot_info['band_a_freqs']}")
    print(f"Central-field FFT coherence band [cycles/rad]: [{band_b_min:.3f}, {band_b_max:.3f}]")
    print(f"Central-field FFT coherence bins used [cycles/rad]: {plot_info['band_b_freqs']}")
    if plot_info["band_a_peak"] is not None:
        print(
            "Planet strongest FFT peak in band A [0.0, 0.025] cycles/rad: "
            f"f={plot_info['band_a_peak'][0]:.6f}, amp={plot_info['band_a_peak'][1]:.6e}"
        )
    if plot_info["band_b_peak"] is not None:
        print(
            "Planet strongest FFT peak in band B [0.120, 0.180] cycles/rad: "
            f"f={plot_info['band_b_peak'][0]:.6f}, amp={plot_info['band_b_peak'][1]:.6e}"
        )
