from __future__ import annotations

import argparse

import numpy as np

from .plotting import _cdi_build_incoherence_maps, _cdi_frequency_selection_spectrum
from .region_shapes import annulus_radii_from_width, build_touching_circle_ring
from .simulator import CoronagraphSimulator

SNR_APERTURE_RADIUS_LAMD = 1.0
SNR_ANNULUS_HALF_WIDTH_LAMD = 1.0
WHOLE_FOCAL_PLANE_PHASE_MODES = {"global", "focal_plane"}


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


def _is_whole_focal_plane_phase_mode(phase_sweep_mode: str) -> bool:
    return str(phase_sweep_mode).strip().lower() in WHOLE_FOCAL_PLANE_PHASE_MODES


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


def _lyot_reference_folder_tag(args: argparse.Namespace, sim_kwargs: dict) -> str:
    if not bool(sim_kwargs.get("perfect_coronagraph", False)):
        return ""
    percent = float(getattr(args, "lyot_reference_percent", 100.0))
    percent_token = f"{percent:.3f}".replace(".", "p")
    return f"_lyot_ref_{percent_token}"


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
    map_info = _cdi_build_incoherence_maps(
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
        if np.any(aperture_mask & planet_mask):
            continue
        if not np.all(annulus_mask[aperture_mask]):
            continue
        aperture_means.append(float(np.mean(incoh[aperture_mask])))
    background_mean = (
        float(np.mean(np.asarray(aperture_means, dtype=float)))
        if len(aperture_means) > 0
        else float("nan")
    )
    background_std = (
        float(np.std(np.asarray(aperture_means, dtype=float)))
        if len(aperture_means) > 0
        else float("nan")
    )

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


def _summarize_roi_snr_trend(
    rows: list[dict[str, float | int]],
    snr_key: str = "snr",
    roi_key: str = "resolved_roi_size_lamD",
    atol: float = 1e-12,
) -> str:
    ordered_pairs: list[tuple[float, float]] = []
    for row in rows:
        roi_value = row.get(roi_key)
        snr_value = row.get(snr_key)
        if roi_value is None or snr_value is None:
            continue
        roi = float(roi_value)
        snr = float(snr_value)
        if not np.isfinite(roi) or not np.isfinite(snr):
            continue
        ordered_pairs.append((roi, snr))
    if len(ordered_pairs) < 2:
        return "insufficient_samples"

    ordered_pairs.sort(key=lambda pair: pair[0])
    diffs = np.diff(np.asarray([pair[1] for pair in ordered_pairs], dtype=float))
    if diffs.size == 0:
        return "insufficient_samples"
    if np.all(np.abs(diffs) <= float(atol)):
        return "flat"
    if np.all(diffs >= -float(atol)):
        return "monotonic_increase"
    if np.all(diffs <= float(atol)):
        return "monotonic_decrease"
    return "non_monotonic"


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
        return (
            [(float(cx), float(cy)) for cx, cy in ring["centers_lamD"]],
            float(ring["resolved_radius_lamD"]),
        )
    if region_shape_name == "ring":
        return (
            [(float(planet_center_lamD[0]), float(planet_center_lamD[1]))],
            float(requested_roi_size_lamD),
        )
    return (
        [(float(planet_center_lamD[0]), float(planet_center_lamD[1]))],
        float(requested_roi_size_lamD),
    )


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
    phase_sweep_mode: str = "regional",
) -> tuple[list[dict[str, float | int]], dict[str, float | int] | None, list[dict[str, object]]]:
    orbit_radius_lamD = float(np.hypot(*planet_center))
    if orbit_radius_lamD <= 0.0:
        return [], None, []
    initial_angle_rad = float(np.arctan2(planet_center[1], planet_center[0]))
    sweep_mode = str(phase_sweep_mode).strip().lower()
    rows: list[dict[str, float | int]] = []
    best_entry: dict[str, float | int] | None = None
    panels: list[dict[str, object]] = []
    for roi_size in roi_sizes:
        if sweep_mode == "mask_rotation":
            roi_centers = []
            roi_radius_eff = float(roi_size)
        elif _is_whole_focal_plane_phase_mode(sweep_mode):
            roi_centers = []
            roi_radius_eff = float(roi_size)
        else:
            roi_centers, roi_radius_eff = _resolve_roi_configuration(
                region_shape_name=region_shape_name,
                requested_roi_size_lamD=float(roi_size),
                orbit_radius_lamD=orbit_radius_lamD,
                initial_angle_rad=initial_angle_rad,
                planet_center_lamD=planet_center,
            )
        stack = np.zeros((phase_offsets.size, 2 * half16, 2 * half16), dtype=float)
        focal_plane_phase_shift_stack: np.ndarray | None = None
        lyot_intensity_stack: np.ndarray | None = None
        lyot_crop_slices: tuple[slice, slice] | None = None
        for i, ph in enumerate(phase_offsets):
            if sweep_mode == "mask_rotation":
                phase_sim = CoronagraphSimulator(
                    **{
                        **sim_local,
                        "companion_offset_lamD": planet_center,
                        "e_final_phase_offset": 0.0,
                        "phase_mask_rotation_rad": float(ph),
                        "focal_local_phase_offset": 0.0,
                        "focal_local_phase_centers_lamD": (),
                        "focal_local_phase_radius_lamD": 0.0,
                    }
                )
            elif _is_whole_focal_plane_phase_mode(sweep_mode):
                phase_sim = CoronagraphSimulator(
                    **{
                        **sim_local,
                        "companion_offset_lamD": planet_center,
                        "e_final_phase_offset": float(ph),
                        "phase_mask_rotation_rad": 0.0,
                        "focal_local_phase_offset": 0.0,
                        "focal_local_phase_centers_lamD": (),
                        "focal_local_phase_radius_lamD": 0.0,
                    }
                )
            else:
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
            phase_result = phase_sim.run()
            stack[i] = phase_result["final_psf_with_ghost"][sl16, sl16]
            if collect_panels:
                if focal_plane_phase_shift_stack is None and _is_whole_focal_plane_phase_mode(sweep_mode):
                    focal_plane_phase_shift_stack = np.zeros_like(stack, dtype=np.float32)
                if focal_plane_phase_shift_stack is not None:
                    focal_plane_phase_shift_stack[i] = float(ph)
                if lyot_crop_slices is None:
                    lyot_stop = np.asarray(phase_result["lyot_stop"], dtype=float)
                    support_y, support_x = np.where(lyot_stop > 0.0)
                    if support_y.size > 0 and support_x.size > 0:
                        y0 = max(int(np.min(support_y)) - 2, 0)
                        y1 = min(int(np.max(support_y)) + 3, lyot_stop.shape[0])
                        x0 = max(int(np.min(support_x)) - 2, 0)
                        x1 = min(int(np.max(support_x)) + 3, lyot_stop.shape[1])
                    else:
                        y0, y1 = 0, lyot_stop.shape[0]
                        x0, x1 = 0, lyot_stop.shape[1]
                    lyot_crop_slices = (slice(y0, y1), slice(x0, x1))
                    lyot_intensity_stack = np.zeros(
                        (phase_offsets.size, y1 - y0, x1 - x0),
                        dtype=np.float32,
                    )
                lyot_field = np.asarray(
                    phase_result["lyot_field_before_reference_subtraction"],
                    dtype=np.complex128,
                )
                y_sl, x_sl = lyot_crop_slices
                lyot_intensity_stack[i] = np.abs(lyot_field[y_sl, x_sl]) ** 2
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
            "planet_flux_ratio": float(sim_local.get("companion_flux_ratio", 0.0)),
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
                    spec = _cdi_frequency_selection_spectrum(
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
                    "phase_sweep_mode": sweep_mode,
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
                    "lyot_intensity_stack": (
                        np.array(lyot_intensity_stack, dtype=np.float32)
                        if lyot_intensity_stack is not None
                        else np.zeros((0, 0, 0), dtype=np.float32)
                    ),
                    "lyot_phase_offsets_rad": np.asarray(phase_offsets, dtype=float),
                    "focal_plane_intensity_stack": (
                        np.array(stack, dtype=np.float32)
                        if _is_whole_focal_plane_phase_mode(sweep_mode)
                        else np.zeros((0, 0, 0), dtype=np.float32)
                    ),
                    "focal_plane_phase_offsets_rad": np.asarray(phase_offsets, dtype=float),
                    "focal_plane_phase_shift_stack": (
                        np.array(focal_plane_phase_shift_stack, dtype=np.float32)
                        if focal_plane_phase_shift_stack is not None
                        else np.zeros((0, 0, 0), dtype=np.float32)
                    ),
                }
            )
    return rows, best_entry, panels
