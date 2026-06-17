from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from .region_shapes import build_touching_circle_ring
from .simulator import CoronagraphSimulator


def make_phase_offsets(n_cycles: int = 8, steps_per_cycle: int = 32) -> np.ndarray:
    if n_cycles < 1:
        raise ValueError("n_cycles must be >= 1.")
    if steps_per_cycle < 2:
        raise ValueError("steps_per_cycle must be >= 2.")
    cycle = np.linspace(0.0, 2.0 * np.pi, steps_per_cycle, endpoint=False, dtype=float)
    return np.tile(cycle, int(n_cycles))


def circles_of_circles_center_sets(
    ring_radius_lamD: float = 8.0,
    circle_radius_lamD: float = 0.6,
    n_relocations: int = 3,
    n_circles: int | None = None,
    spacing_factor: float = 3.0,
    initial_angle_rad: float = 0.0,
) -> list[list[tuple[float, float]]]:
    if ring_radius_lamD <= 0.0:
        raise ValueError("ring_radius_lamD must be > 0.")
    if circle_radius_lamD <= 0.0:
        raise ValueError("circle_radius_lamD must be > 0.")
    if n_relocations < 1:
        raise ValueError("n_relocations must be >= 1.")
    if spacing_factor <= 0.0:
        raise ValueError("spacing_factor must be > 0.")

    if n_circles is None:
        circumference = 2.0 * np.pi * float(ring_radius_lamD)
        pitch_lamD = float(spacing_factor) * 2.0 * float(circle_radius_lamD)
        n_circles_eff = max(3, int(np.round(circumference / max(pitch_lamD, 1e-12))))
    else:
        n_circles_eff = int(n_circles)
        if n_circles_eff < 3:
            raise ValueError("n_circles must be >= 3.")

    dtheta = 2.0 * np.pi / float(n_circles_eff)
    base_angles = np.arange(n_circles_eff, dtype=float) * dtheta + float(initial_angle_rad)
    center_sets: list[list[tuple[float, float]]] = []
    for k in range(int(n_relocations)):
        shift = float(k) * dtheta / float(n_relocations)
        angles = base_angles + shift
        centers = [
            (float(ring_radius_lamD * np.cos(a)), float(ring_radius_lamD * np.sin(a)))
            for a in angles
        ]
        center_sets.append(centers)
    return center_sets


def run_local_phase_stack(
    sim_kwargs: dict,
    phase_offsets_rad: np.ndarray,
    local_phase_centers_lamD: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    local_phase_radius_lamD: float,
) -> tuple[np.ndarray, dict]:
    if len(local_phase_centers_lamD) == 0:
        raise ValueError("local_phase_centers_lamD must contain at least one center.")
    if local_phase_radius_lamD <= 0.0:
        raise ValueError("local_phase_radius_lamD must be > 0.")

    phases = np.asarray(phase_offsets_rad, dtype=float)
    if phases.ndim != 1 or phases.size < 2:
        raise ValueError("phase_offsets_rad must be a 1D array with at least 2 samples.")

    centers = tuple((float(x), float(y)) for x, y in local_phase_centers_lamD)
    local_kwargs = dict(sim_kwargs)
    local_kwargs["e_final_phase_offset"] = 0.0
    local_kwargs["companion_flux_ratio"] = 0.0
    local_kwargs["companion_offset_lamD"] = (0.0, 0.0)

    stack: np.ndarray | None = None
    template_result: dict | None = None
    for i, phase in enumerate(phases):
        frame_kwargs = dict(local_kwargs)
        frame_kwargs["focal_local_phase_offset"] = float(phase)
        frame_kwargs["focal_local_phase_centers_lamD"] = centers
        frame_kwargs["focal_local_phase_radius_lamD"] = float(local_phase_radius_lamD)
        result = CoronagraphSimulator(**frame_kwargs).run()
        frame = result["final_psf_with_ghost"].astype(float)
        if stack is None:
            stack = np.empty((phases.size, frame.shape[0], frame.shape[1]), dtype=float)
            template_result = result
        stack[i] = frame
    return stack, template_result if template_result is not None else {}


def run_local_phase_stack_for_center_sets(
    sim_kwargs: dict,
    phase_offsets_rad: np.ndarray,
    center_sets_lamD: list[list[tuple[float, float]]],
    local_phase_radius_lamD: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if len(center_sets_lamD) == 0:
        raise ValueError("center_sets_lamD must contain at least one set.")
    if local_phase_radius_lamD <= 0.0:
        raise ValueError("local_phase_radius_lamD must be > 0.")

    phases = np.asarray(phase_offsets_rad, dtype=float)
    if phases.ndim != 1 or phases.size < 2:
        raise ValueError("phase_offsets_rad must be a 1D array with at least 2 samples.")

    local_kwargs = dict(sim_kwargs)
    local_kwargs["e_final_phase_offset"] = 0.0
    local_kwargs["companion_flux_ratio"] = 0.0
    local_kwargs["companion_offset_lamD"] = (0.0, 0.0)

    n_frames = len(center_sets_lamD) * phases.size
    stack: np.ndarray | None = None
    frame_phases = np.empty(n_frames, dtype=float)
    frame_set_indices = np.empty(n_frames, dtype=int)
    template_result: dict | None = None
    pos = 0
    for set_idx, centers in enumerate(center_sets_lamD):
        if len(centers) == 0:
            raise ValueError("Each center set must contain at least one center.")
        centers_tuple = tuple((float(x), float(y)) for x, y in centers)
        for phase in phases:
            frame_kwargs = dict(local_kwargs)
            frame_kwargs["focal_local_phase_offset"] = float(phase)
            frame_kwargs["focal_local_phase_centers_lamD"] = centers_tuple
            frame_kwargs["focal_local_phase_radius_lamD"] = float(local_phase_radius_lamD)
            result = CoronagraphSimulator(**frame_kwargs).run()
            frame = result["final_psf_with_ghost"].astype(float)
            if stack is None:
                stack = np.empty((n_frames, frame.shape[0], frame.shape[1]), dtype=float)
                template_result = result
            stack[pos] = frame
            frame_phases[pos] = float(phase)
            frame_set_indices[pos] = int(set_idx)
            pos += 1

    return (
        stack if stack is not None else np.empty((0, 0, 0), dtype=float),
        frame_phases,
        frame_set_indices,
        template_result if template_result is not None else {},
    )


def run_two_source_local_phase_stack(
    sim_kwargs: dict,
    phase_offsets_rad: np.ndarray,
    local_phase_centers_lamD: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    local_phase_radius_lamD: float,
    companion_offset_lamD: tuple[float, float] = (8.0, 0.0),
    companion_flux_ratio: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if companion_flux_ratio <= 0.0:
        raise ValueError("companion_flux_ratio must be > 0.")
    if len(local_phase_centers_lamD) == 0:
        raise ValueError("local_phase_centers_lamD must contain at least one center.")
    if local_phase_radius_lamD <= 0.0:
        raise ValueError("local_phase_radius_lamD must be > 0.")

    phases = np.asarray(phase_offsets_rad, dtype=float)
    if phases.ndim != 1 or phases.size < 2:
        raise ValueError("phase_offsets_rad must be a 1D array with at least 2 samples.")

    centers = tuple((float(x), float(y)) for x, y in local_phase_centers_lamD)
    local_kwargs = dict(sim_kwargs)
    local_kwargs["e_final_phase_offset"] = 0.0
    local_kwargs["companion_flux_ratio"] = 0.0
    local_kwargs["companion_offset_lamD"] = (0.0, 0.0)
    star_shift = tuple(float(v) for v in local_kwargs.get("focal_shift_pixels", (0.0, 0.0)))
    companion_shift = (
        float(star_shift[0]) - float(companion_offset_lamD[0]) * float(local_kwargs["focal_sampling"]),
        float(star_shift[1]) - float(companion_offset_lamD[1]) * float(local_kwargs["focal_sampling"]),
    )

    stack_total: np.ndarray | None = None
    stack_star: np.ndarray | None = None
    stack_comp: np.ndarray | None = None
    template_result: dict | None = None
    for i, phase in enumerate(phases):
        frame_kwargs = dict(local_kwargs)
        frame_kwargs["focal_local_phase_offset"] = float(phase)
        frame_kwargs["focal_local_phase_centers_lamD"] = centers
        frame_kwargs["focal_local_phase_radius_lamD"] = float(local_phase_radius_lamD)

        star_kwargs = dict(frame_kwargs)
        star_kwargs["focal_shift_pixels"] = star_shift
        star_res = CoronagraphSimulator(**star_kwargs).run()
        i_star = star_res["final_psf_with_ghost"].astype(float)

        comp_kwargs = dict(frame_kwargs)
        comp_kwargs["focal_shift_pixels"] = companion_shift
        comp_kwargs["source_amplitude"] = float(np.sqrt(companion_flux_ratio))
        comp_kwargs["normalization_peak"] = float(star_res["normalization_peak"])
        comp_res = CoronagraphSimulator(**comp_kwargs).run()
        i_comp = comp_res["final_psf_with_ghost"].astype(float)

        if stack_total is None:
            stack_total = np.empty((phases.size, i_star.shape[0], i_star.shape[1]), dtype=float)
            stack_star = np.empty_like(stack_total)
            stack_comp = np.empty_like(stack_total)
            template_result = star_res
        stack_star[i] = i_star
        stack_comp[i] = i_comp
        stack_total[i] = i_star + i_comp

    return (
        stack_total if stack_total is not None else np.empty((0, 0, 0), dtype=float),
        stack_star if stack_star is not None else np.empty((0, 0, 0), dtype=float),
        stack_comp if stack_comp is not None else np.empty((0, 0, 0), dtype=float),
        template_result if template_result is not None else {},
    )


def coherence_map_from_phase_stack(
    stack: np.ndarray,
    phase_offsets_rad: np.ndarray,
    zero_phase_atol: float = 1e-6,
    eps: float = 1e-12,
) -> dict:
    data = np.asarray(stack, dtype=float)
    phases = np.asarray(phase_offsets_rad, dtype=float)
    if data.ndim != 3:
        raise ValueError("stack must have shape (n_frames, ny, nx).")
    if phases.ndim != 1 or phases.size != data.shape[0]:
        raise ValueError("phase_offsets_rad must be 1D with length matching stack frames.")

    wrapped = np.mod(phases, 2.0 * np.pi)
    zero_mask = np.isclose(wrapped, 0.0, atol=zero_phase_atol) | np.isclose(
        wrapped, 2.0 * np.pi, atol=zero_phase_atol
    )
    if not np.any(zero_mask):
        zero_idx = int(np.argmin(np.minimum(wrapped, 2.0 * np.pi - wrapped)))
        zero_mask = np.zeros_like(wrapped, dtype=bool)
        zero_mask[zero_idx] = True

    mean_map_all_phases = np.mean(data, axis=0)
    ref = mean_map_all_phases
    ref_safe = np.maximum(ref, eps)
    normalized = data / ref_safe[None, :, :]
    exp_term = np.exp(-1j * phases)[:, None, None]
    modulation = np.mean(normalized * exp_term, axis=0)
    coherence_map = np.abs(modulation) ** 2

    return {
        "reference_zero_phase_median": ref,
        "reference_mean_map_all_phases": mean_map_all_phases,
        "coherence_map_raw": coherence_map,
        "zero_phase_frame_mask": zero_mask,
    }


def incoherence_map_from_coherence(
    coherence_map: np.ndarray,
    p: float = 1.0,
    alpha: float | None = None,
    alpha_scale: float = 1.05,
    iterations: int = 1,
    eps: float = 1e-12,
) -> dict:
    if p <= 0.0:
        raise ValueError("p must be > 0.")
    if iterations < 1:
        raise ValueError("iterations must be >= 1.")
    if alpha_scale <= 1.0 and alpha is None:
        raise ValueError("alpha_scale must be > 1.0 when alpha is not provided.")

    coh = np.asarray(coherence_map, dtype=float)
    coh_safe = np.maximum(coh, eps)
    incoh_raw = 1.0 / np.power(coh_safe, p)
    if alpha is not None:
        alpha_used = float(alpha)
    else:
        finite = incoh_raw[np.isfinite(incoh_raw)]
        if finite.size == 0:
            raise RuntimeError("No finite incoherence values available to infer alpha.")
        baseline = float(np.percentile(finite, 99.9))
        alpha_used = float(alpha_scale * max(baseline, eps))

    incoh = incoh_raw.copy()
    coh_processed = None
    for _ in range(iterations):
        coh_processed = np.maximum(alpha_used - incoh, eps)
        incoh = 1.0 / coh_processed

    return {
        "incoherence_map_raw": incoh_raw,
        "coherence_map_processed": coh_processed,
        "incoherence_map_processed": incoh,
        "alpha_used": alpha_used,
        "p_used": float(p),
        "iterations_used": int(iterations),
    }


def lamd_to_pixel_xy(center_lamD: tuple[float, float], n_fft: int, focal_sampling: float) -> tuple[float, float]:
    c = (float(n_fft) - 1.0) / 2.0
    x_px = c + float(center_lamD[0]) * float(focal_sampling)
    y_px = c + float(center_lamD[1]) * float(focal_sampling)
    return x_px, y_px


def snr_annulus(
    image: np.ndarray,
    star_center_px: tuple[float, float],
    target_center_px: tuple[float, float],
    target_radius_px: float = 2.0,
    annulus_half_width_px: float = 1.0,
    eps: float = 1e-20,
) -> dict:
    if target_radius_px <= 0.0:
        raise ValueError("target_radius_px must be > 0.")
    if annulus_half_width_px <= 0.0:
        raise ValueError("annulus_half_width_px must be > 0.")

    data = np.asarray(image, dtype=float)
    ny, nx = data.shape
    y, x = np.indices((ny, nx), dtype=float)
    sx, sy = float(star_center_px[0]), float(star_center_px[1])
    tx, ty = float(target_center_px[0]), float(target_center_px[1])

    r_star = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
    target_dist = math.hypot(tx - sx, ty - sy)
    annulus = np.abs(r_star - target_dist) <= float(annulus_half_width_px)
    target_disk = (x - tx) ** 2 + (y - ty) ** 2 <= float(target_radius_px) ** 2
    annulus_noise = annulus & (~target_disk)
    if not np.any(target_disk):
        raise RuntimeError("Target aperture is empty; increase target_radius_px.")
    if np.sum(annulus_noise) < 10:
        raise RuntimeError("Annulus contains too few noise pixels; increase annulus_half_width_px.")

    target_sum = float(np.sum(data[target_disk]))

    anchor_angle = math.atan2(ty - sy, tx - sx)
    ring = build_touching_circle_ring(
        requested_region_radius_lamD=float(target_radius_px),
        orbit_radius_lamD=float(target_dist),
        anchor_angle_rad=float(anchor_angle),
        rotation_fraction=0.0,
    )

    aperture_sums: list[float] = []
    for cx_rel, cy_rel in ring["centers_lamD"]:
        cx = sx + float(cx_rel)
        cy = sy + float(cy_rel)
        aperture_mask = (x - cx) ** 2 + (y - cy) ** 2 <= float(target_radius_px) ** 2
        if not np.any(aperture_mask):
            continue
        if np.any(aperture_mask & target_disk):
            continue
        if not np.all(annulus_noise[aperture_mask]):
            continue
        aperture_sums.append(float(np.sum(data[aperture_mask])))

    noise_std = float(np.std(np.asarray(aperture_sums, dtype=float))) if len(aperture_sums) > 0 else float("nan")
    snr = target_sum / max(noise_std, eps)
    return {
        "target_sum": target_sum,
        "noise_std_annulus": noise_std,
        "snr": snr,
        "target_distance_from_star_px": target_dist,
    }


def save_cdi_maps_figure(
    coherence_map: np.ndarray,
    incoherence_raw: np.ndarray,
    incoherence_processed: np.ndarray,
    save_path: str = "cdi_maps.png",
) -> None:
    coh = np.asarray(coherence_map, dtype=float)
    incoh_r = np.asarray(incoherence_raw, dtype=float)
    incoh_p = np.asarray(incoherence_processed, dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    for ax, title, img in zip(
        axes,
        ["Raw Coherence Map", "Raw Incoherence Map", "Processed Incoherence Map"],
        [coh, incoh_r, incoh_p],
    ):
        im = ax.imshow(np.log10(img + 1e-12), cmap="inferno", origin="lower")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)


def gaussian_companion_psf(
    n_fft: int,
    focal_sampling: float,
    center_lamD: tuple[float, float],
    flux_ratio_to_peak: float = 1e-3,
    fwhm_lamD: float = 1.0,
) -> np.ndarray:
    if flux_ratio_to_peak <= 0.0:
        raise ValueError("flux_ratio_to_peak must be > 0.")
    if fwhm_lamD <= 0.0:
        raise ValueError("fwhm_lamD must be > 0.")

    pix = np.arange(int(n_fft), dtype=float)
    xx, yy = np.meshgrid(pix, pix)
    x0, y0 = lamd_to_pixel_xy(center_lamD, n_fft=int(n_fft), focal_sampling=float(focal_sampling))
    sigma_px = (float(fwhm_lamD) * float(focal_sampling)) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    rr2 = (xx - x0) ** 2 + (yy - y0) ** 2
    g = np.exp(-0.5 * rr2 / max(sigma_px * sigma_px, 1e-20))
    return float(flux_ratio_to_peak) * g


def add_incoherent_companion_to_stack(stack: np.ndarray, companion_map: np.ndarray) -> np.ndarray:
    data = np.asarray(stack, dtype=float)
    comp = np.asarray(companion_map, dtype=float)
    if data.ndim != 3:
        raise ValueError("stack must have shape (n_frames, ny, nx).")
    if comp.shape != data.shape[1:]:
        raise ValueError("companion_map shape must match stack frame shape.")
    return data + comp[None, :, :]


def validate_with_synthetic_companion(
    sim_kwargs: dict,
    phase_offsets_rad: np.ndarray,
    local_phase_centers_lamD: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    local_phase_radius_lamD: float,
    companion_center_lamD: tuple[float, float] = (8.0, 0.0),
    companion_flux_ratio_to_peak: float = 1e-3,
    companion_fwhm_lamD: float = 1.0,
    validation_model: str = "two-source-forward",
    incoh_p: float = 1.0,
    alpha: float | None = None,
    alpha_scale: float = 1.05,
    iterations: int = 1,
    target_radius_lamD: float = 0.5,
    ring_half_width_lamD: float = 0.5,
) -> dict:
    model = str(validation_model).strip().lower()
    if model not in {"two-source-forward", "gaussian-additive"}:
        raise ValueError("validation_model must be 'two-source-forward' or 'gaussian-additive'.")

    companion = None
    stack_star = None
    stack_comp = None
    if model == "two-source-forward":
        stack, stack_star, stack_comp, template = run_two_source_local_phase_stack(
            sim_kwargs=sim_kwargs,
            phase_offsets_rad=phase_offsets_rad,
            local_phase_centers_lamD=local_phase_centers_lamD,
            local_phase_radius_lamD=local_phase_radius_lamD,
            companion_offset_lamD=companion_center_lamD,
            companion_flux_ratio=float(companion_flux_ratio_to_peak),
        )
    else:
        stack_base, template = run_local_phase_stack(
            sim_kwargs=sim_kwargs,
            phase_offsets_rad=phase_offsets_rad,
            local_phase_centers_lamD=local_phase_centers_lamD,
            local_phase_radius_lamD=local_phase_radius_lamD,
        )
        companion = gaussian_companion_psf(
            n_fft=int(template["n_fft"]),
            focal_sampling=float(template["focal_sampling"]),
            center_lamD=companion_center_lamD,
            flux_ratio_to_peak=float(companion_flux_ratio_to_peak),
            fwhm_lamD=float(companion_fwhm_lamD),
        )
        stack = add_incoherent_companion_to_stack(stack_base, companion)

    coh = coherence_map_from_phase_stack(stack=stack, phase_offsets_rad=phase_offsets_rad)
    incoh = incoherence_map_from_coherence(
        coherence_map=coh["coherence_map_raw"],
        p=float(incoh_p),
        alpha=alpha,
        alpha_scale=float(alpha_scale),
        iterations=int(iterations),
    )

    n_fft = int(template["n_fft"])
    samp = float(template["focal_sampling"])
    star_center = ((n_fft - 1.0) / 2.0, (n_fft - 1.0) / 2.0)
    target_center = lamd_to_pixel_xy(companion_center_lamD, n_fft=n_fft, focal_sampling=samp)
    target_radius_px = float(target_radius_lamD) * samp
    ring_half_width_px = float(ring_half_width_lamD) * samp

    snr_raw = snr_annulus(
        image=incoh["incoherence_map_raw"],
        star_center_px=star_center,
        target_center_px=target_center,
        target_radius_px=target_radius_px,
        annulus_half_width_px=ring_half_width_px,
    )
    snr_processed = snr_annulus(
        image=incoh["incoherence_map_processed"],
        star_center_px=star_center,
        target_center_px=target_center,
        target_radius_px=target_radius_px,
        annulus_half_width_px=ring_half_width_px,
    )
    gain = snr_processed["snr"] / max(snr_raw["snr"], 1e-20)

    return {
        "validation_model": model,
        "template_result": template,
        "stack_with_companion": stack,
        "companion_map": companion,
        "stack_star_only": stack_star,
        "stack_companion_only": stack_comp,
        "coherence": coh,
        "incoherence": incoh,
        "snr_raw": snr_raw,
        "snr_processed": snr_processed,
        "snr_gain_processed_over_raw": float(gain),
    }
