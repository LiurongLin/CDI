from __future__ import annotations

import argparse
import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from .cdi_analysis import (
    SNR_ANNULUS_HALF_WIDTH_LAMD,
    SNR_APERTURE_RADIUS_LAMD,
    _compute_incoherence_map,
    _compute_incoherence_map_info,
    _evaluate_best_roi_for_planet_center,
    _focal_shift_lamD,
    _inclusive_float_range,
    _is_whole_focal_plane_phase_mode,
    _lyot_reference_folder_tag,
    _local_phase_region_kwargs,
    _noise_aperture_centers_lamD,
    _phase_screen_folder_tag,
    _planet_region_centered_snr,
    _planet_region_snr,
    _planet_region_snr_from_coherence,
    _polar_to_cartesian_lamD,
    _select_reference_speckle_centers_lamD,
    _summarize_roi_snr_trend,
    _theta_back_and_forth,
)
from .cdi_reports import (
    _save_focal_plane_field_phase_grid_png,
    _save_focal_plane_phase_grid_png,
    _save_focal_plane_phase_shift_grid_png,
    _save_grouped_roi_size_focal_plane_field_phase_pngs,
    _save_grouped_roi_size_focal_plane_pngs,
    _save_grouped_roi_size_focal_plane_phase_shift_pngs,
    _save_grouped_roi_size_lyot_plane_field_phase_pngs,
    _save_lyot_plane_phase_grid_png,
    _save_lyot_plane_field_phase_grid_png,
    _save_grouped_roi_size_lyot_plane_pngs,
    _save_map_panel_summary_png,
    _save_grouped_roi_size_coherence_pdfs,
    _save_grouped_roi_size_incoherence_pdfs,
    _save_planet_locations_on_mean_final_psf,
    _save_planet_position_snr_summary_pdf,
    _save_ring_of_circle_rotation_gif,
    _save_ring_rotation_probe_fft_page,
    _save_roi_size_coherence_pdf_for_planet_location,
    _save_roi_size_fft_spectra_pdf_for_planet_location,
    _save_roi_size_incoherence_pdf_for_planet_location,
    _save_roi_size_map_pdf_for_planet_location,
    _save_roi_size_max_minus_coherence_pdf_for_planet_location,
    _write_rgb_gif,
)
from .plotting import (
    _cdi_build_incoherence_maps,
    plot_cdi_planet_phase_outputs,
)
from .region_shapes import (
    annulus_radii_from_width,
    build_touching_circle_ring,
    normalize_region_shape,
)
from .simulator import CoronagraphSimulator


RESULTS_ROOT_DIR = "results"


def _compact_float_tag(value: float, precision: int = 3) -> str:
    return f"{float(value):.{int(precision)}f}".replace(".", "p")


def _roi_shape_folder_name(region_shape: str) -> str:
    return f"shape_{normalize_region_shape(region_shape)}"


def _roi_shape_folder_parts_for_mode(region_shape: str, phase_sweep_mode: str) -> list[str]:
    mode = str(phase_sweep_mode).strip().lower()
    if mode == "mask_rotation" or _is_whole_focal_plane_phase_mode(mode):
        return []
    return [_roi_shape_folder_name(region_shape)]


def _modulation_sweep_folder_name(phase_sweep_mode: str) -> str:
    mode = str(phase_sweep_mode).strip().lower()
    return f"mode_{mode}"


def _source_state_folder_name(
    include_star: bool,
    include_companion: bool,
    coherent_ring_speckle_count: int = 0,
    coherent_ring_speckle_intensity: float = 1.0,
) -> str:
    star_state = "star_enabled" if bool(include_star) else "star_disabled"
    companion_state = "planet_enabled" if bool(include_companion) else "planet_disabled"
    speckles_enabled = (
        int(coherent_ring_speckle_count) > 0
        and float(coherent_ring_speckle_intensity) > 0.0
    )
    speckle_state = "speckles_enabled" if speckles_enabled else "speckles_disabled"
    return f"{star_state}_{companion_state}_{speckle_state}"


def _contrast_ratio_folder_name(contrast_ratio_token: str) -> str:
    return f"contrast_ratio_{contrast_ratio_token}"


def _mask_rotation_angles_rad(n_steps: int) -> np.ndarray:
    if int(n_steps) < 1:
        raise ValueError("n_steps must be >= 1.")
    return np.linspace(0.0, 2.0 * np.pi, int(n_steps), endpoint=False, dtype=float)


def _effective_roi_sizes_for_mode(
    roi_sizes: np.ndarray,
    phase_sweep_mode: str,
) -> np.ndarray:
    values = np.asarray(roi_sizes, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("roi_sizes must be a non-empty 1D array.")
    mode = str(phase_sweep_mode).strip().lower()
    if mode == "mask_rotation" or _is_whole_focal_plane_phase_mode(mode):
        return np.array([float(values[0])], dtype=float)
    return values


def _save_perfect_coronagraph_unmodulated_final_psf(
    *,
    sim_local: dict,
    output_dir: str,
    local_region_radius_lamD: float,
    mask_output_tag: str,
    phase_cycles_tag: str,
    phase_sweep_mode_tag: str,
    single_region_tag: str,
    ghost_suffix: str,
) -> str:
    """Save the final PSF with all focal-plane modulation disabled."""
    baseline = CoronagraphSimulator(
        **{
            **sim_local,
            "e_final_phase_offset": 0.0,
            "phase_mask_rotation_rad": 0.0,
            "focal_local_phase_offset": 0.0,
            "focal_local_phase_centers_lamD": (),
            "focal_local_phase_radius_lamD": 0.0,
            "focal_local_phase_ring_center_lamD": (0.0, 0.0),
            "focal_local_phase_inner_radius_lamD": 0.0,
            "focal_local_phase_outer_radius_lamD": 0.0,
        }
    ).run()

    psf = np.asarray(baseline["final_psf_with_ghost"], dtype=float)
    n_fft = int(baseline["n_fft"])
    samp = float(baseline["focal_sampling"])
    crop_lamD = 16.0
    half = int(0.5 * (2.0 * crop_lamD) * samp)
    cc = n_fft // 2
    sl = slice(cc - half, cc + half)

    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    im = ax.imshow(
        np.log10(np.maximum(psf[sl, sl], 1e-12)),
        origin="lower",
        cmap="inferno",
        vmin=-8,
        vmax=0,
        extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD],
    )
    ax.set_title("Perfect corongraph final PSF without focal-plane phase offset")
    ax.set_xlabel(r"$\lambda/D$")
    ax.set_ylabel(r"$\lambda/D$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\log_{10}$ intensity")

    out_path = (
        f"{output_dir}/cdi_planet_final_psf_no_focal_phase_offset_32lamD_local_"
        f"{float(local_region_radius_lamD):.3f}_{mask_output_tag}{phase_cycles_tag}"
        f"{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png"
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_perfect_coronagraph_subtracted_final_psf(
    *,
    sim_local: dict,
    output_dir: str,
    local_region_radius_lamD: float,
    mask_output_tag: str,
    phase_cycles_tag: str,
    phase_sweep_mode_tag: str,
    single_region_tag: str,
    ghost_suffix: str,
) -> str:
    """Save the ghost-free coronagraphic PSF after the reference wavefront subtraction."""
    result = CoronagraphSimulator(**sim_local).run()

    psf = np.asarray(result["coronagraphic_psf"], dtype=float)
    n_fft = int(result["n_fft"])
    samp = float(result["focal_sampling"])
    crop_lamD = 8.0
    half = int(0.5 * (2.0 * crop_lamD) * samp)
    cc = n_fft // 2
    sl = slice(cc - half, cc + half)

    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    im = ax.imshow(
        np.log10(np.maximum(psf[sl, sl], 1e-12)),
        origin="lower",
        cmap="inferno",
        vmin=-8,
        vmax=0,
        extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD],
    )
    ax.set_title("Perfect corongraph coronagraphic PSF after reference subtraction")
    ax.set_xlabel(r"$\lambda/D$")
    ax.set_ylabel(r"$\lambda/D$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\log_{10}$ intensity")

    out_path = (
        f"{output_dir}/cdi_planet_final_psf_after_reference_subtraction_16lamD_local_"
        f"{float(local_region_radius_lamD):.3f}_{mask_output_tag}{phase_cycles_tag}"
        f"{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png"
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_perfect_coronagraph_subtracted_final_psf_star_only(
    *,
    sim_local: dict,
    output_dir: str,
    local_region_radius_lamD: float,
    mask_output_tag: str,
    phase_cycles_tag: str,
    phase_sweep_mode_tag: str,
    single_region_tag: str,
    ghost_suffix: str,
) -> str:
    """Save the ghost-free star-only coronagraphic PSF after reference subtraction."""
    result = CoronagraphSimulator(
        **{
            **sim_local,
            "companion_flux_ratio": 0.0,
            "companion_offset_lamD": (0.0, 0.0),
        }
    ).run()

    psf = np.asarray(result["coronagraphic_psf_star"], dtype=float)
    n_fft = int(result["n_fft"])
    samp = float(result["focal_sampling"])
    crop_lamD = 8.0
    half = int(0.5 * (2.0 * crop_lamD) * samp)
    cc = n_fft // 2
    sl = slice(cc - half, cc + half)

    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    im = ax.imshow(
        np.log10(np.maximum(psf[sl, sl], 1e-12)),
        origin="lower",
        cmap="inferno",
        vmin=-8,
        vmax=0,
        extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD],
    )
    ax.set_title("Perfect corongraph star-only coronagraphic PSF after reference subtraction")
    ax.set_xlabel(r"$\lambda/D$")
    ax.set_ylabel(r"$\lambda/D$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\log_{10}$ intensity")

    out_path = (
        f"{output_dir}/cdi_planet_final_psf_after_reference_subtraction_star_only_16lamD_local_"
        f"{float(local_region_radius_lamD):.3f}_{mask_output_tag}{phase_cycles_tag}"
        f"{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png"
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_perfect_coronagraph_diagnostic_psfs(
    *,
    sim_local: dict,
    output_dir: str,
    local_region_radius_lamD: float,
    mask_output_tag: str,
    phase_cycles_tag: str,
    phase_sweep_mode_tag: str,
    single_region_tag: str,
    ghost_suffix: str,
) -> None:
    try:
        subtracted_psf_path = _save_perfect_coronagraph_subtracted_final_psf(
            sim_local=sim_local,
            output_dir=output_dir,
            local_region_radius_lamD=local_region_radius_lamD,
            mask_output_tag=mask_output_tag,
            phase_cycles_tag=phase_cycles_tag,
            phase_sweep_mode_tag=phase_sweep_mode_tag,
            single_region_tag=single_region_tag,
            ghost_suffix=ghost_suffix,
        )
        print(
            "Saved perfect corongraph final PSF after reference subtraction: "
            f"{subtracted_psf_path}"
        )
    except Exception as exc:
        print(
            "Could not save perfect corongraph final PSF after reference subtraction: "
            f"{exc}"
        )
    try:
        star_only_subtracted_psf_path = _save_perfect_coronagraph_subtracted_final_psf_star_only(
            sim_local=sim_local,
            output_dir=output_dir,
            local_region_radius_lamD=local_region_radius_lamD,
            mask_output_tag=mask_output_tag,
            phase_cycles_tag=phase_cycles_tag,
            phase_sweep_mode_tag=phase_sweep_mode_tag,
            single_region_tag=single_region_tag,
            ghost_suffix=ghost_suffix,
        )
        print(
            "Saved perfect corongraph star-only final PSF after reference subtraction: "
            f"{star_only_subtracted_psf_path}"
        )
    except Exception as exc:
        print(
            "Could not save perfect corongraph star-only final PSF after reference subtraction: "
            f"{exc}"
        )
    try:
        no_phase_psf_path = _save_perfect_coronagraph_unmodulated_final_psf(
            sim_local=sim_local,
            output_dir=output_dir,
            local_region_radius_lamD=local_region_radius_lamD,
            mask_output_tag=mask_output_tag,
            phase_cycles_tag=phase_cycles_tag,
            phase_sweep_mode_tag=phase_sweep_mode_tag,
            single_region_tag=single_region_tag,
            ghost_suffix=ghost_suffix,
        )
        print(
            "Saved perfect corongraph final PSF without focal-plane phase offset: "
            f"{no_phase_psf_path}"
        )
    except Exception as exc:
        print(
            "Could not save perfect corongraph final PSF without focal-plane phase offset: "
            f"{exc}"
        )


def _inclusive_int_range(start: int, stop: int, step: int) -> np.ndarray:
    start_i = int(start)
    stop_i = int(stop)
    step_i = int(step)
    if step_i <= 0:
        raise ValueError("step must be > 0.")
    if stop_i < start_i:
        raise ValueError("stop must be >= start.")
    return np.arange(start_i, stop_i + 1, step_i, dtype=int)


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
) -> list[dict[str, float | int]]:
    os.makedirs(sweep_output_dir, exist_ok=True)
    poster_figure = bool(getattr(args, "plot_poster_figure", False))
    region_shape_name = normalize_region_shape(args.region_shape)
    phase_sweep_mode = str(getattr(args, "phase_sweep_mode", "regional")).strip().lower()
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
    roi_sizes = _effective_roi_sizes_for_mode(
        _inclusive_float_range(args.roi_size_min, args.roi_size_max, args.roi_size_step),
        phase_sweep_mode,
    )
    if phase_sweep_mode == "mask_rotation":
        print(
            "[planet-roi-polar] mask_rotation mode active: "
            "ROI-size inputs are ignored; evaluating a single placeholder ROI sample."
        )
    elif _is_whole_focal_plane_phase_mode(phase_sweep_mode):
        print(
            "[planet-roi-polar] whole focal-plane phase mode active: "
            "ROI-size inputs are ignored; evaluating a single placeholder ROI sample."
        )

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
        f"_r{_compact_float_tag(args.planet_position_radius_min)}-"
        f"{_compact_float_tag(args.planet_position_radius_max)}-"
        f"{_compact_float_tag(args.planet_position_radius_step)}"
    )
    theta_tag = (
        f"_t{_compact_float_tag(args.planet_position_theta_min_deg)}-"
        f"{_compact_float_tag(args.planet_position_theta_max_deg)}-"
        f"{_compact_float_tag(args.planet_position_theta_step_deg)}"
    )
    roi_tag = (
        f"_roi{_compact_float_tag(args.roi_size_min)}-"
        f"{_compact_float_tag(args.roi_size_max)}-"
        f"{_compact_float_tag(args.roi_size_step)}"
    )
    perfect_coronagraph = bool(sim_local.get("perfect_coronagraph", False))
    lyot_reference_sweep_min = float(args.lyot_reference_percent_sweep_min)
    lyot_reference_sweep_max = float(args.lyot_reference_percent_sweep_max)
    lyot_reference_sweep_step = float(args.lyot_reference_percent_sweep_step)
    if (
        perfect_coronagraph
        and np.isclose(lyot_reference_sweep_min, 100.0)
        and np.isclose(lyot_reference_sweep_max, 100.0)
        and np.isclose(lyot_reference_sweep_step, 0.0)
        and not np.isclose(float(args.lyot_reference_percent), 100.0)
    ):
        lyot_reference_sweep_min = float(args.lyot_reference_percent)
        lyot_reference_sweep_max = float(args.lyot_reference_percent)
        lyot_reference_sweep_step = 0.0
    lyot_reference_percent_vals = _inclusive_float_range(
        lyot_reference_sweep_min,
        lyot_reference_sweep_max,
        lyot_reference_sweep_step,
    ) if perfect_coronagraph else np.array([float(args.lyot_reference_percent)], dtype=float)
    lyot_reference_tag = (
        f"_lr{_compact_float_tag(lyot_reference_sweep_min)}-"
        f"{_compact_float_tag(lyot_reference_sweep_max)}-"
        f"{_compact_float_tag(lyot_reference_sweep_step)}"
    ) if perfect_coronagraph else ""
    if perfect_coronagraph:
        for lyot_reference_percent in lyot_reference_percent_vals:
            diagnostic_sim_local = dict(sim_local)
            diagnostic_sim_local["lyot_reference_scale"] = float(lyot_reference_percent) / 100.0
            diagnostic_single_region_tag = single_region_tag
            if lyot_reference_percent_vals.size > 1:
                diagnostic_single_region_tag = (
                    f"_lr{_compact_float_tag(lyot_reference_percent)}"
                    f"{single_region_tag}"
                )
            _save_perfect_coronagraph_diagnostic_psfs(
                sim_local=diagnostic_sim_local,
                output_dir=sweep_output_dir,
                local_region_radius_lamD=float(args.local_region_radius),
                mask_output_tag=mask_output_tag,
                phase_cycles_tag=phase_cycles_tag,
                phase_sweep_mode_tag=phase_sweep_mode_tag,
                single_region_tag=diagnostic_single_region_tag,
                ghost_suffix=ghost_suffix,
            )
    phase_cycles = float(args.phase_cycles)
    phase_offsets = np.linspace(0.0, 2.0 * np.pi * phase_cycles, int(args.phase_step), endpoint=True)
    rows: list[dict[str, float | int]] = []
    location_panels: list[dict[str, object]] = []

    for lyot_reference_percent in lyot_reference_percent_vals:
        sweep_sim_local = dict(sim_local)
        if perfect_coronagraph:
            sweep_sim_local["lyot_reference_scale"] = float(lyot_reference_percent) / 100.0
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
                    sim_local=sweep_sim_local,
                    phase_offsets=phase_offsets,
                    sl16=sl16,
                    half16=half16,
                    xx16=xx16,
                    yy16=yy16,
                    incoherence_map_mode=incoherence_map_mode,
                    collect_panels=True,
                    phase_sweep_mode=phase_sweep_mode,
                )
                for row in sample_rows:
                    if perfect_coronagraph:
                        row["lyot_reference_percent"] = float(lyot_reference_percent)
                rows.extend(sample_rows)
                if len(panels) > 0:
                    location_tag = (
                        (f"lr{_compact_float_tag(lyot_reference_percent)}_" if perfect_coronagraph else "")
                        + f"planet_r_{_compact_float_tag(radius_lamD)}"
                        + f"_theta_{_compact_float_tag(theta_deg)}"
                    )
                    location_dir = os.path.join(sweep_output_dir, location_tag)
                    os.makedirs(location_dir, exist_ok=True)
                    for panel in panels:
                        panel["planet_radius_lamD"] = float(radius_lamD)
                        panel["planet_theta_deg"] = float(theta_deg)
                        if perfect_coronagraph:
                            panel["lyot_reference_percent"] = float(lyot_reference_percent)
                    location_panels.append(
                        {
                            **({"lyot_reference_percent": float(lyot_reference_percent)} if perfect_coronagraph else {}),
                            "planet_radius_lamD": float(radius_lamD),
                            "planet_theta_deg": float(theta_deg),
                            "planet_center_lamD": (float(planet_center[0]), float(planet_center[1])),
                            "panels": list(panels),
                        }
                    )
                    location_pdf = os.path.join(
                        location_dir,
                        "planet_position_roi_size_sweep_incoherence_maps_with_snr_24lamD",
                    )
                    location_pdf = f"{location_pdf}_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_tag}{ghost_suffix}.pdf"
                    _save_roi_size_incoherence_pdf_for_planet_location(
                        output_path=location_pdf,
                        region_shape_name=region_shape_name,
                        panels=panels,
                        extent=extent,
                        poster_figure=poster_figure,
                    )
                    location_coh_pdf = os.path.join(
                        location_dir,
                        "planet_position_roi_size_sweep_coherence_maps_with_snr_24lamD",
                    )
                    location_coh_pdf = f"{location_coh_pdf}_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_tag}{ghost_suffix}.pdf"
                    _save_roi_size_coherence_pdf_for_planet_location(
                        output_path=location_coh_pdf,
                        region_shape_name=region_shape_name,
                        panels=panels,
                        extent=extent,
                        poster_figure=poster_figure,
                    )
                    if not poster_figure:
                        location_max_minus_coh_pdf = os.path.join(
                            location_dir,
                            "planet_position_roi_size_sweep_max_minus_coherence_maps_24lamD",
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
                                f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_tag}{ghost_suffix}.pdf",
                            )
                            _save_roi_size_fft_spectra_pdf_for_planet_location(
                                output_path=location_fft_pdf,
                                panels=panels,
                            )
                        location_csv = os.path.join(
                            location_dir,
                            "planet_position_roi_size_sweep_table_"
                            f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_tag}{ghost_suffix}.csv",
                        )
                        with open(location_csv, "w", newline="", encoding="utf-8") as fh:
                            writer = csv.DictWriter(
                                fh,
                                fieldnames=[
                                    "planet_flux_ratio",
                                    *(
                                        ["lyot_reference_percent"]
                                        if perfect_coronagraph
                                        else []
                                    ),
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
                    snr_trend = _summarize_roi_snr_trend(sample_rows)
                    print(
                        "[planet-roi-polar] "
                        + (
                            f"lyot_ref={float(lyot_reference_percent):.3f}% "
                            if perfect_coronagraph
                            else ""
                        )
                        + f"planet=(r={float(radius_lamD):+.3f}, theta={float(theta_deg):+.3f} deg) "
                        + f"xy=({planet_center[0]:+.3f}, {planet_center[1]:+.3f}) "
                        + f"best_snr={float(best_entry['snr']):.6e} "
                        + f"best_roi={float(best_entry['resolved_roi_size_lamD']):.3f} λ/D "
                        + f"roi_snr_trend={snr_trend}"
                    )

    if not poster_figure:
        grouped_incoherence_pdfs = _save_grouped_roi_size_incoherence_pdfs(
            output_dir=sweep_output_dir,
            base_name=(
                "planet_position_roi_size_sweep_incoherence_maps_with_snr_24lamD_"
                f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{lyot_reference_tag}{ghost_suffix}"
            ),
            region_shape_name=region_shape_name,
            location_panels=location_panels,
            extent=extent,
            poster_figure=poster_figure,
        )
        grouped_coherence_pdfs = _save_grouped_roi_size_coherence_pdfs(
            output_dir=sweep_output_dir,
            base_name=(
                "planet_position_roi_size_sweep_coherence_maps_with_snr_24lamD_"
                f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{lyot_reference_tag}{ghost_suffix}"
            ),
            region_shape_name=region_shape_name,
            location_panels=location_panels,
            extent=extent,
            poster_figure=poster_figure,
        )
        grouped_lyot_pngs = _save_grouped_roi_size_lyot_plane_pngs(
            output_dir=sweep_output_dir,
            base_name=(
                "planet_position_roi_size_sweep_lyot_plane_phase_grid_"
                f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{lyot_reference_tag}{ghost_suffix}"
            ),
            location_panels=location_panels,
        )
        grouped_lyot_phase_pngs = _save_grouped_roi_size_lyot_plane_field_phase_pngs(
            output_dir=sweep_output_dir,
            base_name=(
                "planet_position_roi_size_sweep_lyot_plane_field_phase_grid_"
                f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{lyot_reference_tag}{ghost_suffix}"
            ),
            location_panels=location_panels,
        )
        grouped_focal_pngs = _save_grouped_roi_size_focal_plane_pngs(
            output_dir=sweep_output_dir,
            base_name=(
                "planet_position_roi_size_sweep_focal_plane_phase_grid_"
                f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{lyot_reference_tag}{ghost_suffix}"
            ),
            location_panels=location_panels,
        )
        grouped_focal_field_phase_pngs = _save_grouped_roi_size_focal_plane_field_phase_pngs(
            output_dir=sweep_output_dir,
            base_name=(
                "planet_position_roi_size_sweep_focal_plane_field_phase_grid_"
                f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{lyot_reference_tag}{ghost_suffix}"
            ),
            location_panels=location_panels,
        )
        grouped_focal_phase_shift_pngs = _save_grouped_roi_size_focal_plane_phase_shift_pngs(
            output_dir=sweep_output_dir,
            base_name=(
                "planet_position_roi_size_sweep_focal_plane_phase_shift_grid_"
                f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{lyot_reference_tag}{ghost_suffix}"
            ),
            location_panels=location_panels,
        )
        out_csv = os.path.join(
            sweep_output_dir,
            "planet_position_roi_size_sweep_table_"
            f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{lyot_reference_tag}{ghost_suffix}.csv",
        )
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "planet_flux_ratio",
                    *(
                        ["lyot_reference_percent"]
                        if perfect_coronagraph
                        else []
                    ),
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
            f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{lyot_reference_tag}{ghost_suffix}.pdf",
        )
        _save_planet_position_snr_summary_pdf(
            output_path=out_snr_summary_pdf,
            location_panels=location_panels,
        )

        print(f"Saved planet-position/ROI-size sweep table: {out_csv}")
        print(f"Saved planet-position/ROI-size sweep SNR summary PDF: {out_snr_summary_pdf}")
        for path in grouped_incoherence_pdfs:
            print(f"Saved grouped incoherence map PDF: {path}")
        for path in grouped_coherence_pdfs:
            print(f"Saved grouped coherence map PDF: {path}")
        for path in grouped_lyot_pngs:
            print(f"Saved grouped Lyot-plane intensity grid PNG: {path}")
        for path in grouped_lyot_phase_pngs:
            print(f"Saved grouped Lyot-plane field-phase grid PNG: {path}")
        for path in grouped_focal_pngs:
            print(f"Saved grouped focal-plane phase-grid PNG: {path}")
        for path in grouped_focal_field_phase_pngs:
            print(f"Saved grouped focal-plane field-phase grid PNG: {path}")
        for path in grouped_focal_phase_shift_pngs:
            print(f"Saved grouped focal-plane phase-shift-map PNG: {path}")
    else:
        print("Poster figure mode enabled: emitted coherence and incoherence PDFs only.")
    return rows


def _run_planet_position_brightness_sweep(
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
    flux_ratio_vals = _inclusive_float_range(
        args.planet_flux_ratio_sweep_min,
        args.planet_flux_ratio_sweep_max,
        args.planet_flux_ratio_sweep_step,
    )
    combined_rows: list[dict[str, float | int]] = []
    phase_sweep_mode = str(getattr(args, "phase_sweep_mode", "regional")).strip().lower()
    if phase_sweep_mode == "mask_rotation":
        print(
            "[planet-position-brightness] mask_rotation mode active: "
            "sampling planet position and flux ratio against one full mask turn."
        )
    for flux_ratio in flux_ratio_vals:
        flux_token = str(f"{float(flux_ratio):.6f}").replace(".", "p")
        flux_dir = os.path.join(
            sweep_output_dir,
            f"planet_flux_ratio_{flux_token}",
        )
        flux_args = argparse.Namespace(**vars(args))
        flux_args.planet_flux_ratio_local = float(flux_ratio)
        flux_sim_local = dict(sim_local)
        flux_sim_local["companion_flux_ratio"] = float(flux_ratio)
        flux_rows = _run_planet_position_roi_size_sweep(
            args=flux_args,
            sim_local=flux_sim_local,
            incoherence_map_mode=incoherence_map_mode,
            sweep_output_dir=flux_dir,
            mask_output_tag=mask_output_tag,
            phase_cycles_tag=phase_cycles_tag,
            phase_sweep_mode_tag=phase_sweep_mode_tag,
            single_region_tag=single_region_tag,
            ghost_suffix=ghost_suffix,
        )
        combined_rows.extend(flux_rows)
        print(
            "[planet-position-brightness] "
            f"completed flux ratio={float(flux_ratio):.6e} with {len(flux_rows)} sampled row(s)."
        )

    out_csv = os.path.join(
        sweep_output_dir,
        "planet_position_brightness_sweep_table_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.csv",
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "planet_flux_ratio",
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
        writer.writerows(combined_rows)
    print(f"Saved planet-position/brightness sweep table: {out_csv}")


def _run_planet_position_map_sweep(
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
    phase_sweep_mode = str(getattr(args, "phase_sweep_mode", "regional")).strip().lower()
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
    roi_sizes = np.array([float(args.local_region_radius)], dtype=float)
    phase_offsets = (
        _mask_rotation_angles_rad(int(args.phase_step))
        if phase_sweep_mode == "mask_rotation"
        else np.linspace(0.0, 2.0 * np.pi * float(args.phase_cycles), int(args.phase_step), endpoint=True)
    )

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

    rows: list[dict[str, float | int]] = []
    summary_panels: list[dict[str, object]] = []
    summary_labels: list[str] = []
    planet_centers: list[tuple[float, float]] = []
    for theta_deg in theta_deg_vals:
        for radius_lamD in radius_vals:
            if float(radius_lamD) <= 0.0:
                continue
            planet_center = _polar_to_cartesian_lamD(
                radius_lamD=float(radius_lamD),
                theta_deg=float(theta_deg),
            )
            sample_rows, _best_entry, panels = _evaluate_best_roi_for_planet_center(
                planet_center=planet_center,
                roi_sizes=roi_sizes,
                region_shape_name=normalize_region_shape(args.region_shape),
                sim_local=sim_local,
                phase_offsets=phase_offsets,
                sl16=sl16,
                half16=half16,
                xx16=xx16,
                yy16=yy16,
                incoherence_map_mode=incoherence_map_mode,
                collect_panels=True,
                phase_sweep_mode=phase_sweep_mode,
            )
            rows.extend(sample_rows)
            if panels:
                summary_panels.append(panels[0])
                summary_labels.append(f"r={float(radius_lamD):.2f}, θ={float(theta_deg):.1f}°")
                planet_centers.append((float(planet_center[0]), float(planet_center[1])))

    out_csv = os.path.join(
        sweep_output_dir,
        "planet_position_map_sweep_table_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.csv",
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "planet_flux_ratio",
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
    out_png = os.path.join(
        sweep_output_dir,
        "planet_position_map_sweep_summary_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_map_panel_summary_png(
        output_path=out_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title="Planet Position Sweep: Incoherence and Coherence Maps",
    )
    out_lyot_png = os.path.join(
        sweep_output_dir,
        "planet_position_map_sweep_lyot_plane_phase_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_lyot_plane_phase_grid_png(
        output_path=out_lyot_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title="Planet Position Sweep: Lyot Plane by Phase Modulation",
    )
    out_lyot_phase_png = os.path.join(
        sweep_output_dir,
        "planet_position_map_sweep_lyot_plane_field_phase_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_lyot_plane_field_phase_grid_png(
        output_path=out_lyot_phase_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title="Planet Position Sweep: Lyot Plane Field Phase by Phase Modulation",
    )
    out_focal_png = os.path.join(
        sweep_output_dir,
        "planet_position_map_sweep_focal_plane_phase_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_focal_plane_phase_grid_png(
        output_path=out_focal_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title="Planet Position Sweep: Focal Plane by Phase Modulation",
    )
    out_focal_field_phase_png = os.path.join(
        sweep_output_dir,
        "planet_position_map_sweep_focal_plane_field_phase_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_focal_plane_field_phase_grid_png(
        output_path=out_focal_field_phase_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title="Planet Position Sweep: Focal Plane Field Phase by Phase Modulation",
    )
    out_focal_phase_shift_png = os.path.join(
        sweep_output_dir,
        "planet_position_map_sweep_focal_plane_phase_shift_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_focal_plane_phase_shift_grid_png(
        output_path=out_focal_phase_shift_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title="Planet Position Sweep: Applied Focal-Plane Phase Shift",
    )
    out_loc_png = os.path.join(
        sweep_output_dir,
        "planet_position_map_sweep_locations_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_planet_locations_on_mean_final_psf(
        output_path=out_loc_png,
        sim_local=sim_local,
        planet_centers_lamD=planet_centers,
    )
    print(f"Saved planet-position sweep table: {out_csv}")
    print(f"Saved planet-position sweep summary PNG: {out_png}")
    print(f"Saved planet-position sweep Lyot-plane intensity grid: {out_lyot_png}")
    if os.path.exists(out_lyot_phase_png):
        print(f"Saved planet-position sweep Lyot-plane field-phase grid: {out_lyot_phase_png}")
    if os.path.exists(out_focal_png):
        print(f"Saved planet-position sweep focal-plane phase grid: {out_focal_png}")
    if os.path.exists(out_focal_field_phase_png):
        print(f"Saved planet-position sweep focal-plane field-phase grid: {out_focal_field_phase_png}")
    if os.path.exists(out_focal_phase_shift_png):
        print(f"Saved planet-position sweep focal-plane phase-shift-map grid: {out_focal_phase_shift_png}")
    print(f"Saved planet-position sweep location overview: {out_loc_png}")


def _run_planet_flux_ratio_map_sweep(
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
    phase_sweep_mode = str(getattr(args, "phase_sweep_mode", "regional")).strip().lower()
    flux_ratio_vals = _inclusive_float_range(
        args.planet_flux_ratio_sweep_min,
        args.planet_flux_ratio_sweep_max,
        args.planet_flux_ratio_sweep_step,
    )
    roi_sizes = np.array([float(args.local_region_radius)], dtype=float)
    planet_center = (float(args.planet_offset_x_local), float(args.planet_offset_y_local))
    phase_offsets = (
        _mask_rotation_angles_rad(int(args.phase_step))
        if phase_sweep_mode == "mask_rotation"
        else np.linspace(0.0, 2.0 * np.pi * float(args.phase_cycles), int(args.phase_step), endpoint=True)
    )

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

    rows: list[dict[str, float | int]] = []
    summary_panels: list[dict[str, object]] = []
    summary_labels: list[str] = []
    for flux_ratio in flux_ratio_vals:
        flux_sim_local = dict(sim_local)
        flux_sim_local["companion_flux_ratio"] = float(flux_ratio)
        sample_rows, _best_entry, panels = _evaluate_best_roi_for_planet_center(
            planet_center=planet_center,
            roi_sizes=roi_sizes,
            region_shape_name=normalize_region_shape(args.region_shape),
            sim_local=flux_sim_local,
            phase_offsets=phase_offsets,
            sl16=sl16,
            half16=half16,
            xx16=xx16,
            yy16=yy16,
            incoherence_map_mode=incoherence_map_mode,
            collect_panels=True,
            phase_sweep_mode=phase_sweep_mode,
        )
        rows.extend(sample_rows)
        if panels:
            summary_panels.append(panels[0])
            summary_labels.append(f"flux={float(flux_ratio):.3e}")

    out_csv = os.path.join(
        sweep_output_dir,
        "planet_flux_ratio_map_sweep_table_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.csv",
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "planet_flux_ratio",
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
    out_png = os.path.join(
        sweep_output_dir,
        "planet_flux_ratio_map_sweep_summary_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_map_panel_summary_png(
        output_path=out_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title=(
            "Planet Brightness Sweep: Incoherence and Coherence Maps\n"
            f"fixed position=({planet_center[0]:+.2f}, {planet_center[1]:+.2f}) λ/D"
        ),
    )
    out_lyot_png = os.path.join(
        sweep_output_dir,
        "planet_flux_ratio_map_sweep_lyot_plane_phase_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_lyot_plane_phase_grid_png(
        output_path=out_lyot_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title=(
            "Planet Brightness Sweep: Lyot Plane by Phase Modulation\n"
            f"fixed position=({planet_center[0]:+.2f}, {planet_center[1]:+.2f}) λ/D"
        ),
    )
    out_lyot_phase_png = os.path.join(
        sweep_output_dir,
        "planet_flux_ratio_map_sweep_lyot_plane_field_phase_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_lyot_plane_field_phase_grid_png(
        output_path=out_lyot_phase_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title=(
            "Planet Brightness Sweep: Lyot Plane Field Phase by Phase Modulation\n"
            f"fixed position=({planet_center[0]:+.2f}, {planet_center[1]:+.2f}) λ/D"
        ),
    )
    out_focal_png = os.path.join(
        sweep_output_dir,
        "planet_flux_ratio_map_sweep_focal_plane_phase_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_focal_plane_phase_grid_png(
        output_path=out_focal_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title=(
            "Planet Brightness Sweep: Focal Plane by Phase Modulation\n"
            f"fixed position=({planet_center[0]:+.2f}, {planet_center[1]:+.2f}) λ/D"
        ),
    )
    out_focal_field_phase_png = os.path.join(
        sweep_output_dir,
        "planet_flux_ratio_map_sweep_focal_plane_field_phase_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_focal_plane_field_phase_grid_png(
        output_path=out_focal_field_phase_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title=(
            "Planet Brightness Sweep: Focal Plane Field Phase by Phase Modulation\n"
            f"fixed position=({planet_center[0]:+.2f}, {planet_center[1]:+.2f}) λ/D"
        ),
    )
    out_focal_phase_shift_png = os.path.join(
        sweep_output_dir,
        "planet_flux_ratio_map_sweep_focal_plane_phase_shift_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_focal_plane_phase_shift_grid_png(
        output_path=out_focal_phase_shift_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title=(
            "Planet Brightness Sweep: Applied Focal-Plane Phase Shift\n"
            f"fixed position=({planet_center[0]:+.2f}, {planet_center[1]:+.2f}) λ/D"
        ),
    )
    print(f"Saved planet-flux-ratio sweep table: {out_csv}")
    print(f"Saved planet-flux-ratio sweep summary PNG: {out_png}")
    print(f"Saved planet-flux-ratio sweep Lyot-plane intensity grid: {out_lyot_png}")
    if os.path.exists(out_lyot_phase_png):
        print(f"Saved planet-flux-ratio sweep Lyot-plane field-phase grid: {out_lyot_phase_png}")
    if os.path.exists(out_focal_png):
        print(f"Saved planet-flux-ratio sweep focal-plane phase grid: {out_focal_png}")
    if os.path.exists(out_focal_field_phase_png):
        print(f"Saved planet-flux-ratio sweep focal-plane field-phase grid: {out_focal_field_phase_png}")
    if os.path.exists(out_focal_phase_shift_png):
        print(f"Saved planet-flux-ratio sweep focal-plane phase-shift-map grid: {out_focal_phase_shift_png}")


def _run_mask_rotation_phase_step_sweep(
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
    phase_step_vals = _inclusive_int_range(
        args.phase_step_sweep_min,
        args.phase_step_sweep_max,
        args.phase_step_sweep_step,
    )
    roi_sizes = np.array([float(args.local_region_radius)], dtype=float)
    planet_center = (float(args.planet_offset_x_local), float(args.planet_offset_y_local))

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

    rows: list[dict[str, float | int]] = []
    summary_panels: list[dict[str, object]] = []
    summary_labels: list[str] = []
    for phase_step_count in phase_step_vals:
        phase_offsets = _mask_rotation_angles_rad(int(phase_step_count))
        sample_rows, _best_entry, panels = _evaluate_best_roi_for_planet_center(
            planet_center=planet_center,
            roi_sizes=roi_sizes,
            region_shape_name=normalize_region_shape(args.region_shape),
            sim_local=sim_local,
            phase_offsets=phase_offsets,
            sl16=sl16,
            half16=half16,
            xx16=xx16,
            yy16=yy16,
            incoherence_map_mode=incoherence_map_mode,
            collect_panels=True,
            phase_sweep_mode="mask_rotation",
        )
        for row in sample_rows:
            row_with_step = dict(row)
            row_with_step["phase_step_count"] = int(phase_step_count)
            rows.append(row_with_step)
        if panels:
            summary_panels.append(panels[0])
            summary_labels.append(f"steps={int(phase_step_count)}")

    out_csv = os.path.join(
        sweep_output_dir,
        "mask_rotation_phase_step_sweep_table_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.csv",
    )
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "phase_step_count",
                "planet_flux_ratio",
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
    out_png = os.path.join(
        sweep_output_dir,
        "mask_rotation_phase_step_sweep_summary_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_map_panel_summary_png(
        output_path=out_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title=(
            "Mask Rotation Step Sweep: Incoherence and Coherence Maps\n"
            f"fixed position=({planet_center[0]:+.2f}, {planet_center[1]:+.2f}) λ/D, "
            f"flux={float(args.planet_flux_ratio_local):.3e}"
        ),
    )
    out_lyot_phase_png = os.path.join(
        sweep_output_dir,
        "mask_rotation_phase_step_sweep_lyot_plane_field_phase_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_lyot_plane_field_phase_grid_png(
        output_path=out_lyot_phase_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title=(
            "Mask Rotation Step Sweep: Lyot Plane Field Phase by Phase Modulation\n"
            f"fixed position=({planet_center[0]:+.2f}, {planet_center[1]:+.2f}) λ/D, "
            f"flux={float(args.planet_flux_ratio_local):.3e}"
        ),
    )
    out_lyot_png = os.path.join(
        sweep_output_dir,
        "mask_rotation_phase_step_sweep_lyot_plane_phase_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_lyot_plane_phase_grid_png(
        output_path=out_lyot_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title=(
            "Mask Rotation Step Sweep: Lyot Plane by Phase Modulation\n"
            f"fixed position=({planet_center[0]:+.2f}, {planet_center[1]:+.2f}) λ/D, "
            f"flux={float(args.planet_flux_ratio_local):.3e}"
        ),
    )
    out_focal_png = os.path.join(
        sweep_output_dir,
        "mask_rotation_phase_step_sweep_focal_plane_phase_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_focal_plane_phase_grid_png(
        output_path=out_focal_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title=(
            "Mask Rotation Step Sweep: Focal Plane by Phase Modulation\n"
            f"fixed position=({planet_center[0]:+.2f}, {planet_center[1]:+.2f}) λ/D, "
            f"flux={float(args.planet_flux_ratio_local):.3e}"
        ),
    )
    out_focal_field_phase_png = os.path.join(
        sweep_output_dir,
        "mask_rotation_phase_step_sweep_focal_plane_field_phase_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_focal_plane_field_phase_grid_png(
        output_path=out_focal_field_phase_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title=(
            "Mask Rotation Step Sweep: Focal Plane Field Phase by Phase Modulation\n"
            f"fixed position=({planet_center[0]:+.2f}, {planet_center[1]:+.2f}) λ/D, "
            f"flux={float(args.planet_flux_ratio_local):.3e}"
        ),
    )
    out_focal_phase_shift_png = os.path.join(
        sweep_output_dir,
        "mask_rotation_phase_step_sweep_focal_plane_phase_shift_grid_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png",
    )
    _save_focal_plane_phase_shift_grid_png(
        output_path=out_focal_phase_shift_png,
        panels=summary_panels,
        panel_labels=summary_labels,
        figure_title=(
            "Mask Rotation Step Sweep: Applied Focal-Plane Phase Shift\n"
            f"fixed position=({planet_center[0]:+.2f}, {planet_center[1]:+.2f}) λ/D, "
            f"flux={float(args.planet_flux_ratio_local):.3e}"
        ),
    )
    print(f"Saved mask-rotation phase-step sweep table: {out_csv}")
    print(f"Saved mask-rotation phase-step sweep summary PNG: {out_png}")
    print(f"Saved mask-rotation phase-step sweep Lyot-plane intensity grid: {out_lyot_png}")
    if os.path.exists(out_lyot_phase_png):
        print(f"Saved mask-rotation phase-step sweep Lyot-plane field-phase grid: {out_lyot_phase_png}")
    if os.path.exists(out_focal_png):
        print(f"Saved mask-rotation phase-step sweep focal-plane phase grid: {out_focal_png}")
    if os.path.exists(out_focal_field_phase_png):
        print(f"Saved mask-rotation phase-step sweep focal-plane field-phase grid: {out_focal_field_phase_png}")
    if os.path.exists(out_focal_phase_shift_png):
        print(f"Saved mask-rotation phase-step sweep focal-plane phase-shift-map grid: {out_focal_phase_shift_png}")


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
    phase_sweep_mode = str(getattr(args, "phase_sweep_mode", "regional")).strip().lower()

    roi_min = float(args.roi_size_min)
    roi_max = float(args.roi_size_max)
    roi_step = float(args.roi_size_step)
    roi_sizes = _effective_roi_sizes_for_mode(
        np.arange(roi_min, roi_max + 0.5 * roi_step, roi_step, dtype=float),
        phase_sweep_mode,
    )
    if phase_sweep_mode == "mask_rotation":
        print(
            "[roi-size-sweep] mask_rotation mode active: "
            "ROI-size inputs are ignored; evaluating a single placeholder ROI sample."
        )
    roi_min_tag = _compact_float_tag(roi_min)
    roi_max_tag = _compact_float_tag(roi_max)
    roi_step_tag = _compact_float_tag(roi_step)
    roi_sweep_tag = f"_roi{roi_min_tag}-{roi_max_tag}-{roi_step_tag}"
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
                    if phase_sweep_mode == "mask_rotation":
                        phase_sim = CoronagraphSimulator(
                            **{
                                **sim_local,
                                "e_final_phase_offset": 0.0,
                                "phase_mask_rotation_rad": float(ph),
                                "focal_local_phase_offset": 0.0,
                                "focal_local_phase_centers_lamD": (),
                                "focal_local_phase_radius_lamD": 0.0,
                            }
                        )
                    elif _is_whole_focal_plane_phase_mode(phase_sweep_mode):
                        phase_sim = CoronagraphSimulator(
                            **{
                                **sim_local,
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
    rot_max_tag = _compact_float_tag(rot_max)
    rot_step_tag = _compact_float_tag(rot_step)
    rot_sweep_tag = f"_rot{rot_max_tag}-{rot_step_tag}"
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
                if _is_whole_focal_plane_phase_mode(str(args.phase_sweep_mode)):
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
            map_info = _cdi_build_incoherence_maps(
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


def run_cdi_planet_phase(
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
    phase_sweep_mode = str(args.phase_sweep_mode).strip().lower()
    mask_rotation_mode = phase_sweep_mode == "mask_rotation"
    modulation_sweep_dir = os.path.join(
        RESULTS_ROOT_DIR,
        _source_state_folder_name(
            bool(sim_kwargs.get("include_star", True)),
            bool(sim_kwargs.get("include_companion", True)),
            int(sim_kwargs.get("coherent_ring_speckle_count", 0)),
            float(sim_kwargs.get("coherent_ring_speckle_intensity", 1.0)),
        ),
        _modulation_sweep_folder_name(phase_sweep_mode),
    )
    if int(args.fov_count) < 1:
        raise ValueError("--fov-count must be >= 1.")
    if int(args.fov_centers_count) < 1:
        raise ValueError("--fov-centers-count must be >= 1.")
    if (
        not mask_rotation_mode
        and region_shape_name not in {"ring", "ring_of_circle"}
        and int(args.fov_count) > int(args.fov_centers_count)
    ):
        raise ValueError("--fov-count must be <= --fov-centers-count.")
    if float(args.local_region_radius) <= 0.0:
        raise ValueError("--local-region-radius must be > 0.")
    if int(args.phase_step) < 2:
        raise ValueError("--phase-step must be >= 2.")
    if float(args.planet_flux_ratio_local) < 0.0:
        raise ValueError("--planet-flux-ratio-local must be >= 0.")
    if float(args.lyot_reference_percent) < 0.0:
        raise ValueError("--lyot-reference-percent must be >= 0.")
    if sum(
        int(bool(getattr(args, name, False)))
        for name in (
            "roi_size_sweep",
            "planet_position_map_sweep",
            "planet_position_roi_size_sweep",
            "planet_flux_ratio_map_sweep",
            "mask_rotation_phase_step_sweep",
            "planet_position_brightness_sweep",
        )
    ) > 1:
        raise ValueError(
            "Enable at most one of --roi-size-sweep, --planet-position-map-sweep, "
            "--planet-position-roi-size-sweep, --planet-flux-ratio-map-sweep, "
            "--mask-rotation-phase-step-sweep, "
            "--planet-position-brightness-sweep."
        )
    if bool(getattr(args, "planet_position_map_sweep", False)):
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
        if str(getattr(args, "phase_mask_type", "")).strip().lower() in {"perfect_corongraph", "perfect_coronagraph"}:
            if float(args.lyot_reference_percent_sweep_min) < 0.0:
                raise ValueError("--lyot-reference-percent-sweep-min must be >= 0.")
            if float(args.lyot_reference_percent_sweep_max) < float(args.lyot_reference_percent_sweep_min):
                raise ValueError("--lyot-reference-percent-sweep-max must be >= --lyot-reference-percent-sweep-min.")
            if float(args.lyot_reference_percent_sweep_step) < 0.0 or (
                float(args.lyot_reference_percent_sweep_step) == 0.0
                and not np.isclose(
                    float(args.lyot_reference_percent_sweep_min),
                    float(args.lyot_reference_percent_sweep_max),
                )
            ):
                raise ValueError(
                    "--lyot-reference-percent-sweep-step must be > 0 unless "
                    "--lyot-reference-percent-sweep-min == --lyot-reference-percent-sweep-max."
                )
    if bool(getattr(args, "planet_position_brightness_sweep", False)):
        if float(args.local_region_radius) <= 0.0:
            raise ValueError("--local-region-radius must be > 0.")
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
        if float(args.planet_flux_ratio_sweep_min) < 0.0:
            raise ValueError("--planet-flux-ratio-sweep-min must be >= 0.")
        if float(args.planet_flux_ratio_sweep_max) < float(args.planet_flux_ratio_sweep_min):
            raise ValueError("--planet-flux-ratio-sweep-max must be >= --planet-flux-ratio-sweep-min.")
        if float(args.planet_flux_ratio_sweep_step) < 0.0 or (
            float(args.planet_flux_ratio_sweep_step) == 0.0
            and not np.isclose(float(args.planet_flux_ratio_sweep_min), float(args.planet_flux_ratio_sweep_max))
        ):
            raise ValueError("--planet-flux-ratio-sweep-step must be > 0 unless --planet-flux-ratio-sweep-min == --planet-flux-ratio-sweep-max.")
    if bool(getattr(args, "planet_flux_ratio_map_sweep", False)):
        if float(args.planet_flux_ratio_sweep_min) < 0.0:
            raise ValueError("--planet-flux-ratio-sweep-min must be >= 0.")
        if float(args.planet_flux_ratio_sweep_max) < float(args.planet_flux_ratio_sweep_min):
            raise ValueError("--planet-flux-ratio-sweep-max must be >= --planet-flux-ratio-sweep-min.")
        if float(args.planet_flux_ratio_sweep_step) < 0.0 or (
            float(args.planet_flux_ratio_sweep_step) == 0.0
            and not np.isclose(float(args.planet_flux_ratio_sweep_min), float(args.planet_flux_ratio_sweep_max))
        ):
            raise ValueError("--planet-flux-ratio-sweep-step must be > 0 unless --planet-flux-ratio-sweep-min == --planet-flux-ratio-sweep-max.")
    if bool(getattr(args, "mask_rotation_phase_step_sweep", False)):
        if not mask_rotation_mode:
            raise ValueError("--mask-rotation-phase-step-sweep requires --phase-sweep-mode mask_rotation.")
        if int(args.phase_step_sweep_min) < 2:
            raise ValueError("--phase-step-sweep-min must be >= 2.")
        if int(args.phase_step_sweep_max) < int(args.phase_step_sweep_min):
            raise ValueError("--phase-step-sweep-max must be >= --phase-step-sweep-min.")
        if int(args.phase_step_sweep_step) <= 0:
            raise ValueError("--phase-step-sweep-step must be > 0.")
    if bool(getattr(args, "ring_rotation_sweep", False)):
        if region_shape_name != "ring_of_circle":
            raise ValueError("--ring-rotation-sweep requires --region-shape ring_of_circle.")
        if float(args.ring_rotation_sweep_max) < 0.0 or float(args.ring_rotation_sweep_max) > 1.0:
            raise ValueError("--ring-rotation-sweep-max must be within [0, 1].")
        if float(args.ring_rotation_sweep_step) <= 0.0:
            raise ValueError("--ring-rotation-sweep-step must be > 0.")
    if mask_rotation_mode and bool(getattr(args, "ring_rotation_sweep", False)):
        raise ValueError("--ring-rotation-sweep is incompatible with --phase-sweep-mode mask_rotation.")

    local_kwargs = dict(sim_kwargs)
    cdi_secondary = (
        float(args.secondary_ratio_local)
        if args.secondary_ratio_local is not None
        else float(local_kwargs.get("secondary_diameter_ratio", 0.0))
    )
    if cdi_secondary <= 0.0:
        cdi_secondary = 0.25
    local_kwargs["secondary_diameter_ratio"] = float(cdi_secondary)

    phase_screen_folder_tag = _phase_screen_folder_tag(args, sim_kwargs)
    lyot_reference_folder_tag = _lyot_reference_folder_tag(args, sim_kwargs)
    fixed_center = (float(args.planet_offset_x_local), float(args.planet_offset_y_local))
    ring_radius_lamD = float(np.hypot(*fixed_center))
    initial_angle_rad = float(np.arctan2(fixed_center[1], fixed_center[0]))
    if mask_rotation_mode:
        centers = [fixed_center]
        effective_args.fov_count = 1
        effective_args.fov_centers_count = 1
        print(
            "Mask-rotation modulation: "
            f"{int(args.phase_step)} equally spaced angle step(s) over 360 deg."
        )
    elif region_shape_name == "ring_of_circle":
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
    contrast_ratio_dir = os.path.join(
        modulation_sweep_dir,
        _contrast_ratio_folder_name(float_filename_token(args.planet_flux_ratio_local, precision=6)),
    )
    cdi_planet_ratio_dir = os.path.join(
        contrast_ratio_dir,
        (
            "planet"
            f"_x_{float_filename_token(planet_center[0], precision=3)}"
            f"_y_{float_filename_token(planet_center[1], precision=3)}"
            f"_pov_r_{float_filename_token(effective_args.local_region_radius, precision=3)}"
            f"{phase_screen_folder_tag}"
            f"{lyot_reference_folder_tag}"
        ),
    )
    cdi_planet_ratio_dir_no_pov = os.path.join(
        contrast_ratio_dir,
        (
            "planet"
            f"_x_{float_filename_token(planet_center[0], precision=3)}"
            f"_y_{float_filename_token(planet_center[1], precision=3)}"
            f"{phase_screen_folder_tag}"
            f"{lyot_reference_folder_tag}"
        ),
    )

    cdi_phase_cycles = float(args.phase_cycles)
    if mask_rotation_mode:
        n_fov_groups = 1
        phase_steps_per_fov = int(args.phase_step)
        phase_steps_total = phase_steps_per_fov
        phase_offsets = _mask_rotation_angles_rad(phase_steps_total)
        print(
            "Mask rotation sampling: "
            f"{phase_steps_total} step(s), "
            f"{360.0 / float(phase_steps_total):.3f} deg per step."
        )
    else:
        n_fov_groups = int(np.ceil(float(fov_centers_count) / float(fov_count)))
        phase_steps_per_fov = int(args.phase_step)
        phase_steps_total = max(2, phase_steps_per_fov * max(n_fov_groups, 1))
        phase_offsets = np.linspace(
            0.0,
            2.0 * np.pi * cdi_phase_cycles * float(n_fov_groups),
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
    print(f"Using phase mask for cdi-planet-phase: {sim_local['phase_mask'].__class__.__name__}")
    single_fov_orbit_radius = float(ring_radius_lamD)

    if bool(getattr(args, "roi_size_sweep", False)):
        sweep_folder = os.path.join(
            cdi_planet_ratio_dir_no_pov,
            "roi_size_sweep",
            *_roi_shape_folder_parts_for_mode(region_shape_name, phase_sweep_mode),
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
    if bool(getattr(args, "planet_position_map_sweep", False)):
        sweep_folder = os.path.join(
            cdi_planet_ratio_dir_no_pov,
            "planet_position_map_sweep",
        )
        _run_planet_position_map_sweep(
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
        print("Planet-position map sweep enabled: skipped standard outputs.")
        return
    if bool(getattr(args, "planet_position_roi_size_sweep", False)):
        lyot_reference_sweep_min = float(args.lyot_reference_percent_sweep_min)
        lyot_reference_sweep_max = float(args.lyot_reference_percent_sweep_max)
        lyot_reference_sweep_step = float(args.lyot_reference_percent_sweep_step)
        if (
            bool(sim_local.get("perfect_coronagraph", False))
            and np.isclose(lyot_reference_sweep_min, 100.0)
            and np.isclose(lyot_reference_sweep_max, 100.0)
            and np.isclose(lyot_reference_sweep_step, 0.0)
            and not np.isclose(float(args.lyot_reference_percent), 100.0)
        ):
            lyot_reference_sweep_min = float(args.lyot_reference_percent)
            lyot_reference_sweep_max = float(args.lyot_reference_percent)
            lyot_reference_sweep_step = 0.0
        lyot_reference_sweep_folder_tag = ""
        if bool(sim_local.get("perfect_coronagraph", False)):
            lyot_reference_sweep_folder_tag = (
                "_lyot_ref_sweep_"
                f"{float_filename_token(lyot_reference_sweep_min, precision=3)}"
                f"_{float_filename_token(lyot_reference_sweep_max, precision=3)}"
                f"_{float_filename_token(lyot_reference_sweep_step, precision=3)}"
            )
        polar_root = os.path.join(
            contrast_ratio_dir,
            (
                "planet_position_polar"
                f"_r_{float_filename_token(float(args.planet_position_radius_min), precision=3)}"
                f"_{float_filename_token(float(args.planet_position_radius_max), precision=3)}"
                f"_{float_filename_token(float(args.planet_position_radius_step), precision=3)}"
                f"_theta_{float_filename_token(float(args.planet_position_theta_min_deg), precision=3)}"
                f"_{float_filename_token(float(args.planet_position_theta_max_deg), precision=3)}"
                f"_{float_filename_token(float(args.planet_position_theta_step_deg), precision=3)}"
                f"{phase_screen_folder_tag}"
                f"{lyot_reference_folder_tag}"
                f"{lyot_reference_sweep_folder_tag}"
            ),
        )
        sweep_folder = os.path.join(
            polar_root,
            "planet_position_roi_size_sweep",
            *_roi_shape_folder_parts_for_mode(region_shape_name, phase_sweep_mode),
        )
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
    if bool(getattr(args, "planet_flux_ratio_map_sweep", False)):
        contrast_sweep_dir = os.path.join(
            modulation_sweep_dir,
            (
                "contrast_ratio_sweep"
                f"_{float_filename_token(float(args.planet_flux_ratio_sweep_min), precision=6)}"
                f"_{float_filename_token(float(args.planet_flux_ratio_sweep_max), precision=6)}"
                f"_{float_filename_token(float(args.planet_flux_ratio_sweep_step), precision=6)}"
            ),
        )
        sweep_folder = os.path.join(
            contrast_sweep_dir,
            (
                "planet"
                f"_x_{float_filename_token(planet_center[0], precision=3)}"
                f"_y_{float_filename_token(planet_center[1], precision=3)}"
                f"{phase_screen_folder_tag}"
                f"{lyot_reference_folder_tag}"
            ),
            "planet_flux_ratio_map_sweep",
        )
        _run_planet_flux_ratio_map_sweep(
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
        print("Planet-flux-ratio map sweep enabled: skipped standard outputs.")
        return
    if bool(getattr(args, "mask_rotation_phase_step_sweep", False)):
        sweep_folder = os.path.join(
            cdi_planet_ratio_dir_no_pov,
            "mask_rotation_phase_step_sweep",
        )
        _run_mask_rotation_phase_step_sweep(
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
        print("Mask-rotation phase-step sweep enabled: skipped standard outputs.")
        return
    if bool(getattr(args, "planet_position_brightness_sweep", False)):
        contrast_sweep_dir = os.path.join(
            modulation_sweep_dir,
            (
                "contrast_ratio_sweep"
                f"_{float_filename_token(float(args.planet_flux_ratio_sweep_min), precision=6)}"
                f"_{float_filename_token(float(args.planet_flux_ratio_sweep_max), precision=6)}"
                f"_{float_filename_token(float(args.planet_flux_ratio_sweep_step), precision=6)}"
            ),
        )
        polar_root = os.path.join(
            contrast_sweep_dir,
            (
                "planet_position_brightness"
                f"_r_{float_filename_token(float(args.planet_position_radius_min), precision=3)}"
                f"_{float_filename_token(float(args.planet_position_radius_max), precision=3)}"
                f"_{float_filename_token(float(args.planet_position_radius_step), precision=3)}"
                f"_theta_{float_filename_token(float(args.planet_position_theta_min_deg), precision=3)}"
                f"_{float_filename_token(float(args.planet_position_theta_max_deg), precision=3)}"
                f"_{float_filename_token(float(args.planet_position_theta_step_deg), precision=3)}"
                f"{phase_screen_folder_tag}"
                f"{lyot_reference_folder_tag}"
            ),
        )
        sweep_folder = os.path.join(polar_root, "planet_position_brightness_sweep")
        brightness_args = argparse.Namespace(**vars(effective_args))
        brightness_args.roi_size_min = float(effective_args.local_region_radius)
        brightness_args.roi_size_max = float(effective_args.local_region_radius)
        brightness_args.roi_size_step = 0.0
        _run_planet_position_brightness_sweep(
            args=brightness_args,
            sim_local=sim_local,
            incoherence_map_mode=str(getattr(brightness_args, "incoherence_map_mode", "fft_band")),
            sweep_output_dir=sweep_folder,
            mask_output_tag=mask_output_tag,
            phase_cycles_tag=phase_cycles_tag,
            phase_sweep_mode_tag=phase_sweep_mode_tag,
            single_region_tag=single_region_tag,
            ghost_suffix=ghost_suffix,
        )
        print("Planet-position/brightness sweep enabled: skipped standard outputs.")
        return
    if bool(getattr(args, "ring_rotation_sweep", False)):
        sweep_folder = os.path.join(
            cdi_planet_ratio_dir_no_pov,
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

    os.makedirs(cdi_planet_ratio_dir, exist_ok=True)
    base = CoronagraphSimulator(**sim_local).run()
    n_fft = int(base["n_fft"])
    samp = float(base["focal_sampling"])
    pix = np.arange(n_fft, dtype=float)
    c = (n_fft - 1.0) / 2.0
    x_lamD = (pix - c) / samp
    y_lamD = (pix - c) / samp
    xx, yy = np.meshgrid(x_lamD, y_lamD)
    if mask_rotation_mode:
        roi_eval_radius_lamD = float(SNR_APERTURE_RADIUS_LAMD)
        roi_masks = [
            (xx - planet_center[0]) ** 2 + (yy - planet_center[1]) ** 2 <= roi_eval_radius_lamD ** 2
        ]
    elif region_shape_name == "ring":
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
    group_cycle_span = 2.0 * np.pi * max(float(cdi_phase_cycles), 1e-12)

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
        if _is_whole_focal_plane_phase_mode(phase_sweep_mode):
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
        elif mask_rotation_mode:
            phase_active_region_idx[i] = -1
            phase_sim = CoronagraphSimulator(
                **{
                    **sim_local,
                    "e_final_phase_offset": 0.0,
                    "phase_mask_rotation_rad": float(ph),
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
            prefix="cdi-planet-phase",
        )

    # Intentionally skip CDI FITS cube export to keep CDI outputs minimal.

    try:
        from matplotlib import cm

        gif_name = (
            f"{cdi_planet_ratio_dir}/cdi_planet_final_psf_16lamD_local_{float(effective_args.local_region_radius):.3f}_"
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

    if bool(sim_local.get("perfect_coronagraph", False)):
        try:
            subtracted_psf_path = _save_perfect_coronagraph_subtracted_final_psf(
                sim_local=sim_local,
                output_dir=cdi_planet_ratio_dir,
                local_region_radius_lamD=float(effective_args.local_region_radius),
                mask_output_tag=mask_output_tag,
                phase_cycles_tag=phase_cycles_tag,
                phase_sweep_mode_tag=phase_sweep_mode_tag,
                single_region_tag=single_region_tag,
                ghost_suffix=ghost_suffix,
            )
            print(
                "Saved perfect corongraph final PSF after reference subtraction: "
                f"{subtracted_psf_path}"
            )
        except Exception as exc:
            print(
                "Could not save perfect corongraph final PSF after reference subtraction: "
                f"{exc}"
            )
        try:
            star_only_subtracted_psf_path = _save_perfect_coronagraph_subtracted_final_psf_star_only(
                sim_local=sim_local,
                output_dir=cdi_planet_ratio_dir,
                local_region_radius_lamD=float(effective_args.local_region_radius),
                mask_output_tag=mask_output_tag,
                phase_cycles_tag=phase_cycles_tag,
                phase_sweep_mode_tag=phase_sweep_mode_tag,
                single_region_tag=single_region_tag,
                ghost_suffix=ghost_suffix,
            )
            print(
                "Saved perfect corongraph star-only final PSF after reference subtraction: "
                f"{star_only_subtracted_psf_path}"
            )
        except Exception as exc:
            print(
                "Could not save perfect corongraph star-only final PSF after reference subtraction: "
                f"{exc}"
            )
        try:
            no_phase_psf_path = _save_perfect_coronagraph_unmodulated_final_psf(
                sim_local=sim_local,
                output_dir=cdi_planet_ratio_dir,
                local_region_radius_lamD=float(effective_args.local_region_radius),
                mask_output_tag=mask_output_tag,
                phase_cycles_tag=phase_cycles_tag,
                phase_sweep_mode_tag=phase_sweep_mode_tag,
                single_region_tag=single_region_tag,
                ghost_suffix=ghost_suffix,
            )
            print(
                "Saved perfect corongraph final PSF without focal-plane phase offset: "
                f"{no_phase_psf_path}"
            )
        except Exception as exc:
            print(
                "Could not save perfect corongraph final PSF without focal-plane phase offset: "
                f"{exc}"
            )

    if region_shape_name == "ring_of_circle":
        try:
            ring_shift_gif = (
                f"{cdi_planet_ratio_dir}/cdi_planet_ring_of_circle_shift_"
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

    plot_info = plot_cdi_planet_phase_outputs(
        args=effective_args,
        base=base,
        centers=centers,
        planet_region_idx=planet_region_idx,
        cdi_planet_ratio_dir=cdi_planet_ratio_dir,
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
    print(f"Saved CDI overlay plot: {plot_info['out_overlay']}")
    print(f"Saved combined FFT+overlay plot: {plot_info['out_fft_overlay']}")
    print(f"Saved coherence/incoherence map plot: {plot_info['out_maps']}")
    if plot_info.get("out_selection_spectrum"):
        print(f"Saved frequency-selection FFT spectrum plot: {plot_info['out_selection_spectrum']}")
    if plot_info.get("out_maps_per_fov_pdf"):
        print(f"Saved per-active-FOV incoherence-map PDF: {plot_info['out_maps_per_fov_pdf']}")
    print(f"Incoherence map mode: {plot_info['incoherence_map_mode']}")
    print(
        "Incoherence-map planet SNR (planet-aperture mean / annulus-aperture std): "
        f"{plot_info['incoherence_planet_snr']:.6e}"
    )
    print(
        "  mean(planet region) = "
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
    print("Sampled center pixels (y, x) for CDI regions:")
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


run_coc_planet_phase = run_cdi_planet_phase
