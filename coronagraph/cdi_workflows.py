from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from .cdi_analysis import (
    SNR_ANNULUS_HALF_WIDTH_LAMD,
    SNR_APERTURE_RADIUS_LAMD,
    _compute_incoherence_map,
    _evaluate_best_roi_for_planet_center,
    _inclusive_float_range,
    _local_phase_region_kwargs,
    _planet_region_snr,
    _polar_to_cartesian_lamD,
    _summarize_roi_snr_trend,
    _theta_back_and_forth,
)
from .cdi_reports import (
    _save_grouped_roi_size_coherence_pdfs,
    _save_grouped_roi_size_incoherence_pdfs,
    _save_planet_position_snr_summary_pdf,
    _save_ring_rotation_probe_fft_page,
    _save_roi_size_coherence_pdf_for_planet_location,
    _save_roi_size_fft_spectra_pdf_for_planet_location,
    _save_roi_size_incoherence_pdf_for_planet_location,
    _save_roi_size_max_minus_coherence_pdf_for_planet_location,
)
from .plotting import _cdi_build_incoherence_maps
from .region_shapes import annulus_radii_from_width, build_touching_circle_ring, normalize_region_shape
from .simulator import CoronagraphSimulator


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
    poster_figure = bool(getattr(args, "plot_poster_figure", False))
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
    perfect_coronagraph = bool(sim_local.get("perfect_coronagraph", False))
    lyot_reference_sweep_min = float(getattr(args, "lyot_reference_percent_sweep_min", 100.0))
    lyot_reference_sweep_max = float(getattr(args, "lyot_reference_percent_sweep_max", 100.0))
    lyot_reference_sweep_step = float(getattr(args, "lyot_reference_percent_sweep_step", 0.0))
    if (
        perfect_coronagraph
        and np.isclose(lyot_reference_sweep_min, 100.0)
        and np.isclose(lyot_reference_sweep_max, 100.0)
        and np.isclose(lyot_reference_sweep_step, 0.0)
        and not np.isclose(float(getattr(args, "lyot_reference_percent", 100.0)), 100.0)
    ):
        lyot_reference_sweep_min = float(args.lyot_reference_percent)
        lyot_reference_sweep_max = float(args.lyot_reference_percent)
        lyot_reference_sweep_step = 0.0
    lyot_reference_percent_vals = (
        _inclusive_float_range(
            lyot_reference_sweep_min,
            lyot_reference_sweep_max,
            lyot_reference_sweep_step,
        )
        if perfect_coronagraph
        else np.array([float(getattr(args, "lyot_reference_percent", 100.0))], dtype=float)
    )

    phase_cycles = float(args.phase_cycles)
    phase_offsets = np.linspace(0.0, 2.0 * np.pi * phase_cycles, int(args.phase_step), endpoint=True)
    rows: list[dict[str, float | int]] = []
    location_panels: list[dict[str, object]] = []

    for lyot_reference_percent in lyot_reference_percent_vals:
        sweep_sim_local = dict(sim_local)
        if perfect_coronagraph:
            sweep_sim_local["lyot_reference_scale"] = float(lyot_reference_percent) / 100.0
        for theta_deg in theta_deg_vals:
            for radius_lamD in radius_vals:
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
                )
                for row in sample_rows:
                    if perfect_coronagraph:
                        row["lyot_reference_percent"] = float(lyot_reference_percent)
                rows.extend(sample_rows)
                if len(panels) > 0:
                    location_tag = (
                        (f"lr_{float(lyot_reference_percent):.3f}_" if perfect_coronagraph else "")
                        + f"planet_r_{float(radius_lamD):+.3f}_theta_{float(theta_deg):+.3f}"
                    ).replace(".", "p").replace("+", "")
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
                        "planet_position_roi_size_sweep_incoherence_maps_with_snr_24lamD_"
                        f"{location_tag}",
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
                        "planet_position_roi_size_sweep_coherence_maps_with_snr_24lamD_"
                        f"{location_tag}",
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
                    f"planet=(r={float(radius_lamD):+.3f}, theta={float(theta_deg):+.3f} deg) "
                    f"xy=({planet_center[0]:+.3f}, {planet_center[1]:+.3f}) "
                    f"best_snr={float(best_entry['snr']):.6e} "
                    f"best_roi={float(best_entry['resolved_roi_size_lamD']):.3f} λ/D "
                    f"roi_snr_trend={snr_trend}"
                )

    if not poster_figure:
        grouped_incoherence_pdfs = _save_grouped_roi_size_incoherence_pdfs(
            output_dir=sweep_output_dir,
            base_name=(
                "planet_position_roi_size_sweep_incoherence_maps_with_snr_24lamD_"
                f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{ghost_suffix}"
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
                f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{ghost_suffix}"
            ),
            region_shape_name=region_shape_name,
            location_panels=location_panels,
            extent=extent,
            poster_figure=poster_figure,
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
            f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{radius_tag}{theta_tag}{roi_tag}{ghost_suffix}.pdf",
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
    else:
        print("Poster figure mode enabled: emitted coherence and incoherence PDFs only.")


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
                ax_m.set_title(f"ROI Size Sweep Incoherence Map: {roi_r:.3f} λ/D ROI | 24x24 λ/D crop")
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
                    plt.Circle((0.0, 0.0), float(max(orbit_radius_lamD - SNR_ANNULUS_HALF_WIDTH_LAMD, 0.0)), fill=False, edgecolor="orange", linewidth=1.2, linestyle="--")
                )
                ax_m.add_patch(
                    plt.Circle((0.0, 0.0), float(orbit_radius_lamD + SNR_ANNULUS_HALF_WIDTH_LAMD), fill=False, edgecolor="orange", linewidth=1.2, linestyle="--", label="annulus")
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
                ax.plot(theta_arr[order], snr_arr[order], "-o", lw=1.3, ms=4.0, color=curve_color, label=f"ROI r={roi_r:.2f} λ/D")
        # poster block omitted in extraction; preserved behavior for standard outputs below
    with PdfPages(out_psf_pdf) as pdf:
        for roi_r in roi_sizes:
            fig_case, ax_case = plt.subplots(1, 1, figsize=(7.0, 6.2), constrained_layout=True)
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
            ax_case.set_title(f"Final PSF with All Regions (ROI r={roi_r:.2f} λ/D -> resolved {case_radius:.2f} λ/D, N={len(case_centers)})")
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
        ax.set_title("Planet-region SNR vs ROI Width (ring)" if region_shape_name == "ring" else "Planet-region SNR vs ROI Size (ring_of_circle)")
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
                if str(args.phase_sweep_mode).strip().lower() in {"global", "focal_plane"}:
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
                im_coh = ax_coh.imshow(coherence_map, origin="lower", cmap="magma", extent=coh_extent)
                im_inv = ax_inv.imshow(inverse_coherence_log_map, origin="lower", cmap="cividis", vmin=inv_vmin, vmax=inv_vmax, extent=coh_extent)
                for ax_map in axes_coh:
                    for j, (cx, cy) in enumerate(centers):
                        edge = "lime" if j == 0 else "cyan"
                        ax_map.add_patch(plt.Circle((cx, cy), region_radius_lamD, fill=False, edgecolor=edge, linewidth=1.1))
                    ax_map.plot([planet_center_lamD[0]], [planet_center_lamD[1]], marker="+", color="white", markersize=8, linestyle="None")
                    ax_map.set_xlabel("x [λ/D]")
                    ax_map.set_ylabel("y [λ/D]")
                ax_coh.set_title(f"Coherence |FFT({selected_target_freq:.3f})| / |FFT(0)|\nu={rot_u:.2f}")
                ax_inv.set_title(f"log10(1 / Coherence) |FFT(0)| / |FFT({selected_target_freq:.3f})|\nu={rot_u:.2f} | mode={incoherence_map_mode} (p5-p95)")
                fig_coh.colorbar(im_coh, ax=ax_coh, fraction=0.046, pad=0.04)
                cbar_inv = fig_coh.colorbar(im_inv, ax=ax_inv, fraction=0.046, pad=0.04)
                cbar_inv.set_label("log10(1 / coherence)")
                pdf_coh.savefig(fig_coh)
                plt.close(fig_coh)

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
