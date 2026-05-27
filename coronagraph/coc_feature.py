from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

from .cdi import circles_of_circles_center_sets
from .plotting import plot_coc_planet_phase_outputs
from .simulator import CoronagraphSimulator


def _run_roi_size_sweep_snr_vs_theta(
    args: argparse.Namespace,
    sim_local: dict,
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

    def _theta_back_and_forth(n: int, max_abs: float = np.pi) -> np.ndarray:
        if n <= 1:
            return np.array([0.0], dtype=float)
        half = max(1, n // 2)
        step = float(max_abs) / float(half)
        seq: list[float] = [0.0]
        k = 1
        while len(seq) < n:
            seq.append(-k * step)
            if len(seq) < n:
                seq.append(+k * step)
            k += 1
        arr = np.asarray(seq[:n], dtype=float)
        arr[np.abs(arr) < 1e-14] = 0.0
        return arr

    roi_dir = sweep_output_dir
    os.makedirs(roi_dir, exist_ok=True)

    roi_min = float(args.roi_size_min)
    roi_max = float(args.roi_size_max)
    roi_step = float(args.roi_size_step)
    roi_sizes = np.arange(roi_min, roi_max + 0.5 * roi_step, roi_step, dtype=float)
    roi_min_tag = f"{roi_min:.3f}".replace(".", "p")
    roi_max_tag = f"{roi_max:.3f}".replace(".", "p")
    roi_step_tag = f"{roi_step:.3f}".replace(".", "p")
    roi_sweep_tag = f"_rmin_{roi_min_tag}_rmax_{roi_max_tag}_rstep_{roi_step_tag}"
    fixed_planet_eval_radius_lamD = 0.5
    snr_eps = 1e-12

    phase_cycles = float(args.phase_cycles)
    phase_offsets = np.linspace(0.0, 2.0 * np.pi * phase_cycles, int(args.phase_step), endpoint=True)
    theta_samples = max(1, int(args.fov_centers_count))
    theta_rel = _theta_back_and_forth(theta_samples, max_abs=np.pi)
    print(f"ROI-size sweep theta samples [rad]: {theta_rel.tolist()}")

    base = CoronagraphSimulator(**sim_local).run()
    n_fft = int(base["n_fft"])
    samp = float(base["focal_sampling"])
    central_box_lamD = 16.0
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
        "roi_size_sweep_incoherence_maps_with_snr_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_sweep_tag}{ghost_suffix}.pdf",
    )
    out_table_csv = os.path.join(
        roi_dir,
        "roi_size_sweep_snr_theta_table_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{roi_sweep_tag}{ghost_suffix}.csv",
    )
    psf_crop_lamD = 8.0
    half_crop = int(psf_crop_lamD * samp)
    cc = n_fft // 2
    sl_crop = slice(cc - half_crop, cc + half_crop)
    table_rows: list[dict[str, float]] = []
    with PdfPages(out_incoh_pdf) as pdf_incoh:
        # Use high-contrast categorical colors for better curve distinguishability.
        base_cmap = plt.get_cmap("tab10" if len(roi_sizes) <= 10 else "tab20")
        curve_colors = [base_cmap(i % base_cmap.N) for i in range(max(len(roi_sizes), 1))]
        for roi_r in roi_sizes:
            color_idx = int(np.where(np.isclose(roi_sizes, roi_r))[0][0]) if len(roi_sizes) > 0 else 0
            curve_color = curve_colors[color_idx % len(curve_colors)]
            snr_vals: list[float] = []
            for th_rel in theta_rel:
                th = float(initial_angle_rad + th_rel)
                ctr = (float(orbit_radius_lamD * np.cos(th)), float(orbit_radius_lamD * np.sin(th)))

                stack = np.zeros((phase_offsets.size, 2 * half16, 2 * half16), dtype=float)
                for i, ph in enumerate(phase_offsets):
                    phase_sim = CoronagraphSimulator(
                        **{
                            **sim_local,
                            "e_final_phase_offset": 0.0,
                            "focal_local_phase_offset": float(ph),
                            "focal_local_phase_centers_lamD": (ctr,),
                            "focal_local_phase_radius_lamD": float(roi_r),
                        }
                    )
                    img = phase_sim.run()["final_psf_with_ghost"]
                    stack[i] = img[sl16, sl16]

                phase_series = np.asarray(phase_offsets, dtype=float)
                if phase_series.size > 2 and np.isclose(phase_series[0], 0.0) and np.isclose(phase_series[-1], float(phase_series.max())):
                    phase_series = phase_series[:-1]
                    stack = stack[:-1]
                dphi = float(np.mean(np.diff(phase_series)))
                freq = np.fft.fftfreq(stack.shape[0], d=dphi)
                fft_cube = np.fft.fft(stack, axis=0)
                band_a = (np.abs(freq) >= 0.0) & (np.abs(freq) <= 0.02)
                fft_a = np.zeros_like(fft_cube, dtype=np.complex128)
                fft_a[band_a] = fft_cube[band_a]
                incoh = np.mean(np.fft.ifft(fft_a, axis=0).real, axis=0)

                # Evaluate SNR at the fixed planet region across all FOV-center locations.
                planet_mask = (
                    (xx16 - float(planet_center_lamD[0])) ** 2
                    + (yy16 - float(planet_center_lamD[1])) ** 2
                    <= float(fixed_planet_eval_radius_lamD) ** 2
                )
                peak = float(np.max(incoh[planet_mask])) if np.any(planet_mask) else float("nan")
                rr = np.sqrt(xx16**2 + yy16**2)
                annulus_mask = (rr >= (orbit_radius_lamD - 0.5)) & (rr <= (orbit_radius_lamD + 0.5))
                ann_vals = incoh[annulus_mask]
                med = float(np.median(ann_vals)) if ann_vals.size > 0 else float("nan")
                if np.isfinite(peak) and np.isfinite(med):
                    med_safe = med if abs(med) > snr_eps else (snr_eps if med >= 0.0 else -snr_eps)
                    snr = float(peak / med_safe)
                else:
                    snr = float("nan")
                snr_vals.append(snr)
                table_rows.append(
                    {
                        "roi_radius_lamD": float(roi_r),
                        "theta_rel_rad": float(th_rel),
                        "active_center_x_lamD": float(ctr[0]),
                        "active_center_y_lamD": float(ctr[1]),
                        "planet_peak": float(peak),
                        "annulus_median": float(med),
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
                ax_m.add_patch(
                    plt.Circle(
                        (float(planet_center_lamD[0]), float(planet_center_lamD[1])),
                        float(fixed_planet_eval_radius_lamD),
                        fill=False,
                        edgecolor="white",
                        linewidth=1.6,
                        linestyle="-",
                        label="planet region (r=0.5 λ/D)",
                    )
                )
                ax_m.add_patch(
                    plt.Circle(
                        (0.0, 0.0),
                        float(max(orbit_radius_lamD - 0.5, 0.0)),
                        fill=False,
                        edgecolor="orange",
                        linewidth=1.2,
                        linestyle="--",
                    )
                )
                ax_m.add_patch(
                    plt.Circle(
                        (0.0, 0.0),
                        float(orbit_radius_lamD + 0.5),
                        fill=False,
                        edgecolor="orange",
                        linewidth=1.2,
                        linestyle="--",
                        label="annulus",
                    )
                )
                ax_m.plot([ctr[0]], [ctr[1]], marker="o", markersize=4.5, color="cyan", linestyle="None", label="active FOV center")
                th_rel_disp = 0.0 if abs(float(th_rel)) < 1e-10 else float(th_rel)
                ax_m.set_title(f"Incoherence Map | ROI r={roi_r:.2f} λ/D | theta={th_rel_disp:+.3f} rad")
                ax_m.set_xlabel("x [λ/D]")
                ax_m.set_ylabel("y [λ/D]")
                ax_m.legend(loc="upper right", fontsize=8)
                ax_m.text(
                    0.02,
                    0.98,
                    f"SNR={snr:.6e}\npeak={peak:.6e}\nannulus median={med:.6e}\nplanet eval r=0.5 λ/D",
                    transform=ax_m.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.60, edgecolor="none"),
                )
                pdf_incoh.savefig(fig_m)
                plt.close(fig_m)

            theta_arr = np.asarray(theta_rel, dtype=float)
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
    # Write all ROI-size PSF overlays into one PDF (one page per ROI radius).
    with PdfPages(out_psf_pdf) as pdf:
        for roi_r in roi_sizes:
            fig_case, ax_case = plt.subplots(1, 1, figsize=(7.0, 6.2), constrained_layout=True)
            # Re-run one representative final image for this ROI size at final phase of first group.
            ctr0 = centers_lamD[0] if len(centers_lamD) > 0 else (0.0, 0.0)
            phase_sim = CoronagraphSimulator(
                **{
                    **sim_local,
                    "e_final_phase_offset": 0.0,
                    "focal_local_phase_offset": float(phase_offsets[-1] if phase_offsets.size > 0 else 0.0),
                    "focal_local_phase_centers_lamD": tuple((float(cx), float(cy)) for cx, cy in centers_lamD),
                    "focal_local_phase_radius_lamD": float(roi_r),
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
            for j, (cx, cy) in enumerate(centers_lamD):
                edge = "lime" if j == 0 else "cyan"
                ax_case.add_patch(plt.Circle((cx, cy), float(roi_r), fill=False, edgecolor=edge, linewidth=1.4))
                ax_case.text(cx, cy, str(j), color="white", fontsize=7, ha="center", va="center")
            ax_case.plot([ctr0[0]], [ctr0[1]], marker="+", color="white", markersize=9, linestyle="None")
            ax_case.set_title(f"Final PSF with All Circle Regions (ROI r={roi_r:.2f} λ/D)")
            ax_case.set_xlabel("x [λ/D]")
            ax_case.set_ylabel("y [λ/D]")
            pdf.savefig(fig_case)
            plt.close(fig_case)

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
                "theta_rel_rad",
                "active_center_x_lamD",
                "active_center_y_lamD",
                "planet_peak",
                "annulus_median",
                "snr",
            ],
        )
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Saved ROI-size sweep SNR-vs-theta plot: {out_path}")
    print(f"Saved ROI-size sweep final-PSF region overlays (PDF): {out_psf_pdf}")
    print(f"Saved ROI-size sweep incoherence maps with SNR (PDF): {out_incoh_pdf}")
    print(f"Saved ROI-size sweep theta/SNR table (CSV): {out_table_csv}")


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
    if int(args.fov_count) < 1:
        raise ValueError("--fov-count must be >= 1.")
    if int(args.fov_centers_count) < 1:
        raise ValueError("--fov-centers-count must be >= 1.")
    if int(args.fov_count) > int(args.fov_centers_count):
        raise ValueError("--fov-count must be <= --fov-centers-count.")
    if float(args.local_region_radius) <= 0.0:
        raise ValueError("--local-region-radius must be > 0.")
    if int(args.phase_step) < 2:
        raise ValueError("--phase-step must be >= 2.")
    if float(args.planet_flux_ratio_local) < 0.0:
        raise ValueError("--planet-flux-ratio-local must be >= 0.")
    if bool(getattr(args, "roi_size_sweep", False)):
        if float(args.roi_size_min) <= 0.0:
            raise ValueError("--roi-size-min must be > 0.")
        if float(args.roi_size_max) < float(args.roi_size_min):
            raise ValueError("--roi-size-max must be >= roi-size-min.")
        if float(args.roi_size_step) <= 0.0:
            raise ValueError("--roi-size-step must be > 0.")

    local_kwargs = dict(sim_kwargs)
    coc_secondary = (
        float(args.secondary_ratio_local)
        if args.secondary_ratio_local is not None
        else float(local_kwargs.get("secondary_diameter_ratio", 0.0))
    )
    if coc_secondary <= 0.0:
        coc_secondary = 0.25
    local_kwargs["secondary_diameter_ratio"] = float(coc_secondary)
    if local_kwargs["spider_width_pixels"] <= 0.0:
        local_kwargs["spider_width_pixels"] = 0.25

    fixed_center = (float(args.planet_offset_x_local), float(args.planet_offset_y_local))
    ring_radius_lamD = float(np.hypot(*fixed_center))
    initial_angle_rad = float(np.arctan2(fixed_center[1], fixed_center[0]))
    fov_count = int(args.fov_count)
    fov_centers_count = int(args.fov_centers_count)
    if fov_centers_count == 1:
        centers = [fixed_center]
    else:
        centers = circles_of_circles_center_sets(
            ring_radius_lamD=ring_radius_lamD,
            circle_radius_lamD=float(args.local_region_radius),
            n_relocations=1,
            n_circles=fov_centers_count,
            initial_angle_rad=initial_angle_rad,
        )[0]
        centers = [
            (
                float(ring_radius_lamD * np.cos(np.arctan2(cy, cx))),
                float(ring_radius_lamD * np.sin(np.arctan2(cy, cx))),
            )
            for cx, cy in centers
        ]

    d2 = [(cx - fixed_center[0]) ** 2 + (cy - fixed_center[1]) ** 2 for cx, cy in centers]
    planet_region_idx = int(np.argmin(d2))
    planet_center = centers[planet_region_idx]
    coc_planet_ratio_dir = (
        f"coc_planet_ratio_{float_filename_token(args.planet_flux_ratio_local, precision=6)}"
        f"_planet_x_{float_filename_token(planet_center[0], precision=3)}"
        f"_y_{float_filename_token(planet_center[1], precision=3)}"
        f"_pov_r_{float_filename_token(args.local_region_radius, precision=3)}"
    )
    os.makedirs(coc_planet_ratio_dir, exist_ok=True)
    coc_planet_ratio_dir_no_pov = (
        f"coc_planet_ratio_{float_filename_token(args.planet_flux_ratio_local, precision=6)}"
        f"_planet_x_{float_filename_token(planet_center[0], precision=3)}"
        f"_y_{float_filename_token(planet_center[1], precision=3)}"
    )

    coc_phase_cycles = float(args.phase_cycles)
    n_fov_groups = int(np.ceil(float(fov_centers_count) / float(fov_count)))
    phase_offsets = np.linspace(
        0.0,
        2.0 * np.pi * coc_phase_cycles * float(n_fov_groups),
        int(args.phase_step),
        endpoint=True,
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
            args=args,
            sim_local=sim_local,
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

    base = CoronagraphSimulator(**sim_local).run()
    n_fft = int(base["n_fft"])
    samp = float(base["focal_sampling"])
    pix = np.arange(n_fft, dtype=float)
    c = (n_fft - 1.0) / 2.0
    x_lamD = (pix - c) / samp
    y_lamD = (pix - c) / samp
    xx, yy = np.meshgrid(x_lamD, y_lamD)
    roi_masks = [
        (xx - xc) ** 2 + (yy - yc) ** 2 <= float(args.local_region_radius) ** 2
        for xc, yc in centers
    ]

    centers_tuple = tuple((float(cx), float(cy)) for cx, cy in centers)
    # Single-FOV cycle stepping is on the star-centered ring that passes through
    # the planet location, so star-planet distance remains constant.
    single_fov_step_lamD = (
        float(args.single_region_step_diameter_fraction)
        * 2.0
        * float(args.local_region_radius)
    )
    if single_fov_orbit_radius > 0.0:
        single_fov_dtheta = single_fov_step_lamD / single_fov_orbit_radius
    else:
        single_fov_dtheta = 0.0
    group_cycle_span = 2.0 * np.pi * max(float(coc_phase_cycles), 1e-12)

    integrated_intensity = np.zeros((len(centers), phase_offsets.size), dtype=float)
    center_pixels_yx = np.zeros((len(centers), 2), dtype=int)
    center_pixel_intensity = np.zeros((len(centers), phase_offsets.size), dtype=float)
    phase_psf_cube = np.zeros((phase_offsets.size, n_fft, n_fft), dtype=np.float32)
    phase_active_region_idx = np.full(phase_offsets.size, -1, dtype=np.int16)
    central_box_lamD = 16.0
    half16 = int(0.5 * central_box_lamD * samp)
    cc16 = n_fft // 2
    sl16 = slice(cc16 - half16, cc16 + half16)
    central_phase_stack = np.zeros((phase_offsets.size, 2 * half16, 2 * half16), dtype=float)
    for j, (cx, cy) in enumerate(centers):
        x_idx = int(np.clip(np.round(c + cx * samp), 0, n_fft - 1))
        y_idx = int(np.clip(np.round(c + cy * samp), 0, n_fft - 1))
        center_pixels_yx[j] = np.array([y_idx, x_idx], dtype=int)

    start_time = time.perf_counter()
    for i, ph in enumerate(phase_offsets):
        if str(args.phase_sweep_mode).strip().lower() == "global":
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
                    "focal_local_phase_centers_lamD": tuple((float(cx), float(cy)) for cx, cy in active_centers),
                    "focal_local_phase_radius_lamD": float(args.local_region_radius),
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
            f"{coc_planet_ratio_dir}/coc_planet_final_psf_16lamD_local_{float(args.local_region_radius):.3f}_"
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
        saved_gif = False
        try:
            import imageio.v2 as imageio

            imageio.mimsave(gif_name, list(rgb8), duration=0.12, loop=0)
            saved_gif = True
        except Exception:
            try:
                from PIL import Image

                frames = [Image.fromarray(frame, mode="RGB") for frame in rgb8]
                if len(frames) > 0:
                    frames[0].save(
                        gif_name,
                        save_all=True,
                        append_images=frames[1:],
                        duration=120,
                        loop=0,
                    )
                    saved_gif = True
            except Exception:
                saved_gif = False

        if not saved_gif:
            raise RuntimeError(
                "GIF export requires either 'imageio' or 'Pillow' (PIL) to be installed."
            )
        print(f"Saved central 16x16 λ/D GIF: {gif_name}")
    except Exception as exc:
        print(f"Could not save central 16x16 λ/D GIF: {exc}")

    plot_info = plot_coc_planet_phase_outputs(
        args=args,
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
    if plot_info.get("out_maps_per_fov_pdf"):
        print(f"Saved per-active-FOV incoherence-map PDF: {plot_info['out_maps_per_fov_pdf']}")
    print(
        "Incoherence-map planet SNR (peak / annulus median): "
        f"{plot_info['incoherence_planet_snr']:.6e}"
    )
    print(
        "  peak(planet region) = "
        f"{plot_info['incoherence_planet_region_peak']:.6e}"
    )
    print(
        "  median(annulus, r={:.3f} λ/D, width={:.3f} λ/D) = {:.6e}".format(
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
    print(f"Central-field iFFT incoherence band [cycles/rad]: [{band_a_min:.3f}, {band_a_max:.3f}]")
    print(f"Central-field iFFT incoherence bins used [cycles/rad]: {plot_info['band_a_freqs']}")
    print(f"Central-field iFFT coherence band [cycles/rad]: [{band_b_min:.3f}, {band_b_max:.3f}]")
    print(f"Central-field iFFT coherence bins used [cycles/rad]: {plot_info['band_b_freqs']}")
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
    roi_min_tag = f"{roi_min:.3f}".replace(".", "p")
    roi_max_tag = f"{roi_max:.3f}".replace(".", "p")
    roi_step_tag = f"{roi_step:.3f}".replace(".", "p")
    roi_sweep_tag = f"_rmin_{roi_min_tag}_rmax_{roi_max_tag}_rstep_{roi_step_tag}"
