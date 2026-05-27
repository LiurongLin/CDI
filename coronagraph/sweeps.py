from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .masks import RoddierPhaseMask
from .simulator import CoronagraphSimulator

EXTRA_LOCAL_PHASE_CENTER_LAMD = (1.0, -1.0)


def _find_top_local_maxima(
    image: np.ndarray,
    focal_sampling: float,
    n_peaks: int = 4,
    r_min_lamD: float = 2.5,
    r_max_lamD: float = 8.0,
    min_separation_lamD: float = 1.2,
) -> list[dict]:
    # Rank candidate bright points in an annulus and keep isolated local maxima.
    # This gives robust region seeds near the speckle ring instead of the PSF core.
    n = image.shape[0]
    c = (n - 1) / 2.0
    y, x = np.indices(image.shape, dtype=float)
    x_lamD = (x - c) / focal_sampling
    y_lamD = (y - c) / focal_sampling
    r_lamD = np.sqrt(x_lamD**2 + y_lamD**2)

    ring = (r_lamD >= r_min_lamD) & (r_lamD <= r_max_lamD)
    ys, xs = np.where(ring)
    values = image[ys, xs]
    order = np.argsort(values)[::-1]

    min_sep2 = min_separation_lamD**2
    peaks: list[dict] = []
    for idx in order:
        yy = ys[idx]
        xx = xs[idx]
        v = image[yy, xx]
        if not np.isfinite(v):
            continue
        if yy <= 0 or yy >= n - 1 or xx <= 0 or xx >= n - 1:
            continue
        # Strict 3x3 local-maximum condition.
        patch = image[yy - 1 : yy + 2, xx - 1 : xx + 2]
        if v < np.max(patch):
            continue
        x0 = x_lamD[yy, xx]
        y0 = y_lamD[yy, xx]
        # Enforce minimum spacing so we do not pick multiple pixels from one lobe.
        too_close = False
        for p in peaks:
            dx = x0 - p["x_lamD"]
            dy = y0 - p["y_lamD"]
            if dx * dx + dy * dy < min_sep2:
                too_close = True
                break
        if too_close:
            continue
        peaks.append(
            {
                "value": float(v),
                "x_lamD": float(x0),
                "y_lamD": float(y0),
                "r_lamD": float(r_lamD[yy, xx]),
                "theta_rad": float(np.arctan2(y0, x0)),
            }
        )
        if len(peaks) >= n_peaks:
            break
    return peaks


def sweep_local_region_phase_peaks(
    sim_kwargs: dict,
    n_phase_samples: int = 101,
    phase_min_rad: float = 0.0,
    phase_max_rad: float = 2.0 * np.pi,
    region_radius_lamD: float = 0.6,
    outward_step_lamD: float = 2.0,
    keep_region_index: int = 0,
    align_to_reference_azimuth: bool = True,
    region_centers_lamD: list[tuple[float, float]] | None = None,
    phase_sweep_mode: str = "regional",
    region_shape: str = "circle",
    fov_count: int = 5,
    single_region_ring_radius_lamD: float | None = None,
    single_region_step_diameter_fraction: float = 0.25,
) -> dict:
    """
    Sweep local phase applied only in selected regions and track per-region peak intensities.
    Exactly five regions are used.

    Workflow:
    Auto mode:
    1) Detect four bright speckle regions from a baseline run.
    2) Keep one region fixed (keep_region_index), move the other three farther out radially.
    3) Add one extra fixed region at (1, -1) lambda/D.
    4) Apply phase shift only inside those circular regions at the first focal plane.
    5) Measure each region's peak intensity vs phase shift.

    Manual mode:
    - If region_centers_lamD is provided, it must contain exactly 5 centers and
      they are used exactly as provided.
    """
    if n_phase_samples < 2:
        raise ValueError("n_phase_samples must be >= 2.")
    if phase_max_rad <= phase_min_rad:
        raise ValueError("phase_max_rad must be greater than phase_min_rad.")
    if region_radius_lamD <= 0.0:
        raise ValueError("region_radius_lamD must be > 0.")
    if outward_step_lamD <= 0.0:
        raise ValueError("outward_step_lamD must be > 0.")
    if single_region_step_diameter_fraction <= 0.0:
        raise ValueError("single_region_step_diameter_fraction must be > 0.")
    if str(region_shape).strip().lower() != "circle":
        raise ValueError("region_shape currently supports only 'circle'.")
    if int(fov_count) < 1:
        raise ValueError("fov_count must be >= 1.")
    sweep_mode = str(phase_sweep_mode).strip().lower()
    if sweep_mode not in {"regional", "global"}:
        raise ValueError("phase_sweep_mode must be 'regional' or 'global'.")

    local_kwargs = dict(sim_kwargs)
    # Baseline run uses zero global phase offset; phase control is local only.
    local_kwargs["e_final_phase_offset"] = 0.0
    base = CoronagraphSimulator(**local_kwargs).run()

    detected = _find_top_local_maxima(
        base["final_psf_with_ghost"],
        focal_sampling=float(base["focal_sampling"]),
        n_peaks=4,
    )
    if len(detected) < 4:
        raise RuntimeError(
            f"Could not find 4 well-separated speckle peaks (found {len(detected)})."
        )

    manual_mode = region_centers_lamD is not None
    if manual_mode:
        expected_n = int(fov_count)
        if len(region_centers_lamD) != expected_n:
            raise ValueError(
                f"region_centers_lamD must contain exactly {expected_n} (x,y) center(s)."
            )
        moved_centers = [(float(x), float(y)) for x, y in region_centers_lamD]
    else:
        if not (0 <= keep_region_index < 4):
            raise ValueError("keep_region_index must be in [0, 3].")

        # Keep one detected region unchanged. The other three are moved outward.
        ref = detected[keep_region_index]
        moved_centers = [(ref["x_lamD"], ref["y_lamD"])]
        others = [p for i, p in enumerate(detected) if i != keep_region_index]
        others.sort(key=lambda p: p["r_lamD"])

        # Optional geometry mode: align all moved regions on the kept-region azimuth.
        ref_theta = ref["theta_rad"]
        for j, p in enumerate(others, start=1):
            theta = ref_theta if align_to_reference_azimuth else p["theta_rad"]
            new_r = max(float(p["r_lamD"]), float(ref["r_lamD"]) + j * outward_step_lamD)
            moved_centers.append((new_r * np.cos(theta), new_r * np.sin(theta)))

    if not manual_mode:
        moved_centers.append(EXTRA_LOCAL_PHASE_CENTER_LAMD)

    # Match requested FOV count.
    if len(moved_centers) > int(fov_count):
        moved_centers = moved_centers[: int(fov_count)]
    elif len(moved_centers) < int(fov_count):
        x0, y0 = moved_centers[0]
        start_theta = float(np.arctan2(y0, x0))
        ring_r = (
            float(single_region_ring_radius_lamD)
            if single_region_ring_radius_lamD is not None
            else float(np.hypot(x0, y0))
        )
        if ring_r <= 0.0:
            raise ValueError("single_region_ring_radius_lamD must be > 0.")
        step_lamD = float(single_region_step_diameter_fraction) * 2.0 * float(region_radius_lamD)
        dtheta = step_lamD / ring_r
        if dtheta <= 0.0:
            raise ValueError("Computed angular step must be > 0.")
        extra_needed = int(fov_count) - len(moved_centers)
        for k in range(extra_needed):
            th = start_theta + (k + 1) * dtheta
            moved_centers.append((ring_r * np.cos(th), ring_r * np.sin(th)))

    # Build lambda/D coordinate grids for region-based evaluation masks.
    n = int(base["n_fft"])
    c = (n - 1) / 2.0
    pix = np.arange(n, dtype=float)
    x_lamD = (pix - c) / float(base["focal_sampling"])
    y_lamD = (pix - c) / float(base["focal_sampling"])
    xx_lamD, yy_lamD = np.meshgrid(x_lamD, y_lamD)
    region_masks = [
        (xx_lamD - xc) ** 2 + (yy_lamD - yc) ** 2 <= region_radius_lamD**2
        for xc, yc in moved_centers
    ]

    phase_offsets = np.linspace(float(phase_min_rad), float(phase_max_rad), n_phase_samples)
    region_peak_curves = np.zeros((len(region_masks), n_phase_samples), dtype=float)

    for i, phase in enumerate(phase_offsets):
        phase_kwargs = dict(local_kwargs)
        if sweep_mode == "global":
            # Global phase injection using e_final_phase_offset.
            phase_kwargs["e_final_phase_offset"] = float(phase)
            phase_kwargs["focal_local_phase_offset"] = 0.0
            phase_kwargs["focal_local_phase_centers_lamD"] = ()
            phase_kwargs["focal_local_phase_radius_lamD"] = 0.0
            result = CoronagraphSimulator(**phase_kwargs).run()
            i_final = result["final_psf_with_ghost"]
            for k, reg in enumerate(region_masks):
                region_peak_curves[k, i] = float(np.max(i_final[reg]))
        else:
            # Local phase injection at first focal plane.
            phase_kwargs["focal_local_phase_offset"] = float(phase)
            phase_kwargs["focal_local_phase_centers_lamD"] = tuple(moved_centers)
            phase_kwargs["focal_local_phase_radius_lamD"] = float(region_radius_lamD)
            phase_kwargs["e_final_phase_offset"] = 0.0
            result = CoronagraphSimulator(**phase_kwargs).run()
            i_final = result["final_psf_with_ghost"]
            for k, reg in enumerate(region_masks):
                region_peak_curves[k, i] = float(np.max(i_final[reg]))

    return {
        "phase_offsets_rad": phase_offsets,
        "region_peak_curves": region_peak_curves,
        "region_centers_lamD": np.array(moved_centers, dtype=float),
        "region_radius_lamD": float(region_radius_lamD),
        "detected_peak_positions_lamD": np.array(
            [[p["x_lamD"], p["y_lamD"]] for p in detected], dtype=float
        ),
        "keep_region_index": int(keep_region_index),
        "align_to_reference_azimuth": bool(align_to_reference_azimuth),
        "outward_step_lamD": float(outward_step_lamD),
        "manual_centers_used": bool(manual_mode),
        "region_shape": str(region_shape).strip().lower(),
        "fov_count": int(len(moved_centers)),
        "single_region_ring_radius_lamD": (
            float(single_region_ring_radius_lamD)
            if single_region_ring_radius_lamD is not None
            else np.nan
        ),
        "single_region_step_diameter_fraction": float(single_region_step_diameter_fraction),
        "single_region_n_circles": int(len(moved_centers)),
        "phase_sweep_mode": sweep_mode,
        "phase_application_plane": (
            "first_focal_plane_local_regions" if sweep_mode == "regional" else "global_phase_offset"
        ),
        "extra_local_phase_center_lamD": np.array(EXTRA_LOCAL_PHASE_CENTER_LAMD, dtype=float),
    }


def sweep_roddier_radius_for_peak_match(
    sim_kwargs: dict,
    radius_min: float = 0.60,
    radius_max: float = 0.45,
    n_radius_samples: int = 41,
    phase_rad: float = np.pi,
    save_path: str | None = "roddier_radius_peak_match.png",
) -> dict:
    """
    Explore Roddier radius_lamD and find best match where coronagraphic and ghost peaks are equal.

    Returns a dictionary with sweep arrays and best radius summary.
    """
    if n_radius_samples < 2:
        raise ValueError("n_radius_samples must be >= 2.")
    if radius_max <= radius_min:
        raise ValueError("radius_max must be greater than radius_min.")

    radii = np.linspace(radius_min, radius_max, n_radius_samples)
    peak_coron = np.zeros_like(radii)
    peak_ghost = np.zeros_like(radii)
    peak_delta = np.zeros_like(radii)

    for i, radius in enumerate(radii):
        local_kwargs = dict(sim_kwargs)
        local_kwargs["phase_mask"] = RoddierPhaseMask(radius_lamD=float(radius), phase_rad=phase_rad)
        result = CoronagraphSimulator(**local_kwargs).run()
        peak_coron[i] = np.max(result["coronagraphic_psf"])
        peak_ghost[i] = np.max(result["ghost_psf"])
        peak_delta[i] = peak_coron[i] - peak_ghost[i]

    best_idx = int(np.argmin(np.abs(peak_delta)))
    best_radius = float(radii[best_idx])
    best_coron = float(peak_coron[best_idx])
    best_ghost = float(peak_ghost[best_idx])
    best_abs_diff = float(np.abs(peak_delta[best_idx]))
    best_rel_diff = best_abs_diff / max(best_ghost, 1e-20)

    if save_path is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

        axes[0].plot(radii, peak_coron, label="coronagraphic peak", color="tab:blue")
        axes[0].plot(radii, peak_ghost, label="ghost peak", color="tab:orange")
        axes[0].axvline(best_radius, color="black", ls="--", lw=1.2, alpha=0.8)
        axes[0].set_xlabel(r"Roddier radius $r$ [$\lambda/D$]")
        axes[0].set_ylabel("Peak intensity")
        axes[0].set_title("Peak Intensity vs Roddier Radius")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(radii, peak_delta, color="tab:green", label="coron - ghost")
        axes[1].axhline(0.0, color="black", lw=1.0, alpha=0.8)
        axes[1].axvline(best_radius, color="black", ls="--", lw=1.2, alpha=0.8)
        axes[1].set_xlabel(r"Roddier radius $r$ [$\lambda/D$]")
        axes[1].set_ylabel("Peak difference")
        axes[1].set_title("Peak Matching Error")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        backend = plt.get_backend().lower()
        if "agg" not in backend:
            plt.show()
        else:
            plt.close(fig)

    return {
        "radii_lamD": radii,
        "peak_coron": peak_coron,
        "peak_ghost": peak_ghost,
        "peak_delta": peak_delta,
        "best_index": best_idx,
        "best_radius_lamD": best_radius,
        "best_peak_coron": best_coron,
        "best_peak_ghost": best_ghost,
        "best_abs_difference": best_abs_diff,
        "best_relative_difference_to_ghost": best_rel_diff,
    }


def sweep_roddier_phase_for_peak_match(
    sim_kwargs: dict,
    radius_lamD: float = 0.53,
    phase_min_rad: float = 0.0,
    phase_max_rad: float = 2.0 * np.pi,
    n_phase_samples: int = 181,
    save_path: str | None = "roddier_phase_peak_match.png",
) -> dict:
    """
    Explore Roddier phase_rad at fixed radius and find best peak match to ghost peak.

    Returns a dictionary with sweep arrays and best phase summary.
    """
    if n_phase_samples < 2:
        raise ValueError("n_phase_samples must be >= 2.")
    if phase_max_rad <= phase_min_rad:
        raise ValueError("phase_max_rad must be greater than phase_min_rad.")

    phases = np.linspace(phase_min_rad, phase_max_rad, n_phase_samples)
    peak_coron = np.zeros_like(phases)
    peak_ghost = np.zeros_like(phases)
    peak_delta = np.zeros_like(phases)

    for i, phase_rad in enumerate(phases):
        local_kwargs = dict(sim_kwargs)
        local_kwargs["phase_mask"] = RoddierPhaseMask(
            radius_lamD=float(radius_lamD),
            phase_rad=float(phase_rad),
        )
        result = CoronagraphSimulator(**local_kwargs).run()
        peak_coron[i] = np.max(result["coronagraphic_psf"])
        peak_ghost[i] = np.max(result["ghost_psf"])
        peak_delta[i] = peak_coron[i] - peak_ghost[i]

    best_idx = int(np.argmin(np.abs(peak_delta)))
    best_phase = float(phases[best_idx])
    best_coron = float(peak_coron[best_idx])
    best_ghost = float(peak_ghost[best_idx])
    best_abs_diff = float(np.abs(peak_delta[best_idx]))
    best_rel_diff = best_abs_diff / max(best_ghost, 1e-20)

    if save_path is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)

        axes[0].plot(phases, peak_coron, label="coronagraphic peak", color="tab:blue")
        axes[0].plot(phases, peak_ghost, label="ghost peak", color="tab:orange")
        axes[0].axvline(best_phase, color="black", ls="--", lw=1.2, alpha=0.8)
        axes[0].set_xlabel(r"Roddier phase shift $\phi$ [rad]")
        axes[0].set_ylabel("Peak intensity")
        axes[0].set_title(f"Peak Intensity vs Roddier Phase (radius={radius_lamD:.3f} λ/D)")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(phases, peak_delta, color="tab:green", label="coron - ghost")
        axes[1].axhline(0.0, color="black", lw=1.0, alpha=0.8)
        axes[1].axvline(best_phase, color="black", ls="--", lw=1.2, alpha=0.8)
        axes[1].set_xlabel(r"Roddier phase shift $\phi$ [rad]")
        axes[1].set_ylabel("Peak difference")
        axes[1].set_title("Peak Matching Error")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        backend = plt.get_backend().lower()
        if "agg" not in backend:
            plt.show()
        else:
            plt.close(fig)

    return {
        "radius_lamD": float(radius_lamD),
        "phases_rad": phases,
        "peak_coron": peak_coron,
        "peak_ghost": peak_ghost,
        "peak_delta": peak_delta,
        "best_index": best_idx,
        "best_phase_rad": best_phase,
        "best_peak_coron": best_coron,
        "best_peak_ghost": best_ghost,
        "best_abs_difference": best_abs_diff,
        "best_relative_difference_to_ghost": best_rel_diff,
    }
