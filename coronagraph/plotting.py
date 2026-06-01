from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from .simulator import CoronagraphSimulator
from .sweeps import sweep_local_region_phase_peaks


def _theta_back_and_forth(n: int, max_abs: float = np.pi) -> np.ndarray:
    if n <= 1:
        return np.array([0.0], dtype=float)
    raw = np.arange(n, dtype=float) * (2.0 * np.pi / float(n))
    wrapped = ((raw + np.pi) % (2.0 * np.pi)) - np.pi
    order = sorted(
        range(n),
        key=lambda i: (abs(float(wrapped[i])), 0 if float(wrapped[i]) <= 0.0 else 1),
    )
    arr = wrapped[np.asarray(order, dtype=int)]
    arr[np.abs(arr) < 1e-14] = 0.0
    return arr


def plot_results(result: dict, save_path: str = "charge2_coronagraph_simulation.png") -> None:
    n_fft = result["n_fft"]
    samp = result["focal_sampling"]

    fig = plt.figure(figsize=(30, 10))
    gs = fig.add_gridspec(2, 5)
    ax_pupil = fig.add_subplot(gs[0, 0])
    ax_img0 = fig.add_subplot(gs[0, 1])
    ax_img1 = fig.add_subplot(gs[0, 2])
    ax_img2 = fig.add_subplot(gs[0, 3])
    ax_img3 = fig.add_subplot(gs[0, 4])
    ax_curve = fig.add_subplot(gs[1, :])

    crop_lamD = 20
    half = int(crop_lamD * samp)
    c = n_fft // 2
    sl = slice(c - half, c + half)

    im_pupil = ax_pupil.imshow(
        result["pupil"],
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    ax_pupil.set_title("Entrance Pupil", fontsize=16, fontweight="bold")
    ax_pupil.set_xlabel("Pixels")
    ax_pupil.set_ylabel("Pixels")
    fig.colorbar(im_pupil, ax=ax_pupil, fraction=0.046, pad=0.04)

    im_no_coron = ax_img0.imshow(
        np.log10(result["coronagraphic_psf"][sl, sl] + 1e-12),
        cmap="inferno",
        vmin=-8,
        vmax=0,
        extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD],
    )
    ax_img0.set_title("Coronagraphic PSF (No Ghost)", fontsize=16, fontweight="bold")
    ax_img0.set_xlabel("λ/D")
    ax_img0.set_ylabel("λ/D")
    fig.colorbar(im_no_coron, ax=ax_img0, fraction=0.046, pad=0.04)

    im_ghost = ax_img1.imshow(
        np.log10(result["ghost_psf"][sl, sl] + 1e-12),
        cmap="inferno",
        vmin=-8,
        vmax=0,
        extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD],
    )
    ax_img1.set_title("Ghost PSF (No Interference)", fontsize=16, fontweight="bold")
    ax_img1.set_xlabel("λ/D")
    ax_img1.set_ylabel("λ/D")
    fig.colorbar(im_ghost, ax=ax_img1, fraction=0.046, pad=0.04)

    interference = result["interference_term"][sl, sl]
    vmax_interf = np.percentile(np.abs(interference), 99.5)
    vmax_interf = max(vmax_interf, 1e-20)
    im_interference = ax_img2.imshow(
        interference,
        cmap="coolwarm",
        extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD],
    )
    ax_img2.set_title("Interference Term", fontsize=16, fontweight="bold")
    ax_img2.set_xlabel("λ/D")
    ax_img2.set_ylabel("λ/D")
    fig.colorbar(im_interference, ax=ax_img2, fraction=0.046, pad=0.04)

    im_final = ax_img3.imshow(
        np.log10(result["final_psf_with_ghost"][sl, sl] + 1e-12),
        cmap="inferno",
        vmin=-8,
        vmax=0,
        extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD],
    )
    ax_img3.set_title("Final Intensity (C+G+Interference)", fontsize=16, fontweight="bold")
    ax_img3.set_xlabel("λ/D")
    ax_img3.set_ylabel("λ/D")
    fig.colorbar(im_final, ax=ax_img3, fraction=0.046, pad=0.04)

    r = result["radial_r_lamD"]
    pc = result["radial_coron"]
    pghost = result["radial_ghost_only_no_interference"]
    pinterf = np.abs(result["radial_ghost_only_with_interference"] - pghost)
    pfinal = result["radial_with_ghost"]
    m = (r > 0) & (r <= 20)

    ax_curve.plot(r[m], pc[m], label="coronagraphic", color="tab:blue")
    ax_curve.plot(r[m], pghost[m], label="ghost", color="tab:orange")
    ax_curve.plot(r[m], np.maximum(pinterf[m], 1e-20), label="|interference|", color="tab:red")
    ax_curve.plot(r[m], pfinal[m], label="final (C+G+interference)", color="tab:green")

    y_data = np.concatenate([pc[m], pghost[m], np.maximum(pinterf[m], 1e-20), pfinal[m]])
    y_pos = y_data[np.isfinite(y_data) & (y_data > 0.0)]
    if y_pos.size > 0:
        y_min = 10.0 ** np.floor(np.log10(np.min(y_pos)))
        y_max = 10.0 ** np.ceil(np.log10(np.max(y_pos)))
        if np.isclose(y_min, y_max):
            y_min /= 10.0
            y_max *= 10.0
    else:
        y_min, y_max = 1e-12, 1.0

    ax_curve.set_yscale("log")
    ax_curve.set_ylim(y_min, y_max)
    ax_curve.set_xlim(0, 20)
    ax_curve.set_xlabel("Radius [λ/D]")
    ax_curve.set_ylabel("Normalized intensity")
    ax_curve.set_title("Contrast Curves", fontsize=16, fontweight="bold")
    ax_curve.grid(alpha=0.3)
    ax_curve.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)


def save_phase_mask_fits(result: dict, fits_path: str = "phase_mask.fits") -> None:
    """Save phase mask map (radians) to a FITS file."""
    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ImportError("astropy is required to save FITS files.") from exc

    phase_map = np.angle(result["mask"]).astype(np.float32)
    hdu = fits.PrimaryHDU(phase_map)
    hdu.header["BUNIT"] = "rad"
    hdu.header["MASK"] = result.get("phase_mask_name", "unknown")
    hdu.header["PM_SAMP"] = float(result.get("phase_mask_sampling", np.nan))
    hdu.writeto(fits_path, overwrite=True)


def plot_phase_offset_metrics(
    sim_kwargs: dict,
    n_phase_samples: int = 101,
    save_path: str = "phase_offset_peak_total_intensity.png",
) -> None:
    """Sweep e_final_phase_offset from 0 to 2*pi and plot peak/total intensity metrics."""
    phase_offsets = np.linspace(0.0, 2 * np.pi, n_phase_samples)

    peak_coron = np.zeros_like(phase_offsets)
    peak_ghost = np.zeros_like(phase_offsets)
    peak_interf = np.zeros_like(phase_offsets)

    total_coron = np.zeros_like(phase_offsets)
    total_ghost = np.zeros_like(phase_offsets)
    total_interf = np.zeros_like(phase_offsets)

    for i, phase_offset in enumerate(phase_offsets):
        local_kwargs = dict(sim_kwargs)
        local_kwargs["e_final_phase_offset"] = float(phase_offset)
        result = CoronagraphSimulator(**local_kwargs).run()

        coron = result["coronagraphic_psf"]
        ghost = result["ghost_psf"]
        interf = result["interference_term"]

        peak_coron[i] = np.max(coron)
        peak_ghost[i] = np.max(ghost)
        peak_interf[i] = np.max(np.abs(interf))

        total_coron[i] = np.sum(coron)
        total_ghost[i] = np.sum(ghost)
        total_interf[i] = np.sum(interf)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    ax0 = axes[0]
    ax0.plot(phase_offsets, peak_coron, label="coronagraphic", color="tab:blue")
    ax0.plot(phase_offsets, peak_ghost, label="ghost", color="tab:orange")
    ax0.plot(phase_offsets, peak_interf, label="|interference|", color="tab:red")
    ax0.set_xlabel(r"$e\_final\_phase\_offset$ [rad]")
    ax0.set_ylabel("Peak intensity")
    ax0.set_title("Peak vs Phase Offset")
    ax0.grid(alpha=0.3)
    ax0.legend()

    ax1 = axes[1]
    ax1.plot(phase_offsets, total_coron, label="coronagraphic", color="tab:blue")
    ax1.plot(phase_offsets, total_ghost, label="ghost", color="tab:orange")
    ax1.plot(phase_offsets, total_interf, label="interference (signed)", color="tab:red")
    ax1.set_xlabel(r"$e\_final\_phase\_offset$ [rad]")
    ax1.set_ylabel("Total intensity (sum over image)")
    ax1.set_title("Total vs Phase Offset")
    ax1.grid(alpha=0.3)
    ax1.legend()

    for ax in axes:
        ax.set_xticks([0.0, np.pi, 2 * np.pi])
        ax.set_xticklabels(["0", r"$\pi$", r"2*$\pi$"])

    fig.savefig(save_path, dpi=160, bbox_inches="tight")

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)


def plot_phase_offset_combined_metrics(
    sim_kwargs: dict,
    n_phase_samples: int = 101,
    save_path: str = "phase_offset_combined_peak_total_intensity.png",
) -> None:
    """Sweep e_final_phase_offset from 0 to 2*pi for combined (C+G+I) intensity metrics."""
    phase_offsets = np.linspace(0.0, 2 * np.pi, n_phase_samples)
    peak_combined = np.zeros_like(phase_offsets)
    total_combined = np.zeros_like(phase_offsets)

    for i, phase_offset in enumerate(phase_offsets):
        local_kwargs = dict(sim_kwargs)
        local_kwargs["e_final_phase_offset"] = float(phase_offset)
        result = CoronagraphSimulator(**local_kwargs).run()
        combined = result["final_psf_with_ghost"]
        peak_combined[i] = np.max(combined)
        total_combined[i] = np.sum(combined)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].plot(phase_offsets, peak_combined, color="black", lw=2.0)
    axes[0].set_title("Peak of Combined Intensity (C+G+I)")
    axes[0].set_xlabel(r"$e\_final\_phase\_offset$ [rad]")
    axes[0].set_ylabel("Peak intensity")
    axes[0].grid(alpha=0.3)

    axes[1].plot(phase_offsets, total_combined, color="black", lw=2.0)
    axes[1].set_title("Total of Combined Intensity (C+G+I)")
    axes[1].set_xlabel(r"$e\_final\_phase\_offset$ [rad]")
    axes[1].set_ylabel("Total intensity (sum over image)")
    axes[1].grid(alpha=0.3)

    for ax in axes:
        ax.set_xticks([0.0, np.pi, 2 * np.pi])
        ax.set_xticklabels(["0", r"$\pi$", r"2*$\pi$"])

    fig.savefig(save_path, dpi=160, bbox_inches="tight")

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)


def plot_local_region_phase_peak_metrics(
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
    save_path: str = "local_region_phase_peak_intensity.png",
) -> dict:
    """
    Plot per-region peak intensity vs localized phase shift for four selected regions.
    """
    # Generate region geometry + phase-response curves from the localized sweep.
    sweep = sweep_local_region_phase_peaks(
        sim_kwargs=sim_kwargs,
        n_phase_samples=n_phase_samples,
        phase_min_rad=phase_min_rad,
        phase_max_rad=phase_max_rad,
        region_radius_lamD=region_radius_lamD,
        outward_step_lamD=outward_step_lamD,
        keep_region_index=keep_region_index,
        align_to_reference_azimuth=align_to_reference_azimuth,
        region_centers_lamD=region_centers_lamD,
        phase_sweep_mode=phase_sweep_mode,
        region_shape=region_shape,
        fov_count=fov_count,
        single_region_ring_radius_lamD=single_region_ring_radius_lamD,
        single_region_step_diameter_fraction=single_region_step_diameter_fraction,
    )

    phase_offsets = sweep["phase_offsets_rad"]
    peaks = sweep["region_peak_curves"]
    centers = sweep["region_centers_lamD"]
    single_moving = int(sweep.get("fov_count", 0)) == 1

    # Recompute a baseline PSF only for the left-panel context image.
    local_kwargs = dict(sim_kwargs)
    local_kwargs["e_final_phase_offset"] = 0.0
    baseline = CoronagraphSimulator(**local_kwargs).run()
    psf = baseline["final_psf_with_ghost"]
    n_fft = baseline["n_fft"]
    samp = baseline["focal_sampling"]
    crop_lamD = 12.0
    half = int(crop_lamD * samp)
    c = n_fft // 2
    sl = slice(c - half, c + half)

    from matplotlib.patches import Circle

    fig, (ax_img, ax_curve) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    # Left panel: where the local phase is being applied.
    ax_img.imshow(
        np.log10(psf[sl, sl] + 1e-12),
        origin="lower",
        cmap="inferno",
        vmin=-8,
        vmax=0,
        extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD],
    )
    for i, (xc, yc) in enumerate(centers):
        ax_img.add_patch(
            Circle((xc, yc), radius=region_radius_lamD, fill=False, edgecolor="cyan", linewidth=1.6)
        )
        ax_img.text(xc, yc, str(i + 1), color="white", fontsize=8, ha="center", va="center")
    if str(phase_sweep_mode).strip().lower() == "global":
        ax_img.set_title("Selected Regions (Global Phase Sweep)")
    elif single_moving:
        ax_img.set_title("Selected Regions (Single-Region Moving-Center Sweep)")
    else:
        ax_img.set_title("Selected Regions (Phase Applied at First Focal Plane)")
    ax_img.set_xlabel("x [λ/D]")
    ax_img.set_ylabel("y [λ/D]")

    # Right panel: requested metric (peak intensity in each region vs phase).
    for i in range(peaks.shape[0]):
        xc, yc = centers[i]
        region_curve = peaks[i]

        ax_curve.plot(
          phase_offsets,
          region_curve,
          lw=2.0 if i == 0 else 1.8,
          label=f"region {i+1}: ({xc:+.2f}, {yc:+.2f}) λ/D",
        )
    ax_curve.set_xlabel("Local phase shift [rad]")
    ax_curve.set_ylabel("Peak intensity in region")
    if str(phase_sweep_mode).strip().lower() == "global":
        ax_curve.set_title("Peak Intensity vs Global Phase Shift")
    elif single_moving:
        ax_curve.set_title("Peak Intensity vs Single-Region Moving-Center Sweep")
    else:
        ax_curve.set_title("Peak Intensity vs Localized Phase Shift")
    ax_curve.grid(alpha=0.3)
    phase_mid = 0.5 * (float(phase_offsets[0]) + float(phase_offsets[-1]))
    ax_curve.set_xticks([float(phase_offsets[0]), phase_mid, float(phase_offsets[-1])])
    ax_curve.set_xticklabels(
        [
            f"{phase_offsets[0]/np.pi:.1f}π",
            f"{phase_mid/np.pi:.1f}π",
            f"{phase_offsets[-1]/np.pi:.1f}π",
        ]
    )
    ax_curve.legend(fontsize=9)

    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)
    return sweep


def plot_local_region0_peak_fft(
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
    save_path: str = "local_region0_peak_fft.png",
) -> dict:
    """
    Use all region center-pixel intensity sequences vs local phase and plot their Fourier spectra.
    """
    sweep = sweep_local_region_phase_peaks(
        sim_kwargs=sim_kwargs,
        n_phase_samples=n_phase_samples,
        phase_min_rad=phase_min_rad,
        phase_max_rad=phase_max_rad,
        region_radius_lamD=region_radius_lamD,
        outward_step_lamD=outward_step_lamD,
        keep_region_index=keep_region_index,
        align_to_reference_azimuth=align_to_reference_azimuth,
        region_centers_lamD=region_centers_lamD,
        phase_sweep_mode=phase_sweep_mode,
        region_shape=region_shape,
        fov_count=fov_count,
        single_region_ring_radius_lamD=single_region_ring_radius_lamD,
        single_region_step_diameter_fraction=single_region_step_diameter_fraction,
    )

    phase_offsets = np.asarray(sweep["phase_offsets_rad"], dtype=float)
    centers_lamD = np.asarray(sweep["region_centers_lamD"], dtype=float)
    single_moving = int(sweep.get("fov_count", 0)) == 1

    base_kwargs = dict(sim_kwargs)
    base_kwargs["e_final_phase_offset"] = 0.0
    base = CoronagraphSimulator(**base_kwargs).run()
    n_fft = int(base["n_fft"])
    samp = float(base["focal_sampling"])
    c = (n_fft - 1.0) / 2.0
    center_pixel_yx = np.zeros((centers_lamD.shape[0], 2), dtype=int)
    for r in range(centers_lamD.shape[0]):
        x_idx = int(np.clip(np.round(c + centers_lamD[r, 0] * samp), 0, n_fft - 1))
        y_idx = int(np.clip(np.round(c + centers_lamD[r, 1] * samp), 0, n_fft - 1))
        center_pixel_yx[r] = np.array([y_idx, x_idx], dtype=int)

    seqs = np.zeros((centers_lamD.shape[0], phase_offsets.size), dtype=float)
    for i, phase in enumerate(phase_offsets):
        frame_kwargs = dict(sim_kwargs)
        if str(phase_sweep_mode).strip().lower() == "global":
            frame_kwargs["e_final_phase_offset"] = float(phase)
            frame_kwargs["focal_local_phase_offset"] = 0.0
            frame_kwargs["focal_local_phase_centers_lamD"] = ()
            frame_kwargs["focal_local_phase_radius_lamD"] = 0.0
        else:
            frame_kwargs["e_final_phase_offset"] = 0.0
            frame_kwargs["focal_local_phase_offset"] = float(phase)
            frame_kwargs["focal_local_phase_centers_lamD"] = tuple(
                (float(x), float(y)) for x, y in sweep["region_centers_lamD"]
            )
            frame_kwargs["focal_local_phase_radius_lamD"] = float(region_radius_lamD)
        frame = CoronagraphSimulator(**frame_kwargs).run()["final_psf_with_ghost"]
        for r in range(centers_lamD.shape[0]):
            y_idx = int(center_pixel_yx[r, 0])
            x_idx = int(center_pixel_yx[r, 1])
            seqs[r, i] = float(frame[y_idx, x_idx])

    # Avoid duplicated endpoint sample (0 and 2*pi) for cleaner FFT bins.
    if (
        phase_offsets.size > 2
        and np.isclose(phase_offsets[0], 0.0)
        and np.isclose(phase_offsets[-1], 2.0 * np.pi)
    ):
        phase_offsets = phase_offsets[:-1]
        seqs = seqs[:, :-1]

    dphi = float(np.mean(np.diff(phase_offsets)))
    fft_vals = np.fft.fft(seqs, axis=1)
    freqs = np.fft.fftfreq(seqs.shape[1], d=dphi)
    amp = np.abs(fft_vals) / max(seqs.shape[1], 1)
    pos = freqs >= 0.0

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for r in range(seqs.shape[0]):
        ax0.plot(
            phase_offsets,
            seqs[r],
            lw=2.0 if r == 0 else 1.7,
            label=f"region {r+1}",
        )
    if str(phase_sweep_mode).strip().lower() == "global":
        ax0.set_title("Region Center-Pixel Intensity vs Global Phase")
    elif single_moving:
        ax0.set_title("Region Center-Pixel Intensity vs Single-Region Moving-Center Sweep")
    else:
        ax0.set_title("Region Center-Pixel Intensity vs Local Phase")
    ax0.set_xlabel("Local phase shift [rad]")
    ax0.set_ylabel("Center-pixel intensity")
    phase_mid = 0.5 * (float(phase_offsets[0]) + float(phase_offsets[-1]))
    ax0.set_xticks([float(phase_offsets[0]), phase_mid, float(phase_offsets[-1])])
    ax0.set_xticklabels(
        [
            f"{phase_offsets[0]/np.pi:.1f}π",
            f"{phase_mid/np.pi:.1f}π",
            f"{phase_offsets[-1]/np.pi:.1f}π",
        ]
    )
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=9)

    for r in range(amp.shape[0]):
        ax1.plot(freqs[pos], amp[r, pos], lw=2.0 if r == 0 else 1.7, label=f"region {r+1}")
    if single_moving:
        ax1.set_title("Fourier Transform Magnitude (Single-Region Moving-Center Sweep)")
    else:
        ax1.set_title("Fourier Transform Magnitude (All Region Sequences)")
    ax1.set_xlabel("Frequency [cycles/rad]")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)

    return {
        **sweep,
        "region_center_pixels_yx": center_pixel_yx,
        "region_phase_offsets_rad": phase_offsets,
        "region_center_pixel_sequences": seqs,
        "region_fft_frequency_cycles_per_rad": freqs,
        "region_fft_amplitudes": amp,
    }


def _coc_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = max(int(window), 1)
    w = min(w, max(int(x.size), 1))
    if w <= 1:
        return x.copy()
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(x, kernel, mode="same")


def _coc_top_peak_indices(y: np.ndarray, score: np.ndarray, n_keep: int = 3) -> np.ndarray:
    if y.size < 3:
        return np.array([], dtype=int)
    is_peak = np.zeros_like(y, dtype=bool)
    is_peak[1:-1] = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    cand = np.where(is_peak)[0]
    if cand.size == 0:
        return np.array([], dtype=int)
    order = np.argsort(score[cand])[::-1]
    keep = cand[order[: max(int(n_keep), 1)]]
    keep.sort()
    return keep


def _coc_fft_peak_filters(freqs: np.ndarray, mag: np.ndarray, n_keep: int = 3) -> dict:
    x = np.asarray(freqs, dtype=float)
    y = np.asarray(mag, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0.0)
    x = x[valid]
    y = y[valid]
    if y.size < 3:
        return {"freqs": x, "mag": y, "f1_idx": np.array([], dtype=int), "f2_idx": np.array([], dtype=int)}

    y_s = _coc_moving_average(y, window=5)
    baseline_s = _coc_moving_average(y_s, window=21)
    prom1 = y_s - baseline_s
    mad1 = np.median(np.abs(prom1 - np.median(prom1))) + 1e-20
    thr1 = np.median(prom1) + 3.0 * mad1
    idx1 = _coc_top_peak_indices(y_s, prom1, n_keep=max(n_keep * 2, 3))
    idx1 = idx1[prom1[idx1] > thr1]
    if y_s.size > 0:
        idx1 = np.unique(np.concatenate([np.array([0], dtype=int), idx1]))
    idx1 = idx1[:n_keep]

    baseline_b = _coc_moving_average(y, window=31)
    hp = y - baseline_b
    thr2 = np.percentile(hp, 90.0)
    idx2 = _coc_top_peak_indices(y, hp, n_keep=max(n_keep * 2, 3))
    idx2 = idx2[hp[idx2] > thr2]
    if y.size > 0:
        idx2 = np.unique(np.concatenate([np.array([0], dtype=int), idx2]))
    idx2 = idx2[:n_keep]
    return {"freqs": x, "mag": y, "f1_idx": idx1, "f2_idx": idx2}


def _coc_strongest_peak_in_band(freqs: np.ndarray, mag: np.ndarray, fmin: float, fmax: float) -> tuple[float, float] | None:
    x = np.asarray(freqs, dtype=float)
    y = np.asarray(mag, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x >= float(fmin)) & (x <= float(fmax))
    if not np.any(m):
        return None
    xx = x[m]
    yy = y[m]
    i = int(np.argmax(yy))
    return float(xx[i]), float(yy[i])


def _plot_coc_planet_phase_outputs_impl(
    args,
    base: dict,
    centers: list[tuple[float, float]],
    planet_region_idx: int,
    coc_planet_ratio_dir: str,
    mask_output_tag: str,
    phase_cycles_tag: str,
    phase_sweep_mode_tag: str,
    single_region_tag: str,
    ghost_suffix: str,
    phase_offsets: np.ndarray,
    roi_masks: list[np.ndarray],
    integrated_intensity: np.ndarray,
    central_phase_stack: np.ndarray,
) -> dict:
    from matplotlib.patches import Circle
    coc_phase_cycles = (
        float(args.coc_phase_cycles)
        if getattr(args, "coc_phase_cycles", None) is not None
        else float(args.local_phase_cycles)
    )

    n_fft = int(base["n_fft"])
    samp = float(base["focal_sampling"])
    ring_radius_lamD = float(max(np.hypot(cx, cy) for cx, cy in centers))
    crop_lamD = max(8.0, ring_radius_lamD + 2.0)
    half = int(crop_lamD * samp)
    cc = n_fft // 2
    sl = slice(cc - half, cc + half)

    integrated_intensity_norm = integrated_intensity / 1
    integrated_intensity_display = integrated_intensity_norm.copy()
    if int(args.fov_count) == 1 and str(args.phase_sweep_mode).strip().lower() != "global":
        for i, ph in enumerate(phase_offsets):
            active_idx = int(np.floor(float(ph) / (2.0 * np.pi) + 1e-12))
            active_idx = int(np.clip(active_idx, 0, len(centers) - 1))
            for j in range(len(centers)):
                if j != active_idx:
                    integrated_intensity_display[j, i] = np.nan

    central_box_lamD = 16.0
    phase_map_fft = phase_offsets.copy()
    central_stack_fft = central_phase_stack.copy()
    if (
        phase_map_fft.size > 2
        and np.isclose(phase_map_fft[0], 0.0)
        and np.isclose(phase_map_fft[-1], 2.0 * np.pi * coc_phase_cycles)
    ):
        phase_map_fft = phase_map_fft[:-1]
        central_stack_fft = central_stack_fft[:-1]
    dphi_map = float(np.mean(np.diff(phase_map_fft)))
    freq_bins = np.fft.fftfreq(central_stack_fft.shape[0], d=dphi_map)
    fft_cube = np.fft.fft(central_stack_fft, axis=0)

    band_a_min, band_a_max = 0.0, 0.02
    band_b_min, band_b_max = 0.14, 0.17
    abs_freq = np.abs(freq_bins)
    band_a_mask = (abs_freq >= band_a_min) & (abs_freq <= band_a_max)
    band_b_mask = (abs_freq >= band_b_min) & (abs_freq <= band_b_max)
    fft_band_a = np.zeros_like(fft_cube, dtype=np.complex128)
    fft_band_a[band_a_mask] = fft_cube[band_a_mask]
    recon_band_a = np.fft.ifft(fft_band_a, axis=0).real
    incoherence_map = np.mean(recon_band_a, axis=0)

    n_band_a_bins = int(np.count_nonzero(band_a_mask))
    if n_band_a_bins > 0 and int(np.count_nonzero(band_b_mask)) > 0:
        band_a_ref = np.sum(np.abs(fft_cube[band_a_mask]), axis=0) / float(n_band_a_bins)
        band_a_ref = np.maximum(band_a_ref, 1e-20)
        fft_ratio = np.zeros_like(fft_cube, dtype=np.complex128)
        fft_ratio[band_b_mask] = fft_cube[band_b_mask] / band_a_ref[None, :, :]
        recon_ratio = np.fft.ifft(fft_ratio, axis=0).real
        coherence_map = np.max(np.abs(recon_ratio), axis=0)
    else:
        coherence_map = np.zeros(fft_cube.shape[1:], dtype=float)

    fig_maps, axes_maps = plt.subplots(1, 3, figsize=(13.8, 4.3), constrained_layout=True)
    extent16 = [-0.5 * central_box_lamD, 0.5 * central_box_lamD, -0.5 * central_box_lamD, 0.5 * central_box_lamD]
    im0 = axes_maps[0].imshow(np.log10(np.mean(central_stack_fft, axis=0) + 1e-12), origin="lower", cmap="inferno", extent=extent16)
    axes_maps[0].set_title("Central 16x16 λ/D Mean Intensity")
    axes_maps[0].set_xlabel("x [λ/D]")
    axes_maps[0].set_ylabel("y [λ/D]")
    fig_maps.colorbar(im0, ax=axes_maps[0], fraction=0.046, pad=0.04)
    im1 = axes_maps[1].imshow(incoherence_map, origin="lower", cmap="viridis", extent=extent16)
    axes_maps[1].set_title("Incoherence (iFFT band: 0-0.02)")
    axes_maps[1].set_xlabel("x [λ/D]")
    axes_maps[1].set_ylabel("y [λ/D]")
    fig_maps.colorbar(im1, ax=axes_maps[1], fraction=0.046, pad=0.04)
    # Planet-region peak and annular median (1 λ/D width) on incoherence map.
    # SNR is defined as: peak(planet region) / median(annulus containing planet radius).
    n16y, n16x = incoherence_map.shape
    x16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, n16x, endpoint=False)
    y16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, n16y, endpoint=False)
    xx16, yy16 = np.meshgrid(x16, y16)
    planet_center = centers[planet_region_idx]
    planet_r_lamD = float(np.hypot(planet_center[0], planet_center[1]))
    planet_eval_radius_lamD = 0.5
    planet_region_mask_16 = (
        (xx16 - planet_center[0]) ** 2 + (yy16 - planet_center[1]) ** 2
        <= float(planet_eval_radius_lamD) ** 2
    )
    planet_peak_incoh = (
        float(np.max(incoherence_map[planet_region_mask_16]))
        if np.any(planet_region_mask_16)
        else float("nan")
    )
    ring_half_width = 0.5
    annulus_mask_16 = (
        (np.sqrt(xx16**2 + yy16**2) >= (planet_r_lamD - ring_half_width))
        & (np.sqrt(xx16**2 + yy16**2) <= (planet_r_lamD + ring_half_width))
    )
    annulus_vals = incoherence_map[annulus_mask_16]
    annulus_median_incoh = float(np.median(annulus_vals)) if annulus_vals.size > 0 else float("nan")
    incoh_snr = (
        float(planet_peak_incoh / annulus_median_incoh)
        if np.isfinite(planet_peak_incoh) and np.isfinite(annulus_median_incoh) and annulus_median_incoh != 0.0
        else float("nan")
    )
    # Visualize annulus on the incoherence map.
    axes_maps[1].add_patch(
        Circle((0.0, 0.0), planet_r_lamD - ring_half_width, fill=False, edgecolor="white", linewidth=1.1, linestyle="--")
    )
    axes_maps[1].add_patch(
        Circle((0.0, 0.0), planet_r_lamD + ring_half_width, fill=False, edgecolor="white", linewidth=1.1, linestyle="--")
    )
    axes_maps[1].add_patch(
        Circle((planet_center[0], planet_center[1]), float(planet_eval_radius_lamD), fill=False, edgecolor="yellow", linewidth=1.1)
    )
    axes_maps[1].text(
        0.02,
        0.98,
        (
            f"SNR={incoh_snr:.3e}\n"
            f"peak={planet_peak_incoh:.3e}\n"
            f"med={annulus_median_incoh:.3e}"
        ),
        transform=axes_maps[1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
    )
    im2 = axes_maps[2].imshow(coherence_map, origin="lower", cmap="magma", extent=extent16)
    axes_maps[2].set_title("Coherence (iFFT of band-B / band-A ratio)")
    axes_maps[2].set_xlabel("x [λ/D]")
    axes_maps[2].set_ylabel("y [λ/D]")
    fig_maps.colorbar(im2, ax=axes_maps[2], fraction=0.046, pad=0.04)
    out_maps = f"{coc_planet_ratio_dir}/coc_planet_coherence_incoherence_maps_16lamD_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png"
    fig_maps.savefig(out_maps, dpi=170, bbox_inches="tight")
    plt.close(fig_maps)

    out_maps_per_fov_pdf = None
    if bool(getattr(args, "build_map_per_fov", False)):
        out_maps_per_fov_pdf = (
            f"{coc_planet_ratio_dir}/coc_planet_incoherence_maps_per_fov_16lamD_"
            f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.pdf"
        )
        phase_series = np.asarray(phase_offsets, dtype=float)
        stack_series = np.asarray(central_phase_stack, dtype=float)
        if (
            phase_series.size > 2
            and np.isclose(phase_series[0], 0.0)
            and np.isclose(phase_series[-1], 2.0 * np.pi * coc_phase_cycles)
        ):
            phase_series = phase_series[:-1]
            stack_series = stack_series[:-1]

        def _active_intervals() -> list[tuple[str, np.ndarray]]:
            mode = str(args.phase_sweep_mode).strip().lower()
            if mode == "global":
                return [("global", np.ones(phase_series.size, dtype=bool))]
            if int(args.fov_count) == 1:
                n_positions = max(1, int(args.fov_centers_count))
                intervals: list[tuple[str, np.ndarray]] = []
                for pos_idx in range(n_positions):
                    lo = float(pos_idx) * 2.0 * np.pi
                    hi = float(pos_idx + 1) * 2.0 * np.pi
                    mask = (phase_series >= lo) & (
                        (phase_series <= hi) if pos_idx == n_positions - 1 else (phase_series < hi)
                    )
                    if np.any(mask):
                        intervals.append((f"fov pos {pos_idx + 1}", mask))
                return intervals
            return [(f"fov {j + 1}", np.ones(phase_series.size, dtype=bool)) for j in range(int(args.fov_count))]

        def _active_fov_center(label: str) -> tuple[float, float] | None:
            mode = str(args.phase_sweep_mode).strip().lower()
            if mode == "global":
                return None
            if int(args.fov_count) == 1:
                try:
                    pos_num = int(str(label).split()[-1])
                    pos_idx = max(pos_num - 1, 0)
                except Exception:
                    pos_idx = 0
                planet_center_local = centers[planet_region_idx]
                orbit_r = float(np.hypot(planet_center_local[0], planet_center_local[1]))
                base_theta_local = float(np.arctan2(planet_center_local[1], planet_center_local[0]))
                theta_rel = _theta_back_and_forth(max(1, int(args.fov_centers_count)), max_abs=np.pi)
                pos_idx = min(pos_idx, theta_rel.size - 1)
                th = base_theta_local + float(theta_rel[pos_idx])
                return float(orbit_r * np.cos(th)), float(orbit_r * np.sin(th))
            try:
                fov_num = int(str(label).split()[-1])
                fov_idx = max(min(fov_num - 1, len(centers) - 1), 0)
            except Exception:
                fov_idx = 0
            return float(centers[fov_idx][0]), float(centers[fov_idx][1])

        intervals = _active_intervals()
        with PdfPages(out_maps_per_fov_pdf) as pdf:
            for label, pm in intervals:
                idx = np.where(pm)[0]
                if idx.size < 3:
                    continue
                local_phase = phase_series[idx]
                local_stack = stack_series[idx]
                dphi_local = float(np.mean(np.diff(local_phase)))
                local_freq = np.fft.fftfreq(local_stack.shape[0], d=dphi_local)
                local_fft = np.fft.fft(local_stack, axis=0)
                local_abs = np.abs(local_freq)
                local_band_a = (local_abs >= band_a_min) & (local_abs <= band_a_max)
                if not np.any(local_band_a):
                    continue
                local_fft_a = np.zeros_like(local_fft, dtype=np.complex128)
                local_fft_a[local_band_a] = local_fft[local_band_a]
                local_recon_a = np.fft.ifft(local_fft_a, axis=0).real
                local_incoh = np.mean(local_recon_a, axis=0)

                planet_peak_local = (
                    float(np.max(local_incoh[planet_region_mask_16]))
                    if np.any(planet_region_mask_16)
                    else float("nan")
                )
                local_annulus_vals = local_incoh[annulus_mask_16]
                local_median = (
                    float(np.median(local_annulus_vals))
                    if local_annulus_vals.size > 0
                    else float("nan")
                )
                local_snr = (
                    float(planet_peak_local / local_median)
                    if np.isfinite(planet_peak_local) and np.isfinite(local_median) and local_median != 0.0
                    else float("nan")
                )

                fig_fov, ax_fov = plt.subplots(1, 1, figsize=(7.0, 6.0), constrained_layout=True)
                im_fov = ax_fov.imshow(local_incoh, origin="lower", cmap="viridis", extent=extent16)
                ax_fov.set_title(f"Incoherence Map per Active FOV: {label}")
                ax_fov.set_xlabel("x [λ/D]")
                ax_fov.set_ylabel("y [λ/D]")
                fig_fov.colorbar(im_fov, ax=ax_fov, fraction=0.046, pad=0.04)
                active_center = _active_fov_center(label)
                if active_center is not None:
                    ax_fov.add_patch(
                        Circle(
                            (active_center[0], active_center[1]),
                            float(args.local_region_radius),
                            fill=False,
                            edgecolor="red",
                            linewidth=1.6,
                        )
                    )
                    ax_fov.text(
                        active_center[0],
                        active_center[1],
                        "FOV",
                        color="red",
                        fontsize=8,
                        ha="center",
                        va="center",
                    )
                ax_fov.add_patch(
                    Circle((0.0, 0.0), planet_r_lamD - ring_half_width, fill=False, edgecolor="white", linewidth=1.1, linestyle="--")
                )
                ax_fov.add_patch(
                    Circle((0.0, 0.0), planet_r_lamD + ring_half_width, fill=False, edgecolor="white", linewidth=1.1, linestyle="--")
                )
                ax_fov.add_patch(
                    Circle((planet_center[0], planet_center[1]), float(planet_eval_radius_lamD), fill=False, edgecolor="yellow", linewidth=1.1)
                )
                ax_fov.text(
                    0.02,
                    0.98,
                    (
                        f"SNR={local_snr:.3e}\n"
                        f"peak={planet_peak_local:.3e}\n"
                        f"med={local_median:.3e}"
                    ),
                    transform=ax_fov.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
                )
                pdf.savefig(fig_fov)
                plt.close(fig_fov)

    single_interval_mode = int(args.fov_count) == 1 and str(args.phase_sweep_mode).strip().lower() != "global"
    out_curve = None

    phase_fft = phase_offsets.copy()
    norm_fft_seq = integrated_intensity_norm.copy()
    if (
        phase_fft.size > 2
        and np.isclose(phase_fft[0], 0.0)
        and np.isclose(phase_fft[-1], 2.0 * np.pi * coc_phase_cycles)
    ):
        phase_fft = phase_fft[:-1]
        norm_fft_seq = norm_fft_seq[:, :-1]
    dphi = float(np.mean(np.diff(phase_fft)))
    fft_vals = np.fft.fft(norm_fft_seq, axis=1)
    freqs = np.fft.fftfreq(norm_fft_seq.shape[1], d=dphi)
    amp = np.abs(fft_vals) / max(norm_fft_seq.shape[1], 1)
    pos = freqs >= 0.0

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.8, 5.0), constrained_layout=True)
    for j, (cx, cy) in enumerate(centers):
        lw = 2.3 if j == planet_region_idx else 1.6
        label = f"region {j}: ({cx:+.2f}, {cy:+.2f})" + (" [planet]" if j == planet_region_idx else "")
        if single_interval_mode:
            lo = float(j) * 2.0 * np.pi
            hi = float(j + 1) * 2.0 * np.pi
            pm = (phase_fft >= lo) & ((phase_fft <= hi) if j == len(centers) - 1 else (phase_fft < hi))
            ax0.plot(phase_fft[pm], integrated_intensity_display[j, : norm_fft_seq.shape[1]][pm], lw=lw, alpha=0.95, label=label)
        else:
            ax0.plot(phase_fft, integrated_intensity_display[j, : norm_fft_seq.shape[1]], lw=lw, alpha=0.95, label=label)
    ax0.set_title("CoC Normalized ROI Peak Intensity vs Local Phase")
    ax0.set_xlabel("Local phase shift [rad]")
    ax0.set_ylabel("ROI peak intensity / region mean")
    ax0.set_ylim(0.3, 1.6)
    phase_mid = 0.5 * (float(phase_fft[0]) + float(phase_fft[-1]))
    ax0.set_xticks([float(phase_fft[0]), phase_mid, float(phase_fft[-1])])
    ax0.set_xticklabels([f"{phase_fft[0]/np.pi:.1f}π", f"{phase_mid/np.pi:.1f}π", f"{phase_fft[-1]/np.pi:.1f}π"])
    ax0.grid(alpha=0.3)
    ax0.legend(fontsize=8, ncol=2)
    for j in range(len(centers)):
        lw = 2.3 if j == planet_region_idx else 1.6
        ax1.plot(freqs[pos], amp[j, pos], lw=lw, alpha=0.95, label=f"region {j}" + (" [planet]" if j == planet_region_idx else ""))
    planet_fft = _coc_fft_peak_filters(freqs[pos], amp[planet_region_idx, pos], n_keep=3)
    pfreq = planet_fft["freqs"]
    pmag = planet_fft["mag"]
    p1 = planet_fft["f1_idx"]
    p2 = planet_fft["f2_idx"]
    if p1.size > 0:
        ax1.scatter(pfreq[p1], pmag[p1], marker="o", s=44, facecolors="none", edgecolors="black", linewidths=1.3, label="filter-1 peaks", zorder=5)
    if p2.size > 0:
        ax1.scatter(pfreq[p2], pmag[p2], marker="x", s=44, color="magenta", linewidths=1.3, label="filter-2 peaks", zorder=5)
    ax1.axvspan(0.0, 0.025, color="gold", alpha=0.10, label="band A: 0-0.025")
    ax1.axvspan(0.120, 0.180, color="cyan", alpha=0.18, label="band B: 0.120-0.180")
    ax1.set_title("CoC Normalized ROI Intensity FFT Magnitude")
    ax1.set_xlabel("Frequency [cycles/rad]")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, ncol=2)
    out_fft = None
    plt.close(fig)

    single_fov_mode = int(args.fov_count) == 1 and str(args.phase_sweep_mode).strip().lower() != "global"
    fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(6.7, 6.0), constrained_layout=True)
    im = ax_overlay.imshow(np.log10(base["final_psf_with_ghost"][sl, sl] + 1e-12), origin="lower", cmap="inferno", vmin=-8, vmax=0, extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD])
    for j, (cx, cy) in enumerate(centers):
        col = "lime" if j == planet_region_idx else "cyan"
        ax_overlay.add_patch(Circle((cx, cy), float(args.local_region_radius), fill=False, edgecolor=col, linewidth=1.5))
        ax_overlay.text(cx, cy, str(j), color="white", fontsize=8, ha="center", va="center")
    ax_overlay.set_title("Final PSF with 8 Circles (planet region in green)")
    ax_overlay.set_xlabel("x [λ/D]")
    ax_overlay.set_ylabel("y [λ/D]")
    if single_fov_mode:
        planet_center = centers[planet_region_idx]
        orbit_r = float(np.hypot(planet_center[0], planet_center[1]))
        n_positions = max(1, int(args.fov_centers_count))
        base_theta = float(np.arctan2(planet_center[1], planet_center[0]))
        theta_rel = _theta_back_and_forth(n_positions, max_abs=np.pi)
        cycle_centers: list[tuple[float, float]] = []
        for th_rel in theta_rel:
            th = base_theta + float(th_rel)
            cx = float(orbit_r * np.cos(th))
            cy = float(orbit_r * np.sin(th))
            cycle_centers.append((cx, cy))
        for k, (cx, cy) in enumerate(cycle_centers, start=1):
            ax_overlay.add_patch(
                Circle(
                    (cx, cy),
                    float(args.local_region_radius),
                    fill=False,
                    edgecolor="white",
                    linewidth=1.0,
                    alpha=0.75,
                )
            )
            ax_overlay.text(cx, cy, str(k), color="yellow", fontsize=7, ha="center", va="center")
        ax_overlay.plot(
            [planet_center[0]],
            [planet_center[1]],
            marker="+",
            markersize=10,
            color="white",
            linestyle="None",
            label="planet center",
        )
        ax_overlay.legend(fontsize=8, loc="upper right")
    fig_overlay.colorbar(im, ax=ax_overlay, fraction=0.046, pad=0.04)
    out_overlay = f"{coc_planet_ratio_dir}/coc_planet_region_overlay_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png"
    fig_overlay.savefig(out_overlay, dpi=170, bbox_inches="tight")
    plt.close(fig_overlay)

    fig_combined, axes_combined = plt.subplots(1, 3, figsize=(16.2, 5.0), constrained_layout=True)
    ax0_c, ax1_c, ax2_c = axes_combined
    cc16 = n_fft // 2
    half16 = int(0.5 * central_box_lamD * samp)
    sl16 = slice(cc16 - half16, cc16 + half16)
    pos_c = freq_bins >= 0.0
    # Stable per-region colors (works for many regions without repeating quickly).
    n_color_items = max(int(np.ceil(coc_phase_cycles)), len(centers), 1)
    palette = plt.cm.hsv(np.linspace(0.0, 1.0, n_color_items, endpoint=False))
    if single_fov_mode and len(centers) == 1:
        # Build one trace per cycle-position for single-FOV mode.
        planet_center = centers[planet_region_idx]
        orbit_r = float(np.hypot(planet_center[0], planet_center[1]))
        n_positions = max(1, int(args.fov_centers_count))
        base_theta = float(np.arctan2(planet_center[1], planet_center[0]))
        theta_rel = _theta_back_and_forth(n_positions, max_abs=np.pi)
        cycle_centers: list[tuple[float, float]] = []
        for th_rel in theta_rel:
            th = base_theta + float(th_rel)
            cx = float(orbit_r * np.cos(th))
            cy = float(orbit_r * np.sin(th))
            cycle_centers.append((cx, cy))

        central_roi_phase_traces = np.full((n_positions, phase_map_fft.size), np.nan, dtype=float)
        cycle_order = list(range(len(cycle_centers)))
        if len(cycle_order) > 1:
            # Planet position is cycle 1 (index 0): draw it last/on top.
            cycle_order = cycle_order[1:] + [0]
        for j in cycle_order:
            cx, cy = cycle_centers[j]
            clr = palette[j % len(palette)]
            xx16, yy16 = np.meshgrid(
                np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, central_stack_fft.shape[2], endpoint=False),
                np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, central_stack_fft.shape[1], endpoint=False),
            )
            m16 = (xx16 - cx) ** 2 + (yy16 - cy) ** 2 <= float(args.local_region_radius) ** 2
            if not np.any(m16):
                continue
            # Full-sequence trace for this cycle-position (used by FFT).
            vals_all = np.max(central_stack_fft[:, m16], axis=1)
            central_roi_phase_traces[j, :] = vals_all

            # Intensity-vs-phase panel: display only the corresponding cycle interval.
            lo = float(2 * j) * 2.0 * np.pi
            hi = float(2 * j + 2) * 2.0 * np.pi
            pm = (phase_map_fft >= lo) & ((phase_map_fft <= hi) if j == n_positions - 1 else (phase_map_fft < hi))
            idx = np.where(pm)[0]
            if idx.size == 0:
                continue
            vals = vals_all[idx]
            is_planet_pos = (j == 0)
            ax0_c.plot(
                phase_map_fft[idx],
                vals,
                linestyle="-",
                linewidth=2.2 if is_planet_pos else 1.4,
                alpha=1.0 if is_planet_pos else 0.85,
                color=clr,
                zorder=6 if is_planet_pos else 4,
                label=f"fov pos {j+1}: ({cx:+.2f}, {cy:+.2f})",
            )

            # FFT panel: use the whole phase-shift process for this cycle-position trace.
            if vals_all.size > 2:
                fft_local = np.fft.fft(vals_all)
                freq_local = np.fft.fftfreq(vals_all.size, d=dphi_map)
                amp_local = np.abs(fft_local) / max(vals_all.size, 1)
                pos_local = freq_local >= 0.0
                ax1_c.plot(
                    freq_local[pos_local],
                    amp_local[pos_local],
                    linestyle="-",
                    marker="o",
                    markersize=4.2 if is_planet_pos else 2.8,
                    linewidth=2.2 if is_planet_pos else 1.2,
                    alpha=1.0 if is_planet_pos else 0.8,
                    color=clr,
                    zorder=10 if is_planet_pos else 4,
                    label=f"fov pos {j+1}",
                )
    else:
        central_roi_masks = [m[sl16, sl16] for m in roi_masks]
        central_roi_phase_traces = np.zeros((len(centers), central_stack_fft.shape[0]), dtype=float)
        for j, m16 in enumerate(central_roi_masks):
            central_roi_phase_traces[j] = np.max(central_stack_fft[:, m16], axis=1) if np.any(m16) else 0.0
        central_roi_fft = np.fft.fft(central_roi_phase_traces, axis=1)
        central_roi_amp = np.abs(central_roi_fft) / max(central_roi_phase_traces.shape[1], 1)
        for j, (cx, cy) in enumerate(centers):
            label = f"region {j}: ({cx:+.2f}, {cy:+.2f})" + (" [planet]" if j == planet_region_idx else "")
            clr = palette[j % len(palette)]
            if single_interval_mode:
                lo = float(j) * 2.0 * np.pi
                hi = float(j + 1) * 2.0 * np.pi
                pm = (phase_map_fft >= lo) & ((phase_map_fft <= hi) if j == len(centers) - 1 else (phase_map_fft < hi))
                ax0_c.plot(phase_map_fft[pm], central_roi_phase_traces[j, pm], linestyle="-", linewidth=1.6, alpha=0.95, color=clr, label=label)
            else:
                ax0_c.plot(phase_map_fft, central_roi_phase_traces[j], linestyle="-", linewidth=1.6, alpha=0.95, color=clr, label=label)
        order = [j for j in range(len(centers)) if j != planet_region_idx] + [planet_region_idx]
        for j in order:
            is_planet = j == planet_region_idx
            clr = palette[j % len(palette)]
            ax1_c.plot(
                freq_bins[pos_c],
                central_roi_amp[j, pos_c],
                linestyle="None",
                marker="o",
                markersize=3.0,
                alpha=0.95,
                color=clr,
                zorder=6 if is_planet else 4,
                label=f"region {j}" + (" [planet]" if is_planet else ""),
            )
    ax0_c.set_title("Central 16x16 ROI Peak Intensity vs Local Phase")
    ax0_c.set_xlabel("Local phase shift [rad]")
    ax0_c.set_ylabel("ROI peak intensity")
    phase_mid = 0.5 * (float(phase_map_fft[0]) + float(phase_map_fft[-1]))
    ax0_c.set_xticks([float(phase_map_fft[0]), phase_mid, float(phase_map_fft[-1])])
    ax0_c.set_xticklabels([f"{phase_map_fft[0]/np.pi:.1f}π", f"{phase_mid/np.pi:.1f}π", f"{phase_map_fft[-1]/np.pi:.1f}π"])
    ax0_c.grid(alpha=0.3)
    ax0_c.legend(
        fontsize=8,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    ax1_c.axvspan(0.0, 0.025, color="gold", alpha=0.10, label="band A: 0-0.025")
    ax1_c.axvspan(0.120, 0.180, color="cyan", alpha=0.18, label="band B: 0.120-0.180")
    ax1_c.set_title("Central 16x16 ROI Peak-Trace FFT Magnitude")
    ax1_c.set_xlabel("Frequency [cycles/rad]")
    ax1_c.set_ylabel("Amplitude")
    ax1_c.grid(alpha=0.3)
    ax1_c.legend(fontsize=8, ncol=1)
    im_combined = ax2_c.imshow(np.log10(base["final_psf_with_ghost"][sl, sl] + 1e-12), origin="lower", cmap="inferno", vmin=-8, vmax=0, extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD])
    for j, (cx, cy) in enumerate(centers):
        col = "lime" if j == planet_region_idx else "cyan"
        ax2_c.add_patch(Circle((cx, cy), float(args.local_region_radius), fill=False, edgecolor=col, linewidth=1.5))
        ax2_c.text(cx, cy, str(j), color="white", fontsize=8, ha="center", va="center")
    ax2_c.set_title("Final PSF with 8 Circles (planet region in green)")
    ax2_c.set_xlabel("x [λ/D]")
    ax2_c.set_ylabel("y [λ/D]")
    if single_fov_mode:
        planet_center = centers[planet_region_idx]
        orbit_r = float(np.hypot(planet_center[0], planet_center[1]))
        n_positions = max(1, int(args.fov_centers_count))
        base_theta = float(np.arctan2(planet_center[1], planet_center[0]))
        theta_rel = _theta_back_and_forth(n_positions, max_abs=np.pi)
        for th_rel in theta_rel:
            th = base_theta + float(th_rel)
            cx = float(orbit_r * np.cos(th))
            cy = float(orbit_r * np.sin(th))
            ax2_c.add_patch(
                Circle(
                    (cx, cy),
                    float(args.local_region_radius),
                    fill=False,
                    edgecolor="white",
                    linewidth=1.0,
                    alpha=0.7,
                )
            )
    fig_combined.colorbar(im_combined, ax=ax2_c, fraction=0.046, pad=0.04)
    out_fft_overlay = f"{coc_planet_ratio_dir}/coc_planet_normalized_roi_fft_local_{float(args.local_region_radius):.3f}_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}_with_overlay.png"
    fig_combined.savefig(out_fft_overlay, dpi=170, bbox_inches="tight")
    plt.close(fig_combined)

    band_a_peak = _coc_strongest_peak_in_band(freqs[pos], amp[planet_region_idx, pos], 0.0, 0.025)
    band_b_peak = _coc_strongest_peak_in_band(freqs[pos], amp[planet_region_idx, pos], 0.120, 0.180)
    return {
        "out_curve": out_curve,
        "out_fft": out_fft,
        "out_overlay": out_overlay,
        "out_fft_overlay": out_fft_overlay,
        "out_maps": out_maps,
        "out_maps_per_fov_pdf": out_maps_per_fov_pdf,
        "f1_freqs": [float(v) for v in pfreq[p1]] if p1.size > 0 else [],
        "f2_freqs": [float(v) for v in pfreq[p2]] if p2.size > 0 else [],
        "band_a_freqs": [float(v) for v in freq_bins[band_a_mask]],
        "band_b_freqs": [float(v) for v in freq_bins[band_b_mask]],
        "band_a_peak": band_a_peak,
        "band_b_peak": band_b_peak,
        "band_a_bounds": (band_a_min, band_a_max),
        "band_b_bounds": (band_b_min, band_b_max),
        "incoherence_planet_region_peak": planet_peak_incoh,
        "incoherence_annulus_median": annulus_median_incoh,
        "incoherence_planet_snr": incoh_snr,
        "incoherence_annulus_radius_lamD": planet_r_lamD,
        "incoherence_annulus_width_lamD": 1.0,
    }


def plot_coc_fov_position_sweep(
    args,
    sim_local: dict,
    coc_planet_ratio_dir: str,
    mask_output_tag: str,
    phase_cycles_tag: str,
    phase_sweep_mode_tag: str,
    single_region_tag: str,
    ghost_suffix: str,
    phase_offsets: np.ndarray,
    orbit_radius_lamD: float,
    initial_angle_rad: float,
    central_phase_stack: np.ndarray | None = None,
    trace_centers_lamD: list[tuple[float, float]] | None = None,
) -> dict | None:
    use_coc_trace = bool(getattr(args, "coc_fov_circle_of_circles_trace", False))
    n_steps = int(getattr(args, "coc_fov_position_steps", 0))
    if (not use_coc_trace) and n_steps <= 0:
        return None
    if str(getattr(args, "phase_sweep_mode", "")).strip().lower() == "global":
        return None

    base = CoronagraphSimulator(**sim_local).run()
    n_fft = int(base["n_fft"])
    samp = float(base["focal_sampling"])
    central_box_lamD = 16.0
    half16 = int(0.5 * central_box_lamD * samp)
    cc16 = n_fft // 2
    sl16 = slice(cc16 - half16, cc16 + half16)
    local_region_radius = float(args.local_region_radius)

    centers: list[tuple[float, float]] = []
    maps: list[np.ndarray] = []
    thetas_rad: list[float] = []
    snrs: list[float] = []
    ring_half_width = 0.5
    eval_radius_lamD = 0.5

    if use_coc_trace and central_phase_stack is not None:
        # Reuse already-produced central phase stack (same idea as per-active-FOV maps)
        # and compute one incoherence map per active POV interval.
        phase_series = np.asarray(phase_offsets, dtype=float)
        stack_series = np.asarray(central_phase_stack, dtype=float)
        if (
            phase_series.size > 2
            and np.isclose(phase_series[0], 0.0)
            and np.isclose(phase_series[-1], float(phase_series.max()))
        ):
            phase_series = phase_series[:-1]
            stack_series = stack_series[:-1]

        coc_phase_cycles = (
            float(getattr(args, "coc_phase_cycles"))
            if getattr(args, "coc_phase_cycles", None) is not None
            else float(getattr(args, "local_phase_cycles", 1.0))
        )
        n_positions = max(1, int(getattr(args, "fov_centers_count", 1)))
        theta_rel = _theta_back_and_forth(n_positions, max_abs=np.pi)
        for pos_idx in range(n_positions):
            lo = float(pos_idx) * 2.0 * np.pi
            hi = float(pos_idx + 1) * 2.0 * np.pi
            pm = (phase_series >= lo) & ((phase_series <= hi) if pos_idx == n_positions - 1 else (phase_series < hi))
            idx = np.where(pm)[0]
            if idx.size < 3:
                continue

            th = float(initial_angle_rad + float(theta_rel[pos_idx]))
            ctr = (
                float(orbit_radius_lamD * np.cos(th)),
                float(orbit_radius_lamD * np.sin(th)),
            )
            centers.append(ctr)
            thetas_rad.append(th)

            local_phase = phase_series[idx]
            local_stack = stack_series[idx]
            dphi = float(np.mean(np.diff(local_phase)))
            freq = np.fft.fftfreq(local_stack.shape[0], d=dphi)
            fft_cube = np.fft.fft(local_stack, axis=0)
            band_a = (np.abs(freq) >= 0.0) & (np.abs(freq) <= 0.02)
            fft_a = np.zeros_like(fft_cube, dtype=np.complex128)
            fft_a[band_a] = fft_cube[band_a]
            incoh = np.mean(np.fft.ifft(fft_a, axis=0).real, axis=0)
            maps.append(incoh)

            n16y, n16x = incoh.shape
            x16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, n16x, endpoint=False)
            y16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, n16y, endpoint=False)
            xx16, yy16 = np.meshgrid(x16, y16)
            planet_mask = (xx16 - ctr[0]) ** 2 + (yy16 - ctr[1]) ** 2 <= eval_radius_lamD**2
            peak = float(np.max(incoh[planet_mask])) if np.any(planet_mask) else float("nan")
            rr = np.sqrt(xx16**2 + yy16**2)
            annulus_mask = (rr >= (orbit_radius_lamD - ring_half_width)) & (rr <= (orbit_radius_lamD + ring_half_width))
            ann_vals = incoh[annulus_mask]
            med = float(np.median(ann_vals)) if ann_vals.size > 0 else float("nan")
            snr = float(peak / med) if np.isfinite(peak) and np.isfinite(med) and med != 0.0 else float("nan")
            snrs.append(snr)
    else:
        # Fallback: explicit center sweep (uniform circle or provided centers).
        if use_coc_trace:
            centers = list(trace_centers_lamD or [])
        else:
            for k in range(n_steps):
                th = float(initial_angle_rad + (2.0 * np.pi * k) / float(n_steps))
                ctr = (
                    float(orbit_radius_lamD * np.cos(th)),
                    float(orbit_radius_lamD * np.sin(th)),
                )
                centers.append(ctr)
        if len(centers) == 0:
            return None
        n_centers = len(centers)
        for idx_ctr, ctr in enumerate(centers):
            print(f"[fov-position-sweep] POV {idx_ctr + 1}/{n_centers} center=({ctr[0]:+.3f}, {ctr[1]:+.3f})")
            th = float(np.arctan2(ctr[1], ctr[0]))
            thetas_rad.append(th)
            stack = np.zeros((phase_offsets.size, 2 * half16, 2 * half16), dtype=float)
            for i, ph in enumerate(np.asarray(phase_offsets, dtype=float)):
                phase_sim = CoronagraphSimulator(
                    **{
                        **sim_local,
                        "e_final_phase_offset": 0.0,
                        "focal_local_phase_offset": float(ph),
                        "focal_local_phase_centers_lamD": (ctr,),
                        "focal_local_phase_radius_lamD": local_region_radius,
                    }
                )
                img = phase_sim.run()["final_psf_with_ghost"]
                stack[i] = img[sl16, sl16]

            phase_series = np.asarray(phase_offsets, dtype=float)
            if (
                phase_series.size > 2
                and np.isclose(phase_series[0], 0.0)
                and np.isclose(phase_series[-1], float(phase_series.max()))
            ):
                phase_series = phase_series[:-1]
                stack = stack[:-1]
            dphi = float(np.mean(np.diff(phase_series)))
            freq = np.fft.fftfreq(stack.shape[0], d=dphi)
            fft_cube = np.fft.fft(stack, axis=0)
            band_a = (np.abs(freq) >= 0.0) & (np.abs(freq) <= 0.02)
            fft_a = np.zeros_like(fft_cube, dtype=np.complex128)
            fft_a[band_a] = fft_cube[band_a]
            incoh = np.mean(np.fft.ifft(fft_a, axis=0).real, axis=0)
            maps.append(incoh)

            n16y, n16x = incoh.shape
            x16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, n16x, endpoint=False)
            y16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, n16y, endpoint=False)
            xx16, yy16 = np.meshgrid(x16, y16)
            planet_mask = (xx16 - ctr[0]) ** 2 + (yy16 - ctr[1]) ** 2 <= eval_radius_lamD**2
            peak = float(np.max(incoh[planet_mask])) if np.any(planet_mask) else float("nan")
            rr = np.sqrt(xx16**2 + yy16**2)
            annulus_mask = (rr >= (orbit_radius_lamD - ring_half_width)) & (rr <= (orbit_radius_lamD + ring_half_width))
            ann_vals = incoh[annulus_mask]
            med = float(np.median(ann_vals)) if ann_vals.size > 0 else float("nan")
            snr = float(peak / med) if np.isfinite(peak) and np.isfinite(med) and med != 0.0 else float("nan")
            snrs.append(snr)

    extent16 = [-0.5 * central_box_lamD, 0.5 * central_box_lamD, -0.5 * central_box_lamD, 0.5 * central_box_lamD]
    valid = np.asarray([v for v in snrs if np.isfinite(v)], dtype=float)
    mean_snr = float(np.mean(valid)) if valid.size > 0 else float("nan")

    out_maps_pdf = (
        f"{coc_planet_ratio_dir}/coc_planet_fov_position_sweep_incoherence_per_pov_16lamD_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.pdf"
    )
    with PdfPages(out_maps_pdf) as pdf:
        for i, (ctr, incoh, snr, th) in enumerate(zip(centers, maps, snrs, thetas_rad)):
            fig_m, ax_m = plt.subplots(1, 1, figsize=(7.0, 6.0), constrained_layout=True)
            im = ax_m.imshow(incoh, origin="lower", cmap="viridis", extent=extent16)
            fig_m.colorbar(im, ax=ax_m, fraction=0.046, pad=0.04)
            ax_m.set_title(f"FOV Position Sweep Incoherence: POV {i+1}")
            ax_m.set_xlabel("x [λ/D]")
            ax_m.set_ylabel("y [λ/D]")
            ax_m.plot(ctr[0], ctr[1], marker="o", markersize=5, color="white")
            ax_m.add_patch(
                plt.Circle((0.0, 0.0), orbit_radius_lamD, fill=False, edgecolor="white", linewidth=1.2, linestyle="--")
            )
            ax_m.text(
                0.02,
                0.98,
                f"theta={th:.4f} rad\nSNR={snr:.3e}",
                transform=ax_m.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55, edgecolor="none"),
            )
            pdf.savefig(fig_m)
            plt.close(fig_m)

    out_path = (
        f"{coc_planet_ratio_dir}/coc_planet_fov_position_sweep_final_psf_and_snr_vs_theta_"
        f"{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png"
    )
    fig, (ax_map, ax_curve) = plt.subplots(1, 2, figsize=(13.2, 5.8), constrained_layout=True)
    crop_lamD = 8.0
    half = int(crop_lamD * samp)
    cc = n_fft // 2
    sl = slice(cc - half, cc + half)
    im = ax_map.imshow(
        np.log10(base["final_psf_with_ghost"][sl, sl] + 1e-12),
        origin="lower",
        cmap="inferno",
        vmin=-8,
        vmax=0,
        extent=[-crop_lamD, crop_lamD, -crop_lamD, crop_lamD],
    )
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    ax_map.set_xlabel("x [λ/D]")
    ax_map.set_ylabel("y [λ/D]")
    trace_label = "circle-of-circles" if use_coc_trace else "uniform-circle"
    ax_map.set_title(f"Final PSF + POV Regions ({trace_label})")
    for ctr in centers:
        ax_map.add_patch(
            plt.Circle(
                (ctr[0], ctr[1]),
                float(local_region_radius),
                fill=False,
                edgecolor="white",
                linewidth=1.2,
            )
        )

    # Define theta so that planet location is theta=0.
    theta_planet = float(initial_angle_rad)
    theta_rel = np.asarray([float(np.arctan2(np.sin(th - theta_planet), np.cos(th - theta_planet))) for th in thetas_rad], dtype=float)
    snr_arr = np.asarray(snrs, dtype=float)
    order = np.argsort(theta_rel)
    ax_curve.plot(theta_rel[order], snr_arr[order], "-o", lw=1.6, ms=4.0, color="tab:blue")
    ax_curve.axvline(0.0, color="tab:red", lw=1.2, ls="--", alpha=0.8)
    ax_curve.set_xlabel("theta relative to planet [rad] (planet = 0)")
    ax_curve.set_ylabel("SNR")
    ax_curve.set_title("SNR vs theta")
    ax_curve.grid(alpha=0.3)
    ax_curve.text(
        0.02,
        0.98,
        f"mean SNR={mean_snr:.3e}",
        transform=ax_curve.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
    )
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {
        "out_final_psf_and_snr_vs_theta": out_path,
        "out_maps_per_pov_pdf": out_maps_pdf,
        "centers_lamD": centers,
        "theta_rel_planet_rad_per_pov": theta_rel.tolist(),
        "theta_rad_per_pov": thetas_rad,
        "snr_per_pov": snrs,
        "snr_circle_mean": mean_snr,
        "trace_mode": "circle-of-circles" if use_coc_trace else "uniform-circle",
    }


class CocPlanetPhasePlotter:
    def __init__(
        self,
        args,
        base: dict,
        centers: list[tuple[float, float]],
        planet_region_idx: int,
        coc_planet_ratio_dir: str,
        mask_output_tag: str,
        phase_cycles_tag: str,
        phase_sweep_mode_tag: str,
        single_region_tag: str,
        ghost_suffix: str,
        phase_offsets: np.ndarray,
        roi_masks: list[np.ndarray],
        integrated_intensity: np.ndarray,
        central_phase_stack: np.ndarray,
    ) -> None:
        self.args = args
        self.base = base
        self.centers = centers
        self.planet_region_idx = int(planet_region_idx)
        self.coc_planet_ratio_dir = coc_planet_ratio_dir
        self.mask_output_tag = mask_output_tag
        self.phase_cycles_tag = phase_cycles_tag
        self.phase_sweep_mode_tag = phase_sweep_mode_tag
        self.single_region_tag = single_region_tag
        self.ghost_suffix = ghost_suffix
        self.phase_offsets = phase_offsets
        self.roi_masks = roi_masks
        self.integrated_intensity = integrated_intensity
        self.central_phase_stack = central_phase_stack

    def plot(self) -> dict:
        return _plot_coc_planet_phase_outputs_impl(
            args=self.args,
            base=self.base,
            centers=self.centers,
            planet_region_idx=self.planet_region_idx,
            coc_planet_ratio_dir=self.coc_planet_ratio_dir,
            mask_output_tag=self.mask_output_tag,
            phase_cycles_tag=self.phase_cycles_tag,
            phase_sweep_mode_tag=self.phase_sweep_mode_tag,
            single_region_tag=self.single_region_tag,
            ghost_suffix=self.ghost_suffix,
            phase_offsets=self.phase_offsets,
            roi_masks=self.roi_masks,
            integrated_intensity=self.integrated_intensity,
            central_phase_stack=self.central_phase_stack,
        )


def plot_coc_planet_phase_outputs(
    args,
    base: dict,
    centers: list[tuple[float, float]],
    planet_region_idx: int,
    coc_planet_ratio_dir: str,
    mask_output_tag: str,
    phase_cycles_tag: str,
    phase_sweep_mode_tag: str,
    single_region_tag: str,
    ghost_suffix: str,
    phase_offsets: np.ndarray,
    roi_masks: list[np.ndarray],
    integrated_intensity: np.ndarray,
    central_phase_stack: np.ndarray,
) -> dict:
    return CocPlanetPhasePlotter(
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
    ).plot()
