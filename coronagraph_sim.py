import numpy as np
import matplotlib.pyplot as plt
import argparse
from abc import ABC, abstractmethod


class PhaseMask(ABC):
    """Base class for focal-plane phase masks."""

    @abstractmethod
    def transmission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return complex transmission map on focal-plane coordinates."""


class FlatPhaseMask(PhaseMask):
    """Constant phase mask: exp(i * phase_rad), defaulting to zero phase shift."""

    def __init__(self, phase_rad: float = 0.0):
        self.phase_rad = float(phase_rad)

    def transmission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(1j * self.phase_rad) * np.ones_like(x, dtype=np.complex128)


class NoPhaseMask(PhaseMask):
    """Backward-compatible alias for a zero-phase flat mask."""

    def transmission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.ones_like(x, dtype=np.complex128)


class VortexPhaseMask(PhaseMask):
    """Vortex phase mask: exp(i * charge * theta)."""

    def __init__(self, charge: int = 2):
        self.charge = charge

    def transmission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        theta = np.arctan2(y, x)
        return np.exp(1j * self.charge * theta)


class FQPMPhaseMask(PhaseMask):
    """Four-Quadrant Phase Mask (pi phase shift in alternating quadrants)."""

    def transmission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        q = np.sign(x) * np.sign(y)
        phase = np.where(q >= 0, 0.0, np.pi)
        return np.exp(1j * phase)


class RoddierPhaseMask(PhaseMask):
    """Roddier phase mask: central disk with a phase shift, zero phase outside."""

    def __init__(self, radius_lamD: float = 0.53, phase_rad: float = np.pi):
        self.radius_lamD = float(radius_lamD)
        self.phase_rad = float(phase_rad)

    def transmission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        phase = np.where(r <= self.radius_lamD, self.phase_rad, 0.0)
        return np.exp(1j * phase)


class CoronagraphSimulator:
    """
    Fraunhofer propagation simulator for a Lyot coronagraph architecture.

    Parameters
    ----------
    pupil_pixels : int
        Entrance pupil diameter in pixels.
    focal_sampling : int or float
        Focal-plane sampling in pixels per (lambda/D).
    phase_mask_sampling : int or float or None
        Sampling of the focal-plane phase mask in pixels per (lambda/D).
        If None, it matches focal_sampling.
    phase_mask : PhaseMask
        Focal-plane phase mask object.
    lyot_scale : float
        Lyot stop diameter scaling relative to entrance pupil diameter.
    ghost_fraction : float
        Incoherent refractive ghost intensity fraction relative to the chosen source PSF.
        Example: 0.005 means 0.5%.
    ghost_source : str
        Which field seeds the ghost:
        - "direct": direct focal field (no Lyot stop)
        - "coronagraphic": coronagraphic focal field
        - "phase_mask_refraction": ghost generated at phase-mask plane from
          incident focal field, then propagated through Lyot stop
    ghost_offset_lamD : tuple[float, float]
        Ghost offset (dx, dy) in lambda/D in the focal plane.
    focal_shift_pixels : tuple[float, float]
        Global focal-plane shift (dx, dy) in pixels, implemented by a linear
        phase ramp at the entrance pupil.
    ghost_phase_rad : float
        Constant phase applied to the ghost field (radians).
    ghost_coherence : float
        Mutual coherence factor gamma in [0, 1] used in
        2*Re{gamma * E_psf * E_ghost*}. Use 0 for no-interference sum.
    """

    def __init__(
        self,
        pupil_pixels: int = 100,
        focal_sampling: float = 10.0,
        phase_mask_sampling: float | None = None,
        phase_mask: PhaseMask | None = None,
        lyot_scale: float = 0.95,
        ghost_fraction: float = 0.005,
        ghost_source: str = "direct",
        ghost_offset_lamD: tuple[float, float] = (0.0, 0.0),
        focal_shift_pixels: tuple[float, float] = (0.5, 0.5),
        ghost_phase_rad: float = 0.0,
        ghost_coherence: float = 1.0,
        e_final_phase_offset: float = 0.0,
    ):
        self.pupil_pixels = int(pupil_pixels)
        self.focal_sampling = float(focal_sampling)
        self.phase_mask_sampling = (
            self.focal_sampling if phase_mask_sampling is None else float(phase_mask_sampling)
        )
        self.phase_mask = phase_mask if phase_mask is not None else VortexPhaseMask(charge=2)
        self.lyot_scale = float(lyot_scale)
        self.ghost_fraction = float(ghost_fraction)
        self.ghost_source = str(ghost_source).lower()
        if self.ghost_source not in {"direct", "coronagraphic", "phase_mask_refraction"}:
            raise ValueError(
                "ghost_source must be 'direct', 'coronagraphic', or 'phase_mask_refraction'."
            )
        self.ghost_offset_lamD = (float(ghost_offset_lamD[0]), float(ghost_offset_lamD[1]))
        self.focal_shift_pixels = (float(focal_shift_pixels[0]), float(focal_shift_pixels[1]))
        self.ghost_phase_rad = float(ghost_phase_rad)
        self.ghost_coherence = float(ghost_coherence)
        self.e_final_phase_offset = float(e_final_phase_offset)
        if not (0.0 <= self.ghost_coherence <= 1.0):
            raise ValueError("ghost_coherence must be in [0, 1].")

        # Ensures focal-plane sampling = N_fft / D_pixels.
        self.n_fft = int(np.ceil(self.pupil_pixels * self.focal_sampling))

        self._x, self._y = self._centered_coordinates(self.n_fft)

    @staticmethod
    def _centered_fft2(field: np.ndarray) -> np.ndarray:
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))

    @staticmethod
    def _centered_ifft2(field: np.ndarray) -> np.ndarray:
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))

    @staticmethod
    def _centered_coordinates(n: int) -> tuple[np.ndarray, np.ndarray]:
        y, x = np.indices((n, n), dtype=float)
        c = (n - 1) / 2.0
        return x - c, y - c

    @staticmethod
    def _radial_profile(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = image.shape[0]
        y, x = np.indices(image.shape, dtype=float)
        c = (n - 1) / 2.0
        r = np.sqrt((x - c) ** 2 + (y - c) ** 2)
        r_int = r.astype(int)

        tbin = np.bincount(r_int.ravel(), image.ravel())
        nr = np.bincount(r_int.ravel())
        profile = tbin / np.maximum(nr, 1)
        return np.arange(len(profile)), profile

    @staticmethod
    def _annular_delta_stats(
        reference: np.ndarray, target: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return annular std(delta) and annular mean(abs(delta)), delta = target - reference."""
        delta = target - reference
        n = delta.shape[0]
        y, x = np.indices(delta.shape, dtype=float)
        c = (n - 1) / 2.0
        r_int = np.sqrt((x - c) ** 2 + (y - c) ** 2).astype(int)

        nr = np.bincount(r_int.ravel())
        sum_d = np.bincount(r_int.ravel(), weights=delta.ravel())
        sum_d2 = np.bincount(r_int.ravel(), weights=(delta.ravel() ** 2))
        sum_abs_d = np.bincount(r_int.ravel(), weights=np.abs(delta.ravel()))

        mean_d = sum_d / np.maximum(nr, 1)
        mean_d2 = sum_d2 / np.maximum(nr, 1)
        std_d = np.sqrt(np.maximum(mean_d2 - mean_d**2, 0.0))
        mean_abs_d = sum_abs_d / np.maximum(nr, 1)
        return np.arange(len(nr)), std_d, mean_abs_d

    def circular_pupil(self, diameter_pixels: float) -> np.ndarray:
        r = np.sqrt(self._x**2 + self._y**2)
        return (r <= diameter_pixels / 2.0).astype(float)

    def _sampled_phase_mask(self) -> np.ndarray:
        """Evaluate phase mask with optional independent spatial sampling."""
        x_lamD = self._x / self.focal_sampling
        y_lamD = self._y / self.focal_sampling

        if np.isclose(self.phase_mask_sampling, self.focal_sampling):
            return self.phase_mask.transmission(x_lamD, y_lamD)

        # Pixelate the mask on its own sampling grid.
        xq = np.round(x_lamD * self.phase_mask_sampling) / self.phase_mask_sampling
        yq = np.round(y_lamD * self.phase_mask_sampling) / self.phase_mask_sampling
        return self.phase_mask.transmission(xq, yq)

    def _apply_focal_shift_phase_ramp(self, e_pupil: np.ndarray) -> np.ndarray:
        """Apply focal-plane subpixel shift using a phase ramp in the entrance pupil."""
        dx, dy = self.focal_shift_pixels
        if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
            return e_pupil
        phase_ramp = np.exp(-1j * 2.0 * np.pi * (dx * self._x + dy * self._y) / self.n_fft)
        return e_pupil * phase_ramp

    @staticmethod
    def _shift_intensity(image: np.ndarray, dx_pix: int, dy_pix: int) -> np.ndarray:
        """Integer-pixel shift used for ghost placement."""
        return np.roll(np.roll(image, shift=dy_pix, axis=0), shift=dx_pix, axis=1)

    @staticmethod
    def _shift_complex(field: np.ndarray, dx_pix: int, dy_pix: int) -> np.ndarray:
        """Integer-pixel shift for complex field placement."""
        return np.roll(np.roll(field, shift=dy_pix, axis=0), shift=dx_pix, axis=1)

    def run(self) -> dict:
        entrance_pupil = self.circular_pupil(self.pupil_pixels)
        e_pupil = entrance_pupil.astype(np.complex128)
        e_pupil = self._apply_focal_shift_phase_ramp(e_pupil)

        e_focal_before_mask = self._centered_fft2(e_pupil)

        mask = self._sampled_phase_mask()
        e_focal_after_mask = e_focal_before_mask*(1-np.sqrt(self.ghost_fraction)) * mask *np.exp(-1j * self.e_final_phase_offset)

        e_lyot = self._centered_ifft2(e_focal_after_mask)

        lyot_stop = self.circular_pupil(self.lyot_scale * self.pupil_pixels)
        e_lyot_stopped = e_lyot * lyot_stop

        if self.e_final_phase_offset:
            e_final = self._centered_fft2(e_lyot_stopped)
        else:
            e_final = self._centered_fft2(e_lyot_stopped)



        i_direct = np.abs(e_focal_before_mask) ** 2
        i_coron = np.abs(e_final) ** 2

        norm = np.max(i_direct)
        e_direct_norm = e_focal_before_mask / np.sqrt(norm)
        e_coron_norm = e_final / np.sqrt(norm)
        i_direct = np.abs(e_direct_norm) ** 2
        i_coron = np.abs(e_coron_norm) ** 2

        # Coherent ghost model (field-level combination with interference)
        # Apply ghost fraction at the beginning of the selected ghost path.
        amp_scale = np.sqrt(self.ghost_fraction)
        if self.ghost_source == "direct":
            ghost_seed_field = amp_scale * e_direct_norm
        elif self.ghost_source == "coronagraphic":
            ghost_seed_field = amp_scale * e_coron_norm
        else:
            # phase_mask_refraction:
            # ghost originates in the phase-mask plane from the masked focal field,
            # then propagates to Lyot plane, is clipped by Lyot stop, and
            # propagates to final focal plane.
            e_ghost_focal_from_mask = amp_scale * e_focal_before_mask
            e_ghost_lyot = self._centered_ifft2(e_ghost_focal_from_mask)
            e_ghost_lyot_stopped = e_ghost_lyot * lyot_stop
            e_ghost_final = self._centered_fft2(e_ghost_lyot_stopped)
            ghost_seed_field = e_ghost_final / np.sqrt(norm)
        dx_pix = int(np.round(self.ghost_offset_lamD[0] * self.focal_sampling))
        dy_pix = int(np.round(self.ghost_offset_lamD[1] * self.focal_sampling))
        ghost_field = (
            self._shift_complex(ghost_seed_field, dx_pix=dx_pix, dy_pix=dy_pix)
            * np.exp(1j * self.ghost_phase_rad)
        )
        ghost_psf = np.abs(ghost_field) ** 2

        i_final_no_interference = i_coron + ghost_psf
        interference_term = 2.0 * np.real(
            self.ghost_coherence * e_coron_norm * np.conj(ghost_field)
        )
        # interference_term = 2.0 * np.sqrt(i_coron*ghost_psf) * self.ghost_coherence * np.cos(self.e_final_phase_offset)
        i_final_with_ghost = i_final_no_interference + interference_term
        ghost_only_no_interference = ghost_psf
        ghost_only_with_interference = ghost_psf + interference_term
        # Enforce non-negative intensity after interference term
        # i_final_with_ghost = np.maximum(i_final_with_ghost, 0.0)

        r_pix = np.sqrt(self._x**2 + self._y**2)
        r_lamD_map = r_pix / self.focal_sampling

        rr, prof_direct = self._radial_profile(i_direct)
        _, prof_coron = self._radial_profile(i_coron)
        _, prof_no_interference = self._radial_profile(i_final_no_interference)
        _, prof_with_ghost = self._radial_profile(i_final_with_ghost)
        _, prof_ghost_only_no_interference = self._radial_profile(ghost_only_no_interference)
        _, prof_ghost_only_with_interference = self._radial_profile(ghost_only_with_interference)
        rr_delta, delta_std, delta_abs_mean = self._annular_delta_stats(
            i_final_no_interference, i_final_with_ghost
        )

        return {
            "n_fft": self.n_fft,
            "focal_sampling": self.focal_sampling,
            "phase_mask_sampling": self.phase_mask_sampling,
            "phase_mask_name": self.phase_mask.__class__.__name__,
            "ghost_fraction": self.ghost_fraction,
            "ghost_source": self.ghost_source,
            "ghost_offset_lamD": self.ghost_offset_lamD,
            "focal_shift_pixels": self.focal_shift_pixels,
            "ghost_phase_rad": self.ghost_phase_rad,
            "ghost_coherence": self.ghost_coherence,
            "pupil": entrance_pupil,
            "mask": mask,
            "lyot_field": e_lyot,
            "lyot_stop": lyot_stop,
            "direct_psf": i_direct,
            "coronagraphic_psf": i_coron,
            "ghost_field": ghost_field,
            "ghost_psf": ghost_psf,
            "ghost_only_no_interference": ghost_only_no_interference,
            "ghost_only_with_interference": ghost_only_with_interference,
            "interference_term": interference_term,
            "final_psf_no_interference": i_final_no_interference,
            "final_psf_with_ghost": i_final_with_ghost,
            "r_lamD_map": r_lamD_map,
            "radial_r_lamD": rr / self.focal_sampling,
            "radial_direct": prof_direct,
            "radial_coron": prof_coron,
            "radial_no_interference": prof_no_interference,
            "radial_with_ghost": prof_with_ghost,
            "radial_ghost_only_no_interference": prof_ghost_only_no_interference,
            "radial_ghost_only_with_interference": prof_ghost_only_with_interference,
            "radial_delta_r_lamD": rr_delta / self.focal_sampling,
            "radial_delta_std": delta_std,
            "radial_delta_abs_mean": delta_abs_mean,
            "e_final_phase_offset": self.e_final_phase_offset,
        }



def plot_results(result: dict, save_path: str = "charge2_coronagraph_simulation.png") -> None:
    n_fft = result["n_fft"]
    samp = result["focal_sampling"]

    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(2, 4)
    ax_img0 = fig.add_subplot(gs[0, 0])
    ax_img1 = fig.add_subplot(gs[0, 1])
    ax_img2 = fig.add_subplot(gs[0, 2])
    ax_img3 = fig.add_subplot(gs[0, 3])
    ax_curve = fig.add_subplot(gs[1, :])

    crop_lamD = 20
    half = int(crop_lamD * samp)
    c = n_fft // 2
    sl = slice(c - half, c + half)

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
        # vmin=-vmax_interf,
        # vmax=vmax_interf,
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
    ax_curve.set_yscale("log")
    ax_curve.set_ylim(1e-8, 10)
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
    """Sweep e_final_phase_offset from 0 to pi and plot peak/total intensity metrics."""
    phase_offsets = np.linspace(0.0, 2*np.pi, n_phase_samples)

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
        ax.set_xticks([0.0, np.pi, 2*np.pi])
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
    """Sweep e_final_phase_offset from 0 to pi for combined (C+G+I) intensity metrics."""
    phase_offsets = np.linspace(0.0, 2*np.pi, n_phase_samples)
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
        ax.set_xticks([0.0, np.pi, 2*np.pi])
        ax.set_xticklabels(["0", r"$\pi$", r"2*$\pi$"])

    fig.savefig(save_path, dpi=160, bbox_inches="tight")

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)


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


def _default_sim_kwargs() -> dict:
    return dict(
        pupil_pixels=100,
        focal_sampling=10,
        phase_mask_sampling=10,
        phase_mask=RoddierPhaseMask(phase_rad=float(3.281219)),
        lyot_scale=1,
        ghost_fraction=0.005,
        ghost_source="phase_mask_refraction",
        ghost_offset_lamD=(0.0, 0.0),
        focal_shift_pixels=(0.5, 0.5),
        ghost_phase_rad=0.0,
        ghost_coherence=1.0,
        e_final_phase_offset=np.pi,
    )


def _print_run_header(result: dict) -> None:
    print(f"FFT grid size: {result['n_fft']} x {result['n_fft']}")
    print(f"Focal-plane sampling: {result['focal_sampling']} px/(λ/D)")
    print(f"Phase-mask sampling: {result['phase_mask_sampling']} px/(λ/D)")
    print(f"Phase mask: {result['phase_mask_name']}")
    print(f"Ghost fraction: {result['ghost_fraction'] * 100:.3f}% of {result['ghost_source']} PSF")
    print(f"Ghost offset: {result['ghost_offset_lamD']} λ/D")
    print(f"Global focal shift: {result['focal_shift_pixels']} px")
    print(f"Ghost phase: {result['ghost_phase_rad']:.3f} rad")
    print(f"Ghost coherence gamma: {result['ghost_coherence']:.3f}")
    print("e_final_phase_offset: {:.3f} rad".format(result["e_final_phase_offset"]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected coronagraph simulation features independently."
    )
    parser.add_argument(
        "--feature",
        nargs="+",
        choices=["single", "phase", "combined", "radius-match", "phase-match", "all"],
        default=["single"],
        help="Feature(s) to run. Use 'all' to run everything.",
    )
    parser.add_argument(
        "--phase-samples",
        type=int,
        default=101,
        help="Number of phase samples for phase sweeps.",
    )
    parser.add_argument("--radius-min", type=float, default=0.2, help="Min Roddier radius [λ/D].")
    parser.add_argument("--radius-max", type=float, default=1.2, help="Max Roddier radius [λ/D].")
    parser.add_argument(
        "--radius-samples",
        type=int,
        default=41,
        help="Number of radius samples for radius-match sweep.",
    )
    parser.add_argument(
        "--roddier-radius",
        type=float,
        default=0.53,
        help="Fixed Roddier radius [λ/D] for phase-match sweep.",
    )
    parser.add_argument(
        "--phase-match-min",
        type=float,
        default=0.0,
        help="Minimum phase_rad [rad] for phase-match sweep.",
    )
    parser.add_argument(
        "--phase-match-max",
        type=float,
        default=2.0 * np.pi,
        help="Maximum phase_rad [rad] for phase-match sweep.",
    )
    parser.add_argument(
        "--phase-match-samples",
        type=int,
        default=181,
        help="Number of phase samples for phase-match sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    features = set(args.feature)
    if "all" in features:
        features = {"single", "phase", "combined", "radius-match", "phase-match"}

    sim_kwargs = _default_sim_kwargs()

    result = None
    if {"single", "phase", "combined"} & features:
        result = CoronagraphSimulator(**sim_kwargs).run()
        _print_run_header(result)

    if "single" in features:
        out = f"charge2_coronagraph_simulation_{sim_kwargs['e_final_phase_offset']}.png"
        plot_results(result, save_path=out)
        save_phase_mask_fits(result, fits_path="phase_mask.fits")
        print(f"Saved simulation plot: {out}")
        print("Saved phase mask FITS: phase_mask.fits")

    if "phase" in features:
        out = (
            "phase_offset_peak_total_intensity_"
            f"phase_mask_{result['phase_mask_name']}_ghost_fraction_{result['ghost_fraction'] * 100:.3f}%.png"
        )
        plot_phase_offset_metrics(
            sim_kwargs=sim_kwargs,
            n_phase_samples=args.phase_samples,
            save_path=out,
        )
        print(f"Saved phase-offset sweep plot: {out}")

    if "combined" in features:
        out = (
            "phase_offset_combined_peak_total_intensity_"
            f"phase_mask_{result['phase_mask_name']}_ghost_fraction_{result['ghost_fraction'] * 100:.3f}%.png"
        )
        plot_phase_offset_combined_metrics(
            sim_kwargs=sim_kwargs,
            n_phase_samples=args.phase_samples,
            save_path=out,
        )
        print(f"Saved combined phase-offset plot: {out}")

    if "radius-match" in features:
        match = sweep_roddier_radius_for_peak_match(
            sim_kwargs=sim_kwargs,
            radius_min=args.radius_min,
            radius_max=args.radius_max,
            n_radius_samples=args.radius_samples,
            phase_rad=np.pi,
            save_path="roddier_radius_peak_match.png",
        )
        print("Saved radius sweep plot: roddier_radius_peak_match.png")
        print(f"Best Roddier radius_lamD: {match['best_radius_lamD']:.4f} λ/D")
        print(
            "At best radius: coron peak = "
            f"{match['best_peak_coron']:.6e}, ghost peak = {match['best_peak_ghost']:.6e}, "
            f"|delta| = {match['best_abs_difference']:.6e} "
            f"({100 * match['best_relative_difference_to_ghost']:.3f}% of ghost peak)"
        )

    if "phase-match" in features:
        match = sweep_roddier_phase_for_peak_match(
            sim_kwargs=sim_kwargs,
            radius_lamD=args.roddier_radius,
            phase_min_rad=args.phase_match_min,
            phase_max_rad=args.phase_match_max,
            n_phase_samples=args.phase_match_samples,
            save_path="roddier_phase_peak_match.png",
        )
        print("Saved phase sweep plot: roddier_phase_peak_match.png")
        print(f"Fixed Roddier radius_lamD: {match['radius_lamD']:.4f} λ/D")
        print(f"Best phase_rad: {match['best_phase_rad']:.6f} rad")
        print(
            "At best phase: coron peak = "
            f"{match['best_peak_coron']:.6e}, ghost peak = {match['best_peak_ghost']:.6e}, "
            f"|delta| = {match['best_abs_difference']:.6e} "
            f"({100 * match['best_relative_difference_to_ghost']:.3f}% of ghost peak)"
        )

    # Examples for later mask swapping:
    # sim.phase_mask = FlatPhaseMask(phase_rad=0.0)  # no phase shift at focal plane
    # sim.phase_mask = NoPhaseMask()
    # sim.phase_mask = RoddierPhaseMask(radius_lamD=0.53, phase_rad=np.pi)
    # sim.phase_mask = FQPMPhaseMask()
    # sim.phase_mask = VortexPhaseMask(charge=4)
    # sim.phase_mask_sampling = 5   # coarser mask resolution
    # sim.phase_mask_sampling = 20  # finer mask resolution
    # sim.ghost_fraction = 0.002      # 0.2%
    # sim.ghost_source = "coronagraphic"
    # sim.ghost_offset_lamD = (5, -3) # move ghost by (dx, dy) lambda/D
    # sim.ghost_phase_rad = np.pi / 3
    # sim.ghost_coherence = 0.7


if __name__ == "__main__":
    main()
