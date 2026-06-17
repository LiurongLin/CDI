from __future__ import annotations

from pathlib import Path

import numpy as np

from .masks import PhaseMask, VortexPhaseMask


PHASE_SCREEN_DIR = Path(__file__).resolve().parent.parent / "phase_screen"
PHASE_SCREEN_PATHS = {
    "0": PHASE_SCREEN_DIR / "TROIA_phase_screens_new_jitter0percentLamdaOverD.fits",
    "5": PHASE_SCREEN_DIR / "TROIA_phase_screens_new_jitter5percentLamdaOverD.fits",
    "10": PHASE_SCREEN_DIR / "TROIA_phase_screens_new_jitter10percentLamdaOverD.fits",
    "20": PHASE_SCREEN_DIR / "TROIA_phase_screens_new_jitter20percentLamdaOverD.fits",
}


def resolve_phase_screen_path(jitter_choice: str | int | None) -> str | None:
    """Map a jitter choice to a bundled phase-screen FITS path."""
    if jitter_choice is None:
        return None
    token = str(jitter_choice).strip().lower()
    if token in {"", "none", "off"}:
        return None
    path = PHASE_SCREEN_PATHS.get(token)
    if path is None:
        raise ValueError(
            f"Unsupported phase-screen jitter choice: {jitter_choice!r}. "
            "Expected one of: none, 0, 5, 10, 20."
        )
    return str(path)


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
    companion_flux_ratio : float
        Incoherent companion intensity scaling relative to the on-axis source.
        0 means no companion. Example: 1e-3 means 0.1% intensity.
    companion_offset_lamD : tuple[float, float]
        Off-axis companion angular offset (dx, dy) in lambda/D.
    include_companion_ghost : bool
        If False, ghost and companion self-interference are disabled only for the
        companion branch, while the stellar branch keeps the global ghost settings.
    e_final_phase_offset : float
        Global phase offset (radians) applied on the first focal plane.
    focal_local_phase_offset : float
        Local phase offset (radians) applied inside selected first-focal-plane regions.
    focal_local_phase_centers_lamD : tuple[tuple[float, float], ...]
        Region centers (x, y) in lambda/D for local first-focal-plane phase application.
        Supports any number of regions.
    focal_local_phase_shape : str
        Local first-focal-plane region geometry: "circle" or "ring".
    focal_local_phase_radius_lamD : float
        Circular region radius in lambda/D for local first-focal-plane phase application.
    focal_local_phase_inner_radius_lamD : float
        Inner radius in lambda/D when ``focal_local_phase_shape="ring"``.
    focal_local_phase_outer_radius_lamD : float
        Outer radius in lambda/D when ``focal_local_phase_shape="ring"``.
    secondary_diameter_ratio : float
        Central obscuration diameter divided by primary diameter (0 to <1).
    spider_width_pixels : float
        Spider vane width in pupil-grid pixels.
    spider_angles_deg : tuple[float, ...]
        Spider vane orientation angles in degrees (line direction through center).
    pupil_supersample : int
        Subpixel sampling factor per axis for entrance pupil rasterization.
        1 means no supersampling.
    """

    _ENTRANCE_PUPIL_CACHE: dict[tuple, np.ndarray] = {}
    _LYOT_STOP_CACHE: dict[tuple, np.ndarray] = {}
    _PHASE_MASK_CACHE: dict[tuple, np.ndarray] = {}
    _LOCAL_PHASE_REGION_CACHE: dict[tuple, np.ndarray] = {}
    _PHASE_SCREEN_CUBE_CACHE: dict[str, np.ndarray] = {}
    _PHASE_SCREEN_MAP_CACHE: dict[tuple, np.ndarray] = {}

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
        include_ghost: bool = True,
        include_interference: bool = True,
        companion_flux_ratio: float = 0.0,
        companion_offset_lamD: tuple[float, float] = (0.0, 0.0),
        include_companion_ghost: bool = True,
        source_amplitude: float = 1.0,
        normalization_peak: float | None = None,
        e_final_phase_offset: float = 0.0,
        focal_local_phase_offset: float = 0.0,
        focal_local_phase_centers_lamD: tuple[tuple[float, float], ...] = (),
        focal_local_phase_shape: str = "circle",
        focal_local_phase_radius_lamD: float = 0.0,
        focal_local_phase_inner_radius_lamD: float = 0.0,
        focal_local_phase_outer_radius_lamD: float = 0.0,
        secondary_diameter_ratio: float = 0.0,
        spider_width_pixels: float = 0.0,
        spider_angles_deg: tuple[float, ...] = (0.0, 90.0),
        pupil_supersample: int = 1,
        phase_screen_path: str | None = None,
        phase_screen_index: int = 0,
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
        self.include_ghost = bool(include_ghost)
        # Interference requires ghost; force it off when ghost is disabled.
        self.include_interference = bool(include_interference) and self.include_ghost
        self.companion_flux_ratio = float(companion_flux_ratio)
        self.companion_offset_lamD = (float(companion_offset_lamD[0]), float(companion_offset_lamD[1]))
        self.include_companion_ghost = bool(include_companion_ghost)
        self.source_amplitude = float(source_amplitude)
        self.normalization_peak = (
            None if normalization_peak is None else float(normalization_peak)
        )
        self.e_final_phase_offset = float(e_final_phase_offset)
        self.focal_local_phase_offset = float(focal_local_phase_offset)
        self.focal_local_phase_centers_lamD = tuple(
            (float(x), float(y)) for x, y in focal_local_phase_centers_lamD
        )
        self.focal_local_phase_shape = str(focal_local_phase_shape).strip().lower()
        self.focal_local_phase_radius_lamD = float(focal_local_phase_radius_lamD)
        self.focal_local_phase_inner_radius_lamD = float(focal_local_phase_inner_radius_lamD)
        self.focal_local_phase_outer_radius_lamD = float(focal_local_phase_outer_radius_lamD)
        self.secondary_diameter_ratio = float(secondary_diameter_ratio)
        self.spider_width_pixels = float(spider_width_pixels)
        self.spider_angles_deg = tuple(float(a) for a in spider_angles_deg)
        self.pupil_supersample = int(pupil_supersample)
        self.phase_screen_path = None if phase_screen_path is None else str(phase_screen_path)
        self.phase_screen_index = int(phase_screen_index)
        if not (0.0 <= self.ghost_coherence <= 1.0):
            raise ValueError("ghost_coherence must be in [0, 1].")
        if self.companion_flux_ratio < 0.0:
            raise ValueError("companion_flux_ratio must be >= 0.")
        if self.source_amplitude < 0.0:
            raise ValueError("source_amplitude must be >= 0.")
        if self.normalization_peak is not None and self.normalization_peak <= 0.0:
            raise ValueError("normalization_peak must be > 0 when provided.")
        if not (0.0 <= self.secondary_diameter_ratio < 1.0):
            raise ValueError("secondary_diameter_ratio must be in [0, 1).")
        if self.spider_width_pixels < 0.0:
            raise ValueError("spider_width_pixels must be >= 0.")
        if self.pupil_supersample < 1:
            raise ValueError("pupil_supersample must be >= 1.")
        if self.phase_screen_index < 0:
            raise ValueError("phase_screen_index must be >= 0.")
        if self.focal_local_phase_radius_lamD < 0.0:
            raise ValueError("focal_local_phase_radius_lamD must be >= 0.")
        if self.focal_local_phase_shape not in {"circle", "ring"}:
            raise ValueError("focal_local_phase_shape must be 'circle' or 'ring'.")
        if self.focal_local_phase_inner_radius_lamD < 0.0:
            raise ValueError("focal_local_phase_inner_radius_lamD must be >= 0.")
        if self.focal_local_phase_outer_radius_lamD < 0.0:
            raise ValueError("focal_local_phase_outer_radius_lamD must be >= 0.")
        if self.focal_local_phase_shape == "ring":
            if self.focal_local_phase_outer_radius_lamD <= self.focal_local_phase_inner_radius_lamD:
                raise ValueError(
                    "focal_local_phase_outer_radius_lamD must be greater than "
                    "focal_local_phase_inner_radius_lamD for ring regions."
                )

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

    @staticmethod
    def _phase_mask_signature(mask: PhaseMask) -> tuple:
        params = []
        for key, value in sorted(mask.__dict__.items()):
            if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
                params.append((key, float(value) if isinstance(value, (np.floating, float)) else value))
            else:
                params.append((key, repr(value)))
        return (mask.__class__.__name__, tuple(params))

    def _entrance_pupil_binary(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Binary entrance-pupil transmission at provided coordinates."""
        r = np.sqrt(x**2 + y**2)
        primary_radius = self.pupil_pixels / 2.0
        pupil = (r <= primary_radius)

        if self.secondary_diameter_ratio > 0.0:
            secondary_radius = 0.5 * self.pupil_pixels * self.secondary_diameter_ratio
            pupil &= r > secondary_radius

        return pupil

    @staticmethod
    def _casted_spider(start: tuple[float, float], end: tuple[float, float], width: float):
        """Return a callable spider transmission for a finite-width segment."""
        sx, sy = float(start[0]), float(start[1])
        ex, ey = float(end[0]), float(end[1])
        vx = ex - sx
        vy = ey - sy
        v2 = vx * vx + vy * vy
        half_w = width / 2.0

        def transmission(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            if v2 <= 0.0:
                dist = np.sqrt((x - sx) ** 2 + (y - sy) ** 2)
                return (dist > half_w).astype(float)
            t = ((x - sx) * vx + (y - sy) * vy) / v2
            t = np.clip(t, 0.0, 1.0)
            cx = sx + t * vx
            cy = sy + t * vy
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            return (dist > half_w).astype(float)

        return transmission

    def _evaluate_supersampled(
        self, transmission, supersample: int = 8
    ) -> np.ndarray:
        """Evaluate a transmission callable with simple subpixel supersampling."""
        sub_offsets = (np.arange(supersample, dtype=float) + 0.5) / supersample - 0.5
        accum = np.zeros_like(self._x, dtype=float)
        for oy in sub_offsets:
            for ox in sub_offsets:
                accum += transmission(self._x + ox, self._y + oy)
        return accum / float(supersample * supersample)

    def _spider_arms(self, dims: float, normalized: bool = True) -> np.ndarray:
        pupil_diameter = 4.0  # meter
        spider_width = 1.958e-2  # meter
        central_obscuration_ratio = 0.25
        spider_offset = np.array([0.0, 0.37251], dtype=float)  # meter

        if normalized:
            spider_width /= pupil_diameter
            spider_offset = np.array([x / pupil_diameter * 100.0 for x in spider_offset], dtype=float)
            pupil_diameter = dims
            spider_width *= dims

        _ = central_obscuration_ratio  # Kept for parity with the provided model.

        mirror_edge1 = (
            pupil_diameter / (2.0 * np.sqrt(2.0)),
            pupil_diameter / (2.0 * np.sqrt(2.0)),
        )
        mirror_edge2 = (
            -pupil_diameter / (2.0 * np.sqrt(2.0)),
            pupil_diameter / (2.0 * np.sqrt(2.0)),
        )
        mirror_edge3 = (
            pupil_diameter / (2.0 * np.sqrt(2.0)),
            -pupil_diameter / (2.0 * np.sqrt(2.0)),
        )
        mirror_edge4 = (
            -pupil_diameter / (2.0 * np.sqrt(2.0)),
            -pupil_diameter / (2.0 * np.sqrt(2.0)),
        )

        spider1 = self._casted_spider(tuple(spider_offset), mirror_edge1, spider_width)
        spider2 = self._casted_spider(tuple(spider_offset), mirror_edge2, spider_width)
        spider3 = self._casted_spider(tuple(-spider_offset), mirror_edge3, spider_width)
        spider4 = self._casted_spider(tuple(-spider_offset), mirror_edge4, spider_width)

        spider1 = self._evaluate_supersampled(spider1, 8)
        spider2 = self._evaluate_supersampled(spider2, 8)
        spider3 = self._evaluate_supersampled(spider3, 8)
        spider4 = self._evaluate_supersampled(spider4, 8)

        return spider1 * spider2 * spider3 * spider4

    def entrance_pupil(self) -> np.ndarray:
        """Primary mirror with optional central obscuration and spider vanes."""
        cache_key = (
            self.n_fft,
            self.pupil_pixels,
            self.secondary_diameter_ratio,
            self.spider_width_pixels,
            self.spider_angles_deg,
            self.pupil_supersample,
        )
        cached = self._ENTRANCE_PUPIL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        ss = self.pupil_supersample
        if ss == 1:
            pupil = self._entrance_pupil_binary(self._x, self._y).astype(float)
        else:
            sub_offsets = (np.arange(ss, dtype=float) + 0.5) / ss - 0.5
            accum = np.zeros_like(self._x, dtype=float)
            for oy in sub_offsets:
                for ox in sub_offsets:
                    accum += self._entrance_pupil_binary(self._x + ox, self._y + oy).astype(float)
            pupil = accum / float(ss * ss)

        if self.spider_width_pixels > 0.0:
            pupil *= self._spider_arms(self.pupil_pixels, normalized=True)
        self._ENTRANCE_PUPIL_CACHE[cache_key] = pupil
        return pupil

    def _sampled_phase_mask(self) -> np.ndarray:
        """Evaluate phase mask with optional independent spatial sampling."""
        cache_key = (
            self.n_fft,
            self.focal_sampling,
            self.phase_mask_sampling,
            self._phase_mask_signature(self.phase_mask),
        )
        cached = self._PHASE_MASK_CACHE.get(cache_key)
        if cached is not None:
            return cached

        x_lamD = self._x / self.focal_sampling
        y_lamD = self._y / self.focal_sampling

        if np.isclose(self.phase_mask_sampling, self.focal_sampling):
            sampled = self.phase_mask.transmission(x_lamD, y_lamD)
            self._PHASE_MASK_CACHE[cache_key] = sampled
            return sampled

        # Pixelate the mask on its own sampling grid.
        xq = np.round(x_lamD * self.phase_mask_sampling) / self.phase_mask_sampling
        yq = np.round(y_lamD * self.phase_mask_sampling) / self.phase_mask_sampling
        sampled = self.phase_mask.transmission(xq, yq)
        self._PHASE_MASK_CACHE[cache_key] = sampled
        return sampled

    def _local_focal_phase_map(self) -> np.ndarray:
        """
        Piecewise-constant focal-plane phase map (radians) on the first focal plane.
        Non-zero only inside configured local regions.
        """
        if np.isclose(self.focal_local_phase_offset, 0.0):
            return np.zeros((self.n_fft, self.n_fft), dtype=float)

        if self.focal_local_phase_shape == "ring":
            if self.focal_local_phase_outer_radius_lamD <= self.focal_local_phase_inner_radius_lamD:
                return np.zeros((self.n_fft, self.n_fft), dtype=float)
            cache_key = (
                self.n_fft,
                self.focal_sampling,
                self.focal_local_phase_shape,
                self.focal_local_phase_inner_radius_lamD,
                self.focal_local_phase_outer_radius_lamD,
            )
        else:
            if (
                self.focal_local_phase_radius_lamD <= 0.0
                or len(self.focal_local_phase_centers_lamD) == 0
            ):
                return np.zeros((self.n_fft, self.n_fft), dtype=float)
            cache_key = (
                self.n_fft,
                self.focal_sampling,
                self.focal_local_phase_shape,
                self.focal_local_phase_radius_lamD,
                self.focal_local_phase_centers_lamD,
            )
        region = self._LOCAL_PHASE_REGION_CACHE.get(cache_key)
        if region is None:
            x_lamD = self._x / self.focal_sampling
            y_lamD = self._y / self.focal_sampling
            if self.focal_local_phase_shape == "ring":
                rr = np.sqrt(x_lamD**2 + y_lamD**2)
                region = (
                    (rr >= self.focal_local_phase_inner_radius_lamD)
                    & (rr <= self.focal_local_phase_outer_radius_lamD)
                )
            else:
                region = np.zeros((self.n_fft, self.n_fft), dtype=bool)
                r2 = self.focal_local_phase_radius_lamD**2
                for xc, yc in self.focal_local_phase_centers_lamD:
                    region |= (x_lamD - xc) ** 2 + (y_lamD - yc) ** 2 <= r2
            self._LOCAL_PHASE_REGION_CACHE[cache_key] = region

        return self.focal_local_phase_offset * region.astype(float)

    def _lyot_stop(self) -> np.ndarray:
        cache_key = (self.n_fft, self.pupil_pixels, self.lyot_scale)
        cached = self._LYOT_STOP_CACHE.get(cache_key)
        if cached is not None:
            return cached
        lyot_stop = self.circular_pupil(self.lyot_scale * self.pupil_pixels)
        self._LYOT_STOP_CACHE[cache_key] = lyot_stop
        return lyot_stop

    @staticmethod
    def _load_phase_screen_cube(path: str) -> np.ndarray:
        cached = CoronagraphSimulator._PHASE_SCREEN_CUBE_CACHE.get(path)
        if cached is not None:
            return cached
        try:
            from astropy.io import fits
        except ImportError as exc:
            raise ImportError("astropy is required to load phase-screen FITS files.") from exc
        with fits.open(path, memmap=True) as hdul:
            data = hdul[0].data
            if data is None:
                raise ValueError(f"Phase-screen FITS file has no primary data: {path}")
            cube = np.asarray(data, dtype=float)
        CoronagraphSimulator._PHASE_SCREEN_CUBE_CACHE[path] = cube
        return cube

    @staticmethod
    def _resample_phase_screen(screen: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
        """Bilinear-like separable resampling using NumPy interpolation only."""
        src = np.asarray(screen, dtype=float)
        if src.shape == output_shape:
            return src.copy()

        out_h, out_w = output_shape
        in_h, in_w = src.shape
        x_old = np.linspace(0.0, 1.0, in_w)
        x_new = np.linspace(0.0, 1.0, out_w)
        tmp = np.empty((in_h, out_w), dtype=float)
        for row in range(in_h):
            tmp[row, :] = np.interp(x_new, x_old, src[row, :])

        y_old = np.linspace(0.0, 1.0, in_h)
        y_new = np.linspace(0.0, 1.0, out_h)
        out = np.empty((out_h, out_w), dtype=float)
        for col in range(out_w):
            out[:, col] = np.interp(y_new, y_old, tmp[:, col])
        return out

    def _pupil_phase_screen_map(self) -> np.ndarray:
        if self.phase_screen_path is None:
            return np.zeros((self.n_fft, self.n_fft), dtype=float)

        cache_key = (
            self.n_fft,
            self.pupil_pixels,
            self.phase_screen_path,
            self.phase_screen_index,
        )
        cached = self._PHASE_SCREEN_MAP_CACHE.get(cache_key)
        if cached is not None:
            return cached

        cube = self._load_phase_screen_cube(self.phase_screen_path)
        if cube.ndim == 3:
            if self.phase_screen_index >= cube.shape[0]:
                raise IndexError(
                    f"phase_screen_index={self.phase_screen_index} is out of bounds for "
                    f"{self.phase_screen_path} with {cube.shape[0]} screens."
                )
            screen = cube[self.phase_screen_index]
        elif cube.ndim == 2:
            screen = cube
        else:
            raise ValueError(
                f"Phase-screen data must be 2D or 3D, got shape {cube.shape} from {self.phase_screen_path}."
            )

        sampled = self._resample_phase_screen(np.asarray(screen, dtype=float), (self.pupil_pixels, self.pupil_pixels))
        phase_map = np.zeros((self.n_fft, self.n_fft), dtype=float)
        start = (self.n_fft - self.pupil_pixels) // 2
        stop = start + self.pupil_pixels
        phase_map[start:stop, start:stop] = sampled
        self._PHASE_SCREEN_MAP_CACHE[cache_key] = phase_map
        return phase_map

    def _apply_focal_shift_phase_ramp(self, e_pupil: np.ndarray) -> np.ndarray:
        """Apply focal-plane subpixel shift using a phase ramp in the entrance pupil."""
        dx, dy = self.focal_shift_pixels
        if np.isclose(dx, 0.0) and np.isclose(dy, 0.0):
            return e_pupil
        phase_ramp = np.exp(-1j * 2.0 * np.pi * (dx * self._x + dy * self._y) / self.n_fft)
        return e_pupil * phase_ramp

    def _single_source_result_for_shift(
        self,
        focal_shift_pixels: tuple[float, float],
        include_ghost: bool | None = None,
        include_interference: bool | None = None,
        source_amplitude: float = 1.0,
        normalization_peak: float | None = None,
    ) -> dict:
        """Run a clone simulator for a single coherent source at a given shift."""
        use_include_ghost = self.include_ghost if include_ghost is None else bool(include_ghost)
        if include_interference is None:
            use_include_interference = self.include_interference and use_include_ghost
        else:
            use_include_interference = bool(include_interference) and use_include_ghost
        clone = CoronagraphSimulator(
            pupil_pixels=self.pupil_pixels,
            focal_sampling=self.focal_sampling,
            phase_mask_sampling=self.phase_mask_sampling,
            phase_mask=self.phase_mask,
            lyot_scale=self.lyot_scale,
            ghost_fraction=self.ghost_fraction,
            ghost_source=self.ghost_source,
            ghost_offset_lamD=self.ghost_offset_lamD,
            focal_shift_pixels=focal_shift_pixels,
            ghost_phase_rad=self.ghost_phase_rad,
            ghost_coherence=self.ghost_coherence,
            include_ghost=use_include_ghost,
            include_interference=use_include_interference,
            companion_flux_ratio=0.0,
            companion_offset_lamD=(0.0, 0.0),
            include_companion_ghost=self.include_companion_ghost,
            source_amplitude=source_amplitude,
            normalization_peak=normalization_peak,
            e_final_phase_offset=self.e_final_phase_offset,
            focal_local_phase_offset=self.focal_local_phase_offset,
            focal_local_phase_centers_lamD=self.focal_local_phase_centers_lamD,
            focal_local_phase_shape=self.focal_local_phase_shape,
            focal_local_phase_radius_lamD=self.focal_local_phase_radius_lamD,
            focal_local_phase_inner_radius_lamD=self.focal_local_phase_inner_radius_lamD,
            focal_local_phase_outer_radius_lamD=self.focal_local_phase_outer_radius_lamD,
            secondary_diameter_ratio=self.secondary_diameter_ratio,
            spider_width_pixels=self.spider_width_pixels,
            spider_angles_deg=self.spider_angles_deg,
            pupil_supersample=self.pupil_supersample,
            phase_screen_path=self.phase_screen_path,
            phase_screen_index=self.phase_screen_index,
        )
        return clone.run()

    @staticmethod
    def _shift_complex(field: np.ndarray, dx_pix: int, dy_pix: int) -> np.ndarray:
        """Integer-pixel shift for complex field placement."""
        return np.roll(np.roll(field, shift=dy_pix, axis=0), shift=dx_pix, axis=1)

    def run(self) -> dict:
        entrance_pupil = self.entrance_pupil()
        e_pupil = self.source_amplitude * entrance_pupil.astype(np.complex128)
        pupil_phase_screen = self._pupil_phase_screen_map()
        e_pupil *= np.exp(1j * pupil_phase_screen)
        e_pupil = self._apply_focal_shift_phase_ramp(e_pupil)

        e_focal_before_mask = self._centered_fft2(e_pupil)

        mask = self._sampled_phase_mask()
        local_focal_phase = self._local_focal_phase_map()
        e_focal_after_mask = (
            e_focal_before_mask
            * (1 - np.sqrt(self.ghost_fraction))
            * mask
            * np.exp(-1j * (self.e_final_phase_offset + local_focal_phase))
        )

        e_lyot = self._centered_ifft2(e_focal_after_mask)

        lyot_stop = self._lyot_stop()
        e_lyot_stopped = e_lyot * lyot_stop

        e_final = self._centered_fft2(e_lyot_stopped)

        i_direct = np.abs(e_focal_before_mask) ** 2
        i_coron = np.abs(e_final) ** 2

        if self.normalization_peak is not None:
            norm = self.normalization_peak
        else:
            norm = np.max(i_direct)
        e_direct_norm = e_focal_before_mask / np.sqrt(norm)
        e_coron_norm = e_final / np.sqrt(norm)
        i_direct = np.abs(e_direct_norm) ** 2
        i_coron = np.abs(e_coron_norm) ** 2

        if self.include_ghost:
            amp_scale = np.sqrt(self.ghost_fraction)
        else:
            amp_scale = 0.0

        if self.ghost_source == "direct":
            ghost_seed_field = amp_scale * e_direct_norm
        elif self.ghost_source == "coronagraphic":
            ghost_seed_field = amp_scale * e_coron_norm
        else:
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
        if self.include_ghost and self.include_interference:
            interference_term = 2.0 * np.real(
                self.ghost_coherence * e_coron_norm * np.conj(ghost_field)
            )
        else:
            interference_term = np.zeros_like(i_coron)
        i_final_with_ghost = i_final_no_interference + interference_term
        ghost_only_no_interference = ghost_psf
        ghost_only_with_interference = ghost_psf + interference_term

        r_pix = np.sqrt(self._x**2 + self._y**2)
        r_lamD_map = r_pix / self.focal_sampling

        companion_present = self.companion_flux_ratio > 0.0
        if companion_present:
            comp_shift = (
                self.focal_shift_pixels[0] - self.companion_offset_lamD[0] * self.focal_sampling,
                self.focal_shift_pixels[1] - self.companion_offset_lamD[1] * self.focal_sampling,
            )
            comp = self._single_source_result_for_shift(
                comp_shift,
                include_ghost=self.include_ghost and self.include_companion_ghost,
                include_interference=self.include_interference and self.include_companion_ghost,
                source_amplitude=np.sqrt(self.companion_flux_ratio),
                normalization_peak=norm,
            )

            i_direct_comp = comp["direct_psf"]
            i_coron_comp = comp["coronagraphic_psf"]
            ghost_psf_comp = comp["ghost_psf"]
            ghost_only_no_interference_comp = comp["ghost_only_no_interference"]
            ghost_only_with_interference_comp = comp["ghost_only_with_interference"]
            interference_term_comp = comp["interference_term"]
            i_final_no_interference_comp = comp["final_psf_no_interference"]
            i_final_with_ghost_comp = comp["final_psf_with_ghost"]
        else:
            i_direct_comp = np.zeros_like(i_direct)
            i_coron_comp = np.zeros_like(i_coron)
            ghost_psf_comp = np.zeros_like(ghost_psf)
            ghost_only_no_interference_comp = np.zeros_like(ghost_only_no_interference)
            ghost_only_with_interference_comp = np.zeros_like(ghost_only_with_interference)
            interference_term_comp = np.zeros_like(interference_term)
            i_final_no_interference_comp = np.zeros_like(i_final_no_interference)
            i_final_with_ghost_comp = np.zeros_like(i_final_with_ghost)

        i_direct_total = i_direct + i_direct_comp
        i_coron_total = i_coron + i_coron_comp
        ghost_psf_total = ghost_psf + ghost_psf_comp
        ghost_only_no_interference_total = (
            ghost_only_no_interference + ghost_only_no_interference_comp
        )
        ghost_only_with_interference_total = (
            ghost_only_with_interference + ghost_only_with_interference_comp
        )
        interference_term_total = interference_term + interference_term_comp
        i_final_no_interference_total = i_final_no_interference + i_final_no_interference_comp
        i_final_with_ghost_total = i_final_with_ghost + i_final_with_ghost_comp

        rr, prof_direct = self._radial_profile(i_direct_total)
        _, prof_coron = self._radial_profile(i_coron_total)
        _, prof_no_interference = self._radial_profile(i_final_no_interference_total)
        _, prof_with_ghost = self._radial_profile(i_final_with_ghost_total)
        _, prof_ghost_only_no_interference = self._radial_profile(ghost_only_no_interference_total)
        _, prof_ghost_only_with_interference = self._radial_profile(ghost_only_with_interference_total)
        rr_delta, delta_std, delta_abs_mean = self._annular_delta_stats(
            i_final_no_interference_total, i_final_with_ghost_total
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
            "include_ghost": self.include_ghost,
            "include_interference": self.include_interference,
            "secondary_diameter_ratio": self.secondary_diameter_ratio,
            "spider_width_pixels": self.spider_width_pixels,
            "spider_angles_deg": self.spider_angles_deg,
            "pupil_supersample": self.pupil_supersample,
            "phase_screen_path": self.phase_screen_path,
            "phase_screen_index": self.phase_screen_index,
            "companion_flux_ratio": self.companion_flux_ratio,
            "companion_offset_lamD": self.companion_offset_lamD,
            "include_companion_ghost": self.include_companion_ghost,
            "source_amplitude": self.source_amplitude,
            "normalization_peak": norm,
            "pupil": entrance_pupil,
            "pupil_phase_screen": pupil_phase_screen,
            "mask": mask,
            "lyot_field": e_lyot,
            "lyot_stop": lyot_stop,
            "direct_psf": i_direct_total,
            "direct_field": e_direct_norm,
            "coronagraphic_psf": i_coron_total,
            "coronagraphic_field": e_coron_norm,
            "ghost_field": ghost_field,
            "ghost_psf": ghost_psf_total,
            "ghost_only_no_interference": ghost_only_no_interference_total,
            "ghost_only_with_interference": ghost_only_with_interference_total,
            "interference_term": interference_term_total,
            "final_psf_no_interference": i_final_no_interference_total,
            "final_psf_with_ghost": i_final_with_ghost_total,
            "direct_psf_star": i_direct,
            "coronagraphic_psf_star": i_coron,
            "ghost_psf_star": ghost_psf,
            "interference_term_star": interference_term,
            "final_psf_with_ghost_star": i_final_with_ghost,
            "direct_psf_companion": i_direct_comp,
            "coronagraphic_psf_companion": i_coron_comp,
            "ghost_psf_companion": ghost_psf_comp,
            "interference_term_companion": interference_term_comp,
            "final_psf_with_ghost_companion": i_final_with_ghost_comp,
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
            "focal_local_phase_offset": self.focal_local_phase_offset,
            "focal_local_phase_shape": self.focal_local_phase_shape,
            "focal_local_phase_radius_lamD": self.focal_local_phase_radius_lamD,
            "focal_local_phase_centers_lamD": self.focal_local_phase_centers_lamD,
            "focal_local_phase_inner_radius_lamD": self.focal_local_phase_inner_radius_lamD,
            "focal_local_phase_outer_radius_lamD": self.focal_local_phase_outer_radius_lamD,
        }
