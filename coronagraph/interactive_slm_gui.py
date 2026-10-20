from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .masks import NoPhaseMask
from .simulator import CoronagraphSimulator


@dataclass
class SLMLyotConfig:
    pupil_pixels: int = 100
    focal_sampling: float = 10.0
    phase_mask_sampling: float | None = 10.0
    phase_mask: object = field(default_factory=NoPhaseMask)
    lyot_scale: float = 1.0
    ghost_fraction: float = 0.0
    ghost_source: str = "phase_mask_refraction"
    ghost_offset_lamD: tuple[float, float] = (0.0, 0.0)
    focal_shift_pixels: tuple[float, float] = (0.5, 0.5)
    ghost_phase_rad: float = 0.0
    ghost_coherence: float = 1.0
    include_ghost: bool = False
    include_interference: bool = False
    companion_offset_lamD: tuple[float, float] = (4.0, 0.0)
    include_companion_ghost: bool = False
    coherent_ring_speckle_count: int = 3
    custom_speckle_offsets_lamD: tuple[tuple[float, float], ...] | None = None
    secondary_diameter_ratio: float = 0.0
    spider_width_pixels: float = 0.0
    spider_angles_deg: tuple[float, ...] = (0.0, 90.0)
    pupil_supersample: int = 1
    phase_screen_path: str | None = None
    phase_screen_index: int = 0
    lyot_reference_scale: float = 1.0
    perfect_coronagraph: bool = True


def default_slm_gui_config() -> SLMLyotConfig:
    return SLMLyotConfig(
        companion_offset_lamD=(11.0, 0.0),
        coherent_ring_speckle_count=1,
        custom_speckle_offsets_lamD=((-11.0, 0.0),),
    )


def _slm_safe_token(value: object) -> str:
    token = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return "".join(ch for ch in token if ch.isalnum() or ch == "_") or "run"


def _slm_coord_token(value: object) -> str:
    number = float(value)
    prefix = "m" if number < 0.0 else "p"
    return prefix + f"{abs(number):.3f}".replace(".", "p")


def _slm_artifact_directory(now: datetime | None = None) -> Path:
    timestamp = now if now is not None else datetime.now()
    directory = Path("interactive_slm_logs") / timestamp.strftime("%Y-%m-%d")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _slm_artifact_path(
    *,
    planet_offset_lamD: tuple[float, float],
    process_kind: str,
    suffix: str,
    now: datetime | None = None,
) -> Path:
    timestamp = now if now is not None else datetime.now()
    px, py = planet_offset_lamD
    stem = (
        f"{timestamp.strftime('%H%M%S')}"
        f"_planet_x_{_slm_coord_token(px)}"
        f"_y_{_slm_coord_token(py)}"
        f"_{_slm_safe_token(process_kind)}"
    )
    return _slm_artifact_directory(timestamp) / f"{stem}_{suffix}"


def _positive_ratio(value: float, name: str) -> float:
    ratio = float(value)
    if ratio <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return ratio


class SLMLyotResponseCalculator:
    """Compute component-wise Lyot-plane response to a focal-plane phase mask.

    Each source propagates coherently across the whole focal plane: the selected
    and unselected portions interfere with each other in the Lyot plane. Star
    and speckles also combine coherently; star and planet combine in intensity.
    ``include_interference`` controls ghost interference only, not this coherence.
    """

    def __init__(self, config: SLMLyotConfig | None = None):
        self.config = config if config is not None else SLMLyotConfig()
        self._template = self._make_simulator()
        self.n_fft = self._template.n_fft
        self.focal_sampling = self._template.focal_sampling
        self.x_lamD = self._template._x / self.focal_sampling
        self.y_lamD = self._template._y / self.focal_sampling
        self.lyot_stop = self._template._lyot_stop()
        self._unit_unmodulated_cache: dict[tuple, np.ndarray] = {}

    def _make_simulator(
        self,
        phase_map: np.ndarray | None = None,
        *,
        lyot_reference_scale: float | None = None,
    ) -> CoronagraphSimulator:
        cfg = self.config
        return CoronagraphSimulator(
            pupil_pixels=cfg.pupil_pixels,
            focal_sampling=cfg.focal_sampling,
            phase_mask_sampling=cfg.phase_mask_sampling,
            phase_mask=cfg.phase_mask,
            lyot_scale=cfg.lyot_scale,
            ghost_fraction=cfg.ghost_fraction,
            ghost_source=cfg.ghost_source,
            ghost_offset_lamD=cfg.ghost_offset_lamD,
            focal_shift_pixels=cfg.focal_shift_pixels,
            ghost_phase_rad=cfg.ghost_phase_rad,
            ghost_coherence=cfg.ghost_coherence,
            include_ghost=cfg.include_ghost,
            include_interference=cfg.include_interference,
            include_star=True,
            include_companion=False,
            companion_flux_ratio=0.0,
            companion_offset_lamD=cfg.companion_offset_lamD,
            include_companion_ghost=cfg.include_companion_ghost,
            coherent_ring_speckle_count=0,
            coherent_ring_speckle_intensity=0.0,
            source_amplitude=1.0,
            normalization_peak=None,
            e_final_phase_offset=0.0,
            focal_plane_phase_map_rad=phase_map,
            secondary_diameter_ratio=cfg.secondary_diameter_ratio,
            spider_width_pixels=cfg.spider_width_pixels,
            spider_angles_deg=cfg.spider_angles_deg,
            pupil_supersample=cfg.pupil_supersample,
            phase_screen_path=cfg.phase_screen_path,
            phase_screen_index=cfg.phase_screen_index,
            lyot_reference_scale=(
                cfg.lyot_reference_scale
                if lyot_reference_scale is None
                else float(lyot_reference_scale)
            ),
            perfect_coronagraph=cfg.perfect_coronagraph,
        )

    def _reference_field(
        self,
        sim: CoronagraphSimulator,
        *,
        apply_focal_plane_modulation: bool = False,
    ) -> np.ndarray | None:
        if not self.config.perfect_coronagraph:
            return None
        reference = sim._run_single_branch(
            phase_screen_enabled=False,
            apply_focal_plane_modulation=apply_focal_plane_modulation,
        )
        return np.asarray(reference["lyot_field_before_reference_subtraction"], dtype=np.complex128)

    def _single_lyot(
        self,
        sim: CoronagraphSimulator,
        *,
        shift_pixels: tuple[float, float] | None = None,
        source_amplitude: float = 1.0,
        phase_screen_enabled: bool = True,
        lyot_reference_field: np.ndarray | None = None,
    ) -> np.ndarray:
        branch = self._single_lyot_branch(
            sim,
            shift_pixels=shift_pixels,
            source_amplitude=source_amplitude,
            phase_screen_enabled=phase_screen_enabled,
            lyot_reference_field=lyot_reference_field,
        )
        return np.asarray(branch["lyot_field"], dtype=np.complex128)

    def _single_lyot_branch(
        self,
        sim: CoronagraphSimulator,
        *,
        shift_pixels: tuple[float, float] | None = None,
        source_amplitude: float = 1.0,
        phase_screen_enabled: bool = True,
        lyot_reference_field: np.ndarray | None = None,
    ) -> dict:
        branch = sim._run_single_branch(
            focal_shift_pixels=shift_pixels,
            source_amplitude=source_amplitude,
            phase_screen_enabled=phase_screen_enabled,
            lyot_reference_field=lyot_reference_field,
            apply_focal_plane_modulation=True,
        )
        return {
            "lyot_field": np.asarray(branch["lyot_field"], dtype=np.complex128),
            "lyot_field_before_reference_subtraction": np.asarray(
                branch["lyot_field_before_reference_subtraction"],
                dtype=np.complex128,
            ),
            "lyot_reference_field": np.asarray(
                branch["lyot_reference_field"],
                dtype=np.complex128,
            ),
        }

    def _cached_unmodulated_unit_lyot(
        self,
        *,
        cache_name: str,
        shift_pixels: tuple[float, float] | None = None,
        phase_screen_enabled: bool = True,
        use_perfect_reference: bool = False,
    ) -> np.ndarray:
        shift_key = None if shift_pixels is None else (float(shift_pixels[0]), float(shift_pixels[1]))
        key = (cache_name, shift_key, bool(phase_screen_enabled), bool(use_perfect_reference))
        cached = self._unit_unmodulated_cache.get(key)
        if cached is not None:
            return cached

        sim = self._make_simulator()
        reference = self._reference_field(sim) if use_perfect_reference else None
        field = self._single_lyot(
            sim,
            shift_pixels=shift_pixels,
            source_amplitude=1.0,
            phase_screen_enabled=phase_screen_enabled,
            lyot_reference_field=reference,
        )
        self._unit_unmodulated_cache[key] = field
        return field

    def _speckle_offsets(self) -> list[tuple[float, float]]:
        cfg = self.config
        if cfg.custom_speckle_offsets_lamD is not None:
            return [
                (float(x), float(y))
                for x, y in cfg.custom_speckle_offsets_lamD
            ]
        sim = CoronagraphSimulator(
            pupil_pixels=cfg.pupil_pixels,
            focal_sampling=cfg.focal_sampling,
            companion_offset_lamD=cfg.companion_offset_lamD,
            coherent_ring_speckle_count=cfg.coherent_ring_speckle_count,
            coherent_ring_speckle_intensity=1.0,
        )
        return sim._coherent_ring_speckle_offsets_lamD()

    def source_pixel(
        self,
        source_offset_lamD: tuple[float, float],
    ) -> tuple[int, int]:
        """Return the nearest focal-grid pixel for a source offset in lambda/D."""
        cx = (self.n_fft - 1) / 2.0
        cy = (self.n_fft - 1) / 2.0
        x = int(round(cx + float(source_offset_lamD[0]) * self.focal_sampling))
        y = int(round(cy + float(source_offset_lamD[1]) * self.focal_sampling))
        return x, y

    def place_mask_template(
        self,
        template_mask: np.ndarray,
        source_offset_lamD: tuple[float, float],
        *,
        anchor_pixel: tuple[int, int] | None = None,
    ) -> dict:
        """Translate one binary mask template so its anchor lands on a source position."""
        template = np.asarray(template_mask, dtype=bool)
        if template.shape != (self.n_fft, self.n_fft):
            raise ValueError(f"SLM template must have shape ({self.n_fft}, {self.n_fft}).")
        if anchor_pixel is None:
            anchor = (int(round((self.n_fft - 1) / 2.0)), int(round((self.n_fft - 1) / 2.0)))
        else:
            anchor = (int(anchor_pixel[0]), int(anchor_pixel[1]))
        if not (0 <= anchor[0] < self.n_fft and 0 <= anchor[1] < self.n_fft):
            raise ValueError("Mask template anchor is outside the SLM grid.")

        target = self.source_pixel(source_offset_lamD)
        if not (0 <= target[0] < self.n_fft and 0 <= target[1] < self.n_fft):
            raise ValueError(
                "Source position is outside the SLM grid: "
                f"{source_offset_lamD} lambda/D maps to pixel {target}."
            )

        yy, xx = np.nonzero(template)
        placed = np.zeros_like(template, dtype=bool)
        dx = target[0] - anchor[0]
        dy = target[1] - anchor[1]
        if xx.size:
            new_x = xx + dx
            new_y = yy + dy
            inside = (
                (new_x >= 0)
                & (new_y >= 0)
                & (new_x < self.n_fft)
                & (new_y < self.n_fft)
            )
            if not np.all(inside):
                clipped = int(np.count_nonzero(~inside))
                raise ValueError(
                    "Placed mask would clip at the SLM grid boundary: "
                    f"{clipped} selected template pixels are outside the grid. "
                    f"Target pixel is {target}."
                )
            placed[new_y, new_x] = True

        return {
            "mask": placed,
            "anchor_pixel": anchor,
            "target_pixel": target,
            "source_offset_lamD": (float(source_offset_lamD[0]), float(source_offset_lamD[1])),
            "translation_pixels": (int(dx), int(dy)),
            "selected_pixels": int(np.count_nonzero(placed)),
        }

    def propagate(
        self,
        slm_region_mask: np.ndarray,
        *,
        include_star: bool = True,
        include_planet: bool = True,
        include_speckles: bool = False,
        selected_speckle_indices: tuple[int, ...] | list[int] | None = None,
        star_planet_ratio: float = 500.0,
        star_speckle_ratio: float = 100.0,
        phase_modulation_rad: float = np.pi,
        reference_phase_modulation_rad: float | None = None,
        lyot_reference_scale: float | None = None,
    ) -> dict:
        region = np.asarray(slm_region_mask, dtype=bool)
        if region.shape != (self.n_fft, self.n_fft):
            raise ValueError(f"SLM mask must have shape ({self.n_fft}, {self.n_fft}).")

        star_planet_ratio = _positive_ratio(star_planet_ratio, "star_planet_ratio")
        star_speckle_ratio = float(star_speckle_ratio)
        phase_modulation_rad = float(phase_modulation_rad)
        if not np.isfinite(phase_modulation_rad):
            raise ValueError("phase_modulation_rad must be finite.")
        reference_phase_modulation_rad = (
            phase_modulation_rad
            if reference_phase_modulation_rad is None
            else float(reference_phase_modulation_rad)
        )
        if not np.isfinite(reference_phase_modulation_rad):
            raise ValueError("reference_phase_modulation_rad must be finite.")
        reference_scale = (
            float(self.config.lyot_reference_scale)
            if lyot_reference_scale is None
            else float(lyot_reference_scale)
        )
        if reference_scale < 0.0:
            raise ValueError("lyot_reference_scale must be >= 0.")
        phase_map = np.where(region, phase_modulation_rad, 0.0).astype(float)
        reference_phase_map = np.where(region, reference_phase_modulation_rad, 0.0).astype(float)

        mod = self._make_simulator(phase_map=phase_map, lyot_reference_scale=reference_scale)
        ref_sim = self._make_simulator(
            phase_map=reference_phase_map,
            lyot_reference_scale=reference_scale,
        )
        mod_ref = (
            self._reference_field(ref_sim, apply_focal_plane_modulation=True)
            if include_star
            else None
        )

        zero = np.zeros((self.n_fft, self.n_fft), dtype=np.complex128)
        fields: dict[str, dict[str, np.ndarray]] = {}

        if include_star:
            e0 = self._cached_unmodulated_unit_lyot(
                cache_name="star",
                use_perfect_reference=self.config.perfect_coronagraph,
            )
            branch = self._single_lyot_branch(mod, lyot_reference_field=mod_ref)
            em = branch["lyot_field"]
            fields["star"] = {
                "unmodulated": e0,
                "modulated": em,
                "delta": em - e0,
                "raw_modulated": branch["lyot_field_before_reference_subtraction"],
                "reference": branch["lyot_reference_field"],
            }
        else:
            fields["star"] = {
                "unmodulated": zero,
                "modulated": zero,
                "delta": zero,
                "raw_modulated": zero,
                "reference": zero,
            }

        if include_planet:
            planet_amp = 1.0 / np.sqrt(star_planet_ratio)
            shift = (
                self.config.focal_shift_pixels[0]
                - self.config.companion_offset_lamD[0] * self.focal_sampling,
                self.config.focal_shift_pixels[1]
                - self.config.companion_offset_lamD[1] * self.focal_sampling,
            )
            e0 = planet_amp * self._cached_unmodulated_unit_lyot(
                cache_name="planet",
                shift_pixels=shift,
            )
            branch = self._single_lyot_branch(
                mod,
                shift_pixels=shift,
                source_amplitude=planet_amp,
                lyot_reference_field=None,
            )
            em = branch["lyot_field"]
            fields["planet"] = {
                "unmodulated": e0,
                "modulated": em,
                "delta": em - e0,
                "raw_modulated": branch["lyot_field_before_reference_subtraction"],
                "reference": branch["lyot_reference_field"],
            }
        else:
            fields["planet"] = {
                "unmodulated": zero,
                "modulated": zero,
                "delta": zero,
                "raw_modulated": zero,
                "reference": zero,
            }

        speckle_offsets_all = self._speckle_offsets() if include_speckles else []
        if selected_speckle_indices is not None:
            selected = {int(idx) for idx in selected_speckle_indices}
            speckle_offsets = [
                offset
                for idx, offset in enumerate(speckle_offsets_all)
                if idx in selected
            ]
        else:
            speckle_offsets = speckle_offsets_all
        if speckle_offsets:
            star_speckle_ratio = _positive_ratio(star_speckle_ratio, "star_speckle_ratio")
            speckle_amp = 1.0 / np.sqrt(star_speckle_ratio)
            e0_total = np.zeros_like(zero)
            em_total = np.zeros_like(zero)
            raw_total = np.zeros_like(zero)
            for offset_lamD in speckle_offsets:
                shift = (
                    self.config.focal_shift_pixels[0] - offset_lamD[0] * self.focal_sampling,
                    self.config.focal_shift_pixels[1] - offset_lamD[1] * self.focal_sampling,
                )
                e0_total += speckle_amp * self._cached_unmodulated_unit_lyot(
                    cache_name=f"speckle:{offset_lamD[0]:.12g},{offset_lamD[1]:.12g}",
                    shift_pixels=shift,
                )
                branch = self._single_lyot_branch(
                    mod,
                    shift_pixels=shift,
                    source_amplitude=speckle_amp,
                    lyot_reference_field=None,
                )
                em_total += branch["lyot_field"]
                raw_total += branch["lyot_field_before_reference_subtraction"]
            fields["speckle"] = {
                "unmodulated": e0_total,
                "modulated": em_total,
                "delta": em_total - e0_total,
                "raw_modulated": raw_total,
                "reference": zero,
            }
        else:
            fields["speckle"] = {
                "unmodulated": zero,
                "modulated": zero,
                "delta": zero,
                "raw_modulated": zero,
                "reference": zero,
            }

        fields["coherent"] = {
            "unmodulated": fields["star"]["unmodulated"] + fields["speckle"]["unmodulated"],
            "modulated": fields["star"]["modulated"] + fields["speckle"]["modulated"],
            "delta": fields["star"]["delta"] + fields["speckle"]["delta"],
            "raw_modulated": fields["star"]["raw_modulated"] + fields["speckle"]["raw_modulated"],
            "reference": fields["star"]["reference"],
        }
        incoherent_delta_amp = np.sqrt(
            np.abs(fields["star"]["delta"]) ** 2
            + np.abs(fields["planet"]["delta"]) ** 2
        )
        fields["incoherent"] = {
            "unmodulated": np.zeros_like(zero, dtype=float),
            "modulated": incoherent_delta_amp,
            "delta": incoherent_delta_amp,
            "raw_modulated": incoherent_delta_amp,
            "reference": np.zeros_like(zero, dtype=float),
        }

        metrics = self._metrics(fields, include_star, include_planet, bool(speckle_offsets))
        return {
            "fields": fields,
            "metrics": metrics,
            "slm_region_mask": region,
            "phase_map_rad": phase_map,
            "lyot_stop": self.lyot_stop,
            "speckle_offsets_lamD": tuple(speckle_offsets),
            "selected_speckle_indices": (
                tuple(range(len(speckle_offsets_all)))
                if selected_speckle_indices is None
                else tuple(int(idx) for idx in selected_speckle_indices)
            ),
            "star_planet_ratio": star_planet_ratio,
            "star_speckle_ratio": star_speckle_ratio,
            "phase_modulation_rad": phase_modulation_rad,
            "reference_phase_modulation_rad": reference_phase_modulation_rad,
            "lyot_reference_scale": reference_scale,
            "include_star": bool(include_star),
            "include_planet": bool(include_planet),
            "include_speckles": bool(include_speckles),
        }

    def science_intensity_sweep(
        self, slm_region_mask: np.ndarray, *, radius_lamD: float = 0.5,
        selected_speckle_index: int = 0, phase_steps: int = 16,
        measurement_plane: str = "science", measure_all_companions: bool = False,
        measurement_region_names: list[str] | None = None,
        phase_start_rad: float = 0.0, phase_span_rad: float = 2.0 * np.pi,
        include_star: bool = True, include_planet: bool = True,
        include_speckles: bool = True, selected_speckle_indices=None,
        star_planet_ratio: float = 500.0, star_speckle_ratio: float = 100.0,
        lyot_reference_scale: float | None = None,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        """Measure total physical intensity in fixed circular source apertures.

        The simulator supplies the stopped science fields and ghost fields.
        Star and speckles add as fields; the planet adds as intensity. No
        star-only or phase-zero intensity is removed from either aperture.
        """
        region = np.array(slm_region_mask, dtype=bool, copy=True)
        if region.shape != (self.n_fft, self.n_fft):
            raise ValueError("SLM mask has the wrong shape.")
        radius = float(radius_lamD)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Measurement radius must be finite and > 0 lambda/D.")
        offsets = self._speckle_offsets()
        plane = str(measurement_plane).strip().lower()
        if plane not in {"focal", "science"}:
            raise ValueError("measurement_plane must be 'focal' or 'science'.")
        target = int(selected_speckle_index)
        if not measure_all_companions and (target < 0 or target >= len(offsets)):
            raise ValueError("Select an existing target speckle for measurement.")
        n = int(phase_steps)
        start, span = float(phase_start_rad), float(phase_span_rad)
        if n < 2 or not np.isfinite(start) or not np.isfinite(span) or span <= 0:
            raise ValueError("Use at least 2 phases and a finite positive phase span.")
        ratios = [_positive_ratio(star_planet_ratio, "star_planet_ratio"),
                  _positive_ratio(star_speckle_ratio, "star_speckle_ratio")]
        if not np.all(np.isfinite(ratios)):
            raise ValueError("Source ratios must be finite.")
        scale = self.config.lyot_reference_scale if lyot_reference_scale is None else float(lyot_reference_scale)
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("Reference scale must be finite and nonnegative.")
        centers = (
            {f"speckle_{i + 1}": offset for i, offset in enumerate(offsets)}
            if measure_all_companions else {"speckle": offsets[target]}
        )
        centers["planet"] = self.config.companion_offset_lamD
        if measurement_region_names is not None:
            if not measurement_region_names or any(name not in centers for name in measurement_region_names):
                raise ValueError("Choose existing science measurement regions.")
            centers = {name: centers[name] for name in measurement_region_names}
        masks, regions = {}, {}
        for name, (x, y) in centers.items():
            if (x - radius < self.x_lamD.min() or x + radius > self.x_lamD.max()
                    or y - radius < self.y_lamD.min() or y + radius > self.y_lamD.max()):
                raise ValueError(f"{name} measurement circle crosses the science image boundary.")
            mask = (self.x_lamD - x)**2 + (self.y_lamD - y)**2 <= radius**2
            if not np.any(mask):
                raise ValueError(f"{name} measurement circle contains no pixel centers; increase its radius.")
            masks[name] = mask
            regions[name] = {"center_lamD": [float(x), float(y)], "radius_lamD": radius,
                             "pixel_count": int(mask.sum())}
        # One unit-star direct PSF peak, independent of phase and source toggles.
        norm = float(self._template._run_single_branch(
            apply_focal_plane_modulation=False)["normalization_peak"])
        selected = set(range(len(offsets))) if selected_speckle_indices is None else {int(i) for i in selected_speckle_indices}
        phases = start + span * np.arange(n) / n
        curves = {name: np.zeros(n) for name in masks}
        preview = None
        display_max = 0.0
        field_key = "coronagraphic_field" if plane == "science" else "focal_plane_field_after_mask"
        for i, phase in enumerate(phases):
            if progress_callback is not None:
                progress_callback({"phase_index": i + 1, "phase_count": n, "phase_rad": float(phase)})
            sim = self._make_simulator(np.where(region, phase, 0.0), lyot_reference_scale=scale)
            reference = self._reference_field(sim, apply_focal_plane_modulation=True) if include_star else None
            coherent = np.zeros(region.shape, dtype=complex)
            ghost = np.zeros_like(coherent)
            if include_star:
                branch = sim._run_single_branch(normalization_peak=norm, lyot_reference_field=reference)
                coherent += branch[field_key]
                ghost += branch["ghost_field"] if plane == "science" else 0
            # Preserve the simulator's companion/speckle ghost switch.
            sim.include_ghost = self.config.include_ghost and self.config.include_companion_ghost
            if include_speckles:
                for j, offset in enumerate(offsets):
                    if j not in selected:
                        continue
                    shift = tuple(self.config.focal_shift_pixels[k] - offset[k] * self.focal_sampling for k in (0, 1))
                    branch = sim._run_single_branch(focal_shift_pixels=shift,
                        source_amplitude=1 / np.sqrt(ratios[1]), normalization_peak=norm)
                    coherent += branch[field_key]
                    ghost += branch["ghost_field"] if plane == "science" else 0
            intensity = np.abs(coherent)**2 + np.abs(ghost)**2
            if plane == "science" and self.config.include_ghost and self.config.include_interference:
                intensity += 2 * np.real(self.config.ghost_coherence * coherent * np.conj(ghost))
            if include_planet:
                shift = tuple(self.config.focal_shift_pixels[k] - self.config.companion_offset_lamD[k] * self.focal_sampling for k in (0, 1))
                branch = sim._run_single_branch(focal_shift_pixels=shift,
                    source_amplitude=1 / np.sqrt(ratios[0]), normalization_peak=norm)
                intensity += (branch["final_psf_with_ghost"] if plane == "science"
                              else np.abs(branch[field_key])**2)
            for name, mask in masks.items():
                curves[name][i] = np.mean(intensity[mask])
            if preview is None:
                preview = intensity.copy()
            display_max = max(display_max, float(intensity.max()))
        return {
            "phases": phases, "mean_intensities": curves, "regions": regions,
            "metrics": {name: {"maximum": float(v.max()), "minimum": float(v.min()),
                                "peak_to_peak": float(v.max() - v.min())} for name, v in curves.items()},
            "normalization_peak": norm,
            "intensity_units": "unit-star direct PSF peak",
            "measurement_plane": "final science focal plane" if plane == "science" else "SLM focal plane before Lyot stop",
            "measurement_plane_key": plane,
            "measure_all_companions": bool(measure_all_companions),
            "background_subtraction": "none",
            "phase_start_rad": start, "phase_span_rad": span,
            "preview_phase_rad": float(phases[0]), "preview_intensity": preview,
            "display_max": display_max, "slm_region_mask": region,
            "source_state": {"include_star": bool(include_star), "include_planet": bool(include_planet),
                "include_speckles": bool(include_speckles), "selected_speckle_indices": sorted(selected),
                "target_speckle_index": None if measure_all_companions else target, "speckle_offsets_lamD": offsets,
                "planet_offset_lamD": list(self.config.companion_offset_lamD),
                "perfect_coronagraph": self.config.perfect_coronagraph,
                "include_ghost": self.config.include_ghost,
                "include_companion_ghost": self.config.include_companion_ghost,
                "include_interference": self.config.include_interference,
                "ghost_fraction": self.config.ghost_fraction, "ghost_coherence": self.config.ghost_coherence,
                "star_planet_ratio": ratios[0],
                "star_speckle_ratio": ratios[1], "lyot_reference_scale": float(scale)},
        }

    def calibrate_science_phase_zero_intensity(
        self, *, selected_speckle_index: int = 0, radius_lamD: float = 0.5,
        include_star: bool = True, include_planet: bool = True, include_speckles: bool = True,
        selected_speckle_indices=None, star_planet_ratio: float = 500.0,
        star_speckle_ratio: float = 100.0, lyot_reference_scale: float | None = None,
    ) -> dict:
        """Match total target-speckle/planet ROI means at zero applied SLM phase.

        Solve the quadratic dependence on one common speckle amplitude. Full
        physical images provide the calibration samples; no background is
        removed from any measured or displayed intensity. The calibrated ratio
        is held fixed in all subsequent phases and mask placements.
        """
        if not include_planet or not include_speckles:
            raise ValueError("Science phase-zero matching requires the planet and speckles enabled.")
        target = int(selected_speckle_index)
        offsets = self._speckle_offsets()
        selected = list(range(len(offsets))) if selected_speckle_indices is None else list(selected_speckle_indices)
        if target < 0 or target >= len(offsets) or target not in selected:
            raise ValueError("Enable the selected target speckle for science phase-zero matching.")
        base_ratio = _positive_ratio(star_speckle_ratio, "star_speckle_ratio")
        if not np.isfinite(base_ratio):
            raise ValueError("star_speckle_ratio must be finite.")
        zero = np.zeros((self.n_fft, self.n_fft), dtype=bool)
        region_name = f"speckle_{target + 1}"
        settings = dict(radius_lamD=radius_lamD, measure_all_companions=True,
                        measurement_plane="science", phase_steps=2,
                        include_star=include_star, include_planet=True,
                        selected_speckle_indices=selected,
                        star_planet_ratio=star_planet_ratio, lyot_reference_scale=lyot_reference_scale)
        def sample(relative_amplitude):
            result = self.science_intensity_sweep(zero, **settings,
                include_speckles=relative_amplitude > 0,
                star_speckle_ratio=base_ratio / relative_amplitude**2 if relative_amplitude else base_ratio)
            means = result["mean_intensities"]
            return (float(means[region_name][0]), float(means["planet"][0]), result["normalization_peak"])
        samples = [sample(value) for value in (0.0, 1.0, 2.0)]
        differences = [speckle - planet for speckle, planet, _ in samples]
        d = differences[0]
        c = (differences[2] - 2 * differences[1] + d) / 2
        b = differences[1] - d - c
        tol = 64 * np.finfo(float).eps * max(abs(c), abs(b), abs(d), np.finfo(float).tiny)
        if abs(c) <= tol:
            roots = [1.0] if abs(b) <= tol and abs(d) <= tol else ([-d / b] if abs(b) > tol else [])
        else:
            discriminant = b*b - 4*c*d
            disc_tol = 64 * np.finfo(float).eps * max(b*b, abs(4*c*d), np.finfo(float).tiny)
            if discriminant < -disc_tol:
                roots = []
            else:
                q = -0.5 * (b + np.copysign(np.sqrt(max(0.0, discriminant)), b))
                roots = [q/c, d/q] if q else [-b/(2*c)]
        roots = [float(value) for value in roots if np.isfinite(value) and value > 0]
        if not roots:
            raise ValueError("Cannot match these science-region intensities with a positive common speckle amplitude. Change the target or source settings.")
        factor = min(roots, key=lambda value: abs(np.log(value)))
        ratio = base_ratio / factor**2
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("Science phase-zero matching produced an invalid Star/Speckle ratio.")
        speckle_mean, planet_mean, norm = sample(factor)
        residual = speckle_mean - planet_mean
        if abs(residual) > 1e-8 * max(abs(speckle_mean), abs(planet_mean), 1e-30):
            raise ValueError("Science phase-zero matching failed its physical-intensity verification.")
        return {
            "measurement_plane": "final science focal plane", "phase_rad": 0.0,
            "quantity": "mean total physical intensity in fixed source regions",
            "target_region": region_name, "reference_region": "planet",
            "radius_lamD": float(radius_lamD), "star_speckle_ratio": float(ratio),
            "input_star_speckle_ratio": float(base_ratio), "relative_speckle_amplitude": factor,
            "speckle_phase_zero_mean_intensity": speckle_mean,
            "planet_phase_zero_mean_intensity": planet_mean,
            "residual": float(residual), "normalization_peak": float(norm),
            "selected_speckle_indices": selected,
        }

    def science_comparison_sweeps(
        self, template_mask: np.ndarray, *, comparison_type: str = "separate",
        source_names: list[str] | None = None, progress_callback=None, **scan_settings,
    ) -> dict:
        """Compare fixed science apertures while scanning one SLM placement at a time.

        Every scan includes the same enabled sources and records every aperture,
        including cross-response in apertures away from the modulated source.
        """
        if comparison_type not in {"separate", "shared_mask"}:
            raise ValueError("comparison_type must be 'separate' or 'shared_mask'.")
        template = np.array(template_mask, dtype=bool, copy=True)
        centers = {f"speckle_{i+1}": offset for i, offset in enumerate(self._speckle_offsets())}
        centers["planet"] = self.config.companion_offset_lamD
        names = list(centers) if source_names is None else list(source_names)
        if not names or len(set(names)) != len(names) or any(name not in centers for name in names):
            raise ValueError("Select existing, distinct companion names to scan.")
        # Center the selected template itself, even when it was drawn off-axis.
        yy, xx = np.nonzero(template)
        anchor = ((int(round(float(xx.mean()))), int(round(float(yy.mean())))) if xx.size
                  else (int(round((self.n_fft - 1) / 2)),) * 2)
        # Validate every placement before starting any propagation.
        placements = ({"shared_mask": {"mask": template}} if comparison_type == "shared_mask" else {
            name: self.place_mask_template(template, centers[name], anchor_pixel=anchor) for name in names})
        settings = dict(scan_settings)
        settings.update(measurement_plane="science", measure_all_companions=True)
        sweeps = {}
        for index, (name, placement) in enumerate(placements.items()):
            def report(update, name=name, index=index):
                if progress_callback is not None:
                    progress_callback({"active_run": f"science:{name}", "candidate": name,
                                       "candidate_count": len(placements), "completed_runs": index, **update})
            sweeps[name] = self.science_intensity_sweep(placement["mask"],
                                                       progress_callback=report, **settings)
        first = next(iter(sweeps.values()))
        if any(item["normalization_peak"] != first["normalization_peak"]
               or item["regions"] != first["regions"] for item in sweeps.values()):
            raise RuntimeError("Science comparison must use common normalization and fixed apertures.")
        own_metrics = {name: sweeps[name if comparison_type == "separate" else "shared_mask"]["metrics"][name]
                       for name in names}
        return {
            "measurement_plane": "final science focal plane", "comparison_type": comparison_type,
            "source_names": names, "mask_anchor_pixel": anchor, "sweeps": sweeps,
            "placements": {name: {k: v for k, v in placement.items() if k != "mask"}
                           for name, placement in placements.items()},
            "own_region_metrics": own_metrics,
            "modulation_matrix": {name: {region: metric["peak_to_peak"] for region, metric in run["metrics"].items()}
                                  for name, run in sweeps.items()},
            "normalization_peak": first["normalization_peak"], "regions": first["regions"],
            "slm_template_mask": template,
        }

    def phase_sweep(
        self,
        slm_region_mask: np.ndarray,
        *,
        phase_steps: int = 4,
        phase_start_rad: float = 0.0,
        phase_span_rad: float = 2.0 * np.pi,
        subtraction_mode: str = "field",
        lyot_reference_scale: float | None = None,
        include_star: bool = True,
        include_planet: bool = True,
        include_speckles: bool = False,
        selected_speckle_indices: tuple[int, ...] | list[int] | None = None,
        star_planet_ratio: float = 500.0,
        star_speckle_ratio: float = 100.0,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        n_steps = int(phase_steps)
        if n_steps < 2:
            raise ValueError("phase_steps must be >= 2.")
        phase_start = float(phase_start_rad)
        phase_span = float(phase_span_rad)
        if not np.isfinite(phase_start) or not np.isfinite(phase_span):
            raise ValueError("phase_start_rad and phase_span_rad must be finite.")
        mode = str(subtraction_mode).strip().lower().replace("_", "-")
        if mode in {"electric", "electric-field", "field", "field-subtraction"}:
            mode = "field"
        elif mode in {"intensity", "intensity-subtraction"}:
            mode = "intensity"
        else:
            raise ValueError("subtraction_mode must be 'field' or 'intensity'.")
        phases = phase_start + phase_span * np.arange(n_steps, dtype=float) / float(n_steps)
        powers = {
            "star": np.zeros(n_steps, dtype=float),
            "speckle": np.zeros(n_steps, dtype=float),
            "coherent": np.zeros(n_steps, dtype=float),
            "planet": np.zeros(n_steps, dtype=float),
            "incoherent": np.zeros(n_steps, dtype=float),
        }

        last_result: dict | None = None
        for idx, phase in enumerate(phases):
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase_index": int(idx + 1),
                        "phase_count": int(n_steps),
                        "phase_rad": float(phase),
                    }
                )
            result = self.propagate(
                slm_region_mask,
                include_star=include_star,
                include_planet=include_planet,
                include_speckles=include_speckles,
                selected_speckle_indices=selected_speckle_indices,
                star_planet_ratio=star_planet_ratio,
                star_speckle_ratio=star_speckle_ratio,
                phase_modulation_rad=float(phase),
                lyot_reference_scale=lyot_reference_scale,
            )
            last_result = result
            fields = result["fields"]
            if mode == "intensity":
                powers["star"][idx] = self._integrated_intensity_difference(
                    fields["star"]["raw_modulated"],
                    fields["star"]["reference"],
                )
                powers["speckle"][idx] = self._integrated_power(fields["speckle"]["raw_modulated"])
                powers["coherent"][idx] = self._integrated_intensity_difference(
                    fields["coherent"]["raw_modulated"],
                    fields["coherent"]["reference"],
                )
            else:
                powers["star"][idx] = self._integrated_power(fields["star"]["modulated"])
                powers["speckle"][idx] = self._integrated_power(fields["speckle"]["modulated"])
                powers["coherent"][idx] = self._integrated_power(fields["coherent"]["modulated"])
            powers["planet"][idx] = self._integrated_power(fields["planet"]["modulated"])
            powers["incoherent"][idx] = powers["star"][idx] + powers["planet"][idx]

        metrics = {
            name: self._phase_power_metrics(power, phases)
            for name, power in powers.items()
        }
        responses = {
            "speckle_response": powers["coherent"] - powers["star"],
            "planet_response": powers["incoherent"] - powers["star"],
        }
        metrics.update(
            {
                name: self._phase_power_metrics(response, phases)
                for name, response in responses.items()
            }
        )
        metrics["ratios"] = {
            "R_response_harmonic": self._safe_ratio(
                metrics["speckle_response"]["M_harmonic"],
                metrics["planet_response"]["M_harmonic"],
            ),
            "R_response_pp": self._safe_ratio(
                metrics["speckle_response"]["M_pp"],
                metrics["planet_response"]["M_pp"],
            ),
        }

        return {
            "phases": phases,
            "powers": powers,
            "responses": responses,
            "metrics": metrics,
            "phase_steps": n_steps,
            "phase_start_rad": phase_start,
            "phase_span_rad": phase_span,
            "subtraction_mode": mode,
            "lyot_reference_scale": (
                float(self.config.lyot_reference_scale)
                if lyot_reference_scale is None
                else float(lyot_reference_scale)
            ),
            "slm_region_mask": np.asarray(slm_region_mask, dtype=bool),
            "lyot_stop": self.lyot_stop,
            "speckle_offsets_lamD": (
                () if last_result is None else last_result["speckle_offsets_lamD"]
            ),
            "selected_speckle_indices": (
                () if last_result is None else last_result["selected_speckle_indices"]
            ),
            "star_planet_ratio": _positive_ratio(star_planet_ratio, "star_planet_ratio"),
            "star_speckle_ratio": float(star_speckle_ratio),
            "include_star": bool(include_star),
            "include_planet": bool(include_planet),
            "include_speckles": bool(include_speckles),
        }

    def calibrate_star_speckle_ratio_for_phase_zero_match(
        self,
        *,
        selected_speckle_indices: tuple[int, ...] | list[int] | None = None,
        star_planet_ratio: float = 500.0,
        lyot_reference_scale: float | None = None,
        subtraction_mode: str = "field",
    ) -> dict:
        """Find star/speckle ratio that matches coherent and incoherent power at phase zero."""
        star_planet_ratio = _positive_ratio(star_planet_ratio, "star_planet_ratio")
        mode = str(subtraction_mode).strip().lower().replace("_", "-")
        if mode in {"electric", "electric-field", "field", "field-subtraction"}:
            mode = "field"
        elif mode in {"intensity", "intensity-subtraction"}:
            mode = "intensity"
        else:
            raise ValueError("subtraction_mode must be 'field' or 'intensity'.")

        zero_mask = np.zeros((self.n_fft, self.n_fft), dtype=bool)
        star_only = self.propagate(
            zero_mask,
            include_star=True,
            include_planet=False,
            include_speckles=False,
            star_planet_ratio=star_planet_ratio,
            star_speckle_ratio=1.0,
            phase_modulation_rad=0.0,
            lyot_reference_scale=lyot_reference_scale,
        )
        planet_only = self.propagate(
            zero_mask,
            include_star=False,
            include_planet=True,
            include_speckles=False,
            star_planet_ratio=star_planet_ratio,
            star_speckle_ratio=1.0,
            phase_modulation_rad=0.0,
            lyot_reference_scale=lyot_reference_scale,
        )
        speckle_unit = self.propagate(
            zero_mask,
            include_star=False,
            include_planet=False,
            include_speckles=True,
            selected_speckle_indices=selected_speckle_indices,
            star_planet_ratio=star_planet_ratio,
            star_speckle_ratio=1.0,
            phase_modulation_rad=0.0,
            lyot_reference_scale=lyot_reference_scale,
        )

        if not speckle_unit["speckle_offsets_lamD"]:
            raise ValueError("At least one selected speckle is required for phase-zero matching.")

        if mode == "intensity":
            star_field = (
                np.asarray(star_only["fields"]["star"]["raw_modulated"], dtype=np.complex128)
                - np.asarray(star_only["fields"]["star"]["reference"], dtype=np.complex128)
            )
            speckle_field = np.asarray(
                speckle_unit["fields"]["speckle"]["raw_modulated"],
                dtype=np.complex128,
            )
        else:
            star_field = np.asarray(
                star_only["fields"]["star"]["modulated"],
                dtype=np.complex128,
            )
            speckle_field = np.asarray(
                speckle_unit["fields"]["speckle"]["modulated"],
                dtype=np.complex128,
            )
        planet_power = self._integrated_power(planet_only["fields"]["planet"]["modulated"])

        stop = self.lyot_stop.astype(bool)
        c = float(np.sum(np.abs(speckle_field[stop]) ** 2))
        b = float(np.real(np.sum(star_field[stop] * np.conj(speckle_field[stop]))))
        if c <= 0.0:
            raise ValueError("Selected speckle has zero Lyot-stop power at phase zero.")

        discriminant = b * b + c * planet_power
        if discriminant < 0.0:
            raise ValueError("No real phase-zero matching solution exists.")
        amplitude = (-b + float(np.sqrt(discriminant))) / c
        if amplitude <= 0.0 or not np.isfinite(amplitude):
            raise ValueError("No positive phase-zero matching solution exists.")

        ratio = 1.0 / (amplitude * amplitude)
        coherent_power = self._integrated_power(star_field + amplitude * speckle_field)
        incoherent_power = self._integrated_power(star_field) + planet_power
        return {
            "star_speckle_ratio": float(ratio),
            "speckle_amplitude": float(amplitude),
            "coherent_phase_zero_power": float(coherent_power),
            "incoherent_phase_zero_power": float(incoherent_power),
            "planet_phase_zero_power": float(planet_power),
            "selected_speckle_indices": (
                tuple(range(len(self._speckle_offsets())))
                if selected_speckle_indices is None
                else tuple(int(idx) for idx in selected_speckle_indices)
            ),
        }

    def comparison_sweeps(
        self,
        template_mask: np.ndarray,
        *,
        selected_speckle_index: int = 0,
        comparison_type: str = "separate",
        phase_steps: int = 8,
        phase_start_rad: float = 0.0,
        phase_span_rad: float = 2.0 * np.pi,
        subtraction_mode: str = "field",
        lyot_reference_scale: float | None = None,
        include_star: bool = True,
        include_planet: bool = True,
        include_speckles: bool = True,
        star_planet_ratio: float = 500.0,
        star_speckle_ratio: float = 100.0,
        progress_callback: Callable[[dict], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> dict:
        speckle_offsets = self._speckle_offsets()
        if not speckle_offsets:
            raise ValueError("At least one speckle position is required for comparison.")
        idx = int(selected_speckle_index)
        if idx < 0 or idx >= len(speckle_offsets):
            raise IndexError(f"selected_speckle_index must be between 0 and {len(speckle_offsets) - 1}.")
        selected_indices = (idx,)
        speckle_offset = speckle_offsets[idx]
        planet_offset = self.config.companion_offset_lamD

        if comparison_type not in {"separate", "shared_mask"}:
            raise ValueError("comparison_type must be separate or shared_mask.")
        if comparison_type == "shared_mask":
            # Preserve the editor's absolute pixels, including an off-source center.
            completed = {}
            placement = {
                "mask": np.asarray(template_mask, dtype=bool),
                "translation_pixels": (0, 0),
                "selected_pixels": int(np.count_nonzero(template_mask)),
            }
            if stop_requested is None or not stop_requested():
                completed["shared_mask"] = self.phase_sweep(
                    placement["mask"], phase_steps=phase_steps,
                    phase_start_rad=phase_start_rad, phase_span_rad=phase_span_rad,
                    subtraction_mode=subtraction_mode, lyot_reference_scale=lyot_reference_scale,
                    include_star=include_star, include_planet=include_planet,
                    include_speckles=include_speckles, selected_speckle_indices=selected_indices,
                    star_planet_ratio=star_planet_ratio, star_speckle_ratio=star_speckle_ratio,
                    progress_callback=progress_callback,
                )
            return self._comparison_result(completed, placement, placement, comparison_type="shared_mask")

        speckle_placement = self.place_mask_template(template_mask, speckle_offset)
        planet_placement = self.place_mask_template(template_mask, planet_offset)

        def stopped() -> bool:
            return bool(stop_requested is not None and stop_requested())

        completed: dict[str, dict] = {}
        if progress_callback is not None:
            progress_callback({"active_run": "speckle", "completed_runs": 0, "total_runs": 2})
        if stopped():
            return self._comparison_result(completed, speckle_placement, planet_placement)
        completed["speckle_position"] = self.phase_sweep(
            speckle_placement["mask"],
            phase_steps=phase_steps,
            phase_start_rad=phase_start_rad,
            phase_span_rad=phase_span_rad,
            subtraction_mode=subtraction_mode,
            lyot_reference_scale=lyot_reference_scale,
            include_star=include_star,
            include_planet=include_planet,
            include_speckles=include_speckles,
            selected_speckle_indices=selected_indices,
            star_planet_ratio=star_planet_ratio,
            star_speckle_ratio=star_speckle_ratio,
            progress_callback=(
                None
                if progress_callback is None
                else lambda update: progress_callback(
                    {"active_run": "speckle", "completed_runs": 0, "total_runs": 2, **update}
                )
            ),
        )

        if progress_callback is not None:
            progress_callback({"active_run": "planet", "completed_runs": 1, "total_runs": 2})
        if not stopped():
            completed["planet_position"] = self.phase_sweep(
                planet_placement["mask"],
                phase_steps=phase_steps,
                phase_start_rad=phase_start_rad,
                phase_span_rad=phase_span_rad,
                subtraction_mode=subtraction_mode,
                lyot_reference_scale=lyot_reference_scale,
                include_star=include_star,
                include_planet=include_planet,
                include_speckles=include_speckles,
                selected_speckle_indices=selected_indices,
                star_planet_ratio=star_planet_ratio,
                star_speckle_ratio=star_speckle_ratio,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda update: progress_callback(
                        {"active_run": "planet", "completed_runs": 1, "total_runs": 2, **update}
                    )
                ),
            )
        if progress_callback is not None:
            progress_callback({"active_run": "idle", "completed_runs": len(completed), "total_runs": 2})
        return self._comparison_result(completed, speckle_placement, planet_placement)

    def _comparison_result(
        self,
        sweeps: dict[str, dict],
        speckle_placement: dict,
        planet_placement: dict,
        *,
        comparison_type: str = "separate",
    ) -> dict:
        original_sweeps = sweeps
        shared = comparison_type == "shared_mask"
        if shared:
            sweeps = ({"speckle_position": sweeps["shared_mask"], "planet_position": sweeps["shared_mask"]}
                      if "shared_mask" in sweeps else {})
        metrics: dict[str, object] = {"complete": set(sweeps) == {"speckle_position", "planet_position"}}
        if "speckle_position" in sweeps:
            metrics["main_speckle_pp"] = float(
                sweeps["speckle_position"]["metrics"]["speckle_response"]["M_pp"]
            )
            metrics["speckle_run_planet_pp"] = float(
                sweeps["speckle_position"]["metrics"]["planet_response"]["M_pp"]
            )
        if "planet_position" in sweeps:
            metrics["main_planet_pp"] = float(
                sweeps["planet_position"]["metrics"]["planet_response"]["M_pp"]
            )
            metrics["planet_run_speckle_pp"] = float(
                sweeps["planet_position"]["metrics"]["speckle_response"]["M_pp"]
            )
            metrics["main_planet_mean"] = float(
                sweeps["planet_position"]["metrics"]["planet_response"]["P_mean"]
            )
        if "main_speckle_pp" in metrics and "main_planet_pp" in metrics:
            denominator = float(metrics["main_planet_pp"])
            metrics["ratio_pp"] = self._safe_ratio(float(metrics["main_speckle_pp"]), denominator)
            scale = max(abs(float(metrics["main_speckle_pp"])), 1.0)
            metrics["ratio_denominator_near_zero"] = bool(abs(denominator) <= scale * 1e-12)
        return {
            "measurement_plane": "lyot_stop",
            "primary_metric": "peak_to_peak",
            "comparison_type": comparison_type,
            "sweeps": original_sweeps,
            "metrics": metrics,
            "placements": ({"shared_mask": {key: value for key, value in speckle_placement.items() if key != "mask"}} if shared else {
                "speckle_position": {
                    key: value
                    for key, value in speckle_placement.items()
                    if key != "mask"
                },
                "planet_position": {
                    key: value
                    for key, value in planet_placement.items()
                    if key != "mask"
                },
            }),
        }

    def optimize_common_mask(
        self,
        initial_template_mask: np.ndarray,
        *,
        optimization_mode: str = "pixel_groups",
        comparison_type: str = "separate",
        measurement_plane: str = "lyot", science_radius_lamD: float = 0.5,
        selected_speckle_indices=None,
        circle_center_pixel: tuple[int, int] | None = None,
        circle_radius_min_px: int | None = None,
        circle_radius_max_px: int | None = None,
        ring_inner_radius_min_px: int = 10,
        ring_inner_radius_max_px: int = 30,
        ring_outer_radius_min_px: int = 30,
        ring_outer_radius_max_px: int = 50,
        ring_inner_radius_step_px: int = 5,
        ring_outer_radius_step_px: int = 5,
        selected_speckle_index: int = 0,
        phase_steps: int = 4,
        phase_start_rad: float = 0.0,
        phase_span_rad: float = 2.0 * np.pi,
        subtraction_mode: str = "field",
        lyot_reference_scale: float | None = None,
        include_star: bool = True,
        include_planet: bool = True,
        include_speckles: bool = True,
        star_planet_ratio: float = 500.0,
        star_speckle_ratio: float = 100.0,
        search_radius_px: int = 24,
        group_size_px: int = 4,
        iterations: int = 16,
        planet_penalty: float = 1.0,
        random_seed: int = 1,
        progress_callback: Callable[[dict], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> dict:
        plane = str(measurement_plane).lower()
        if plane not in {"lyot", "science"}:
            raise ValueError("Optimization measurement_plane must be 'lyot' or 'science'.")
        if plane == "science":
            selected_speckle_index = 0  # Score only s1 and the planet; other enabled light remains.
            enabled = set(range(len(self._speckle_offsets()))) if selected_speckle_indices is None else set(selected_speckle_indices)
            if not include_planet or not include_speckles or 0 not in enabled:
                raise ValueError("Science optimization requires s1 and the planet enabled.")
        template = np.asarray(initial_template_mask, dtype=bool)
        if template.shape != (self.n_fft, self.n_fft):
            raise ValueError(f"SLM template must have shape ({self.n_fft}, {self.n_fft}).")
        if comparison_type not in {"separate", "shared_mask"}:
            raise ValueError("comparison_type must be separate or shared_mask.")
        mode = str(optimization_mode).strip().lower().replace("-", "_")
        group = max(1, int(group_size_px))
        radius = max(1 if mode in {"circle", "circle_size", "circle_radius", "ring_radii"} else group,
                     int(search_radius_px))
        n_iter = max(1, int(iterations))
        rng = np.random.default_rng(int(random_seed))

        anchor = (int(round((self.n_fft - 1) / 2.0)), int(round((self.n_fft - 1) / 2.0)))
        cy, cx = anchor[1], anchor[0]
        y0 = max(0, cy - radius)
        y1 = min(self.n_fft, cy + radius + 1)
        x0 = max(0, cx - radius)
        x1 = min(self.n_fft, cx + radius + 1)
        block_origins = [
            (y, x)
            for y in range(y0, y1, group)
            for x in range(x0, x1, group)
        ]

        def stopped() -> bool:
            return bool(stop_requested is not None and stop_requested())

        def square_template(size: int) -> np.ndarray:
            mask = np.zeros_like(template, dtype=bool)
            half = max(1, int(size)) // 2
            mask[max(0, cy - half) : min(self.n_fft, cy + half + 1),
                 max(0, cx - half) : min(self.n_fft, cx + half + 1)] = True
            return mask

        candidates: list[tuple[str, np.ndarray]]
        ring_grid = None
        if mode in {"circle", "circle_size", "circle_radius"}:
            if circle_center_pixel is None:
                circle_center = anchor
            else:
                circle_center = (int(circle_center_pixel[0]), int(circle_center_pixel[1]))
            if not (
                0 <= circle_center[0] < self.n_fft
                and 0 <= circle_center[1] < self.n_fft
            ):
                raise ValueError("Circle center is outside the SLM grid.")
            current_radius = int(round(np.sqrt(np.count_nonzero(template) / np.pi)))
            if circle_radius_min_px is None:
                r_min = max(1, current_radius - radius)
            else:
                r_min = int(circle_radius_min_px)
            if circle_radius_max_px is None:
                r_max = max(r_min, current_radius + radius)
            else:
                r_max = int(circle_radius_max_px)
            if r_min < 1 or r_max < r_min:
                raise ValueError("Circle radius range must have 1 <= min radius <= max radius.")
            r_max = min(
                r_max,
                circle_center[0],
                circle_center[1],
                self.n_fft - 1 - circle_center[0],
                self.n_fft - 1 - circle_center[1],
            )
            if r_max < r_min:
                raise ValueError("Circle radius range clips at the SLM grid boundary.")
            radii = np.linspace(r_min, r_max, n_iter, dtype=float)
            radii = sorted({int(round(value)) for value in radii})
            yy, xx = np.indices(template.shape)
            candidates = []
            for candidate_radius in radii:
                mask = (
                    (xx - circle_center[0]) ** 2
                    + (yy - circle_center[1]) ** 2
                    <= int(candidate_radius) ** 2
                )
                candidates.append((f"circle_radius_{int(candidate_radius)}px", mask))
        elif mode == "ring_radii":
            center = anchor if circle_center_pixel is None else tuple(map(int, circle_center_pixel))
            inner_min, inner_max = int(ring_inner_radius_min_px), int(ring_inner_radius_max_px)
            outer_min, outer_max = int(ring_outer_radius_min_px), int(ring_outer_radius_max_px)
            inner_step = int(ring_inner_radius_step_px)
            outer_step = int(ring_outer_radius_step_px)
            if inner_step < 1 or outer_step < 1 or inner_min < 0 or inner_max < inner_min or outer_min < 1 or outer_max < outer_min:
                raise ValueError("Ring radius ranges must be increasing, nonnegative, with a positive step and outer radius.")
            boundary = min(center[0], center[1], self.n_fft - 1 - center[0], self.n_fft - 1 - center[1])
            if outer_max > boundary:
                raise ValueError("Ring outer radius range clips at the SLM grid boundary.")
            ring_grid = {"inner_radii_px": list(range(inner_min, inner_max + 1, inner_step)),
                         "outer_radii_px": list(range(outer_min, outer_max + 1, outer_step))}
            ring_pairs = [(inner, outer)
                          for outer in ring_grid["outer_radii_px"]
                          for inner in ring_grid["inner_radii_px"]
                          if inner < outer]
            if not ring_pairs:
                raise ValueError("Ring search needs at least one inner radius smaller than the outer radius.")
            yy, xx = np.indices(template.shape)
            distance2 = (xx - center[0])**2 + (yy - center[1])**2
            candidates = [(f"ring_inner_{inner}px_outer_{outer}px",
                           (distance2 >= inner**2) & (distance2 <= outer**2))
                          for inner, outer in ring_pairs]
        else:
            candidates = [
                ("empty_reference", np.zeros_like(template, dtype=bool)),
                ("center_block_reference", square_template(group)),
                ("initial_template", template.copy()),
            ]
            current = template.copy()
            for idx in range(n_iter):
                candidate = current.copy()
                toggle_count = max(1, min(len(block_origins), int(rng.integers(1, 4))))
                chosen = rng.choice(len(block_origins), size=toggle_count, replace=False)
                for choice in np.atleast_1d(chosen):
                    by, bx = block_origins[int(choice)]
                    block = np.s_[by : min(self.n_fft, by + group), bx : min(self.n_fft, bx + group)]
                    candidate[block] = ~candidate[block]
                candidates.append((f"random_toggle_{idx + 1}", candidate))
                current = candidate

        history: list[dict] = []
        best_index: int | None = None
        best_score = -np.inf
        for idx, (name, mask) in enumerate(candidates):
            if stopped():
                break
            if progress_callback is not None:
                progress_callback(
                    {
                        "active_run": "optimize",
                        "candidate": idx + 1,
                        "candidate_count": len(candidates),
                        "candidate_name": name,
                    }
                )
            if plane == "science":
                def science_progress(update):
                    if progress_callback is not None:
                        progress_callback({**update, "active_run": "science_optimize",
                            "placement": update.get("candidate"), "candidate": idx + 1,
                            "candidate_count": len(candidates), "candidate_name": name})
                comparison = self.science_comparison_sweeps(
                    mask, comparison_type=comparison_type, source_names=["speckle_1", "planet"],
                    measurement_region_names=["speckle_1", "planet"], radius_lamD=science_radius_lamD,
                    phase_steps=phase_steps, phase_start_rad=phase_start_rad, phase_span_rad=phase_span_rad,
                    include_star=include_star, include_planet=include_planet, include_speckles=include_speckles,
                    selected_speckle_indices=selected_speckle_indices,
                    star_planet_ratio=star_planet_ratio, star_speckle_ratio=star_speckle_ratio,
                    lyot_reference_scale=lyot_reference_scale, progress_callback=science_progress)
                own = comparison["own_region_metrics"]
                planet_run = comparison["sweeps"]["planet" if comparison_type == "separate" else "shared_mask"]
                ratio = self._safe_ratio(own["speckle_1"]["peak_to_peak"], own["planet"]["peak_to_peak"])
                comparison["metrics"] = {
                    "main_speckle_pp": own["speckle_1"]["peak_to_peak"],
                    "main_planet_pp": own["planet"]["peak_to_peak"], "ratio_pp": ratio,
                    "main_planet_mean": float(np.mean(planet_run["mean_intensities"]["planet"])),
                    "complete": True,
                    "ratio_denominator_near_zero": bool(own["planet"]["peak_to_peak"] <= 1e-12 * max(own["planet"]["maximum"], 1e-30)),
                }
            else:
                comparison = self.comparison_sweeps(
                    mask,
                    comparison_type=comparison_type,
                    selected_speckle_index=selected_speckle_index,
                    phase_steps=phase_steps,
                    phase_start_rad=phase_start_rad,
                    phase_span_rad=phase_span_rad,
                    subtraction_mode=subtraction_mode,
                    lyot_reference_scale=lyot_reference_scale,
                    include_star=include_star,
                    include_planet=include_planet,
                    include_speckles=include_speckles,
                    star_planet_ratio=star_planet_ratio,
                    star_speckle_ratio=star_speckle_ratio,
                    stop_requested=stop_requested,
                )
            metrics = comparison["metrics"]
            speckle_pp = float(metrics.get("main_speckle_pp", 0.0))
            planet_pp = float(metrics.get("main_planet_pp", np.inf))
            ratio_pp = metrics.get("ratio_pp")
            score = -np.inf if ratio_pp is None else float(ratio_pp)
            item = {
                "index": idx,
                "name": name,
                "mask_pixels": int(np.count_nonzero(mask)),
                "speckle_pp": speckle_pp,
                "planet_pp": planet_pp,
                "planet_mean": float(metrics.get("main_planet_mean", np.nan)),
                "ratio_pp": ratio_pp,
                "ratio_denominator_near_zero": bool(
                    metrics.get("ratio_denominator_near_zero", False)
                ),
                "score": float(score),
                "mask": mask.copy(),
                "comparison": comparison,
            }
            if mode in {"circle", "circle_size", "circle_radius"}:
                item["radius_px"] = int(radii[idx])
            elif mode == "ring_radii":
                item["inner_radius_px"], item["outer_radius_px"] = ring_pairs[idx]
            history.append(item)
            if score > best_score:
                best_score = float(score)
                best_index = len(history) - 1

        pareto_indices: list[int] = []
        for i, item in enumerate(history):
            dominated = False
            for j, other in enumerate(history):
                if i == j:
                    continue
                if (
                    other["speckle_pp"] >= item["speckle_pp"]
                    and other["planet_pp"] <= item["planet_pp"]
                    and (
                        other["speckle_pp"] > item["speckle_pp"]
                        or other["planet_pp"] < item["planet_pp"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(i)

        if progress_callback is not None:
            progress_callback({"active_run": "idle"})

        return {
            "measurement_plane": plane,
            "scored_regions": ["speckle_1", "planet"] if plane == "science" else None,
            "science_radius_lamD": float(science_radius_lamD) if plane == "science" else None,
            "star_planet_ratio": float(star_planet_ratio), "star_speckle_ratio": float(star_speckle_ratio),
            "selected_speckle_indices": None if selected_speckle_indices is None else list(selected_speckle_indices),
            "primary_metric": "peak_to_peak",
            "optimization_mode": mode,
            "comparison_type": comparison_type,
            "ring_grid": ring_grid,
            "search_radius_px": int(radius),
            "group_size_px": int(group),
            "iterations": int(n_iter),
            "planet_penalty": float(planet_penalty),
            "random_seed": int(random_seed),
            "history": history,
            "best_index": best_index,
            "pareto_indices": pareto_indices,
            "stopped": stopped(),
        }

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float | None:
        numerator = float(numerator)
        denominator = float(denominator)
        if denominator == 0.0:
            return None
        return numerator / denominator

    def _integrated_power(self, field: np.ndarray) -> float:
        stop = self.lyot_stop.astype(bool)
        values = np.asarray(field)
        return float(np.sum(np.abs(values[stop]) ** 2))

    def _integrated_intensity_difference(
        self,
        raw_field: np.ndarray,
        reference_field: np.ndarray,
    ) -> float:
        stop = self.lyot_stop.astype(bool)
        raw = np.asarray(raw_field, dtype=np.complex128)
        reference = np.asarray(reference_field, dtype=np.complex128)
        return float(np.sum(np.abs(raw[stop]) ** 2 - np.abs(reference[stop]) ** 2))

    @staticmethod
    def _phase_power_metrics(power: np.ndarray, phases: np.ndarray) -> dict[str, float]:
        values = np.asarray(power, dtype=float)
        phase_values = np.asarray(phases, dtype=float)
        mean = float(np.mean(values))
        centered = values - mean
        c1 = np.mean(values * np.exp(-1j * phase_values))
        metrics = {
            "P_mean": mean,
            "M_pp": float(np.max(values) - np.min(values)),
            "M_rms": float(np.sqrt(np.mean(centered ** 2))),
            "M_harmonic": float(np.abs(c1)),
            "phase_response": float(np.angle(c1)),
        }
        pi_indices = np.where(np.isclose(np.mod(phase_values, 2.0 * np.pi), np.pi))[0]
        zero_indices = np.where(np.isclose(np.mod(phase_values, 2.0 * np.pi), 0.0))[0]
        if zero_indices.size and pi_indices.size:
            metrics["delta_pi_minus_zero"] = float(values[pi_indices[0]] - values[zero_indices[0]])
        return metrics

    def _metrics(
        self,
        fields: dict[str, dict[str, np.ndarray]],
        include_star: bool,
        include_planet: bool,
        include_speckles: bool,
    ) -> dict[str, float]:
        stop = self.lyot_stop.astype(bool)

        def norm(name: str) -> float:
            delta = np.asarray(fields[name]["delta"], dtype=np.complex128)
            return float(np.sqrt(np.sum(np.abs(delta[stop]) ** 2)))

        metrics: dict[str, float] = {}
        if include_star:
            metrics["A_star"] = norm("star")
        if include_speckles:
            metrics["A_speckle"] = norm("speckle")
        if include_star or include_speckles:
            metrics["A_coherent"] = norm("coherent")
        if include_planet:
            metrics["A_planet"] = norm("planet")
        if (include_star or include_speckles) and include_planet:
            metrics["R"] = metrics["A_coherent"] / (metrics["A_planet"] + 1e-30)
        if include_planet:
            incoherent = np.asarray(fields["incoherent"]["delta"], dtype=float)
            metrics["A_incoherent"] = float(np.sqrt(np.sum(incoherent[stop] ** 2)))
        if (include_star or include_speckles) and include_planet:
            metrics["R1"] = metrics["A_coherent"] / (metrics["A_incoherent"] + 1e-30)
        return metrics


def _slm_science_npz_payload(result: dict) -> dict:
    """Export measured samples and fixed geometry without object/pickle arrays."""
    payload = {
        "phases_rad": result["phases"],
        "normalization_peak": result["normalization_peak"],
        "preview_intensity": result["preview_intensity"],
        "preview_phase_rad": result["preview_phase_rad"],
        "slm_region_mask": result["slm_region_mask"],
        "intensity_units": result["intensity_units"],
        "source_state_json": json.dumps(result["source_state"]),
    }
    payload["measurement_plane"] = result["measurement_plane"]
    payload["phase_zero_match_json"] = json.dumps(result.get("phase_zero_match"))
    for name in result["regions"]:
        payload[f"{name}_mean_intensity"] = result["mean_intensities"][name]
        for key, value in result["regions"][name].items():
            payload[f"{name}_{key}"] = value
        for key, value in result["metrics"][name].items():
            payload[f"{name}_{key}"] = value
    return payload


def save_slm_science_result(path: str | Path, result: dict) -> None:
    np.savez_compressed(path, **_slm_science_npz_payload(result))


def save_slm_science_comparison(path: str | Path, comparison: dict) -> None:
    payload = {"slm_template_mask": comparison["slm_template_mask"],
               "comparison_type": comparison["comparison_type"],
               "normalization_peak": comparison["normalization_peak"],
               "source_names": np.asarray(comparison["source_names"])}
    for placement, run in comparison["sweeps"].items():
        payload.update({f"{placement}__{key}": value for key, value in _slm_science_npz_payload(run).items()})
    def json_value(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, tuple):
            return list(value)
        raise TypeError(type(value).__name__)
    payload["comparison_metadata_json"] = json.dumps({
        key: value for key, value in comparison.items() if key not in {"sweeps", "slm_template_mask"}
    }, default=json_value)
    np.savez_compressed(path, **payload)


def save_slm_lyot_result(path: str | Path, result: dict) -> None:
    path = Path(path)
    fields = result["fields"]
    payload = {
        "slm_region_mask": np.asarray(result["slm_region_mask"], dtype=bool),
        "phase_map_rad": np.asarray(result["phase_map_rad"], dtype=float),
        "lyot_stop": np.asarray(result["lyot_stop"], dtype=float),
        "star_planet_ratio": np.asarray(result["star_planet_ratio"], dtype=float),
        "star_speckle_ratio": np.asarray(result["star_speckle_ratio"], dtype=float),
        "phase_modulation_rad": np.asarray(
            result.get("phase_modulation_rad", np.pi),
            dtype=float,
        ),
        "reference_phase_modulation_rad": np.asarray(
            result.get("reference_phase_modulation_rad", 0.0),
            dtype=float,
        ),
        "lyot_reference_scale": np.asarray(
            result.get("lyot_reference_scale", 1.0),
            dtype=float,
        ),
        "include_star": np.asarray(result["include_star"], dtype=bool),
        "include_planet": np.asarray(result["include_planet"], dtype=bool),
        "include_speckles": np.asarray(result["include_speckles"], dtype=bool),
        "speckle_offsets_lamD": np.asarray(result["speckle_offsets_lamD"], dtype=float),
        "selected_speckle_indices": np.asarray(
            result.get("selected_speckle_indices", ()),
            dtype=int,
        ),
    }
    for name in ("star", "planet", "speckle", "coherent"):
        payload[f"delta_E_L_{name}"] = np.asarray(fields[name]["delta"], dtype=np.complex128)
        payload[f"E_L_{name}_0"] = np.asarray(fields[name]["unmodulated"], dtype=np.complex128)
        payload[f"E_L_{name}_mod"] = np.asarray(fields[name]["modulated"], dtype=np.complex128)
    payload["delta_E_L_incoherent_amp"] = np.asarray(fields["incoherent"]["delta"], dtype=float)
    for key, value in result["metrics"].items():
        payload[key] = np.asarray(value, dtype=float)
    np.savez_compressed(path, **payload)


def launch_gui() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseButton
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle
    from matplotlib.ticker import MaxNLocator
    from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

    calculator = SLMLyotResponseCalculator(default_slm_gui_config())
    region = np.zeros((calculator.n_fft, calculator.n_fft), dtype=bool)
    undo_stack: list[np.ndarray] = []
    last_result: dict | None = None
    last_sweep: dict | None = None
    last_science: dict | None = None
    science_active = False
    science_window: dict = {}
    display_mode = {"value": "abs"}
    drawing_tool = {"value": "circle"}
    paint_mode = {"value": "paint"}
    brush_radius = {"value": 5}
    circle_start: tuple[int, int] | None = None
    last_draw_pixel: tuple[int, int] | None = None
    stroke_active = {"value": False}
    syncing_controls = {"value": False}

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.0, 1.0, 1.0, 0.68], height_ratios=[1.0, 1.0, 0.15])
    ax_mask = fig.add_subplot(gs[0:2, 0])
    ax_star = fig.add_subplot(gs[0, 1])
    ax_planet = fig.add_subplot(gs[0, 2])
    ax_speckle = fig.add_subplot(gs[1, 1])
    ax_coherent = fig.add_subplot(gs[1, 2])
    ax_results = fig.add_subplot(gs[0:2, 3])
    ax_results.axis("off")
    ax_phase_sweep = fig.add_subplot(gs[2, 1:3])

    slm_cmap = ListedColormap(["#f4f4f4", "#d7191c"])
    extent = [
        float(np.min(calculator.x_lamD)),
        float(np.max(calculator.x_lamD)),
        float(np.min(calculator.y_lamD)),
        float(np.max(calculator.y_lamD)),
    ]
    lyot_x = calculator._template._x / float(calculator.config.pupil_pixels)
    lyot_y = calculator._template._y / float(calculator.config.pupil_pixels)
    center = calculator.n_fft // 2
    lyot_half_width = max(2, int(round(0.75 * float(calculator.config.pupil_pixels))))
    lyot_row_slice = slice(max(0, center - lyot_half_width), min(calculator.n_fft, center + lyot_half_width + 1))
    lyot_col_slice = slice(max(0, center - lyot_half_width), min(calculator.n_fft, center + lyot_half_width + 1))
    lyot_view = np.s_[lyot_row_slice, lyot_col_slice]
    lyot_extent = [
        float(np.min(lyot_x[lyot_view])),
        float(np.max(lyot_x[lyot_view])),
        float(np.min(lyot_y[lyot_view])),
        float(np.max(lyot_y[lyot_view])),
    ]
    mask_image = ax_mask.imshow(
        region.astype(float),
        origin="lower",
        cmap=slm_cmap,
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
    )
    ax_mask.set_title("SLM pi region: 0 px")
    ax_mask.set_xlabel("x [lambda/D]")
    ax_mask.set_ylabel("y [lambda/D]")
    brush_cursor = Circle(
        (0.0, 0.0),
        radius=brush_radius["value"] / calculator.focal_sampling,
        fill=False,
        edgecolor="#1d70b8",
        linewidth=1.2,
        alpha=0.85,
        visible=False,
        zorder=10,
    )
    ax_mask.add_patch(brush_cursor)

    image_axes = {
        "star": (ax_star, "Delta E_L star"),
        "planet": (ax_planet, "Delta E_L planet"),
        "speckle": (ax_speckle, "Delta E_L speckle"),
        "coherent": (ax_coherent, "Delta E_L coherent"),
    }
    images = {}
    for key, (ax, title) in image_axes.items():
        images[key] = ax.imshow(
            np.zeros_like(region[lyot_view], dtype=float),
            origin="lower",
            cmap="viridis",
            extent=lyot_extent,
        )
        ax.contour(
            calculator.lyot_stop[lyot_view],
            levels=[0.5],
            colors="white",
            linewidths=0.8,
            extent=lyot_extent,
        )
        ax.set_title(title)
        ax.set_xlabel("x [pupil D]")
        ax.set_ylabel("y [pupil D]")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    ax_measurement_plane = fig.add_axes([0.78, 0.895, 0.20, 0.075])
    ax_measurement_plane.set_title("Intensity measurement plane", fontsize=9)
    measurement_plane_radio = RadioButtons(ax_measurement_plane, ["Lyot plane", "Final science"], active=1)

    ax_checks = fig.add_axes([0.78, 0.76, 0.12, 0.12])
    source_checks = CheckButtons(ax_checks, ["Star", "Planet", "Speckles"], [True, True, True])
    ax_ratio_planet = fig.add_axes([0.78, 0.69, 0.12, 0.04])
    planet_ratio_box = TextBox(ax_ratio_planet, "Star/Planet", initial="500")
    ax_phase_modulation = fig.add_axes([0.91, 0.69, 0.08, 0.04])
    phase_modulation_box = TextBox(ax_phase_modulation, "Phase", initial=f"{np.pi:.6f}")
    ax_reference_scale = fig.add_axes([0.78, 0.62, 0.08, 0.04])
    reference_scale_box = TextBox(ax_reference_scale, "Ref %", initial="80")
    ax_ratio_speckle = fig.add_axes([0.78, 0.665, 0.20, 0.018])
    ax_ratio_speckle.axis("off")
    ax_ratio_speckle.text(
        0.0,
        0.5,
        "Star/Speckle: set in Intensity window",
        va="center",
        fontsize=8,
    )
    ax_phase_steps = fig.add_axes([0.91, 0.62, 0.08, 0.04])
    phase_steps_box = TextBox(ax_phase_steps, "Steps", initial="6")
    ax_subtraction_mode = fig.add_axes([0.91, 0.555, 0.08, 0.055])
    subtraction_mode_radio = RadioButtons(ax_subtraction_mode, ["field", "intensity"])
    ax_tool = fig.add_axes([0.78, 0.48, 0.12, 0.11])
    tool_radio = RadioButtons(ax_tool, ["freehand", "circle", "ring", "rect", "sector"], active=1)
    ax_paint_mode = fig.add_axes([0.91, 0.48, 0.08, 0.08])
    paint_mode_radio = RadioButtons(ax_paint_mode, ["paint", "erase"])
    ax_brush = fig.add_axes([0.78, 0.445, 0.20, 0.025])
    brush_slider = Slider(ax_brush, "Brush px", 1, 25, valinit=brush_radius["value"], valstep=1)
    ax_mode = fig.add_axes([0.78, 0.32, 0.12, 0.13])
    mode_radio = RadioButtons(ax_mode, ["abs", "phase", "real", "imag"])
    fig.text(
        0.78,
        0.295,
        "Shape coords: mask[y, x]; x left->right, y bottom->top (origin=lower).",
        fontsize=7.5,
        color="#444444",
    )
    center_default = int(round((calculator.n_fft - 1) / 2.0))
    shape_boxes: dict[str, TextBox] = {
        "center_x": TextBox(fig.add_axes([0.78, 0.265, 0.08, 0.024]), "Cx px", initial=str(center_default)),
        "center_y": TextBox(fig.add_axes([0.90, 0.265, 0.08, 0.024]), "Cy px", initial=str(center_default)),
        "radius": TextBox(fig.add_axes([0.78, 0.235, 0.08, 0.024]), "R px", initial="30"),
        "inner": TextBox(fig.add_axes([0.90, 0.235, 0.08, 0.024]), "Rin", initial="20"),
        "outer": TextBox(fig.add_axes([0.78, 0.205, 0.08, 0.024]), "Rout", initial="40"),
        "width": TextBox(fig.add_axes([0.90, 0.205, 0.08, 0.024]), "W", initial="60"),
        "height": TextBox(fig.add_axes([0.78, 0.175, 0.08, 0.024]), "H", initial="40"),
        "start": TextBox(fig.add_axes([0.90, 0.175, 0.08, 0.024]), "A0", initial="0"),
        "end": TextBox(fig.add_axes([0.78, 0.145, 0.08, 0.024]), "A1", initial="90"),
    }
    shape_status = fig.text(0.90, 0.148, "", fontsize=7.5, color="#9b1c1c")

    button_specs = [
        ("Apply", [0.08, 0.04, 0.09, 0.04]),
        ("Clear", [0.19, 0.04, 0.09, 0.04]),
        ("Undo", [0.30, 0.04, 0.09, 0.04]),
        ("Save Mask", [0.41, 0.04, 0.10, 0.04]),
        ("Load Mask", [0.53, 0.04, 0.10, 0.04]),
        ("Save Result", [0.65, 0.04, 0.11, 0.04]),
        ("Eval Mod", [0.78, 0.04, 0.10, 0.04]),
        ("Save Log", [0.90, 0.04, 0.09, 0.04]),
        ("Save Fig", [0.90, 0.09, 0.09, 0.04]),
        ("Intensity", [0.78, 0.09, 0.10, 0.04]),
    ]
    buttons = {label: Button(fig.add_axes(pos), label) for label, pos in button_specs}

    def clear_phase_sweep_plot() -> None:
        ax_phase_sweep.clear()
        ax_phase_sweep.set_title("Integrated Lyot-stop power vs phase")
        ax_phase_sweep.set_xlabel("phase [rad]")
        ax_phase_sweep.set_ylabel("power")
        ax_phase_sweep.set_xlim(0.0, 2.0 * np.pi)
        ax_phase_sweep.set_xticks([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi])
        ax_phase_sweep.set_xticklabels(["0", "pi/2", "pi", "3pi/2", "2pi"])
        ax_phase_sweep.grid(True, alpha=0.25)

    clear_phase_sweep_plot()

    def component_for_display(field: np.ndarray) -> np.ndarray:
        if display_mode["value"] == "phase":
            return np.angle(field)
        if display_mode["value"] == "real":
            return np.real(field)
        if display_mode["value"] == "imag":
            return np.imag(field)
        return np.abs(field)

    def redraw_mask() -> None:
        mask_image.set_data(region.astype(float))
        ax_mask.set_title(f"SLM pi region: {int(np.count_nonzero(region))} px")
        fig.canvas.draw_idle()

    def redraw_cursor(pixel: tuple[int, int] | None) -> None:
        if pixel is None:
            brush_cursor.set_visible(False)
        else:
            x, y = pixel
            brush_cursor.center = (
                (x - (region.shape[1] - 1) / 2.0) / calculator.focal_sampling,
                (y - (region.shape[0] - 1) / 2.0) / calculator.focal_sampling,
            )
            brush_cursor.radius = brush_radius["value"] / calculator.focal_sampling
            brush_cursor.set_edgecolor("#1d70b8" if paint_mode["value"] == "paint" else "#222222")
        brush_cursor.set_visible(drawing_tool["value"] == "freehand")
        fig.canvas.draw_idle()

    def set_shape_status(message: str = "", error: bool = False) -> None:
        shape_status.set_text(message)
        shape_status.set_color("#9b1c1c" if error else "#444444")
        fig.canvas.draw_idle()

    def _parse_box(name: str, default: float) -> float:
        text = shape_boxes[name].text.strip()
        if text == "":
            return float(default)
        return float(text)

    def _shape_params() -> dict[str, float]:
        width = region.shape[1]
        height = region.shape[0]
        cx = int(round(_parse_box("center_x", (width - 1) / 2.0)))
        cy = int(round(_parse_box("center_y", (height - 1) / 2.0)))
        if cx < 0 or cx >= width or cy < 0 or cy >= height:
            raise ValueError(f"Center must satisfy 0 <= x < {width}, 0 <= y < {height}.")
        radius = max(0.0, _parse_box("radius", 30.0))
        inner = max(0.0, _parse_box("inner", 20.0))
        outer = max(0.0, _parse_box("outer", 40.0))
        if outer <= inner:
            raise ValueError("Outer radius must be > inner radius.")
        rect_width = max(0.0, _parse_box("width", 60.0))
        rect_height = max(0.0, _parse_box("height", 40.0))
        return {
            "center_x": float(cx),
            "center_y": float(cy),
            "radius": radius,
            "inner": inner,
            "outer": outer,
            "width": rect_width,
            "height": rect_height,
            "start": _parse_box("start", 0.0),
            "end": _parse_box("end", 90.0),
        }

    def _angle_sector_mask(theta_deg: np.ndarray, start_deg: float, end_deg: float) -> np.ndarray:
        start = float(start_deg) % 360.0
        end = float(end_deg) % 360.0
        theta = theta_deg % 360.0
        if np.isclose(start, end):
            return np.ones_like(theta, dtype=bool)
        if start < end:
            return (theta >= start) & (theta <= end)
        return (theta >= start) | (theta <= end)

    def build_parameterized_mask(tool: str, params: dict[str, float]) -> np.ndarray:
        yy, xx = np.indices(region.shape, dtype=float)
        cx = params["center_x"]
        cy = params["center_y"]
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        if tool == "circle":
            return rr <= params["radius"]
        if tool in {"ring", "annulus"}:
            return (rr >= params["inner"]) & (rr <= params["outer"])
        if tool == "rect":
            half_w = 0.5 * params["width"]
            half_h = 0.5 * params["height"]
            return (np.abs(xx - cx) <= half_w) & (np.abs(yy - cy) <= half_h)
        if tool == "sector":
            theta_deg = np.degrees(np.arctan2(yy - cy, xx - cx))
            return (
                (rr >= params["inner"])
                & (rr <= params["outer"])
                & _angle_sector_mask(theta_deg, params["start"], params["end"])
            )
        return np.zeros_like(region, dtype=bool)

    def update_mask_from_shape_controls(_text: str | None = None) -> None:
        if syncing_controls["value"] or drawing_tool["value"] == "freehand":
            return
        try:
            params = _shape_params()
            region[:, :] = build_parameterized_mask(drawing_tool["value"], params)
        except Exception as exc:
            set_shape_status(str(exc), error=True)
            return
        set_shape_status("shape preview updated")
        redraw_mask()

    def set_shape_center_fields(x: int, y: int) -> None:
        syncing_controls["value"] = True
        try:
            shape_boxes["center_x"].set_val(str(int(np.clip(x, 0, region.shape[1] - 1))))
            shape_boxes["center_y"].set_val(str(int(np.clip(y, 0, region.shape[0] - 1))))
        finally:
            syncing_controls["value"] = False

    def set_shape_numeric_field(name: str, value: float) -> None:
        syncing_controls["value"] = True
        try:
            shape_boxes[name].set_val(str(int(round(max(0.0, float(value))))))
        finally:
            syncing_controls["value"] = False

    def update_results_text(result: dict | None, message: str = "") -> None:
        ax_results.clear()
        ax_results.axis("off")
        if message:
            ax_results.text(0.0, 1.0, message, va="top", family="monospace")
        elif result is None:
            ax_results.text(0.0, 1.0, "Draw a region and press Apply.", va="top")
        else:
            rows = [f"{k} = {v:.6e}" for k, v in result["metrics"].items()]
            ax_results.text(0.0, 1.0, "\n".join(rows), va="top", family="monospace")
        fig.canvas.draw_idle()

    def update_phase_sweep_result(sweep: dict) -> None:
        clear_phase_sweep_plot()
        phases = np.asarray(sweep["phases"], dtype=float)
        plot_specs = (
            ("coherent", "coherent star + speckles", "#005ea8", 2.6, "-", "o", 1.0),
            ("incoherent", "incoherent star + planet", "#c51b29", 2.6, "--", "s", 1.0),
            ("star", "star", "#9f6b00", 1.2, ":", "^", 0.72),
            ("speckle", "speckles", "#7a3db8", 1.2, "-.", "D", 0.72),
            ("planet", "planet", "#007f91", 1.2, (0, (2, 3)), "x", 0.72),
        )
        for key, label, color, width, linestyle, marker, alpha in plot_specs:
            values = np.asarray(sweep["powers"][key], dtype=float)
            x = np.concatenate([phases, [2.0 * np.pi]])
            y = np.concatenate([values, [values[0]]])
            ax_phase_sweep.plot(
                x,
                y,
                label=label,
                color=color,
                linewidth=width,
                linestyle=linestyle,
                marker=marker,
                markersize=4.5 if width > 2 else 3.5,
                markerfacecolor="white",
                alpha=alpha,
            )
        primary = np.concatenate(
            [
                np.asarray(sweep["powers"]["coherent"], dtype=float),
                np.asarray(sweep["powers"]["incoherent"], dtype=float),
            ]
        )
        finite = primary[np.isfinite(primary)]
        if finite.size:
            ymin = float(np.min(finite))
            ymax = float(np.max(finite))
            span = ymax - ymin
            mean_scale = max(abs(ymin), abs(ymax), 1.0)
            if span < mean_scale * 1e-6:
                center = 0.5 * (ymin + ymax)
                span = mean_scale * 1e-6
                ymin = center - 0.5 * span
                ymax = center + 0.5 * span
            pad = 0.08 * (ymax - ymin)
            ax_phase_sweep.set_ylim(ymin - pad, ymax + pad)
        ax_phase_sweep.legend(loc="best", fontsize=7, framealpha=0.88)

        metrics = sweep["metrics"]

        def format_ratio(value: float | None) -> str:
            return "undefined" if value is None else f"{value:.6e}"

        rows = [
            f"R_response_harmonic = {format_ratio(metrics['ratios']['R_response_harmonic'])}",
            f"R_response_pp = {format_ratio(metrics['ratios']['R_response_pp'])}",
            "",
        ]
        for name in ("speckle_response", "planet_response", "coherent", "incoherent", "star", "speckle", "planet"):
            item = metrics[name]
            rows.extend(
                [
                    f"{name}:",
                    f"  M_pp = {item['M_pp']:.6e}",
                    f"  M_rms = {item['M_rms']:.6e}",
                    f"  M_harmonic = {item['M_harmonic']:.6e}",
                    f"  phase_response = {item['phase_response']:.6f}",
                ]
            )
        ax_results.clear()
        ax_results.axis("off")
        ax_results.text(0.0, 1.0, "\n".join(rows), va="top", family="monospace", fontsize=8)
        fig.canvas.draw_idle()

    def refresh_maps() -> None:
        if last_result is None:
            return
        for key, image in images.items():
            data = component_for_display(last_result["fields"][key]["delta"])[lyot_view]
            image.set_data(data)
            finite = data[np.isfinite(data)]
            if finite.size:
                image.set_clim(float(np.min(finite)), float(np.max(finite)))
        update_results_text(last_result)

    def scan_science_intensity(_event=None) -> None:
        nonlocal last_science, science_active
        from matplotlib.colors import LogNorm

        if measurement_plane_radio.value_selected == "Lyot plane":
            evaluate_modulation()
            return
        status = science_window["status"]
        try:
            states = source_checks.get_status()
            status.set_text("Scanning physical intensity...")
            science_window["figure"].canvas.draw_idle()
            measured = calculator.science_intensity_sweep(
                region,
                radius_lamD=float(science_window["radius"].text),
                measurement_plane="science",
                measure_all_companions=True,
                phase_steps=int(phase_steps_box.text),
                include_star=states[0], include_planet=states[1], include_speckles=states[2],
                star_planet_ratio=float(planet_ratio_box.text),
                star_speckle_ratio=float(science_window["speckle_ratio"].text),
                lyot_reference_scale=float(reference_scale_box.text) / 100.0,
            )
        except Exception as exc:
            status.set_text(f"Intensity scan failed: {exc}")
            science_window["figure"].canvas.draw_idle()
            return
        last_science = measured
        science_active = True
        science_window["figure"].set_size_inches(14, max(8, 4 + 0.45 * len(measured["regions"])))
        ax = science_window["image_axes"]
        ax.clear()
        vmax = max(measured["display_max"], 1e-29)
        image = ax.imshow(measured["preview_intensity"], origin="lower", extent=extent,
                          cmap="inferno", norm=LogNorm(vmin=vmax * 1e-8, vmax=vmax))
        science_window["colorbar_axes"].clear()
        science_window["figure"].colorbar(image, cax=science_window["colorbar_axes"],
                                        label="Intensity / unit-star direct PSF peak")
        colors = {name: ("#007f91" if name == "planet" else plt.get_cmap("tab10")(i % 10))
                  for i, name in enumerate(measured["regions"])}
        for name, aperture in measured["regions"].items():
            x, y = aperture["center_lamD"]
            radius = aperture["radius_lamD"]
            color = "#00e5ff" if name == "planet" else "#ffe000"
            ax.add_patch(Circle((x, y), radius, fill=False, edgecolor=color, linewidth=1.8))
            ax.annotate(name, (x, y + radius), xytext=(0, 5), textcoords="offset points",
                        ha="center", color=color, fontsize=10)
        half = max(abs(v) for r in measured["regions"].values() for v in r["center_lamD"]) + radius + 2
        ax.set_xlim(max(extent[0], -half), min(extent[1], half))
        ax.set_ylim(max(extent[2], -half), min(extent[3], half))
        ax.set(title=f"{measurement_plane_radio.value_selected}: phase {measured['preview_phase_rad']:.3f} rad",
               xlabel="x [lambda/D]", ylabel="y [lambda/D]")
        plot = science_window["curve_axes"]
        plot.clear()
        rows = []
        for name, color in colors.items():
            plot.plot(measured["phases"], measured["mean_intensities"][name], "o-",
                      label=name, color=color, markersize=4)
            item = measured["metrics"][name]
            rows.append(f"{name:12s} {item['maximum']:.5e}  {item['minimum']:.5e}  {item['peak_to_peak']:.5e}")
        plot.set(title="Mean pixel intensity in fixed circles", xlabel="SLM phase [rad]",
                 ylabel="Intensity / unit-star direct PSF peak", xlim=(0, 2 * np.pi))
        plot.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
        plot.legend()
        plot.grid(alpha=0.25)
        metrics = science_window["metrics_axes"]
        metrics.clear()
        metrics.axis("off")
        rows.insert(0, "Region          Maximum      Minimum      Peak-to-peak")
        metrics.text(0, 1, "\n".join(rows), va="top", family="monospace",
                     fontsize=max(5, min(9, 90 / len(rows))))
        status.set_text(
            f"{measurement_plane_radio.value_selected}; {len(measured['phases'])} phases; radius {radius:g} lambda/D. All companions; no background subtraction.\n"
            "Image shows first phase. Main Save Result / Save Log / Save Fig export this measurement.")
        science_window["figure"].canvas.draw_idle()

    def open_science_intensity(_event=None) -> None:
        nonlocal science_active
        if science_window and plt.fignum_exists(science_window["figure"].number):
            scan_science_intensity()
            science_window["figure"].show()
            return
        science_fig = plt.figure(figsize=(14, 8))
        science_fig.suptitle("Final science intensity measurement — all companions")
        science_window.clear()
        science_window.update({
            "figure": science_fig,
            "image_axes": science_fig.add_axes([0.06, 0.30, 0.34, 0.56]),
            "colorbar_axes": science_fig.add_axes([0.42, 0.30, 0.014, 0.56]),
            "curve_axes": science_fig.add_axes([0.59, 0.52, 0.38, 0.32]),
            "metrics_axes": science_fig.add_axes([0.53, 0.20, 0.45, 0.22]),
            "status": science_fig.text(0.07, 0.14, "", fontsize=9),
            "radius": TextBox(science_fig.add_axes([0.25, 0.055, 0.09, 0.045]), "ROI r [lambda/D] ", initial="0.5"),
            "speckle_ratio": TextBox(science_fig.add_axes([0.65, 0.055, 0.09, 0.045]), "Star/Speckle ", initial="100"),
            "scan": Button(science_fig.add_axes([0.82, 0.055, 0.12, 0.045]), "Scan Intensity"),
        })
        science_window["scan"].on_clicked(scan_science_intensity)
        scan_science_intensity()
        science_fig.show()

    def apply(_event=None) -> None:
        nonlocal last_result, science_active
        try:
            states = source_checks.get_status()
            last_result = calculator.propagate(
                region,
                include_star=states[0],
                include_planet=states[1],
                include_speckles=states[2],
                star_planet_ratio=float(planet_ratio_box.text),
                phase_modulation_rad=float(phase_modulation_box.text),
                lyot_reference_scale=float(reference_scale_box.text) / 100.0,
            )
        except Exception as exc:
            update_results_text(None, f"Propagation failed:\n{exc}")
            return
        science_active = False
        clear_phase_sweep_plot()
        refresh_maps()

    def evaluate_modulation(_event=None) -> None:
        nonlocal last_sweep, science_active
        if measurement_plane_radio.value_selected == "Final science":
            open_science_intensity()
            return
        try:
            states = source_checks.get_status()
            last_sweep = calculator.phase_sweep(
                region, phase_steps=int(phase_steps_box.text),
                subtraction_mode=subtraction_mode_radio.value_selected,
                lyot_reference_scale=float(reference_scale_box.text) / 100.0,
                include_star=states[0], include_planet=states[1], include_speckles=states[2],
                star_planet_ratio=float(planet_ratio_box.text))
        except Exception as exc:
            update_results_text(None, f"Lyot phase sweep failed:\n{exc}")
            return
        science_active = False
        update_phase_sweep_result(last_sweep)

    def save_current_mask(_event=None) -> None:
        np.save("interactive_slm_mask.npy", region)
        update_results_text(last_result, "Saved mask to interactive_slm_mask.npy")

    def load_current_mask(_event=None) -> None:
        nonlocal region
        loaded = np.load("interactive_slm_mask.npy")
        if loaded.shape != region.shape:
            update_results_text(last_result, f"Mask shape mismatch: {loaded.shape}")
            return
        undo_stack.append(region.copy())
        region = np.asarray(loaded, dtype=bool)
        redraw_mask()

    def save_current_result(_event=None) -> None:
        if science_active and last_science is not None:
            path = _slm_artifact_path(
                planet_offset_lamD=calculator.config.companion_offset_lamD,
                process_kind=f"{last_science['measurement_plane_key']}_intensity", suffix="result.npz")
            save_slm_science_result(path, last_science)
            update_results_text(last_result, f"Saved intensity result to {path}")
            return
        if last_result is None:
            update_results_text(None, "No propagation result to save.")
            return
        save_slm_lyot_result("interactive_slm_lyot_response.npz", last_result)
        update_results_text(last_result, "Saved result to interactive_slm_lyot_response.npz")

    def _jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, complex):
            return {"real": float(np.real(value)), "imag": float(np.imag(value))}
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return value

    def save_current_log(_event=None) -> None:
        states = source_checks.get_status()
        process_kind = f"{last_science['measurement_plane_key']}_intensity" if science_active else ("phase_sweep" if last_sweep is not None else "single")
        payload = {
            "gui": "interactive_slm_gui",
            "saved_at": datetime.now().isoformat(),
            "process_kind": process_kind,
            "n_fft": int(calculator.n_fft),
            "mask_pixels": int(np.count_nonzero(region)),
            "mask_fraction": float(np.mean(region)),
            "controls": {
                "include_star": bool(states[0]),
                "include_planet": bool(states[1]),
                "include_speckles": bool(states[2]),
                "star_planet_ratio": float(planet_ratio_box.text),
                "science_phase_modulation_rad": float(phase_modulation_box.text),
                "reference_subtraction_percent": float(reference_scale_box.text),
                "phase_steps": int(float(phase_steps_box.text)),
                "subtraction_mode": str(subtraction_mode_radio.value_selected),
                "measurement_plane": measurement_plane_radio.value_selected,
            },
            "single_phase_metrics": None if last_result is None else last_result["metrics"],
            "phase_sweep": last_sweep,
            "science_intensity": None if last_science is None else {
                k: v for k, v in last_science.items() if k not in {"preview_intensity", "slm_region_mask"}
            },
        }
        path = _slm_artifact_path(
            planet_offset_lamD=calculator.config.companion_offset_lamD,
            process_kind=process_kind,
            suffix="log.json",
        )
        path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        update_results_text(last_result, f"Saved log to {path}")

    def save_current_figure(_event=None) -> None:
        process_kind = f"{last_science['measurement_plane_key']}_intensity" if science_active else ("phase_sweep" if last_sweep is not None else "single")
        path = _slm_artifact_path(
            planet_offset_lamD=calculator.config.companion_offset_lamD,
            process_kind=process_kind,
            suffix="figure.png",
        )
        target_fig = science_window["figure"] if science_active else fig
        target_fig.savefig(path, dpi=180)
        update_results_text(last_result, f"Saved figure to {path}")

    def clear(_event=None) -> None:
        undo_stack.append(region.copy())
        region[:, :] = False
        redraw_mask()

    def undo(_event=None) -> None:
        if not undo_stack:
            return
        region[:, :] = undo_stack.pop()
        redraw_mask()

    def to_pixel(event) -> tuple[int, int] | None:
        if event.inaxes is not ax_mask or event.xdata is None or event.ydata is None:
            return None
        x = int(round(event.xdata * calculator.focal_sampling + (region.shape[1] - 1) / 2.0))
        y = int(round(event.ydata * calculator.focal_sampling + (region.shape[0] - 1) / 2.0))
        if x < 0 or y < 0 or x >= region.shape[1] or y >= region.shape[0]:
            return None
        return x, y

    def select_pixel_line(start: tuple[int, int], stop: tuple[int, int]) -> None:
        x0, y0 = start
        x1, y1 = stop
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        xs = np.rint(np.linspace(x0, x1, steps + 1)).astype(int)
        ys = np.rint(np.linspace(y0, y1, steps + 1)).astype(int)
        valid = (xs >= 0) & (ys >= 0) & (xs < region.shape[1]) & (ys < region.shape[0])
        xs = xs[valid]
        ys = ys[valid]
        if xs.size == 0:
            return
        r = int(max(1, brush_radius["value"]))
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        footprint_y, footprint_x = np.nonzero(xx * xx + yy * yy <= r * r)
        offsets_x = footprint_x - r
        offsets_y = footprint_y - r
        target_x = xs[:, None] + offsets_x[None, :]
        target_y = ys[:, None] + offsets_y[None, :]
        inside = (
            (target_x >= 0)
            & (target_y >= 0)
            & (target_x < region.shape[1])
            & (target_y < region.shape[0])
        )
        if paint_mode["value"] == "erase":
            region[target_y[inside], target_x[inside]] = False
        else:
            region[target_y[inside], target_x[inside]] = True

    def paint(pixel: tuple[int, int]) -> None:
        nonlocal last_draw_pixel
        if last_draw_pixel is None:
            select_pixel_line(pixel, pixel)
        else:
            select_pixel_line(last_draw_pixel, pixel)
        last_draw_pixel = pixel
        redraw_mask()

    def paint_event(event) -> None:
        pix = to_pixel(event)
        if pix is None:
            return
        redraw_cursor(pix)
        paint(pix)

    def on_press(event) -> None:
        nonlocal circle_start, last_draw_pixel
        if event.button not in {MouseButton.LEFT, MouseButton.RIGHT}:
            return
        pix = to_pixel(event)
        if pix is None:
            return
        undo_stack.append(region.copy())
        stroke_active["value"] = True
        if event.button is MouseButton.RIGHT:
            paint_mode["value"] = "erase"
        elif event.button is MouseButton.LEFT:
            paint_mode["value"] = "paint"
        if drawing_tool["value"] == "circle":
            circle_start = pix
            set_shape_center_fields(*pix)
            update_mask_from_shape_controls()
        elif drawing_tool["value"] in {"ring", "annulus", "rect", "sector"}:
            circle_start = pix
            set_shape_center_fields(*pix)
            update_mask_from_shape_controls()
        else:
            last_draw_pixel = None
            redraw_cursor(pix)
            paint(pix)

    def on_motion(event) -> None:
        pix = to_pixel(event)
        redraw_cursor(pix)
        if stroke_active["value"] and drawing_tool["value"] == "freehand" and pix is not None:
            paint_event(event)
        elif stroke_active["value"] and drawing_tool["value"] in {"circle", "ring", "annulus", "rect", "sector"} and pix is not None and circle_start is not None:
            x0, y0 = circle_start
            x1, y1 = pix
            dist = float(np.hypot(x1 - x0, y1 - y0))
            if drawing_tool["value"] == "circle":
                set_shape_numeric_field("radius", dist)
            elif drawing_tool["value"] in {"ring", "annulus", "sector"}:
                set_shape_numeric_field("outer", dist)
            elif drawing_tool["value"] == "rect":
                set_shape_numeric_field("width", 2.0 * abs(x1 - x0))
                set_shape_numeric_field("height", 2.0 * abs(y1 - y0))
            update_mask_from_shape_controls()

    def on_release(event) -> None:
        nonlocal circle_start, last_draw_pixel
        stroke_active["value"] = False
        last_draw_pixel = None
        if drawing_tool["value"] not in {"circle", "ring", "annulus", "rect", "sector"} or circle_start is None:
            return
        circle_start = None
        update_mask_from_shape_controls()

    def set_tool(label: str) -> None:
        drawing_tool["value"] = label
        redraw_cursor(None)
        update_mask_from_shape_controls()

    def set_paint_mode(label: str) -> None:
        paint_mode["value"] = label

    def set_brush_radius(value: float) -> None:
        brush_radius["value"] = int(round(float(value)))
        brush_cursor.radius = brush_radius["value"] / calculator.focal_sampling
        fig.canvas.draw_idle()

    def set_mode(label: str) -> None:
        display_mode["value"] = label
        refresh_maps()

    buttons["Intensity"].on_clicked(evaluate_modulation)
    buttons["Apply"].on_clicked(apply)
    buttons["Eval Mod"].on_clicked(evaluate_modulation)
    buttons["Clear"].on_clicked(clear)
    buttons["Undo"].on_clicked(undo)
    buttons["Save Mask"].on_clicked(save_current_mask)
    buttons["Load Mask"].on_clicked(load_current_mask)
    buttons["Save Result"].on_clicked(save_current_result)
    buttons["Save Log"].on_clicked(save_current_log)
    buttons["Save Fig"].on_clicked(save_current_figure)
    tool_radio.on_clicked(set_tool)
    paint_mode_radio.on_clicked(set_paint_mode)
    brush_slider.on_changed(set_brush_radius)
    mode_radio.on_clicked(set_mode)
    for box in shape_boxes.values():
        if hasattr(box, "on_text_change"):
            box.on_text_change(update_mask_from_shape_controls)
        box.on_submit(update_mask_from_shape_controls)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    update_results_text(None)
    plt.show()


if __name__ == "__main__":
    launch_gui()
