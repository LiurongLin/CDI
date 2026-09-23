from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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


def _positive_ratio(value: float, name: str) -> float:
    ratio = float(value)
    if ratio <= 0.0:
        raise ValueError(f"{name} must be > 0.")
    return ratio


class SLMLyotResponseCalculator:
    """Compute component-wise Lyot-plane response to an arbitrary focal-plane pi mask."""

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
            speckle_amp = 1.0 / np.sqrt(star_planet_ratio)
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

    def phase_sweep(
        self,
        slm_region_mask: np.ndarray,
        *,
        phase_steps: int = 4,
        subtraction_mode: str = "field",
        lyot_reference_scale: float | None = None,
        include_star: bool = True,
        include_planet: bool = True,
        include_speckles: bool = False,
        selected_speckle_indices: tuple[int, ...] | list[int] | None = None,
        star_planet_ratio: float = 500.0,
        star_speckle_ratio: float = 100.0,
    ) -> dict:
        n_steps = int(phase_steps)
        if n_steps < 2:
            raise ValueError("phase_steps must be >= 2.")
        mode = str(subtraction_mode).strip().lower().replace("_", "-")
        if mode in {"electric", "electric-field", "field", "field-subtraction"}:
            mode = "field"
        elif mode in {"intensity", "intensity-subtraction"}:
            mode = "intensity"
        else:
            raise ValueError("subtraction_mode must be 'field' or 'intensity'.")
        phases = 2.0 * np.pi * np.arange(n_steps, dtype=float) / float(n_steps)
        powers = {
            "star": np.zeros(n_steps, dtype=float),
            "speckle": np.zeros(n_steps, dtype=float),
            "coherent": np.zeros(n_steps, dtype=float),
            "planet": np.zeros(n_steps, dtype=float),
            "incoherent": np.zeros(n_steps, dtype=float),
        }

        last_result: dict | None = None
        for idx, phase in enumerate(phases):
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
        metrics["ratios"] = {
            "R_mod_harmonic": metrics["coherent"]["M_harmonic"]
            / (metrics["incoherent"]["M_harmonic"] + 1e-30),
            "R_mod_pp": metrics["coherent"]["M_pp"]
            / (metrics["incoherent"]["M_pp"] + 1e-30),
        }

        return {
            "phases": phases,
            "powers": powers,
            "metrics": metrics,
            "phase_steps": n_steps,
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

    calculator = SLMLyotResponseCalculator()
    region = np.zeros((calculator.n_fft, calculator.n_fft), dtype=bool)
    undo_stack: list[np.ndarray] = []
    last_result: dict | None = None
    last_sweep: dict | None = None
    display_mode = {"value": "abs"}
    drawing_tool = {"value": "freehand"}
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

    ax_checks = fig.add_axes([0.78, 0.76, 0.12, 0.12])
    source_checks = CheckButtons(ax_checks, ["Star", "Planet", "Speckles"], [True, True, False])
    ax_ratio_planet = fig.add_axes([0.78, 0.69, 0.12, 0.04])
    planet_ratio_box = TextBox(ax_ratio_planet, "Star/Planet", initial="500")
    ax_phase_modulation = fig.add_axes([0.91, 0.69, 0.08, 0.04])
    phase_modulation_box = TextBox(ax_phase_modulation, "Phase", initial=f"{np.pi:.6f}")
    ax_reference_scale = fig.add_axes([0.78, 0.62, 0.08, 0.04])
    reference_scale_box = TextBox(ax_reference_scale, "Ref %", initial="100")
    ax_ratio_speckle = fig.add_axes([0.78, 0.665, 0.20, 0.018])
    ax_ratio_speckle.axis("off")
    ax_ratio_speckle.text(
        0.0,
        0.5,
        "Speckle intensity = planet",
        va="center",
        fontsize=8,
    )
    ax_phase_steps = fig.add_axes([0.78, 0.555, 0.08, 0.04])
    phase_steps_box = TextBox(ax_phase_steps, "Steps", initial="4")
    ax_subtraction_mode = fig.add_axes([0.91, 0.555, 0.08, 0.055])
    subtraction_mode_radio = RadioButtons(ax_subtraction_mode, ["field", "intensity"])
    ax_tool = fig.add_axes([0.78, 0.48, 0.12, 0.11])
    tool_radio = RadioButtons(ax_tool, ["freehand", "circle", "ring", "rect", "sector"])
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
        rows = [
            f"R_mod_harmonic = {metrics['ratios']['R_mod_harmonic']:.6e}",
            f"R_mod_pp = {metrics['ratios']['R_mod_pp']:.6e}",
            "",
        ]
        for name in ("coherent", "incoherent", "star", "speckle", "planet"):
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

    def apply(_event=None) -> None:
        nonlocal last_result
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
        clear_phase_sweep_plot()
        refresh_maps()

    def evaluate_modulation(_event=None) -> None:
        nonlocal last_sweep
        try:
            states = source_checks.get_status()
            last_sweep = calculator.phase_sweep(
                region,
                phase_steps=int(float(phase_steps_box.text)),
                subtraction_mode=subtraction_mode_radio.value_selected,
                lyot_reference_scale=float(reference_scale_box.text) / 100.0,
                include_star=states[0],
                include_planet=states[1],
                include_speckles=states[2],
                star_planet_ratio=float(planet_ratio_box.text),
            )
        except Exception as exc:
            update_results_text(None, f"Phase sweep failed:\n{exc}")
            return
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
        payload = {
            "gui": "interactive_slm_gui",
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
            },
            "single_phase_metrics": None if last_result is None else last_result["metrics"],
            "phase_sweep": last_sweep,
        }
        path = Path("interactive_slm_lyot_log.json")
        path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        update_results_text(last_result, f"Saved log to {path}")

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

    buttons["Apply"].on_clicked(apply)
    buttons["Eval Mod"].on_clicked(evaluate_modulation)
    buttons["Clear"].on_clicked(clear)
    buttons["Undo"].on_clicked(undo)
    buttons["Save Mask"].on_clicked(save_current_mask)
    buttons["Load Mask"].on_clicked(load_current_mask)
    buttons["Save Result"].on_clicked(save_current_result)
    buttons["Save Log"].on_clicked(save_current_log)
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
