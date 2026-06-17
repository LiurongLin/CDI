from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

if __package__:
    from .coc_feature import run_coc_planet_phase
    from .masks import PhaseMask, RoddierPhaseMask, VortexPhaseMask
    from .plotting import (
        plot_local_region0_peak_fft,
        plot_local_region_phase_peak_metrics,
        plot_phase_offset_combined_metrics,
        plot_phase_offset_metrics,
        plot_results,
        save_phase_mask_fits,
    )
    from .region_shapes import normalize_region_shape
    from .simulator import CoronagraphSimulator, resolve_phase_screen_path
    from .sweeps import sweep_roddier_phase_for_peak_match, sweep_roddier_radius_for_peak_match
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from coronagraph.coc_feature import run_coc_planet_phase
    from coronagraph.masks import PhaseMask, RoddierPhaseMask, VortexPhaseMask
    from coronagraph.plotting import (
        plot_local_region0_peak_fft,
        plot_local_region_phase_peak_metrics,
        plot_phase_offset_combined_metrics,
        plot_phase_offset_metrics,
        plot_results,
        save_phase_mask_fits,
    )
    from coronagraph.simulator import CoronagraphSimulator, resolve_phase_screen_path
    from coronagraph.sweeps import (
        sweep_roddier_phase_for_peak_match,
        sweep_roddier_radius_for_peak_match,
    )


def _parse_region_shape(value: str) -> str:
    return normalize_region_shape(value)


def default_sim_kwargs() -> dict:
    return dict(
        pupil_pixels=100,
        focal_sampling=10,
        phase_mask_sampling=10,
        phase_mask=RoddierPhaseMask(radius_lamD=0.53),
        lyot_scale=1,
        ghost_fraction=0.005,
        ghost_source="phase_mask_refraction",
        ghost_offset_lamD=(0.0, 0.0),
        focal_shift_pixels=(0.5, 0.5),
        ghost_phase_rad=0.0,
        ghost_coherence=1.0,
        include_ghost=True,
        include_interference=True,
        include_companion_ghost=True,
        companion_flux_ratio=0.0,
        companion_offset_lamD=(0.0, 0.0),
        e_final_phase_offset=np.pi,
        secondary_diameter_ratio=0.0,
        spider_width_pixels=0.0,
        spider_angles_deg=(0.0, 90.0),
        pupil_supersample=1,
        phase_screen_path=None,
        phase_screen_index=0,
    )


def mask_filename_suffix(mask: PhaseMask) -> str:
    """Return a filename-safe suffix describing mask parameters."""
    if isinstance(mask, RoddierPhaseMask):
        radius = f"{mask.radius_lamD:.4f}".replace(".", "p")
        phase = f"{mask.phase_rad:.6f}".replace(".", "p")
        return f"_radius_{radius}_phase_{phase}"
    if isinstance(mask, VortexPhaseMask):
        return f"_charge_{int(mask.charge)}"
    return ""


def float_filename_token(value: float, precision: int = 3) -> str:
    """Format a float into a filename-safe token."""
    return f"{float(value):.{int(precision)}f}".replace(".", "p")


def build_phase_mask(args: argparse.Namespace) -> PhaseMask:
    mask_type = str(args.phase_mask_type).lower()
    if mask_type == "roddier":
        return RoddierPhaseMask(
            radius_lamD=float(args.roddier_mask_radius),
            phase_rad=float(args.roddier_mask_phase),
        )
    if mask_type == "vortex":
        return VortexPhaseMask(charge=int(args.vortex_charge))
    raise ValueError(f"Unsupported phase-mask type: {mask_type}")


def print_run_header(result: dict) -> None:
    print(f"FFT grid size: {result['n_fft']} x {result['n_fft']}")
    print(f"Focal-plane sampling: {result['focal_sampling']} px/(λ/D)")
    print(f"Phase-mask sampling: {result['phase_mask_sampling']} px/(λ/D)")
    print(f"Phase mask: {result['phase_mask_name']}")
    print(f"Ghost fraction: {result['ghost_fraction'] * 100:.3f}% of {result['ghost_source']} PSF")
    print(f"Ghost offset: {result['ghost_offset_lamD']} λ/D")
    print(f"Global focal shift: {result['focal_shift_pixels']} px")
    print(f"Ghost phase: {result['ghost_phase_rad']:.3f} rad")
    print(f"Ghost coherence gamma: {result['ghost_coherence']:.3f}")
    print(f"Ghost enabled: {result.get('include_ghost', True)}")
    print(f"Interference enabled: {result.get('include_interference', True)}")
    print(f"Companion ghost enabled: {result.get('include_companion_ghost', True)}")
    print(f"Companion flux ratio: {result['companion_flux_ratio']:.3e}")
    print(f"Companion offset: {result['companion_offset_lamD']} λ/D")
    print("e_final_phase_offset: {:.3f} rad".format(result["e_final_phase_offset"]))
    print(f"Secondary diameter ratio: {result['secondary_diameter_ratio']:.3f}")
    print(f"Spider width: {result['spider_width_pixels']:.3f} px")
    print(f"Spider angles: {result['spider_angles_deg']} deg")
    print(f"Pupil supersampling: {result['pupil_supersample']}x")
    if "incoherence_map_mode" in result:
        print(f"Incoherence map mode: {result['incoherence_map_mode']}")
    print(
        "Pupil phase screen: "
        f"{result['phase_screen_path'] if result['phase_screen_path'] is not None else 'disabled'}"
    )
    print(f"Phase screen index: {result['phase_screen_index']}")


def print_progress_bar(
    completed: int,
    total: int,
    start_time: float,
    prefix: str = "Progress",
    width: int = 34,
) -> None:
    if total <= 0:
        return
    frac = min(max(float(completed) / float(total), 0.0), 1.0)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.perf_counter() - start_time, 0.0)
    if completed > 0:
        eta = elapsed * (float(total - completed) / float(completed))
    else:
        eta = float("inf")
    eta_str = f"{eta:6.1f}s" if np.isfinite(eta) else "  inf s"
    msg = (
        f"\r{prefix} [{bar}] {completed:>3}/{total:<3} "
        f"{100.0 * frac:5.1f}%  elapsed {elapsed:6.1f}s  eta {eta_str}"
    )
    print(msg, end="" if completed < total else "\n", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected coronagraph simulation features independently."
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch desktop GUI runner instead of command-line execution.",
    )
    parser.add_argument(
        "--feature",
        nargs="+",
        choices=[
            "single",
            "phase",
            "combined",
            "radius-match",
            "phase-match",
            "local-region-phase",
            "local-region-phase-ft",
            "coc-planet-phase",
            "all",
        ],
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
    parser.add_argument(
        "--phase-mask-type",
        choices=["roddier", "vortex"],
        default="roddier",
        help="Phase mask model used by features (single/phase/combined/local/coc).",
    )
    parser.add_argument(
        "--roddier-mask-radius",
        type=float,
        default=0.53,
        help="Roddier mask radius [λ/D] when --phase-mask-type roddier.",
    )
    parser.add_argument(
        "--roddier-mask-phase",
        type=float,
        default=np.pi,
        help="Roddier mask phase shift [rad] when --phase-mask-type roddier.",
    )
    parser.add_argument(
        "--vortex-charge",
        type=int,
        default=2,
        help="Vortex phase-mask charge when --phase-mask-type vortex.",
    )
    parser.add_argument(
        "--planet-flux-ratio",
        type=float,
        default=0.0,
        help="Incoherent companion intensity ratio relative to star (0 disables).",
    )
    parser.add_argument(
        "--disable-ghost",
        action="store_true",
        help="Disable ghost contribution entirely.",
    )
    parser.add_argument(
        "--disable-interference",
        action="store_true",
        help="Disable coherent interference term (keeps ghost intensity-only if ghost is enabled).",
    )
    parser.add_argument(
        "--disable-companion-ghost",
        action="store_true",
        help="Disable ghost and companion self-interference for the companion branch only.",
    )
    parser.add_argument(
        "--planet-offset-x",
        type=float,
        default=0.0,
        help="Companion x offset [λ/D].",
    )
    parser.add_argument(
        "--planet-offset-y",
        type=float,
        default=0.0,
        help="Companion y offset [λ/D].",
    )
    parser.add_argument(
        "--secondary-ratio",
        type=float,
        default=0.25,
        help="Secondary mirror diameter / primary diameter in [0,1).",
    )
    parser.add_argument(
        "--spider-width",
        type=float,
        default=0.25,
        help="Spider vane width in pupil-grid pixels.",
    )
    parser.add_argument(
        "--spider-angles",
        type=float,
        nargs="+",
        default=[0.0, 90.0],
        help="Spider vane angles in degrees (e.g. 0 60 120).",
    )
    parser.add_argument(
        "--pupil-ss",
        type=int,
        default=8,
        help="Entrance-pupil supersampling factor per axis (>=1).",
    )
    parser.add_argument(
        "--phase-screen-jitter",
        choices=["none", "0", "5", "10", "20"],
        default="none",
        help=(
            "Entrance-pupil phase-screen jitter choice. "
            "Uses the first screen in the selected FITS cube."
        ),
    )
    parser.add_argument(
        "--incoherence-map-mode",
        choices=["fft_band", "lab_fft_ratio"],
        default="fft_band",
        help=(
            "How to build simulation incoherence maps: "
            "'fft_band' keeps the existing low-frequency FFT-band sum, "
            "'lab_fft_ratio' uses the lab-style inverse coherence ratio from the "
            "strongest non-DC FFT peak."
        ),
    )
    parser.add_argument(
        "--local-region-radius",
        type=float,
        default=2,
        help="Circular region radius [λ/D] for localized phase sweep.",
    )
    parser.add_argument(
        "--local-phase-cycles",
        type=float,
        default=1.0,
        help="Number of full 2π local-phase cycles (e.g., 8 -> 0 to 16π).",
    )
    parser.add_argument(
        "--phase-sweep-mode",
        choices=["regional", "global"],
        default="regional",
        help="Phase sweep mode: regional (local focal-plane regions) or global (e_final_phase_offset).",
    )
    parser.add_argument(
        "--local-outward-step",
        type=float,
        default=2.0,
        help="Radial outward spacing [λ/D] when moving 3 auto-detected regions.",
    )
    parser.add_argument(
        "--local-keep-index",
        type=int,
        default=0,
        help="Index [0..3] of the detected region to keep at its original position.",
    )
    parser.add_argument(
        "--local-align-reference-azimuth",
        action="store_true",
        help="Align moved regions on the kept region azimuth direction.",
    )
    parser.add_argument(
        "--local-region-centers",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Manual region centers in λ/D. "
            "Provide 2*N floats for N FOVs: x1 y1 ... xN yN."
        ),
    )
    parser.add_argument(
        "--region-shape",
        type=_parse_region_shape,
        default="circle",
        help="Local FOV region shape: 'circle', 'ring', or 'ring_of_circle'.",
    )
    parser.add_argument(
        "--fov-count",
        type=int,
        default=1,
        help="Number of FOVs phase-shifted simultaneously at each phase step.",
    )
    parser.add_argument(
        "--fov-centers-count",
        type=int,
        default=1,
        help="Total number of FOV centers explored sequentially.",
    )
    parser.add_argument(
        "--single-region-ring-radius",
        type=float,
        default=None,
        help="Ring radius [λ/D] used when auto-expanding FOV centers.",
    )
    parser.add_argument(
        "--ring-rotation-fraction",
        type=float,
        default=0.0,
        help="Normalized ring_of_circle rotation in [0,1]: 0=centered on planet, 1=edge passes through planet center.",
    )
    parser.add_argument(
        "--ring-rotation-sweep",
        action="store_true",
        help="Enable ring_of_circle rotation sweep from 0 to a max fraction with a fixed step.",
    )
    parser.add_argument(
        "--ring-rotation-sweep-max",
        type=float,
        default=1.0,
        help="Maximum ring_of_circle rotation fraction for the sweep.",
    )
    parser.add_argument(
        "--ring-rotation-sweep-step",
        type=float,
        default=0.1,
        help="Step size for the ring_of_circle rotation sweep.",
    )
    parser.add_argument(
        "--phase-step",
        type=int,
        default=61,
        help="Number of phase steps for each ROI.",
    )
    parser.add_argument(
        "--phase-cycles",
        type=float,
        default=1.0,
        help="Number of phase cycles per ROI.",
    )
    parser.add_argument(
        "--planet-offset-x-local",
        type=float,
        default=2.5,
        help="Planet x offset [λ/D] for the ROI-phase simulation.",
    )
    parser.add_argument(
        "--planet-offset-y-local",
        type=float,
        default=-2.5,
        help="Planet y offset [λ/D] for the ROI-phase simulation.",
    )
    parser.add_argument(
        "--secondary-ratio-local",
        type=float,
        default=None,
        help="Secondary ratio for the ROI-phase simulation. Defaults to --secondary-ratio.",
    )
    parser.add_argument(
        "--planet-flux-ratio-local",
        type=float,
        default=1e-2,
        help="Planet intensity ratio relative to star for the ROI-phase simulation.",
    )
    parser.add_argument(
        "--roi-size-sweep",
        action="store_true",
        help="Enable ROI-size sweep. When enabled, local-region-radius is ignored.",
    )
    parser.add_argument(
        "--roi-size-min",
        type=float,
        default=0.5,
        help="Minimum ROI radius [λ/D] for ROI-size sweep.",
    )
    parser.add_argument(
        "--roi-size-max",
        type=float,
        default=3.0,
        help="Maximum ROI radius [λ/D] for ROI-size sweep.",
    )
    parser.add_argument(
        "--roi-size-step",
        type=float,
        default=0.25,
        help="ROI radius step [λ/D] for ROI-size sweep.",
    )
    parser.add_argument(
        "--planet-position-roi-size-sweep",
        action="store_true",
        help="Enable a 2D sweep over planet (radius, theta) location and ROI size.",
    )
    parser.add_argument(
        "--planet-position-radius-min",
        type=float,
        default=0.5,
        help="Minimum planet radius [λ/D] for the 2D planet-position sweep.",
    )
    parser.add_argument(
        "--planet-position-radius-max",
        type=float,
        default=4.0,
        help="Maximum planet radius [λ/D] for the 2D planet-position sweep.",
    )
    parser.add_argument(
        "--planet-position-radius-step",
        type=float,
        default=0.5,
        help="Planet radius step [λ/D] for the 2D planet-position sweep.",
    )
    parser.add_argument(
        "--planet-position-theta-min-deg",
        type=float,
        default=-180.0,
        help="Minimum planet theta [deg] for the 2D planet-position sweep.",
    )
    parser.add_argument(
        "--planet-position-theta-max-deg",
        type=float,
        default=180.0,
        help="Maximum planet theta [deg] for the 2D planet-position sweep.",
    )
    parser.add_argument(
        "--planet-position-theta-step-deg",
        type=float,
        default=15.0,
        help="Planet theta step [deg] for the 2D planet-position sweep.",
    )
    parser.add_argument(
        "--planet-diagonal-roi-size-sweep",
        action="store_true",
        help="Enable a diagonal-only sweep over planet location and ROI size.",
    )
    parser.add_argument(
        "--planet-diagonal-mode",
        choices=["anti", "main"],
        default="anti",
        help="Diagonal to follow: 'anti' uses y=-x, 'main' uses y=x.",
    )
    parser.add_argument(
        "--planet-diagonal-t-min",
        type=float,
        default=0.5,
        help="Minimum diagonal parameter t [λ/D] for diagonal sweep.",
    )
    parser.add_argument(
        "--planet-diagonal-t-max",
        type=float,
        default=4.0,
        help="Maximum diagonal parameter t [λ/D] for diagonal sweep.",
    )
    parser.add_argument(
        "--planet-diagonal-t-step",
        type=float,
        default=0.5,
        help="Diagonal parameter step [λ/D] for diagonal sweep.",
    )
    parser.add_argument("--coc-phase-samples", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--coc-phase-cycles", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--coc-planet-offset-x", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--coc-planet-offset-y", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--coc-secondary-ratio", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--coc-planet-flux-ratio", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--build-map-per-fov",
        action="store_true",
        help="Build and save 16x16 lambda/D incoherence maps per active FOV period as a PDF.",
    )
    parser.add_argument("--coc-fov-position-steps", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--coc-fov-circle-of-circles-trace", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _resolve_features(args: argparse.Namespace) -> set[str]:
    features = set(args.feature)
    if "all" in features:
        return {
            "single",
            "phase",
            "combined",
            "radius-match",
            "phase-match",
            "local-region-phase",
            "local-region-phase-ft",
            "coc-planet-phase",
        }
    return features


def _build_sim_kwargs(args: argparse.Namespace) -> dict:
    sim_kwargs = default_sim_kwargs()
    sim_kwargs["secondary_diameter_ratio"] = float(args.secondary_ratio)
    sim_kwargs["spider_width_pixels"] = float(args.spider_width)
    sim_kwargs["spider_angles_deg"] = tuple(float(a) for a in args.spider_angles)
    sim_kwargs["pupil_supersample"] = int(args.pupil_ss)
    sim_kwargs["phase_screen_path"] = resolve_phase_screen_path(args.phase_screen_jitter)
    sim_kwargs["phase_screen_index"] = 0
    sim_kwargs["companion_flux_ratio"] = float(args.planet_flux_ratio)
    sim_kwargs["companion_offset_lamD"] = (float(args.planet_offset_x), float(args.planet_offset_y))
    sim_kwargs["include_ghost"] = not bool(args.disable_ghost)
    sim_kwargs["include_interference"] = (not bool(args.disable_interference)) and sim_kwargs["include_ghost"]
    sim_kwargs["include_companion_ghost"] = not bool(args.disable_companion_ghost)
    sim_kwargs["phase_mask"] = build_phase_mask(args)
    return sim_kwargs


def _build_output_tags(args: argparse.Namespace, sim_kwargs: dict) -> tuple[str, str, str, str, str]:
    mask_suffix = mask_filename_suffix(sim_kwargs["phase_mask"])
    if isinstance(sim_kwargs["phase_mask"], VortexPhaseMask):
        mask_output_tag = f"vortex_charge_{int(sim_kwargs['phase_mask'].charge)}"
    else:
        mask_output_tag = f"{sim_kwargs['phase_mask'].__class__.__name__}{mask_suffix}"
    effective_cycles = float(args.phase_cycles)
    phase_cycles_tag = f"_cycles_{float_filename_token(effective_cycles, precision=3)}"
    phase_sweep_mode_tag = f"_mode_{str(args.phase_sweep_mode).strip().lower()}"
    region_shape = str(args.region_shape).strip().lower()
    if region_shape == "ring_of_circle":
        rotation_tag = float_filename_token(float(getattr(args, "ring_rotation_fraction", 0.0)), precision=3)
        single_region_tag = f"_shape_{region_shape}_rotfrac_{rotation_tag}"
    else:
        single_region_tag = (
            f"_fovsim_{int(args.fov_count)}_fovcenters_{int(args.fov_centers_count)}"
            f"_shape_{region_shape}"
        )
    ghost_suffix = f"_ghost_{'on' if sim_kwargs['include_ghost'] else 'off'}"
    return mask_output_tag, phase_cycles_tag, phase_sweep_mode_tag, single_region_tag, ghost_suffix


def _parse_manual_centers(args: argparse.Namespace) -> list[tuple[float, float]] | None:
    if args.local_region_centers is None:
        return None
    n_vals = len(args.local_region_centers)
    expected_vals = 2 * int(args.fov_centers_count)
    if n_vals != expected_vals:
        raise ValueError(f"--local-region-centers requires exactly {expected_vals} floats for the selected mode.")
    vals = [float(v) for v in args.local_region_centers]
    return [(vals[i], vals[i + 1]) for i in range(0, n_vals, 2)]


def main() -> None:
    args = parse_args()
    if args.coc_phase_samples is not None:
        args.phase_step = int(args.coc_phase_samples)
    if args.coc_phase_cycles is not None:
        args.phase_cycles = float(args.coc_phase_cycles)
    if args.coc_planet_offset_x is not None:
        args.planet_offset_x_local = float(args.coc_planet_offset_x)
    if args.coc_planet_offset_y is not None:
        args.planet_offset_y_local = float(args.coc_planet_offset_y)
    if args.coc_secondary_ratio is not None:
        args.secondary_ratio_local = float(args.coc_secondary_ratio)
    if args.coc_planet_flux_ratio is not None:
        args.planet_flux_ratio_local = float(args.coc_planet_flux_ratio)
    if bool(args.gui):
        if __package__:
            from .gui import main as gui_main
        else:
            from coronagraph.gui import main as gui_main
        gui_main()
        return
    features = _resolve_features(args)
    sim_kwargs = _build_sim_kwargs(args)
    mask_output_tag, phase_cycles_tag, phase_sweep_mode_tag, single_region_tag, ghost_suffix = _build_output_tags(
        args, sim_kwargs
    )

    result = None
    if {"single", "phase", "combined"} & features:
        result = CoronagraphSimulator(**sim_kwargs).run()
        print_run_header(result)

    if "single" in features:
        out = f"coronagraph_simulation_{sim_kwargs['e_final_phase_offset']}_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png"
        plot_results(result, save_path=out)
        phase_mask_fits = f"phase_mask_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.fits"
        save_phase_mask_fits(result, fits_path=phase_mask_fits)
        print(f"Saved simulation plot: {out}")
        print(f"Saved phase mask FITS: {phase_mask_fits}")

    if "phase" in features:
        out = (
            "phase_offset_peak_total_intensity_"
            f"phase_mask_{mask_output_tag}_"
            f"ghost_fraction_{result['ghost_fraction'] * 100:.3f}%"
            f"{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png"
        )
        plot_phase_offset_metrics(sim_kwargs=sim_kwargs, n_phase_samples=args.phase_samples, save_path=out)
        print(f"Saved phase-offset sweep plot: {out}")

    if "combined" in features:
        out = (
            "phase_offset_combined_peak_total_intensity_"
            f"phase_mask_{mask_output_tag}_"
            f"ghost_fraction_{result['ghost_fraction'] * 100:.3f}%"
            f"{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png"
        )
        plot_phase_offset_combined_metrics(sim_kwargs=sim_kwargs, n_phase_samples=args.phase_samples, save_path=out)
        print(f"Saved combined phase-offset plot: {out}")

    if "radius-match" in features:
        out = f"roddier_radius_peak_match_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png"
        match = sweep_roddier_radius_for_peak_match(
            sim_kwargs=sim_kwargs,
            radius_min=args.radius_min,
            radius_max=args.radius_max,
            n_radius_samples=args.radius_samples,
            phase_rad=np.pi,
            save_path=out,
        )
        print(f"Saved radius sweep plot: {out}")
        print(f"Best Roddier radius_lamD: {match['best_radius_lamD']:.4f} λ/D")

    if "phase-match" in features:
        out = f"roddier_phase_peak_match_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png"
        match = sweep_roddier_phase_for_peak_match(
            sim_kwargs=sim_kwargs,
            radius_lamD=args.roddier_radius,
            phase_min_rad=args.phase_match_min,
            phase_max_rad=args.phase_match_max,
            n_phase_samples=args.phase_match_samples,
            save_path=out,
        )
        print(f"Saved phase sweep plot: {out}")
        print(f"Fixed Roddier radius_lamD: {match['radius_lamD']:.4f} λ/D")
        print(f"Best phase_rad: {match['best_phase_rad']:.6f} rad")

    if "local-region-phase" in features:
        out = f"local_region_phase_peak_intensity_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png"
        sweep = plot_local_region_phase_peak_metrics(
            sim_kwargs=sim_kwargs,
            n_phase_samples=args.phase_samples,
            phase_min_rad=0.0,
            phase_max_rad=2.0 * np.pi * float(args.local_phase_cycles),
            region_radius_lamD=args.local_region_radius,
            outward_step_lamD=args.local_outward_step,
            keep_region_index=args.local_keep_index,
            align_to_reference_azimuth=args.local_align_reference_azimuth,
            region_centers_lamD=_parse_manual_centers(args),
            phase_sweep_mode=str(args.phase_sweep_mode),
            region_shape=str(args.region_shape),
            fov_count=int(args.fov_count),
            single_region_ring_radius_lamD=args.single_region_ring_radius,
            single_region_step_diameter_fraction=0.25,
            ring_rotation_fraction=float(args.ring_rotation_fraction),
            save_path=out,
        )
        print(f"Saved localized region phase sweep plot: {out}")
        print(f"Phase application plane: {sweep['phase_application_plane']}")

    if "local-region-phase-ft" in features:
        out = f"local_regions_center_pixel_fft_{mask_output_tag}{phase_cycles_tag}{phase_sweep_mode_tag}{single_region_tag}{ghost_suffix}.png"
        fft_result = plot_local_region0_peak_fft(
            sim_kwargs=sim_kwargs,
            n_phase_samples=args.phase_samples,
            phase_min_rad=0.0,
            phase_max_rad=2.0 * np.pi * float(args.local_phase_cycles),
            region_radius_lamD=args.local_region_radius,
            outward_step_lamD=args.local_outward_step,
            keep_region_index=args.local_keep_index,
            align_to_reference_azimuth=args.local_align_reference_azimuth,
            region_centers_lamD=_parse_manual_centers(args),
            phase_sweep_mode=str(args.phase_sweep_mode),
            region_shape=str(args.region_shape),
            fov_count=int(args.fov_count),
            single_region_ring_radius_lamD=args.single_region_ring_radius,
            single_region_step_diameter_fraction=0.25,
            ring_rotation_fraction=float(args.ring_rotation_fraction),
            save_path=out,
        )
        print(f"Saved all-region FFT plot: {out}")
        print(f"Sampled {len(fft_result['region_center_pixels_yx'])} center pixels.")

    if "coc-planet-phase" in features:
        run_coc_planet_phase(
            args=args,
            sim_kwargs=sim_kwargs,
            mask_output_tag=mask_output_tag,
            phase_cycles_tag=phase_cycles_tag,
            phase_sweep_mode_tag=phase_sweep_mode_tag,
            single_region_tag=single_region_tag,
            ghost_suffix=ghost_suffix,
            print_progress_bar=print_progress_bar,
            float_filename_token=float_filename_token,
        )


if __name__ == "__main__":
    main()
