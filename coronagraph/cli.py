from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

if __package__:
    from .cdi_feature import run_cdi_planet_phase
    from .masks import NoPhaseMask, PhaseMask, RoddierPhaseMask, VortexPhaseMask
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
    from .roddier_sweeps import sweep_roddier_phase_for_peak_match, sweep_roddier_radius_for_peak_match
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from coronagraph.cdi_feature import run_cdi_planet_phase
    from coronagraph.masks import NoPhaseMask, PhaseMask, RoddierPhaseMask, VortexPhaseMask
    from coronagraph.plotting import (
        plot_local_region0_peak_fft,
        plot_local_region_phase_peak_metrics,
        plot_phase_offset_combined_metrics,
        plot_phase_offset_metrics,
        plot_results,
        save_phase_mask_fits,
    )
    from coronagraph.simulator import CoronagraphSimulator, resolve_phase_screen_path
    from coronagraph.roddier_sweeps import (
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
        include_star=True,
        include_companion=True,
        include_companion_ghost=True,
        coherent_ring_speckle_count=0,
        coherent_ring_speckle_intensity=1.0,
        companion_flux_ratio=0.0,
        companion_offset_lamD=(0.0, 0.0),
        e_final_phase_offset=np.pi,
        secondary_diameter_ratio=0.0,
        spider_width_pixels=0.0,
        spider_angles_deg=(0.0, 90.0),
        pupil_supersample=1,
        phase_screen_path=None,
        phase_screen_index=0,
        lyot_reference_scale=1.0,
    )


def mask_filename_suffix(mask: PhaseMask) -> str:
    """Return a filename-safe suffix describing mask parameters."""
    if isinstance(mask, RoddierPhaseMask):
        radius = f"{mask.radius_lamD:.4f}".replace(".", "p")
        phase = f"{mask.phase_rad:.6f}".replace(".", "p")
        return f"_r{radius}_p{phase}"
    if isinstance(mask, VortexPhaseMask):
        return f"_c{int(mask.charge)}"
    return ""


def float_filename_token(value: float, precision: int = 3) -> str:
    """Format a float into a filename-safe token."""
    return f"{float(value):.{int(precision)}f}".replace(".", "p")


def _shape_tag(region_shape: str) -> str:
    return {
        "circle": "cir",
        "ring": "rng",
        "ring_of_circle": "roc",
    }.get(str(region_shape).strip().lower(), str(region_shape).strip().lower())


def _mode_tag(mode_name: str) -> str:
    return {
        "regional": "reg",
        "global": "gbl",
        "focal_plane": "fpl",
        "mask_rotation": "mrot",
    }.get(str(mode_name).strip().lower(), str(mode_name).strip().lower())


def build_phase_mask(args: argparse.Namespace) -> PhaseMask:
    mask_type = str(args.phase_mask_type).lower()
    if mask_type == "roddier":
        return RoddierPhaseMask(
            radius_lamD=float(args.roddier_mask_radius),
            phase_rad=float(args.roddier_mask_phase),
        )
    if mask_type == "vortex":
        return VortexPhaseMask(charge=int(args.vortex_charge))
    if mask_type in {"perfect_corongraph", "perfect_coronagraph"}:
        return NoPhaseMask()
    raise ValueError(f"Unsupported phase-mask type: {mask_type}")


def print_run_header(result: dict) -> None:
    print(f"FFT grid size: {result['n_fft']} x {result['n_fft']}")
    print(f"Focal-plane sampling: {result['focal_sampling']} px/(λ/D)")
    print(f"Phase-mask sampling: {result['phase_mask_sampling']} px/(λ/D)")
    print(f"Phase mask: {result['phase_mask_name']}")
    print(f"Perfect coronagraph: {result.get('perfect_coronagraph', False)}")
    if result.get("perfect_coronagraph", False):
        print(f"Lyot reference subtraction: {result.get('lyot_reference_scale', 1.0) * 100.0:.3f}%")
    print(f"Ghost fraction: {result['ghost_fraction'] * 100:.3f}% of {result['ghost_source']} PSF")
    print(f"Ghost offset: {result['ghost_offset_lamD']} λ/D")
    print(f"Global focal shift: {result['focal_shift_pixels']} px")
    print(f"Ghost phase: {result['ghost_phase_rad']:.3f} rad")
    print(f"Ghost coherence gamma: {result['ghost_coherence']:.3f}")
    print(f"Ghost enabled: {result.get('include_ghost', True)}")
    print(f"Interference enabled: {result.get('include_interference', True)}")
    print(f"Star enabled: {result.get('include_star', True)}")
    print(f"Companion enabled: {result.get('include_companion', True)}")
    print(f"Companion ghost enabled: {result.get('include_companion_ghost', True)}")
    print(f"Companion flux ratio: {result['companion_flux_ratio']:.3e}")
    print(f"Companion offset: {result['companion_offset_lamD']} λ/D")
    print(f"Coherent ring speckle count: {result.get('coherent_ring_speckle_count', 0)}")
    print(f"Coherent ring speckle intensity: {result.get('coherent_ring_speckle_intensity', 1.0):.3e}")
    if result.get("coherent_ring_speckle_offsets_lamD"):
        print(f"Coherent ring speckle offsets: {result['coherent_ring_speckle_offsets_lamD']} λ/D")
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
        "--interactive-slm-gui",
        action="store_true",
        help="Launch the interactive focal-plane SLM/Lyot diagnostic GUI.",
    )
    parser.add_argument(
        "--html-slm-gui",
        action="store_true",
        help="Launch the browser-based focal-plane SLM/Lyot diagnostic GUI.",
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
            "cdi-planet-phase",
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
        choices=["roddier", "vortex", "perfect_corongraph", "perfect_coronagraph"],
        default="roddier",
        help="Phase mask model used by features (single/phase/combined/local/cdi).",
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
        "--disable-star",
        action="store_true",
        help="Disable the stellar branch while keeping the companion/planet branch.",
    )
    parser.add_argument(
        "--disable-companion",
        action="store_true",
        help=(
            "Disable the companion/planet branch while keeping its flux ratio as the "
            "coherent ring speckle calibration target."
        ),
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
        "--disable-spiders",
        action="store_true",
        help="Disable spider vanes in the entrance pupil.",
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
        "--coherent-ring-speckle-count",
        type=int,
        default=0,
        help=(
            "Number of extra coherent speckles to place on the same angular-separation ring "
            "as the planet, equally spaced in azimuth."
        ),
    )
    parser.add_argument(
        "--coherent-ring-speckle-intensity",
        type=float,
        default=1.0,
        help=(
            "Input intensity of each coherent ring speckle relative to the star source. "
            "Use 1.0 for one star intensity."
        ),
    )
    parser.add_argument(
        "--lyot-reference-percent",
        type=float,
        default=100.0,
        help=(
            "Percentage of the perfect-coronagraph reference wavefront to subtract at the Lyot plane. "
            "Used only with --phase-mask-type perfect_corongraph/perfect_coronagraph."
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
        choices=["regional", "global", "focal_plane", "mask_rotation"],
        default="regional",
        help=(
            "Phase sweep mode: regional (local focal-plane regions), "
            "global/focal_plane (whole focal-plane phase offset), or mask_rotation "
            "(rotate the focal-plane phase mask once per run)."
        ),
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
        "--planet-position-map-sweep",
        action="store_true",
        help="Enable a planet-location sweep with fixed ROI size and fixed planet flux ratio.",
    )
    parser.add_argument(
        "--planet-position-roi-size-sweep",
        action="store_true",
        help="Enable a 2D sweep over planet (radius, theta) location and ROI size.",
    )
    parser.add_argument(
        "--lyot-reference-percent-sweep-min",
        type=float,
        default=100.0,
        help="Minimum Lyot-reference subtraction percentage for the planet-position/ROI-size sweep.",
    )
    parser.add_argument(
        "--lyot-reference-percent-sweep-max",
        type=float,
        default=100.0,
        help="Maximum Lyot-reference subtraction percentage for the planet-position/ROI-size sweep.",
    )
    parser.add_argument(
        "--lyot-reference-percent-sweep-step",
        type=float,
        default=0.0,
        help="Lyot-reference subtraction percentage step for the planet-position/ROI-size sweep.",
    )
    parser.add_argument(
        "--planet-flux-ratio-map-sweep",
        action="store_true",
        help="Enable a planet-brightness sweep with fixed location and fixed ROI size.",
    )
    parser.add_argument(
        "--mask-rotation-phase-step-sweep",
        action="store_true",
        help="Enable a mask-rotation step-count sweep with fixed planet location, brightness, and ROI size.",
    )
    parser.add_argument(
        "--planet-position-brightness-sweep",
        action="store_true",
        help="Enable a sweep over planet (radius, theta) location and planet flux ratio.",
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
        "--planet-flux-ratio-sweep-min",
        type=float,
        default=0.001,
        help="Minimum planet flux ratio for the planet-position/brightness sweep.",
    )
    parser.add_argument(
        "--planet-flux-ratio-sweep-max",
        type=float,
        default=0.010,
        help="Maximum planet flux ratio for the planet-position/brightness sweep.",
    )
    parser.add_argument(
        "--planet-flux-ratio-sweep-step",
        type=float,
        default=0.001,
        help="Planet flux-ratio step for the planet-position/brightness sweep.",
    )
    parser.add_argument(
        "--phase-step-sweep-min",
        type=int,
        default=4,
        help="Minimum phase-step count for the mask-rotation step sweep.",
    )
    parser.add_argument(
        "--phase-step-sweep-max",
        type=int,
        default=24,
        help="Maximum phase-step count for the mask-rotation step sweep.",
    )
    parser.add_argument(
        "--phase-step-sweep-step",
        type=int,
        default=4,
        help="Phase-step increment for the mask-rotation step sweep.",
    )
    parser.add_argument("--cdi-phase-samples", "--coc-phase-samples", dest="cdi_phase_samples", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cdi-phase-cycles", "--coc-phase-cycles", dest="cdi_phase_cycles", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cdi-planet-offset-x", "--coc-planet-offset-x", dest="cdi_planet_offset_x", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cdi-planet-offset-y", "--coc-planet-offset-y", dest="cdi_planet_offset_y", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cdi-secondary-ratio", "--coc-secondary-ratio", dest="cdi_secondary_ratio", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cdi-planet-flux-ratio", "--coc-planet-flux-ratio", dest="cdi_planet_flux_ratio", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--build-map-per-fov",
        action="store_true",
        help="Build and save 16x16 lambda/D incoherence maps per active FOV period as a PDF.",
    )
    parser.add_argument(
        "--plot-poster-figure",
        action="store_true",
        help="For the 2D planet-position ROI sweep, emit poster-styled coherence/incoherence PDFs only.",
    )
    parser.add_argument("--cdi-fov-position-steps", "--coc-fov-position-steps", dest="cdi_fov_position_steps", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--cdi-fov-orbit-trace", "--cdi-fov-circle-of-circles-trace", "--coc-fov-circle-of-circles-trace", dest="cdi_fov_orbit_trace", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _resolve_features(args: argparse.Namespace) -> set[str]:
    features = {("cdi-planet-phase" if item == "coc-planet-phase" else item) for item in args.feature}
    if "all" in features:
        return {
            "single",
            "phase",
            "combined",
            "radius-match",
            "phase-match",
            "local-region-phase",
            "local-region-phase-ft",
            "cdi-planet-phase",
        }
    return features


def _build_sim_kwargs(args: argparse.Namespace) -> dict:
    sim_kwargs = default_sim_kwargs()
    sim_kwargs["secondary_diameter_ratio"] = float(args.secondary_ratio)
    if bool(getattr(args, "disable_spiders", False)):
        sim_kwargs["spider_width_pixels"] = 0.0
        sim_kwargs["spider_angles_deg"] = tuple()
    else:
        sim_kwargs["spider_width_pixels"] = float(args.spider_width)
        sim_kwargs["spider_angles_deg"] = tuple(float(a) for a in args.spider_angles)
    sim_kwargs["pupil_supersample"] = int(args.pupil_ss)
    sim_kwargs["phase_screen_path"] = resolve_phase_screen_path(args.phase_screen_jitter)
    sim_kwargs["phase_screen_index"] = 0
    sim_kwargs["coherent_ring_speckle_count"] = int(args.coherent_ring_speckle_count)
    sim_kwargs["coherent_ring_speckle_intensity"] = float(args.coherent_ring_speckle_intensity)
    sim_kwargs["lyot_reference_scale"] = float(args.lyot_reference_percent) / 100.0
    sim_kwargs["companion_flux_ratio"] = float(args.planet_flux_ratio)
    sim_kwargs["companion_offset_lamD"] = (float(args.planet_offset_x), float(args.planet_offset_y))
    sim_kwargs["include_ghost"] = not bool(args.disable_ghost)
    sim_kwargs["include_interference"] = (not bool(args.disable_interference)) and sim_kwargs["include_ghost"]
    sim_kwargs["include_star"] = not bool(args.disable_star)
    sim_kwargs["include_companion"] = not bool(args.disable_companion)
    sim_kwargs["include_companion_ghost"] = not bool(args.disable_companion_ghost)
    sim_kwargs["perfect_coronagraph"] = str(args.phase_mask_type).lower() in {
        "perfect_corongraph",
        "perfect_coronagraph",
    }
    sim_kwargs["phase_mask"] = build_phase_mask(args)
    return sim_kwargs


def _build_output_tags(args: argparse.Namespace, sim_kwargs: dict) -> tuple[str, str, str, str, str]:
    mask_suffix = mask_filename_suffix(sim_kwargs["phase_mask"])
    if bool(sim_kwargs.get("perfect_coronagraph", False)):
        mask_output_tag = "pcg"
    elif isinstance(sim_kwargs["phase_mask"], VortexPhaseMask):
        mask_output_tag = f"vx{int(sim_kwargs['phase_mask'].charge)}"
    else:
        mask_output_tag = f"rd{mask_suffix}"
    effective_cycles = float(args.phase_cycles)
    mode_name = str(args.phase_sweep_mode).strip().lower()
    if mode_name == "mask_rotation":
        phase_cycles_tag = f"_rs{int(args.phase_step)}"
    else:
        phase_cycles_tag = f"_cy{float_filename_token(effective_cycles, precision=3)}"
    phase_sweep_mode_tag = f"_m{_mode_tag(mode_name)}"
    region_shape = str(args.region_shape).strip().lower()
    if region_shape == "ring_of_circle":
        rotation_tag = float_filename_token(float(getattr(args, "ring_rotation_fraction", 0.0)), precision=3)
        single_region_tag = f"_{_shape_tag(region_shape)}rf{rotation_tag}"
    else:
        single_region_tag = f"_f{int(args.fov_count)}c{int(args.fov_centers_count)}{_shape_tag(region_shape)}"
    ghost_suffix = f"_g{1 if sim_kwargs['include_ghost'] else 0}"
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
    if args.cdi_phase_samples is not None:
        args.phase_step = int(args.cdi_phase_samples)
    if args.cdi_phase_cycles is not None:
        args.phase_cycles = float(args.cdi_phase_cycles)
    if args.cdi_planet_offset_x is not None:
        args.planet_offset_x_local = float(args.cdi_planet_offset_x)
    if args.cdi_planet_offset_y is not None:
        args.planet_offset_y_local = float(args.cdi_planet_offset_y)
    if args.cdi_secondary_ratio is not None:
        args.secondary_ratio_local = float(args.cdi_secondary_ratio)
    if args.cdi_planet_flux_ratio is not None:
        args.planet_flux_ratio_local = float(args.cdi_planet_flux_ratio)
    args.coc_fov_position_steps = int(getattr(args, "cdi_fov_position_steps", 0))
    args.coc_fov_circle_of_circles_trace = bool(getattr(args, "cdi_fov_orbit_trace", False))
    if bool(args.html_slm_gui):
        if __package__:
            from .html_slm_gui import launch_html_gui
        else:
            from coronagraph.html_slm_gui import launch_html_gui
        launch_html_gui()
        return
    if bool(args.interactive_slm_gui):
        if __package__:
            from .interactive_slm_gui import launch_gui as slm_gui_main
        else:
            from coronagraph.interactive_slm_gui import launch_gui as slm_gui_main
        slm_gui_main()
        return
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

    if "cdi-planet-phase" in features:
        run_cdi_planet_phase(
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
