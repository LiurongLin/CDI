from __future__ import annotations

from coronagraph import (
    CoronagraphSimulator,
    FQPMPhaseMask,
    FlatPhaseMask,
    NoPhaseMask,
    PhaseMask,
    RoddierPhaseMask,
    VortexPhaseMask,
    default_sim_kwargs,
    main,
    mask_filename_suffix,
    plot_phase_offset_combined_metrics,
    plot_phase_offset_metrics,
    plot_results,
    print_run_header,
    save_phase_mask_fits,
    sweep_roddier_phase_for_peak_match,
    sweep_roddier_radius_for_peak_match,
)
from coronagraph.cli import parse_args


def _default_sim_kwargs() -> dict:
    return default_sim_kwargs()


def _mask_filename_suffix(mask: PhaseMask) -> str:
    return mask_filename_suffix(mask)


def _print_run_header(result: dict) -> None:
    print_run_header(result)


def _parse_args():
    return parse_args()


if __name__ == "__main__":
    main()
