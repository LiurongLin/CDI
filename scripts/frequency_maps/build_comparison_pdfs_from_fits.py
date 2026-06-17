#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build high-resolution comparison PDFs directly from coherence and "
            "incoherence FITS maps, with overlays for the planet aperture, SNR "
            "annulus, and ring boundaries."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("CDI_data/12.06.26"),
        help="Base directory containing ring_planet_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("CDI_data/12.06.26/comparison_pdfs_from_fits"),
        help="Directory to write the PDF outputs.",
    )
    parser.add_argument(
        "--image-center-x",
        type=float,
        default=340.0,
        help="Image center x for ring and annulus overlays.",
    )
    parser.add_argument(
        "--image-center-y",
        type=float,
        default=263.0,
        help="Image center y for ring and annulus overlays.",
    )
    parser.add_argument(
        "--planet-radius",
        type=float,
        default=12.0,
        help="Planet aperture radius in pixels.",
    )
    parser.add_argument(
        "--annulus-width",
        type=float,
        default=24.0,
        help="SNR annulus width in pixels.",
    )
    parser.add_argument(
        "--incoh-percentile",
        type=float,
        default=98.0,
        help="Upper percentile for incoherence display scaling.",
    )
    parser.add_argument(
        "--coh-percentile",
        type=float,
        default=99.0,
        help="Upper percentile for coherence display scaling.",
    )
    parser.add_argument(
        "--px-per-lambda-d",
        type=float,
        default=12.295081967213115,
        help="Camera sampling in pixels per lambda/D for ring-width labeling.",
    )
    return parser.parse_args()


def read_fits_image(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D FITS image in {path}, got shape {image.shape}")
    return image


def display_limits(image: np.ndarray, percentile: float) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanpercentile(finite, percentile))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def ring_width_px_from_name(name: str) -> float:
    match = re.search(r"ring_(\d+(?:\.\d+)?)id", name)
    if match is None:
        raise ValueError(f"Could not extract ring width from {name}")
    return float(match.group(1)) * 12.0


def base_key_from_name(name: str) -> str:
    key = name
    key = key.replace("_coherence_ratio_", "::")
    key = key.replace("_incoherence_ratio_", "::")
    return key.split("::")[0]


def ring_order(name: str) -> float:
    match = re.search(r"ring_(\d+(?:\.\d+)?)id", name)
    return float(match.group(1)) if match else 1e9


def load_snr_map(summary_csv: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with summary_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            values[Path(row["file"]).stem] = float(row["snr"])
    return values


def draw_overlays(
    ax: plt.Axes,
    stem: str,
    planet_x: float,
    planet_y: float,
    planet_radius: float,
    image_center_x: float,
    image_center_y: float,
    annulus_width: float,
) -> None:
    ring_width_px = ring_width_px_from_name(stem)
    ring_mid_radius = float(np.hypot(planet_x - image_center_x, planet_y - image_center_y))
    ring_inner_radius = ring_mid_radius - ring_width_px / 2.0
    ring_outer_radius = ring_mid_radius + ring_width_px / 2.0
    annulus_inner_radius = ring_mid_radius - annulus_width / 2.0
    annulus_outer_radius = ring_mid_radius + annulus_width / 2.0

    for radius in (ring_inner_radius, ring_outer_radius):
        ax.add_patch(
            Circle(
                (image_center_x, image_center_y),
                radius,
                edgecolor="lime",
                facecolor="none",
                linewidth=0.8,
                linestyle="--",
            )
        )
    for radius in (annulus_inner_radius, annulus_outer_radius):
        ax.add_patch(
            Circle(
                (image_center_x, image_center_y),
                radius,
                edgecolor="white",
                facecolor="none",
                linewidth=0.8,
            )
        )
    ax.add_patch(
        Circle(
            (planet_x, planet_y),
            planet_radius,
            edgecolor="cyan",
            facecolor="none",
            linewidth=0.9,
        )
    )


def matches_group(group: str, name: str) -> bool:
    if group == "location_397_312_no_phase":
        return "withphasescreen" not in name and "_combined_" in name
    if group == "location_397_312_with_phase":
        return "withphasescreen" in name
    if group == "location_417_345_no_phase":
        return "phasescreen" not in name and "_combined_" in name
    if group == "location_417_345_with_phase":
        return "phasescreen" in name
    return False


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = {
        "location_397_312_no_phase": {
            "folder": args.base_dir / "ring_planet_7id/second_peak_coherence_maps",
            "planet": (397.0, 312.0),
            "snr_csv": args.base_dir / "ring_planet_7id/second_peak_coherence_maps/incoherence_snr_summary.csv",
        },
        "location_397_312_with_phase": {
            "folder": args.base_dir / "ring_planet_7id/second_peak_coherence_maps",
            "planet": (397.0, 312.0),
            "snr_csv": args.base_dir / "ring_planet_7id/second_peak_coherence_maps/incoherence_snr_summary.csv",
        },
        "location_417_345_no_phase": {
            "folder": args.base_dir / "ring_planet_9.9id/second_peak_coherence_maps",
            "planet": (417.0, 345.0),
            "snr_csv": args.base_dir / "ring_planet_9.9id/second_peak_coherence_maps/incoherence_snr_summary.csv",
        },
        "location_417_345_with_phase": {
            "folder": args.base_dir / "ring_planet_9.9id/second_peak_coherence_maps",
            "planet": (417.0, 345.0),
            "snr_csv": args.base_dir / "ring_planet_9.9id/second_peak_coherence_maps/incoherence_snr_summary.csv",
        },
    }

    for group, info in groups.items():
        folder = info["folder"]
        planet_x, planet_y = info["planet"]
        snr_values = load_snr_map(info["snr_csv"])

        fits_paths = sorted(
            [p for p in folder.glob("*.fits") if matches_group(group, p.name)],
            key=lambda p: (ring_order(p.name), p.name),
        )
        by_stem: dict[str, dict[str, Path]] = {}
        for path in fits_paths:
            key = base_key_from_name(path.name)
            by_stem.setdefault(key, {})
            if "_coherence_ratio_" in path.name:
                by_stem[key]["coh"] = path
            elif "_incoherence_ratio_" in path.name:
                by_stem[key]["incoh"] = path

        pdf_path = args.output_dir / f"{group}_comparison_from_fits.pdf"
        with PdfPages(pdf_path) as pdf:
            for stem in sorted(by_stem, key=ring_order):
                coh_path = by_stem[stem].get("coh")
                incoh_path = by_stem[stem].get("incoh")
                if coh_path is None or incoh_path is None:
                    continue

                coh = read_fits_image(coh_path)
                incoh = read_fits_image(incoh_path)
                ring_width_px = ring_width_px_from_name(stem)
                ring_width_lambda_d = ring_width_px / args.px_per_lambda_d
                coh_vmin, coh_vmax = display_limits(coh, args.coh_percentile)
                incoh_vmin, incoh_vmax = display_limits(incoh, args.incoh_percentile)
                snr = snr_values.get(stem, float("nan"))

                fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), dpi=200)
                fig.suptitle(
                    f"{stem} | ring width = {ring_width_lambda_d:.2f} lambda/D",
                    fontsize=12,
                )

                panels = [
                    (axes[0], coh, "Coherence", coh_vmin, coh_vmax, "coherence"),
                    (axes[1], incoh, "Incoherence", incoh_vmin, incoh_vmax, "incoherence"),
                ]
                for ax, image, title, vmin, vmax, cbar_label in panels:
                    im = ax.imshow(image, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
                    draw_overlays(
                        ax,
                        stem=stem,
                        planet_x=planet_x,
                        planet_y=planet_y,
                        planet_radius=args.planet_radius,
                        image_center_x=args.image_center_x,
                        image_center_y=args.image_center_y,
                        annulus_width=args.annulus_width,
                    )
                    ax.set_title(title)
                    ax.set_xlabel("x [px]")
                    ax.set_ylabel("y [px]")
                    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label(cbar_label)
                    if title == "Incoherence" and np.isfinite(snr):
                        ax.text(
                            0.02,
                            0.98,
                            f"SNR={snr:.3f}",
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                            color="white",
                            fontsize=10,
                            bbox={"facecolor": "black", "alpha": 0.5, "pad": 4, "edgecolor": "none"},
                        )

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        print(pdf_path)


if __name__ == "__main__":
    main()
