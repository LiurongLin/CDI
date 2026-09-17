#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.patches import Circle


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render an incoherence map using the ROI/planet-location sweep SNR, "
            "optionally side by side with one raw PSF frame from the source cube."
        )
    )
    parser.add_argument("--input-fits", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--sweep-csv", type=Path, required=True)
    parser.add_argument("--planet-x", type=float, required=True)
    parser.add_argument("--planet-y", type=float, required=True)
    parser.add_argument("--image-center-x", type=float, required=True)
    parser.add_argument("--image-center-y", type=float, required=True)
    parser.add_argument("--planet-separation-lamd", type=float, default=None)
    parser.add_argument("--px-per-lamd", type=float, default=None)
    parser.add_argument("--percentile", type=float, default=98.0)
    parser.add_argument("--ring-width-scale-px", type=float, default=12.0)
    parser.add_argument("--raw-psf-cube", type=Path, default=None)
    parser.add_argument("--raw-frame-index", type=int, default=0)
    parser.add_argument("--raw-percentile", type=float, default=99.9)
    parser.add_argument("--crop-size-px", type=int, default=400)
    parser.add_argument("--show-raw-psf", action="store_true")
    return parser.parse_args()


def ring_width_px_from_name(name: str, scale_px: float) -> float | None:
    match = re.search(r"ring_(\d+(?:\.\d+)?)id", name)
    if match is None:
        return None
    return float(match.group(1)) * float(scale_px)


def load_fits_image(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D FITS image in {path}, got shape {image.shape}")
    return image


def load_raw_psf_frame(path: Path, frame_index: int) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D FITS cube in {path}, got shape {cube.shape}")
    resolved_index = int(np.clip(frame_index, 0, cube.shape[0] - 1))
    return np.asarray(cube[resolved_index], dtype=np.float64)


def log10_image(image: np.ndarray, floor: float = 1.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    return np.log10(np.maximum(arr, 0.0) + float(floor))


def infer_raw_psf_cube_path(input_fits: Path) -> Path | None:
    stem = input_fits.stem
    stem = re.sub(r"_incoherence_ratio_.*$", "", stem)
    candidate = input_fits.parent.parent / f"{stem}.fit"
    return candidate if candidate.exists() else None


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


def centered_crop_bounds(image_shape: tuple[int, int], crop_size_px: int) -> tuple[int, int, int, int]:
    ny, nx = image_shape
    crop = int(max(1, crop_size_px))
    crop_x = min(crop, nx)
    crop_y = min(crop, ny)
    x0 = max(0, (nx - crop_x) // 2)
    y0 = max(0, (ny - crop_y) // 2)
    x1 = x0 + crop_x
    y1 = y0 + crop_y
    return x0, x1, y0, y1


def wrap_angle_deg(theta_deg: float) -> float:
    return float((theta_deg + 360.0) % 360.0)


def select_best_sweep_row(
    sweep_csv: Path,
    target_radius_lamd: float,
    target_theta_deg: float,
) -> dict[str, float]:
    groups: dict[tuple[float, float], list[dict[str, float]]] = {}
    with sweep_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = {key: float(value) for key, value in raw_row.items()}
            theta_deg = wrap_angle_deg(math.degrees(row["planet_theta_rad"]))
            key = (row["orbit_radius_lamD"], theta_deg)
            groups.setdefault(key, []).append(row)

    if not groups:
        raise ValueError(f"No sweep rows found in {sweep_csv}")

    target_x = float(target_radius_lamd) * math.cos(math.radians(target_theta_deg))
    target_y = float(target_radius_lamd) * math.sin(math.radians(target_theta_deg))

    best_group_key: tuple[float, float] | None = None
    best_group_distance = float("inf")
    for radius_lamd, theta_deg in groups:
        sample_x = float(radius_lamd) * math.cos(math.radians(theta_deg))
        sample_y = float(radius_lamd) * math.sin(math.radians(theta_deg))
        distance = float(math.hypot(sample_x - target_x, sample_y - target_y))
        if distance < best_group_distance:
            best_group_distance = distance
            best_group_key = (radius_lamd, theta_deg)

    assert best_group_key is not None
    return max(groups[best_group_key], key=lambda row: row["snr"])


def add_planet_annotations(
    ax: plt.Axes,
    image_shape: tuple[int, int],
    planet_x: float,
    planet_y: float,
    center_x: float,
    center_y: float,
    ring_inner_radius: float | None,
    ring_outer_radius: float | None,
    snr_value: float | None,
) -> None:
    if ring_inner_radius is not None:
        ax.add_patch(
            Circle(
                (center_x, center_y),
                ring_inner_radius,
                edgecolor="lime",
                facecolor="none",
                linewidth=0.8,
                linestyle="--",
            )
        )
    if ring_outer_radius is not None:
        ax.add_patch(
            Circle(
                (center_x, center_y),
                ring_outer_radius,
                edgecolor="lime",
                facecolor="none",
                linewidth=0.8,
                linestyle="--",
            )
        )

    arrow_tail_x = min(float(image_shape[1]) - 55.0, float(planet_x) + 42.0)
    arrow_tail_y = min(float(image_shape[0]) - 40.0, float(planet_y) + 38.0)
    dx = arrow_tail_x - float(planet_x)
    dy = arrow_tail_y - float(planet_y)
    norm = math.hypot(dx, dy)
    head_offset_px = 11.0
    if norm > 0.0:
        arrow_head_x = float(planet_x) + head_offset_px * dx / norm
        arrow_head_y = float(planet_y) + head_offset_px * dy / norm
    else:
        arrow_head_x = float(planet_x)
        arrow_head_y = float(planet_y)
    ax.annotate(
        "planet",
        xy=(arrow_head_x, arrow_head_y),
        xytext=(arrow_tail_x, arrow_tail_y),
        color="cyan",
        fontsize=10,
        ha="left",
        va="bottom",
        arrowprops={
            "arrowstyle": "->",
            "lw": 1.4,
            "color": "cyan",
            "shrinkA": 0.0,
            "shrinkB": 2.0,
        },
    )
    if snr_value is not None:
        ax.text(
            0.02,
            0.98,
            f"SNR={snr_value:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=9,
            bbox={"facecolor": "black", "alpha": 0.58, "pad": 4, "edgecolor": "none"},
        )


def main() -> None:
    args = parse_args()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)

    incoh = load_fits_image(args.input_fits)
    x0, x1, y0, y1 = centered_crop_bounds(incoh.shape, args.crop_size_px)
    incoh_cropped = np.asarray(incoh[y0:y1, x0:x1], dtype=np.float64)
    incoh_vmin, incoh_vmax = display_limits(incoh_cropped, args.percentile)

    dx_px = float(args.planet_x - args.image_center_x)
    dy_px = float(args.planet_y - args.image_center_y)
    orbit_radius_px = float(math.hypot(dx_px, dy_px))
    theta_deg = wrap_angle_deg(math.degrees(math.atan2(dy_px, dx_px)))

    if args.px_per_lamd is not None:
        px_per_lamd = float(args.px_per_lamd)
        target_radius_lamd = orbit_radius_px / px_per_lamd
    elif args.planet_separation_lamd is not None:
        target_radius_lamd = float(args.planet_separation_lamd)
        px_per_lamd = orbit_radius_px / target_radius_lamd
    else:
        raise ValueError("Provide either --px-per-lamd or --planet-separation-lamd")

    sweep_row = select_best_sweep_row(
        sweep_csv=args.sweep_csv,
        target_radius_lamd=target_radius_lamd,
        target_theta_deg=theta_deg,
    )
    sweep_snr = float(sweep_row["snr"])

    ring_width_px = ring_width_px_from_name(args.input_fits.stem, args.ring_width_scale_px)
    ring_inner_radius = None
    ring_outer_radius = None
    if ring_width_px is not None:
        ring_inner_radius = orbit_radius_px - 0.5 * ring_width_px
        ring_outer_radius = orbit_radius_px + 0.5 * ring_width_px

    raw_psf = None
    resolved_raw_frame_index = None
    raw_psf_path = args.raw_psf_cube or infer_raw_psf_cube_path(args.input_fits)
    if args.show_raw_psf:
        if raw_psf_path is None:
            raise ValueError("Could not infer the raw PSF cube path; pass --raw-psf-cube explicitly.")
        with fits.open(raw_psf_path, memmap=True) as hdul:
            cube = np.asarray(hdul[0].data, dtype=np.float64)
        if cube.ndim != 3:
            raise ValueError(f"Expected a 3D FITS cube in {raw_psf_path}, got shape {cube.shape}")
        resolved_raw_frame_index = int(np.clip(args.raw_frame_index, 0, cube.shape[0] - 1))
        raw_psf = np.asarray(cube[resolved_raw_frame_index], dtype=np.float64)
        raw_psf_log10 = log10_image(raw_psf[y0:y1, x0:x1])
        raw_vmin, raw_vmax = display_limits(raw_psf_log10, args.raw_percentile)
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), dpi=180, sharey=True)
        raw_ax, incoh_ax = axes
        raw_im = raw_ax.imshow(
            raw_psf_log10,
            origin="lower",
            cmap="magma",
            vmin=raw_vmin,
            vmax=raw_vmax,
            extent=[x0, x1, y0, y1],
        )
        add_planet_annotations(
            ax=raw_ax,
            image_shape=raw_psf.shape,
            planet_x=float(args.planet_x),
            planet_y=float(args.planet_y),
            center_x=float(args.image_center_x),
            center_y=float(args.image_center_y),
            ring_inner_radius=None,
            ring_outer_radius=None,
            snr_value=None,
        )
        raw_ax.set_xlabel("x [px]")
        raw_ax.set_ylabel("y [px]")
        raw_ax.set_aspect("equal")
        raw_ax.set_xlim(x0, x1)
        raw_ax.set_ylim(y0, y1)
        raw_ax.tick_params(direction="out", length=3.5, width=0.8)
        fig.colorbar(
            raw_im,
            ax=raw_ax,
            orientation="horizontal",
            fraction=0.035,
            pad=0.10,
            label=r"$\log_{10}(\mathrm{intensity})$",
        )
        fig.subplots_adjust(left=0.055, right=0.992, top=0.992, bottom=0.12, wspace=0.02)
    else:
        fig, incoh_ax = plt.subplots(1, 1, figsize=(7.2, 7.2), dpi=180, constrained_layout=True)

    incoh_im = incoh_ax.imshow(
        incoh_cropped,
        origin="lower",
        cmap="magma",
        vmin=incoh_vmin,
        vmax=incoh_vmax,
        extent=[x0, x1, y0, y1],
    )
    add_planet_annotations(
        ax=incoh_ax,
        image_shape=incoh.shape,
        planet_x=float(args.planet_x),
        planet_y=float(args.planet_y),
        center_x=float(args.image_center_x),
        center_y=float(args.image_center_y),
        ring_inner_radius=ring_inner_radius,
        ring_outer_radius=ring_outer_radius,
        snr_value=sweep_snr,
    )
    incoh_ax.set_xlabel("x [px]")
    incoh_ax.set_aspect("equal")
    incoh_ax.set_xlim(x0, x1)
    incoh_ax.set_ylim(y0, y1)
    incoh_ax.tick_params(direction="out", length=3.5, width=0.8)
    fig.colorbar(
        incoh_im,
        ax=incoh_ax,
        orientation="horizontal",
        fraction=0.035,
        pad=0.10,
        label=f"Incoherence Value (max={args.percentile:g}th pct)",
    )

    fig.savefig(args.output_png, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] wrote {args.output_png}")
    if raw_psf_path is not None and args.show_raw_psf:
        print(f"[info] raw psf cube={raw_psf_path}")
        print(f"[info] raw psf frame index={resolved_raw_frame_index}")


if __name__ == "__main__":
    main()
