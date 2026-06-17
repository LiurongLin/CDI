#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class Region:
    label: str
    center_x: float
    center_y: float
    radius: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build region-sum time series and FFT plots for a planet aperture and "
            "additional apertures at the same angular separation."
        )
    )
    parser.add_argument("cube_path", type=Path, help="Combined FITS cube path.")
    parser.add_argument("--planet-x", type=float, default=285.0, help="Planet aperture x center.")
    parser.add_argument("--planet-y", type=float, default=229.0, help="Planet aperture y center.")
    parser.add_argument(
        "--center-x",
        type=float,
        default=None,
        help="Annulus center x. Defaults to the image midpoint.",
    )
    parser.add_argument(
        "--center-y",
        type=float,
        default=None,
        help="Annulus center y. Defaults to the image midpoint.",
    )
    parser.add_argument("--radius", type=float, default=14.0, help="Aperture radius in pixels.")
    parser.add_argument(
        "--n-regions",
        type=int,
        default=6,
        help="Total number of equally spaced apertures on the annulus, including the planet.",
    )
    parser.add_argument("--dt", type=float, default=0.1, help="Sampling interval.")
    parser.add_argument(
        "--label-first-peak",
        action="store_true",
        help="Annotate the DC/first peak within 0 to 1 Hz for each region.",
    )
    parser.add_argument(
        "--label-second-peak",
        action="store_true",
        help="Annotate the strongest non-DC peak within 0 to 1 Hz for each region.",
    )
    parser.add_argument(
        "--subtract-mean",
        action="store_true",
        help="Subtract the mean from each region time series before FFT.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the cube folder.",
    )
    return parser.parse_args()


def read_cube(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube in {path}, got shape {cube.shape}")
    return cube


def build_regions(
    cube: np.ndarray,
    planet_x: float,
    planet_y: float,
    center_x: float | None,
    center_y: float | None,
    radius: float,
    n_regions: int,
) -> list[Region]:
    ny, nx = cube.shape[1:]
    cx = (nx - 1.0) / 2.0 if center_x is None else center_x
    cy = (ny - 1.0) / 2.0 if center_y is None else center_y
    dx = planet_x - cx
    dy = planet_y - cy
    rho = float(np.hypot(dx, dy))
    theta0 = float(np.arctan2(dy, dx))

    regions = [Region(label="planet", center_x=planet_x, center_y=planet_y, radius=radius)]
    for idx in range(1, n_regions):
        theta = theta0 + idx * (2.0 * np.pi / n_regions)
        x = cx + rho * np.cos(theta)
        y = cy + rho * np.sin(theta)
        regions.append(Region(label=f"annulus_{idx}", center_x=x, center_y=y, radius=radius))
    return regions


def circular_mask(shape: tuple[int, int], region: Region) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - region.center_x) ** 2 + (yy - region.center_y) ** 2 <= region.radius ** 2


def region_sums(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return cube[:, mask].sum(axis=1)


def fft_magnitude(values: np.ndarray, dt: float, subtract_mean: bool) -> tuple[np.ndarray, np.ndarray]:
    signal = values - np.mean(values) if subtract_mean else values
    freqs = np.fft.rfftfreq(signal.size, d=dt)
    magnitude = np.abs(np.fft.rfft(signal))
    return freqs, magnitude


def second_peak_index(freqs: np.ndarray, magnitude: np.ndarray) -> int | None:
    band = (freqs > 0.0) & (freqs <= 1.0)
    if not np.any(band):
        return None
    band_indices = np.flatnonzero(band)
    return int(band_indices[np.argmax(magnitude[band])])


def first_peak_index(freqs: np.ndarray, magnitude: np.ndarray) -> int | None:
    band = (freqs >= 0.0) & (freqs <= 1.0)
    if not np.any(band):
        return None
    band_indices = np.flatnonzero(band)
    return int(band_indices[np.argmax(magnitude[band])])


def main() -> None:
    args = parse_args()
    cube = read_cube(args.cube_path)
    outdir = args.cube_path.parent if args.outdir is None else args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    regions = build_regions(
        cube,
        planet_x=args.planet_x,
        planet_y=args.planet_y,
        center_x=args.center_x,
        center_y=args.center_y,
        radius=args.radius,
        n_regions=args.n_regions,
    )

    masks = {region.label: circular_mask(cube.shape[1:], region) for region in regions}
    sums = {region.label: region_sums(cube, masks[region.label]) for region in regions}

    csv_path = outdir / f"{args.cube_path.parent.name}_{args.cube_path.stem}_annulus_region_sums.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["label", "center_x", "center_y", "frame_index", "sum"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for region in regions:
            for frame_index, value in enumerate(sums[region.label]):
                writer.writerow(
                    {
                        "label": region.label,
                        "center_x": region.center_x,
                        "center_y": region.center_y,
                        "frame_index": frame_index,
                        "sum": float(value),
                    }
                )

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for region in regions:
        freqs, mag = fft_magnitude(sums[region.label], args.dt, args.subtract_mean)
        label = f"{region.label} ({region.center_x:.1f}, {region.center_y:.1f})"
        (line,) = ax.plot(freqs, mag, linewidth=1.6, label=label)
        if args.label_first_peak:
            peak_idx = first_peak_index(freqs, mag)
            if peak_idx is not None:
                peak_freq = freqs[peak_idx]
                peak_mag = mag[peak_idx]
                ax.scatter([peak_freq], [peak_mag], color=line.get_color(), s=18, zorder=4)
                ax.text(
                    peak_freq,
                    peak_mag,
                    f" {peak_freq:.3f} Hz",
                    color=line.get_color(),
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )
        if args.label_second_peak:
            peak_idx = second_peak_index(freqs, mag)
            if peak_idx is not None:
                peak_freq = freqs[peak_idx]
                peak_mag = mag[peak_idx]
                ax.scatter([peak_freq], [peak_mag], color=line.get_color(), s=18, zorder=4)
                ax.text(
                    peak_freq,
                    peak_mag,
                    f" {peak_freq:.3f} Hz",
                    color=line.get_color(),
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )
    title_suffix = " (mean-subtracted)" if args.subtract_mean else " (raw)"
    ax.set_title(f"FFT Annulus Apertures: {args.cube_path.name}{title_suffix}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("FFT magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    suffix = "mean_subtracted" if args.subtract_mean else "raw"
    plot_path = outdir / f"{args.cube_path.parent.name}_{args.cube_path.stem}_annulus_region_sums_fft_{suffix}.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"[ok] wrote {csv_path}")
    print(f"[ok] wrote {plot_path}")
    for region in regions:
        print(
            f"  {region.label}: center=({region.center_x:.3f}, {region.center_y:.3f}) "
            f"radius={region.radius:.1f}"
        )


if __name__ == "__main__":
    main()
