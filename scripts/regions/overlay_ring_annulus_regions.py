#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.patches import Circle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay annulus aperture regions on a mean FITS image."
    )
    parser.add_argument("image_path", type=Path, help="Mean FITS image path.")
    parser.add_argument(
        "--planet-x",
        type=float,
        default=285.0,
        help="Planet aperture x center.",
    )
    parser.add_argument(
        "--planet-y",
        type=float,
        default=229.0,
        help="Planet aperture y center.",
    )
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
    parser.add_argument(
        "--radius",
        type=float,
        default=14.0,
        help="Aperture radius in pixels.",
    )
    parser.add_argument(
        "--n-regions",
        type=int,
        default=6,
        help="Total number of equally spaced apertures on the annulus, including the planet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D FITS image in {path}, got shape {image.shape}")
    return image


def build_regions(
    shape: tuple[int, int],
    planet_x: float,
    planet_y: float,
    center_x: float | None,
    center_y: float | None,
    radius: float,
    n_regions: int,
) -> list[tuple[str, float, float]]:
    ny, nx = shape
    cx = (nx - 1.0) / 2.0 if center_x is None else center_x
    cy = (ny - 1.0) / 2.0 if center_y is None else center_y
    dx = planet_x - cx
    dy = planet_y - cy
    rho = float(np.hypot(dx, dy))
    theta0 = float(np.arctan2(dy, dx))

    regions = [("planet", planet_x, planet_y)]
    for idx in range(1, n_regions):
        theta = theta0 + idx * (2.0 * np.pi / n_regions)
        x = cx + rho * np.cos(theta)
        y = cy + rho * np.sin(theta)
        regions.append((f"annulus_{idx}", x, y))
    return regions


def main() -> None:
    args = parse_args()
    image = read_image(args.image_path)
    regions = build_regions(
        image.shape,
        planet_x=args.planet_x,
        planet_y=args.planet_y,
        center_x=args.center_x,
        center_y=args.center_y,
        radius=args.radius,
        n_regions=args.n_regions,
    )

    vmin = float(np.percentile(image, 1))
    vmax = float(np.percentile(image, 99))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(image, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    for label, x, y in regions:
        color = "cyan" if label == "planet" else "white"
        ax.add_patch(Circle((x, y), args.radius, fill=False, edgecolor=color, linewidth=1.2))
        ax.text(
            x + args.radius + 2,
            y,
            label,
            color=color,
            fontsize=8,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.45, edgecolor="none"),
        )

    ax.set_title(args.image_path.stem)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
