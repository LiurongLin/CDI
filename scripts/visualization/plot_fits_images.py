#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render 2D FITS images to PNG for quick inspection."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input 2D FITS images.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Directory to save PNGs. Defaults to each FITS file's directory.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap name.",
    )
    parser.add_argument(
        "--vmin-percentile",
        type=float,
        default=None,
        help="Optional lower percentile for display scaling.",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=None,
        help="Optional upper percentile for display scaling.",
    )
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D FITS image in {path}, got shape {image.shape}")
    return image


def render_png(
    path: Path,
    image: np.ndarray,
    outdir: Path | None,
    cmap: str,
    vmin_percentile: float | None,
    vmax_percentile: float | None,
) -> Path:
    target_dir = path.parent if outdir is None else outdir
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{path.stem}.png"

    vmin = None if vmin_percentile is None else float(np.percentile(image, vmin_percentile))
    vmax = None if vmax_percentile is None else float(np.percentile(image, vmax_percentile))

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(image, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(path.stem)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()

    for path in args.inputs:
        image = read_image(path)
        output_path = render_png(
            path,
            image,
            args.outdir,
            args.cmap,
            args.vmin_percentile,
            args.vmax_percentile,
        )
        print(f"[ok] wrote {output_path}")


if __name__ == "__main__":
    main()
