#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine coherence or incoherence FITS maps by dataset group "
            "(planet location and phase-mask state)."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input FITS maps to combine.")
    parser.add_argument("--output", required=True, type=Path, help="Output FITS path.")
    parser.add_argument(
        "--combine",
        choices=("mean", "median"),
        default="mean",
        help="Pixelwise combine function.",
    )
    parser.add_argument(
        "--png-percentile",
        type=float,
        default=None,
        help="Optional upper percentile cap for PNG visualization.",
    )
    parser.add_argument(
        "--png-label",
        default="combined map",
        help="Colorbar label for PNG visualization.",
    )
    return parser.parse_args()


def read_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        image = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D FITS image in {path}, got shape {image.shape}")
    return image, header


def write_fits(path: Path, data: np.ndarray, header: fits.Header, comment: str) -> None:
    out_header = header.copy()
    for key in ("NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "BITPIX"):
        if key in out_header:
            del out_header[key]
    out_header["COMMENT"] = comment
    fits.writeto(path, np.asarray(data, dtype=np.float32), out_header, overwrite=True)


def save_png(path: Path, image: np.ndarray, title: str, colorbar_label: str, percentile: float | None) -> None:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        vmin = 0.0
        vmax = 1.0
    else:
        vmin = float(np.nanmin(finite))
        if percentile is None:
            vmax = float(np.nanmax(finite))
        else:
            vmax = float(np.nanpercentile(finite, percentile))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = float(np.nanmax(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(image, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    images: list[np.ndarray] = []
    first_header: fits.Header | None = None
    expected_shape: tuple[int, int] | None = None

    for path in args.inputs:
        image, header = read_image(path)
        if first_header is None:
            first_header = header
            expected_shape = image.shape
        elif image.shape != expected_shape:
            raise ValueError(f"Shape mismatch for {path}: expected {expected_shape}, got {image.shape}")
        images.append(image)
        print(f"[input] {path}")

    stack = np.stack(images, axis=0)
    combined = np.nanmean(stack, axis=0) if args.combine == "mean" else np.nanmedian(stack, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_fits(
        args.output,
        combined,
        first_header,
        comment=f"Pixelwise {args.combine} of {len(images)} input maps.",
    )
    print(f"[ok] wrote {args.output}")

    png_path = args.output.with_suffix(".png")
    save_png(
        png_path,
        combined,
        args.output.stem,
        args.png_label,
        args.png_percentile,
    )
    print(f"[ok] wrote {png_path}")


if __name__ == "__main__":
    main()
