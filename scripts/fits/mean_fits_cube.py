#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Take the mean over the frame axis of 3D FITS cubes."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input 3D FITS cube paths.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to each input cube directory.",
    )
    return parser.parse_args()


def read_cube(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D FITS cube in {path}, got shape {cube.shape}")
    return cube, header


def write_mean_image(path: Path, image: np.ndarray, header: fits.Header) -> None:
    out_header = header.copy()
    for key in ("NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "BITPIX"):
        if key in out_header:
            del out_header[key]
    out_header["BUNIT"] = "mean_signal"
    out_header["COMMENT"] = "Mean image over frame axis from FITS cube."
    fits.writeto(path, np.asarray(image, dtype=np.float32), out_header, overwrite=True)


def main() -> None:
    args = parse_args()

    for input_path in args.inputs:
        cube, header = read_cube(input_path)
        mean_image = np.mean(cube, axis=0)
        target_dir = input_path.parent if args.outdir is None else args.outdir
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{input_path.stem}_mean.fits"
        write_mean_image(output_path, mean_image, header)
        print(f"[file] {input_path}")
        print(f"  wrote {output_path}")


if __name__ == "__main__":
    main()
