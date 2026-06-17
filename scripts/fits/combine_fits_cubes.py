#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate multiple 3D FITS cubes along the frame axis."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Input FITS cube paths in the order they should be concatenated.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output FITS cube path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def read_cube(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        cube = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube in {path}, got shape {cube.shape}")
    return cube, header


def main() -> None:
    args = parse_args()

    cubes: list[np.ndarray] = []
    first_header: fits.Header | None = None
    expected_frame_shape: tuple[int, int] | None = None

    for path in args.inputs:
        cube, header = read_cube(path)
        if first_header is None:
            first_header = header
            expected_frame_shape = cube.shape[1:]
        elif cube.shape[1:] != expected_frame_shape:
            raise ValueError(
                f"Spatial shape mismatch for {path}: expected {expected_frame_shape}, "
                f"got {cube.shape[1:]}"
            )
        cubes.append(cube)
        print(f"[file] {path} frames={cube.shape[0]}")

    combined_cube = np.concatenate(cubes, axis=0)
    output_header = first_header.copy()
    output_header["HISTORY"] = "Combined with combine_fits_cubes.py"
    output_header["NSOURCE"] = (len(args.inputs), "Number of cubes concatenated")
    fits.writeto(args.output, combined_cube, header=output_header, overwrite=args.overwrite)

    print(f"[ok] wrote {args.output}")
    print(f"  combined_frames={combined_cube.shape[0]} shape={combined_cube.shape}")


if __name__ == "__main__":
    main()
