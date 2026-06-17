#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 2D FFT-ratio map from two FFT frequency windows in a FITS cube."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input FITS cube paths.")
    parser.add_argument("--dt", type=float, default=0.1, help="Sampling interval.")
    parser.add_argument("--num-fmin", type=float, default=0.7, help="Numerator band min Hz.")
    parser.add_argument("--num-fmax", type=float, default=1.3, help="Numerator band max Hz.")
    parser.add_argument("--den-fmin", type=float, default=0.0, help="Denominator band min Hz.")
    parser.add_argument("--den-fmax", type=float, default=0.5, help="Denominator band max Hz.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path.cwd(),
        help="Output directory.",
    )
    return parser.parse_args()


def read_cube(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube in {path}, got shape {cube.shape}")
    return cube, header


def write_fits(path: Path, data: np.ndarray, header: fits.Header, bunit: str, comment: str) -> None:
    out_header = header.copy()
    for key in ("NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "BITPIX"):
        if key in out_header:
            del out_header[key]
    out_header["BUNIT"] = bunit
    out_header["COMMENT"] = comment
    fits.writeto(path, np.asarray(data, dtype=np.float32), out_header, overwrite=True)


def fft_ratio_map(
    cube: np.ndarray,
    dt: float,
    num_fmin: float,
    num_fmax: float,
    den_fmin: float,
    den_fmax: float,
    eps: float = 1e-20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    freqs = np.fft.fftfreq(cube.shape[0], d=dt)
    fft_cube = np.fft.fft(cube, axis=0)
    abs_freq = np.abs(freqs)

    num_mask = (abs_freq >= num_fmin) & (abs_freq <= num_fmax)
    den_mask = (abs_freq >= den_fmin) & (abs_freq <= den_fmax)
    if not np.any(num_mask):
        raise RuntimeError(f"No FFT bins found in numerator band [{num_fmin}, {num_fmax}] Hz.")
    if not np.any(den_mask):
        raise RuntimeError(f"No FFT bins found in denominator band [{den_fmin}, {den_fmax}] Hz.")

    num_ref = np.sum(np.abs(fft_cube[num_mask]), axis=0) / float(np.count_nonzero(num_mask))
    den_ref = np.sum(np.abs(fft_cube[den_mask]), axis=0) / float(np.count_nonzero(den_mask))
    den_ref = np.maximum(den_ref, eps)
    ratio_map = num_ref / den_ref

    return freqs[num_mask], freqs[den_mask], ratio_map


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        cube, header = read_cube(path)
        num_freqs, den_freqs, ratio_map = fft_ratio_map(
            cube,
            dt=args.dt,
            num_fmin=args.num_fmin,
            num_fmax=args.num_fmax,
            den_fmin=args.den_fmin,
            den_fmax=args.den_fmax,
        )

        prefix = f"{path.parent.name}_{path.stem}"
        band_tag = (
            f"ratio_{args.num_fmin:.3f}_{args.num_fmax:.3f}Hz_"
            f"over_{args.den_fmin:.3f}_{args.den_fmax:.3f}Hz"
        ).replace(".", "p")

        map_path = args.outdir / f"{prefix}_{band_tag}_reconstruction_map.fits"
        write_fits(
            map_path,
            ratio_map,
            header,
            bunit="fft_ratio",
            comment="Per-pixel mean numerator FFT magnitude divided by mean denominator FFT magnitude.",
        )

        print(f"[file] {path}")
        print(f"  numerator_freqs={num_freqs}")
        print(f"  denominator_freqs={den_freqs}")
        print(f"  wrote {map_path}")


if __name__ == "__main__":
    main()
