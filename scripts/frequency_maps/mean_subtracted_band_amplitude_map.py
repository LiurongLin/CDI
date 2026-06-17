#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subtract the per-pixel mean before FFT and build an amplitude map from "
            "the strongest FFT component inside a frequency band."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input FITS cube paths.")
    parser.add_argument("--dt", type=float, default=0.1, help="Sampling interval.")
    parser.add_argument("--fmin", type=float, default=0.7, help="Lower band edge in Hz.")
    parser.add_argument("--fmax", type=float, default=1.0, help="Upper band edge in Hz.")
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


def amplitude_map_after_mean_subtraction(
    cube: np.ndarray, dt: float, fmin: float, fmax: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered_cube = cube - np.mean(cube, axis=0, keepdims=True)
    freqs = np.fft.rfftfreq(centered_cube.shape[0], d=dt)
    spectrum = np.fft.rfft(centered_cube, axis=0)
    mask = (freqs >= fmin) & (freqs <= fmax)
    band_freqs = freqs[mask]
    band_spectrum = spectrum[mask, :, :]
    if band_freqs.size == 0:
        raise RuntimeError(f"No FFT bins found inside [{fmin}, {fmax}] Hz.")

    magnitude = np.abs(band_spectrum)
    peak_index = np.argmax(magnitude, axis=0)
    amplitude_map = np.take_along_axis(
        magnitude,
        peak_index[None, :, :],
        axis=0,
    )[0]
    dominant_freq_map = band_freqs[peak_index]
    return band_freqs, amplitude_map, dominant_freq_map


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        cube, header = read_cube(path)
        band_freqs, amplitude_map, dominant_freq_map = amplitude_map_after_mean_subtraction(
            cube, args.dt, args.fmin, args.fmax
        )
        freq_tag = f"{args.fmin:.3f}_{args.fmax:.3f}Hz".replace(".", "p")
        prefix = f"{path.parent.name}_{path.stem}"

        amplitude_path = args.outdir / f"{prefix}_mean_subtracted_band_{freq_tag}_amplitude_map.fits"
        dominant_freq_path = args.outdir / f"{prefix}_mean_subtracted_band_{freq_tag}_dominant_freq_map.fits"

        write_fits(
            amplitude_path,
            amplitude_map,
            header,
            bunit="fft_amplitude",
            comment="Peak FFT amplitude inside the selected band after mean subtraction.",
        )
        write_fits(
            dominant_freq_path,
            dominant_freq_map,
            header,
            bunit="Hz",
            comment="Dominant FFT frequency inside the selected band after mean subtraction.",
        )

        print(f"[file] {path}")
        print(f"  band_freqs={band_freqs}")
        print(f"  wrote {amplitude_path}")
        print(f"  wrote {dominant_freq_path}")


if __name__ == "__main__":
    main()
