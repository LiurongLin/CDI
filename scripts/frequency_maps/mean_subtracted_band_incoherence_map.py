#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subtract the per-pixel mean before FFT, keep a frequency band, "
            "reconstruct with IFFT, and build a band-limited incoherence map."
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
    parser.add_argument(
        "--save-filtered-cube",
        action="store_true",
        help="Also save the reconstructed band-limited cube.",
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


def reconstruct_band_after_mean_subtraction(
    cube: np.ndarray, dt: float, fmin: float, fmax: float
) -> tuple[np.ndarray, np.ndarray]:
    centered_cube = cube - np.mean(cube, axis=0, keepdims=True)
    freqs = np.fft.rfftfreq(centered_cube.shape[0], d=dt)
    spectrum = np.fft.rfft(centered_cube, axis=0)
    mask = (freqs >= fmin) & (freqs <= fmax)
    filtered = np.zeros_like(spectrum)
    filtered[mask, :, :] = spectrum[mask, :, :]
    reconstructed = np.fft.irfft(filtered, n=centered_cube.shape[0], axis=0)
    return freqs[mask], reconstructed


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        cube, header = read_cube(path)
        kept_freqs, filtered_cube = reconstruct_band_after_mean_subtraction(
            cube, args.dt, args.fmin, args.fmax
        )
        incoherence_map = np.sqrt(np.mean(filtered_cube ** 2, axis=0))
        freq_tag = f"{args.fmin:.3f}_{args.fmax:.3f}Hz".replace(".", "p")
        prefix = f"{path.parent.name}_{path.stem}"

        map_path = args.outdir / f"{prefix}_mean_subtracted_band_{freq_tag}_incoherence_map.fits"
        write_fits(
            map_path,
            incoherence_map,
            header,
            bunit="band_rms",
            comment="RMS of mean-subtracted, band-limited IFFT-reconstructed cube.",
        )

        print(f"[file] {path}")
        print(f"  frames={cube.shape[0]} kept_freqs={kept_freqs}")
        print(f"  wrote {map_path}")

        if args.save_filtered_cube:
            cube_path = (
                args.outdir / f"{prefix}_mean_subtracted_band_{freq_tag}_filtered_cube.fits"
            )
            write_fits(
                cube_path,
                filtered_cube,
                header,
                bunit="filtered_signal",
                comment="Mean-subtracted, band-limited cube reconstructed by IFFT.",
            )
            print(f"  wrote {cube_path}")


if __name__ == "__main__":
    main()
