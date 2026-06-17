#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a map by FFT over time, isolating one frequency bin, IFFT back to "
            "time domain, and taking the mean over frames."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input FITS cube paths.")
    parser.add_argument("--dt", type=float, default=0.1, help="Sampling interval.")
    parser.add_argument(
        "--target-freq",
        type=float,
        default=0.88235,
        help="Target frequency to isolate.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path.cwd(),
        help="Output directory.",
    )
    parser.add_argument(
        "--save-filtered-cube",
        action="store_true",
        help="Also save the reconstructed single-frequency cube.",
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


def isolate_single_frequency(
    cube: np.ndarray, dt: float, target_freq: float
) -> tuple[float, int, np.ndarray]:
    n_frames = cube.shape[0]
    freqs = np.fft.rfftfreq(n_frames, d=dt)
    target_index = int(np.argmin(np.abs(freqs - target_freq)))
    selected_freq = float(freqs[target_index])
    spectrum = np.fft.rfft(cube, axis=0)
    filtered_spectrum = np.zeros_like(spectrum)
    filtered_spectrum[target_index, :, :] = spectrum[target_index, :, :]
    filtered_cube = np.fft.irfft(filtered_spectrum, n=n_frames, axis=0)
    return selected_freq, target_index, filtered_cube


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        cube, header = read_cube(path)
        selected_freq, selected_index, filtered_cube = isolate_single_frequency(
            cube, args.dt, args.target_freq
        )
        mean_map = np.mean(filtered_cube, axis=0)
        freq_tag = f"{selected_freq:.5f}Hz".replace(".", "p")

        map_path = args.outdir / f"{path.stem}_ifft_mean_map_{freq_tag}.fits"
        write_fits(
            map_path,
            mean_map,
            header,
            bunit="ifft_mean_signal",
            comment="Mean of time-domain cube reconstructed from one FFT frequency bin.",
        )

        print(f"[file] {path}")
        print(
            f"  target_freq={args.target_freq:.5f} selected_freq={selected_freq:.8f} "
            f"selected_index={selected_index}"
        )
        print(f"  wrote {map_path}")

        if args.save_filtered_cube:
            cube_path = args.outdir / f"{path.stem}_ifft_filtered_cube_{freq_tag}.fits"
            write_fits(
                cube_path,
                filtered_cube,
                header,
                bunit="ifft_filtered_signal",
                comment="Time-domain cube reconstructed from one FFT frequency bin.",
            )
            print(f"  wrote {cube_path}")


if __name__ == "__main__":
    main()
