#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build coherence and incoherence maps from a FITS cube using FFT windows "
            "without baseline subtraction."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input FITS cube paths.")
    parser.add_argument("--dt", type=float, default=0.1, help="Sampling interval.")
    parser.add_argument(
        "--coh-freq",
        type=float,
        default=0.0,
        help="Center frequency for the coherence window.",
    )
    parser.add_argument(
        "--incoh-freq",
        type=float,
        default=0.88235,
        help="Center frequency for the incoherence window.",
    )
    parser.add_argument(
        "--coh-half-width",
        type=float,
        default=1e-9,
        help="Half-width of the coherence window in Hz.",
    )
    parser.add_argument(
        "--incoh-half-width",
        type=float,
        default=1e-9,
        help="Half-width of the incoherence window in Hz.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path.cwd(),
        help="Output directory.",
    )
    parser.add_argument(
        "--save-reconstructed-cubes",
        action="store_true",
        help="Also save the IFFT-reconstructed cubes for the two windows.",
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


def reconstruct_window(
    cube: np.ndarray, dt: float, center_freq: float, half_width: float
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = cube.shape[0]
    freqs = np.fft.rfftfreq(n_frames, d=dt)
    spectrum = np.fft.rfft(cube, axis=0)
    mask = (freqs >= center_freq - half_width) & (freqs <= center_freq + half_width)
    filtered = np.zeros_like(spectrum)
    filtered[mask, :, :] = spectrum[mask, :, :]
    reconstructed = np.fft.irfft(filtered, n=n_frames, axis=0)
    return freqs[mask], reconstructed


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        cube, header = read_cube(path)
        coh_freqs, coh_cube = reconstruct_window(
            cube, args.dt, args.coh_freq, args.coh_half_width
        )
        incoh_freqs, incoh_cube = reconstruct_window(
            cube, args.dt, args.incoh_freq, args.incoh_half_width
        )

        coherence_map = np.mean(coh_cube, axis=0)
        incoherence_map = np.mean(incoh_cube, axis=0)

        coh_tag = f"{args.coh_freq:.5f}Hz".replace(".", "p")
        incoh_tag = f"{args.incoh_freq:.5f}Hz".replace(".", "p")

        coh_path = args.outdir / f"{path.stem}_coherence_map_{coh_tag}.fits"
        incoh_path = args.outdir / f"{path.stem}_incoherence_map_{incoh_tag}.fits"

        write_fits(
            coh_path,
            coherence_map,
            header,
            bunit="ifft_mean_signal",
            comment="Mean of IFFT-reconstructed cube from coherence frequency window.",
        )
        write_fits(
            incoh_path,
            incoherence_map,
            header,
            bunit="ifft_mean_signal",
            comment="Mean of IFFT-reconstructed cube from incoherence frequency window.",
        )

        print(f"[file] {path}")
        print(f"  coherence_freqs={coh_freqs}")
        print(f"  incoherence_freqs={incoh_freqs}")
        print(f"  wrote {coh_path}")
        print(f"  wrote {incoh_path}")

        if args.save_reconstructed_cubes:
            coh_cube_path = args.outdir / f"{path.stem}_coherence_cube_{coh_tag}.fits"
            incoh_cube_path = args.outdir / f"{path.stem}_incoherence_cube_{incoh_tag}.fits"
            write_fits(
                coh_cube_path,
                coh_cube,
                header,
                bunit="ifft_signal",
                comment="IFFT-reconstructed cube from coherence frequency window.",
            )
            write_fits(
                incoh_cube_path,
                incoh_cube,
                header,
                bunit="ifft_signal",
                comment="IFFT-reconstructed cube from incoherence frequency window.",
            )
            print(f"  wrote {coh_cube_path}")
            print(f"  wrote {incoh_cube_path}")


if __name__ == "__main__":
    main()
