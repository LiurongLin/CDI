#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bandpass each pixel time series in a FITS cube, reconstruct the filtered "
            "signal with an IFFT, and build a band-limited incoherence map."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input FITS cube paths.")
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Sampling interval between frames.",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=0.5,
        help="Lower frequency bound of the retained band.",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=1.3,
        help="Upper frequency bound of the retained band.",
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
        help="Also save the band-limited reconstructed time cube.",
    )
    return parser.parse_args()


def read_cube(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube in {path}, got shape {cube.shape}")
    return cube, header


def write_fits(
    path: Path,
    data: np.ndarray,
    header: fits.Header,
    bunit: str,
    comment: str,
) -> None:
    out_header = header.copy()
    for key in ("NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "BITPIX"):
        if key in out_header:
            del out_header[key]
    out_header["BUNIT"] = bunit
    out_header["COMMENT"] = comment
    fits.writeto(path, np.asarray(data, dtype=np.float32), out_header, overwrite=True)


def bandpass_cube(
    cube: np.ndarray, dt: float, fmin: float, fmax: float
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = cube.shape[0]
    freqs = np.fft.rfftfreq(n_frames, d=dt)
    spectrum = np.fft.rfft(cube, axis=0)
    keep = (freqs >= fmin) & (freqs <= fmax)
    filtered_spectrum = np.zeros_like(spectrum)
    filtered_spectrum[keep, :, :] = spectrum[keep, :, :]
    filtered_cube = np.fft.irfft(filtered_spectrum, n=n_frames, axis=0)
    return freqs[keep], filtered_cube


def band_limited_incoherence_map(
    original_cube: np.ndarray, filtered_cube: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    mean_map = np.mean(original_cube, axis=0)
    var_map = np.var(filtered_cube, axis=0, ddof=1 if filtered_cube.shape[0] > 1 else 0)
    rms_map = np.sqrt(np.mean(filtered_cube ** 2, axis=0))
    incoherence_map = var_map / np.maximum(mean_map * mean_map, eps)
    return incoherence_map, rms_map


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        cube, header = read_cube(path)
        kept_freqs, filtered_cube = bandpass_cube(cube, args.dt, args.fmin, args.fmax)
        incoherence_map, rms_map = band_limited_incoherence_map(cube, filtered_cube)

        freq_tag = f"{args.fmin:.3f}_{args.fmax:.3f}Hz".replace(".", "p")

        incoh_path = args.outdir / f"{path.stem}_bandpass_{freq_tag}_incoherence_map.fits"
        rms_path = args.outdir / f"{path.stem}_bandpass_{freq_tag}_rms_map.fits"

        write_fits(
            incoh_path,
            incoherence_map,
            header,
            bunit="band_var_over_mean2",
            comment="Band-limited incoherence map from IFFT-reconstructed pixel time series.",
        )
        write_fits(
            rms_path,
            rms_map,
            header,
            bunit="band_rms",
            comment="Band-limited RMS map from IFFT-reconstructed pixel time series.",
        )
        print(f"[file] {path}")
        print(f"  frames={cube.shape[0]} kept_bins={kept_freqs.size} kept_freqs={kept_freqs}")
        print(f"  wrote {incoh_path}")
        print(f"  wrote {rms_path}")

        if args.save_filtered_cube:
            filtered_cube_path = (
                args.outdir / f"{path.stem}_bandpass_{freq_tag}_filtered_cube.fits"
            )
            write_fits(
                filtered_cube_path,
                filtered_cube,
                header,
                bunit="filtered_signal",
                comment="Band-limited cube reconstructed by IFFT after FFT masking.",
            )
            print(f"  wrote {filtered_cube_path}")


if __name__ == "__main__":
    main()
