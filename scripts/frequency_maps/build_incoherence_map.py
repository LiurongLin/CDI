#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def temporal_incoherence_map(cube: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    data = np.asarray(cube, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D cube (n_frames, ny, nx), got shape {data.shape}")
    mean_map = np.mean(data, axis=0)
    var_map = np.var(data, axis=0, ddof=1 if data.shape[0] > 1 else 0)
    return var_map / np.maximum(mean_map * mean_map, eps)


def temporal_coherence_map(cube: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    incoh = temporal_incoherence_map(cube, eps=eps)
    return 1.0 / np.maximum(incoh, eps)


def read_cube(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        cube = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()
    return cube, header


def write_fits(
    path: Path,
    image: np.ndarray,
    base_header: fits.Header | None = None,
    bunit: str = "arb",
    comment: str = "",
) -> None:
    hdr = fits.Header() if base_header is None else base_header.copy()
    for k in ("NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "BITPIX"):
        if k in hdr:
            del hdr[k]
    hdr["BUNIT"] = bunit
    if comment:
        hdr["COMMENT"] = comment
    fits.writeto(path, np.asarray(image, dtype=np.float32), hdr, overwrite=True)


def maybe_save_png(path: Path, image: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(image, origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.colorbar(im, ax=ax, label="incoherence (var/mean²)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build incoherence maps from FITS cubes using temporal normalized variance "
            "at each pixel: var(I_t) / mean(I_t)^2."
        )
    )
    p.add_argument("inputs", nargs="+", type=Path, help="Input FITS cube paths.")
    p.add_argument("--outdir", type=Path, default=Path.cwd(), help="Output directory.")
    p.add_argument(
        "--combine",
        choices=("none", "mean", "median"),
        default="median",
        help="How to combine individual incoherence maps into one final map.",
    )
    p.add_argument("--save-png", action="store_true", help="Also save PNG preview(s).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    incoh_maps: list[np.ndarray] = []
    coh_maps: list[np.ndarray] = []
    first_header: fits.Header | None = None

    for in_path in args.inputs:
        cube, hdr = read_cube(in_path)
        if first_header is None:
            first_header = hdr
        incoh = temporal_incoherence_map(cube)
        coh = temporal_coherence_map(cube)
        incoh_maps.append(incoh)
        coh_maps.append(coh)

        out_incoh_fits = outdir / f"{in_path.stem}_incoherence_map.fits"
        write_fits(
            out_incoh_fits,
            incoh,
            hdr,
            bunit="var_over_mean2",
            comment="Temporal incoherence map: var(I_t)/mean(I_t)^2 per pixel.",
        )
        print(f"[ok] wrote {out_incoh_fits}")

        out_coh_fits = outdir / f"{in_path.stem}_coherence_map.fits"
        write_fits(
            out_coh_fits,
            coh,
            hdr,
            bunit="mean2_over_var",
            comment="Temporal coherence map: mean(I_t)^2/var(I_t) per pixel.",
        )
        print(f"[ok] wrote {out_coh_fits}")

        if args.save_png:
            out_incoh_png = outdir / f"{in_path.stem}_incoherence_map.png"
            maybe_save_png(out_incoh_png, incoh, f"Incoherence map: {in_path.name}")
            print(f"[ok] wrote {out_incoh_png}")
            out_coh_png = outdir / f"{in_path.stem}_coherence_map.png"
            maybe_save_png(out_coh_png, coh, f"Coherence map: {in_path.name}")
            print(f"[ok] wrote {out_coh_png}")

    if args.combine != "none":
        incoh_stack = np.stack(incoh_maps, axis=0)
        coh_stack = np.stack(coh_maps, axis=0)
        combine_fn = np.nanmean if args.combine == "mean" else np.nanmedian
        incoh_combined = combine_fn(incoh_stack, axis=0)
        coh_combined = combine_fn(coh_stack, axis=0)

        incoh_name = f"incoherence_map_combined_{args.combine}"
        out_incoh_fits = outdir / f"{incoh_name}.fits"
        write_fits(
            out_incoh_fits,
            incoh_combined,
            first_header,
            bunit="var_over_mean2",
            comment="Combined temporal incoherence map.",
        )
        print(f"[ok] wrote {out_incoh_fits}")

        coh_name = f"coherence_map_combined_{args.combine}"
        out_coh_fits = outdir / f"{coh_name}.fits"
        write_fits(
            out_coh_fits,
            coh_combined,
            first_header,
            bunit="mean2_over_var",
            comment="Combined temporal coherence map.",
        )
        print(f"[ok] wrote {out_coh_fits}")
        if args.save_png:
            out_incoh_png = outdir / f"{incoh_name}.png"
            maybe_save_png(out_incoh_png, incoh_combined, f"Combined incoherence map ({args.combine})")
            print(f"[ok] wrote {out_incoh_png}")
            out_coh_png = outdir / f"{coh_name}.png"
            maybe_save_png(out_coh_png, coh_combined, f"Combined coherence map ({args.combine})")
            print(f"[ok] wrote {out_coh_png}")


if __name__ == "__main__":
    main()
