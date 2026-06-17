#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 3D FITS cube as a frame contact sheet and GIF."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input 3D FITS cube paths.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory. Defaults to each cube directory.",
    )
    parser.add_argument(
        "--cmap",
        default="magma",
        help="Matplotlib colormap for frames.",
    )
    parser.add_argument(
        "--gif-ms",
        type=int,
        default=180,
        help="Frame duration for GIF in milliseconds.",
    )
    parser.add_argument(
        "--vmin-percentile",
        type=float,
        default=2.0,
        help="Lower percentile for display scaling.",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=98.0,
        help="Upper percentile for display scaling.",
    )
    parser.add_argument(
        "--flip-vertical",
        action="store_true",
        help="Flip each frame vertically before rendering.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Apply shifted log10 scaling before rendering.",
    )
    parser.add_argument(
        "--log10-positive",
        action="store_true",
        help="Apply log10(max(image, 1)) before rendering.",
    )
    return parser.parse_args()


def read_cube(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D FITS cube in {path}, got shape {cube.shape}")
    return cube


def normalize_frame(frame: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    scaled = np.clip((frame - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def prepare_cube_for_display(
    cube: np.ndarray,
    log_scale: bool,
    log10_positive: bool,
) -> np.ndarray:
    if log10_positive:
        return np.log10(np.maximum(cube, 1.0))
    if not log_scale:
        return cube
    floor = np.min(cube)
    shifted = cube - floor
    return np.log10(shifted + 1.0)


def save_contact_sheet(
    path: Path,
    cube: np.ndarray,
    cmap: str,
    flip_vertical: bool,
    vmin_percentile: float,
    vmax_percentile: float,
    log_scale: bool,
    log10_positive: bool,
) -> Path:
    n_frames = cube.shape[0]
    ncols = min(6, n_frames)
    nrows = math.ceil(n_frames / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.4 * nrows))
    axes_array = np.atleast_1d(axes).ravel()

    display_cube = prepare_cube_for_display(cube, log_scale, log10_positive)
    vmin = float(np.percentile(display_cube, vmin_percentile))
    vmax = float(np.percentile(display_cube, vmax_percentile))

    for idx, ax in enumerate(axes_array):
        if idx < n_frames:
            frame = np.flipud(display_cube[idx]) if flip_vertical else display_cube[idx]
            im = ax.imshow(frame, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"Frame {idx}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= n_frames:
            ax.axis("off")

    fig.suptitle(path.stem, fontsize=12)
    fig.tight_layout()
    output_path = Path(f"{path}.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_gif(
    path: Path,
    cube: np.ndarray,
    cmap: str,
    gif_ms: int,
    flip_vertical: bool,
    vmin_percentile: float,
    vmax_percentile: float,
    log_scale: bool,
    log10_positive: bool,
) -> Path:
    colormap = plt.get_cmap(cmap)
    display_cube = prepare_cube_for_display(cube, log_scale, log10_positive)
    vmin = float(np.percentile(display_cube, vmin_percentile))
    vmax = float(np.percentile(display_cube, vmax_percentile))

    frames: list[Image.Image] = []
    for frame in display_cube:
        frame_to_render = np.flipud(frame) if flip_vertical else frame
        gray = normalize_frame(frame_to_render, vmin, vmax)
        rgba = (colormap(gray / 255.0) * 255).astype(np.uint8)
        frames.append(Image.fromarray(rgba[:, :, :3], mode="RGB"))

    output_path = Path(f"{path}.gif")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=gif_ms,
        loop=0,
    )
    return output_path


def main() -> None:
    args = parse_args()

    for input_path in args.inputs:
        cube = read_cube(input_path)
        target_dir = input_path.parent if args.outdir is None else args.outdir
        target_dir.mkdir(parents=True, exist_ok=True)
        base_path = target_dir / f"{input_path.parent.name}_{input_path.stem}"

        png_path = save_contact_sheet(
            base_path,
            cube,
            args.cmap,
            args.flip_vertical,
            args.vmin_percentile,
            args.vmax_percentile,
            args.log_scale,
            args.log10_positive,
        )
        gif_path = save_gif(
            base_path,
            cube,
            args.cmap,
            args.gif_ms,
            args.flip_vertical,
            args.vmin_percentile,
            args.vmax_percentile,
            args.log_scale,
            args.log10_positive,
        )

        print(f"[file] {input_path}")
        print(f"  wrote {png_path}")
        print(f"  wrote {gif_path}")


if __name__ == "__main__":
    main()
