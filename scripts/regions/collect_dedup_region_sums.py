#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class CircularRegion:
    name: str
    center_x: int
    center_y: int
    radius: float


REGIONS = (
    CircularRegion(name="region_316_233_r14", center_x=316, center_y=233, radius=14.0),
    CircularRegion(name="region_285_229_r14", center_x=285, center_y=229, radius=14.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect frame-by-frame summed intensities in fixed circular regions "
            "from deduplicated FITS cubes."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Deduplicated FITS files or directories to scan recursively.",
    )
    parser.add_argument(
        "--suffix",
        default="_dedup",
        help="Only process FITS files whose stem ends with this suffix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/liurong/Documents/CDI/CDI_data/1.6.26/dedup_region_sums.csv"),
        help="CSV file to write the combined results to.",
    )
    return parser.parse_args()


def collect_dedup_paths(inputs: list[Path], suffix: str) -> list[Path]:
    paths: list[Path] = []
    for entry in inputs:
        if entry.is_dir():
            paths.extend(
                sorted(
                    path
                    for path in entry.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in {".fit", ".fits"}
                    and path.stem.endswith(suffix)
                )
            )
        elif (
            entry.is_file()
            and entry.suffix.lower() in {".fit", ".fits"}
            and entry.stem.endswith(suffix)
        ):
            paths.append(entry)
        else:
            raise FileNotFoundError(
                f"No deduplicated FITS file or directory found at {entry}"
            )
    return paths


def read_cube(path: Path) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube in {path}, got shape {cube.shape}")
    return cube


def circular_mask(shape: tuple[int, int], region: CircularRegion) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - region.center_x) ** 2 + (yy - region.center_y) ** 2 <= region.radius ** 2


def region_sums(cube: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = cube[:, mask]
    return masked.sum(axis=1)


def main() -> None:
    args = parse_args()
    fits_paths = collect_dedup_paths(args.inputs, args.suffix)
    if not fits_paths:
        raise SystemExit("No deduplicated FITS files found.")

    rows: list[dict[str, object]] = []

    for path in fits_paths:
        cube = read_cube(path)
        masks = {region.name: circular_mask(cube.shape[1:], region) for region in REGIONS}
        sums = {name: region_sums(cube, mask) for name, mask in masks.items()}

        print(f"[file] {path}")
        print(f"  frames={cube.shape[0]}")

        for frame_index in range(cube.shape[0]):
            row = {
                "file": str(path),
                "parent_dir": path.parent.name,
                "frame_index": frame_index,
            }
            for region in REGIONS:
                row[region.name] = float(sums[region.name][frame_index])
            rows.append(row)

        for region in REGIONS:
            print(
                f"  {region.name}: "
                f"min={np.min(sums[region.name]):.6f} "
                f"max={np.max(sums[region.name]):.6f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file", "parent_dir", "frame_index"] + [region.name for region in REGIONS]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
