#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits


@dataclass(frozen=True)
class DuplicateMatch:
    removed_index: int
    kept_index: int
    max_abs_diff: float
    mean_abs_diff: float
    changed_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove repeated frames from 3D FITS cubes and save cleaned cubes "
            "as new FITS files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="FITS files or directories to scan recursively for .fit/.fits cubes.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_dedup",
        help="Suffix appended to the original stem for the cleaned FITS file.",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1.0,
        help="Maximum absolute per-pixel difference allowed for duplicate frames.",
    )
    parser.add_argument(
        "--max-changed-pixels",
        type=int,
        default=1,
        help="Maximum number of pixels allowed to differ for duplicate frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cleaned FITS files.",
    )
    return parser.parse_args()


def collect_fits_paths(inputs: list[Path], output_suffix: str) -> list[Path]:
    paths: list[Path] = []
    for entry in inputs:
        if entry.is_dir():
            paths.extend(
                sorted(
                    path
                    for path in entry.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in {".fit", ".fits"}
                    and not path.stem.endswith(output_suffix)
                )
            )
        elif (
            entry.is_file()
            and entry.suffix.lower() in {".fit", ".fits"}
            and not entry.stem.endswith(output_suffix)
        ):
            paths.append(entry)
        else:
            raise FileNotFoundError(f"No FITS file or directory found at {entry}")
    return paths


def read_cube(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[0].data)
        header = hdul[0].header.copy()
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D cube in {path}, got shape {data.shape}")
    return data, header


def compare_frames(frame_a: np.ndarray, frame_b: np.ndarray) -> tuple[float, float, int]:
    diff = np.abs(
        np.asarray(frame_a, dtype=np.float64) - np.asarray(frame_b, dtype=np.float64)
    )
    return float(diff.max()), float(diff.mean()), int(np.count_nonzero(diff))


def find_duplicates(
    cube: np.ndarray, abs_tol: float, max_changed_pixels: int
) -> tuple[list[int], list[DuplicateMatch]]:
    keep_indices: list[int] = []
    removed: list[DuplicateMatch] = []

    for frame_index, frame in enumerate(cube):
        matched_duplicate: DuplicateMatch | None = None
        for kept_index in keep_indices:
            max_abs_diff, mean_abs_diff, changed_pixels = compare_frames(
                frame, cube[kept_index]
            )
            if max_abs_diff <= abs_tol and changed_pixels <= max_changed_pixels:
                matched_duplicate = DuplicateMatch(
                    removed_index=frame_index,
                    kept_index=kept_index,
                    max_abs_diff=max_abs_diff,
                    mean_abs_diff=mean_abs_diff,
                    changed_pixels=changed_pixels,
                )
                break

        if matched_duplicate is None:
            keep_indices.append(frame_index)
        else:
            removed.append(matched_duplicate)

    return keep_indices, removed


def build_output_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}.fits")


def write_clean_cube(
    output_path: Path,
    clean_cube: np.ndarray,
    header: fits.Header,
    removed: list[DuplicateMatch],
    overwrite: bool,
) -> None:
    clean_header = header.copy()
    clean_header["HISTORY"] = "Duplicate frames removed by dedupe_fits_cubes.py"
    clean_header["DUPREM"] = (len(removed), "Removed duplicate frames")
    if removed:
        clean_header["DUPKEEP"] = (
            ",".join(str(item.kept_index) for item in removed[:20]),
            "Kept frame indices",
        )
        clean_header["DUPDROP"] = (
            ",".join(str(item.removed_index) for item in removed[:20]),
            "Removed frame indices",
        )
    fits.writeto(output_path, clean_cube, header=clean_header, overwrite=overwrite)


def process_cube(
    path: Path,
    output_suffix: str,
    abs_tol: float,
    max_changed_pixels: int,
    overwrite: bool,
) -> None:
    cube, header = read_cube(path)
    keep_indices, removed = find_duplicates(cube, abs_tol, max_changed_pixels)
    clean_cube = cube[keep_indices]
    output_path = build_output_path(path, output_suffix)
    write_clean_cube(output_path, clean_cube, header, removed, overwrite=overwrite)

    print(f"[file] {path}")
    print(f"  original_frames={cube.shape[0]} cleaned_frames={clean_cube.shape[0]}")
    print(f"  output={output_path}")
    if removed:
        print("  removed_frames:")
        for item in removed:
            print(
                "    "
                f"drop {item.removed_index} -> keep {item.kept_index} "
                f"(max_abs_diff={item.max_abs_diff:.6g}, "
                f"changed_pixels={item.changed_pixels}, "
                f"mean_abs_diff={item.mean_abs_diff:.6g})"
            )
    else:
        print("  removed_frames: none")


def main() -> None:
    args = parse_args()
    fits_paths = collect_fits_paths(args.inputs, args.output_suffix)
    if not fits_paths:
        raise SystemExit("No FITS files found.")

    for path in fits_paths:
        process_cube(
            path=path,
            output_suffix=args.output_suffix,
            abs_tol=args.abs_tol,
            max_changed_pixels=args.max_changed_pixels,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
