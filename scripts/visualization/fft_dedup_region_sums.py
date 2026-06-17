#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and plot FFT spectra of deduplicated aperture-sum time series, "
            "with both regions shown on the same figure for each cube."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="CSV produced by the region-sum collection script.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Sampling interval between consecutive data points.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/home/liurong/Documents/CDI/CDI_data/1.6.26/dedup_region_fft_plots"),
        help="Directory to save the per-cube FFT plots.",
    )
    parser.add_argument(
        "--subtract-mean",
        action="store_true",
        help="Subtract the time-series mean before FFT.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[row["file"]].append(row)
    return grouped


def fft_magnitude(values: np.ndarray, dt: float, subtract_mean: bool) -> tuple[np.ndarray, np.ndarray]:
    signal = values - np.mean(values) if subtract_mean else values
    freqs = np.fft.rfftfreq(signal.size, d=dt)
    spectrum = np.fft.rfft(signal)
    magnitude = np.abs(spectrum)
    return freqs, magnitude


def plot_cube_fft(
    rows: list[dict[str, str]],
    outdir: Path,
    dt: float,
    subtract_mean: bool,
) -> Path:
    rows = sorted(rows, key=lambda item: int(item["frame_index"]))
    cube_path = Path(rows[0]["file"])
    parent_name = rows[0]["parent_dir"]

    region_a = np.asarray([float(row["region_316_233_r14"]) for row in rows], dtype=np.float64)
    region_b = np.asarray([float(row["region_285_229_r14"]) for row in rows], dtype=np.float64)

    freqs_a, mag_a = fft_magnitude(region_a, dt, subtract_mean)
    freqs_b, mag_b = fft_magnitude(region_b, dt, subtract_mean)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        freqs_a,
        mag_a,
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="(316, 233), r=14",
    )
    ax.plot(
        freqs_b,
        mag_b,
        marker="s",
        linewidth=1.8,
        markersize=4,
        label="planet (285, 229), r=14",
    )
    title_suffix = " (mean-subtracted)" if subtract_mean else " (raw)"
    ax.set_title(f"FFT: {cube_path.name}{title_suffix}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("FFT magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = outdir / f"{parent_name}_{cube_path.stem}_region_sums_fft.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    grouped_rows = load_rows(args.csv_path)
    if not grouped_rows:
        raise SystemExit("No rows found in CSV.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    for _, rows in sorted(grouped_rows.items()):
        output_path = plot_cube_fft(rows, args.outdir, args.dt, args.subtract_mean)
        print(f"[ok] wrote {output_path}")


if __name__ == "__main__":
    main()
