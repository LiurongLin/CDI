#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


REGION_COLUMNS = (
    "region_316_233_r14",
    "region_285_229_r14",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot aperture-sum time series for each deduplicated FITS cube, with both "
            "regions shown on the same figure."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="CSV produced by the region-sum collection script.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/home/liurong/Documents/CDI/CDI_data/1.6.26/dedup_region_plots"),
        help="Directory to save the per-cube plots.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            grouped[row["file"]].append(row)
    return grouped


def plot_cube(rows: list[dict[str, str]], outdir: Path) -> Path:
    rows = sorted(rows, key=lambda item: int(item["frame_index"]))
    cube_path = Path(rows[0]["file"])
    parent_name = rows[0]["parent_dir"]
    x_values = [int(row["frame_index"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        x_values,
        [float(row["region_316_233_r14"]) for row in rows],
        marker="o",
        linewidth=1.8,
        markersize=4,
        label="(316, 233), r=14",
    )
    ax.plot(
        x_values,
        [float(row["region_285_229_r14"]) for row in rows],
        marker="s",
        linewidth=1.8,
        markersize=4,
        label="planet (285, 229), r=14",
    )
    ax.set_title(cube_path.name)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Summed pixel intensity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = outdir / f"{parent_name}_{cube_path.stem}_region_sums.png"
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
        output_path = plot_cube(rows, args.outdir)
        print(f"[ok] wrote {output_path}")


if __name__ == "__main__":
    main()
