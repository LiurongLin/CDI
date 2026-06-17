#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


GROUP_PATTERN = re.compile(r"^(?P<stem>.+?)(?:_(?P<part>[12]))?\.fit$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pipeline for ring-planet FITS cubes: combine base/_1/_2 files, "
            "remove repeated frames, extract aperture time series, plot FFTs, and build "
            "coherence/incoherence maps from the second FFT peak."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing the input FITS cubes.")
    parser.add_argument("--dt-ms", type=float, default=1.0, help="Frame spacing in milliseconds.")
    parser.add_argument("--sum-x", type=float, required=True, help="Aperture center x.")
    parser.add_argument("--sum-y", type=float, required=True, help="Aperture center y.")
    parser.add_argument("--sum-radius", type=float, required=True, help="Aperture radius in pixels.")
    parser.add_argument("--circle-x", type=float, default=None, help="Planet circle x for PNG previews.")
    parser.add_argument("--circle-y", type=float, default=None, help="Planet circle y for PNG previews.")
    parser.add_argument(
        "--circle-radius",
        type=float,
        default=14.0,
        help="Planet circle radius for PNG previews.",
    )
    parser.add_argument(
        "--image-center-x",
        type=float,
        default=None,
        help="Image center x for drawing ring overlays on map previews.",
    )
    parser.add_argument(
        "--image-center-y",
        type=float,
        default=None,
        help="Image center y for drawing ring overlays on map previews.",
    )
    parser.add_argument(
        "--dedupe-abs-tol",
        type=float,
        default=1.0,
        help="Maximum absolute per-pixel difference for duplicate-frame detection.",
    )
    parser.add_argument(
        "--dedupe-max-changed-pixels",
        type=int,
        default=1,
        help="Maximum changed-pixel count for duplicate-frame detection.",
    )
    parser.add_argument(
        "--incoherence-png-percentile",
        type=float,
        default=98.0,
        help="Upper percentile for incoherence PNG color scaling.",
    )
    parser.add_argument(
        "--snr-planet-radius",
        type=float,
        default=12.0,
        help="Planet aperture radius in pixels for the incoherence-map SNR metric.",
    )
    parser.add_argument(
        "--snr-annulus-width",
        type=float,
        default=24.0,
        help="Annulus width in pixels for the incoherence-map SNR metric.",
    )
    parser.add_argument(
        "--snr-top-n",
        type=int,
        default=50,
        help="Number of brightest planet-region pixels to average for the incoherence-map signal metric.",
    )
    return parser.parse_args()


def read_cube(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        cube = np.asarray(hdul[0].data, dtype=np.float64)
        header = hdul[0].header.copy()
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube in {path}, got shape {cube.shape}")
    return cube, header


def write_cube(path: Path, cube: np.ndarray, header: fits.Header, comment: str) -> None:
    out_header = header.copy()
    out_header["HISTORY"] = comment
    fits.writeto(path, np.asarray(cube, dtype=np.float32), out_header, overwrite=True)


def write_image_fits(path: Path, data: np.ndarray, header: fits.Header, bunit: str, comment: str) -> None:
    out_header = header.copy()
    for key in ("NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "BITPIX"):
        if key in out_header:
            del out_header[key]
    out_header["BUNIT"] = bunit
    out_header["COMMENT"] = comment
    fits.writeto(path, np.asarray(data, dtype=np.float32), out_header, overwrite=True)


def collect_triplets(input_dir: Path) -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted(input_dir.glob("*.fit")):
        if path.stem.endswith("_combined"):
            continue
        match = GROUP_PATTERN.match(path.name)
        if not match:
            continue
        stem = match.group("stem")
        part = match.group("part") or "0"
        groups[stem][part] = path
    return {stem: parts for stem, parts in groups.items() if {"0", "1", "2"}.issubset(parts)}


def dedupe_combined_cube(
    cube: np.ndarray, abs_tol: float, max_changed_pixels: int
) -> tuple[np.ndarray, list[int]]:
    keep_indices: list[int] = []
    for frame_index, frame in enumerate(cube):
        is_duplicate = False
        for kept_index in keep_indices:
            diff = np.abs(frame - cube[kept_index])
            if float(diff.max()) <= abs_tol and int(np.count_nonzero(diff)) <= max_changed_pixels:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(frame_index)
    return cube[keep_indices], keep_indices


def combine_triplets(
    input_dir: Path,
    abs_tol: float,
    max_changed_pixels: int,
) -> list[Path]:
    outputs: list[Path] = []
    for stem, parts in sorted(collect_triplets(input_dir).items()):
        cubes: list[np.ndarray] = []
        header: fits.Header | None = None
        frame_shape: tuple[int, int] | None = None
        source_names: list[str] = []
        for part in ("0", "1", "2"):
            path = parts[part]
            cube, this_header = read_cube(path)
            if header is None:
                header = this_header
                frame_shape = cube.shape[1:]
            elif cube.shape[1:] != frame_shape:
                raise ValueError(f"Spatial shape mismatch in group {stem}")
            cubes.append(cube)
            source_names.append(path.name)

        combined = np.concatenate(cubes, axis=0)
        cleaned, keep_indices = dedupe_combined_cube(
            combined, abs_tol=abs_tol, max_changed_pixels=max_changed_pixels
        )
        output_path = input_dir / f"{stem}_combined.fit"
        out_header = header.copy()
        out_header["NSOURCE"] = (3, "Number of source cubes combined")
        out_header["DUPREM"] = (combined.shape[0] - cleaned.shape[0], "Removed duplicate frames")
        out_header["COMMENT"] = "Combined base/_1/_2 FITS cubes and removed repeated frames."
        out_header["SRCFILES"] = (",".join(source_names)[:68], "Source FITS files (truncated)")
        fits.writeto(output_path, np.asarray(cleaned, dtype=np.float32), out_header, overwrite=True)
        print(f"[combine] {stem}: {combined.shape[0]} -> {cleaned.shape[0]} frames")
        outputs.append(output_path)
    return outputs


def circular_mask(shape: tuple[int, int], center_x: float, center_y: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2


def annulus_mask(
    shape: tuple[int, int],
    center_x: float,
    center_y: float,
    inner_radius: float,
    outer_radius: float,
) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    rr2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
    return (rr2 >= inner_radius ** 2) & (rr2 <= outer_radius ** 2)


def plot_timeseries_single_cube(rows: list[dict[str, object]], plot_dir: Path) -> Path:
    rows = sorted(rows, key=lambda item: int(item["frame_index"]))
    cube_path = Path(str(rows[0]["file"]))
    x_ms = [float(row["time_ms"]) for row in rows]
    y_sum = [float(row["region_sum"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x_ms, y_sum, linewidth=1.8)
    ax.set_title(cube_path.name)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Summed pixel intensity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = plot_dir / f"{cube_path.stem}_region_sum_timeseries.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_timeseries_overlay(grouped_rows: dict[str, list[dict[str, object]]], plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for file_path, rows in sorted(grouped_rows.items()):
        rows = sorted(rows, key=lambda item: int(item["frame_index"]))
        x_ms = [float(row["time_ms"]) for row in rows]
        y_sum = [float(row["region_sum"]) for row in rows]
        ax.plot(x_ms, y_sum, linewidth=1.4, label=Path(file_path).stem)

    ax.set_title("Region-Sum Time Series")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Summed pixel intensity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    output_path = plot_dir / "all_combined_region_sum_timeseries.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def build_timeseries(
    combined_paths: list[Path],
    center_x: float,
    center_y: float,
    radius: float,
    dt_ms: float,
    output_csv: Path,
    plot_dir: Path,
) -> dict[str, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    plot_dir.mkdir(parents=True, exist_ok=True)

    for path in combined_paths:
        cube, _ = read_cube(path)
        sums = cube[:, circular_mask(cube.shape[1:], center_x, center_y, radius)].sum(axis=1)
        for frame_index, value in enumerate(sums):
            row = {
                "file": str(path),
                "parent_dir": path.parent.name,
                "frame_index": frame_index,
                "time_ms": frame_index * dt_ms,
                "center_x": center_x,
                "center_y": center_y,
                "radius": radius,
                "region_sum": float(value),
            }
            rows.append(row)
            grouped_rows[str(path)].append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "file",
            "parent_dir",
            "frame_index",
            "time_ms",
            "center_x",
            "center_y",
            "radius",
            "region_sum",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for cube_rows in grouped_rows.values():
        plot_timeseries_single_cube(cube_rows, plot_dir)
    plot_timeseries_overlay(grouped_rows, plot_dir)
    print(f"[timeseries] wrote {output_csv}")
    return grouped_rows


def build_fft_plots(
    grouped_rows: dict[str, list[dict[str, object]]],
    raw_dir: Path,
    mean_sub_dir: Path,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    mean_sub_dir.mkdir(parents=True, exist_ok=True)
    for file_path, rows in sorted(grouped_rows.items()):
        rows_as_str = [{key: str(value) for key, value in row.items()} for row in rows]
        plot_fft_single_cube(file_path, rows_as_str, raw_dir, subtract_mean=False)
        plot_fft_single_cube(file_path, rows_as_str, mean_sub_dir, subtract_mean=True)
    grouped_as_str = {
        file_path: [{key: str(value) for key, value in row.items()} for row in rows]
        for file_path, rows in grouped_rows.items()
    }
    plot_fft_overlay(grouped_as_str, raw_dir, subtract_mean=False)
    plot_fft_overlay(grouped_as_str, mean_sub_dir, subtract_mean=True)
    print(f"[fft] wrote {raw_dir}")
    print(f"[fft] wrote {mean_sub_dir}")


def fft_from_rows(
    rows: list[dict[str, str]], subtract_mean: bool
) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(rows, key=lambda item: int(item["frame_index"]))
    times_ms = np.array([float(row["time_ms"]) for row in rows], dtype=np.float64)
    values = np.array([float(row["region_sum"]) for row in rows], dtype=np.float64)
    dt_seconds = (times_ms[1] - times_ms[0]) / 1000.0
    signal = values - np.mean(values) if subtract_mean else values
    freqs = np.fft.rfftfreq(signal.size, d=dt_seconds)
    magnitude = np.abs(np.fft.rfft(signal))
    return freqs, magnitude


def plot_fft_single_cube(
    file_path: str,
    rows: list[dict[str, str]],
    outdir: Path,
    subtract_mean: bool,
) -> Path:
    freqs, magnitude = fft_from_rows(rows, subtract_mean=subtract_mean)
    cube_path = Path(file_path)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(freqs, magnitude, linewidth=1.8)
    ax.set_title(f"{cube_path.name} FFT")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("FFT magnitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    suffix = "mean_subtracted" if subtract_mean else "raw"
    output_path = outdir / f"{cube_path.stem}_fft_{suffix}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_fft_overlay(
    grouped_rows: dict[str, list[dict[str, str]]],
    outdir: Path,
    subtract_mean: bool,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for file_path, rows in sorted(grouped_rows.items()):
        freqs, magnitude = fft_from_rows(rows, subtract_mean=subtract_mean)
        ax.plot(freqs, magnitude, linewidth=1.4, label=Path(file_path).stem)

    ax.set_title("Region-Sum FFT")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("FFT magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    suffix = "mean_subtracted" if subtract_mean else "raw"
    output_path = outdir / f"all_combined_region_sum_fft_{suffix}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def second_peak_frequency(rows: list[dict[str, object]]) -> tuple[float, int]:
    rows = sorted(rows, key=lambda item: int(item["frame_index"]))
    times_ms = np.array([float(row["time_ms"]) for row in rows], dtype=np.float64)
    values = np.array([float(row["region_sum"]) for row in rows], dtype=np.float64)
    dt_seconds = (times_ms[1] - times_ms[0]) / 1000.0
    freqs = np.fft.rfftfreq(values.size, d=dt_seconds)
    magnitude = np.abs(np.fft.rfft(values - np.mean(values)))
    valid = np.flatnonzero(freqs > 0.0)
    peak_index = int(valid[np.argmax(magnitude[valid])])
    return float(freqs[peak_index]), peak_index


def build_ratio_maps(
    cube: np.ndarray, selected_index: int, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectrum = np.fft.rfft(cube, axis=0)
    dc_map = np.abs(spectrum[0])
    peak_map = np.abs(spectrum[selected_index])
    ratio_map = peak_map / np.maximum(dc_map, eps)
    return dc_map, peak_map, ratio_map


def ring_overlay_radii(
    cube_path: Path,
    image_center_x: float | None,
    image_center_y: float | None,
    planet_x: float | None,
    planet_y: float | None,
) -> tuple[float | None, float | None]:
    if (
        image_center_x is None
        or image_center_y is None
        or planet_x is None
        or planet_y is None
    ):
        return None, None

    match = re.search(r"ring_(\d+(?:\.\d+)?)id", cube_path.stem)
    if match is None:
        return None, None

    ring_id_width = float(match.group(1))
    ring_width_px = ring_id_width * 12.0
    ring_mid_radius = float(np.hypot(planet_x - image_center_x, planet_y - image_center_y))
    ring_inner_radius = ring_mid_radius - ring_width_px / 2.0
    ring_outer_radius = ring_mid_radius + ring_width_px / 2.0
    return ring_inner_radius, ring_outer_radius


def save_coherence_png(
    path: Path,
    image: np.ndarray,
    title: str,
    colorbar_label: str,
    circle_x: float | None,
    circle_y: float | None,
    circle_radius: float,
    ring_center_x: float | None,
    ring_center_y: float | None,
    ring_inner_radius: float | None,
    ring_outer_radius: float | None,
) -> None:
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(image, origin="lower", cmap="magma")
    if ring_center_x is not None and ring_center_y is not None:
        if ring_inner_radius is not None:
            ax.add_patch(
                Circle(
                    (ring_center_x, ring_center_y),
                    ring_inner_radius,
                    edgecolor="lime",
                    facecolor="none",
                    linewidth=0.8,
                    linestyle="--",
                )
            )
        if ring_outer_radius is not None:
            ax.add_patch(
                Circle(
                    (ring_center_x, ring_center_y),
                    ring_outer_radius,
                    edgecolor="lime",
                    facecolor="none",
                    linewidth=0.8,
                    linestyle="--",
                )
            )
    if circle_x is not None and circle_y is not None:
        ax.add_patch(
            Circle((circle_x, circle_y), circle_radius, edgecolor="cyan", facecolor="none", linewidth=0.9)
        )
    ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_incoherence_png(
    path: Path,
    image: np.ndarray,
    title: str,
    percentile: float,
    circle_x: float | None,
    circle_y: float | None,
    circle_radius: float,
    ring_center_x: float | None,
    ring_center_y: float | None,
    ring_inner_radius: float | None,
    ring_outer_radius: float | None,
    snr_annulus_center_x: float | None,
    snr_annulus_center_y: float | None,
    snr_annulus_inner_radius: float | None,
    snr_annulus_outer_radius: float | None,
    snr_value: float | None,
) -> None:
    from matplotlib.patches import Circle

    finite = image[np.isfinite(image)]
    if finite.size == 0:
        vmin = 0.0
        vmax = 1.0
    else:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanpercentile(finite, percentile))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = float(np.nanmax(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(image, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    if ring_center_x is not None and ring_center_y is not None:
        if ring_inner_radius is not None:
            ax.add_patch(
                Circle(
                    (ring_center_x, ring_center_y),
                    ring_inner_radius,
                    edgecolor="lime",
                    facecolor="none",
                    linewidth=0.8,
                    linestyle="--",
                )
            )
        if ring_outer_radius is not None:
            ax.add_patch(
                Circle(
                    (ring_center_x, ring_center_y),
                    ring_outer_radius,
                    edgecolor="lime",
                    facecolor="none",
                    linewidth=0.8,
                    linestyle="--",
                )
            )
    if snr_annulus_center_x is not None and snr_annulus_center_y is not None:
        if snr_annulus_inner_radius is not None:
            ax.add_patch(
                Circle(
                    (snr_annulus_center_x, snr_annulus_center_y),
                    snr_annulus_inner_radius,
                    edgecolor="white",
                    facecolor="none",
                    linewidth=0.8,
                )
            )
        if snr_annulus_outer_radius is not None:
            ax.add_patch(
                Circle(
                    (snr_annulus_center_x, snr_annulus_center_y),
                    snr_annulus_outer_radius,
                    edgecolor="white",
                    facecolor="none",
                    linewidth=0.8,
                )
            )
    if circle_x is not None and circle_y is not None:
        ax.add_patch(
            Circle((circle_x, circle_y), circle_radius, edgecolor="cyan", facecolor="none", linewidth=0.9)
        )
    if snr_value is not None and np.isfinite(snr_value):
        ax.text(
            0.02,
            0.98,
            f"SNR={snr_value:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=10,
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 4, "edgecolor": "none"},
        )
    ax.set_title(title)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.colorbar(im, ax=ax, label=f"1 / coherence (max={percentile:g}th pct)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_coherence_and_incoherence_maps(
    combined_paths: list[Path],
    grouped_rows: dict[str, list[dict[str, object]]],
    output_dir: Path,
    circle_x: float | None,
    circle_y: float | None,
    circle_radius: float,
    image_center_x: float | None,
    image_center_y: float | None,
    incoherence_png_percentile: float,
    snr_planet_radius: float,
    snr_annulus_width: float,
    snr_top_n: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    snr_rows: list[dict[str, object]] = []
    for cube_path in combined_paths:
        rows = grouped_rows[str(cube_path)]
        peak_freq, peak_index = second_peak_frequency(rows)
        cube, header = read_cube(cube_path)
        dc_map, peak_map, ratio_map = build_ratio_maps(cube, peak_index, eps=1e-12)
        freq_tag = f"{peak_freq:.5f}Hz".replace(".", "p")
        ring_inner_radius, ring_outer_radius = ring_overlay_radii(
            cube_path,
            image_center_x=image_center_x,
            image_center_y=image_center_y,
            planet_x=circle_x,
            planet_y=circle_y,
        )

        ratio_path = output_dir / f"{cube_path.stem}_coherence_ratio_{freq_tag}_over_0Hz.fits"
        peak_path = output_dir / f"{cube_path.stem}_fft_amplitude_{freq_tag}.fits"
        dc_path = output_dir / f"{cube_path.stem}_fft_amplitude_0Hz.fits"
        incoh_path = output_dir / f"{cube_path.stem}_incoherence_ratio_{freq_tag}_over_0Hz.fits"

        write_image_fits(
            ratio_path,
            ratio_map,
            header,
            bunit="fft_ratio",
            comment="Per-pixel ratio |FFT(second_peak)| / |FFT(0Hz)|.",
        )
        write_image_fits(
            peak_path,
            peak_map,
            header,
            bunit="fft_amplitude",
            comment="Per-pixel FFT magnitude at the selected second-peak frequency.",
        )
        write_image_fits(
            dc_path,
            dc_map,
            header,
            bunit="fft_amplitude",
            comment="Per-pixel FFT magnitude at 0 Hz.",
        )
        save_coherence_png(
            ratio_path.with_suffix(".png"),
            ratio_map,
            f"{cube_path.name} coherence ratio {peak_freq:.3f} Hz / 0 Hz",
            "|FFT(f2)| / |FFT(0)|",
            circle_x,
            circle_y,
            circle_radius,
            image_center_x,
            image_center_y,
            ring_inner_radius,
            ring_outer_radius,
        )

        incoherence = 1.0 / np.maximum(ratio_map, 1e-12)
        if (
            circle_x is not None
            and circle_y is not None
            and image_center_x is not None
            and image_center_y is not None
        ):
            planet_mask = circular_mask(
                incoherence.shape,
                center_x=circle_x,
                center_y=circle_y,
                radius=snr_planet_radius,
            )
            annulus_mid_radius = float(
                np.hypot(circle_x - image_center_x, circle_y - image_center_y)
            )
            snr_annulus_inner_radius = annulus_mid_radius - snr_annulus_width / 2.0
            snr_annulus_outer_radius = annulus_mid_radius + snr_annulus_width / 2.0
            noise_mask = annulus_mask(
                incoherence.shape,
                center_x=image_center_x,
                center_y=image_center_y,
                inner_radius=snr_annulus_inner_radius,
                outer_radius=snr_annulus_outer_radius,
            )
            planet_values = np.sort(np.asarray(incoherence[planet_mask], dtype=np.float64))
            top_n = max(1, min(int(snr_top_n), planet_values.size))
            signal_mean = float(np.mean(planet_values[-top_n:]))
            noise_median = float(np.median(incoherence[noise_mask]))
            snr_value = signal_mean / max(noise_median, 1e-12)
        else:
            snr_annulus_inner_radius = None
            snr_annulus_outer_radius = None
            signal_mean = float("nan")
            noise_median = float("nan")
            snr_value = float("nan")

        write_image_fits(
            incoh_path,
            incoherence,
            header,
            bunit="inverse_fft_ratio",
            comment="Pointwise inverse of the coherence ratio map.",
        )
        save_incoherence_png(
            incoh_path.with_suffix(".png"),
            incoherence,
            incoh_path.name,
            incoherence_png_percentile,
            circle_x,
            circle_y,
            circle_radius,
            image_center_x,
            image_center_y,
            ring_inner_radius,
            ring_outer_radius,
            image_center_x,
            image_center_y,
            snr_annulus_inner_radius,
            snr_annulus_outer_radius,
            snr_value,
        )
        snr_rows.append(
            {
                "file": cube_path.name,
                "second_peak_hz": peak_freq,
                "planet_x": circle_x,
                "planet_y": circle_y,
                "planet_radius_px": snr_planet_radius,
                "annulus_center_x": image_center_x,
                "annulus_center_y": image_center_y,
                "annulus_width_px": snr_annulus_width,
                "signal_top_n": snr_top_n,
                "signal_top_n_mean": signal_mean,
                "noise_median": noise_median,
                "snr": snr_value,
            }
        )
        print(f"[maps] wrote coherence/incoherence maps for {cube_path.name}")

    summary_path = output_dir / "incoherence_snr_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "file",
            "second_peak_hz",
            "planet_x",
            "planet_y",
            "planet_radius_px",
            "annulus_center_x",
            "annulus_center_y",
            "annulus_width_px",
            "signal_top_n",
            "signal_top_n_mean",
            "noise_median",
            "snr",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(snr_rows)
    print(f"[maps] wrote {summary_path}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir

    combined_paths = combine_triplets(
        input_dir,
        abs_tol=args.dedupe_abs_tol,
        max_changed_pixels=args.dedupe_max_changed_pixels,
    )
    if not combined_paths:
        raise SystemExit("No complete base/_1/_2 triplets found.")

    stem = f"region_sum_{int(args.sum_x)}_{int(args.sum_y)}_r{int(args.sum_radius)}"
    timeseries_csv = input_dir / f"{stem}_combined.csv"
    timeseries_plot_dir = input_dir / f"{stem}_plots"
    fft_raw_dir = input_dir / f"{stem}_fft_raw"
    fft_mean_sub_dir = input_dir / f"{stem}_fft_mean_subtracted"
    map_dir = input_dir / "second_peak_coherence_maps"

    grouped_rows = build_timeseries(
        combined_paths=combined_paths,
        center_x=args.sum_x,
        center_y=args.sum_y,
        radius=args.sum_radius,
        dt_ms=args.dt_ms,
        output_csv=timeseries_csv,
        plot_dir=timeseries_plot_dir,
    )
    build_fft_plots(grouped_rows, raw_dir=fft_raw_dir, mean_sub_dir=fft_mean_sub_dir)
    build_coherence_and_incoherence_maps(
        combined_paths=combined_paths,
        grouped_rows=grouped_rows,
        output_dir=map_dir,
        circle_x=args.circle_x,
        circle_y=args.circle_y,
        circle_radius=args.circle_radius,
        image_center_x=args.image_center_x,
        image_center_y=args.image_center_y,
        incoherence_png_percentile=args.incoherence_png_percentile,
        snr_planet_radius=args.snr_planet_radius,
        snr_annulus_width=args.snr_annulus_width,
        snr_top_n=args.snr_top_n,
    )
    print("[ok] pipeline complete")


if __name__ == "__main__":
    main()
