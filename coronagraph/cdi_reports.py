from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from .region_shapes import annulus_radii_from_width, build_touching_circle_ring
from .simulator import CoronagraphSimulator


def _has_roi_indication_mode(phase_sweep_mode: str) -> bool:
    return str(phase_sweep_mode).strip().lower() not in {
        "global",
        "focal_plane",
        "mask_rotation",
    }


def _write_rgb_gif(rgb8_frames: np.ndarray, gif_path: str, duration_ms: int = 500) -> None:
    saved_gif = False
    try:
        import imageio.v2 as imageio

        imageio.mimsave(gif_path, list(rgb8_frames), duration=duration_ms / 1000.0, loop=0)
        saved_gif = True
    except Exception:
        try:
            from PIL import Image

            frames = [Image.fromarray(frame, mode="RGB") for frame in rgb8_frames]
            if len(frames) > 0:
                frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=duration_ms,
                    loop=0,
                )
                saved_gif = True
        except Exception:
            saved_gif = False

    if not saved_gif:
        raise RuntimeError("GIF export requires either 'imageio' or 'Pillow' (PIL) to be installed.")


def _save_roi_size_incoherence_pdf_for_planet_location(
    *,
    output_path: str,
    region_shape_name: str,
    panels: list[dict[str, object]],
    extent: list[float],
    poster_figure: bool = False,
    include_mean_map_page: bool | None = None,
) -> None:
    _save_roi_size_map_pdf_for_planet_location(
        output_path=output_path,
        region_shape_name=region_shape_name,
        panels=panels,
        extent=extent,
        map_key="incoherence_map",
        title_prefix="Incoherence",
        cmap="inferno",
        snr_key="snr",
        snr_label="SNR",
        curve_snr_key="snr",
        curve_snr_label="SNR",
        vmax_percentile=None,
        include_mean_map_page=(not bool(poster_figure) if include_mean_map_page is None else bool(include_mean_map_page)),
        poster_figure=poster_figure,
    )


def _save_roi_size_coherence_pdf_for_planet_location(
    *,
    output_path: str,
    region_shape_name: str,
    panels: list[dict[str, object]],
    extent: list[float],
    poster_figure: bool = False,
) -> None:
    _save_roi_size_map_pdf_for_planet_location(
        output_path=output_path,
        region_shape_name=region_shape_name,
        panels=panels,
        extent=extent,
        map_key="coherence_map",
        title_prefix="Coherence",
        cmap="inferno",
        snr_key="snr",
        snr_label="SNR",
        curve_snr_key="snr",
        curve_snr_label="SNR",
        vmax_percentile=None,
        poster_figure=poster_figure,
    )


def _build_planet_position_map_report_groups(
    location_panels: list[dict[str, object]],
) -> list[dict[str, object]]:
    if len(location_panels) == 0:
        return []

    def _f(value: object) -> float:
        return float(value)

    def _format_tag(prefix: str, value: float) -> str:
        return f"{prefix}_{value:.3f}".replace(".", "p")

    def _location_sort_key(item: dict[str, object]) -> tuple[float, float, float]:
        return (
            _f(item["lyot_reference_percent"]) if "lyot_reference_percent" in item else 100.0,
            _f(item["planet_radius_lamD"]),
            _f(item["planet_theta_deg"]),
        )

    radius_values = sorted({_f(item["planet_radius_lamD"]) for item in location_panels})
    theta_values = sorted({_f(item["planet_theta_deg"]) for item in location_panels})
    roi_values = sorted(
        {
            _f(panel["requested_roi_size_lamD"])
            for item in location_panels
            for panel in item["panels"]
        }
    )
    varying = [
        name
        for name, values in (
            ("radius", radius_values),
            ("roi", roi_values),
            ("theta", theta_values),
        )
        if len(values) > 1
    ]
    sorted_locations = sorted(location_panels, key=_location_sort_key)

    if len(varying) <= 1:
        if varying == ["roi"]:
            panel_collections = [list(sorted_locations[0]["panels"])]
        else:
            panel_collections = [list(item["panels"]) for item in sorted_locations]
        return [
            {
                "file_tag": "grouped_all",
                "panel_collections": panel_collections,
            }
        ]

    if "radius" in varying:
        groups: list[dict[str, object]] = []
        for radius_value in radius_values:
            collections = [
                list(item["panels"])
                for item in sorted_locations
                if np.isclose(_f(item["planet_radius_lamD"]), radius_value)
            ]
            if collections:
                groups.append(
                    {
                        "file_tag": f"grouped_by_radius_{_format_tag('r', radius_value)}",
                        "panel_collections": collections,
                    }
                )
        return groups

    if "roi" in varying:
        groups = []
        for roi_value in roi_values:
            collections = []
            for item in sorted_locations:
                matched = [
                    panel
                    for panel in item["panels"]
                    if np.isclose(_f(panel["requested_roi_size_lamD"]), roi_value)
                ]
                if matched:
                    collections.append(matched)
            if collections:
                groups.append(
                    {
                        "file_tag": f"grouped_by_roi_{_format_tag('roi', roi_value)}",
                        "panel_collections": collections,
                    }
                )
        return groups

    groups = []
    for theta_value in theta_values:
        collections = [
            list(item["panels"])
            for item in sorted_locations
            if np.isclose(_f(item["planet_theta_deg"]), theta_value)
        ]
        if collections:
            groups.append(
                {
                    "file_tag": f"grouped_by_theta_{_format_tag('theta', theta_value)}",
                    "panel_collections": collections,
                }
            )
    return groups


def _save_grouped_roi_size_map_pdfs(
    *,
    output_dir: str,
    base_name: str,
    region_shape_name: str,
    location_panels: list[dict[str, object]],
    extent: list[float],
    save_panel_pdf,
    poster_figure: bool = False,
) -> list[str]:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm

    groups = _build_planet_position_map_report_groups(location_panels)
    saved_paths: list[str] = []
    if len(groups) == 0:
        return saved_paths

    if poster_figure:
        rows_per_page = 2
        title_fontsize = 13
        label_fontsize = 14
        tick_fontsize = 12
        snr_fontsize = 12
        overlay_linewidth = 2.2
        spine_linewidth = 1.6
        annotation_size = 12
        figsize_scale = (5.6, 4.9)
    else:
        rows_per_page = 3
        title_fontsize = 9
        label_fontsize = 10
        tick_fontsize = 9
        snr_fontsize = 8
        overlay_linewidth = 1.3
        spine_linewidth = 0.8
        annotation_size = 8
        figsize_scale = (4.2, 4.0)

    for group in groups:
        output_path = os.path.join(output_dir, f"{base_name}_{group['file_tag']}.pdf")
        panel_collections = list(group["panel_collections"])
        if len(panel_collections) == 0:
            continue
        ncols = max(len(panels) for panels in panel_collections)
        with PdfPages(output_path) as pdf:
            for start in range(0, len(panel_collections), rows_per_page):
                chunk = panel_collections[start:start + rows_per_page]
                nrows = len(chunk)
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(figsize_scale[0] * ncols, figsize_scale[1] * nrows),
                    constrained_layout=True,
                    squeeze=False,
                )
                axes_arr = np.asarray(axes, dtype=object)
                shared_im = None
                cbar_label = None
                for row_idx, panels in enumerate(chunk):
                    panels_sorted = sorted(panels, key=lambda p: float(p["requested_roi_size_lamD"]))
                    for col_idx in range(ncols):
                        ax = axes_arr[row_idx, col_idx]
                        if col_idx >= len(panels_sorted):
                            ax.axis("off")
                            continue
                        panel = panels_sorted[col_idx]
                        map_key = "incoherence_map" if save_panel_pdf is _save_roi_size_incoherence_pdf_for_planet_location else "coherence_map"
                        title_prefix = "Incoherence" if map_key == "incoherence_map" else "Coherence"
                        map_arr = np.asarray(panel[map_key], dtype=float)
                        finite_vals = map_arr[np.isfinite(map_arr)]
                        if finite_vals.size > 0:
                            vmax = float(np.max(finite_vals))
                            if map_key == "coherence_map":
                                vmin = 0.0
                            else:
                                vmin = 0.0
                        else:
                            vmin = 0.0
                            vmax = 1.0
                        vmax = max(vmax, 1e-20)
                        if vmax <= vmin:
                            vmax = vmin + 1e-20
                        imshow_kwargs: dict[str, object] = {}
                        if map_key == "incoherence_map":
                            positive_vals = finite_vals[finite_vals > 0.0]
                            if positive_vals.size > 0:
                                log_vmin = float(np.percentile(positive_vals, 5.0))
                                log_vmax = float(np.percentile(positive_vals, 99.9))
                            else:
                                log_vmin = 1e-20
                                log_vmax = 1.0
                            log_vmin = max(log_vmin, 1e-20)
                            log_vmax = max(log_vmax, log_vmin * (1.0 + 1e-12))
                            imshow_kwargs["norm"] = LogNorm(vmin=log_vmin, vmax=log_vmax)
                            cbar_label = "Incoherence value"
                        else:
                            imshow_kwargs["vmin"] = vmin
                            imshow_kwargs["vmax"] = vmax
                            cbar_label = "Coherence value"
                        shared_im = ax.imshow(
                            map_arr,
                            origin="lower",
                            cmap="inferno",
                            extent=extent,
                            **imshow_kwargs,
                        )
                        planet_center = panel["planet_center_lamD"]
                        orbit_radius_lamD = float(panel["orbit_radius_lamD"])
                        roi_centers = panel["roi_centers_lamD"]
                        roi_size = float(panel["resolved_roi_size_lamD"])
                        has_roi_indication_mode = _has_roi_indication_mode(
                            str(panel.get("phase_sweep_mode", "regional"))
                        )
                        ax.annotate(
                            "Planet",
                            xy=(float(planet_center[0]), float(planet_center[1])),
                            xytext=(
                                extent[1] - 1.2 if float(planet_center[0]) >= 0.0 else extent[0] + 1.2,
                                float(planet_center[1]) + 1.2,
                            ),
                            color="white",
                            fontsize=annotation_size,
                            ha="right" if float(planet_center[0]) >= 0.0 else "left",
                            va="bottom",
                            arrowprops=dict(arrowstyle="->", color="white", lw=1.0),
                        )
                        if has_roi_indication_mode:
                            if region_shape_name == "ring":
                                ring_rmin_lamD, ring_rmax_lamD = annulus_radii_from_width(
                                    mid_radius_lamD=orbit_radius_lamD,
                                    width_lamD=roi_size,
                                )
                                ax.add_patch(plt.Circle((0.0, 0.0), float(ring_rmin_lamD), fill=False, edgecolor="lime", linewidth=overlay_linewidth))
                                ax.add_patch(plt.Circle((0.0, 0.0), float(ring_rmax_lamD), fill=False, edgecolor="cyan", linewidth=overlay_linewidth))
                            else:
                                for j, (cx, cy) in enumerate(roi_centers):
                                    edge = "lime" if j == 0 else "cyan"
                                    ax.add_patch(plt.Circle((float(cx), float(cy)), roi_size, fill=False, edgecolor=edge, linewidth=overlay_linewidth))
                        location_line = (
                            f"r={float(panel.get('planet_radius_lamD', panel['orbit_radius_lamD'])):.2f} λ/D, "
                            f"θ={float(panel.get('planet_theta_deg', 0.0)):.1f}°"
                        )
                        if "lyot_reference_percent" in panel:
                            location_line += f", LR={float(panel['lyot_reference_percent']):.1f}%"
                        roi_line = f"ROI {float(panel['requested_roi_size_lamD']):.2f} → {roi_size:.2f} λ/D"
                        snr_line = f"SNR {float(panel['snr']):.3e}" if "snr" in panel else title_prefix
                        ax.set_title(
                            f"{location_line}\n{roi_line}\n{snr_line}",
                            fontsize=title_fontsize,
                            pad=10,
                        )
                        ax.set_xlim(extent[0], extent[1])
                        ax.set_ylim(extent[2], extent[3])
                        ax.set_aspect("equal")
                        ax.set_xlabel("x [λ/D]", fontsize=label_fontsize)
                        ax.set_ylabel("y [λ/D]", fontsize=label_fontsize)
                        ax.tick_params(axis="both", labelsize=tick_fontsize, width=spine_linewidth, length=3.5)
                        for spine in ax.spines.values():
                            spine.set_linewidth(spine_linewidth)
                if shared_im is not None and cbar_label is not None:
                    cbar = fig.colorbar(shared_im, ax=axes_arr.ravel().tolist(), fraction=0.025, pad=0.02)
                    cbar.set_label(cbar_label, fontsize=label_fontsize)
                    cbar.ax.tick_params(labelsize=tick_fontsize)
                pdf.savefig(fig, transparent=poster_figure)
                plt.close(fig)
        saved_paths.append(output_path)
    return saved_paths


def _save_grouped_roi_size_incoherence_pdfs(
    *,
    output_dir: str,
    base_name: str,
    region_shape_name: str,
    location_panels: list[dict[str, object]],
    extent: list[float],
    poster_figure: bool = False,
) -> list[str]:
    return _save_grouped_roi_size_map_pdfs(
        output_dir=output_dir,
        base_name=base_name,
        region_shape_name=region_shape_name,
        location_panels=location_panels,
        extent=extent,
        save_panel_pdf=_save_roi_size_incoherence_pdf_for_planet_location,
        poster_figure=poster_figure,
    )


def _save_grouped_roi_size_coherence_pdfs(
    *,
    output_dir: str,
    base_name: str,
    region_shape_name: str,
    location_panels: list[dict[str, object]],
    extent: list[float],
    poster_figure: bool = False,
) -> list[str]:
    return _save_grouped_roi_size_map_pdfs(
        output_dir=output_dir,
        base_name=base_name,
        region_shape_name=region_shape_name,
        location_panels=location_panels,
        extent=extent,
        save_panel_pdf=_save_roi_size_coherence_pdf_for_planet_location,
        poster_figure=poster_figure,
    )


def _save_grouped_roi_size_lyot_plane_pngs(
    *,
    output_dir: str,
    base_name: str,
    location_panels: list[dict[str, object]],
) -> list[str]:
    groups = _build_planet_position_map_report_groups(location_panels)
    saved_paths: list[str] = []
    if len(groups) == 0:
        return saved_paths

    for group in groups:
        grouped_panels: list[dict[str, object]] = []
        grouped_labels: list[str] = []
        for collection in list(group["panel_collections"]):
            panels_sorted = sorted(collection, key=lambda p: float(p["requested_roi_size_lamD"]))
            for panel in panels_sorted:
                grouped_panels.append(panel)
                grouped_labels.append(f"ROI={float(panel['requested_roi_size_lamD']):.2f}")
        if len(grouped_panels) == 0:
            continue
        output_path = os.path.join(output_dir, f"{base_name}_{group['file_tag']}.png")
        _save_lyot_plane_phase_grid_png(
            output_path=output_path,
            panels=grouped_panels,
            panel_labels=grouped_labels,
            figure_title="Lyot Plane by Phase Modulation",
        )
        saved_paths.append(output_path)
    return saved_paths


def _save_grouped_roi_size_lyot_plane_field_phase_pngs(
    *,
    output_dir: str,
    base_name: str,
    location_panels: list[dict[str, object]],
) -> list[str]:
    groups = _build_planet_position_map_report_groups(location_panels)
    saved_paths: list[str] = []
    if len(groups) == 0:
        return saved_paths

    for group in groups:
        grouped_panels: list[dict[str, object]] = []
        grouped_labels: list[str] = []
        for collection in list(group["panel_collections"]):
            panels_sorted = sorted(collection, key=lambda p: float(p["requested_roi_size_lamD"]))
            for panel in panels_sorted:
                grouped_panels.append(panel)
                grouped_labels.append(f"ROI={float(panel['requested_roi_size_lamD']):.2f}")
        if len(grouped_panels) == 0:
            continue
        output_path = os.path.join(output_dir, f"{base_name}_{group['file_tag']}.png")
        _save_lyot_plane_field_phase_grid_png(
            output_path=output_path,
            panels=grouped_panels,
            panel_labels=grouped_labels,
            figure_title="Lyot Plane Field Phase by Phase Modulation",
        )
        saved_paths.append(output_path)
    return saved_paths


def _save_grouped_roi_size_focal_plane_pngs(
    *,
    output_dir: str,
    base_name: str,
    location_panels: list[dict[str, object]],
) -> list[str]:
    groups = _build_planet_position_map_report_groups(location_panels)
    saved_paths: list[str] = []
    if len(groups) == 0:
        return saved_paths

    for group in groups:
        grouped_panels: list[dict[str, object]] = []
        grouped_labels: list[str] = []
        for collection in list(group["panel_collections"]):
            panels_sorted = sorted(collection, key=lambda p: float(p["requested_roi_size_lamD"]))
            for panel in panels_sorted:
                grouped_panels.append(panel)
                grouped_labels.append(f"ROI={float(panel['requested_roi_size_lamD']):.2f}")
        if len(grouped_panels) == 0:
            continue
        output_path = os.path.join(output_dir, f"{base_name}_{group['file_tag']}.png")
        _save_focal_plane_phase_grid_png(
            output_path=output_path,
            panels=grouped_panels,
            panel_labels=grouped_labels,
            figure_title="Focal Plane by Phase Modulation",
        )
        if os.path.exists(output_path):
            saved_paths.append(output_path)
    return saved_paths


def _save_grouped_roi_size_focal_plane_field_phase_pngs(
    *,
    output_dir: str,
    base_name: str,
    location_panels: list[dict[str, object]],
) -> list[str]:
    groups = _build_planet_position_map_report_groups(location_panels)
    saved_paths: list[str] = []
    if len(groups) == 0:
        return saved_paths

    for group in groups:
        grouped_panels: list[dict[str, object]] = []
        grouped_labels: list[str] = []
        for collection in list(group["panel_collections"]):
            panels_sorted = sorted(collection, key=lambda p: float(p["requested_roi_size_lamD"]))
            for panel in panels_sorted:
                grouped_panels.append(panel)
                grouped_labels.append(f"ROI={float(panel['requested_roi_size_lamD']):.2f}")
        if len(grouped_panels) == 0:
            continue
        output_path = os.path.join(output_dir, f"{base_name}_{group['file_tag']}.png")
        _save_focal_plane_field_phase_grid_png(
            output_path=output_path,
            panels=grouped_panels,
            panel_labels=grouped_labels,
            figure_title="Focal Plane Field Phase by Phase Modulation",
        )
        if os.path.exists(output_path):
            saved_paths.append(output_path)
    return saved_paths


def _save_grouped_roi_size_focal_plane_phase_shift_pngs(
    *,
    output_dir: str,
    base_name: str,
    location_panels: list[dict[str, object]],
) -> list[str]:
    groups = _build_planet_position_map_report_groups(location_panels)
    saved_paths: list[str] = []
    if len(groups) == 0:
        return saved_paths

    for group in groups:
        grouped_panels: list[dict[str, object]] = []
        grouped_labels: list[str] = []
        for collection in list(group["panel_collections"]):
            panels_sorted = sorted(collection, key=lambda p: float(p["requested_roi_size_lamD"]))
            for panel in panels_sorted:
                grouped_panels.append(panel)
                grouped_labels.append(f"ROI={float(panel['requested_roi_size_lamD']):.2f}")
        if len(grouped_panels) == 0:
            continue
        output_path = os.path.join(output_dir, f"{base_name}_{group['file_tag']}.png")
        _save_focal_plane_phase_shift_grid_png(
            output_path=output_path,
            panels=grouped_panels,
            panel_labels=grouped_labels,
            figure_title="Applied Focal-Plane Phase Shift",
        )
        if os.path.exists(output_path):
            saved_paths.append(output_path)
    return saved_paths


def _save_roi_size_max_minus_coherence_pdf_for_planet_location(
    *,
    output_path: str,
    region_shape_name: str,
    panels: list[dict[str, object]],
    extent: list[float],
) -> None:
    _save_roi_size_map_pdf_for_planet_location(
        output_path=output_path,
        region_shape_name=region_shape_name,
        panels=panels,
        extent=extent,
        map_key="max_minus_coherence_map",
        title_prefix="max(Coherence) - Coherence",
        cmap="viridis",
        snr_key="snr",
        snr_label="SNR",
        curve_snr_key="snr",
        curve_snr_label="SNR",
        vmax_percentile=99.0,
    )


def _save_roi_size_map_pdf_for_planet_location(
    *,
    output_path: str,
    region_shape_name: str,
    panels: list[dict[str, object]],
    extent: list[float],
    map_key: str,
    title_prefix: str,
    cmap: str = "viridis",
    snr_key: str | None = None,
    snr_label: str | None = None,
    curve_snr_key: str = "snr",
    curve_snr_label: str = "SNR",
    vmin_percentile: float | None = None,
    vmax_percentile: float | None = 98.0,
    include_mean_map_page: bool = False,
    poster_figure: bool = False,
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm

    if len(panels) == 0:
        return

    panels_sorted = sorted(panels, key=lambda p: float(p["requested_roi_size_lamD"]))
    curve_snr_keys = ("snr",)
    curve_snr_all = np.asarray(
        [
            float(panel[key])
            for panel in panels_sorted
            for key in curve_snr_keys
            if key in panel and np.isfinite(float(panel[key]))
        ],
        dtype=float,
    )
    if curve_snr_all.size > 0:
        curve_snr_ymin = float(np.min(curve_snr_all))
        curve_snr_ymax = float(np.max(curve_snr_all))
        if curve_snr_ymax <= curve_snr_ymin:
            pad = max(abs(curve_snr_ymax) * 0.05, 1e-6)
        else:
            pad = 0.05 * (curve_snr_ymax - curve_snr_ymin)
        curve_snr_ylim = (curve_snr_ymin - pad, curve_snr_ymax + pad)
    else:
        curve_snr_ylim = None
    n_panels = len(panels_sorted)
    ncols = min(4, max(2, int(np.ceil(np.sqrt(n_panels)))))
    nrows_maps = int(np.ceil(float(n_panels) / float(ncols)))
    map_title_size = 18 if poster_figure else 8
    map_subtitle_size = 15 if poster_figure else 8
    axis_label_size = 16 if poster_figure else 10
    tick_label_size = 13 if poster_figure else 10
    annotation_size = 14 if poster_figure else 8
    overlay_linewidth = 2.8 if poster_figure else 1.4
    arrow_linewidth = 2.0 if poster_figure else 1.0
    spine_linewidth = 2.0 if poster_figure else 0.8
    panel_title_pad = 16 if poster_figure else 6
    if poster_figure:
        figure_kwargs = {
            "figsize": (5.8 * ncols, 5.4 * nrows_maps),
            "constrained_layout": False,
        }
    else:
        figure_kwargs = {
            "figsize": (4.4 * ncols, 3.6 * nrows_maps),
            "constrained_layout": True,
        }

    with PdfPages(output_path) as pdf:
        fig = plt.figure(**figure_kwargs)
        if poster_figure:
            gs = fig.add_gridspec(nrows_maps, ncols, wspace=0.06, hspace=0.08)
        else:
            gs = fig.add_gridspec(nrows_maps, ncols)
        first_panel = panels_sorted[0]
        has_roi_indication_mode = _has_roi_indication_mode(
            str(first_panel.get("phase_sweep_mode", "regional"))
        )
        first_ax = None
        poster_colorbar_image = None
        poster_colorbar_label = None

        for idx, panel in enumerate(panels_sorted):
            subplot_kwargs = {}
            if first_ax is not None:
                subplot_kwargs["sharex"] = first_ax
                subplot_kwargs["sharey"] = first_ax
            ax = fig.add_subplot(gs[idx // ncols, idx % ncols], **subplot_kwargs)
            if first_ax is None:
                first_ax = ax
            map_arr = np.asarray(panel[map_key], dtype=float)
            finite_vals = map_arr[np.isfinite(map_arr)]
            if finite_vals.size > 0:
                if vmax_percentile is None:
                    vmax = float(np.max(finite_vals))
                else:
                    vmax = float(np.percentile(finite_vals, float(vmax_percentile)))
                if vmin_percentile is None:
                    vmin = 0.0
                else:
                    vmin = float(np.percentile(finite_vals, float(vmin_percentile)))
            else:
                vmin = 0.0
                vmax = 1.0
            vmax = max(vmax, 1e-20)
            if vmax <= vmin:
                vmax = vmin + 1e-20
            cbar_label = None
            imshow_kwargs: dict[str, object] = {}
            if map_key == "incoherence_map":
                positive_vals = finite_vals[finite_vals > 0.0]
                if positive_vals.size > 0:
                    log_vmin = float(np.percentile(positive_vals, 5.0))
                    log_vmax = float(np.percentile(positive_vals, 99.9))
                else:
                    log_vmin = 1e-20
                    log_vmax = 1.0
                log_vmin = max(log_vmin, 1e-20)
                log_vmax = max(log_vmax, log_vmin * (1.0 + 1e-12))
                imshow_kwargs["norm"] = LogNorm(vmin=log_vmin, vmax=log_vmax)
            if poster_figure:
                cbar_label = f"{title_prefix} value"
                if map_key != "incoherence_map":
                    if vmax_percentile is None:
                        imshow_kwargs["vmax"] = float(np.max(finite_vals)) if finite_vals.size > 0 else 1.0
                    else:
                        imshow_kwargs["vmax"] = (
                            float(np.percentile(finite_vals, float(vmax_percentile)))
                            if finite_vals.size > 0
                            else 1.0
                        )
            im = ax.imshow(
                map_arr,
                origin="lower",
                cmap=cmap,
                extent=extent,
                **(
                    imshow_kwargs
                    if (poster_figure or map_key == "incoherence_map")
                    else {"vmin": vmin, "vmax": vmax}
                ),
            )
            rounded_clip = None
            if poster_figure:
                ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
                rounded_clip = FancyBboxPatch(
                    (0.0, 0.0),
                    1.0,
                    1.0,
                    boxstyle="round,pad=0.0,rounding_size=0.045",
                    transform=ax.transAxes,
                    facecolor="none",
                    edgecolor="none",
                )
                ax.add_patch(rounded_clip)
                im.set_clip_path(rounded_clip)
            planet_center = panel["planet_center_lamD"]
            orbit_radius_lamD = float(panel["orbit_radius_lamD"])
            roi_centers = panel["roi_centers_lamD"]
            roi_size = float(panel["resolved_roi_size_lamD"])
            planet_x = float(planet_center[0])
            planet_y = float(planet_center[1])
            x_text = extent[1] - 2.2 if planet_x >= 0.0 else extent[0] + 2.2
            x_align = "right" if planet_x >= 0.0 else "left"
            ax.annotate(
                "Planet",
                xy=(planet_x, planet_y),
                xytext=(x_text, planet_y + 2.1),
                color="white",
                fontsize=annotation_size,
                ha=x_align,
                va="bottom",
                fontweight="bold" if poster_figure else None,
                bbox=(
                    dict(boxstyle="round,pad=0.18", facecolor="black", edgecolor="none", alpha=0.88)
                    if poster_figure
                    else None
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color="white",
                    lw=1.4 if poster_figure else arrow_linewidth,
                    shrinkB=10 if poster_figure else 6,
                ),
            )
            if has_roi_indication_mode:
                if region_shape_name == "ring":
                    ring_rmin_lamD, ring_rmax_lamD = annulus_radii_from_width(
                        mid_radius_lamD=orbit_radius_lamD,
                        width_lamD=roi_size,
                    )
                    patch_inner = plt.Circle((0.0, 0.0), float(ring_rmin_lamD), fill=False, edgecolor="lime", linewidth=overlay_linewidth)
                    patch_outer = plt.Circle((0.0, 0.0), float(ring_rmax_lamD), fill=False, edgecolor="cyan", linewidth=overlay_linewidth)
                    if rounded_clip is not None:
                        patch_inner.set_clip_path(rounded_clip)
                        patch_outer.set_clip_path(rounded_clip)
                    ax.add_patch(patch_inner)
                    ax.add_patch(patch_outer)
                else:
                    for j, (cx, cy) in enumerate(roi_centers):
                        edge = "lime" if j == 0 else "cyan"
                        patch = plt.Circle((float(cx), float(cy)), roi_size, fill=False, edgecolor=edge, linewidth=overlay_linewidth)
                        if rounded_clip is not None:
                            patch.set_clip_path(rounded_clip)
                        ax.add_patch(patch)
            if not poster_figure:
                title_lines = [
                    f"ROI {float(panel['requested_roi_size_lamD']):.2f} → {roi_size:.2f} λ/D",
                ]
                if snr_key is not None and snr_label is not None and snr_key in panel:
                    title_lines.append(f"{str(snr_label)} = {float(panel[snr_key]):.3e}")
                ax.set_title(
                    "\n".join(title_lines),
                    fontsize=map_title_size,
                    pad=panel_title_pad + 4,
                    fontweight="bold" if poster_figure else None,
                )
            if snr_key is not None and snr_label is not None and snr_key in panel:
                if poster_figure:
                    ax.text(
                        0.03,
                        0.97,
                        f"{str(snr_label)} = {float(panel[snr_key]):.3e}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=map_subtitle_size,
                        fontweight="bold",
                        color="white",
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="black", edgecolor="white", linewidth=1.4, alpha=0.88),
                    )
            if not poster_figure:
                ax.set_xlabel("x [λ/D]", fontsize=axis_label_size)
                ax.set_ylabel("y [λ/D]", fontsize=axis_label_size)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal")
            ax.tick_params(axis="both", labelsize=tick_label_size, width=spine_linewidth, length=6 if poster_figure else 3.5)
            if poster_figure:
                row_idx = idx // ncols
                col_idx = idx % ncols
                if row_idx < nrows_maps - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                if col_idx > 0:
                    ax.tick_params(axis="y", labelleft=False)
            for spine in ax.spines.values():
                spine.set_linewidth(spine_linewidth)
                if poster_figure:
                    spine.set_alpha(0.0)
            if poster_figure:
                poster_colorbar_image = im
                poster_colorbar_label = cbar_label
        for idx in range(n_panels, nrows_maps * ncols):
            subplot_kwargs = {}
            if first_ax is not None:
                subplot_kwargs["sharex"] = first_ax
                subplot_kwargs["sharey"] = first_ax
            ax_unused = fig.add_subplot(gs[idx // ncols, idx % ncols], **subplot_kwargs)
            ax_unused.axis("off")

        if poster_figure and poster_colorbar_image is not None and poster_colorbar_label is not None:
            fig.subplots_adjust(left=0.055, right=0.88, bottom=0.06, top=0.97, wspace=0.06, hspace=0.08)
            fig.supxlabel("x [λ/D]", fontsize=axis_label_size, fontweight="bold")
            fig.supylabel("y [λ/D]", fontsize=axis_label_size, fontweight="bold")
            cbar = fig.colorbar(
                poster_colorbar_image,
                ax=fig.axes,
                fraction=0.028,
                pad=0.025,
            )
            cbar.set_label(poster_colorbar_label, fontsize=axis_label_size)
            cbar.ax.tick_params(labelsize=tick_label_size, width=spine_linewidth, length=5)
            cbar.outline.set_linewidth(spine_linewidth)
            cbar.ax.set_facecolor((0.0, 0.0, 0.0, 0.0))

        pdf.savefig(fig, transparent=poster_figure)
        plt.close(fig)

        if include_mean_map_page:
            mean_stack = np.stack(
                [np.asarray(panel[map_key], dtype=float) for panel in panels_sorted],
                axis=0,
            )
            mean_map = np.mean(mean_stack, axis=0)
            finite_vals = mean_map[np.isfinite(mean_map)]
            if finite_vals.size > 0:
                if vmax_percentile is None:
                    mean_vmax = float(np.nanmax(finite_vals))
                else:
                    mean_vmax = float(np.percentile(finite_vals, float(vmax_percentile)))
                if vmin_percentile is None:
                    mean_vmin = 0.0
                else:
                    mean_vmin = float(np.percentile(finite_vals, float(vmin_percentile)))
            else:
                mean_vmin = 0.0
                mean_vmax = 1.0
            mean_vmax = max(mean_vmax, 1e-20)
            if mean_vmax <= mean_vmin:
                mean_vmax = mean_vmin + 1e-20

            fig_mean, ax_mean = plt.subplots(1, 1, figsize=(8.0, 7.2), constrained_layout=True)
            im_mean = ax_mean.imshow(
                mean_map,
                origin="lower",
                cmap=cmap,
                extent=extent,
                vmin=mean_vmin,
                vmax=mean_vmax,
            )
            fig_mean.colorbar(im_mean, ax=ax_mean, fraction=0.046, pad=0.04)
            first_panel = panels_sorted[0]
            planet_center = first_panel["planet_center_lamD"]
            planet_x = float(planet_center[0])
            planet_y = float(planet_center[1])
            x_text = extent[1] - 0.8 if planet_x >= 0.0 else extent[0] + 0.8
            x_align = "right" if planet_x >= 0.0 else "left"
            ax_mean.annotate(
                "Planet",
                xy=(planet_x, planet_y),
                xytext=(x_text, planet_y + 0.9),
                color="white",
                fontsize=9,
                ha=x_align,
                va="bottom",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.1),
            )
            requested_roi = np.asarray(
                [float(panel["requested_roi_size_lamD"]) for panel in panels_sorted],
                dtype=float,
            )
            title_lines = [
                f"Mean {title_prefix.lower()} map across ROI sizes",
                "ROI {:.2f} to {:.2f} λ/D ({:d} maps)".format(
                    float(np.min(requested_roi)),
                    float(np.max(requested_roi)),
                    int(requested_roi.size),
                ),
            ]
            ax_mean.set_title("\n".join(title_lines), fontsize=10, pad=8)
            ax_mean.set_xlabel("x [λ/D]")
            ax_mean.set_ylabel("y [λ/D]")
            pdf.savefig(fig_mean)
            plt.close(fig_mean)


def _save_roi_size_fft_spectra_pdf_for_planet_location(
    *,
    output_path: str,
    panels: list[dict[str, object]],
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    if len(panels) == 0:
        return

    panels_sorted = sorted(panels, key=lambda p: float(p["requested_roi_size_lamD"]))
    with PdfPages(output_path) as pdf:
        for panel in panels_sorted:
            spec_freq = np.asarray(panel.get("selection_nonnegative_freqs", np.array([], dtype=float)), dtype=float)
            spec_mag = np.asarray(panel.get("selection_nonnegative_mag", np.array([], dtype=float)), dtype=float)
            if spec_freq.size == 0 or spec_mag.size == 0:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2), constrained_layout=True)
            ax.plot(spec_freq, spec_mag, color="tab:orange", lw=2.0, label="Planet aperture")
            ref_spectra = list(panel.get("reference_spectra", []))
            ref_colors = plt.cm.Blues(np.linspace(0.45, 0.85, max(len(ref_spectra), 1)))
            for color, ref_spec in zip(ref_colors, ref_spectra):
                ref_freq = np.asarray(ref_spec.get("nonnegative_freqs", np.array([], dtype=float)), dtype=float)
                ref_mag = np.asarray(ref_spec.get("nonnegative_mag", np.array([], dtype=float)), dtype=float)
                if ref_freq.size == 0 or ref_mag.size == 0:
                    continue
                center = ref_spec.get("center_lamD", (float("nan"), float("nan")))
                ax.plot(
                    ref_freq,
                    ref_mag,
                    color=color,
                    lw=1.2,
                    alpha=0.95,
                    label=(
                        f"{ref_spec.get('label', 'Speckle')} "
                        f"({float(center[0]):+.2f}, {float(center[1]):+.2f})"
                    ),
                )
            ax.axvline(float(panel["selected_target_freq"]), color="crimson", lw=1.3, ls="--")
            ax.set_xlabel("frequency [cycles/rad]")
            ax.set_ylabel("|FFT|")
            ax.grid(alpha=0.3)
            ax.set_title(
                "Planet-Aperture FFT Spectrum | ROI {:.2f} -> {:.2f} | f={:.4f}".format(
                    float(panel["requested_roi_size_lamD"]),
                    float(panel["resolved_roi_size_lamD"]),
                    float(panel["selected_target_freq"]),
                )
            )
            ax.legend(fontsize=8, loc="best")
            pdf.savefig(fig)
            plt.close(fig)


def _save_planet_locations_on_mean_final_psf(
    *,
    output_path: str,
    sim_local: dict,
    planet_centers_lamD: list[tuple[float, float]],
    crop_lamD: float = 12.0,
) -> None:
    if len(planet_centers_lamD) == 0:
        return

    psf_stack: list[np.ndarray] = []
    n_fft: int | None = None
    samp: float | None = None
    for ctr in planet_centers_lamD:
        result = CoronagraphSimulator(
            **{
                **sim_local,
                "companion_offset_lamD": (float(ctr[0]), float(ctr[1])),
                "e_final_phase_offset": 0.0,
                "focal_local_phase_offset": 0.0,
            }
        ).run()
        psf_stack.append(np.asarray(result["final_psf_with_ghost"], dtype=float))
        if n_fft is None:
            n_fft = int(result["n_fft"])
            samp = float(result["focal_sampling"])

    if n_fft is None or samp is None or len(psf_stack) == 0:
        return

    mean_psf = np.mean(np.stack(psf_stack, axis=0), axis=0)
    half = int(float(crop_lamD) * float(samp))
    cc = int(n_fft // 2)
    sl = slice(cc - half, cc + half)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 7.0), constrained_layout=True)
    im = ax.imshow(
        np.log10(mean_psf[sl, sl] + 1e-12),
        origin="lower",
        cmap="inferno",
        vmin=-8,
        vmax=0,
        extent=[-float(crop_lamD), float(crop_lamD), -float(crop_lamD), float(crop_lamD)],
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(planet_centers_lamD), 1), endpoint=False))
    for idx, ctr in enumerate(planet_centers_lamD):
        color = palette[idx % len(palette)]
        ax.plot([float(ctr[0])], [float(ctr[1])], marker="o", markersize=6.5, color=color, linestyle="None")
        ax.text(
            float(ctr[0]),
            float(ctr[1]),
            str(idx + 1),
            color="white",
            fontsize=9,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="black", alpha=0.45, edgecolor="none"),
        )
    ax.set_title("Planet Locations on Mean Final PSF")
    ax.set_xlabel("x [λ/D]")
    ax.set_ylabel("y [λ/D]")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_planet_position_snr_summary_pdf(
    *,
    output_path: str,
    location_panels: list[dict[str, object]],
) -> None:
    from matplotlib.backends.backend_pdf import PdfPages

    if len(location_panels) == 0:
        return

    per_page = 6
    ncols = 2
    nrows = 3
    summary_snr_keys = ("snr",)
    summary_snr_all = np.asarray(
        [
            float(panel[key])
            for item in location_panels
            for panel in item["panels"]
            for key in summary_snr_keys
            if key in panel and np.isfinite(float(panel[key]))
        ],
        dtype=float,
    )
    if summary_snr_all.size > 0:
        summary_ymin = float(np.min(summary_snr_all))
        summary_ymax = float(np.max(summary_snr_all))
        if summary_ymax <= summary_ymin:
            summary_pad = max(abs(summary_ymax) * 0.05, 1e-6)
        else:
            summary_pad = 0.05 * (summary_ymax - summary_ymin)
        summary_ylim = (summary_ymin - summary_pad, summary_ymax + summary_pad)
    else:
        summary_ylim = None
    with PdfPages(output_path) as pdf:
        for start in range(0, len(location_panels), per_page):
            chunk = location_panels[start:start + per_page]
            fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 13.0), constrained_layout=True)
            axes_flat = np.asarray(axes).ravel()
            for ax, item in zip(axes_flat, chunk):
                panels_sorted = sorted(
                    list(item["panels"]),
                    key=lambda p: float(p["requested_roi_size_lamD"]),
                )
                requested_roi = np.asarray(
                    [float(panel["requested_roi_size_lamD"]) for panel in panels_sorted],
                    dtype=float,
                )
                resolved_roi = np.asarray(
                    [float(panel["resolved_roi_size_lamD"]) for panel in panels_sorted],
                    dtype=float,
                )
                vals = np.asarray([float(panel["snr"]) for panel in panels_sorted], dtype=float)
                ax.plot(requested_roi, vals, "-o", lw=1.6, ms=4.0, color="tab:blue", label="SNR")
                ax.set_xlabel("requested ROI size [λ/D]")
                ax.set_ylabel("SNR")
                ax.grid(alpha=0.3)
                if summary_ylim is not None:
                    ax.set_ylim(*summary_ylim)
                title = (
                    f"r={float(item['planet_radius_lamD']):.3f} λ/D, "
                    f"theta={float(item['planet_theta_deg']):.1f} deg"
                    + (
                        f", LR={float(item['lyot_reference_percent']):.1f}%"
                        if "lyot_reference_percent" in item
                        else ""
                    )
                    + "\n"
                    +
                    f"xy=({float(item['planet_center_lamD'][0]):+.3f}, {float(item['planet_center_lamD'][1]):+.3f})"
                )
                ax.set_title(title)
                ax.legend(fontsize=7, loc="best")
                if np.any(np.abs(resolved_roi - requested_roi) > 1e-12):
                    ax2 = ax.twinx()
                    ax2.plot(requested_roi, resolved_roi, "--s", lw=1.2, ms=3.5, color="tab:orange")
                    ax2.set_ylabel("resolved ROI [λ/D]", color="tab:orange")
                    ax2.tick_params(axis="y", labelcolor="tab:orange")
            for ax in axes_flat[len(chunk):]:
                ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

        radius_groups: dict[float, list[dict[str, object]]] = {}
        for item in location_panels:
            radius_key = float(item["planet_radius_lamD"])
            radius_groups.setdefault(radius_key, []).append(item)

        grouped_radii = sorted(radius_groups.keys())
        if len(grouped_radii) > 0:
            for start in range(0, len(grouped_radii), per_page):
                radius_chunk = grouped_radii[start:start + per_page]
                fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 13.0), constrained_layout=True)
                axes_flat = np.asarray(axes).ravel()
                for ax, radius_key in zip(axes_flat, radius_chunk):
                    items = radius_groups[radius_key]
                    roi_to_snrs: dict[float, list[float]] = {}
                    roi_to_resolved: dict[float, list[float]] = {}
                    for item in items:
                        for panel in item["panels"]:
                            if "snr" not in panel:
                                continue
                            requested_roi = float(panel["requested_roi_size_lamD"])
                            roi_to_snrs.setdefault(requested_roi, []).append(float(panel["snr"]))
                            roi_to_resolved.setdefault(requested_roi, []).append(float(panel["resolved_roi_size_lamD"]))
                    requested_roi = np.asarray(sorted(roi_to_snrs.keys()), dtype=float)
                    mean_snr = np.asarray(
                        [float(np.mean(np.asarray(roi_to_snrs[roi], dtype=float))) for roi in requested_roi],
                        dtype=float,
                    )
                    mean_resolved = np.asarray(
                        [float(np.mean(np.asarray(roi_to_resolved[roi], dtype=float))) for roi in requested_roi],
                        dtype=float,
                    )
                    ax.plot(
                        requested_roi,
                        mean_snr,
                        "-o",
                        lw=1.8,
                        ms=4.4,
                        color="tab:purple",
                        label="Mean SNR over theta",
                    )
                    ax.set_xlabel("requested ROI size [λ/D]")
                    ax.set_ylabel("mean SNR over theta")
                    ax.grid(alpha=0.3)
                    if summary_ylim is not None:
                        ax.set_ylim(*summary_ylim)
                    theta_vals = sorted({float(item["planet_theta_deg"]) for item in items})
                    ax.set_title(
                        f"r={radius_key:.3f} λ/D\n"
                        f"mean over theta ({len(theta_vals)} samples)"
                    )
                    ax.legend(fontsize=7, loc="best")
                    if np.any(np.abs(mean_resolved - requested_roi) > 1e-12):
                        ax2 = ax.twinx()
                        ax2.plot(requested_roi, mean_resolved, "--s", lw=1.2, ms=3.5, color="tab:orange")
                        ax2.set_ylabel("mean resolved ROI [λ/D]", color="tab:orange")
                        ax2.tick_params(axis="y", labelcolor="tab:orange")
                for ax in axes_flat[len(radius_chunk):]:
                    ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)


def _save_map_panel_summary_png(
    *,
    output_path: str,
    panels: list[dict[str, object]],
    panel_labels: list[str],
    figure_title: str,
) -> None:
    from matplotlib.colors import LogNorm

    if len(panels) == 0 or len(panels) != len(panel_labels):
        return

    n_cases = len(panels)
    cases_per_row = min(3, max(1, n_cases))
    nrows_per_section = int(np.ceil(float(n_cases) / float(cases_per_row)))
    ncols = cases_per_row
    total_rows = 2 * nrows_per_section

    incoh_blocks = []
    coh_blocks = []
    for panel in panels:
        incoh = np.asarray(panel["incoherence_map"], dtype=float)
        coh = np.asarray(panel["coherence_map"], dtype=float)
        incoh_blocks.append(incoh[np.isfinite(incoh)].ravel())
        coh_blocks.append(coh[np.isfinite(coh)].ravel())
    incoh_vals = np.concatenate(incoh_blocks) if incoh_blocks else np.array([], dtype=float)
    coh_vals = np.concatenate(coh_blocks) if coh_blocks else np.array([], dtype=float)

    incoh_pos = incoh_vals[incoh_vals > 0.0]
    if incoh_pos.size > 0:
        incoh_vmin = max(float(np.percentile(incoh_pos, 5.0)), 1e-20)
        incoh_vmax = max(float(np.percentile(incoh_pos, 99.7)), incoh_vmin * (1.0 + 1e-12))
    else:
        incoh_vmin, incoh_vmax = 1e-20, 1.0
    if coh_vals.size > 0:
        coh_vmin = float(np.min(coh_vals))
        coh_vmax = float(np.percentile(coh_vals, 99.7))
        if coh_vmax <= coh_vmin:
            coh_vmax = coh_vmin + 1e-12
    else:
        coh_vmin, coh_vmax = 0.0, 1.0

    fig, axes = plt.subplots(
        total_rows,
        ncols,
        figsize=(4.6 * ncols, 4.0 * total_rows),
        constrained_layout=True,
    )
    axes_arr = np.atleast_2d(axes)
    incoh_artist = None
    coh_artist = None
    extent = [-12.0, 12.0, -12.0, 12.0]

    for idx, (panel, label) in enumerate(zip(panels, panel_labels)):
        row = idx // cases_per_row
        col = idx % cases_per_row
        ax_incoh = axes_arr[row, col]
        ax_coh = axes_arr[nrows_per_section + row, col]
        incoh = np.asarray(panel["incoherence_map"], dtype=float)
        coh = np.asarray(panel["coherence_map"], dtype=float)
        planet_x, planet_y = panel.get("planet_center_lamD", (0.0, 0.0))
        snr = float(panel.get("snr", float("nan")))

        incoh_artist = ax_incoh.imshow(
            incoh,
            origin="lower",
            cmap="inferno",
            extent=extent,
            norm=LogNorm(vmin=incoh_vmin, vmax=incoh_vmax),
        )
        coh_artist = ax_coh.imshow(
            coh,
            origin="lower",
            cmap="inferno",
            extent=extent,
            vmin=coh_vmin,
            vmax=coh_vmax,
        )
        for ax, title in ((ax_incoh, "Incoherence"), (ax_coh, "Coherence")):
            x_text = extent[1] - 2.4 if float(planet_x) >= 0.0 else extent[0] + 2.4
            y_text = float(planet_y) + 1.8
            ha = "right" if float(planet_x) >= 0.0 else "left"
            ax.annotate(
                "planet",
                xy=(float(planet_x), float(planet_y)),
                xytext=(x_text, y_text),
                color="white",
                fontsize=9,
                ha=ha,
                va="center",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.4),
                bbox=dict(boxstyle="round,pad=0.18", facecolor="black", alpha=0.45, edgecolor="none"),
            )
            ax.set_title(f"{title}\n{label}\nSNR={snr:.3e}", fontsize=10)
            ax.set_xlabel("x [λ/D]")
            ax.set_ylabel("y [λ/D]")

    for row in range(nrows_per_section):
        for col in range(cases_per_row):
            idx = row * cases_per_row + col
            if idx >= n_cases:
                axes_arr[row, col].axis("off")
                axes_arr[nrows_per_section + row, col].axis("off")

    if incoh_artist is not None:
        fig.colorbar(incoh_artist, ax=axes_arr[:nrows_per_section, :], fraction=0.018, pad=0.02)
    if coh_artist is not None:
        fig.colorbar(coh_artist, ax=axes_arr[nrows_per_section:, :], fraction=0.018, pad=0.02)
    if nrows_per_section > 0:
        axes_arr[0, 0].text(
            0.0,
            1.18,
            "Incoherence Maps",
            transform=axes_arr[0, 0].transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        axes_arr[nrows_per_section, 0].text(
            0.0,
            1.18,
            "Coherence Maps",
            transform=axes_arr[nrows_per_section, 0].transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    fig.suptitle(figure_title, fontsize=15)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_lyot_plane_phase_grid_png(
    *,
    output_path: str,
    panels: list[dict[str, object]],
    panel_labels: list[str],
    figure_title: str,
) -> None:
    from matplotlib.colors import LogNorm

    if len(panels) == 0 or len(panels) != len(panel_labels):
        return

    lyot_stacks: list[np.ndarray] = []
    phase_vectors: list[np.ndarray] = []
    positive_values: list[np.ndarray] = []
    max_cols = 0
    for panel in panels:
        stack = np.asarray(panel.get("lyot_intensity_stack", np.zeros((0, 0, 0))), dtype=float)
        phase_vec = np.asarray(panel.get("lyot_phase_offsets_rad", np.array([], dtype=float)), dtype=float)
        if stack.ndim != 3 or stack.shape[0] == 0 or phase_vec.size != stack.shape[0]:
            return
        lyot_stacks.append(stack)
        phase_vectors.append(phase_vec)
        max_cols = max(max_cols, int(stack.shape[0]))
        pos = stack[np.isfinite(stack) & (stack > 0.0)]
        if pos.size > 0:
            positive_values.append(pos)

    if max_cols == 0:
        return

    if positive_values:
        all_pos = np.concatenate(positive_values)
        vmin = max(float(np.percentile(all_pos, 5.0)), 1e-20)
        vmax = max(float(np.percentile(all_pos, 99.7)), vmin * (1.0 + 1e-12))
        shared_norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        shared_norm = None

    nrows = len(panels)
    fig, axes = plt.subplots(
        nrows,
        max_cols,
        figsize=(2.2 * max_cols, 2.35 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_arr = np.asarray(axes, dtype=object)
    shared_artist = None

    for row_idx, (panel, label, stack, phase_vec) in enumerate(
        zip(panels, panel_labels, lyot_stacks, phase_vectors)
    ):
        sweep_mode = str(panel.get("phase_sweep_mode", "regional")).strip().lower()
        for col_idx in range(max_cols):
            ax = axes_arr[row_idx, col_idx]
            if col_idx >= stack.shape[0]:
                ax.axis("off")
                continue
            image = np.asarray(stack[col_idx], dtype=float)
            if shared_norm is not None:
                shared_artist = ax.imshow(image, origin="lower", cmap="inferno", norm=shared_norm)
            else:
                vmax = float(np.max(image)) if np.isfinite(np.max(image)) else 1.0
                vmax = vmax if vmax > 0.0 else 1.0
                shared_artist = ax.imshow(
                    image,
                    origin="lower",
                    cmap="inferno",
                    vmin=0.0,
                    vmax=vmax,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            phase_value = float(phase_vec[col_idx])
            if sweep_mode == "mask_rotation":
                title = f"rot={phase_value / np.pi:.2f}pi"
            else:
                title = f"phi={phase_value / np.pi:.2f}pi"
            ax.set_title(title, fontsize=8, pad=4)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9)

    if shared_artist is not None:
        cbar = fig.colorbar(
            shared_artist,
            ax=axes_arr.ravel().tolist(),
            fraction=0.03,
            pad=0.02,
            shrink=0.96,
        )
        cbar.set_label("Lyot intensity", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(figure_title, fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_lyot_plane_field_phase_grid_png(
    *,
    output_path: str,
    panels: list[dict[str, object]],
    panel_labels: list[str],
    figure_title: str,
) -> None:
    if len(panels) == 0 or len(panels) != len(panel_labels):
        return

    lyot_stacks: list[np.ndarray] = []
    phase_vectors: list[np.ndarray] = []
    max_cols = 0
    for panel in panels:
        stack = np.asarray(panel.get("lyot_phase_stack", np.zeros((0, 0, 0))), dtype=float)
        phase_vec = np.asarray(panel.get("lyot_phase_offsets_rad", np.array([], dtype=float)), dtype=float)
        if stack.ndim != 3 or stack.shape[0] == 0 or phase_vec.size != stack.shape[0]:
            return
        lyot_stacks.append(stack)
        phase_vectors.append(phase_vec)
        max_cols = max(max_cols, int(stack.shape[0]))

    if max_cols == 0:
        return

    nrows = len(panels)
    fig, axes = plt.subplots(
        nrows,
        max_cols,
        figsize=(2.2 * max_cols, 2.35 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_arr = np.asarray(axes, dtype=object)
    shared_artist = None

    for row_idx, (panel, label, stack, phase_vec) in enumerate(
        zip(panels, panel_labels, lyot_stacks, phase_vectors)
    ):
        sweep_mode = str(panel.get("phase_sweep_mode", "regional")).strip().lower()
        for col_idx in range(max_cols):
            ax = axes_arr[row_idx, col_idx]
            if col_idx >= stack.shape[0]:
                ax.axis("off")
                continue
            image = np.asarray(stack[col_idx], dtype=float)
            shared_artist = ax.imshow(
                image,
                origin="lower",
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            phase_value = float(phase_vec[col_idx])
            if sweep_mode == "mask_rotation":
                title = f"rot={phase_value / np.pi:.2f}pi"
            else:
                title = f"phi={phase_value / np.pi:.2f}pi"
            ax.set_title(title, fontsize=8, pad=4)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9)

    if shared_artist is not None:
        cbar = fig.colorbar(
            shared_artist,
            ax=axes_arr.ravel().tolist(),
            fraction=0.03,
            pad=0.02,
            shrink=0.96,
        )
        cbar.set_label("Lyot phase [rad]", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(figure_title, fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_focal_plane_phase_grid_png(
    *,
    output_path: str,
    panels: list[dict[str, object]],
    panel_labels: list[str],
    figure_title: str,
) -> None:
    from matplotlib.colors import LogNorm

    if len(panels) == 0 or len(panels) != len(panel_labels):
        return

    focal_stacks: list[np.ndarray] = []
    phase_vectors: list[np.ndarray] = []
    positive_values: list[np.ndarray] = []
    max_cols = 0
    for panel in panels:
        stack = np.asarray(panel.get("focal_plane_intensity_stack", np.zeros((0, 0, 0))), dtype=float)
        phase_vec = np.asarray(panel.get("focal_plane_phase_offsets_rad", np.array([], dtype=float)), dtype=float)
        if stack.ndim != 3 or stack.shape[0] == 0 or phase_vec.size != stack.shape[0]:
            return
        focal_stacks.append(stack)
        phase_vectors.append(phase_vec)
        max_cols = max(max_cols, int(stack.shape[0]))
        pos = stack[np.isfinite(stack) & (stack > 0.0)]
        if pos.size > 0:
            positive_values.append(pos)

    if max_cols == 0:
        return

    if positive_values:
        all_pos = np.concatenate(positive_values)
        vmin = max(float(np.percentile(all_pos, 5.0)), 1e-20)
        vmax = max(float(np.percentile(all_pos, 99.7)), vmin * (1.0 + 1e-12))
        shared_norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        shared_norm = None

    nrows = len(panels)
    fig, axes = plt.subplots(
        nrows,
        max_cols,
        figsize=(2.2 * max_cols, 2.35 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_arr = np.asarray(axes, dtype=object)
    shared_artist = None

    for row_idx, (panel, label, stack, phase_vec) in enumerate(
        zip(panels, panel_labels, focal_stacks, phase_vectors)
    ):
        sweep_mode = str(panel.get("phase_sweep_mode", "regional")).strip().lower()
        for col_idx in range(max_cols):
            ax = axes_arr[row_idx, col_idx]
            if col_idx >= stack.shape[0]:
                ax.axis("off")
                continue
            image = np.asarray(stack[col_idx], dtype=float)
            if shared_norm is not None:
                shared_artist = ax.imshow(image, origin="lower", cmap="inferno", norm=shared_norm)
            else:
                vmax = float(np.max(image)) if np.isfinite(np.max(image)) else 1.0
                vmax = vmax if vmax > 0.0 else 1.0
                shared_artist = ax.imshow(
                    image,
                    origin="lower",
                    cmap="inferno",
                    vmin=0.0,
                    vmax=vmax,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            phase_value = float(phase_vec[col_idx])
            if sweep_mode == "mask_rotation":
                title = f"rot={phase_value / np.pi:.2f}pi"
            else:
                title = f"phi={phase_value / np.pi:.2f}pi"
            ax.set_title(title, fontsize=8, pad=4)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9)

    if shared_artist is not None:
        cbar = fig.colorbar(
            shared_artist,
            ax=axes_arr.ravel().tolist(),
            fraction=0.03,
            pad=0.02,
            shrink=0.96,
        )
        cbar.set_label("Focal-plane intensity", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(figure_title, fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_focal_plane_field_phase_grid_png(
    *,
    output_path: str,
    panels: list[dict[str, object]],
    panel_labels: list[str],
    figure_title: str,
) -> None:
    if len(panels) == 0 or len(panels) != len(panel_labels):
        return

    focal_phase_stacks: list[np.ndarray] = []
    phase_vectors: list[np.ndarray] = []
    max_cols = 0
    for panel in panels:
        stack = np.asarray(panel.get("focal_plane_field_phase_stack", np.zeros((0, 0, 0))), dtype=float)
        phase_vec = np.asarray(panel.get("focal_plane_phase_offsets_rad", np.array([], dtype=float)), dtype=float)
        if stack.ndim != 3 or stack.shape[0] == 0 or phase_vec.size != stack.shape[0]:
            return
        focal_phase_stacks.append(stack)
        phase_vectors.append(phase_vec)
        max_cols = max(max_cols, int(stack.shape[0]))

    if max_cols == 0:
        return

    nrows = len(panels)
    fig, axes = plt.subplots(
        nrows,
        max_cols,
        figsize=(2.2 * max_cols, 2.35 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_arr = np.asarray(axes, dtype=object)
    shared_artist = None

    for row_idx, (panel, label, stack, phase_vec) in enumerate(
        zip(panels, panel_labels, focal_phase_stacks, phase_vectors)
    ):
        sweep_mode = str(panel.get("phase_sweep_mode", "regional")).strip().lower()
        for col_idx in range(max_cols):
            ax = axes_arr[row_idx, col_idx]
            if col_idx >= stack.shape[0]:
                ax.axis("off")
                continue
            image = np.asarray(stack[col_idx], dtype=float)
            shared_artist = ax.imshow(
                image,
                origin="lower",
                cmap="twilight",
                vmin=-np.pi,
                vmax=np.pi,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            phase_value = float(phase_vec[col_idx])
            if sweep_mode == "mask_rotation":
                title = f"rot={phase_value / np.pi:.2f}pi"
            else:
                title = f"phi={phase_value / np.pi:.2f}pi"
            ax.set_title(title, fontsize=8, pad=4)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9)

    if shared_artist is not None:
        cbar = fig.colorbar(
            shared_artist,
            ax=axes_arr.ravel().tolist(),
            fraction=0.03,
            pad=0.02,
            shrink=0.96,
        )
        cbar.set_label("Focal-plane field phase [rad]", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(figure_title, fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_focal_plane_phase_shift_grid_png(
    *,
    output_path: str,
    panels: list[dict[str, object]],
    panel_labels: list[str],
    figure_title: str,
) -> None:
    if len(panels) == 0 or len(panels) != len(panel_labels):
        return

    phase_stacks: list[np.ndarray] = []
    phase_vectors: list[np.ndarray] = []
    max_cols = 0
    for panel in panels:
        stack = np.asarray(panel.get("focal_plane_phase_shift_stack", np.zeros((0, 0, 0))), dtype=float)
        phase_vec = np.asarray(panel.get("focal_plane_phase_offsets_rad", np.array([], dtype=float)), dtype=float)
        if stack.ndim != 3 or stack.shape[0] == 0 or phase_vec.size != stack.shape[0]:
            return
        phase_stacks.append(stack)
        phase_vectors.append(phase_vec)
        max_cols = max(max_cols, int(stack.shape[0]))

    if max_cols == 0:
        return

    nrows = len(panels)
    fig, axes = plt.subplots(
        nrows,
        max_cols,
        figsize=(2.2 * max_cols, 2.35 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes_arr = np.asarray(axes, dtype=object)
    shared_artist = None

    for row_idx, (panel, label, stack, phase_vec) in enumerate(
        zip(panels, panel_labels, phase_stacks, phase_vectors)
    ):
        sweep_mode = str(panel.get("phase_sweep_mode", "regional")).strip().lower()
        for col_idx in range(max_cols):
            ax = axes_arr[row_idx, col_idx]
            if col_idx >= stack.shape[0]:
                ax.axis("off")
                continue
            image = np.mod(np.asarray(stack[col_idx], dtype=float), 2.0 * np.pi)
            shared_artist = ax.imshow(
                image,
                origin="lower",
                cmap="twilight",
                vmin=0.0,
                vmax=2.0 * np.pi,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            phase_value = float(phase_vec[col_idx])
            if sweep_mode == "mask_rotation":
                title = f"rot={phase_value / np.pi:.2f}pi"
            else:
                title = f"phi={phase_value / np.pi:.2f}pi"
            ax.set_title(title, fontsize=8, pad=4)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9)

    if shared_artist is not None:
        cbar = fig.colorbar(
            shared_artist,
            ax=axes_arr.ravel().tolist(),
            fraction=0.03,
            pad=0.02,
            shrink=0.96,
        )
        cbar.set_label("Applied phase shift [rad], wrapped", fontsize=11)
        cbar.set_ticks([0.0, np.pi, 2.0 * np.pi])
        cbar.set_ticklabels(["0", "pi", "2pi"])
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle(figure_title, fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_ring_rotation_probe_fft_page(
    pdf,
    img_last: np.ndarray,
    stack: np.ndarray,
    phase_series: np.ndarray,
    probes: list[dict[str, float | int]],
    rotation_fraction: float,
    applied_rotation_rad: float,
    region_centers: list[tuple[float, float]],
    region_radius_lamD: float,
    planet_center_lamD: tuple[float, float],
    sl_crop: slice,
    psf_crop_lamD: float,
    n_fft: int,
    focal_sampling: float,
) -> None:
    if img_last is None or stack.size == 0 or phase_series.size < 2 or len(probes) == 0:
        return

    dphi = float(np.mean(np.diff(phase_series)))
    freq = np.fft.fftfreq(int(phase_series.size), d=dphi)
    pos = freq >= 0.0
    crop_center = int(n_fft // 2)
    crop_half = int((sl_crop.stop - sl_crop.start) // 2)
    c_full = (float(n_fft) - 1.0) / 2.0

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), constrained_layout=True)
    ax_ts, ax_fft, ax_img = axes
    palette = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(probes), 1), endpoint=False))
    fft_xticks: list[float] = []

    for j, probe in enumerate(probes):
        color = palette[j % len(palette)]
        x_idx = int(probe["x_idx"])
        y_idx = int(probe["y_idx"])
        x_local = x_idx - (crop_center - crop_half)
        y_local = y_idx - (crop_center - crop_half)
        if y_local < 0 or y_local >= stack.shape[1] or x_local < 0 or x_local >= stack.shape[2]:
            continue
        trace = stack[:, y_local, x_local]
        fft_trace = np.fft.fft(trace)
        amp = np.abs(fft_trace) / max(trace.size, 1)
        label = "planet center" if j == 0 else f"probe {j}"
        freq_pos = freq[pos]
        amp_pos = amp[pos]
        fft_xticks = [float(v) for v in freq_pos]
        ax_ts.plot(
            phase_series,
            trace,
            lw=2.0 if j == 0 else 1.2,
            alpha=1.0 if j == 0 else 0.85,
            color=color,
            label=label,
        )
        ax_fft.plot(
            freq_pos,
            amp_pos,
            lw=2.0 if j == 0 else 1.2,
            alpha=1.0 if j == 0 else 0.85,
            color=color,
            marker="o",
            markersize=3.5 if j == 0 else 2.5,
            label=label,
        )
        x_lamD = (float(x_idx) - c_full) / float(focal_sampling)
        y_lamD = (float(y_idx) - c_full) / float(focal_sampling)
        ax_img.plot(
            [x_lamD],
            [y_lamD],
            marker="o",
            markersize=4.0 if j == 0 else 3.0,
            color=color,
            linestyle="None",
        )
        ax_img.text(x_lamD, y_lamD, str(j), color="white", fontsize=10, ha="left", va="bottom")

    phase_mid = 0.5 * (float(phase_series[0]) + float(phase_series[-1]))
    ax_ts.set_title("Fixed Probe-Pixel Time Series")
    ax_ts.set_xlabel("Local phase shift [rad]")
    ax_ts.set_ylabel("Intensity")
    ax_ts.set_xticks([float(phase_series[0]), phase_mid, float(phase_series[-1])])
    ax_ts.set_xticklabels([f"{phase_series[0]/np.pi:.1f}π", f"{phase_mid/np.pi:.1f}π", f"{phase_series[-1]/np.pi:.1f}π"])
    ax_ts.grid(alpha=0.3)
    ax_ts.legend(fontsize=10, ncol=1, loc="best")

    ax_fft.set_title("Fixed Probe-Pixel FFT Magnitude")
    ax_fft.set_xlabel("Frequency [cycles/rad]")
    ax_fft.set_ylabel("Amplitude")
    if fft_xticks:
        ax_fft.set_xticks(fft_xticks)
        ax_fft.set_xticklabels([f"{tick:.3f}" for tick in fft_xticks], rotation=90, fontsize=6)
    ax_fft.grid(alpha=0.3)
    ax_fft.legend(fontsize=10, ncol=1, loc="best")

    im = ax_img.imshow(
        np.log10(img_last[sl_crop, sl_crop] + 1e-12),
        origin="lower",
        cmap="inferno",
        vmin=-8,
        vmax=0,
        extent=[-psf_crop_lamD, psf_crop_lamD, -psf_crop_lamD, psf_crop_lamD],
    )
    for j, (cx, cy) in enumerate(region_centers):
        edge = "lime" if j == 0 else "cyan"
        ax_img.add_patch(plt.Circle((cx, cy), region_radius_lamD, fill=False, edgecolor=edge, linewidth=1.2))
    ax_img.plot([planet_center_lamD[0]], [planet_center_lamD[1]], marker="+", color="white", markersize=9, linestyle="None")
    ax_img.set_title(
        f"Probe Pixels on Rotated Ring\nu={rotation_fraction:.2f}, rot={applied_rotation_rad:.3f} rad"
    )
    ax_img.set_xlabel("x [λ/D]")
    ax_img.set_ylabel("y [λ/D]")
    fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
    pdf.savefig(fig)
    plt.close(fig)


def _save_ring_of_circle_rotation_gif(
    gif_path: str,
    requested_region_radius_lamD: float,
    orbit_radius_lamD: float,
    anchor_angle_rad: float,
    fixed_center_lamD: tuple[float, float],
    resolved_region_radius_lamD: float,
    n_circles: int,
    n_frames: int = 21,
) -> None:
    frames: list[np.ndarray] = []
    max_extent = max(orbit_radius_lamD + 2.5 * resolved_region_radius_lamD, orbit_radius_lamD + 1.5, 2.0)
    u_values = np.linspace(0.0, 1.0, max(2, int(n_frames)))
    for u in u_values:
        ring = build_touching_circle_ring(
            requested_region_radius_lamD=requested_region_radius_lamD,
            orbit_radius_lamD=orbit_radius_lamD,
            anchor_angle_rad=anchor_angle_rad,
            rotation_fraction=float(u),
            min_circles=int(n_circles),
        )
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.2), constrained_layout=True)
        ax.set_aspect("equal")
        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        ax.grid(alpha=0.2)
        ax.add_patch(plt.Circle((0.0, 0.0), orbit_radius_lamD, fill=False, edgecolor="0.65", linestyle="--", linewidth=1.2))
        ax.plot(0.0, 0.0, marker="+", color="black", markersize=8, markeredgewidth=1.5)
        ax.plot(fixed_center_lamD[0], fixed_center_lamD[1], marker="o", color="tab:red", markersize=6)
        for idx, (cx, cy) in enumerate(ring["centers_lamD"]):
            edge_color = "tab:green" if idx == 0 else "tab:blue"
            line_width = 2.0 if idx == 0 else 1.4
            ax.add_patch(
                plt.Circle((cx, cy), ring["resolved_radius_lamD"], fill=False, edgecolor=edge_color, linewidth=line_width)
            )
            ax.plot(cx, cy, marker=".", color=edge_color, markersize=5)
        ax.set_title("ring_of_circle rotation")
        ax.set_xlabel("x [λ/D]")
        ax.set_ylabel("y [λ/D]")
        ax.text(
            0.02,
            0.98,
            (
                f"u={u:.2f}\n"
                f"rotation={ring['applied_rotation_rad']:.4f} rad\n"
                f"edge target={ring['edge_cut_rotation_rad']:.4f} rad\n"
                f"r={ring['resolved_radius_lamD']:.4f} λ/D, N={ring['n_circles']}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)

    _write_rgb_gif(np.asarray(frames, dtype=np.uint8), gif_path, duration_ms=250)
