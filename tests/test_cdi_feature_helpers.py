from __future__ import annotations

import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from coronagraph.cdi_feature import (
    _evaluate_best_roi_for_planet_center,
    _inclusive_float_range,
    _planet_region_snr,
    _polar_to_cartesian_lamD,
    _run_planet_position_roi_size_sweep,
    _roi_shape_folder_name,
    _summarize_roi_snr_trend,
)
from coronagraph.cdi_reports import (
    _build_planet_position_map_report_groups,
    _save_focal_plane_phase_grid_png,
    _save_focal_plane_phase_shift_grid_png,
    _save_lyot_plane_phase_grid_png,
)
from coronagraph.simulator import CoronagraphSimulator
from coronagraph.region_shapes import build_touching_circle_ring


class CocFeatureHelperTests(unittest.TestCase):
    @staticmethod
    def _make_location_panel(radius: float, theta: float, roi_values: list[float]) -> dict[str, object]:
        return {
            "planet_radius_lamD": radius,
            "planet_theta_deg": theta,
            "planet_center_lamD": (radius, theta),
            "panels": [
                {
                    "requested_roi_size_lamD": roi,
                    "resolved_roi_size_lamD": roi,
                    "planet_center_lamD": (radius, theta),
                    "orbit_radius_lamD": radius,
                    "roi_centers_lamD": [(radius, theta)],
                    "snr": roi,
                    "phase_sweep_mode": "regional",
                }
                for roi in roi_values
            ],
        }

    def test_inclusive_float_range_includes_stop(self) -> None:
        values = _inclusive_float_range(0.5, 1.0, 0.25)
        np.testing.assert_allclose(values, np.array([0.5, 0.75, 1.0]))

    def test_inclusive_float_range_allows_single_value_with_zero_step(self) -> None:
        values = _inclusive_float_range(10.0, 10.0, 0.0)
        np.testing.assert_allclose(values, np.array([10.0]))

    def test_roi_shape_folder_name_normalizes_aliases(self) -> None:
        self.assertEqual(_roi_shape_folder_name("ring"), "shape_ring")
        self.assertEqual(_roi_shape_folder_name("ring-of-circles"), "shape_ring_of_circle")

    def test_polar_to_cartesian_lamd(self) -> None:
        x, y = _polar_to_cartesian_lamD(2.0, 90.0)
        self.assertAlmostEqual(x, 0.0, places=12)
        self.assertAlmostEqual(y, 2.0, places=12)

        x2, y2 = _polar_to_cartesian_lamD(3.0, -180.0)
        self.assertAlmostEqual(x2, -3.0, places=12)
        self.assertAlmostEqual(y2, 0.0, places=12)
        self.assertAlmostEqual(math.hypot(x2, y2), 3.0, places=12)

    def test_planet_region_snr_uses_aperture_mean_and_annulus_aperture_std(self) -> None:
        x = np.linspace(-2.0, 2.0, 401)
        y = np.linspace(-2.0, 2.0, 401)
        xx, yy = np.meshgrid(x, y)
        incoh = np.zeros_like(xx)

        eval_radius = 0.5
        orbit_radius = 1.0
        planet_center = (1.0, 0.0)
        ring = build_touching_circle_ring(
            requested_region_radius_lamD=eval_radius,
            orbit_radius_lamD=orbit_radius,
            anchor_angle_rad=0.0,
            rotation_fraction=0.0,
        )

        background_levels = [1.0, 2.0, 3.0, 4.0, 5.0]
        level_idx = 0
        for cx, cy in ring["centers_lamD"]:
            mask = ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) <= eval_radius ** 2
            if np.hypot(float(cx) - planet_center[0], float(cy) - planet_center[1]) < 1e-12:
                incoh[mask] = 10.0
            else:
                incoh[mask] = background_levels[level_idx]
                level_idx += 1

        signal_mean, noise_std, snr = _planet_region_snr(
            incoh=incoh,
            xx=xx,
            yy=yy,
            planet_center_lamD=planet_center,
            orbit_radius_lamD=orbit_radius,
            eval_radius_lamD=eval_radius,
            annulus_half_width_lamD=0.5,
        )

        planet_mask = ((xx - planet_center[0]) ** 2 + (yy - planet_center[1]) ** 2) <= eval_radius ** 2
        annulus_mask = (np.sqrt(xx**2 + yy**2) >= (orbit_radius - 0.5)) & (np.sqrt(xx**2 + yy**2) <= (orbit_radius + 0.5))
        expected_signal = float(np.mean(incoh[planet_mask]))
        expected_aperture_means: list[float] = []
        for cx, cy in ring["centers_lamD"]:
            mask = ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) <= eval_radius ** 2
            if np.any(mask & planet_mask):
                continue
            if not np.all(annulus_mask[mask]):
                continue
            expected_aperture_means.append(float(np.mean(incoh[mask])))
        expected_background_mean = float(np.mean(np.asarray(expected_aperture_means, dtype=float)))
        expected_noise = float(np.std(np.asarray(expected_aperture_means, dtype=float)))

        self.assertAlmostEqual(signal_mean, expected_signal, places=6)
        self.assertAlmostEqual(noise_std, expected_noise, places=6)
        self.assertAlmostEqual(
            snr,
            (expected_signal - expected_background_mean) / expected_noise,
            places=6,
        )

    def test_evaluate_best_roi_returns_none_at_origin(self) -> None:
        rows, best, panels = _evaluate_best_roi_for_planet_center(
            planet_center=(0.0, 0.0),
            roi_sizes=np.array([0.5, 1.0]),
            region_shape_name="circle",
            sim_local={},
            phase_offsets=np.array([0.0, 1.0]),
            sl16=slice(0, 1),
            half16=1,
            xx16=np.zeros((1, 1)),
            yy16=np.zeros((1, 1)),
        )
        self.assertEqual(rows, [])
        self.assertIsNone(best)
        self.assertEqual(panels, [])

    def test_evaluate_best_roi_collects_lyot_phase_stack(self) -> None:
        sim_local = {
            "pupil_pixels": 24,
            "focal_sampling": 4.0,
            "ghost_fraction": 0.0,
            "include_ghost": False,
            "include_interference": False,
            "companion_flux_ratio": 1e-3,
        }
        base = CoronagraphSimulator(**sim_local).run()
        n_fft = int(base["n_fft"])
        samp = float(base["focal_sampling"])
        central_box_lamD = 12.0
        half16 = int(0.5 * central_box_lamD * samp)
        cc16 = n_fft // 2
        sl16 = slice(cc16 - half16, cc16 + half16)
        x16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, 2 * half16, endpoint=False)
        y16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, 2 * half16, endpoint=False)
        xx16, yy16 = np.meshgrid(x16, y16)

        phase_offsets = np.array([0.0, np.pi], dtype=float)
        rows, best, panels = _evaluate_best_roi_for_planet_center(
            planet_center=(3.0, 0.0),
            roi_sizes=np.array([0.6], dtype=float),
            region_shape_name="circle",
            sim_local=sim_local,
            phase_offsets=phase_offsets,
            sl16=sl16,
            half16=half16,
            xx16=xx16,
            yy16=yy16,
            collect_panels=True,
            phase_sweep_mode="regional",
        )

        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(best)
        self.assertEqual(len(panels), 1)
        lyot_stack = np.asarray(panels[0]["lyot_intensity_stack"], dtype=float)
        np.testing.assert_allclose(np.asarray(panels[0]["lyot_phase_offsets_rad"], dtype=float), phase_offsets)
        self.assertEqual(lyot_stack.shape[0], phase_offsets.size)
        self.assertGreater(lyot_stack.shape[1], 0)
        self.assertGreater(lyot_stack.shape[2], 0)
        self.assertTrue(np.all(lyot_stack >= 0.0))

    def test_evaluate_best_roi_focal_plane_mode_uses_no_local_roi_regions(self) -> None:
        sim_local = {
            "pupil_pixels": 24,
            "focal_sampling": 4.0,
            "ghost_fraction": 0.0,
            "include_ghost": False,
            "include_interference": False,
            "companion_flux_ratio": 1e-3,
        }
        base = CoronagraphSimulator(**sim_local).run()
        n_fft = int(base["n_fft"])
        samp = float(base["focal_sampling"])
        central_box_lamD = 12.0
        half16 = int(0.5 * central_box_lamD * samp)
        cc16 = n_fft // 2
        sl16 = slice(cc16 - half16, cc16 + half16)
        x16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, 2 * half16, endpoint=False)
        y16 = np.linspace(-0.5 * central_box_lamD, 0.5 * central_box_lamD, 2 * half16, endpoint=False)
        xx16, yy16 = np.meshgrid(x16, y16)

        rows, best, panels = _evaluate_best_roi_for_planet_center(
            planet_center=(3.0, 0.0),
            roi_sizes=np.array([0.6], dtype=float),
            region_shape_name="circle",
            sim_local=sim_local,
            phase_offsets=np.array([0.0, np.pi], dtype=float),
            sl16=sl16,
            half16=half16,
            xx16=xx16,
            yy16=yy16,
            collect_panels=True,
            phase_sweep_mode="focal_plane",
        )

        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(best)
        self.assertEqual(len(panels), 1)
        self.assertEqual(rows[0]["n_circles"], 0)
        self.assertEqual(panels[0]["phase_sweep_mode"], "focal_plane")
        self.assertEqual(panels[0]["roi_centers_lamD"], [])
        focal_stack = np.asarray(panels[0]["focal_plane_intensity_stack"], dtype=float)
        np.testing.assert_allclose(
            np.asarray(panels[0]["focal_plane_phase_offsets_rad"], dtype=float),
            np.array([0.0, np.pi], dtype=float),
        )
        self.assertEqual(focal_stack.shape[0], 2)
        self.assertGreater(focal_stack.shape[1], 0)
        self.assertGreater(focal_stack.shape[2], 0)
        self.assertTrue(np.all(focal_stack >= 0.0))
        phase_shift_stack = np.asarray(panels[0]["focal_plane_phase_shift_stack"], dtype=float)
        self.assertEqual(phase_shift_stack.shape, focal_stack.shape)
        np.testing.assert_allclose(phase_shift_stack[0], 0.0)
        np.testing.assert_allclose(phase_shift_stack[1], np.pi)

    def test_planet_position_roi_sweep_applies_user_lyot_reference_percent(self) -> None:
        captured_scales: list[float] = []

        def fake_evaluate(**kwargs):
            captured_scales.append(float(kwargs["sim_local"]["lyot_reference_scale"]))
            return [], None, []

        args = unittest.mock.Mock()
        args.plot_poster_figure = False
        args.region_shape = "circle"
        args.phase_sweep_mode = "regional"
        args.planet_position_radius_min = 3.0
        args.planet_position_radius_max = 3.0
        args.planet_position_radius_step = 0.0
        args.planet_position_theta_min_deg = 0.0
        args.planet_position_theta_max_deg = 0.0
        args.planet_position_theta_step_deg = 0.0
        args.roi_size_min = 0.5
        args.roi_size_max = 0.5
        args.roi_size_step = 0.0
        args.lyot_reference_percent = 65.0
        args.lyot_reference_percent_sweep_min = 100.0
        args.lyot_reference_percent_sweep_max = 100.0
        args.lyot_reference_percent_sweep_step = 0.0
        args.phase_cycles = 1.0
        args.phase_step = 2
        args.local_region_radius = 0.5

        sim_local = {
            "pupil_pixels": 16,
            "focal_sampling": 2.0,
            "ghost_fraction": 0.0,
            "include_ghost": False,
            "include_interference": False,
            "perfect_coronagraph": True,
            "lyot_reference_scale": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("coronagraph.cdi_feature._evaluate_best_roi_for_planet_center", side_effect=fake_evaluate):
                with mock.patch("coronagraph.cdi_feature._save_perfect_coronagraph_diagnostic_psfs"):
                    _run_planet_position_roi_size_sweep(
                        args=args,
                        sim_local=sim_local,
                        incoherence_map_mode="fft_band",
                        sweep_output_dir=tmpdir,
                        mask_output_tag="pcg",
                        phase_cycles_tag="_cy1p000",
                        phase_sweep_mode_tag="_mreg",
                        single_region_tag="_f1",
                        ghost_suffix="_g0",
                    )

        self.assertEqual(captured_scales, [0.65])

    def test_summarize_roi_snr_trend_detects_monotonic_increase(self) -> None:
        rows = [
            {"resolved_roi_size_lamD": 3.0, "snr": -0.5},
            {"resolved_roi_size_lamD": 1.0, "snr": -2.0},
            {"resolved_roi_size_lamD": 2.0, "snr": -1.0},
        ]

        self.assertEqual(_summarize_roi_snr_trend(rows), "monotonic_increase")

    def test_summarize_roi_snr_trend_detects_non_monotonic(self) -> None:
        rows = [
            {"resolved_roi_size_lamD": 1.0, "snr": 1.0},
            {"resolved_roi_size_lamD": 2.0, "snr": 3.0},
            {"resolved_roi_size_lamD": 3.0, "snr": 2.0},
        ]

        self.assertEqual(_summarize_roi_snr_trend(rows), "non_monotonic")

    def test_planet_position_map_groups_single_varying_radius_stays_together(self) -> None:
        groups = _build_planet_position_map_report_groups(
            [
                self._make_location_panel(5.0, 45.0, [1.0]),
                self._make_location_panel(6.0, 45.0, [1.0]),
            ]
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["file_tag"], "grouped_all")
        self.assertEqual(len(groups[0]["panel_collections"]), 2)

    def test_planet_position_map_groups_prioritize_radius_over_roi_and_theta(self) -> None:
        groups = _build_planet_position_map_report_groups(
            [
                self._make_location_panel(5.0, 0.0, [1.0, 2.0]),
                self._make_location_panel(5.0, 90.0, [1.0, 2.0]),
                self._make_location_panel(6.0, 0.0, [1.0, 2.0]),
                self._make_location_panel(6.0, 90.0, [1.0, 2.0]),
            ]
        )

        self.assertEqual([group["file_tag"] for group in groups], [
            "grouped_by_radius_r_5p000",
            "grouped_by_radius_r_6p000",
        ])
        self.assertEqual([len(group["panel_collections"]) for group in groups], [2, 2])

    def test_planet_position_map_groups_prioritize_roi_when_radius_fixed(self) -> None:
        groups = _build_planet_position_map_report_groups(
            [
                self._make_location_panel(5.0, 0.0, [1.0, 2.0]),
                self._make_location_panel(5.0, 90.0, [1.0, 2.0]),
            ]
        )

        self.assertEqual([group["file_tag"] for group in groups], [
            "grouped_by_roi_roi_1p000",
            "grouped_by_roi_roi_2p000",
        ])
        self.assertEqual([len(group["panel_collections"]) for group in groups], [2, 2])

    def test_save_lyot_plane_phase_grid_png_writes_file(self) -> None:
        panel = {
            "phase_sweep_mode": "regional",
            "lyot_phase_offsets_rad": np.array([0.0, np.pi], dtype=float),
            "lyot_intensity_stack": np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[2.0, 3.0], [4.0, 5.0]],
                ],
                dtype=float,
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "lyot_grid.png")
            _save_lyot_plane_phase_grid_png(
                output_path=output_path,
                panels=[panel],
                panel_labels=["flux=1e-3"],
                figure_title="Lyot Plane Grid",
            )
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_save_focal_plane_phase_grid_png_writes_file(self) -> None:
        panel = {
            "phase_sweep_mode": "focal_plane",
            "focal_plane_phase_offsets_rad": np.array([0.0, np.pi], dtype=float),
            "focal_plane_intensity_stack": np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[2.0, 3.0], [4.0, 5.0]],
                ],
                dtype=float,
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "focal_grid.png")
            _save_focal_plane_phase_grid_png(
                output_path=output_path,
                panels=[panel],
                panel_labels=["flux=1e-3"],
                figure_title="Focal Plane Grid",
            )
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_save_focal_plane_phase_shift_grid_png_writes_file(self) -> None:
        panel = {
            "phase_sweep_mode": "focal_plane",
            "focal_plane_phase_offsets_rad": np.array([0.0, np.pi], dtype=float),
            "focal_plane_phase_shift_stack": np.array(
                [
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[np.pi, np.pi], [np.pi, np.pi]],
                ],
                dtype=float,
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "focal_phase_shift_grid.png")
            _save_focal_plane_phase_shift_grid_png(
                output_path=output_path,
                panels=[panel],
                panel_labels=["flux=1e-3"],
                figure_title="Focal Plane Phase Shift Grid",
            )
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)


if __name__ == "__main__":
    unittest.main()
