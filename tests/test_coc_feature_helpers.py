from __future__ import annotations

import math
import unittest

import numpy as np

from coronagraph.coc_feature import (
    _evaluate_best_roi_for_planet_center,
    _inclusive_float_range,
    _planet_region_snr,
    _polar_to_cartesian_lamD,
)
from coronagraph.region_shapes import build_touching_circle_ring


class CocFeatureHelperTests(unittest.TestCase):
    def test_inclusive_float_range_includes_stop(self) -> None:
        values = _inclusive_float_range(0.5, 1.0, 0.25)
        np.testing.assert_allclose(values, np.array([0.5, 0.75, 1.0]))

    def test_inclusive_float_range_allows_single_value_with_zero_step(self) -> None:
        values = _inclusive_float_range(10.0, 10.0, 0.0)
        np.testing.assert_allclose(values, np.array([10.0]))

    def test_polar_to_cartesian_lamd(self) -> None:
        x, y = _polar_to_cartesian_lamD(2.0, 90.0)
        self.assertAlmostEqual(x, 0.0, places=12)
        self.assertAlmostEqual(y, 2.0, places=12)

        x2, y2 = _polar_to_cartesian_lamD(3.0, -180.0)
        self.assertAlmostEqual(x2, -3.0, places=12)
        self.assertAlmostEqual(y2, 0.0, places=12)
        self.assertAlmostEqual(math.hypot(x2, y2), 3.0, places=12)

    def test_planet_region_snr_uses_aperture_sum_and_annulus_aperture_std(self) -> None:
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

        signal_sum, noise_std, snr = _planet_region_snr(
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
        expected_signal = float(np.sum(incoh[planet_mask]))
        expected_aperture_sums: list[float] = []
        for cx, cy in ring["centers_lamD"]:
            mask = ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) <= eval_radius ** 2
            if np.any(mask & planet_mask):
                continue
            if not np.all(annulus_mask[mask]):
                continue
            expected_aperture_sums.append(float(np.sum(incoh[mask])))
        expected_noise = float(np.std(np.asarray(expected_aperture_sums, dtype=float)))

        self.assertAlmostEqual(signal_sum, expected_signal, places=6)
        self.assertAlmostEqual(noise_std, expected_noise, places=6)
        self.assertAlmostEqual(snr, expected_signal / expected_noise, places=6)

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


if __name__ == "__main__":
    unittest.main()
