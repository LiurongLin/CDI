from __future__ import annotations

import math
import unittest

from coronagraph.region_shapes import (
    annulus_radii_from_width,
    build_touching_circle_ring,
    normalize_region_shape,
)


class TouchingCircleRingTests(unittest.TestCase):
    def test_region_shape_aliases_normalize(self) -> None:
        self.assertEqual(normalize_region_shape("ring"), "ring")
        self.assertEqual(normalize_region_shape("annulus"), "ring")
        self.assertEqual(normalize_region_shape("ring_of_circle"), "ring_of_circle")
        self.assertEqual(normalize_region_shape("ring-of-circles"), "ring_of_circle")

    def test_snaps_to_nearest_valid_radius_and_preserves_touching_spacing(self) -> None:
        ring = build_touching_circle_ring(
            requested_region_radius_lamD=2.0,
            orbit_radius_lamD=4.5,
            anchor_angle_rad=0.3,
        )

        self.assertEqual(ring["n_circles"], 7)
        self.assertAlmostEqual(ring["resolved_radius_lamD"], 4.5 * math.sin(math.pi / 7.0))
        self.assertAlmostEqual(math.atan2(ring["centers_lamD"][0][1], ring["centers_lamD"][0][0]), 0.3)

        x0, y0 = ring["centers_lamD"][0]
        x1, y1 = ring["centers_lamD"][1]
        spacing = math.hypot(x1 - x0, y1 - y0)
        self.assertAlmostEqual(spacing, 2.0 * ring["resolved_radius_lamD"], places=9)

    def test_rotation_fraction_matches_option_a_edge_cut_target(self) -> None:
        centered = build_touching_circle_ring(
            requested_region_radius_lamD=2.0,
            orbit_radius_lamD=4.5,
            anchor_angle_rad=0.3,
            rotation_fraction=0.0,
        )
        shifted = build_touching_circle_ring(
            requested_region_radius_lamD=2.0,
            orbit_radius_lamD=4.5,
            anchor_angle_rad=0.3,
            rotation_fraction=1.0,
        )

        self.assertAlmostEqual(centered["applied_rotation_rad"], 0.0)
        self.assertAlmostEqual(shifted["applied_rotation_rad"], shifted["edge_cut_rotation_rad"])

        px = 4.5 * math.cos(0.3)
        py = 4.5 * math.sin(0.3)
        cx, cy = shifted["centers_lamD"][0]
        self.assertAlmostEqual(math.hypot(cx - px, cy - py), shifted["resolved_radius_lamD"], places=9)

    def test_caps_large_requested_radius_at_three_circles(self) -> None:
        ring = build_touching_circle_ring(
            requested_region_radius_lamD=10.0,
            orbit_radius_lamD=5.0,
            anchor_angle_rad=0.0,
        )

        self.assertEqual(ring["n_circles"], 3)
        self.assertAlmostEqual(ring["resolved_radius_lamD"], 5.0 * math.sin(math.pi / 3.0))

    def test_annulus_radii_from_width_keeps_planet_separation_as_midpoint(self) -> None:
        rmin, rmax = annulus_radii_from_width(mid_radius_lamD=4.5, width_lamD=1.0)

        self.assertAlmostEqual(rmin, 4.0)
        self.assertAlmostEqual(rmax, 5.0)
        self.assertAlmostEqual(0.5 * (rmin + rmax), 4.5)
        self.assertAlmostEqual(rmax - rmin, 1.0)


if __name__ == "__main__":
    unittest.main()
