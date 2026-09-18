from __future__ import annotations

import unittest

import numpy as np

from coronagraph.cli import _mode_tag
from coronagraph.cdi_feature import _effective_roi_sizes_for_mode, _mask_rotation_angles_rad


class CdiMaskRotationTests(unittest.TestCase):
    def test_mask_rotation_angles_cover_one_turn_without_repeat(self) -> None:
        angles = _mask_rotation_angles_rad(5)
        expected = 2.0 * np.pi * np.arange(5, dtype=float) / 5.0
        np.testing.assert_allclose(angles, expected)
        self.assertLess(float(angles[-1]), 2.0 * np.pi)

    def test_mask_rotation_angles_require_positive_step_count(self) -> None:
        with self.assertRaises(ValueError):
            _mask_rotation_angles_rad(0)

    def test_mask_rotation_collapses_roi_size_sweep_to_single_placeholder(self) -> None:
        roi_sizes = np.array([0.5, 1.0, 1.5], dtype=float)
        collapsed = _effective_roi_sizes_for_mode(roi_sizes, "mask_rotation")
        np.testing.assert_allclose(collapsed, np.array([0.5]))

    def test_focal_plane_collapses_roi_size_sweep_to_single_placeholder(self) -> None:
        roi_sizes = np.array([0.5, 1.0, 1.5], dtype=float)
        collapsed = _effective_roi_sizes_for_mode(roi_sizes, "focal_plane")
        np.testing.assert_allclose(collapsed, np.array([0.5]))

    def test_focal_plane_mode_uses_compact_tag(self) -> None:
        self.assertEqual(_mode_tag("focal_plane"), "fpl")


if __name__ == "__main__":
    unittest.main()
