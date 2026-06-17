from __future__ import annotations

import tempfile
import unittest

import numpy as np
from astropy.io import fits

from coronagraph.simulator import CoronagraphSimulator


class SimulatorLocalPhaseRegionTests(unittest.TestCase):
    def test_ring_local_phase_map_builds_annulus(self) -> None:
        sim = CoronagraphSimulator(
            pupil_pixels=32,
            focal_sampling=10.0,
            focal_local_phase_offset=1.0,
            focal_local_phase_shape="ring",
            focal_local_phase_inner_radius_lamD=2.0,
            focal_local_phase_outer_radius_lamD=3.0,
        )

        phase_map = sim._local_focal_phase_map()
        c = (sim.n_fft - 1) / 2.0
        x = sim._x / sim.focal_sampling
        y = sim._y / sim.focal_sampling
        rr = np.sqrt(x**2 + y**2)

        self.assertEqual(float(phase_map[int(round(c)), int(round(c))]), 0.0)
        annulus_mask = (rr >= 2.0) & (rr <= 3.0)
        self.assertTrue(np.any(phase_map[annulus_mask] > 0.0))
        self.assertTrue(np.all(phase_map[rr < 2.0] == 0.0))
        self.assertTrue(np.all(phase_map[rr > 3.0] == 0.0))

    def test_phase_screen_uses_first_cube_plane_on_pupil_grid(self) -> None:
        cube = np.zeros((2, 4, 4), dtype=float)
        cube[0] = np.arange(16, dtype=float).reshape(4, 4)
        cube[1] = -1.0

        with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
            fits.writeto(tmp.name, cube, overwrite=True)
            sim = CoronagraphSimulator(
                pupil_pixels=4,
                focal_sampling=1.0,
                phase_screen_path=tmp.name,
                phase_screen_index=0,
            )

            phase_map = sim._pupil_phase_screen_map()
            np.testing.assert_allclose(phase_map, cube[0])

    def test_phase_screen_changes_propagated_psf(self) -> None:
        cube = np.zeros((1, 6, 6), dtype=float)
        cube[0, :, 3:] = np.pi / 2.0

        with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
            fits.writeto(tmp.name, cube, overwrite=True)
            base = CoronagraphSimulator(
                pupil_pixels=6,
                focal_sampling=2.0,
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            ).run()
            aberrated = CoronagraphSimulator(
                pupil_pixels=6,
                focal_sampling=2.0,
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
                phase_screen_path=tmp.name,
                phase_screen_index=0,
            ).run()

            self.assertFalse(np.allclose(base["direct_psf"], aberrated["direct_psf"]))
            phase_map = aberrated["pupil_phase_screen"]
            start = (phase_map.shape[0] - cube.shape[-1]) // 2
            stop = start + cube.shape[-1]
            np.testing.assert_allclose(phase_map[start:stop, start:stop], cube[0])


if __name__ == "__main__":
    unittest.main()
