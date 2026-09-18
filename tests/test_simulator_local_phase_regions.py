from __future__ import annotations

import tempfile
import unittest

import numpy as np
from astropy.io import fits

from coronagraph.masks import PhaseMask
from coronagraph.simulator import CoronagraphSimulator


class _LinearXPhaseMask(PhaseMask):
    def transmission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(1j * x)


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

    def test_coherent_ring_speckles_add_expected_offsets(self) -> None:
        result = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            companion_flux_ratio=1e-3,
            companion_offset_lamD=(4.0, 0.0),
            coherent_ring_speckle_count=2,
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
        ).run()

        self.assertEqual(result["coherent_ring_speckle_count"], 2)
        self.assertEqual(len(result["coherent_ring_speckle_offsets_lamD"]), 2)
        expected = {
            (-2.0, round(2.0 * np.sqrt(3.0), 6)),
            (-2.0, round(-2.0 * np.sqrt(3.0), 6)),
        }
        actual = {
            (round(float(x), 6), round(float(y), 6))
            for x, y in result["coherent_ring_speckle_offsets_lamD"]
        }
        self.assertEqual(actual, expected)
        self.assertGreater(np.max(result["coronagraphic_psf_star"]), 0.0)

    def test_disable_star_keeps_companion_branch_only(self) -> None:
        result = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            companion_flux_ratio=1e-3,
            companion_offset_lamD=(4.0, 0.0),
            include_star=False,
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
        ).run()

        self.assertFalse(result["include_star"])
        self.assertAlmostEqual(float(np.max(result["final_psf_with_ghost_star"])), 0.0)
        self.assertGreater(float(np.max(result["final_psf_with_ghost_companion"])), 0.0)
        np.testing.assert_allclose(
            result["final_psf_with_ghost"],
            result["final_psf_with_ghost_companion"],
        )

    def test_disable_star_and_companion_keeps_coherent_ring_speckles_only(self) -> None:
        result = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            companion_flux_ratio=1e-3,
            companion_offset_lamD=(4.0, 0.0),
            coherent_ring_speckle_intensity=1e-3,
            include_star=False,
            include_companion=False,
            coherent_ring_speckle_count=2,
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
        ).run()
        star_only = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            companion_flux_ratio=1e-3,
            companion_offset_lamD=(4.0, 0.0),
            coherent_ring_speckle_intensity=1e-3,
            include_star=True,
            include_companion=False,
            coherent_ring_speckle_count=0,
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
        ).run()

        self.assertFalse(result["include_star"])
        self.assertFalse(result["include_companion"])
        self.assertEqual(result["coherent_ring_speckle_count"], 2)
        self.assertEqual(len(result["coherent_ring_speckle_offsets_lamD"]), 2)
        self.assertEqual(len(result["coherent_ring_speckle_source_amplitudes"]), 2)
        self.assertTrue(
            all(float(amplitude) > 0.0 for amplitude in result["coherent_ring_speckle_source_amplitudes"])
        )
        self.assertAlmostEqual(float(np.max(result["final_psf_with_ghost_companion"])), 0.0)
        self.assertGreater(float(np.max(result["final_psf_with_ghost_star"])), 0.0)
        np.testing.assert_allclose(
            result["final_psf_with_ghost"],
            result["final_psf_with_ghost_star"],
        )
        self.assertGreater(
            float(np.max(np.abs(result["lyot_field_before_reference_subtraction"]))),
            0.0,
        )
        self.assertFalse(
            np.allclose(
                result["lyot_field_before_reference_subtraction"],
                star_only["lyot_field_before_reference_subtraction"],
            )
        )

    def test_coherent_ring_speckles_use_configured_star_relative_intensity(self) -> None:
        common_kwargs = dict(
            pupil_pixels=24,
            focal_sampling=4.0,
            companion_flux_ratio=0.0,
            companion_offset_lamD=(4.0, 0.0),
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
            e_final_phase_offset=0.0,
            focal_local_phase_offset=0.0,
            focal_local_phase_centers_lamD=(),
            focal_local_phase_radius_lamD=0.0,
        )

        with_speckles_sim = CoronagraphSimulator(
            **{
                **common_kwargs,
                "coherent_ring_speckle_count": 2,
                "coherent_ring_speckle_intensity": 2.5e-3,
            }
        )
        with_speckles = with_speckles_sim.run()

        self.assertEqual(len(with_speckles["coherent_ring_speckle_source_amplitudes"]), 2)
        for amplitude in with_speckles["coherent_ring_speckle_source_amplitudes"]:
            self.assertAlmostEqual(float(amplitude), np.sqrt(2.5e-3))
        self.assertEqual(len(with_speckles["coherent_ring_speckle_offsets_lamD"]), 2)
        self.assertGreater(float(np.max(with_speckles["final_psf_with_ghost"])), 0.0)

    def test_phase_mask_rotation_rotates_mask_coordinates(self) -> None:
        base = CoronagraphSimulator(
            pupil_pixels=8,
            focal_sampling=2.0,
            phase_mask=_LinearXPhaseMask(),
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
        )
        rotated = CoronagraphSimulator(
            pupil_pixels=8,
            focal_sampling=2.0,
            phase_mask=_LinearXPhaseMask(),
            phase_mask_rotation_rad=np.pi / 2.0,
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
        )

        base_mask = base._sampled_phase_mask()
        rotated_mask = rotated._sampled_phase_mask()
        expected = np.exp(1j * (base._y / base.focal_sampling))

        self.assertFalse(np.allclose(base_mask, rotated_mask))
        np.testing.assert_allclose(rotated_mask, expected)

    def test_perfect_coronagraph_cancels_unaberrated_star_at_lyot_plane(self) -> None:
        result = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            secondary_diameter_ratio=0.25,
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
            perfect_coronagraph=True,
        ).run()

        self.assertTrue(result["perfect_coronagraph"])
        self.assertGreater(np.max(np.abs(result["lyot_reference_field"])), 0.0)
        self.assertLess(np.max(np.abs(result["lyot_field"])), 1e-10)
        self.assertLess(np.max(result["coronagraphic_psf_star"]), 1e-12)

    def test_perfect_coronagraph_preserves_companion_branch(self) -> None:
        result = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            secondary_diameter_ratio=0.25,
            companion_flux_ratio=1e-3,
            companion_offset_lamD=(3.0, 0.0),
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
            perfect_coronagraph=True,
        ).run()

        self.assertLess(np.max(result["coronagraphic_psf_star"]), 1e-12)
        self.assertGreater(np.max(result["coronagraphic_psf_companion"]), 0.0)
        self.assertGreater(np.max(result["coronagraphic_psf"]), 0.0)

    def test_perfect_coronagraph_reference_branch_skips_focal_plane_modulation(self) -> None:
        baseline = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            secondary_diameter_ratio=0.25,
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
        ).run()
        modulated = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            secondary_diameter_ratio=0.25,
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
            perfect_coronagraph=True,
            e_final_phase_offset=np.pi / 2.0,
            focal_local_phase_offset=np.pi / 3.0,
            focal_local_phase_centers_lamD=((1.5, 0.0),),
            focal_local_phase_radius_lamD=0.5,
        ).run()

        np.testing.assert_allclose(
            modulated["lyot_reference_field"],
            baseline["lyot_field_before_reference_subtraction"],
        )
        self.assertGreater(np.max(np.abs(modulated["lyot_field"])), 0.0)

    def test_perfect_coronagraph_partial_reference_subtraction_scales_lyot_reference(self) -> None:
        baseline = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            secondary_diameter_ratio=0.25,
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
        ).run()
        partial = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            secondary_diameter_ratio=0.25,
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
            perfect_coronagraph=True,
            lyot_reference_scale=0.5,
        ).run()

        self.assertAlmostEqual(partial["lyot_reference_scale"], 0.5)
        np.testing.assert_allclose(
            partial["lyot_reference_field"],
            0.5 * baseline["lyot_field_before_reference_subtraction"],
        )
        self.assertGreater(np.max(np.abs(partial["lyot_field"])), 0.0)
        self.assertGreater(np.max(partial["coronagraphic_psf_star"]), 0.0)

    def test_perfect_coronagraph_keeps_speckles_in_full_output_but_not_reference_branch(self) -> None:
        baseline = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            secondary_diameter_ratio=0.25,
            companion_flux_ratio=1e-3,
            companion_offset_lamD=(4.0, 0.0),
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
            perfect_coronagraph=True,
            coherent_ring_speckle_count=0,
        ).run()
        with_speckles = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            secondary_diameter_ratio=0.25,
            companion_flux_ratio=1e-3,
            companion_offset_lamD=(4.0, 0.0),
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
            perfect_coronagraph=True,
            coherent_ring_speckle_count=3,
        ).run()

        self.assertEqual(len(with_speckles["coherent_ring_speckle_offsets_lamD"]), 3)
        np.testing.assert_allclose(
            with_speckles["lyot_reference_field"],
            baseline["lyot_reference_field"],
        )
        self.assertFalse(
            np.allclose(
                with_speckles["coronagraphic_psf_star"],
                baseline["coronagraphic_psf_star"],
            )
        )
        self.assertFalse(
            np.allclose(
                with_speckles["final_psf_with_ghost"],
                baseline["final_psf_with_ghost"],
            )
        )

    def test_perfect_coronagraph_reference_branch_is_star_only(self) -> None:
        baseline = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            secondary_diameter_ratio=0.25,
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
            perfect_coronagraph=True,
        ).run()
        with_companion_and_speckles = CoronagraphSimulator(
            pupil_pixels=16,
            focal_sampling=2.0,
            secondary_diameter_ratio=0.25,
            companion_flux_ratio=1e-3,
            companion_offset_lamD=(4.0, 0.0),
            coherent_ring_speckle_count=3,
            ghost_fraction=0.0,
            include_ghost=False,
            include_interference=False,
            perfect_coronagraph=True,
        ).run()

        np.testing.assert_allclose(
            with_companion_and_speckles["lyot_reference_field"],
            baseline["lyot_reference_field"],
        )


if __name__ == "__main__":
    unittest.main()
