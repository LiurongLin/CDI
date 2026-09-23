from __future__ import annotations

import tempfile
import unittest

import numpy as np

from coronagraph.interactive_slm_gui import (
    SLMLyotConfig,
    SLMLyotResponseCalculator,
    save_slm_lyot_result,
)
from coronagraph.html_slm_gui import _SLMHtmlServer
from coronagraph.masks import NoPhaseMask


class InteractiveSLMGuiTests(unittest.TestCase):
    def test_planet_intensity_ratio_uses_square_root_field_scaling(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(0.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[calc.n_fft // 2 :, :] = True

        result = calc.propagate(
            mask,
            include_star=True,
            include_planet=True,
            include_speckles=False,
            star_planet_ratio=400.0,
            reference_phase_modulation_rad=0.0,
        )

        self.assertGreater(result["metrics"]["A_star"], 0.0)
        self.assertAlmostEqual(
            result["metrics"]["A_planet"],
            result["metrics"]["A_star"] / np.sqrt(400.0),
            places=10,
        )

    def test_phase_modulation_value_controls_selected_pixels(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(0.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[calc.n_fft // 2 :, :] = True

        result = calc.propagate(
            mask,
            include_star=False,
            include_planet=True,
            phase_modulation_rad=0.25,
        )

        self.assertEqual(result["phase_modulation_rad"], 0.25)
        self.assertTrue(np.all(result["phase_map_rad"][mask] == 0.25))
        self.assertTrue(np.all(result["phase_map_rad"][~mask] == 0.0))

    def test_phase_sweep_reports_first_harmonic_modulation_metrics(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(0.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[calc.n_fft // 2 :, :] = True

        sweep = calc.phase_sweep(
            mask,
            phase_steps=4,
            include_star=True,
            include_planet=True,
            include_speckles=False,
            star_planet_ratio=400.0,
        )
        coherent_power = sweep["powers"]["coherent"]
        phases = sweep["phases"]
        c1 = np.mean(coherent_power * np.exp(-1j * phases))

        np.testing.assert_allclose(phases, [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
        self.assertAlmostEqual(
            sweep["metrics"]["coherent"]["M_harmonic"],
            float(abs(c1)),
            places=10,
        )
        self.assertAlmostEqual(
            sweep["metrics"]["coherent"]["phase_response"],
            float(np.angle(c1)),
            places=10,
        )
        self.assertIn("R_mod_harmonic", sweep["metrics"]["ratios"])
        self.assertIn("R_mod_pp", sweep["metrics"]["ratios"])

    def test_phase_sweep_incoherent_power_sums_star_and_planet_intensities(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(2.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[:, calc.n_fft // 2 :] = True

        sweep = calc.phase_sweep(
            mask,
            phase_steps=4,
            include_star=True,
            include_planet=True,
            include_speckles=False,
            star_planet_ratio=400.0,
        )

        np.testing.assert_allclose(
            sweep["powers"]["incoherent"],
            sweep["powers"]["star"] + sweep["powers"]["planet"],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_phase_sweep_reference_tracks_full_mask_global_phase_response(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(2.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.ones((calc.n_fft, calc.n_fft), dtype=bool)

        field_sweep = calc.phase_sweep(
            mask,
            phase_steps=4,
            subtraction_mode="field",
            include_star=True,
            include_planet=False,
            include_speckles=False,
        )
        intensity_sweep = calc.phase_sweep(
            mask,
            phase_steps=4,
            subtraction_mode="intensity",
            include_star=True,
            include_planet=False,
            include_speckles=False,
        )

        self.assertAlmostEqual(
            field_sweep["metrics"]["star"]["M_harmonic"],
            0.0,
            places=10,
        )
        self.assertEqual(intensity_sweep["subtraction_mode"], "intensity")
        self.assertAlmostEqual(
            intensity_sweep["metrics"]["star"]["M_harmonic"],
            0.0,
            places=10,
        )

    def test_reference_subtraction_scale_controls_tracked_reference_field_response(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(2.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[:, calc.n_fft // 2 :] = True

        full_reference = calc.phase_sweep(
            mask,
            phase_steps=4,
            subtraction_mode="field",
            lyot_reference_scale=1.0,
            include_star=True,
            include_planet=False,
            include_speckles=False,
        )
        no_reference = calc.phase_sweep(
            mask,
            phase_steps=4,
            subtraction_mode="field",
            lyot_reference_scale=0.0,
            include_star=True,
            include_planet=False,
            include_speckles=False,
        )

        self.assertAlmostEqual(full_reference["metrics"]["star"]["M_harmonic"], 0.0, places=10)
        self.assertGreater(no_reference["metrics"]["star"]["M_harmonic"], 0.0)
        self.assertEqual(no_reference["lyot_reference_scale"], 0.0)

    def test_coherent_metric_is_norm_of_complex_sum_not_sum_of_norms(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=10,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(3.0, 0.0),
                coherent_ring_speckle_count=2,
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[:, calc.n_fft // 2 :] = True

        result = calc.propagate(
            mask,
            include_star=True,
            include_planet=False,
            include_speckles=True,
            star_speckle_ratio=100.0,
            lyot_reference_scale=0.0,
        )
        stop = result["lyot_stop"].astype(bool)
        expected = np.sqrt(
            np.sum(np.abs(result["fields"]["coherent"]["delta"][stop]) ** 2)
        )

        self.assertAlmostEqual(result["metrics"]["A_coherent"], float(expected), places=10)
        self.assertNotAlmostEqual(
            result["metrics"]["A_coherent"],
            result["metrics"]["A_star"] + result["metrics"]["A_speckle"],
            places=10,
        )

    def test_can_select_subset_of_coherent_speckles(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=10,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(3.0, 0.0),
                coherent_ring_speckle_count=3,
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[:, calc.n_fft // 2 :] = True

        one = calc.propagate(
            mask,
            include_star=False,
            include_planet=False,
            include_speckles=True,
            selected_speckle_indices=[1],
            star_speckle_ratio=100.0,
        )
        all_speckles = calc.propagate(
            mask,
            include_star=False,
            include_planet=False,
            include_speckles=True,
            star_speckle_ratio=100.0,
        )

        self.assertEqual(one["selected_speckle_indices"], (1,))
        self.assertEqual(len(one["speckle_offsets_lamD"]), 1)
        self.assertEqual(len(all_speckles["speckle_offsets_lamD"]), 3)
        self.assertFalse(
            np.allclose(
                one["fields"]["speckle"]["delta"],
                all_speckles["fields"]["speckle"]["delta"],
            )
        )

    def test_each_speckle_uses_same_focal_plane_intensity_as_planet(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=10,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(3.0, 0.0),
                custom_speckle_offsets_lamD=((3.0, 0.0),),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[:, calc.n_fft // 2 :] = True

        result = calc.propagate(
            mask,
            include_star=False,
            include_planet=True,
            include_speckles=True,
            star_planet_ratio=400.0,
            star_speckle_ratio=1.0,
        )

        np.testing.assert_allclose(
            result["fields"]["speckle"]["delta"],
            result["fields"]["planet"]["delta"],
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertAlmostEqual(
            result["metrics"]["A_speckle"],
            result["metrics"]["A_planet"],
            places=10,
        )

    def test_incoherent_map_combines_star_and_planet_without_speckles(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=10,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(3.0, 0.0),
                coherent_ring_speckle_count=1,
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[:, calc.n_fft // 2 :] = True

        result = calc.propagate(
            mask,
            include_star=True,
            include_planet=True,
            include_speckles=True,
            star_planet_ratio=400.0,
            star_speckle_ratio=100.0,
        )
        expected_map = np.sqrt(
            np.abs(result["fields"]["star"]["delta"]) ** 2
            + np.abs(result["fields"]["planet"]["delta"]) ** 2
        )
        stop = result["lyot_stop"].astype(bool)
        expected_a = float(np.sqrt(np.sum(expected_map[stop] ** 2)))

        np.testing.assert_allclose(result["fields"]["incoherent"]["delta"], expected_map)
        self.assertFalse(
            np.allclose(
                result["fields"]["incoherent"]["delta"],
                np.sqrt(
                    np.abs(result["fields"]["coherent"]["delta"]) ** 2
                    + np.abs(result["fields"]["planet"]["delta"]) ** 2
                ),
            )
        )
        self.assertAlmostEqual(result["metrics"]["A_incoherent"], expected_a, places=10)
        self.assertAlmostEqual(
            result["metrics"]["R1"],
            result["metrics"]["A_coherent"] / (result["metrics"]["A_incoherent"] + 1e-30),
            places=10,
        )

    def test_disabling_star_skips_lyot_reference_subtraction(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(2.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )

        def fail_reference(_sim):
            raise AssertionError("No Lyot reference should be built when the star is disabled.")

        calc._reference_field = fail_reference
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[calc.n_fft // 2 :, :] = True

        result = calc.propagate(
            mask,
            include_star=False,
            include_planet=True,
            include_speckles=False,
            star_planet_ratio=400.0,
        )

        stop = result["lyot_stop"].astype(bool)
        expected_a = float(
            np.sqrt(np.sum(np.abs(result["fields"]["planet"]["delta"][stop]) ** 2))
        )
        self.assertNotIn("A_star", result["metrics"])
        self.assertNotIn("A_coherent", result["metrics"])
        self.assertNotIn("R1", result["metrics"])
        self.assertAlmostEqual(result["metrics"]["A_incoherent"], expected_a, places=10)

    def test_save_result_writes_complex_fields_and_source_state(self) -> None:
        calc = SLMLyotResponseCalculator(
            SLMLyotConfig(
                pupil_pixels=8,
                focal_sampling=2.0,
                phase_mask=NoPhaseMask(),
                companion_offset_lamD=(0.0, 0.0),
                ghost_fraction=0.0,
                include_ghost=False,
                include_interference=False,
            )
        )
        mask = np.zeros((calc.n_fft, calc.n_fft), dtype=bool)
        mask[calc.n_fft // 2 :, :] = True
        result = calc.propagate(mask, include_star=True, include_planet=True)

        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            save_slm_lyot_result(tmp.name, result)
            data = np.load(tmp.name)
            self.assertIn("delta_E_L_star", data.files)
            self.assertIn("delta_E_L_planet", data.files)
            self.assertIn("delta_E_L_coherent", data.files)
            self.assertTrue(bool(data["include_star"]))
            self.assertTrue(np.iscomplexobj(data["delta_E_L_star"]))
            self.assertIn("phase_modulation_rad", data.files)

    def test_html_config_includes_slm_source_markers(self) -> None:
        server = _SLMHtmlServer()
        payload = server.config_payload()
        labels = {marker["label"] for marker in payload["source_markers"]}

        self.assertIn("star", labels)
        self.assertIn("planet", labels)
        self.assertTrue(any(label.startswith("s") for label in labels))
        self.assertTrue(
            all("index" in marker for marker in payload["source_markers"] if marker.get("kind") == "speckle")
        )
        self.assertEqual(payload["n_fft"], server.calculator.n_fft)

    def test_html_source_geometry_can_move_planet_and_speckles(self) -> None:
        server = _SLMHtmlServer()
        payload = server.update_source_geometry(
            planet_x=2.5,
            planet_y=-1.25,
            speckle_offsets=[[1.0, 2.0], [-3.5, 0.5]],
        )
        markers = payload["source_markers"]
        planet = next(marker for marker in markers if marker["kind"] == "planet")
        speckles = [marker for marker in markers if marker["kind"] == "speckle"]

        self.assertEqual((planet["x"], planet["y"]), (2.5, -1.25))
        self.assertEqual(
            [(marker["x"], marker["y"]) for marker in speckles],
            [(1.0, 2.0), (-3.5, 0.5)],
        )
        self.assertEqual(server.calculator.config.companion_offset_lamD, (2.5, -1.25))
        self.assertEqual(
            server.calculator.config.custom_speckle_offsets_lamD,
            ((1.0, 2.0), (-3.5, 0.5)),
        )

    def test_html_source_geometry_can_remove_all_speckles(self) -> None:
        server = _SLMHtmlServer()
        payload = server.update_source_geometry(
            planet_x=4.0,
            planet_y=0.0,
            speckle_offsets=[],
        )
        self.assertFalse(
            any(marker["kind"] == "speckle" for marker in payload["source_markers"])
        )

        mask = np.zeros((server.calculator.n_fft, server.calculator.n_fft), dtype=bool)
        result = server.calculator.propagate(
            mask,
            include_star=False,
            include_planet=False,
            include_speckles=True,
        )

        self.assertEqual(result["speckle_offsets_lamD"], ())
        self.assertEqual(result["selected_speckle_indices"], ())
        self.assertNotIn("A_speckle", result["metrics"])


if __name__ == "__main__":
    unittest.main()
