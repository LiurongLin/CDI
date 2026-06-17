from __future__ import annotations

import unittest

import numpy as np

from coronagraph.plotting import _coc_build_incoherence_maps


class CocIncoherenceMapModeTests(unittest.TestCase):
    def test_lab_fft_ratio_uses_inverse_of_selected_peak_ratio(self) -> None:
        n_phase = 8
        phase = np.arange(n_phase, dtype=float)
        signal = 5.0 + 2.0 * np.cos(2.0 * np.pi * phase / n_phase)
        central_stack = signal[:, None, None]
        freq_bins = np.fft.fftfreq(n_phase, d=1.0)
        fft_cube = np.fft.fft(central_stack, axis=0)

        info = _coc_build_incoherence_maps(
            freq_bins=freq_bins,
            fft_cube=fft_cube,
            central_stack_fft=central_stack,
            mode="lab_fft_ratio",
        )

        self.assertAlmostEqual(float(info["selected_target_freq"]), 1.0 / n_phase)
        expected_coherence = np.abs(fft_cube[1, 0, 0]) / np.abs(fft_cube[0, 0, 0])
        self.assertAlmostEqual(float(info["coherence_map"][0, 0]), float(expected_coherence))
        self.assertAlmostEqual(
            float(info["incoherence_map"][0, 0]),
            float(1.0 / expected_coherence),
        )

    def test_fft_band_mode_keeps_low_frequency_band_sum(self) -> None:
        n_phase = 64
        phase = np.arange(n_phase, dtype=float)
        signal = 3.0 + np.cos(2.0 * np.pi * 0.015625 * phase)
        central_stack = signal[:, None, None]
        freq_bins = np.fft.fftfreq(n_phase, d=1.0)
        fft_cube = np.fft.fft(central_stack, axis=0)

        info = _coc_build_incoherence_maps(
            freq_bins=freq_bins,
            fft_cube=fft_cube,
            central_stack_fft=central_stack,
            mode="fft_band",
        )

        band_mask = (np.abs(freq_bins) >= 0.0) & (np.abs(freq_bins) <= 0.02)
        expected = np.sum(np.abs(fft_cube[band_mask]), axis=0) / float(np.count_nonzero(band_mask))
        np.testing.assert_allclose(info["incoherence_map"], expected)

    def test_lab_fft_ratio_selects_frequency_from_planet_region_sum(self) -> None:
        n_phase = 8
        phase = np.arange(n_phase, dtype=float)
        stack = np.zeros((n_phase, 1, 2), dtype=float)
        stack[:, 0, 0] = 7.0 + 4.0 * np.cos(2.0 * np.pi * 2.0 * phase / n_phase)
        stack[:, 0, 1] = 50.0 + 20.0 * np.cos(2.0 * np.pi * 1.0 * phase / n_phase)
        freq_bins = np.fft.fftfreq(n_phase, d=1.0)
        fft_cube = np.fft.fft(stack, axis=0)

        info = _coc_build_incoherence_maps(
            freq_bins=freq_bins,
            fft_cube=fft_cube,
            central_stack_fft=stack,
            mode="lab_fft_ratio",
            planet_region_mask=np.array([[True, False]]),
        )

        self.assertAlmostEqual(float(info["selected_target_freq"]), 2.0 / n_phase)
        np.testing.assert_allclose(info["selection_phase_trace"], stack[:, 0, 0])
        self.assertAlmostEqual(float(info["selection_nonnegative_freqs"][0]), 0.0)


if __name__ == "__main__":
    unittest.main()
