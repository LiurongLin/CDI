from __future__ import annotations

import unittest

from coronagraph.gui import Runner


class GuiRunnerTests(unittest.TestCase):
    def test_build_cmd_uses_flux_ratio_map_sweep_flags(self) -> None:
        runner = Runner()
        payload = {
            "phase_mask_type": "vortex",
            "roddier_mask_radius": "0.53",
            "roddier_mask_phase": "3.141592653589793",
            "vortex_charge": "2",
            "spider_width": "0.25",
            "pupil_ss": "8",
            "phase_screen_jitter": "none",
            "incoherence_map_mode": "fft_band",
            "local_region_radius": "2.0",
            "phase_sweep_mode": "mask_rotation",
            "region_shape": "circle",
            "fov_count": "1",
            "fov_centers_count": "1",
            "ring_rotation_fraction": "0.0",
            "ring_rotation_sweep_max": "1.0",
            "ring_rotation_sweep_step": "0.1",
            "phase_step": "8",
            "phase_cycles": "1.0",
            "secondary_ratio_local": "0.25",
            "lyot_reference_percent": "100",
            "coherent_ring_speckles_enabled": "off",
            "planet_flux_ratio_local": "0.01",
            "coherent_ring_speckle_count": "0",
            "roi_size_min": "0.5",
            "roi_size_max": "3.0",
            "roi_size_step": "0.25",
            "planet_position_radius_min": "9.0",
            "planet_position_radius_max": "9.0",
            "planet_position_radius_step": "9.0",
            "planet_position_theta_min_deg": "9.0",
            "planet_position_theta_max_deg": "9.0",
            "planet_position_theta_step_deg": "9.0",
            "planet_position_radius_min_map": "1.5",
            "planet_position_radius_max_map": "2.5",
            "planet_position_radius_step_map": "0.5",
            "planet_position_theta_min_deg_map": "-45.0",
            "planet_position_theta_max_deg_map": "45.0",
            "planet_position_theta_step_deg_map": "15.0",
            "planet_flux_ratio_sweep_min": "0.001",
            "planet_flux_ratio_sweep_max": "0.005",
            "planet_flux_ratio_sweep_step": "0.002",
            "planet_offset_radius_local": "5.0",
            "planet_offset_theta_deg_local": "180.0",
            "spiders_enabled": "on",
            "enable_ring_of_circle_sweep": False,
            "enable_ring_rotation_sweep": False,
            "roi_size_sweep": False,
            "planet_position_roi_size_sweep": False,
            "planet_position_map_sweep": True,
            "planet_flux_ratio_map_sweep": True,
            "planet_position_brightness_sweep": False,
            "disable_ghost": False,
            "disable_interference": False,
            "disable_companion_ghost": False,
            "build_map_per_fov": False,
            "plot_poster_figure": False,
        }

        cmd = runner.build_cmd(payload)

        self.assertIn("--planet-position-map-sweep", cmd)
        self.assertIn("--planet-flux-ratio-map-sweep", cmd)
        self.assertEqual(cmd[cmd.index("--planet-position-radius-min") + 1], "1.5")
        self.assertEqual(cmd[cmd.index("--planet-position-radius-max") + 1], "2.5")
        self.assertEqual(cmd[cmd.index("--planet-position-theta-min-deg") + 1], "-45.0")
        self.assertEqual(cmd[cmd.index("--planet-position-theta-max-deg") + 1], "45.0")
        self.assertEqual(cmd[cmd.index("--planet-flux-ratio-sweep-min") + 1], "0.001")
        self.assertEqual(cmd[cmd.index("--planet-flux-ratio-sweep-max") + 1], "0.005")

    def test_build_cmd_uses_mask_rotation_phase_step_sweep_flags(self) -> None:
        runner = Runner()
        payload = {
            "phase_mask_type": "vortex",
            "roddier_mask_radius": "0.53",
            "roddier_mask_phase": "3.141592653589793",
            "vortex_charge": "2",
            "spider_width": "0.25",
            "pupil_ss": "8",
            "phase_screen_jitter": "none",
            "incoherence_map_mode": "fft_band",
            "local_region_radius": "2.0",
            "phase_sweep_mode": "mask_rotation",
            "region_shape": "circle",
            "fov_count": "1",
            "fov_centers_count": "1",
            "ring_rotation_fraction": "0.0",
            "ring_rotation_sweep_max": "1.0",
            "ring_rotation_sweep_step": "0.1",
            "phase_step": "8",
            "phase_cycles": "1.0",
            "secondary_ratio_local": "0.25",
            "lyot_reference_percent": "100",
            "coherent_ring_speckles_enabled": "off",
            "planet_flux_ratio_local": "0.01",
            "coherent_ring_speckle_count": "0",
            "roi_size_min": "0.5",
            "roi_size_max": "3.0",
            "roi_size_step": "0.25",
            "planet_position_radius_min": "1.5",
            "planet_position_radius_max": "2.5",
            "planet_position_radius_step": "0.5",
            "planet_position_theta_min_deg": "-45.0",
            "planet_position_theta_max_deg": "45.0",
            "planet_position_theta_step_deg": "15.0",
            "planet_flux_ratio_sweep_min": "0.001",
            "planet_flux_ratio_sweep_max": "0.005",
            "planet_flux_ratio_sweep_step": "0.002",
            "phase_step_sweep_min": "4",
            "phase_step_sweep_max": "16",
            "phase_step_sweep_step": "4",
            "planet_offset_radius_local": "5.0",
            "planet_offset_theta_deg_local": "180.0",
            "spiders_enabled": "on",
            "enable_ring_of_circle_sweep": False,
            "enable_ring_rotation_sweep": False,
            "roi_size_sweep": False,
            "planet_position_map_sweep": False,
            "planet_position_roi_size_sweep": False,
            "planet_flux_ratio_map_sweep": False,
            "mask_rotation_phase_step_sweep": True,
            "planet_position_brightness_sweep": False,
            "disable_ghost": False,
            "disable_interference": False,
            "disable_companion_ghost": False,
            "build_map_per_fov": False,
            "plot_poster_figure": False,
        }

        cmd = runner.build_cmd(payload)

        self.assertIn("--mask-rotation-phase-step-sweep", cmd)
        self.assertEqual(cmd[cmd.index("--phase-step-sweep-min") + 1], "4")
        self.assertEqual(cmd[cmd.index("--phase-step-sweep-max") + 1], "16")
        self.assertEqual(cmd[cmd.index("--phase-step-sweep-step") + 1], "4")

    def test_build_cmd_uses_roi_sweep_specific_planet_position_fields(self) -> None:
        runner = Runner()
        payload = {
            "phase_mask_type": "perfect_corongraph",
            "roddier_mask_radius": "0.53",
            "roddier_mask_phase": "3.141592653589793",
            "vortex_charge": "2",
            "spider_width": "0.25",
            "pupil_ss": "8",
            "phase_screen_jitter": "0",
            "incoherence_map_mode": "lab_fft_ratio",
            "local_region_radius": "1.5",
            "phase_sweep_mode": "regional",
            "region_shape": "ring",
            "fov_count": "1",
            "fov_centers_count": "10",
            "ring_rotation_fraction": "0.0",
            "ring_rotation_sweep_max": "1.0",
            "ring_rotation_sweep_step": "0.2",
            "phase_step": "10",
            "phase_cycles": "2",
            "secondary_ratio_local": "0.25",
            "lyot_reference_percent": "65",
            "coherent_ring_speckles_enabled": "on",
            "planet_flux_ratio_local": "0.005",
            "coherent_ring_speckle_count": "2",
            "roi_size_min": "7",
            "roi_size_max": "7",
            "roi_size_step": "0",
            "planet_position_radius_min": "3",
            "planet_position_radius_max": "10",
            "planet_position_radius_step": "1",
            "planet_position_theta_min_deg": "180",
            "planet_position_theta_max_deg": "180",
            "planet_position_theta_step_deg": "0",
            "planet_position_radius_min_roi": "5",
            "planet_position_radius_max_roi": "8",
            "planet_position_radius_step_roi": "1",
            "planet_position_theta_min_deg_roi": "180",
            "planet_position_theta_max_deg_roi": "180",
            "planet_position_theta_step_deg_roi": "0",
            "lyot_reference_percent_min_roi": "40",
            "lyot_reference_percent_max_roi": "80",
            "lyot_reference_percent_step_roi": "20",
            "planet_flux_ratio_sweep_min": "0.001",
            "planet_flux_ratio_sweep_max": "0.011",
            "planet_flux_ratio_sweep_step": "0.002",
            "phase_step_sweep_min": "4",
            "phase_step_sweep_max": "24",
            "phase_step_sweep_step": "5",
            "planet_offset_radius_local": "10.0",
            "planet_offset_theta_deg_local": "180.0",
            "spiders_enabled": "off",
            "enable_ring_of_circle_sweep": False,
            "enable_ring_rotation_sweep": False,
            "roi_size_sweep": False,
            "planet_position_map_sweep": False,
            "planet_position_roi_size_sweep": True,
            "planet_flux_ratio_map_sweep": False,
            "mask_rotation_phase_step_sweep": False,
            "planet_position_brightness_sweep": False,
            "disable_ghost": False,
            "disable_interference": False,
            "disable_companion_ghost": False,
            "build_map_per_fov": False,
            "plot_poster_figure": True,
        }

        cmd = runner.build_cmd(payload)

        self.assertIn("--planet-position-roi-size-sweep", cmd)
        self.assertEqual(cmd[cmd.index("--lyot-reference-percent") + 1], "65")
        self.assertEqual(cmd[cmd.index("--coherent-ring-speckle-count") + 1], "2")
        self.assertEqual(cmd[cmd.index("--lyot-reference-percent-sweep-min") + 1], "40")
        self.assertEqual(cmd[cmd.index("--lyot-reference-percent-sweep-max") + 1], "80")
        self.assertEqual(cmd[cmd.index("--lyot-reference-percent-sweep-step") + 1], "20")
        self.assertEqual(cmd[cmd.index("--planet-position-radius-min") + 1], "5")
        self.assertEqual(cmd[cmd.index("--planet-position-radius-max") + 1], "8")
        self.assertEqual(cmd[cmd.index("--planet-position-radius-step") + 1], "1")

    def test_build_cmd_forces_zero_coherent_ring_speckles_when_toggle_is_off(self) -> None:
        runner = Runner()
        payload = {
            "phase_mask_type": "vortex",
            "roddier_mask_radius": "0.53",
            "roddier_mask_phase": "3.141592653589793",
            "vortex_charge": "2",
            "spider_width": "0.25",
            "pupil_ss": "8",
            "phase_screen_jitter": "none",
            "incoherence_map_mode": "fft_band",
            "local_region_radius": "2.0",
            "phase_sweep_mode": "regional",
            "region_shape": "circle",
            "fov_count": "1",
            "fov_centers_count": "1",
            "ring_rotation_fraction": "0.0",
            "ring_rotation_sweep_max": "1.0",
            "ring_rotation_sweep_step": "0.1",
            "phase_step": "8",
            "phase_cycles": "1.0",
            "secondary_ratio_local": "0.25",
            "lyot_reference_percent": "100",
            "coherent_ring_speckles_enabled": "off",
            "planet_flux_ratio_local": "0.01",
            "coherent_ring_speckle_count": "7",
            "roi_size_min": "0.5",
            "roi_size_max": "3.0",
            "roi_size_step": "0.25",
            "planet_position_radius_min": "1.5",
            "planet_position_radius_max": "2.5",
            "planet_position_radius_step": "0.5",
            "planet_position_theta_min_deg": "-45.0",
            "planet_position_theta_max_deg": "45.0",
            "planet_position_theta_step_deg": "15.0",
            "planet_flux_ratio_sweep_min": "0.001",
            "planet_flux_ratio_sweep_max": "0.005",
            "planet_flux_ratio_sweep_step": "0.002",
            "phase_step_sweep_min": "4",
            "phase_step_sweep_max": "16",
            "phase_step_sweep_step": "4",
            "planet_offset_radius_local": "5.0",
            "planet_offset_theta_deg_local": "180.0",
            "spiders_enabled": "on",
            "enable_ring_of_circle_sweep": False,
            "enable_ring_rotation_sweep": False,
            "roi_size_sweep": False,
            "planet_position_map_sweep": False,
            "planet_position_roi_size_sweep": False,
            "planet_flux_ratio_map_sweep": False,
            "mask_rotation_phase_step_sweep": False,
            "planet_position_brightness_sweep": False,
            "disable_ghost": False,
            "disable_interference": False,
            "disable_companion_ghost": False,
            "build_map_per_fov": False,
            "plot_poster_figure": False,
        }

        cmd = runner.build_cmd(payload)

        self.assertEqual(cmd[cmd.index("--coherent-ring-speckle-count") + 1], "0")


if __name__ == "__main__":
    unittest.main()
