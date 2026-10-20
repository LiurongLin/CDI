from __future__ import annotations

import json
import tempfile
import io
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from pathlib import Path

import numpy as np

from coronagraph.interactive_slm_gui import (
    SLMLyotConfig, SLMLyotResponseCalculator, save_slm_science_result, save_slm_science_comparison,
)
from coronagraph.html_slm_gui import HTML, _SLMHtmlServer, make_handler
from coronagraph.masks import NoPhaseMask


class ScienceIntensityTests(unittest.TestCase):
    def calculator(self, **overrides):
        settings = dict(pupil_pixels=12, focal_sampling=3.0, phase_mask=NoPhaseMask(),
                        companion_offset_lamD=(2.0, 0.5),
                        custom_speckle_offsets_lamD=((-2.0, -0.5),),
                        ghost_fraction=0.0, include_ghost=False)
        return SLMLyotResponseCalculator(SLMLyotConfig(**{**settings, **overrides}))

    def mask(self, calc):
        return (calc.x_lamD + 2)**2 + (calc.y_lamD + 0.5)**2 <= 0.8**2

    def test_means_match_stopped_physical_fields_with_fixed_normalization(self):
        calc = self.calculator()
        mask = self.mask(calc)
        original = mask.copy()
        sweep = calc.science_intensity_sweep(mask, radius_lamD=0.6, phase_steps=7,
            phase_start_rad=0.3, phase_span_rad=4.0, lyot_reference_scale=0.7)
        norm = sweep['normalization_peak']
        for i, phase in enumerate(sweep['phases']):
            propagated = calc.propagate(mask, phase_modulation_rad=phase,
                                       include_speckles=True, lyot_reference_scale=0.7)
            fields = propagated['fields']
            fft = calc._template._centered_fft2
            physical = (np.abs(fft(fields['coherent']['modulated'] * calc.lyot_stop))**2
                        + np.abs(fft(fields['planet']['modulated'] * calc.lyot_stop))**2) / norm
            for name, region in sweep['regions'].items():
                x, y = region['center_lamD']
                aperture = (calc.x_lamD-x)**2 + (calc.y_lamD-y)**2 <= 0.6**2
                self.assertAlmostEqual(sweep['mean_intensities'][name][i], physical[aperture].mean(), places=14)
            if i == 0:
                np.testing.assert_allclose(sweep['preview_intensity'], physical, atol=1e-16)
        np.testing.assert_array_equal(mask, original)
        for name, values in sweep['mean_intensities'].items():
            self.assertEqual(sweep['metrics'][name], dict(maximum=values.max(), minimum=values.min(),
                                                         peak_to_peak=values.max()-values.min()))
        self.assertGreater(sweep['metrics']['speckle']['peak_to_peak'], 1e-8)

    def test_zero_phase_retains_light_and_planet_halo_in_speckle_aperture(self):
        calc = self.calculator()
        mask = np.zeros((calc.n_fft, calc.n_fft), bool)
        sweep = calc.science_intensity_sweep(mask, phase_steps=4, include_star=False,
                                           include_speckles=False)
        # Planet-only light remains in BOTH apertures, without background removal.
        for values in sweep['mean_intensities'].values():
            self.assertTrue(np.all(values > 0))
            np.testing.assert_allclose(values, values[0], rtol=0, atol=1e-18)
        no_light = calc.science_intensity_sweep(mask, phase_steps=2, include_star=False,
                                               include_speckles=False, include_planet=False)
        np.testing.assert_array_equal(no_light['preview_intensity'], 0)
        self.assertEqual(sweep['normalization_peak'], no_light['normalization_peak'])

    def test_ghosts_and_coherence_match_simulator_physics(self):
        calc = self.calculator(perfect_coronagraph=False, include_ghost=True,
            include_companion_ghost=True, ghost_fraction=0.04, include_interference=True,
            ghost_coherence=0.6)
        mask = self.mask(calc)
        sweep = calc.science_intensity_sweep(mask, phase_steps=3, phase_start_rad=0.7)
        sim = calc._make_simulator(np.where(mask, 0.7, 0))
        norm = sweep['normalization_peak']
        star = sim._run_single_branch(normalization_peak=norm)
        def shifted(offset, amplitude):
            return sim._run_single_branch(normalization_peak=norm, source_amplitude=amplitude,
                focal_shift_pixels=tuple(calc.config.focal_shift_pixels[k] - offset[k]*calc.focal_sampling for k in (0, 1)))
        speckle = shifted((-2, -0.5), 0.1)
        planet = shifted((2, 0.5), 1/np.sqrt(500))
        field = star['coronagraphic_field'] + speckle['coronagraphic_field']
        ghost = star['ghost_field'] + speckle['ghost_field']
        expected = np.abs(field)**2 + np.abs(ghost)**2 + 1.2*np.real(field*np.conj(ghost)) + planet['final_psf_with_ghost']
        np.testing.assert_allclose(sweep['preview_intensity'], expected, atol=1e-16)
        incoherent_wrong = star['final_psf_with_ghost'] + speckle['final_psf_with_ghost'] + planet['final_psf_with_ghost']
        self.assertGreater(np.max(np.abs(expected-incoherent_wrong)), 1e-6)

    def test_both_planes_and_all_companion_apertures(self):
        calc = self.calculator(custom_speckle_offsets_lamD=((-2, -0.5), (0, 2)))
        mask = self.mask(calc)
        science = calc.science_intensity_sweep(mask, phase_steps=7, measure_all_companions=True)
        focal = calc.science_intensity_sweep(mask, phase_steps=7, measure_all_companions=True,
                                           measurement_plane="focal")
        self.assertEqual(set(science['regions']), {'speckle_1', 'speckle_2', 'planet'})
        self.assertEqual(science['normalization_peak'], focal['normalization_peak'])
        self.assertEqual(science['regions'], focal['regions'])
        for name, values in focal['mean_intensities'].items():
            # A common local phase changes no physical intensity at the SLM itself.
            np.testing.assert_allclose(values, values[0], atol=1e-16)
        self.assertGreater(science['metrics']['speckle_1']['peak_to_peak'], 1e-8)
        norm = focal['normalization_peak']
        sim = calc._make_simulator(np.where(mask, focal['phases'][0], 0))
        star = sim._run_single_branch(normalization_peak=norm)
        field = star['focal_plane_field_after_mask'].copy()
        for offset in calc._speckle_offsets():
            shift = tuple(calc.config.focal_shift_pixels[k] - offset[k]*calc.focal_sampling for k in (0, 1))
            field += sim._run_single_branch(focal_shift_pixels=shift, source_amplitude=0.1,
                normalization_peak=norm)['focal_plane_field_after_mask']
        shift = tuple(calc.config.focal_shift_pixels[k] - calc.config.companion_offset_lamD[k]*calc.focal_sampling for k in (0, 1))
        planet = sim._run_single_branch(focal_shift_pixels=shift, source_amplitude=1/np.sqrt(500),
                                       normalization_peak=norm)['focal_plane_field_after_mask']
        np.testing.assert_allclose(focal['preview_intensity'], np.abs(field)**2 + np.abs(planet)**2, atol=1e-16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'all.npz'
            save_slm_science_result(path, science)
            with np.load(path) as data:
                for name in science['regions']:
                    np.testing.assert_array_equal(data[f'{name}_mean_intensity'], science['mean_intensities'][name])
                    self.assertEqual(data[f'{name}_peak_to_peak'], science['metrics'][name]['peak_to_peak'])
        with self.assertRaisesRegex(ValueError, 'measurement_plane'):
            calc.science_intensity_sweep(mask, measurement_plane='unknown')

    def test_all_companions_mode_without_speckles_still_measures_planet(self):
        calc = self.calculator(custom_speckle_offsets_lamD=())
        result = calc.science_intensity_sweep(self.mask(calc), measure_all_companions=True, phase_steps=2)
        self.assertEqual(set(result['regions']), {'planet'})
        self.assertGreater(result['mean_intensities']['planet'][0], 0)

    def test_separate_science_masks_center_each_source_and_record_all_regions(self):
        calc = self.calculator(custom_speckle_offsets_lamD=((-2, -0.5), (0, 2)))
        template = self.mask(calc)  # deliberately off-axis, not at the grid center
        original = template.copy()
        result = calc.science_comparison_sweeps(template, phase_steps=5, radius_lamD=0.6,
            lyot_reference_scale=0.8)
        self.assertEqual(set(result['sweeps']), {'speckle_1', 'speckle_2', 'planet'})
        np.testing.assert_array_equal(template, original)
        first = next(iter(result['sweeps'].values()))
        for name, run in result['sweeps'].items():
            self.assertEqual(run['regions'], first['regions'])
            self.assertEqual(run['normalization_peak'], first['normalization_peak'])
            np.testing.assert_array_equal(run['phases'], first['phases'])
            self.assertEqual(np.count_nonzero(run['slm_region_mask']), np.count_nonzero(template))
            target_x, target_y = calc.source_pixel(tuple(run['regions'][name]['center_lamD']))
            yy, xx = np.nonzero(run['slm_region_mask'])
            self.assertLessEqual(abs(xx.mean()-target_x), 0.5)
            self.assertLessEqual(abs(yy.mean()-target_y), 0.5)
            self.assertTrue(run['source_state']['include_star'])
            self.assertEqual(len(run['source_state']['selected_speckle_indices']), 2)
            direct = calc.science_intensity_sweep(run['slm_region_mask'], phase_steps=5,
                radius_lamD=0.6, measure_all_companions=True, lyot_reference_scale=0.8)
            for region, values in run['mean_intensities'].items():
                np.testing.assert_allclose(values, direct['mean_intensities'][region], atol=1e-16)
                self.assertEqual(result['modulation_matrix'][name][region], run['metrics'][region]['peak_to_peak'])
            self.assertEqual(result['own_region_metrics'][name], run['metrics'][name])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'comparison.npz'
            save_slm_science_comparison(path, result)
            with np.load(path, allow_pickle=False) as data:
                for name, run in result['sweeps'].items():
                    for region in run['regions']:
                        np.testing.assert_array_equal(data[f'{name}__{region}_mean_intensity'], run['mean_intensities'][region])
                        self.assertEqual(data[f'{name}__{region}_peak_to_peak'], run['metrics'][region]['peak_to_peak'])
                metadata = json.loads(str(data['comparison_metadata_json']))
                self.assertEqual(metadata['own_region_metrics'], result['own_region_metrics'])

    def test_shared_science_comparison_scans_original_mask_once(self):
        calc = self.calculator()
        template = self.mask(calc)
        with patch.object(calc, 'place_mask_template', side_effect=AssertionError('Shared mask moved')), patch.object(
            calc, 'science_intensity_sweep', wraps=calc.science_intensity_sweep) as scan:
            result = calc.science_comparison_sweeps(template, comparison_type='shared_mask', phase_steps=3)
        self.assertEqual(scan.call_count, 1)
        self.assertEqual(set(result['sweeps']), {'shared_mask'})
        np.testing.assert_array_equal(result['sweeps']['shared_mask']['slm_region_mask'], template)
        for name in result['source_names']:
            self.assertEqual(result['own_region_metrics'][name], result['sweeps']['shared_mask']['metrics'][name])

    def test_science_comparison_http_payload_and_all_placement_exports(self):
        server = _SLMHtmlServer()
        server.calculator = self.calculator(custom_speckle_offsets_lamD=((-2, -0.5), (0, 2)))
        def post(path, payload):
            raw = json.dumps(payload).encode()
            handler = SimpleNamespace(path=path, headers={"Content-Length": str(len(raw))}, rfile=io.BytesIO(raw))
            with patch('coronagraph.html_slm_gui._json_response') as response:
                make_handler(server).do_POST(handler)
                return response.call_args.args[1:]
        settings = dict(mask_b64=server.encode_mask(self.mask(server.calculator)), phase_steps=3,
                        radius_lamD=0.6, comparison_type='separate', measure_all_companions=True,
                        match_phase_zero_power=True)
        status, payload = post('/api/science-intensity', settings)
        self.assertEqual(status, 200)
        self.assertEqual(len(payload['sweeps']), 3)
        self.assertAlmostEqual(payload['phase_zero_match']['speckle_phase_zero_mean_intensity'],
                               payload['phase_zero_match']['planet_phase_zero_mean_intensity'], places=12)
        for run in payload['sweeps'].values():
            self.assertAlmostEqual(run['mean_intensities']['speckle_1'][0], run['mean_intensities']['planet'][0], places=12)
        self.assertEqual(len(set(run['display_max'] for run in payload['sweeps'].values())), 1)
        for name, encoded in payload['placed_masks'].items():
            np.testing.assert_array_equal(server.decode_mask(encoded), server.last_science_comparison['sweeps'][name]['slm_region_mask'])
            self.assertAlmostEqual(payload['sweeps'][name]['source_state']['star_speckle_ratio'], payload['phase_zero_match']['star_speckle_ratio'])
        with tempfile.TemporaryDirectory() as directory:
            server.log_root = Path(directory)
            status, saved = post('/api/save-result', {'process_kind': 'science_comparison'})
            self.assertEqual(status, 200)
            with np.load(saved['path']) as data:
                self.assertIn('planet__planet_mean_intensity', data.files)
                self.assertIn('speckle_2__planet_mean_intensity', data.files)
                match = json.loads(str(data['planet__phase_zero_match_json']))
                self.assertEqual(match['target_region'], 'speckle_1')
            log_path = server.save_log_payload({'process_kind': 'science_comparison', 'science_comparison': payload})
            log = json.loads(log_path.read_text())['science_comparison']
            self.assertEqual(set(log['sweeps']), set(payload['sweeps']))
            self.assertNotIn('placed_masks', log)
            for run in log['sweeps'].values():
                self.assertNotIn('image', run)
                self.assertEqual(set(run['mean_intensities']), {'speckle_1', 'speckle_2', 'planet'})
        status, planet_only = post('/api/science-intensity', {**settings, 'scan_mode': 'planet'})
        self.assertEqual(status, 200)
        self.assertEqual(set(planet_only['sweeps']), {'planet'})
        self.assertEqual(len(planet_only['sweeps']['planet']['mean_intensities']), 3)

    def test_science_phase_zero_match_uses_total_means_and_one_fixed_ratio(self):
        calc = self.calculator(custom_speckle_offsets_lamD=((-2, -0.5), (0, 2)),
            include_ghost=True, include_companion_ghost=True, ghost_fraction=0.01,
            include_interference=True, ghost_coherence=0.6)
        settings = dict(selected_speckle_index=1, selected_speckle_indices=[0, 1],
                        radius_lamD=0.6, star_planet_ratio=700, star_speckle_ratio=100,
                        lyot_reference_scale=0.8)
        match = calc.calibrate_science_phase_zero_intensity(**settings)
        self.assertEqual(match['target_region'], 'speckle_2')
        self.assertGreater(match['star_speckle_ratio'], 0)
        self.assertAlmostEqual(match['speckle_phase_zero_mean_intensity'], match['planet_phase_zero_mean_intensity'], places=12)
        comparison = calc.science_comparison_sweeps(self.mask(calc), phase_steps=5,
            radius_lamD=0.6, star_planet_ratio=700, lyot_reference_scale=0.8,
            star_speckle_ratio=match['star_speckle_ratio'])
        for run in comparison['sweeps'].values():
            self.assertEqual(run['source_state']['star_speckle_ratio'], match['star_speckle_ratio'])
            self.assertAlmostEqual(run['mean_intensities']['speckle_2'][0], run['mean_intensities']['planet'][0], places=12)
            self.assertGreater(run['mean_intensities']['planet'][0], 0)
        self.assertGreater(comparison['own_region_metrics']['planet']['peak_to_peak'], 1e-8)

    def test_science_matching_requires_enabled_target_and_planet(self):
        calc = self.calculator()
        for settings in ({'include_planet':False}, {'include_speckles':False},
                         {'selected_speckle_indices':[]}, {'selected_speckle_index':9}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                calc.calibrate_science_phase_zero_intensity(**settings)
        # Coincident circles already have equal total means at any source amplitude.
        coincident = self.calculator(custom_speckle_offsets_lamD=((2.0, 0.5),))
        match = coincident.calibrate_science_phase_zero_intensity(star_speckle_ratio=120)
        self.assertEqual(match['star_speckle_ratio'], 120)

    def test_science_optimizer_scores_only_s1_planet_with_all_enabled_light(self):
        calc = self.calculator(custom_speckle_offsets_lamD=((-2, -0.5), (0, 2)))
        template = self.mask(calc)
        original = template.copy()
        for comparison_type in ('separate', 'shared_mask'):
            with self.subTest(comparison_type=comparison_type), patch.object(
                calc, 'comparison_sweeps', side_effect=AssertionError('Lyot scorer used for science')):
                result = calc.optimize_common_mask(template, measurement_plane='science',
                    comparison_type=comparison_type, optimization_mode='circle_size',
                    circle_center_pixel=(18, 18), circle_radius_min_px=1, circle_radius_max_px=3,
                    iterations=3, phase_steps=4, science_radius_lamD=0.6,
                    selected_speckle_indices=[0,1], selected_speckle_index=1)
            self.assertEqual(result['scored_regions'], ['speckle_1','planet'])
            self.assertEqual(len(result['history']), 3)
            self.assertEqual(result['best_index'], int(np.argmax([item['score'] for item in result['history']])))
            norms = []
            for item in result['history']:
                comp = item['comparison']
                self.assertEqual(set(comp['own_region_metrics']), {'speckle_1','planet'})
                self.assertEqual(len(comp['sweeps']), 2 if comparison_type=='separate' else 1)
                self.assertEqual(item['speckle_pp'], comp['own_region_metrics']['speckle_1']['peak_to_peak'])
                self.assertEqual(item['planet_pp'], comp['own_region_metrics']['planet']['peak_to_peak'])
                self.assertAlmostEqual(item['score'], item['speckle_pp']/item['planet_pp'])
                for run in comp['sweeps'].values():
                    self.assertEqual(set(run['mean_intensities']), {'speckle_1','planet'})
                    self.assertEqual(run['source_state']['selected_speckle_indices'], [0,1])
                    self.assertTrue(run['source_state']['include_star'])
                    self.assertEqual(run['regions']['speckle_1']['radius_lamD'], 0.6)
                    self.assertEqual(run['regions']['planet']['radius_lamD'], 0.6)
                    norms.append(run['normalization_peak'])
                    # Disabling s2 changes the physical intensity, despite s2 not being scored.
                    isolated = calc.science_intensity_sweep(run['slm_region_mask'], measure_all_companions=True,
                        measurement_region_names=['speckle_1','planet'], phase_steps=4,
                        radius_lamD=0.6, selected_speckle_indices=[0])
                    self.assertFalse(np.allclose(run['mean_intensities']['speckle_1'],
                                                 isolated['mean_intensities']['speckle_1'], rtol=1e-10, atol=1e-16))
            self.assertEqual(len(set(norms)), 1)
        np.testing.assert_array_equal(template, original)
        with self.assertRaisesRegex(ValueError, 's1'):
            calc.optimize_common_mask(template, measurement_plane='science', selected_speckle_indices=[1])
        stopped = calc.optimize_common_mask(template, measurement_plane='science', stop_requested=lambda:True)
        self.assertIsNone(stopped['best_index'])

    def test_science_optimization_api_matching_payload_and_exports(self):
        server = _SLMHtmlServer()
        server.calculator = self.calculator(custom_speckle_offsets_lamD=((-2,-0.5),(0,2)))
        def post(path, payload):
            raw=json.dumps(payload).encode()
            handler=SimpleNamespace(path=path,headers={'Content-Length':str(len(raw))},rfile=io.BytesIO(raw))
            with patch('coronagraph.html_slm_gui._json_response') as response:
                make_handler(server).do_POST(handler)
                return response.call_args.args[1:]
        settings=dict(mask_b64=server.encode_mask(self.mask(server.calculator)), measurement_plane='science',
            selected_speckle_index=1, selected_speckle_indices=[0,1], comparison_type='shared_mask',
            optimization_mode='circle_size', circle_center_x=18, circle_center_y=18,
            circle_radius_min_px=1, circle_radius_max_px=3, iterations=3, phase_steps=4,
            radius_lamD=0.6, match_phase_zero_power=True)
        with patch.object(server.calculator,'calibrate_science_phase_zero_intensity',
                          wraps=server.calculator.calibrate_science_phase_zero_intensity) as calibration:
            status,payload=post('/api/optimize',settings)
        self.assertEqual(status,200)
        self.assertEqual(calibration.call_count,1)
        self.assertEqual(payload['measurement_plane'],'science')
        self.assertEqual(payload['phase_zero_match']['target_region'],'speckle_1')
        json.dumps(payload, allow_nan=False)
        ratios=set()
        for item in server.last_optimization['history']:
            for run in item['comparison']['sweeps'].values():
                ratios.add(run['source_state']['star_speckle_ratio'])
                self.assertAlmostEqual(run['mean_intensities']['speckle_1'][0],run['mean_intensities']['planet'][0],places=12)
        self.assertEqual(len(ratios),1)
        self.assertEqual(set(payload['best_comparison']['own_region_metrics']), {'speckle_1','planet'})
        with tempfile.TemporaryDirectory() as directory:
            server.log_root=Path(directory)
            status,saved=post('/api/save-result',{'process_kind':'science_optimization'})
            self.assertEqual(status,200)
            with np.load(saved['path'],allow_pickle=False) as data:
                self.assertIn('shared_mask__speckle_1_mean_intensity',data.files)
                self.assertIn('shared_mask__planet_mean_intensity',data.files)
                opt=json.loads(str(data['optimization_json']))
                self.assertEqual(opt['best_index'],payload['best_index'])
                self.assertEqual(len(opt['history']),3)
            log=server.concise_physics_log_payload({'process_kind':'science_optimization','optimization':payload,
                'science_comparison':payload['best_comparison']})
            self.assertEqual(log['optimization']['scored_regions'], ['speckle_1','planet'])
            self.assertEqual(log['optimization']['measurement_plane'],'science')
        # Pixel candidates include an empty reference: undefined scores must still be valid JSON.
        status,pixel_payload=post('/api/optimize',{**settings,'match_phase_zero_power':False,
                                                 'optimization_mode':'pixel_groups','iterations':1,'search_radius_px':2})
        self.assertEqual(status,200)
        json.dumps(pixel_payload,allow_nan=False)

    def test_geometry_validation(self):
        calc = self.calculator()
        for radius in (0, -1, float('nan'), 0.001, 10):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                calc.science_intensity_sweep(self.mask(calc), radius_lamD=radius)
        with self.assertRaises(ValueError):
            calc.science_intensity_sweep(self.mask(calc), selected_speckle_index=3)

    def test_json_and_npz_exports_preserve_geometry_curves_metrics(self):
        calc = self.calculator()
        result = calc.science_intensity_sweep(self.mask(calc), phase_steps=5, radius_lamD=0.6)
        server = _SLMHtmlServer()
        server.calculator = calc
        payload = server.science_intensity_payload(result)
        json.dumps(payload, allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'science.npz'
            save_slm_science_result(path, result)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['phases_rad'], result['phases'])
                for name in ('speckle', 'planet'):
                    np.testing.assert_array_equal(saved[f'{name}_mean_intensity'], result['mean_intensities'][name])
                    np.testing.assert_array_equal(saved[f'{name}_center_lamD'], result['regions'][name]['center_lamD'])
                    self.assertEqual(saved[f'{name}_radius_lamD'], 0.6)
                    self.assertEqual(saved[f'{name}_peak_to_peak'], result['metrics'][name]['peak_to_peak'])
                self.assertEqual(json.loads(str(saved['source_state_json']))['target_speckle_index'], 0)
            server.log_root = Path(directory)
            log_path = server.save_log_payload({'process_kind': 'science_intensity', 'science_intensity': payload})
            log = json.loads(log_path.read_text())['science_intensity']
            self.assertNotIn('image', log)
            self.assertEqual(log['regions'], payload['regions'])
            self.assertEqual(log['mean_intensities'], payload['mean_intensities'])
            self.assertEqual(log['metrics'], payload['metrics'])
            self.assertEqual(log['normalization_peak'], payload['normalization_peak'])

    def test_http_scan_and_save_result_routes(self):
        server = _SLMHtmlServer()
        server.calculator = self.calculator()
        mask = self.mask(server.calculator)
        def post(path, payload):
            raw = json.dumps(payload).encode()
            handler = SimpleNamespace(path=path, headers={"Content-Length": str(len(raw))},
                                      rfile=io.BytesIO(raw))
            with patch('coronagraph.html_slm_gui._json_response') as response:
                make_handler(server).do_POST(handler)
                return response.call_args.args[1:]
        status, payload = post('/api/science-intensity', dict(mask_b64=server.encode_mask(mask),
            radius_lamD=0.6, phase_steps=3, phase_start_rad=0.2, phase_span_rad=3.0,
            include_star=False, star_planet_ratio=700, star_speckle_ratio=90))
        self.assertEqual(status, 200)
        self.assertEqual(len(payload['phases']), 3)
        self.assertFalse(payload['source_state']['include_star'])
        self.assertEqual(payload['source_state']['star_planet_ratio'], 700)
        self.assertEqual(server.progress['active_run'], 'science_intensity')
        with tempfile.TemporaryDirectory() as directory:
            server.log_root = Path(directory)
            status, saved = post('/api/save-result', {'process_kind': 'science_intensity'})
            self.assertEqual(status, 200)
            with np.load(saved['path']) as data:
                np.testing.assert_allclose(data['speckle_mean_intensity'], payload['mean_intensities']['speckle'])
        status, error = post('/api/science-intensity', dict(mask_b64=server.encode_mask(mask), radius_lamD=-1))
        self.assertEqual(status, 400)
        self.assertIn('radius', error['error'])

    @unittest.skipUnless(shutil.which("node"), "Node.js required for GUI checks")
    def test_science_gui_render_request_and_export(self):
        self.assertLess(HTML.index('<section id="optimizationResults"'), HTML.index('<section id="scienceResults"'))
        script = HTML.split("<script>", 1)[1].split("</script>", 1)[0].replace("    init();", "")
        harness = r"""
const assert = require('node:assert/strict');
const elements = new Map(), labels = [];
const context = new Proxy({
  createImageData: (w,h) => ({data:new Uint8ClampedArray(w*h*4)}),
  fillText: text => labels.push(text), measureText: text => ({width:text.length*7}),
}, {get: (t,k) => t[k] || (() => {})});
function element(id) { return {id, checked:true, disabled:false, children:[], style:{}, open:false,
  showModal(){this.open=true}, close(){this.open=false},
  closest(){return this.parentElement}, parentElement:{after(){}}, append(...c){this.children.push(...c)},
  appendChild(child){this.children.push(child)}, replaceChildren(){this.children=[]}, getContext:()=>context,
  getBoundingClientRect:()=>({width:600,height:400}),
  toDataURL:()=> 'data:image/png;base64,aA==',
}; }
global.document={getElementById(id){if(!elements.has(id))elements.set(id,element(id));return elements.get(id)},
  createElement: tag=>element(tag), querySelector(selector){return this.getElementById(selector)},
  querySelectorAll:()=>[]};
global.window={devicePixelRatio:1};
"""
        checks = r"""
(async () => {
  state.config = {n_fft:2, focal_sampling:1, source_markers:[]}; state.mask = new Uint8Array(4);
  const result = {image:{shape:[2,2], values:[0.01,0.02,0.03,0.04], extent:[-3,3,-3,3]},
    display_max:0.1, preview_phase_rad:0.3, phase_start_rad:0.3, phase_span_rad:4,
    phases:[0.3,1.3,2.3,3.3], mean_intensities:{speckle:[0.01,0.02,0.03,0.02], planet:[0.02,0.02,0.02,0.02]},
    regions:{speckle:{center_lamD:[-2,0],radius_lamD:0.5}, planet:{center_lamD:[2,0],radius_lamD:0.5}},
    metrics:{speckle:{maximum:0.03,minimum:0.01,peak_to_peak:0.02},planet:{maximum:0.02,minimum:0.02,peak_to_peak:0}}};
  state.lastScience = result; els.modulationPlane.value = 'science'; document.getElementById('displayMode').value = 'phase';
  const original = JSON.stringify(result);
  drawScienceResults();
  assert.equal(JSON.stringify(result), original, 'Rendering never changes measured intensities');
  assert(labels.includes('speckle') && labels.includes('planet'));
  assert(labels.includes('0.300') && labels.includes('4.300'), 'Plot respects arbitrary start/span');
  assert.equal(document.getElementById('scienceMetrics').children.length,6);
  updateResultVisibility(); assert.equal(document.getElementById('scienceResults').hidden,false);
  const figures = figurePayload(); assert('science_intensity_image' in figures && 'science_intensity_curves' in figures);
  assert.equal(currentLogPayload().science_intensity,result);
  const originalShowResults = showResults;
  state.lastScience = null; setBusy=()=>{}; showResults=()=>{};
  els.scienceRadius.value='0.7'; els.targetSpeckle.value='2'; els.referenceSubtractPercent.value='80';
  els.phaseSteps.value='8'; els.phaseStartRad.value='0.4'; els.phaseSpanRad.value='5';
  let request;
  global.fetch = async (url, options)=>{assert.equal(url,'/api/science-intensity'); request=JSON.parse(options.body); return {ok:true,json:async()=>result};};
  await measureScienceIntensity();
  assert.equal(request.radius_lamD,0.7); assert.equal(request.measure_all_companions,true); assert.equal(request.match_phase_zero_power,true);
  assert.equal(request.lyot_reference_scale,0.8); assert.equal(request.phase_steps,8);
  assert.equal(state.lastScience,result); assert.equal(state.lastProcess,'science_intensity');
  assert.deepEqual(Array.from(state.mask),[0,0,0,0], 'Science scan never alters the SLM mask');
  state.lastResult=null; downloadText=()=>{};
  global.fetch = async (url,options)=>{assert.equal(url,'/api/save-result'); assert.equal(JSON.parse(options.body).process_kind,'science_intensity');return {ok:true,json:async()=>({path:'science.npz'})};};
  await saveResult(); assert(els.status.textContent.includes('science.npz'));

  const speckleRun = {...result,
    phase_zero_match:{target_region:'speckle_1',speckle_phase_zero_mean_intensity:0.02,planet_phase_zero_mean_intensity:0.02,star_speckle_ratio:500},
    mean_intensities:{speckle_1:result.mean_intensities.speckle,planet:[0.02,0.02,0.02,0.02]},
    regions:{speckle_1:result.regions.speckle,planet:result.regions.planet},
    metrics:{speckle_1:result.metrics.speckle,planet:result.metrics.planet}};
  const planetRun = {...speckleRun,
    mean_intensities:{speckle_1:[0.01,0.01,0.01,0.01],planet:[0.02,0.04,0.03,0.01]}};
  const comparison = {comparison_type:'separate', sweeps:{speckle_1:speckleRun,planet:planetRun},
    placed_masks:{speckle_1:'AAAAAA==',planet:'AQAAAA=='},
    own_region_metrics:{speckle_1:result.metrics.speckle,planet:result.metrics.planet}};
  const drawSlmOriginal = redrawSlm; redrawSlm=()=>{};
  els.comparisonType.value='separate'; els.modulationPlane.value='science';
  global.fetch=async (url,options)=>{request=JSON.parse(options.body);return {ok:true,json:async()=>comparison}};
  await runComparison('both');
  assert.equal(request.comparison_type,'separate'); assert.equal(request.scan_mode,'both');
  assert.equal(state.lastProcess,'science_comparison'); assert.equal(state.lastScienceComparison,comparison);
  assert.equal(document.getElementById('sciencePhaseZeroMatch').hidden,false);
  assert(document.getElementById('sciencePhaseZeroMatch').textContent.includes('Star/Speckle'));
  assert(document.getElementById('sciencePhaseZeroMatch').textContent.includes('fixed for every phase and placement'));
  assert.equal(els.sciencePlacement.children.length,2);
  const overview = scienceOverviewData(comparison);
  assert.deepEqual(overview.mean_intensities.planet,[0.02,0.04,0.03,0.01],
    'Overview must use the planet-centered scan, not planet light under the speckle-centered mask');
  assert.deepEqual(overview.mean_intensities.speckle_1,result.mean_intensities.speckle);
  assert.equal(state.lastScience,speckleRun,'Detailed view may start on the speckle while overview includes both own-source scans');
  assert('science_source_comparison' in figurePayload(),'Overview can be saved without selecting a placement');
  assert(document.getElementById('scienceOverviewNote').textContent.includes('planet-centered mask'));
  assert(document.getElementById('scienceSelectedPlacement').textContent.includes('speckle_1'));
  const sharedOverview=scienceOverviewData({comparison_type:'shared_mask',sweeps:{shared_mask:speckleRun},own_region_metrics:comparison.own_region_metrics});
  assert.deepEqual(sharedOverview.mean_intensities.planet,[0.02,0.02,0.02,0.02]);
  assert.equal(currentLogPayload().science_comparison,comparison);
  assert.equal(document.getElementById('scienceComparisonSummary').children[0].children.length,3);
  els.sciencePlacement.value='planet'; selectSciencePlacement();
  assert.deepEqual(Array.from(displayedSlmMasks()[0].mask),[1,0,0,0]);
  els.comparisonType.value='shared_mask'; await measureScienceIntensity();
  assert.equal(request.comparison_type,'shared_mask');
  global.fetch=async (url,options)=>{assert.equal(url,'/api/save-result');assert.equal(JSON.parse(options.body).process_kind,'science_comparison');return {ok:true,json:async()=>({path:'comparison.npz'})}};
  await saveResult(); assert(els.status.textContent.includes('comparison.npz'));

  // Use the real dialog/rendering path for science optimization, including every plot.
  showResults = originalShowResults;
  // Final-science optimization keeps its winner/plots on the science plane and scores only s1/planet.
  const sciOpt={measurement_plane:'science',optimization_mode:'circle_size',scored_regions:['speckle_1','planet'],
    best_index:0,best_mask_b64:'AQAAAA==',best_comparison:comparison,pareto_indices:[0],
    history:[{name:'circle_radius_1px',radius_px:1,score:2,speckle_pp:0.02,planet_pp:0.01,ratio_pp:2}]};
  els.modulationPlane.value='science';els.tool.value='circle';els.targetSpeckle.value='1';
  selectedSpeckleIndices=()=>[0,1];
  global.fetch=async(url,options)=>{assert.equal(url,'/api/optimize');request=JSON.parse(options.body);els.modulationPlane.value='lyot';return {ok:true,json:async()=>sciOpt}};
  els.resultsDialog.open=true; els.resultsDialog.scrollTop=800;
  await optimizeMask();
  assert.equal(els.modulationPlane.value,'science','Completed optimization displays the plane that was actually requested');
  assert.equal(els.resultsDialog.open,true);
  assert.equal(els.resultsDialog.scrollTop,0,'Completed optimization opens at the summary, not a previous scroll position');
  assert.equal(document.getElementById('optimizationTitle').textContent,'Final science optimization — s1 vs planet');
  assert(labels.includes('s1 intensity PP (higher preferred)'),'Actual science trade-off plot was rendered');
  assert(labels.includes('Peak-to-peak mean intensity'),'Actual science radius plot was rendered');
  assert(document.getElementById('scienceOptimizationMetrics').children.some(child=>child.innerHTML?.includes('s1 intensity PP')));
  assert.equal(request.measurement_plane,'science');assert.equal(request.radius_lamD,0.7);
  assert.deepEqual(request.selected_speckle_indices,[0,1]);
  assert.equal(state.lastProcess,'science_optimization');assert.equal(selectedResultProcess(),'science_optimization');
  assert.equal(state.lastScienceComparison,comparison);
  updateResultVisibility();
  assert.equal(document.getElementById('optimizationResults').hidden,false);
  assert.equal(document.getElementById('scienceOptimizationMetrics').hidden,false);
  assert('optimization_radius_plot' in figurePayload());
  assert.equal(currentLogPayload().optimization,sciOpt);
  global.fetch=async(url,options)=>{assert.equal(url,'/api/save-result');assert.equal(JSON.parse(options.body).process_kind,'science_optimization');return {ok:true,json:async()=>({path:'science_opt.npz'})}};
  await saveResult();assert(els.status.textContent.includes('science_opt.npz'));
  els.modulationPlane.value='lyot';updateResultVisibility();
  assert.equal(document.getElementById('optimizationResults').hidden,true,'Science optimization stays hidden on Lyot');
  els.modulationPlane.value='science';
  // A stale server must produce an actionable error rather than hiding a Lyot optimization on science.
  const cachedOptimization=state.lastOptimization;
  global.fetch=async()=>({ok:true,json:async()=>({...sciOpt,measurement_plane:'lyot'})});
  await optimizeMask();
  assert(els.status.textContent.includes('Restart the Python GUI server'));
  assert.equal(state.lastOptimization,cachedOptimization);
  showResults=()=>{}; els.resultsDialog.close();
  // Results from both planes remain cached, but only the selected plane is visible/exported.
  const scienceCached = state.lastScienceComparison;
  state.lastResult = {maps:{planet:{}},metrics:{A_planet:1}};
  state.lastSweep = {}; state.lastComparison = {sweeps:{speckle_position:{}}};
  state.lastOptimization = {history:[{}]};
  els.modulationPlane.value='lyot'; refreshSelectedPlaneResults();
  assert.equal(document.getElementById('scienceResults').hidden,true);
  for (const id of ['mapResults','currentPhaseResults','comparisonResults','comparisonSummary','optimizationResults']) {
    assert.equal(document.getElementById(id).hidden,false, `${id} remains available for Lyot`);
  }
  assert.equal(els.metrics.hidden,false);
  assert.equal(Object.keys(figurePayload()).some(key=>key.startsWith('science_')),false);
  assert.equal(document.getElementById('resultsTitle').textContent,'Lyot-plane results');
  const lyotLog=currentLogPayload();
  assert.equal(lyotLog.science_intensity,null); assert.equal(lyotLog.science_comparison,null);
  assert.equal(lyotLog.comparison,state.lastComparison);
  assert.equal(lyotLog.process_kind,'comparison');
  assert.equal(state.lastScienceComparison,scienceCached,'Switching planes preserves previous data');
  els.modulationPlane.value='science'; refreshSelectedPlaneResults();
  assert.equal(document.getElementById('scienceResults').hidden,false);
  for (const id of ['mapResults','currentPhaseResults','comparisonResults','comparisonSummary','optimizationResults']) {
    assert.equal(document.getElementById(id).hidden,true, `${id} is hidden for science`);
  }
  assert.equal(els.metrics.hidden,true);
  assert.equal(Object.keys(figurePayload()).every(key=>key==='slm_mask'||key.startsWith('science_')),true);
  assert.equal(currentLogPayload().phase_sweep,null); assert.equal(currentLogPayload().comparison,null);
  state.lastResult=null; state.lastSweep=null; state.lastComparison=null; state.lastOptimization=null;
  els.modulationPlane.value='lyot'; refreshSelectedPlaneResults();
  assert.equal(document.getElementById('planeResultsEmpty').hidden,false);
  assert.equal(document.getElementById('scienceResults').hidden,true,'No science fallback when Lyot has no results');
  let scienceDraws=0, lyotDraws=0;
  drawScienceResults=()=>scienceDraws++; redrawMaps=()=>lyotDraws++;
  drawPhasePlot=()=>{};drawPhasePlotOn=()=>{};drawRadiusPlot=()=>{};drawRingPlot=()=>{};drawTradeoffPlot=()=>{};
  redrawResults(); assert.equal(scienceDraws,0); assert.equal(lyotDraws,1);
  els.modulationPlane.value='science'; redrawResults(); assert.equal(scienceDraws,1); assert.equal(lyotDraws,1);
  renderSweepMetrics=()=>{};
  els.matchPhaseZeroPower.checked=true; els.modulationPlane.value='lyot';
  global.fetch=async (url,options)=>{assert.equal(url,'/api/phase-sweep');request=JSON.parse(options.body);return {ok:true,json:async()=>({mask_fraction:0.1})}};
  await evaluateModulation(); assert.equal(request.match_phase_zero_power,true,'Lyot Measure modulation sends matching checkbox');
  els.matchPhaseZeroPower.checked=false;
  await evaluateModulation(); assert.equal(request.match_phase_zero_power,false);
  let routed = '';
  evaluateModulation = async () => {routed = 'lyot'};
  measureScienceIntensity = async () => {routed = 'science'};
  els.modulationPlane.value = 'lyot'; await measureSelectedModulation(); assert.equal(routed,'lyot');
  els.modulationPlane.value = 'science'; await measureSelectedModulation(); assert.equal(routed,'science');

})().catch(e=>{console.error(e);process.exitCode=1;});
"""
        result = subprocess.run([shutil.which('node'), '-'], input=harness+script+checks,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
