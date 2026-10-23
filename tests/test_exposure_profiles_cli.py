"""End-to-end continuous exposure replay on independent synthetic geometry."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import dataset as ds
import model_io
from test_height_field_cli import empty_cloud, read_ply


class ExposureCliTests(unittest.TestCase):
    @classmethod
    def cli(cls, *arguments, success=True):
        result = subprocess.run([sys.executable, str(HERE / 'bin_filter.py'), *map(str, arguments)],
                                text=True, capture_output=True, timeout=90)
        if (result.returncode == 0) != success or 'Traceback' in result.stderr:
            raise AssertionError(f'{arguments}\n{result.stdout}\n{result.stderr}')
        return result

    @classmethod
    def register(cls, capture_id, exposure, split, label, points, group):
        path = cls.base / f'{capture_id}.npy'
        np.save(path, points)
        return ds.add_capture(cls.root, capture_id, split, dict(bin_a=label, bin_b=label), exposure,
                              cloud_path=path, cloud_frame='robot', group_id=group)

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name)
        camera, robot = cls.base / 'camera.yaml', cls.base / 'robot.yaml'
        ds.write_yaml(camera, {name: {'rows': 2, 'cols': 2, 'cam_intrinsic': np.eye(3).ravel().tolist(),
                                     'cam_extrinsic': np.eye(4).ravel().tolist()} for name in ('cam1', 'cam2')})
        ds.write_yaml(robot, {'name': 'B2', 'hand_eye': [{'scene_id': 0, 'calibration': [0.] * 6}]})
        cls.root = cls.base / 'dataset'
        ds.init_dataset(cls.root, camera, robot)
        settings = {'bins': {}}
        for name, x in [('bin_a', 0), ('bin_b', 40)]:
            settings['bins'][name] = {'kind': 'height_field', 'bin_id': name,
                'xy_bounds_mm': [[x - 10, -10], [x + 10, 10]], 'z_bounds_mm': [-135, -85],
                'seed_xy_mm': [x, 0], 'max_reference_height_mm': -93, 'min_capture_support': 3,
                'min_tolerance_mm': .7, 'max_tolerance_mm': 3., 'alignment_margin_mm': .3}
        ds.write_yaml(cls.root / 'bin_config.yaml', settings)
        for exposure in (4., 20.):
            for index in range(4):
                cls.register(f'train_{exposure:g}_{index}', exposure, 'train', 'empty',
                             empty_cloud(10 * int(exposure) + index) + [0, 0, .08 * (exposure - 4)],
                             f'train_session_{index}')
        for index, exposure in enumerate((4., 6., 12.5, 18., 20.)):
            cls.register(f'test_{index}', exposure, 'test', 'empty',
                         empty_cloud(900 + index) + [0, 0, .08 * (exposure - 4)], 'empty_test_session')
        cls.probes = np.array([[0., 0, -99.32], [0., 0, -96.32], [3., 3, -90],
                              [40., 0, -99.32], [40., 0, -96.32], [43., 3, -90]])
        cls.register('parts', 12.5, 'test', 'loaded', cls.probes, 'parts_test_layout')
        cls.model_path = cls.base / 'profiles/model.json'
        cls.cli('calibrate', '--dataset', cls.root, '--output', cls.model_path.parent,
                '--exposure-profiles', '--interpolation-margin-mm', .2)
        cls.model = model_io.load_model(cls.model_path)
        cls.hash = ds.sha256(cls.model_path)
        output = cls.base / 'validation'
        cls.cli('validate', '--dataset', cls.root, '--model', cls.model_path, '--output', output)
        cls.report = json.loads((output / 'validation.json').read_text())

    def test_fit_provenance_and_held_out_intermediate_replays(self):
        self.assertNotIn('schema_version', self.model)
        self.assertEqual([p['exposure_ms'] for p in self.model['profiles']], [4., 20.])
        self.assertEqual(len(self.model['training_captures']), 8)
        self.assertEqual(self.report['capture_count'], 6)
        self.assertEqual(len(self.report['exposure_summary']), 12)
        for report in self.report['results']:
            mode = 'exact' if report['structured_light_exposure_ms'] in (4., 20.) else 'interpolated'
            self.assertEqual(report['exposure_selection']['mode'], mode)
            if report['label'] == 'empty':
                self.assertEqual(report['foreground_point_count'], 0)
            self.assertGreater(report['camera_diagnostics'][0]['unknown_point_count'], 0)
        self.assertEqual(ds.sha256(self.model_path), self.hash)

    def test_nonround_exposure_retains_foreground_and_excludes_unknown(self):
        for name, row in [('bin_a', 1), ('bin_b', 4)]:
            output = self.base / f'parts_{name}'
            self.cli('filter', '--dataset', self.root, '--model', self.model_path,
                     '--capture', 'parts', '--bin', name, '--output', output)
            report = json.loads((output / 'report.json').read_text())
            self.assertAlmostEqual(report['exposure_selection']['weight'], 8.5 / 16)
            self.assertEqual(report['foreground_point_count'], 1)
            points, _ = read_ply(output / 'foreground_robot_mm.ply')
            np.testing.assert_allclose(points, self.probes[[row]], atol=1e-5)

    def test_single_exposure_export_remains_exact_only(self):
        output = self.base / 'single'
        self.cli('calibrate', '--dataset', self.root, '--output', output, '--exposure-ms', 4.)
        model = model_io.load_model(output / 'model.json')
        self.assertNotIn('schema_version', model)
        self.assertEqual(len(model['training_captures']), 4)
        rejected = self.base / 'single_rejected'
        result = self.cli('filter', '--dataset', self.root, '--model', output / 'model.json',
                          '--capture', 'parts', '--bin', 'bin_a', '--output', rejected, success=False)
        self.assertIn('exposure differs', result.stderr)
        self.assertFalse(rejected.exists())

    def test_pooled_calibration_replays_all_exposures_with_one_shared_grid(self):
        output = self.base / 'pooled'
        self.cli('calibrate', '--dataset', self.root, '--output', output, '--pool-exposures')
        path = output / 'model.json'
        model = model_io.load_model(path)
        frozen_hash = ds.sha256(path)
        self.assertNotIn('schema_version', model)
        self.assertEqual(model['model_type'], 'offline_height_field_bins')
        self.assertEqual(model['exposure_settings_ms'], [4., 20.])
        self.assertEqual(len(model['training_captures']), 8)
        for key in ('exposure_ms', 'profiles', 'interpolation'):
            self.assertNotIn(key, model)
        for field in model['bins'].values():
            self.assertEqual(field['diagnostics']['capture_count'], 8)
        validation = self.base / 'pooled_validation'
        self.cli('validate', '--dataset', self.root, '--model', path, '--output', validation)
        report = json.loads((validation / 'validation.json').read_text())
        self.assertEqual(report['capture_count'], 6)
        for entry in report['results']:
            self.assertEqual(entry['exposure_selection']['mode'], 'pooled')
            self.assertEqual(entry['foreground_point_count'], 0 if entry['label'] == 'empty' else 1)

        # Provenance describes fitted settings, not a runtime selection range.
        root = self.base / 'pooled_outside_settings'
        root.mkdir()
        shutil.copytree(self.root / 'calibration', root / 'calibration')
        manifest = ds.read_yaml(self.root / 'dataset.yaml')
        entry = copy.deepcopy(next(e for e in manifest['captures'] if e['id'] == 'test_1'))
        entry['exposure_ms'] = entry['structured_light_exposure_ms'] = 25.
        ds.write_yaml(root / 'dataset.yaml', manifest | {'captures': [entry]})
        self.cli('filter', '--dataset', root, '--model', path, '--capture', entry['id'],
                 '--bin', 'bin_a', '--output', self.base / 'pooled_outside_filter')
        self.assertEqual(ds.sha256(path), frozen_hash)

    def test_pooled_flag_is_exclusive_with_profile_and_single_selection(self):
        for selection in (('--exposure-profiles',), ('--exposure-ms', 4.)):
            output = self.base / 'pooled_conflicting_options'
            result = self.cli('calibrate', '--dataset', self.root, '--output', output,
                              '--pool-exposures', *selection, success=False)
            self.assertIn('not allowed with argument', result.stderr)
            self.assertFalse(output.exists())

    def test_group_leakage_and_extrapolation_fail_before_output(self):
        manifest = ds.read_yaml(self.root / 'dataset.yaml')
        test_entry = copy.deepcopy(next(e for e in manifest['captures'] if e['id'] == 'test_1'))
        for variant in ('group', 'extrapolation'):
            root = self.base / variant
            root.mkdir()
            shutil.copytree(self.root / 'calibration', root / 'calibration')
            entry = copy.deepcopy(test_entry)
            if variant == 'group':
                entry['group_id'] = 'train_session_0'
            else:
                entry['exposure_ms'] = entry['structured_light_exposure_ms'] = 21.
            modified = manifest | {'captures': [entry]}
            ds.write_yaml(root / 'dataset.yaml', modified)
            output = self.base / f'rejected_{variant}'
            result = self.cli('validate', '--dataset', root, '--model', self.model_path,
                              '--output', output, success=False)
            self.assertIn('group used for fitting' if variant == 'group' else 'no extrapolation', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
