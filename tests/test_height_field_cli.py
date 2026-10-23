"""Curved-bin CLI regressions using synthetic clouds and temporary files only."""
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
from height_field import classify


def bowl(x, y):
    return -100.0 + 0.04 * x * x + 0.025 * y * y


def empty_cloud(seed, inconsistent=False):
    """Two bowls, an unobserved hole, raised rim, and disconnected lower table."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(-10, 10), np.arange(-10, 10))
    cells = np.column_stack((x.ravel(), y.ravel()))
    bins = []
    for index, shift in enumerate((0.0, 40.0)):
        xy = np.repeat(cells, 16, axis=0) + rng.uniform(0.03, 0.97, (len(cells) * 16, 2))
        inner = np.all(np.abs(xy) < 7, axis=1)
        rim = np.all(np.abs(xy) < 9, axis=1)
        z = np.where(inner, bowl(xy[:, 0], xy[:, 1]), np.where(rim, -88.0, -125.0))
        z += rng.normal(0, 0.025, len(z))
        if inconsistent:
            changed = (xy[:, 0] > -6) & (xy[:, 0] < -4) & (xy[:, 1] > 3) & (xy[:, 1] < 5)
            # Outside the reference-fitting Z band, in opposite directions.
            # These observations must still influence measured uncertainty.
            z[changed] += 40 if index == 0 else -60
        hole = (xy[:, 0] > 2) & (xy[:, 0] < 4) & (xy[:, 1] > 2) & (xy[:, 1] < 4)
        bins.append(np.column_stack((xy[~hole] + [shift, 0], z[~hole])))
    return np.concatenate(bins)


def read_ply(path, colored=False):
    """Read the exact binary XYZ/RGB output, avoiding a viewer dependency."""
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    if colored:
        fields += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    with Path(path).open('rb') as stream:
        count = None
        for _ in range(30):
            line = stream.readline().decode('ascii').strip()
            if line.startswith('element vertex '):
                count = int(line.split()[-1])
            if line == 'end_header':
                break
        if count is None:
            raise AssertionError('Missing PLY vertex count')
        values = np.fromfile(stream, dtype=fields)
    if len(values) != count:
        raise AssertionError('PLY payload length does not match header')
    points = np.column_stack([values[key] for key in ('x', 'y', 'z')])
    colors = np.column_stack([values[key] for key in ('red', 'green', 'blue')]) if colored else None
    return points, colors


class HeightFieldCliTests(unittest.TestCase):
    @classmethod
    def run_cli(cls, *args, success=True):
        result = subprocess.run([sys.executable, str(HERE / 'bin_filter.py'), *map(str, args)],
                                cwd=HERE, capture_output=True, text=True, timeout=90)
        if (result.returncode == 0) != success:
            raise AssertionError(f'CLI returned {result.returncode}: {args}\n{result.stdout}\n{result.stderr}')
        if not success and 'Traceback' in result.stderr:
            raise AssertionError(f'Expected actionable error: {result.stderr}')
        return result

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name)
        camera = cls.base / 'camera.yaml'
        robot = cls.base / 'robot.yaml'
        ds.write_yaml(camera, {name: {'rows': 2, 'cols': 3,
                                     'cam_intrinsic': [2., 0., 0., 0., 2., 0., 0., 0., 1.],
                                     'cam_extrinsic': np.eye(4).ravel().tolist()}
                               for name in ('cam1', 'cam2')})
        ds.write_yaml(robot, {'name': 'B2', 'hand_eye': [
            {'scene_id': 0, 'calibration': [0.] * 6}]})
        cls.dataset = cls.base / 'dataset'
        cls.run_cli('init', '--dataset', cls.dataset, '--camera-calibration', camera,
                    '--robot-calibration', robot)
        settings = {'bins': {}}
        for name, shift in [('bin_a', 0.), ('bin_b', 40.)]:
            settings['bins'][name] = {'kind': 'height_field', 'bin_id': name,
                'xy_bounds_mm': [[shift - 10, -10], [shift + 10, 10]],
                'z_bounds_mm': [-135, -85], 'seed_xy_mm': [shift, 0],
                'max_reference_height_mm': -93, 'min_tolerance_mm': 0.7,
                'max_tolerance_mm': 3.0, 'alignment_margin_mm': 0.3}
        ds.write_yaml(cls.dataset / 'bin_config.yaml', settings)
        for index in range(6):
            cls.register(f'empty_train_{index}', 'train', 'empty',
                         empty_cloud(120 + index, inconsistent=index == 5))
        cls.register('empty_test', 'test', 'empty', empty_cloud(920))
        # One observed background, two real foreground probes (thin and above
        # the rim), three unknown regions, and three out-of-volume probes.
        local = np.array([[0, 0, -100], [0, 0, -98], [1, -1, -80],
                          [3, 3, -90], [9.5, 0, -110], [-5, 4, -90],
                          [0, 0, -110], [0, 0, 70], [-11, 0, -100]], dtype=float)
        cls.probes = np.concatenate([local, local + [40, 0, 0]])
        cls.register('loaded_test', 'test', 'loaded', cls.probes)
        # A loaded training recording is registered but may not fit the model.
        cls.register('loaded_train', 'train', 'loaded', cls.probes + [0, 0, 0.01])
        cls.model_path = cls.base / 'model' / 'model.json'
        cls.run_cli('calibrate', '--dataset', cls.dataset, '--output', cls.model_path.parent,
                    '--max-fit-points', 100)
        cls.model = model_io.load_model(cls.model_path)
        cls.frozen_hash = ds.sha256(cls.model_path)
        cls.reports = {}
        for split in ('train', 'test'):
            output = cls.base / f'validation_{split}'
            cls.run_cli('validate', '--dataset', cls.dataset, '--model', cls.model_path,
                        '--split', split, '--output', output)
            cls.reports[split] = json.loads((output / 'validation.json').read_text())
        cls.manifest, cls.calibration = ds.load_dataset(cls.dataset)

    @classmethod
    def register(cls, name, split, label, points):
        path = cls.base / f'{name}.npy'
        np.save(path, points)
        cls.run_cli('add', '--dataset', cls.dataset, '--cloud', path, '--cloud-frame', 'robot',
                    '--unfiltered', '--id', name, '--split', split, '--bin-a', label,
                    '--bin-b', label, '--exposure-ms', 2.0)

    def test_fit_provenance_uses_only_empty_train_and_validation_does_not_mutate_model(self):
        model = self.model
        self.assertEqual(model['model_type'], 'offline_height_field_bins')
        self.assertIsNone(model['max_fit_points_per_capture_bin'])
        self.assertEqual(model['bin_membership_check']['status'], 'disjoint_xy_interiors')
        self.assertEqual({item['id'] for item in model['training_captures']},
                         {f'empty_train_{index}' for index in range(6)})
        self.assertEqual(model['settings_sha256'], ds.sha256(self.dataset / 'bin_config.yaml'))
        self.assertEqual(set(model['implementation_sha256']),
                         {'bin_filter.py', 'dataset.py', 'model_io.py', 'height_field.py', 'exposure_profiles.py'})
        for name, digest in model['implementation_sha256'].items():
            self.assertEqual(digest, ds.sha256(HERE / name))
        self.assertEqual(model['validation_status'], 'requires_held_out_review')
        self.assertEqual(self.reports['train']['capture_count'], 7)
        self.assertEqual(self.reports['test']['capture_count'], 2)
        for report in self.reports.values():
            self.assertEqual(report['model_sha256'], self.frozen_hash)
            self.assertEqual(report['status'], 'manual_review_required')
        self.assertTrue(all(item['kept_by_multiple_bins'] == 0 for item in report['bin_membership_overlap']))
        self.assertEqual(ds.sha256(self.model_path), self.frozen_hash)

    def test_out_of_band_training_observations_make_region_unknown_without_sample_cap(self):
        for name in ('bin_a', 'bin_b'):
            model = self.model['bins'][name]
            captures = model['diagnostics']['captures']
            self.assertEqual(len(captures), 6)
            self.assertTrue(all(item['input_point_count'] > 6000 for item in captures))
            self.assertLess(captures[-1]['background_fit_point_count'], captures[-1]['input_point_count'])
            self.assertGreater(model['diagnostics']['interpolation_squares_excessive_tolerance'], 0)
            shift = 0. if name == 'bin_a' else 40.
            keep, masks = classify(np.array([[shift - 5, 4, -90.]]), model)
            self.assertFalse(keep[0])
            self.assertTrue(masks['unknown'][0])
            self.assertLessEqual(np.nanmax(np.asarray(model['tolerance_mm'], dtype=float)), 3.)

    def test_masks_and_exports_retain_thin_and_above_rim_parts_but_not_unknown_or_behind(self):
        colors = {'surface': [60, 120, 230], 'kept': [40, 220, 60],
                  'unknown': [190, 70, 210], 'outside': [90, 90, 90]}
        own_reasons = ['surface', 'kept', 'kept', 'unknown', 'unknown', 'unknown',
                       'outside', 'outside', 'outside']
        for name in ('bin_a', 'bin_b'):
            selected = slice(0, 9) if name == 'bin_a' else slice(9, 18)
            expected = np.zeros(18, dtype=bool)
            expected[[1, 2] if name == 'bin_a' else [10, 11]] = True
            keep, masks = classify(self.probes, self.model['bins'][name])
            np.testing.assert_array_equal(keep, expected)
            np.testing.assert_array_equal(sum(mask.astype(int) for mask in masks.values()), np.ones(18))
            for index, reason in enumerate(own_reasons, start=selected.start):
                self.assertTrue(masks[reason][index], f'{name} probe {index}: expected {reason}')
            output = self.base / f'filter_{name}'
            self.run_cli('filter', '--dataset', self.dataset, '--model', self.model_path,
                          '--capture', 'loaded_test', '--bin', name, '--output', output)
            report = json.loads((output / 'report.json').read_text())
            validation = next(item for item in self.reports['test']['results']
                              if item['capture_id'] == 'loaded_test' and item['bin_id'] == name)
            self.assertEqual(report['counts'], validation['counts'])
            self.assertEqual(report['counts'], {'invalid': 0, 'outside': 12, 'unknown': 3,
                                                'surface': 1, 'kept': 2})
            self.assertEqual(report['classified_bin_point_count'], 3)
            self.assertEqual(report['unknown_point_count'], 3)
            foreground, _ = read_ply(output / 'foreground_robot_mm.ply')
            np.testing.assert_allclose(foreground, self.probes[expected], atol=1e-5)
            preview, rgb = read_ply(output / 'classified_preview_robot_mm.ply', colored=True)
            np.testing.assert_allclose(preview, self.probes, atol=1e-5)
            np.testing.assert_array_equal(rgb[selected], [colors[reason] for reason in own_reasons])
            before = ds.sha256(output / 'foreground_robot_mm.ply')
            self.run_cli('filter', '--dataset', self.dataset, '--model', self.model_path,
                          '--capture', 'loaded_test', '--bin', name, '--output', output, success=False)
            self.assertEqual(ds.sha256(output / 'foreground_robot_mm.ply'), before)
        self.assertEqual(ds.sha256(self.model_path), self.frozen_hash)

    def test_independent_empty_has_no_foreground_and_reports_excluded_unknown_coverage(self):
        for report in self.reports['test']['results']:
            if report['capture_id'] != 'empty_test':
                continue
            self.assertEqual(report['foreground_point_count'], 0)
            self.assertEqual(report['empty_leakage_fraction'], 0.)
            self.assertGreater(report['unknown_point_count'], 0)
            self.assertGreater(report['classified_bin_point_count'], 0)
            self.assertEqual(report['classified_bin_point_count'], report['counts']['surface'])
            self.assertTrue(report['review_required'])
            self.assertIn('No summit', report['interpretation'])

    def test_replay_rejects_training_as_test_and_mismatched_calibration_or_exposure(self):
        variants = []
        leaky = copy.deepcopy(self.manifest)
        next(item for item in leaky['captures'] if item['id'] == 'empty_train_0')['split'] = 'test'
        variants.append(('leaky', leaky, 'inputs used for fitting'))
        exposure = copy.deepcopy(self.manifest)
        next(item for item in exposure['captures'] if item['id'] == 'loaded_test')['exposure_ms'] = 3.
        variants.append(('exposure', exposure, 'exposure differs'))
        for name, manifest, error in variants:
            root = self.base / name
            root.mkdir()
            shutil.copytree(self.dataset / 'calibration', root / 'calibration')
            ds.write_yaml(root / 'dataset.yaml', manifest)
            output = self.base / f'rejected_{name}'
            result = self.run_cli('validate', '--dataset', root, '--model', self.model_path,
                                  '--output', output, success=False)
            self.assertIn(error, result.stderr)
            self.assertFalse(output.exists())
        wrong = copy.deepcopy(self.model)
        wrong['calibration_sha256']['camera'] = '0' * 64
        model_path = self.base / 'wrong_calibration.yaml'
        ds.write_yaml(model_path, wrong)
        output = self.base / 'rejected_calibration'
        result = self.run_cli('filter', '--dataset', self.dataset, '--model', model_path,
                              '--capture', 'loaded_test', '--bin', 'bin_a', '--output', output,
                              success=False)
        self.assertIn('Model calibration differs', result.stderr)
        self.assertFalse(output.exists())
        self.assertEqual(ds.sha256(self.model_path), self.frozen_hash)


if __name__ == '__main__':
    unittest.main()
