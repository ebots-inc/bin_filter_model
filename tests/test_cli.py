"""CLI integration using only the generated demo and temporary local files."""
import json
from pathlib import Path
import copy
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


def read_json(path):
    return json.loads(Path(path).read_text())


def ply_vertex_count(path):
    with Path(path).open('rb') as stream:
        for _ in range(30):
            line = stream.readline().decode('ascii').strip()
            if line.startswith('element vertex '):
                return int(line.split()[-1])
            if line == 'end_header':
                break
    raise AssertionError(f'Missing PLY vertex count: {path}')


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name)
        cls.demo = cls.base / 'demo'
        cls.run_cli('demo', '--output', cls.demo)
        cls.dataset = cls.demo / 'dataset'
        cls.manifest, cls.calibration = ds.load_dataset(cls.dataset)
        cls.model_path = cls.demo / 'model' / 'model.json'
        cls.model = model_io.load_model(cls.model_path)
        cls.validation = read_json(cls.demo / 'validation' / 'validation.json')

    @classmethod
    def run_cli(cls, *args, success=True):
        result = subprocess.run([sys.executable, str(HERE / 'bin_filter.py'), *map(str, args)],
                                cwd=HERE, capture_output=True, text=True, timeout=90)
        if (result.returncode == 0) != success:
            raise AssertionError(f'CLI returned {result.returncode}: {args}\n{result.stdout}\n{result.stderr}')
        if not success and 'Traceback' in result.stderr:
            raise AssertionError(f'Expected actionable error, got traceback: {result.stderr}')
        return result

    def test_demo_fits_only_empty_training_and_validates_independent_test_inputs(self):
        entries = {entry['id']: entry for entry in self.manifest['captures']}
        training = self.model['training_captures']
        self.assertGreaterEqual(len(training), 3)
        trained_signatures = {entry['input_signature'] for entry in training}
        for item in training:
            source = entries[item['id']]
            self.assertEqual(source['split'], 'train')
            self.assertEqual(set(source['bins'].values()), {'empty'})
        held_out = [entry for entry in entries.values() if entry['split'] == 'test']
        self.assertTrue(held_out)
        self.assertTrue(trained_signatures.isdisjoint(entry['input_signature'] for entry in held_out))
        self.assertEqual(self.validation['capture_count'], len(held_out))
        self.assertEqual(len(self.validation['results']), 2 * len(held_out))
        self.assertEqual(self.model['online_integration'], 'not_implemented')
        self.assertEqual(self.model['validation_status'], 'requires_held_out_review')
        self.assertEqual(self.validation['status'], 'manual_review_required')

    def test_synthetic_empty_leakage_loaded_foreground_and_written_counts(self):
        manifest_entries = {entry['id']: entry for entry in self.manifest['captures']}
        empty_seen = loaded_seen = False
        for report in self.validation['results']:
            entry = manifest_entries[report['capture_id']]
            source = ds.load_capture_points(self.dataset, entry, self.calibration)
            self.assertEqual(sum(report['counts'].values()), len(source))
            self.assertEqual(report['foreground_point_count'], report['counts']['kept'])
            output = self.demo / 'validation' / entry['id'] / report['bin_id']
            self.assertEqual(ply_vertex_count(output / 'foreground_robot_mm.ply'), report['foreground_point_count'])
            self.assertEqual(ply_vertex_count(output / 'classified_preview_robot_mm.ply'), report['preview_points'])
            self.assertTrue(report['review_required'])
            self.assertIn('No summit', report['interpretation'])
            if report['label'] == 'empty':
                empty_seen = True
                self.assertIsNotNone(report['empty_leakage_fraction'])
                self.assertLess(report['empty_leakage_fraction'], 0.02)
            else:
                loaded_seen = True
                self.assertIsNone(report['empty_leakage_fraction'])
                self.assertGreater(report['foreground_point_count'], 0)
        self.assertTrue(empty_seen and loaded_seen)

    def test_rendered_pixel_truth_measures_retention_and_other_bin_rejection(self):
        from geometry import classify
        for entry in self.manifest['captures']:
            if entry['split'] != 'test':
                continue
            points = ds.load_capture_points(self.dataset, entry, self.calibration)
            truth = []
            for index, item in enumerate(entry['depth_images']):
                depth_path = ds.checked_file(self.dataset, item)
                depth = ds.read_image(depth_path, self.calibration['cameras'][index])
                labels = np.load(depth_path.parent.parent / 'truth' / f'Frame{index}.npy')
                self.assertEqual(labels.shape, depth.shape)
                truth.append(labels[np.isfinite(depth) & (depth > 0)])
            truth = np.concatenate(truth)
            self.assertEqual(len(truth), len(points))
            for bin_index, name in enumerate(('bin_a', 'bin_b'), start=1):
                keep, _ = classify(points, self.model['bins'][name])
                object_pixels = truth == bin_index
                other_object = (truth != 0) & ~object_pixels
                self.assertFalse(np.any(keep & other_object), f'{entry["id"]}: retained other bin object')
                self.assertLess(np.count_nonzero(keep & (truth == 0)) / max(1, np.count_nonzero(truth == 0)),
                                0.001)
                if entry['bins'][name] == 'loaded':
                    self.assertGreater(np.count_nonzero(object_pixels), 20)
                    retention = np.count_nonzero(keep & object_pixels) / np.count_nonzero(object_pixels)
                    self.assertGreater(retention, 0.9, f'{entry["id"]}: synthetic thin object retention')
                    object_points = points[object_pixels]
                    self.assertTrue(np.all((object_points[:, 2] > 99.8) & (object_points[:, 2] < 103.2)))
                    center_x = -70. if name == 'bin_a' else 70.
                    self.assertTrue(np.all(np.abs(object_points[:, 0] - center_x) < 23.))
                else:
                    self.assertFalse(np.any(object_pixels))

    def test_filter_and_inspect_cli_match_validation_and_do_not_overwrite(self):
        report = next(item for item in self.validation['results'] if item['label'] == 'loaded')
        output = self.base / 'one_filter'
        arguments = ('filter', '--dataset', self.dataset, '--capture', report['capture_id'],
                     '--model', self.model_path, '--bin', report['bin_id'], '--output', output)
        self.run_cli(*arguments)
        result = read_json(output / 'report.json')
        self.assertEqual(result['counts'], report['counts'])
        before = (output / 'foreground_robot_mm.ply').read_bytes()
        self.run_cli(*arguments, success=False)
        self.assertEqual((output / 'foreground_robot_mm.ply').read_bytes(), before)
        inspection = self.base / 'inspection'
        self.run_cli('inspect', '--dataset', self.dataset, '--capture', report['capture_id'],
                     '--output', inspection)
        summary = read_json(inspection / 'inspection.json')
        self.assertEqual(summary['frame_id'], 'B2')
        self.assertEqual(summary['length_unit'], 'mm')
        self.assertEqual(summary['point_count'], sum(report['counts'].values()))

    def test_cli_init_add_and_invalid_seed_configuration_are_actionable(self):
        root = self.base / 'new_dataset'
        self.run_cli('init', '--dataset', root,
                     '--camera-calibration', self.dataset / 'calibration/camera.yaml',
                     '--robot-calibration', self.dataset / 'calibration/robot.yaml')
        seed = ds.read_yaml(root / 'bin_config.yaml')
        self.assertEqual(set(seed['bins']), {'bin_a', 'bin_b'})
        self.assertTrue(all(item['interior_polygon_mm'] is None for item in seed['bins'].values()))
        originals = [item for item in self.manifest['captures'] if item['split'] == 'train' and
                     set(item['bins'].values()) == {'empty'}][:3]
        for index, entry in enumerate(originals):
            points = ds.load_capture_points(self.dataset, entry, self.calibration)
            cloud = self.base / f'cli_empty_{index}.npy'
            np.save(cloud, points)
            args = ('add', '--dataset', root, '--cloud', cloud, '--cloud-frame', 'robot',
                    '--id', f'empty_{index}', '--split', 'train', '--bin-a', 'empty', '--bin-b', 'empty',
                    '--exposure-ms', entry['exposure_ms'])
            if index == 0:
                self.run_cli(*args, success=False)
            self.run_cli(*args, '--unfiltered')
        incomplete = self.base / 'incomplete_model'
        result = self.run_cli('calibrate', '--dataset', root, '--output', incomplete, success=False)
        self.assertTrue('floor' in result.stderr or 'transform' in result.stderr or 'seed' in result.stderr)
        self.assertFalse(incomplete.exists())
        seed['bins'].pop('bin_b')
        ds.write_yaml(root / 'bin_config.yaml', seed)
        result = self.run_cli('calibrate', '--dataset', root, '--output', incomplete, success=False)
        self.assertIn('every configured bin', result.stderr)

    def test_cli_named_bin_scene_and_robot_frame_are_preserved(self):
        root = self.base / 'named-bin-scene4'
        robot = self.base / 'A1_scene4.yaml'
        ds.write_yaml(robot, {'name': 'A1', 'hand_eye': [{'scene_id': 4, 'calibration': [0.] * 6}]})
        self.run_cli('init', '--dataset', root, '--scene-id', 4, '--bins', 'supply-tray',
                     '--camera-calibration', self.dataset / 'calibration/camera.yaml', '--robot-calibration', robot)
        points = np.array([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]])
        cloud = self.base / 'named-bin-cloud.npy'
        np.save(cloud, points)
        self.run_cli('add', '--dataset', root, '--cloud', cloud, '--cloud-frame', 'robot', '--unfiltered',
                     '--id', 'empty', '--split', 'train', '--bin-label', 'supply-tray=empty', '--exposure-ms', 4)
        manifest, _ = ds.load_dataset(root)
        self.assertEqual((manifest['scene_id'], manifest['frame_id']), (4, 'A1'))
        self.assertEqual(manifest['captures'][0]['bins'], {'supply-tray': 'empty'})
        output = self.base / 'named-bin-inspect'
        self.run_cli('inspect', '--dataset', root, '--capture', 'empty', '--output', output)
        self.assertEqual(read_json(output / 'inspection.json')['frame_id'], 'A1')
        self.assertIn(b'comment frame A1 units mm', (output / 'raw_robot_mm.ply').read_bytes())

    def test_validate_rejects_reassigned_training_capture_and_model_calibration_mismatch(self):
        leaky = self.base / 'leaky_dataset'
        leaky.mkdir()
        shutil.copytree(self.dataset / 'calibration', leaky / 'calibration')
        manifest = copy.deepcopy(self.manifest)
        trained_id = self.model['training_captures'][0]['id']
        next(item for item in manifest['captures'] if item['id'] == trained_id)['split'] = 'test'
        ds.write_yaml(leaky / 'dataset.yaml', manifest)
        output = self.base / 'leaky_validation'
        result = self.run_cli('validate', '--dataset', leaky, '--model', self.model_path,
                              '--output', output, success=False)
        self.assertIn('inputs used for fitting', result.stderr)
        self.assertFalse(output.exists())
        wrong_model = copy.deepcopy(self.model)
        wrong_model['calibration_sha256']['camera'] = '0' * 64
        model_path = self.base / 'wrong_calibration_model.yaml'
        ds.write_yaml(model_path, wrong_model)
        entry = next(item for item in self.manifest['captures'] if item['split'] == 'test')
        result = self.run_cli('filter', '--dataset', self.dataset, '--model', model_path,
                              '--capture', entry['id'], '--bin', 'bin_a', '--output', output,
                              success=False)
        self.assertIn('Model calibration differs', result.stderr)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
