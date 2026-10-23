"""Collection frontend → Float32 EXRs → importer → pooled fit/replay integration."""
import importlib.util
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


class BuildSessionTests(unittest.TestCase):
    def test_frontend_session_builds_complete_reviewable_package(self):
        try:
            import OpenEXR
        except ImportError:
            self.skipTest('Install requirements.txt to exercise real EXR integration')
        frontend = HERE.parent / 'ebots_system_artifacts/artifacts_cell_4/data_saving/bin_model/prepare_session.py'
        if not frontend.is_file():
            self.skipTest('Collection frontend checkout is unavailable')
        spec = importlib.util.spec_from_file_location('collection_frontend', frontend)
        collection = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collection)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / 'artifacts'
            camera = {name: {'rows': 80, 'cols': 120,
                'cam_intrinsic': [50., 0., 0., 0., 50., 0., 0., 0., 1.],
                'cam_extrinsic': np.eye(4).ravel().tolist()} for name in ('cam1', 'cam2')}
            camera.update(world_r=np.eye(3).ravel().tolist(), world_t=[0., 0., 0.])
            camera_path = 'calibration/deployment/camera/scene0_denali0_params.yaml'
            robot_path = 'calibration/deployment/robot/B2_params.yaml'
            ds.write_yaml(artifacts / camera_path, camera)
            ds.write_yaml(artifacts / robot_path, {'name': 'B2', 'hand_eye': [
                {'scene_id': 0, 'calibration': [0.] * 6}]})
            settings = {'bins': {}}
            for name, xmin in [('bin_a', 0), ('bin_b', 120)]:
                settings['bins'][name] = {'kind': 'height_field',
                    'xy_bounds_mm': [[xmin, 0], [xmin + 100, 140]], 'z_bounds_mm': [90, 110],
                    'seed_xy_mm': [xmin + 50, 70], 'max_reference_height_mm': 109,
                    'cell_size_mm': 10, 'min_capture_support': 3}
            ds.write_yaml(artifacts / 'bin_config.yaml', settings)
            ds.write_yaml(artifacts / 'workflow_configs/bin_calibration.yaml', {'scenes': [{'scene_id': 0,
                'camera': [{'cam_id': 0, 'parameters': ['artifacts/' + camera_path]}]}]})
            recipe = {'scene_id': 0, 'denali_id': 0,
                'frame_id': 'B2', 'length_unit': 'mm', 'exposure_ms': [4, 8, 12, 16, 20],
                'empty_train_rounds': 3, 'empty_test_rounds': 1,
                'loaded_test_rounds': 1, 'gain': [1, 1], 'black_level': [0, 0],
                'camera_calibration': camera_path, 'robot_calibration': robot_path,
                'bin_config': 'bin_config.yaml'}
            folder = collection.prepare_session(artifacts, root / 'sessions', root / 'sessions', recipe, mode='empty')
            session = json.loads((folder / 'session.json').read_text())
            for index, capture in enumerate(session['captures']):
                self.assertEqual(capture['spotlight_exposure_ms'], capture['exposure_ms'])
                directory = folder / capture['path']
                directory.mkdir(parents=True)
                (directory / 'calibration').mkdir()
                for key in ('camera', 'robot'):
                    shutil.copyfile(artifacts / session['calibration'][key]['artifact_path'], directory / 'calibration' / f'{key}.yaml')
                manifest = {'status': 'complete', 'simulated': False,
                    'scene_id': 0, 'frame_id': 'B2', 'length_unit': 'mm', 'depth_axis': 'z',
                    'part_name': capture['part_name'], 'task_name': 'capture', 'files': [], 'images': {},
                    'calibration': {key: {'path': f'calibration/{key}.yaml'} for key in ('camera', 'robot')},
                    'requested_settings': {'path': 'requested_settings.yaml', 'values_are_requested_not_sensor_verified': True}}
                for kind in ('depth', 'spotlight'):
                    (directory / kind).mkdir()
                    manifest['images'][kind] = []
                    for camera_index in range(2):
                        image = np.random.default_rng(index * 2 + camera_index).normal(100, .01, (80, 120)).astype(np.float32)
                        OpenEXR.File({}, {'Y': image}).write(str(directory / kind / f'Frame{camera_index}.exr'))
                        manifest['images'][kind].append({'height': 80, 'width': 120, 'dtype': 'float32'})
                ds.write_yaml(directory / 'requested_settings.yaml', {
                    'structured_light_exposure_ms': [capture['exposure_ms']] * 2,
                    'spotlight_exposure_ms': [capture['spotlight_exposure_ms']] * 2, 'gain': [1, 1], 'black_level': [0, 0],
                    'values_are_requested_not_sensor_verified': True})
                for path in sorted(directory.rglob('*')):
                    if path.is_file():
                        manifest['files'].append({'path': str(path.relative_to(directory)), 'size_bytes': path.stat().st_size})
                ds.write_yaml(directory / 'capture_manifest.yaml', manifest)
            self.assertEqual(collection.verify_session(folder), 20)
            output = root / 'result'
            command = [sys.executable, str(HERE / 'bin_filter.py'), 'build-session', '--session', str(folder), '--output', str(output)]
            process = subprocess.run(command, text=True, capture_output=True, timeout=90)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            model = model_io.load_model(output / 'model')
            self.assertEqual(model['exposure_settings_ms'], [4, 8, 12, 16, 20])
            self.assertEqual(len(model['training_captures']), 15)
            report = json.loads((output / 'validation/validation.json').read_text())
            self.assertEqual(report['capture_count'], 5)
            self.assertEqual(report['status'], 'manual_review_required')
            self.assertTrue(all(item['foreground_point_count'] == 0 for item in report['results']))
            self.assertTrue(all(item['spotlight_exposure_ms'] == item['structured_light_exposure_ms']
                                for item in report['results']))
            self.assertEqual(json.loads((output / 'build.json').read_text())['activation'], 'not_activated')
            dataset, calibration = ds.load_dataset(output / 'dataset')
            self.assertTrue(all(ds.capture_spotlight_exposure(entry) == ds.capture_exposure(entry)
                                for entry in dataset['captures']))
            self.assertGreater(len(ds.load_capture_points(output / 'dataset', dataset['captures'][0], calibration)), 0)
            for document in output.rglob('*'):
                if document.suffix in ('.json', '.yaml'):
                    text = document.read_text()
                    self.assertNotIn('.bin-build-', text, str(document))
                    self.assertNotIn('.bin-session-', text, str(document))
            again = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn('overwrite', again.stderr)


if __name__ == '__main__':
    unittest.main()
