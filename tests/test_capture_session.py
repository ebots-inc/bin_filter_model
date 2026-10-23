"""Factory-session import checks with tiny EXR fixtures and frozen calibration."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as ds
import cv2
import numpy as np


class CaptureSessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.session = self.base / 'session'
        self.session.mkdir()
        calibration = self.session / 'calibration'
        ds.write_yaml(calibration / 'camera.yaml', {name: {'rows': 2, 'cols': 3,
            'cam_intrinsic': np.eye(3).ravel().tolist(), 'cam_extrinsic': np.eye(4).ravel().tolist()}
            for name in ('cam1', 'cam2')})
        ds.write_yaml(calibration / 'robot.yaml', {'name': 'B2', 'hand_eye': [
            {'scene_id': 0, 'calibration': [0.] * 6}]})
        ds.write_yaml(self.session / 'bin_config.yaml', {'bins': {}})
        self.manifest = {'frame_id': 'B2', 'length_unit': 'mm', 'scene_id': 0,
            'calibration': {name: {'path': f'calibration/{name}.yaml'}
                            for name in ('camera', 'robot')},
            'settings': {'path': 'bin_config.yaml'}, 'captures': []}
        for index, exposure in enumerate((4., 20.)):
            capture_id = f'train_{index}'
            relative = f'captures/{capture_id}'
            root = self.session / relative
            root.mkdir(parents=True)
            shutil.copytree(calibration, root / 'calibration')
            capture = {'status': 'complete', 'scene_id': 0, 'simulated': False,
                       'robot_id': 'B2', 'frame_id': 'B2', 'length_unit': 'mm', 'depth_axis': 'z',
                       'images': {}, 'files': [],
                       'calibration': copy.deepcopy(self.manifest['calibration'])}
            for kind in ('depth', 'spotlight'):
                (root / kind).mkdir()
                capture['images'][kind] = []
                for camera in range(2):
                    path = root / kind / f'Frame{camera}.exr'
                    self.write_exr(path, np.full((2, 3), 100. + index + camera, np.float32))
                    capture['images'][kind].append({'height': 2, 'width': 3, 'dtype': 'float32'})
            ds.write_yaml(root / 'requested_settings.yaml', {'structured_light_exposure_ms': [exposure] * 2,
                'spotlight_exposure_ms': [10.] * 2, 'values_are_requested_not_sensor_verified': True})
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    capture['files'].append({'path': str(path.relative_to(root)), 'size_bytes': path.stat().st_size})
            ds.write_yaml(root / 'capture_manifest.yaml', capture)
            self.manifest['captures'].append({'id': capture_id, 'path': relative, 'split': 'train',
                'bins': {'bin_a': 'empty', 'bin_b': 'empty'}, 'exposure_ms': exposure,
                'spotlight_exposure_ms': 10., 'group_id': f'train_{index}'})
        self.save_session()

    def write_exr(self, path, image):
        try:
            import OpenEXR
        except ImportError:
            if not cv2.haveImageWriter('.exr'):
                self.skipTest('OpenEXR dependency not installed and OpenCV has no EXR codec')
            self.assertTrue(cv2.imwrite(str(path), image))
        else:
            with OpenEXR.File({}, {'Y': image}) as source:
                source.write(str(path))

    def save_session(self):
        (self.session / 'session.json').write_text(json.dumps(self.manifest))

    def update_capture_file_list(self, root):
        path = root / 'capture_manifest.yaml'
        capture = ds.read_yaml(path)
        for item in capture['files']:
            source = root / item['path']
            item.update(size_bytes=source.stat().st_size)
        ds.write_yaml(path, capture)

    def test_completed_session_import_keeps_requested_evidence_and_images(self):
        output = self.base / 'dataset'
        manifest = ds.import_session(self.session, output)
        self.assertEqual([ds.capture_exposure(item) for item in manifest['captures']], [4., 20.])
        self.assertTrue(all(item['exposure_evidence'] == 'requested_not_sensor_verified'
                            for item in manifest['captures']))
        _, calibration = ds.load_dataset(output)
        points = ds.load_capture_points(output, manifest['captures'][0], calibration)
        self.assertEqual(points.shape, (12, 3))
        self.assertEqual(manifest['captures'][0]['spotlight_exposure_ms'], 10.)
        self.assertTrue((output / 'bin_config.yaml').is_file())
        with self.assertRaisesRegex(ds.DatasetError, 'overwrite'):
            ds.import_session(self.session, output)

    def test_deployment_calibration_import_supports_one_and_three_bins_other_scenes(self):
        artifacts = self.base / 'artifacts'
        camera = artifacts / 'calibration/deployment/camera/scene4_denali0_params.yaml'
        robot = artifacts / 'calibration/deployment/robot/A1_params.yaml'
        ds.write_yaml(camera, ds.read_yaml(self.session / 'calibration/camera.yaml'))
        ds.write_yaml(robot, {'name': 'A1', 'hand_eye': [
            {'scene_id': scene, 'calibration': [scene * 10., 0., 0., 0., 0., 0.]} for scene in (1, 4)]})
        self.manifest.update(artifacts_root=str(artifacts), frame_id='A1', calibration={
            'camera': {'artifact_path': str(camera.relative_to(artifacts))},
            'robot': {'artifact_path': str(robot.relative_to(artifacts))}})
        shutil.rmtree(self.session / 'calibration')
        for scene, names in ((1, ['supply-tray']), (4, ['incoming.parts', 'resistors', 'rejects'])):
            with self.subTest(scene=scene):
                self.manifest['scene_id'] = scene
                ds.write_yaml(self.session / 'bin_config.yaml', {'bins': {name: {} for name in names}})
                for request in self.manifest['captures']:
                    request['bins'] = {name: 'empty' for name in names}
                    root = self.session / request['path']
                    shutil.copyfile(robot, root / 'calibration/robot.yaml')
                    captured = ds.read_yaml(root / 'capture_manifest.yaml')
                    captured.update(scene_id=scene, frame_id='A1', robot_id='A1')
                    ds.write_yaml(root / 'capture_manifest.yaml', captured)
                    self.update_capture_file_list(root)
                self.save_session()
                output = self.base / f'dataset-scene-{scene}'
                manifest = ds.import_session(self.session, output)
                self.assertEqual(manifest['frame_id'], 'A1')
                self.assertEqual(manifest['scene_id'], scene)
                self.assertEqual(set(manifest['captures'][0]['bins']), set(names))
                _, calibration = ds.load_dataset(output)
                self.assertEqual(calibration['robot_from_vision'][0, 3], scene * 10.)
                self.assertFalse((self.session / 'calibration').exists())
                self.assertTrue((output / 'calibration/robot.yaml').is_file())
        self.manifest['calibration']['camera']['artifact_path'] = '../camera.yaml'
        with self.assertRaises(ds.DatasetError):
            ds.session_calibration_paths(self.session, self.manifest)

    def test_session_cannot_publish_partial_or_calibration_mismatched_dataset(self):
        root = self.session / self.manifest['captures'][1]['path']
        manifest_path = root / 'capture_manifest.yaml'
        original = manifest_path.read_text()
        for failure in ('incomplete', 'calibration', 'simulated', 'requested_exposure', 'asymmetric'):
            with self.subTest(failure=failure):
                manifest_path.write_text(original)
                capture = ds.read_yaml(manifest_path)
                if failure == 'incomplete':
                    capture['status'] = 'incomplete'
                elif failure == 'calibration':
                    other = ds.read_yaml(root / 'calibration/camera.yaml')
                    other['cam1']['cam_intrinsic'][0] = 2.
                    ds.write_yaml(root / 'different_camera.yaml', other)
                    capture['calibration']['camera']['path'] = 'different_camera.yaml'
                elif failure == 'simulated':
                    capture['simulated'] = True
                elif failure == 'requested_exposure':
                    self.manifest['captures'][1]['exposure_ms'] = 12.
                    self.save_session()
                elif failure == 'asymmetric':
                    settings = ds.read_yaml(root / 'requested_settings.yaml')
                    settings['structured_light_exposure_ms'] = [4., 20.]
                    ds.write_yaml(root / 'requested_settings.yaml', settings)
                ds.write_yaml(manifest_path, capture)
                if failure == 'asymmetric':
                    self.update_capture_file_list(root)
                target = self.base / failure
                with self.assertRaises(ds.DatasetError):
                    ds.import_session(self.session, target)
                self.assertFalse(target.exists())
                self.manifest['captures'][1]['exposure_ms'] = 20.
                self.save_session()

    def test_unfinished_exr_folder_never_falls_back_to_legacy_tiff(self):
        root = self.session / self.manifest['captures'][0]['path']
        (root / 'capture_manifest.yaml').unlink()
        with self.assertRaisesRegex(ds.DatasetError, 'no completed'):
            ds.discover_images(root)

    def test_guided_confirmation_and_stage_labels_are_preserved_and_checked(self):
        self.manifest.update(mode='guided', session_id='bin-setup-test', acceptance={'require_resistor': True})
        stage = {'id':'empty_training', 'title':'Empty bins', 'split':'train', 'rounds':1,
                 'bins': {name:{'label':'empty','part_count':0,'placement':'none'} for name in ('bin_a','bin_b')},
                 'capture_ids':[entry['id'] for entry in self.manifest['captures']],
                 'confirmation_path':'stage_confirmations/empty_training.json'}
        self.manifest['stages'] = [stage]
        for entry in self.manifest['captures']:
            entry['stage_id'] = stage['id']
        ds.write_yaml(self.session / 'recipe.yaml', {'operator_recipe':'measured'})
        self.manifest['recipe'] = {'path':'recipe.yaml'}
        self.save_session()
        output = self.base / 'dataset'
        with self.assertRaisesRegex(ds.DatasetError, 'confirmation'):
            ds.import_session(self.session, output)
        confirmation = self.session / stage['confirmation_path']
        confirmation.parent.mkdir()
        ds.write_json(confirmation, {'session_id':'wrong','stage_id':stage['id'],'confirmed_at':'2026-10-23T10:00:00Z'})
        with self.assertRaisesRegex(ds.DatasetError, 'confirmation'):
            ds.import_session(self.session, output)
        ds.write_json(confirmation, {'session_id':'bin-setup-test','stage_id':stage['id'],'confirmed_at':'2026-10-23T10:00:00Z'})
        imported = ds.import_session(self.session, output)
        self.assertEqual(imported['stages'], [stage])
        self.assertEqual(imported['acceptance'], {'require_resistor':True})
        self.assertTrue(all(entry['stage_id']=='empty_training' for entry in imported['captures']))
        self.assertTrue((output / 'recipe.yaml').is_file())
        self.assertEqual(json.loads((output / stage['confirmation_path']).read_text())['session_id'], 'bin-setup-test')
        self.manifest['captures'][0]['bins']['bin_a'] = 'loaded'
        self.save_session()
        with self.assertRaisesRegex(ds.DatasetError, 'labels/split'):
            ds.import_session(self.session, self.base / 'wrong_labels')

    def test_loaded_arrangements_keep_separate_confirmations_and_groups_when_imported(self):
        self.manifest.update(mode='guided', session_id='arrangement-session', stages=[])
        labels = {'bin_a': 'loaded', 'bin_b': 'loaded'}
        regions = {'right': [.80, .35, .98, .65], 'top': [.35, .80, .65, .98]}
        for index, (capture, side) in enumerate(zip(self.manifest['captures'], regions), start=1):
            stage_id = f'all_resistor_wall_{side}_arrangement_01'
            stage = {'id': stage_id, 'split': 'test', 'rounds': 1, 'arrangement_index': 1,
                     'arrangement_count': 3, 'bins': {name: {'label': label} for name, label in labels.items()},
                     'capture_ids': [capture['id']], 'confirmation_path': f'stage_confirmations/{stage_id}.json'}
            for item in stage['bins'].values():
                item.update(placement='wall', wall_side=side, part_count=1, target_region_fraction=regions[side])
            capture.update(split='test', stage_id=stage_id, group_id=f'arrangement-session_{stage_id}', bins=labels)
            self.manifest['stages'].append(stage)
            confirmation = self.session / stage['confirmation_path']
            confirmation.parent.mkdir(exist_ok=True)
            ds.write_json(confirmation, {'session_id': 'arrangement-session', 'stage_id': stage_id,
                                         'confirmed_at': f'2026-10-23T10:0{index}:00Z'})
        self.save_session()
        imported = ds.import_session(self.session, self.base / 'arrangements-dataset')
        self.assertEqual(imported['stages'], self.manifest['stages'])
        self.assertEqual({stage['bins']['bin_a']['wall_side']: stage['bins']['bin_a']['target_region_fraction']
                          for stage in imported['stages']}, regions)
        self.assertTrue(all(item['part_count'] == 1 and item['label'] == 'loaded'
                            for stage in imported['stages'] for item in stage['bins'].values()))
        self.assertEqual(len({entry['group_id'] for entry in imported['captures']}), 2)
        for actual, expected in zip(imported['captures'], self.manifest['captures']):
            self.assertEqual(actual['stage_id'], expected['stage_id'])
            self.assertEqual(actual['group_id'], expected['group_id'])
            self.assertEqual(actual['bins'], labels)
        second = self.manifest['stages'][1]
        (self.session / second['confirmation_path']).unlink()
        with self.assertRaisesRegex(ds.DatasetError, second['id'] + ': operator confirmation is missing'):
            ds.import_session(self.session, self.base / 'missing-arrangement-confirmation')


if __name__ == '__main__':
    unittest.main()
