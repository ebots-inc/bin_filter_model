"""An operator's drawn layout becomes a usable local collection configuration."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as data
import layout_geometry as geometry
import prepare_session as collection
from setup_operations import Operations
from test_layout_geometry import measured_bowl


class VisualLayoutWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts, self.storage = self.root / 'artifacts', self.root / 'data'
        self.camera_path = 'calibration/deployment/camera/scene4_denali0_params.yaml'
        self.robot_path = 'calibration/deployment/robot/station.yaml'
        camera = {'rows': 600, 'cols': 800, 'cam_intrinsic': [100., 0., 400., 0., 100., 300., 0., 0., 1.],
                  'cam_extrinsic': np.eye(4).ravel().tolist()}
        self.camera = {'cam1': deepcopy(camera), 'cam2': deepcopy(camera),
                       'world_r': np.eye(3).ravel().tolist(), 'world_t': [0., 0., 0.]}
        self.robot = {'name': 'station', 'hand_eye': [{'scene_id': 4, 'calibration': [0.] * 6}]}
        self.write(self.artifacts / self.camera_path, self.camera)
        self.write(self.artifacts / self.robot_path, self.robot)
        self.write(self.artifacts / 'workflow_configs/bin_calibration.yaml', {'scenes': [
            {'scene_id': 4, 'camera': [{'cam_id': 0, 'parameters': ['artifacts/' + self.camera_path]}]}]})
        self.operations = Operations(self.artifacts, self.storage, self.artifacts / 'calibration/deployment/bin',
                                     self.artifacts / 'config/pose_estimation/pipeline.yaml')
        self.operations.recipe.update(scene_id=4, frame_id='station', camera_calibration=self.camera_path,
                                      robot_calibration=self.robot_path)
        self.original = self.storage / 'existing-session'
        self.original.mkdir(parents=True)
        (self.original / 'session.json').write_text('{"session_id":"existing-session","captures":[]}')
        (self.original / 'recorded-depth.exr').write_bytes(b'Preserve all original capture bytes')
        self.original_files = self.files(self.original)
        self.artifact_files = self.files(self.artifacts)
        self.preview = self.storage / 'layout_previews/layout-preview-test-01'
        self.preview.mkdir(parents=True)
        self.points = np.concatenate((measured_bowl((500., -300., -90.)),
                                      measured_bowl((650., -300., -90.)))).astype(np.float32)
        np.save(self.preview / 'points.npy', self.points, allow_pickle=False)
        rendered = geometry.render_preview(self.points, max_size=600)
        (self.preview / 'preview.png').write_bytes(base64.b64decode(rendered['preview_png']))
        metadata = {key: value for key, value in rendered.items() if key != 'preview_png'}
        metadata.update(path=str(self.preview), scene_id=4, frame_id='station', length_unit='mm',
                        camera_calibration=self.camera_path, robot_calibration=self.robot_path)
        (self.preview / 'preview.json').write_text(json.dumps(metadata))
        self.write(self.preview / 'capture/calibration/camera.yaml', self.camera)
        self.write(self.preview / 'capture/calibration/robot.yaml', self.robot)
        self.selections = [
            {'bin_id': name, 'rectangle': geometry._pixels(bounds, metadata).ravel().tolist()}
            for name, bounds in [('resistors', [[450., -350.], [550., -250.]]),
                                 ('spares-3', [[600., -350.], [700., -250.]])]]

    @staticmethod
    def write(path, value):
        collection.write_atomic(path, collection.dump_yaml(value))

    @staticmethod
    def files(root):
        return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}

    def execute(self, command, **values):
        return self.operations.execute({'command': command, **values}, lambda _: None, lambda: {})

    def analyze(self, **values):
        return self.execute('layout_analyze', **{'preview_path': str(self.preview),
                                                'bins': self.selections, **values})['layout_analysis']

    def save(self, analysis=None, **values):
        analysis = analysis or self.analyze()
        return self.execute('layout_save', **{'analysis_path': analysis['path'], 'name': 'Resistor station 4',
                                             'confirmed': True, **values})['layout_saved']

    def prepare(self, saved, name='from-visual-layout'):
        return self.execute('prepare', name=name, scene_id=4, camera_calibration=self.camera_path,
                            robot_calibration=self.robot_path, bin_config=saved['path'],
                            bin_ids=['resistors', 'spares-3'])

    def test_review_save_catalog_and_prepare_preserve_existing_capture_session(self):
        analysis = self.analyze()
        self.assertEqual(analysis['preview_path'], str(self.preview))
        self.assertEqual(analysis['preview']['path'], str(self.preview))
        self.assertEqual(analysis['selections'], self.selections)
        self.assertEqual([item['bin_id'] for item in analysis['bins']], ['resistors', 'spares-3'])
        self.assertTrue(analysis['preview_png'])
        self.assertNotEqual(analysis['preview_png'], analysis['preview']['preview_png'])
        saved = self.save(analysis)
        layout = data.read_yaml(saved['path'])
        self.assertEqual(Path(saved['path']).parent, self.storage / 'layouts')
        self.assertEqual(saved['label'], 'Resistor station 4')
        self.assertEqual(saved['scene_id'], 4)
        self.assertEqual(saved['camera_calibration'], self.camera_path)
        self.assertEqual(saved['robot_calibration'], self.robot_path)
        self.assertEqual(layout['scene_id'], 4)
        self.assertEqual(layout['frame_id'], 'station')
        self.assertEqual(list(layout['bins']), ['resistors', 'spares-3'])
        self.assertIn('calibration_reference', layout)
        self.assertEqual(self.files(self.artifacts), self.artifact_files)
        catalog = self.execute('catalog')['configuration']
        self.assertTrue(any(item['path'] == saved['path'] for item in catalog['layouts']))
        prepared = self.prepare(saved)
        self.assertEqual(prepared['session']['scene_id'], 4)
        self.assertEqual(prepared['session']['frame_id'], 'station')
        self.assertEqual(prepared['session']['bin_ids'], ['resistors', 'spares-3'])
        frozen = data.read_yaml(Path(prepared['session_path']) / 'bin_config.yaml')
        self.assertEqual(frozen, layout)
        self.assertEqual(self.files(self.original), self.original_files)
        new_artifacts = set(self.files(self.artifacts)) - set(self.artifact_files)
        self.assertTrue(new_artifacts)
        self.assertTrue(all(path.startswith('config/parts/data_saving/bin_model/sessions/from-visual-layout/')
                            for path in new_artifacts))

    def test_preview_capture_to_saved_layout_needs_no_preexisting_layout(self):
        def camera(folder, session, capture, destination, progress):
            self.assertNotIn('bin_ids', session)
            self.assertFalse((folder / 'bin_config.yaml').exists())
            self.write(destination / 'calibration/camera.yaml', self.camera)
            self.write(destination / 'calibration/robot.yaml', self.robot)
            self.write(destination / 'capture_manifest.yaml', {'status': 'complete', 'scene_id': 4,
                       'frame_id': 'station', 'length_unit': 'mm', 'depth_axis': 'z'})
        with patch('layout_setup.reconstruct_points', return_value=(self.points, [40000, 40000])):
            preview = self.operations.execute({'command': 'layout_preview', 'empty_confirmed': True},
                lambda _: None, lambda: {}, capture_runtime=camera)['layout_preview']
        selections = [{'bin_id': selection['bin_id'], 'rectangle': geometry._pixels(bounds, preview).ravel().tolist()}
                      for selection, bounds in zip(self.selections, [
                          [[450., -350.], [550., -250.]], [[600., -350.], [700., -250.]]])]
        analysis = self.analyze(preview_path=preview['path'], bins=selections)
        saved = self.save(analysis)
        self.assertTrue(Path(saved['path']).is_file())
        self.assertEqual(self.files(self.artifacts), self.artifact_files)
        self.assertEqual(self.files(self.original), self.original_files)

    def test_repeated_readable_name_never_overwrites_an_existing_layout(self):
        analysis = self.analyze()
        first = self.save(analysis)
        contents = Path(first['path']).read_bytes()
        second = self.save(analysis)
        self.assertEqual(Path(first['path']).name, 'resistor-station-4.yaml')
        self.assertEqual(Path(second['path']).name, 'resistor-station-4-02.yaml')
        self.assertEqual(Path(first['path']).read_bytes(), contents)
        self.assertEqual(self.files(self.original), self.original_files)

    def test_requires_review_confirmation_and_rejects_duplicate_ids(self):
        analysis = self.analyze()
        for flag in (False, None, 'true', 1):
            with self.subTest(flag=flag), self.assertRaisesRegex(data.DatasetError, 'Review every'):
                self.save(analysis, confirmed=flag)
        self.assertFalse((self.storage / 'layouts').exists())
        duplicate = deepcopy(self.selections)
        duplicate[1]['bin_id'] = duplicate[0]['bin_id']
        with self.assertRaises(data.DatasetError):
            self.analyze(bins=duplicate)

    def test_calibration_change_rejects_analysis_save_and_new_session(self):
        analysis = self.analyze()
        saved = self.save(analysis)
        self.robot['hand_eye'][0]['calibration'][0] += .1
        self.write(self.artifacts / self.robot_path, self.robot)
        for operation in (self.analyze, lambda: self.save(analysis), lambda: self.prepare(saved)):
            with self.subTest(operation=operation), self.assertRaisesRegex(data.DatasetError, 'calibration changed'):
                operation()
        self.assertFalse((self.storage / 'from-visual-layout').exists())
        self.assertEqual(self.files(self.original), self.original_files)

    def test_preview_and_analysis_cannot_come_from_outside_local_storage(self):
        for path in (self.root / 'other-layout', self.preview / 'capture', '../../outside'):
            with self.subTest(path=path), self.assertRaises(data.DatasetError):
                self.analyze(preview_path=str(path))
        analysis = self.analyze()
        outside = self.root / 'analysis-01'
        outside.mkdir()
        with self.assertRaises(data.DatasetError):
            self.save(analysis, analysis_path=str(outside))
        outside_file = self.root / 'outside.yaml'
        outside_file.write_text('{}')
        draft = Path(analysis['path']) / 'layout.yaml'
        draft.unlink()
        draft.symlink_to(outside_file)
        with self.assertRaisesRegex(data.DatasetError, 'review is incomplete'):
            self.save(analysis)

    def test_shorter_recipe_layout_can_be_prepared_at_each_exposure(self):
        self.operations.recipe['empty_train_rounds'] = 3
        saved = self.save()
        prepared = self.prepare(saved)
        settings = data.read_yaml(Path(prepared['session_path']) / 'bin_config.yaml')
        self.assertTrue(all(bin_config['min_capture_support'] <= 3 for bin_config in settings['bins'].values()))


if __name__ == '__main__':
    unittest.main()
