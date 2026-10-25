"""Layout preview acquisition without layouts, Docker or camera hardware."""
from copy import deepcopy
import base64
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as data
import layout_setup
import prepare_session as collection


class LayoutCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / 'artifacts'
        self.storage = self.root / 'data'
        self.camera_path = 'calibration/deployment/camera/scene4_denali0_params.yaml'
        self.robot_path = 'calibration/deployment/robot/station.yaml'
        camera = {'rows': 6, 'cols': 7, 'cam_intrinsic': [100., 0., 3., 0., 100., 2., 0., 0., 1.],
                  'cam_extrinsic': np.eye(4).ravel().tolist()}
        self.camera = {'cam1': deepcopy(camera), 'cam2': deepcopy(camera),
                       'world_r': np.eye(3).ravel().tolist(), 'world_t': [0., 0., 0.]}
        self.robot = {'name': 'station', 'hand_eye': [{'scene_id': 4, 'calibration': [10., 20., 30., 0., 0., 0.]}]}
        self.write(self.artifacts / self.camera_path, self.camera)
        self.write(self.artifacts / self.robot_path, self.robot)
        self.write(self.artifacts / 'workflow_configs/bin_calibration.yaml', {'scenes': [
            {'scene_id': 4, 'camera': [{'cam_id': 0, 'parameters': ['artifacts/' + self.camera_path]}]}]})
        self.recipe = {'scene_id': 4, 'denali_id': 0, 'gain': [0., 0.], 'black_level': [240, 240],
                       'camera_calibration': self.camera_path, 'robot_calibration': self.robot_path,
                       'spotlight_exposure_ms': 10.}
        self.request = {'empty_confirmed': True, 'scene_id': 4}
        self.rendering = {'preview_png': base64.b64encode(b'preview-png').decode('ascii'),
                          'width_px': 500, 'height_px': 400,
                          'lower_xy_mm': [-10., -20.], 'upper_xy_mm': [210., 150.],
                          'pixels_per_mm': 2., 'margin_px': 30}
        self.depth = np.full((6, 7), 100., np.float32)

    @staticmethod
    def write(path, value):
        collection.write_atomic(path, collection.dump_yaml(value))

    def native_capture(self, folder, session, entry, destination, progress):
        # This must be usable before any bin_config.yaml or collection session.
        self.assertFalse((folder / 'bin_config.yaml').exists())
        self.assertFalse((folder / 'session.json').exists())
        self.assertEqual(session['scene_id'], 4)
        self.assertEqual(session['frame_id'], 'station')
        self.assertEqual(session['calibration']['camera']['artifact_path'], self.camera_path)
        part = self.artifacts / 'config/parts' / (entry['part_name'] + '.yaml')
        self.assertEqual(part.read_bytes(), (folder / 'configs/depth_08ms.yaml').read_bytes())
        settings = data.read_yaml(part)['image']
        self.assertEqual(settings['exposure_time_structure_light'], [8., 8.])
        self.assertEqual(settings['exposure_time_spot_light'], [8., 8.])
        destination.mkdir()
        self.write(destination / 'capture_manifest.yaml', {'status': 'complete', 'scene_id': 4,
                   'frame_id': 'station', 'length_unit': 'mm', 'depth_axis': 'z'})
        for kind, relative in (('camera', self.camera_path), ('robot', self.robot_path)):
            target = destination / f'calibration/{kind}.yaml'
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(self.artifacts / relative, target)

    def capture(self, runtime=None, request=None):
        with patch('layout_setup.data.read_image', return_value=self.depth), \
                patch('layout_setup._render', return_value=self.rendering):
            return layout_setup.capture(self.artifacts, self.storage, self.recipe, request or self.request,
                                        runtime or self.native_capture, lambda _: None)

    def test_capture_without_existing_layout_and_remove_temporary_artifact_config(self):
        result = self.capture()
        self.assertEqual(result['scene_id'], 4)
        self.assertEqual(result['camera_points'], [42, 42])
        self.assertEqual(result['width'], 500)
        self.assertEqual(result['preview_png'], self.rendering['preview_png'])
        self.assertEqual((Path(result['path']) / 'preview.png').read_bytes(), b'preview-png')
        self.assertEqual(result['xy_bounds_mm'], [[-10., -20.], [210., 150.]])
        self.assertEqual(list((self.artifacts / 'config').rglob('*.yaml')), [])
        points, metadata = layout_setup.load_preview(self.storage, result['path'])
        self.assertEqual(points.shape, (84, 3))
        np.testing.assert_allclose(points[0], [7., 18., 130.])
        self.assertNotIn('preview_png', metadata)
        self.assertEqual(metadata['frame_id'], 'station')
        second = self.capture()
        self.assertNotEqual(second['path'], result['path'])
        self.assertTrue(Path(result['path']).is_dir())

    def test_requires_empty_confirmation_before_writing_or_capturing(self):
        runtime = Mock()
        for flag in (False, None, 'true', 1):
            with self.subTest(flag=flag), self.assertRaisesRegex(data.DatasetError, 'Empty all bins'):
                self.capture(runtime, {'empty_confirmed': flag})
        runtime.assert_not_called()
        self.assertFalse(self.storage.exists())

    def test_mismatched_eyes_scene_or_calibration_is_rejected_before_capture(self):
        runtime = Mock()
        self.robot['hand_eye'].append({'scene_id': 2, 'calibration': [0.] * 6})
        self.write(self.artifacts / self.robot_path, self.robot)
        with self.assertRaisesRegex(ValueError, 'exactly one entry for scene 2'):
            self.capture(runtime, {'empty_confirmed': True, 'scene_id': 2})
        runtime.assert_not_called()
        self.assertFalse(self.storage.exists())

    def test_failed_capture_keeps_diagnostic_data_and_removes_artifact_config(self):
        def fail(folder, session, entry, destination, progress):
            (folder / 'failure.log').write_text('Camera unavailable')
            raise RuntimeError('Camera unavailable')
        with self.assertRaisesRegex(RuntimeError, 'Camera unavailable'):
            self.capture(fail)
        self.assertEqual(list((self.artifacts / 'config').rglob('*.yaml')), [])
        self.assertEqual(len(list(self.storage.glob('layout_previews/*/failure.log'))), 1)
        self.assertEqual(list(self.storage.rglob('preview.json')), [])

    def test_calibration_change_during_capture_does_not_make_a_layout(self):
        def changed(*args):
            self.native_capture(*args)
            self.robot['hand_eye'][0]['calibration'][0] = 99.
            self.write(self.artifacts / self.robot_path, self.robot)
        with self.assertRaisesRegex(data.DatasetError, 'Calibration changed'):
            self.capture(changed)
        self.assertEqual(list((self.artifacts / 'config').rglob('*.yaml')), [])
        self.assertEqual(list(self.storage.rglob('preview.json')), [])

    def test_loading_cannot_leave_preview_storage_or_follow_a_cloud_symlink(self):
        result = self.capture()
        for path in (self.root, self.storage / 'not-a-preview', '../../private', result['path'] + '/capture'):
            with self.subTest(path=path), self.assertRaises(data.DatasetError):
                layout_setup.load_preview(self.storage, path)
        cloud = Path(result['path']) / 'points.npy'
        outside = self.root / 'outside.npy'
        cloud.rename(outside)
        cloud.symlink_to(outside)
        with self.assertRaisesRegex(data.DatasetError, 'missing or incomplete'):
            layout_setup.load_preview(self.storage, result['path'])

    def test_empty_depth_is_rejected(self):
        self.depth[:] = np.nan
        with self.assertRaisesRegex(data.DatasetError, 'No valid depth'):
            self.capture()

    def test_layout_reference_uses_only_reconstruction_values_and_selected_scene(self):
        result = self.capture()
        reference = layout_setup.validate_preview_calibration(self.artifacts, result['path'], result)
        self.assertEqual(reference['robot']['hand_eye'][0]['scene_id'], 4)
        self.camera['commissioning_note'] = 'Updated by technician'
        self.robot['hand_eye'].append({'scene_id': 0, 'calibration': [99.] * 6})
        self.write(self.artifacts / self.camera_path, self.camera)
        self.write(self.artifacts / self.robot_path, self.robot)
        self.assertEqual(layout_setup.validate_preview_calibration(self.artifacts, result['path'], result), reference)
        self.robot['hand_eye'][0]['calibration'][0] += .1
        self.write(self.artifacts / self.robot_path, self.robot)
        with self.assertRaisesRegex(data.DatasetError, 'Station calibration changed'):
            layout_setup.validate_preview_calibration(self.artifacts, result['path'], result)

    def test_camera_geometry_change_or_escaping_snapshot_is_rejected(self):
        result = self.capture()
        self.camera['cam1']['cam_intrinsic'][0] += 1
        self.write(self.artifacts / self.camera_path, self.camera)
        with self.assertRaisesRegex(data.DatasetError, 'Station calibration changed'):
            layout_setup.validate_preview_calibration(self.artifacts, result['path'], result)
        source = Path(result['path']) / 'capture/calibration/camera.yaml'
        moved = self.root / 'camera.yaml'
        source.rename(moved)
        source.symlink_to(moved)
        with self.assertRaisesRegex(data.DatasetError, 'preview calibration is missing'):
            layout_setup.validate_preview_calibration(self.artifacts, result['path'], result)


if __name__ == '__main__':
    unittest.main()
