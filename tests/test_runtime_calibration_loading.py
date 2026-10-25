"""Offline validation can use deployed models without duplicate YAML files."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bin_filter
import dataset as data
import model_io


class RuntimeCalibrationLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / 'dataset'
        (self.dataset / 'calibration').mkdir(parents=True)
        self.runtime = self.root / 'runtime/model.json'
        self.runtime.parent.mkdir()
        self.runtime.write_text('{}')
        intrinsic = [100., 0, 32., 0, 100., 24., 0, 0, 1.]
        camera = {'rows': 48, 'cols': 64, 'cam_intrinsic': intrinsic,
                  'cam_extrinsic': np.eye(4).ravel().tolist()}
        self.camera = {'world_r': np.eye(3).ravel().tolist(), 'world_t': [0., 0., 0.],
                       'cam1': deepcopy(camera), 'cam2': deepcopy(camera), 'comment': 'Training copy'}
        self.robot = {'name': 'station', 'hand_eye': [
            {'scene_id': 4, 'calibration': [1., 2., 3., 0., 0., 0.]},
            {'scene_id': 0, 'calibration': [0., 0., 0., 0., 0., 0.]}]}
        self.model = {'model_type': 'calibrated_bin_background', 'scene_id': 4, 'frame_id': 'station',
                      'calibration_reference': model_io.calibration_values(self.camera, self.robot, 4, 'station')}
        self.manifest = {'scene_id': 4, 'frame_id': 'station', 'calibration': {
            key: {'path': f'calibration/{key}.yaml'} for key in ('camera', 'robot')}}
        self.write_snapshots()

    def write_snapshots(self):
        data.write_yaml(self.dataset / 'calibration/camera.yaml', self.camera)
        data.write_yaml(self.dataset / 'calibration/robot.yaml', self.robot)

    def load(self):
        with patch('bin_filter.model_io.load_model', return_value=self.model), \
                patch('modern_model.validate_model'):
            return bin_filter.load_model(SimpleNamespace(model=self.runtime, dataset=self.dataset), self.manifest)

    def test_runtime_metadata_uses_dataset_calibration_without_model_snapshot_directory(self):
        self.assertFalse((self.runtime.parent / 'calibration').exists())
        self.assertEqual(self.load(), self.model)
        # Unrelated scene calibration and notes do not change this model's frame.
        self.robot['hand_eye'][1]['calibration'][0] = 500.
        self.camera['comment'] = 'Current deployed calibration'
        self.write_snapshots()
        self.assertEqual(self.load(), self.model)

    def test_changed_camera_intrinsic_or_selected_hand_eye_is_rejected(self):
        self.camera['cam1']['cam_intrinsic'][0] += 1.
        self.write_snapshots()
        with self.assertRaisesRegex(data.DatasetError, 'calibration differs'):
            self.load()
        self.camera['cam1']['cam_intrinsic'][0] -= 1.
        self.robot['hand_eye'][0]['calibration'][0] += 1.
        self.write_snapshots()
        with self.assertRaisesRegex(data.DatasetError, 'calibration differs'):
            self.load()

    def test_candidate_full_snapshot_validation_remains_supported(self):
        self.model.pop('calibration_reference')
        (self.runtime.parent / 'calibration').mkdir()
        for name in ('camera', 'robot'):
            (self.runtime.parent / 'calibration' / f'{name}.yaml').write_bytes(
                (self.dataset / 'calibration' / f'{name}.yaml').read_bytes())
        self.assertEqual(self.load(), self.model)
        self.camera['cam2']['cam_intrinsic'][0] += 1.
        self.write_snapshots()
        with self.assertRaisesRegex(data.DatasetError, 'calibration differs'):
            self.load()


if __name__ == '__main__':
    unittest.main()
