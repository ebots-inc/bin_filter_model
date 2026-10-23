"""Preview uses the selected deployment scene and places every bin region."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as data
import preview


class PreviewTests(unittest.TestCase):
    def test_preview_uses_selected_scene_and_deployment_calibration_without_session_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts, session = root / 'artifacts', root / 'session'
            session.mkdir()
            camera_path = 'calibration/deployment/camera/scene4_denali0_params.yaml'
            robot_path = 'calibration/deployment/robot/A1_params.yaml'
            data.write_yaml(artifacts / camera_path, {name: {'rows': 2, 'cols': 3,
                'cam_intrinsic': np.eye(3).ravel().tolist(), 'cam_extrinsic': np.eye(4).ravel().tolist()}
                for name in ('cam1', 'cam2')})
            data.write_yaml(artifacts / robot_path, {'name': 'A1', 'hand_eye': [
                {'scene_id': 0, 'calibration': [1000, 0, 0, 0, 0, 0]},
                {'scene_id': 4, 'calibration': [0] * 6}]})
            (session / 'session.json').write_text(json.dumps({'scene_id': 4, 'frame_id': 'A1',
                'artifacts_root': str(artifacts), 'calibration': {'camera': {'artifact_path': camera_path},
                                                               'robot': {'artifact_path': robot_path}}}))
            data.write_yaml(session / 'bin_config.yaml', {'bins': {'supply-tray': {'xy_bounds_mm': [[0, 0], [250, 250]]}}})
            with patch('preview.data.read_image', return_value=np.full((2, 3), 100., dtype=np.float32)):
                result = preview.render(root / 'capture', session)
            self.assertEqual(result['camera_points'], [2, 2])
            self.assertIn('Scene 4 in A1', result['caption'])
            self.assertTrue(result['preview_png'])
            self.assertFalse((session / 'calibration').exists())

    def test_bin_local_regions_project_each_placement_into_robot_coordinates(self):
        settings = {'bins': {}}
        for index, name in enumerate(('incoming.parts', 'supply-tray', 'rejects')):
            settings['bins'][name] = {'xy_bounds_mm': [[-10, -5], [10, 5]], 'z_bounds_mm': [0, 10],
                'coordinate_frame': 'bin', 'robot_from_bin': data.pose_matrix([index * 100, 20, 0, 0, 0, 90]).tolist()}
        regions = preview.robot_regions(settings)
        for index, vertices in enumerate(regions.values()):
            np.testing.assert_allclose(vertices.min(axis=0), [index * 100 - 5, 10])
            np.testing.assert_allclose(vertices.max(axis=0), [index * 100 + 5, 30])


if __name__ == '__main__':
    unittest.main()
