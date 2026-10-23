#!/usr/bin/env python3
"""Hardware-free native helper checks. Run in an isolated ROS_DOMAIN_ID."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from ebots_ros2_common.msg import CameraNodeStatus, DenaliStatus
from ebots_ros2_common.srv import DenaliRequest, DepthImageRequest
import yaml


class NativeCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('ROS_DOMAIN_ID') != '199':
            raise RuntimeError('Tests require the isolated ROS_DOMAIN_ID=199')
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('bin_capture_readiness_fixture')
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.worker = threading.Thread(target=self.executor.spin)
        self.worker.start()
        self.calls = 0

    def tearDown(self):
        self.executor.shutdown()
        self.worker.join()
        self.node.destroy_node()

    def run_check(self, scene=0):
        result = subprocess.run([EXECUTABLE, '--check', '--scene-id', str(scene), '--ready-timeout-sec', '0.8'],
                                text=True, capture_output=True, timeout=10)
        lines = [line for line in result.stdout.splitlines() if line.startswith('{')]
        self.assertTrue(lines, result.stderr)
        return result.returncode, json.loads(lines[-1])

    def services(self, scene=0):
        def forbid_capture(request, response):
            self.calls += 1
            return response
        self.node.create_service(DenaliRequest, f'/denali_camera_scene{scene}', forbid_capture)
        self.node.create_service(DepthImageRequest, f'/eyes_depth{scene}', forbid_capture)

    def status(self, *, scene=0, sensors=3, ready=1, simulated=False):
        publisher = self.node.create_publisher(CameraNodeStatus, '/camera_status', 10)
        message = CameraNodeStatus(scene_id=scene, node_status=ready)
        message.denali_statuses = [DenaliStatus(sensor_en=sensors,
            status='Created simulated Denali0' if simulated else 'Created Denali0')]
        self.node.create_timer(0.05, lambda: publisher.publish(message))

    def test_missing_nodes_report_individual_dependencies(self):
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertFalse(report['ready'])
        self.assertEqual(report['camera_status']['publisher_count'], 0)
        self.assertFalse(report['services']['denali']['ready'])
        self.assertFalse(report['services']['eyes_depth']['ready'])
        self.assertGreaterEqual(len(report['missing']), 3)

    def test_clients_alone_are_not_live_service_servers(self):
        self.node.create_client(DenaliRequest, '/denali_camera_scene0')
        self.node.create_client(DepthImageRequest, '/eyes_depth0')
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertFalse(report['services']['denali']['ready'])
        self.assertFalse(report['services']['eyes_depth']['ready'])

    def test_ready_hardware_status_never_calls_capture_services(self):
        self.services()
        self.status()
        code, report = self.run_check()
        self.assertEqual(code, 0, report)
        self.assertTrue(report['ready'])
        self.assertTrue(report['read_only'])
        self.assertTrue(report['camera_status']['hardware'])
        self.assertFalse(report['eyes_calibration_verified'])
        self.assertEqual(self.calls, 0)

    def test_wrong_scene_is_reported(self):
        self.services()
        self.status(scene=1)
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertEqual(report['camera_status']['observed_scene_ids'], [1])
        self.assertFalse(report['camera_status']['received'])
        self.assertEqual(self.calls, 0)

    def test_scene_one_and_four_use_their_own_camera_services(self):
        for scene in (1, 4):
            self.services(scene)
            self.status(scene=scene)
        for scene in (1, 4):
            with self.subTest(scene=scene):
                code, report = self.run_check(scene)
                self.assertEqual(code, 0, report)
                self.assertEqual(report['scene_id'], scene)
                self.assertEqual(report['services']['denali']['name'], f'/denali_camera_scene{scene}')
                self.assertEqual(report['services']['eyes_depth']['name'], f'/eyes_depth{scene}')
                self.assertTrue(report['read_only'])
                self.assertEqual(self.calls, 0)

    def test_loading_camera_and_single_sensor_are_not_ready(self):
        self.services()
        self.status(ready=2, sensors=1)
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertTrue(any('not ready' in item for item in report['missing']))
        self.assertTrue(any('both sensors' in item for item in report['missing']))
        self.assertEqual(self.calls, 0)

    def test_simulated_camera_cannot_satisfy_hardware_preflight(self):
        self.services()
        self.status(simulated=True)
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertFalse(report['camera_status']['hardware'])
        self.assertEqual(self.calls, 0)

    def test_standalone_capture_and_relocated_frozen_calibration(self):
        with tempfile.TemporaryDirectory(prefix='bin-native-test-') as directory:
            root = Path(directory)
            part_name = 'data_saving/bin_model/sessions/test/depth_04ms'
            shared = root / 'artifacts/data_saving/bin_model/sessions/test/calibration'
            shared.mkdir(parents=True)
            calibration = {'world_r': [1,0,0,0,1,0,0,0,1], 'world_t': [0,0,0]}
            for name in ('cam1', 'cam2'):
                calibration[name] = {'rows': 2, 'cols': 3, 'cam_intrinsic': [1,0,0,0,1,0,0,0,1],
                                    'cam_extrinsic': [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]}
            (shared / 'camera.yaml').write_text(yaml.safe_dump(calibration))
            (shared / 'robot.yaml').write_text(yaml.safe_dump({'name': 'B2',
                'hand_eye': [{'scene_id': 0, 'calibration': [0,0,0,0,0,0]}]}))
            for kind in ('spotlight', 'depth'):
                (root / kind).mkdir()
                for camera in range(2):
                    self.assertTrue(cv2.imwrite(str(root / kind / f'Frame{camera}.exr'),
                                               np.full((2, 3), 100.125, dtype=np.float32)))
            recipe = {'segmentation': {'class_id': 999}, 'image': {
                'exposure_time': [4,4], 'exposure_time_spot_light': [10,10],
                'exposure_time_structure_light': [4,4], 'gain': [0,0], 'black_level': [240,240]},
                'tasks': [{'name': 'capture', 'node': 'perception', 'steps': [{'step': 'capture_data',
                    'cam_id': 0, 'parameters': {'denali_id': 0, 'frame_id': 'B2',
                        'camera_calibration_path': '/not-mounted/session/calibration/camera.yaml',
                        'robot_calibration_path': '/not-mounted/session/calibration/robot.yaml',
                        'sim_camera_dir': str(root)}}]}]}
            config = root / 'part.yaml'
            config.write_text(yaml.safe_dump(recipe))
            command = [EXECUTABLE, '--part-config', str(config), '--part-name', part_name,
                '--camera-calibration', str(shared / 'camera.yaml'), '--robot-calibration', str(shared / 'robot.yaml'),
                '--scene-id', '0', '--output', str(root / 'capture'), '--simulate']
            environment = {**os.environ, 'EBOTS_ROOT': str(root)}
            strict = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(strict.returncode, 2)
            self.assertFalse((root / 'capture').exists())
            result = subprocess.run(command + ['--relocated-session'], env=environment,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = yaml.safe_load((root / 'capture/capture_manifest.yaml').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['part_name'], part_name)
            self.assertTrue(manifest['simulated'])
            self.assertEqual(config.read_text(), yaml.safe_dump(recipe))
            repeated = subprocess.run(command + ['--relocated-session'], env=environment,
                                      capture_output=True, text=True, timeout=10)
            self.assertNotEqual(repeated.returncode, 0)

            # New sessions use the deployed calibration directly. The recorded
            # capture may retain provenance copies, but no session calibration
            # files or relocation override are required for acquisition.
            camera = root / 'artifacts/calibration/deployment/camera/scene4_denali0_params.yaml'
            robot = root / 'artifacts/calibration/deployment/robot/B2_params.yaml'
            camera.parent.mkdir(parents=True)
            robot.parent.mkdir(parents=True)
            camera.write_text(yaml.safe_dump(calibration))
            robot.write_text(yaml.safe_dump({'name': 'B2',
                'hand_eye': [{'scene_id': 4, 'calibration': [0,0,0,0,0,0]}]}))
            params = recipe['tasks'][0]['steps'][0]['parameters']
            params['camera_calibration_path'] = str(camera)
            params['robot_calibration_path'] = str(robot)
            config.write_text(yaml.safe_dump(recipe))
            deployed = list(command)
            for flag, value in (('--camera-calibration', camera), ('--robot-calibration', robot),
                                ('--scene-id', 4), ('--output', root / 'capture-scene4')):
                deployed[deployed.index(flag) + 1] = str(value)
            result = subprocess.run(deployed, env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = yaml.safe_load((root / 'capture-scene4/capture_manifest.yaml').read_text())
            self.assertEqual(manifest['scene_id'], 4)
            self.assertEqual(manifest['calibration']['camera']['source_path'], str(camera))
            self.assertEqual(manifest['calibration']['robot']['source_path'], str(robot))
            self.assertEqual(yaml.safe_load(camera.read_text()), calibration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable', required=True)
    options, remaining = parser.parse_known_args()
    EXECUTABLE = options.executable
    unittest.main(argv=[__file__, *remaining])
