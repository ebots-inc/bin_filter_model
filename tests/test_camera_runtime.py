"""Camera runtime contracts, with synthetic files and every Docker call mocked."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import camera_runtime
import dataset as data
from docker_runtime import DockerRuntime, discover_container


def inspection(*mounts):
    return {'State': {'Running': True}, 'Mounts': [
        {'Type': 'bind', 'Source': str(source), 'Destination': str(destination)}
        for source, destination in mounts]}


def readiness(ready=True):
    return {'ready': ready, 'scene_id': 0, 'ros_domain_id': '77', 'rmw_implementation': 'rmw_fastrtps_cpp',
            'camera_status': {'received': True, 'node_status': 1, 'hardware': True,
                              'age_seconds': .2, 'denali': [{'sensor_en': 3}]},
            'services': {'denali': {'ready': True, 'name': '/camera0'},
                         'eyes_depth': {'ready': ready, 'name': '/eyes_depth0'}},
            'missing': [] if ready else ['Eyes depth service unavailable']}


class DockerMappingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='bin_runtime_map_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts, self.nested = self.root / 'artifacts', self.root / 'artifacts/cache'
        self.runtime = DockerRuntime('camera-test', inspection=inspection(
            (self.artifacts, '/root/artifacts'), (self.nested, '/root/cache'),
            (self.root / 'config-overlay', '/root/artifacts/config')))

    def test_longest_mount_wins_in_each_direction(self):
        self.assertEqual(self.runtime.to_runtime(self.nested / 'image.exr'), '/root/cache/image.exr')
        self.assertEqual(self.runtime.to_runtime(self.artifacts / 'calibration/camera.yaml'),
                         '/root/artifacts/calibration/camera.yaml')
        self.assertEqual(self.runtime.to_host('/root/artifacts/config/part.yaml'),
                         str(self.root / 'config-overlay/part.yaml'))
        self.assertEqual(self.runtime.to_host('/root/cache/image.exr'), str(self.nested / 'image.exr'))

    def test_unmapped_and_traversing_paths_are_rejected(self):
        for path in ('relative/path', '/root/artifacts/../private', '/root/artifacts/config/../../private', '/private'):
            with self.subTest(path=path), self.assertRaises(data.DatasetError):
                self.runtime.to_host(path)
        for path in (self.root / 'outside', self.artifacts / '../outside'):
            with self.subTest(path=path), self.assertRaises(data.DatasetError):
                self.runtime.to_runtime(path)
        self.artifacts.mkdir()
        (self.artifacts / 'outside-link').symlink_to(self.root / 'outside')
        with self.assertRaises(data.DatasetError):
            self.runtime.to_runtime(self.artifacts / 'outside-link/file')

    def test_arguments_are_literal_and_ownership_is_limited_to_shared_output(self):
        argv = ['bin_capture', '--part-name', 'name with spaces;$(touch /tmp/do-not-run)', '--output',
                '/root/artifacts/capture with spaces']
        with patch.dict(os.environ, {'ROS_DOMAIN_ID': '77', 'ROS_LOCALHOST_ONLY': '1'}, clear=True), \
             patch('docker_runtime.os.getuid', return_value=1001), patch('docker_runtime.os.getgid', return_value=1002):
            command = self.runtime.command(argv, outputs=[argv[-1]])
        self.assertEqual(command[:2], ['docker', 'exec'])
        self.assertNotIn('--user', command)
        self.assertIn('ROS_DOMAIN_ID=77', command)
        self.assertIn('ROS_LOCALHOST_ONLY=1', command)
        self.assertEqual(command[-len(argv):], argv)
        script_index = command.index('-c') + 1
        self.assertNotIn(argv[2], command[script_index])
        self.assertIn('trap handoff EXIT', command[script_index])
        self.assertEqual(command[script_index + 1:-len(argv)],
                         ['bin-setup-client', '1001:1002', '1', argv[-1]])
        with self.assertRaises(data.DatasetError):
            self.runtime.command(argv, outputs=['/not-a-bind/output'])

    def test_stopped_container_is_rejected_without_running_docker(self):
        with self.assertRaisesRegex(data.DatasetError, 'stopped'):
            DockerRuntime('camera-test', inspection={'State': {'Running': False}})


    def test_discovery_matches_artifacts_and_capability_without_name_convention(self):
        probes = []
        def run(command, **kwargs):
            if command[1] == 'ps':
                return subprocess.CompletedProcess(command, 0, 'production-camera-7\nother-station\nmisleading_eyes\n', '')
            self.assertEqual(command[:2], ['docker', 'exec'])
            probes.append(command)
            return subprocess.CompletedProcess(command, 0 if command[2] == 'production-camera-7' else 1, '', '')
        def inspect(arguments):
            source = self.root / 'other-artifacts' if arguments[1] == 'other-station' else self.artifacts
            return [inspection((source, '/root/artifacts'))]
        with patch('docker_runtime.subprocess.run', side_effect=run), \
             patch('docker_runtime.docker_json', side_effect=inspect):
            self.assertEqual(discover_container('eyes', self.artifacts), 'production-camera-7')
        self.assertEqual([command[2] for command in probes], ['production-camera-7', 'misleading_eyes'])
        self.assertTrue(all('bin-setup-discovery' in command for command in probes))
        self.assertTrue(all('--check' not in command and '--output' not in command for command in probes))

    def test_ambiguous_discovery_requires_explicit_selection(self):
        def run(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, 'camera-one\ncamera-two\n' if command[1] == 'ps' else '', '')
        with patch('docker_runtime.subprocess.run', side_effect=run), \
             patch('docker_runtime.docker_json', return_value=[inspection((self.artifacts, '/root/artifacts'))]), \
             self.assertRaisesRegex(data.DatasetError, 'camera-one, camera-two.*--camera-container NAME'):
            discover_container('eyes', self.artifacts)


class HostCameraRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='bin_camera_transport_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / 'artifacts'
        self.folder = self.root / 'local-sessions/bin-setup-test'
        self.session = {'session_id': 'bin-setup-test', 'scene_id': 0}
        self.capture = {'id': 'empty_training_01_depth_04ms', 'exposure_ms': 4,
                        'part_name': 'data_saving/bin_model/sessions/bin-setup-test/depth_04ms'}
        self.destination = self.folder / 'captures' / self.capture['id']
        self.shared = self.artifacts / 'data_saving/bin_model/sessions/bin-setup-test'
        self.part = self.artifacts / 'config/parts' / (self.capture['part_name'] + '.yaml')
        self.local_part = self.folder / 'configs/depth_04ms.yaml'
        for path in (self.part, self.local_part):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('image:\n  exposure_time_structure_light: [4.0, 4.0]\n')
        for name, value in (('camera', 'cam1: {width: 4, height: 3}\n'), ('robot', 'name: B2\n')):
            for root in (self.folder, self.shared):
                path = root / f'calibration/{name}.yaml'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value)
        self.remote_files = self.root / 'camera-temporary'
        self.remote_files.mkdir()
        self.runtime = DockerRuntime('eyes-test', inspection=inspection(
            (self.artifacts, '/root/artifacts'), (self.remote_files, '/tmp')))
        self.client = camera_runtime.HostCameraCapture(self.artifacts, container='eyes-test')
        self.runtime_patch = patch.object(self.client, 'runtime', return_value=self.runtime)
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        self.progress = []
        self.transfer_number = 0
        def create_transfer(runtime):
            self.transfer_number += 1
            remote = Path(f'/tmp/bin-model-capture.test-{self.transfer_number}')
            Path(runtime.to_host(remote)).mkdir()
            return remote
        def copy_transfer(runtime, remote, local):
            shutil.copytree(Path(runtime.to_host(remote)) / 'capture', local)
        def remove_transfer(runtime, remote):
            shutil.rmtree(runtime.to_host(remote))
        for name, implementation in [('create_transfer', create_transfer), ('copy_transfer', copy_transfer),
                                      ('remove_transfer', remove_transfer)]:
            patched = patch.object(self.client, name, side_effect=implementation)
            patched.start()
            self.addCleanup(patched.stop)

    def native_output(self, command):
        self.assertEqual(command[:2], ['docker', 'exec'])
        self.assertNotIn('--user', command)
        self.assertIn('--relocated-session', command)
        self.assertEqual(command[command.index('--part-name') + 1], self.capture['part_name'])
        self.assertEqual(Path(self.runtime.to_host(command[command.index('--part-config') + 1])), self.part)
        for name in ('camera', 'robot'):
            path = command[command.index(f'--{name}-calibration') + 1]
            self.assertEqual(Path(self.runtime.to_host(path)), self.shared / f'calibration/{name}.yaml')
        self.assertNotIn('timeout', command)
        self.assertFalse(any(item.startswith('--kill-after') for item in command))
        return Path(self.runtime.to_host(command[command.index('--output') + 1]))

    def completed_process(self, returncode):
        process = MagicMock()
        process.__enter__.return_value = process
        process.wait.return_value = returncode
        return process

    def create_native_capture(self, output, *, complete=True):
        output.mkdir()
        (output / 'depth').mkdir()
        (output / 'depth/Frame0.exr').write_bytes(b'original depth evidence')
        if complete:
            (output / 'capture_manifest.yaml').write_text('status: complete\n')
        else:
            (output / '.capture_in_progress').mkdir()

    def collect(self, capture=None):
        return self.client(self.folder, self.session, capture or self.capture, self.destination, self.progress.append)

    def test_readiness_json_is_exposed_without_requesting_capture(self):
        report = readiness()
        response = subprocess.CompletedProcess([], 0, 'native log line\n' + json.dumps(report) + '\n', '')
        with patch('camera_runtime.subprocess.run', return_value=response) as run:
            result = self.client.check()
        self.assertTrue(result['ready'])
        self.assertEqual(len(result['checks']), 3)
        self.assertTrue(all(check['ready'] for check in result['checks']))
        command = run.call_args.args[0]
        self.assertIn('--check', command)
        self.assertNotIn('--output', command)
        self.assertNotIn('--part-name', command)
        self.assertEqual(run.call_args.kwargs['timeout'], 20)
        self.assertFalse((self.folder / '.transfers').exists())

    def test_missing_service_report_and_invalid_helper_output_fail_closed(self):
        report = readiness(False)
        response = subprocess.CompletedProcess([], 2, json.dumps(report), '')
        with patch('camera_runtime.subprocess.run', return_value=response):
            result = self.client.check()
        self.assertFalse(result['ready'])
        missing = next(check for check in result['checks'] if check['id'] == 'eyes_depth')
        self.assertFalse(missing['ready'])
        self.assertIn('Start Eyes', missing['action'])
        for response in (subprocess.CompletedProcess([], 127, '', 'helper not installed'),
                         subprocess.CompletedProcess([], 0, 'no JSON report', '')):
            with self.subTest(code=response.returncode), patch('camera_runtime.subprocess.run', return_value=response), \
                 self.assertRaises(data.DatasetError):
                self.client.check()

    def test_readiness_passes_selected_scene_to_helper(self):
        for scene in (1, 4):
            response = subprocess.CompletedProcess([], 0, json.dumps(readiness() | {'scene_id': scene}), '')
            with self.subTest(scene=scene), patch('camera_runtime.subprocess.run', return_value=response) as run:
                result = self.client.check(scene)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--scene-id') + 1], str(scene))
            self.assertIn(f'scene {scene}', result['checks'][0]['label'])

    def deployment_references(self):
        entries = {}
        for name in ('camera', 'robot'):
            relative = f'calibration/deployment/{name}/scene4.yaml'
            path = self.artifacts / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((self.folder / f'calibration/{name}.yaml').read_bytes())
            entries[name] = {'artifact_path': relative}
        self.session.update(calibration=entries, artifacts_root=str(self.artifacts), scene_id=4)
        shutil.rmtree(self.folder / 'calibration')
        shutil.rmtree(self.shared / 'calibration')

    def test_deployment_calibration_is_read_directly_without_session_copies(self):
        self.deployment_references()
        def acquire(command, **kwargs):
            self.assertNotIn('--relocated-session', command)
            self.assertEqual(command[command.index('--scene-id') + 1], '4')
            for name in ('camera', 'robot'):
                self.assertEqual(command[command.index(f'--{name}-calibration') + 1],
                                 f'/root/artifacts/calibration/deployment/{name}/scene4.yaml')
            self.create_native_capture(Path(self.runtime.to_host(command[command.index('--output') + 1])))
            return self.completed_process(0)
        with patch('camera_runtime.subprocess.Popen', side_effect=acquire):
            self.collect()
        self.assertTrue((self.destination / 'capture_manifest.yaml').is_file())
        self.assertFalse((self.folder / 'calibration').exists())
        self.assertFalse((self.shared / 'calibration').exists())

    def test_invalid_deployment_references_are_rejected_before_capture(self):
        self.deployment_references()
        for reference in ('/private/camera.yaml', '../camera.yaml', 'config/parts/camera.yaml',
                          'calibration/deployment/robot/scene4.yaml', 'calibration/deployment/camera/missing.yaml'):
            self.session['calibration']['camera']['artifact_path'] = reference
            with self.subTest(reference=reference), patch('camera_runtime.subprocess.Popen') as run, \
                 self.assertRaises(data.DatasetError):
                self.collect()
            run.assert_not_called()
        self.assertFalse((self.folder / '.transfers').exists())

    def test_success_checks_frozen_inputs_and_publishes_after_copy(self):
        before = {path: path.read_bytes() for path in (self.part, self.local_part,
                  self.folder / 'calibration/camera.yaml', self.shared / 'calibration/camera.yaml')}
        def acquire(command, **kwargs):
            self.assertFalse(self.destination.exists())
            self.create_native_capture(self.native_output(command))
            self.assertTrue(kwargs['start_new_session'])
            return self.completed_process(0)
        copytree = shutil.copytree
        def copy_before_publish(source, incoming, *args, **kwargs):
            self.assertFalse(self.destination.exists())
            if Path(incoming).parent == self.destination.parent:
                self.assertTrue(Path(incoming).name.startswith('.incoming-'))
            result = copytree(source, incoming, *args, **kwargs)
            self.assertFalse(self.destination.exists())
            return result
        with patch('camera_runtime.subprocess.Popen', side_effect=acquire), \
             patch('camera_runtime.shutil.copytree', side_effect=copy_before_publish):
            self.collect()
        self.assertEqual((self.destination / 'depth/Frame0.exr').read_bytes(), b'original depth evidence')
        self.assertTrue((self.destination / 'capture_manifest.yaml').is_file())
        self.assertEqual(list((self.folder / '.transfers').iterdir()), [])
        self.assertFalse((self.shared / 'transfers').exists())
        self.assertEqual(list(self.remote_files.iterdir()), [])
        self.assertEqual(list(self.destination.parent.glob('.incoming-*')), [])
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_mismatched_or_missing_frozen_inputs_refuse_native_execution(self):
        original = self.part.read_bytes()
        self.part.write_text('different camera settings')
        with patch('camera_runtime.subprocess.Popen') as run, self.assertRaisesRegex(data.DatasetError, 'frozen settings'):
            self.collect()
        run.assert_not_called()
        self.part.write_bytes(original)
        robot = self.shared / 'calibration/robot.yaml'
        robot.write_text('name: A1\n')
        with patch('camera_runtime.subprocess.Popen') as run, self.assertRaisesRegex(data.DatasetError, 'robot calibration'):
            self.collect()
        run.assert_not_called()
        robot.unlink()
        with patch('camera_runtime.subprocess.Popen') as run, self.assertRaises((OSError, data.DatasetError)):
            self.collect()
        run.assert_not_called()
        with patch('camera_runtime.subprocess.Popen') as run, self.assertRaisesRegex(data.DatasetError, 'registered'):
            self.collect(self.capture | {'part_name': 'data_saving/bin_model/sessions/another/depth_04ms'})
        run.assert_not_called()
        self.assertFalse((self.folder / '.transfers').exists())

    def test_existing_destination_is_never_overwritten_or_dispatched(self):
        self.destination.mkdir(parents=True)
        original = self.destination / 'previous-evidence'
        original.write_bytes(b'keep this acquisition')
        with patch('camera_runtime.subprocess.Popen') as run, self.assertRaisesRegex(data.DatasetError, 'already exists'):
            self.collect()
        run.assert_not_called()
        self.assertEqual(original.read_bytes(), b'keep this acquisition')

    def test_native_failure_preserves_attempt_and_retry_uses_new_staging_folder(self):
        attempts = []
        def acquire(command, **kwargs):
            output = self.native_output(command)
            attempts.append(sorted((self.folder / '.transfers').iterdir())[-1])
            self.create_native_capture(output, complete=len(attempts) > 1)
            kwargs['stdout'].write('camera acquisition failed\n' if len(attempts) == 1 else 'complete\n')
            return self.completed_process(2 if len(attempts) == 1 else 0)
        with patch('camera_runtime.subprocess.Popen', side_effect=acquire):
            with self.assertRaisesRegex(data.DatasetError, 'Temporary data'):
                self.collect()
            self.assertFalse(self.destination.exists())
            self.assertTrue((attempts[0] / 'capture/depth/Frame0.exr').is_file())
            self.collect()
        self.assertNotEqual(attempts[0], attempts[1])
        self.assertTrue(attempts[0].is_dir())
        self.assertFalse(attempts[1].exists())
        self.assertTrue((self.destination / 'capture_manifest.yaml').is_file())

    def test_missing_completion_and_copy_failure_preserve_evidence(self):
        for failure in ('incomplete', 'copy'):
            with self.subTest(failure=failure):
                def acquire(command, **kwargs):
                    output = self.native_output(command)
                    self.create_native_capture(output, complete=failure == 'copy')
                    return self.completed_process(0)
                copying = patch('camera_runtime.shutil.copytree', side_effect=OSError('disk unavailable')) if failure == 'copy' else nullcontext()
                with patch('camera_runtime.subprocess.Popen', side_effect=acquire), copying, \
                     self.assertRaises((data.DatasetError, OSError)):
                    self.collect()
                self.assertFalse(self.destination.exists())
                self.assertTrue(any((self.folder / '.transfers').iterdir()))
                self.assertEqual(list(self.destination.parent.glob('.incoming-*')), [])

    def test_capture_deadline_waits_without_killing_and_preserves_failed_attempt(self):
        attempts = []
        for returncode in (2, 0):
            with self.subTest(returncode=returncode):
                process = self.completed_process(returncode)
                def acquire(command, **kwargs):
                    self.assertTrue(kwargs['start_new_session'])
                    self.assertNotIn('timeout', kwargs)
                    output = self.native_output(command)
                    attempts.append(sorted((self.folder / '.transfers').iterdir())[-1])
                    self.create_native_capture(output, complete=False)
                    def wait(timeout=None):
                        self.assertFalse(kwargs['stdout'].closed)
                        self.assertFalse(self.destination.exists())
                        self.assertTrue((output / 'depth/Frame0.exr').is_file())
                        if timeout is not None:
                            self.assertEqual(timeout, 180)
                            raise subprocess.TimeoutExpired(command, timeout)
                        self.assertTrue(any('Waiting for camera/Eyes to release buffers' in item
                                            for item in self.progress))
                        if returncode == 0:
                            (output / '.capture_in_progress').rmdir()
                            (output / 'capture_manifest.yaml').write_text('status: complete\n')
                        else:
                            kwargs['stdout'].write('native request timed out; buffer cleanup finished\n')
                        return returncode
                    process.wait.side_effect = wait
                    return process
                with patch('camera_runtime.subprocess.Popen', side_effect=acquire), \
                     patch('camera_runtime.subprocess.run', side_effect=AssertionError('capture must not use run')):
                    if returncode:
                        with self.assertRaisesRegex(data.DatasetError, 'buffer cleanup finished'):
                            self.collect()
                        self.assertFalse(self.destination.exists())
                        self.assertTrue(attempts[-1].is_dir())
                    else:
                        self.collect()
                self.assertEqual(process.wait.call_args_list, [call(timeout=180), call()])
                process.kill.assert_not_called()
                process.terminate.assert_not_called()
        self.assertNotEqual(attempts[0], attempts[1])
        self.assertTrue((attempts[0] / 'capture/depth/Frame0.exr').is_file())
        self.assertFalse(attempts[1].exists())
        self.assertTrue((self.destination / 'capture_manifest.yaml').is_file())



class DockerTransferTests(unittest.TestCase):
    def setUp(self):
        self.client = camera_runtime.HostCameraCapture(Path('/tmp/artifacts'), container='eyes-test')
        self.runtime = MagicMock(container='eyes-test')

    def test_helper_is_installed_outside_artifacts(self):
        self.assertEqual(self.client.program_path(self.runtime), '/root/bin_filter_offline/native/bin_capture')
        self.runtime.to_runtime.assert_not_called()

    def test_temporary_directory_is_validated_before_use(self):
        for name in ('/tmp/bin-model-capture.example', '/', '/root/artifacts', '/tmp/bin-model-capture.x/../y'):
            result = subprocess.CompletedProcess([], 0, name + '\n', '')
            with self.subTest(name=name), patch('camera_runtime.subprocess.run', return_value=result) as run:
                if name == '/tmp/bin-model-capture.example':
                    self.assertEqual(self.client.create_transfer(self.runtime), Path(name))
                else:
                    with self.assertRaises(data.DatasetError):
                        self.client.create_transfer(self.runtime)
                self.assertEqual(run.call_args.args[0][:4], ['docker', 'exec', 'eyes-test', 'mktemp'])

    def test_copy_and_cleanup_keep_paths_as_literal_arguments(self):
        result = subprocess.CompletedProcess([], 0, '', '')
        with patch('camera_runtime.subprocess.run', return_value=result) as run:
            self.client.copy_transfer(self.runtime, Path('/tmp/bin-model-capture.example'), Path('/tmp/local folder'))
            self.assertEqual(run.call_args.args[0], ['docker', 'cp',
                'eyes-test:/tmp/bin-model-capture.example/capture', '/tmp/local folder'])
            self.client.remove_transfer(self.runtime, Path('/tmp/bin-model-capture.example'))
            self.assertEqual(run.call_args.args[0], ['docker', 'exec', 'eyes-test', 'rm', '-rf', '--',
                                                   '/tmp/bin-model-capture.example'])

if __name__ == '__main__':
    unittest.main()
