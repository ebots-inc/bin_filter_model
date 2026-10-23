import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import host_replay
import docker_runtime


class Runtime:
    def __init__(self, container):
        self.container = container

    def to_runtime(self, path):
        return Path('/runtime') / path.relative_to('/host')

    def command(self, arguments, *, outputs=()):
        self.assert_outputs = outputs
        return ['docker', 'exec', self.container, *arguments]


class HostReplayTests(unittest.TestCase):
    def test_maps_all_paths_but_retains_camera_values_and_bin(self):
        arguments = []
        for option in sorted(host_replay.PATH_OPTIONS):
            arguments += [option, '/host/capture with spaces/' + option[2:]]
        arguments += ['--exposure-ms', '4,4', '--gain', '1,1', '--black-level', '0,0', '--scene-id', '0', '--bin', 'bin_a']
        with patch.dict(os.environ, {'BIN_SETUP_REPLAY_CONTAINER': 'native',
                                    'BIN_SETUP_REPLAY_EXECUTABLE': '/installed/summit_replay'}, clear=True), \
             patch('host_replay.subprocess.run', return_value=subprocess.CompletedProcess([], 7)) as run:
            self.assertEqual(host_replay.main(arguments, Runtime), 7)
        command = run.call_args.args[0]
        self.assertEqual(command[:7], ['docker', 'exec', 'native',
                                      'timeout', '--kill-after=5s', '150s', '/installed/summit_replay'])
        self.assertFalse(any(value.startswith('/host/') for value in command))
        self.assertIn('/runtime/capture with spaces/model', command)
        self.assertEqual(command[-10:], arguments[-10:])

    def test_requests_ownership_handoff_for_only_declared_output_files(self):
        runtime = Runtime('native')
        with patch.dict(os.environ, {'BIN_SETUP_REPLAY_CONTAINER': 'native'}, clear=True), \
             patch('host_replay.subprocess.run', return_value=subprocess.CompletedProcess([], 0)):
            self.assertEqual(host_replay.main(['--model', '/host/model', '--output', '/host/case/result.json',
                                              '--filtered-cloud', '/host/case/filtered.xyz'], lambda _: runtime), 0)
        self.assertEqual(runtime.assert_outputs, ['/runtime/case/result.json', '/runtime/case/filtered.xyz'])

    def test_unmapped_paths_and_duplicate_options_fail_before_start(self):
        for arguments in (['--model', '/not-mounted/model'], ['--bin', 'bin_a', '--bin', 'bin_b'], ['--output']):
            with self.subTest(arguments=arguments), \
                 patch.dict(os.environ, {'BIN_SETUP_REPLAY_CONTAINER': 'native'}, clear=True), \
                 patch('host_replay.subprocess.run') as run:
                self.assertEqual(host_replay.main(arguments, Runtime), 2)
                run.assert_not_called()

    def test_discovers_replay_container_only_when_replay_is_invoked(self):
        with patch.dict(os.environ, {'BIN_SETUP_ARTIFACTS_ROOT': '/host/artifacts'}, clear=True), \
             patch.object(docker_runtime, 'discover_container', return_value='lazy-native') as discover, \
             patch('host_replay.subprocess.run', return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(host_replay.main(['--model', '/host/model'], Runtime), 0)
        discover.assert_called_once_with('perception', Path('/host/artifacts'))
        self.assertIn('lazy-native', run.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
