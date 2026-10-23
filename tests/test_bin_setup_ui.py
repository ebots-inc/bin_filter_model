"""Standalone controller tests with temporary sessions and no ROS/camera calls."""
from pathlib import Path
import json
import shutil
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'bin_filter_offline'))

import bin_setup
import dataset
from setup_operations import Operations


class StandaloneControllerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='standalone_bin_setup_')
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.artifacts = root / 'artifacts'
        self.data_root = root / 'sessions'
        self.data_root.mkdir()
        source = ROOT / 'ebots_system_artifacts/artifacts_cell_4'
        helper = source / 'data_saving/bin_model'
        recipe = dataset.read_yaml(helper / 'recipe.yaml')
        for relative in ('data_saving/bin_model/prepare_session.py', 'data_saving/bin_model/recipe.yaml',
                         'workflow_configs/bin_calibration.yaml',
                         recipe['camera_calibration'], recipe['robot_calibration'], recipe['bin_config']):
            target = self.artifacts / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        self.ops = Operations(self.artifacts, self.data_root, root / 'models', root / 'pipeline.yaml')
        self.ops.recipe['exposure_ms'] = [4, 8]
        self.captured = []
        self.verified = []
        self.before_capture = None
        self.before_verify = None
        self.connection = mock.Mock(return_value={'ready': True, 'checks': []})
        self.ops.collection.verify_capture = self.verify_capture
        self.controller = self.new_controller()
        self.controller.submit({'command': 'prepare', 'name': 'operator-baseline'})
        self.idle()
        self.assertIsNone(self.controller.error)
        self.folder = self.controller.folder
        self.connection.reset_mock()

    def new_controller(self):
        controller = bin_setup.Controller(self.ops, self.capture, mock.Mock(), check_camera=self.connection)
        self.addCleanup(controller.pool.shutdown, wait=True)
        return controller

    def idle(self, controller=None):
        controller = controller or self.controller
        controller.pool.submit(lambda: None).result(timeout=10)
        self.assertFalse(controller.busy)

    def ready(self, controller=None):
        state = (controller or self.controller).state()
        return {'command': 'capture_stage', 'stage_id': state['current_stage']['id'],
                'ready_token': state['ready_token']}

    def capture(self, folder, session, entry, destination, progress):
        if destination.parent.name == 'captures':
            stage = next(stage for stage in session['stages'] if stage['id'] == entry['stage_id'])
            confirmation = json.loads((folder / stage['confirmation_path']).read_text())
            self.assertEqual(confirmation['session_id'], session['session_id'])
            self.assertEqual(confirmation['stage_id'], stage['id'])
            self.assertTrue(confirmation['gate_token'])
        self.captured.append((entry['id'], destination))
        if self.before_capture:
            self.before_capture(entry, destination)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / 'capture_manifest.yaml').write_text('status: complete\n')
        (destination / 'raw_evidence.txt').write_text(f'acquisition {len(self.captured)}')

    def verify_capture(self, folder, session, entry):
        self.assertTrue((folder / entry['path'] / 'capture_manifest.yaml').is_file())
        if self.before_verify:
            self.before_verify(entry)
        self.verified.append(entry['id'])

    def hold_first_capture(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def wait_once(entry, destination):
            if len(self.captured) == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('Test did not release acquisition')
        self.before_capture = wait_once
        return entered, release

    def test_ready_is_consumed_once_and_next_stage_needs_new_confirmation(self):
        request = self.ready()
        entered, release = self.hold_first_capture()
        self.controller.submit(request)
        self.assertTrue(entered.wait(5))
        self.assertTrue(self.controller.state()['busy'])
        with self.assertRaisesRegex(dataset.DatasetError, 'still running'):
            self.controller.submit(request)
        with self.assertRaisesRegex(dataset.DatasetError, 'still running'):
            self.controller.submit({'command': 'preview'})
        release.set()
        self.idle()
        state = self.controller.state()
        self.assertIsNone(state['error'])
        self.assertEqual(state['current_stage']['id'], 'empty_validation')
        self.assertEqual(len(self.captured), 14)
        self.assertEqual(len(self.verified), 14)
        self.assertFalse((self.folder / state['current_stage']['confirmation_path']).exists())
        with self.assertRaisesRegex(dataset.DatasetError, 'scene changed'):
            self.controller.submit(request)
        session = state['session']
        self.assertEqual(session['collection_interface'], 'standalone')
        self.assertFalse((self.folder / 'collect.xml').exists())
        self.assertTrue((self.artifacts / 'workflow_configs/bin_calibration.yaml').is_file())

    def test_ready_sweep_delivers_matching_spotlight_and_depth_exposures(self):
        pairs = []
        def check_settings(entry, destination):
            pairs.append((entry['exposure_ms'], entry['spotlight_exposure_ms']))
            config = dataset.read_yaml(self.folder / 'configs' / (Path(entry['part_name']).name + '.yaml'))
            self.assertEqual(config['image']['exposure_time_spot_light'], [entry['exposure_ms']] * 2)
            self.assertEqual(config['image']['exposure_time_structure_light'], [entry['exposure_ms']] * 2)
        self.before_capture = check_settings
        self.controller.submit(self.ready())
        self.idle()
        self.assertIsNone(self.controller.state()['error'])
        self.assertEqual(set(pairs), {(4., 4.), (8., 8.)})
        self.assertEqual(len(pairs), 14)

    def test_stop_preserves_completed_capture_and_retry_retakes_entire_arrangement(self):
        request = self.ready()
        entered, release = self.hold_first_capture()
        self.controller.submit(request)
        self.assertTrue(entered.wait(5))
        self.controller.submit({'command': 'stop'})
        release.set()
        self.idle()
        state = self.controller.state()
        self.assertEqual(state['result'], {'stopped': True})
        self.assertEqual(state['current_stage']['id'], request['stage_id'])
        self.assertEqual(state['completed'], 1)
        self.assertEqual(len(self.captured), 1)
        first_path = self.captured[0][1]
        original = (first_path / 'raw_evidence.txt').read_text()
        self.controller.submit(self.ready())
        self.idle()
        state = self.controller.state()
        self.assertEqual(state['current_stage']['id'], 'empty_validation')
        self.assertEqual(state['completed'], 14)
        self.assertEqual(len(self.captured), 15)
        archive = self.folder / 'previous_attempts' / request['stage_id'] / 'attempt-01'
        self.assertEqual((archive / first_path.name / 'raw_evidence.txt').read_text(), original)
        old_confirmation = json.loads((archive / 'confirmation.json').read_text())
        current_confirmation = json.loads((self.folder / 'stage_confirmations' / (request['stage_id'] + '.json')).read_text())
        self.assertEqual(old_confirmation['gate_token'], request['ready_token'])
        self.assertNotEqual(old_confirmation['gate_token'], current_confirmation['gate_token'])
        self.assertNotEqual((first_path / 'raw_evidence.txt').read_text(), original)
        with self.assertRaisesRegex(dataset.DatasetError, 'no capture sweep'):
            self.controller.submit({'command': 'stop'})

    def test_restart_opens_saved_session_at_next_confirmed_checkpoint(self):
        request = self.ready()
        self.controller.submit(request)
        self.idle()
        resumed = self.new_controller()
        state = resumed.state()
        self.assertEqual(resumed.folder, self.folder)
        self.assertEqual(state['completed'], 14)
        self.assertEqual(state['current_stage']['id'], 'empty_validation')
        self.assertNotEqual(state['ready_token'], request['ready_token'])
        self.assertEqual(len(self.captured), 14)
        with self.assertRaisesRegex(dataset.DatasetError, 'scene changed'):
            resumed.submit(request)
        resumed.submit(self.ready(resumed))
        self.idle(resumed)
        self.assertEqual(resumed.state()['completed'], 20)
        self.assertEqual(resumed.state()['current_stage']['id'], 'all_resistor_center_arrangement_01')

    def prepare_loaded_arrangements(self):
        self.controller.submit({'command': 'prepare', 'name': 'loaded-arrangements',
                                'exposures_ms': [4, 8, 12, 16, 20]})
        self.idle()
        self.assertIsNone(self.controller.error)
        self.folder = self.controller.folder
        for _ in range(2):
            self.assertTrue(all(item['label'] == 'empty'
                                for item in self.controller.state()['current_stage']['bins'].values()))
            self.controller.submit(self.ready())
            self.idle()
        self.assertEqual(self.controller.state()['current_stage']['id'], 'all_resistor_center_arrangement_01')
        self.assertEqual(len(self.captured), 50)

    def test_one_ready_captures_only_one_loaded_arrangement_and_requires_a_new_gate(self):
        self.prepare_loaded_arrangements()
        first = self.controller.state()['current_stage']
        request = self.ready()
        self.controller.submit(request)
        self.idle()
        state = self.controller.state()
        self.assertIsNone(state['error'])
        second = state['current_stage']
        self.assertEqual([capture_id for capture_id, _ in self.captured[50:]], first['capture_ids'])
        self.assertEqual(len(self.captured), 55)
        self.assertEqual(second['id'], 'all_resistor_center_arrangement_02')
        self.assertEqual(second['arrangement_index'], 2)
        self.assertEqual(second['bins'], first['bins'])
        self.assertTrue(all(item['label'] == 'loaded' and item['part_count'] == 1
                            for item in first['bins'].values()))
        self.assertEqual(second['completed'], 0)
        self.assertFalse((self.folder / second['confirmation_path']).exists())
        with self.assertRaisesRegex(dataset.DatasetError, 'scene changed'):
            self.controller.submit(request)
        resumed = self.new_controller()
        self.assertEqual(resumed.state()['current_stage']['id'], second['id'])
        self.assertEqual(len(self.captured), 55)
        resumed.submit(self.ready(resumed))
        self.idle(resumed)
        self.assertEqual(len(self.captured), 60)
        self.assertEqual(resumed.state()['current_stage']['id'], 'all_resistor_center_arrangement_03')

    def test_loaded_stop_retry_retakes_only_current_arrangement_and_preserves_prior_one(self):
        self.prepare_loaded_arrangements()
        first = self.controller.state()['current_stage']
        self.controller.submit(self.ready())
        self.idle()
        original_paths = [self.folder / first['confirmation_path'],
                          self.folder / 'completed_stages' / (first['id'] + '.json')]
        for _, path in self.captured[50:]:
            original_paths.extend(item for item in path.iterdir() if item.is_file())
        original = {path: path.read_bytes() for path in original_paths}
        second = self.controller.state()['current_stage']
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def pause_first(entry, destination):
            if entry['id'] == second['capture_ids'][0]:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('Test did not release loaded acquisition')
        self.before_capture = pause_first
        self.controller.submit(self.ready())
        self.assertTrue(entered.wait(5))
        self.controller.submit({'command': 'stop'})
        release.set()
        self.idle()
        self.assertEqual(self.controller.state()['result'], {'stopped': True})
        self.assertEqual(self.controller.state()['current_stage']['id'], second['id'])
        self.assertEqual(self.controller.state()['current_stage']['completed'], 1)
        partial = self.captured[-1][1]
        partial_bytes = (partial / 'raw_evidence.txt').read_bytes()
        self.before_capture = None
        self.controller.submit(self.ready())
        self.idle()
        self.assertIsNone(self.controller.error)
        self.assertEqual(self.controller.state()['current_stage']['id'], 'all_resistor_center_arrangement_03')
        self.assertEqual([capture_id for capture_id, _ in self.captured[-5:]], second['capture_ids'])
        self.assertEqual(self.controller.state()['completed'], 60)
        self.assertEqual({path: path.read_bytes() for path in original}, original)
        self.assertFalse((self.folder / 'previous_attempts' / first['id']).exists())
        archived = self.folder / 'previous_attempts' / second['id'] / 'attempt-01'
        self.assertEqual((archived / partial.name / 'raw_evidence.txt').read_bytes(), partial_bytes)
        self.assertEqual([item.name for item in (self.folder / 'previous_attempts').iterdir()], [second['id']])

    def test_each_wall_side_waits_for_ready_and_captures_its_own_region(self):
        self.prepare_loaded_arrangements()
        for _ in range(3):
            self.controller.submit(self.ready())
            self.idle()
        regions = {'left': [.02, .35, .20, .65], 'right': [.80, .35, .98, .65],
                   'bottom': [.35, .02, .65, .20], 'top': [.35, .80, .65, .98]}
        for side, region in regions.items():
            for arrangement in range(1, 4):
                with self.subTest(side=side, arrangement=arrangement):
                    stage = self.controller.state()['current_stage']
                    suffix = '' if side == 'left' else '_' + side
                    self.assertEqual(stage['id'], f'all_resistor_wall{suffix}_arrangement_{arrangement:02d}')
                    for item in stage['bins'].values():
                        self.assertEqual(item.get('wall_side', 'left'), side)
                        self.assertEqual(item['target_region_fraction'], region)
                        self.assertEqual(item['part_count'], 1)
                        self.assertEqual(item['label'], 'loaded')
                    self.assertEqual(stage['completed'], 0)
                    self.assertFalse((self.folder / stage['confirmation_path']).exists())
                    request = self.ready()
                    before = len(self.captured)
                    self.controller.submit(request)
                    self.idle()
                    self.assertIsNone(self.controller.error)
                    self.assertEqual([capture_id for capture_id, _ in self.captured[before:]], stage['capture_ids'])
                    self.assertEqual(len(self.captured), before + 5)
                    with self.assertRaisesRegex(dataset.DatasetError, 'scene changed'):
                        self.controller.submit(request)
        state = self.controller.state()
        self.assertEqual(state['completed'], 125)
        self.assertEqual(state['current_stage']['id'], 'all_resistor_corner_arrangement_01')
        self.assertFalse((self.folder / state['current_stage']['confirmation_path']).exists())

    def test_failed_final_capture_verification_does_not_complete_stage(self):
        request = self.ready()
        first_stage = self.controller.state()['current_stage']
        def fail_last(entry):
            if entry['id'] == first_stage['capture_ids'][-1]:
                raise ValueError('Invalid exposure evidence')
        self.before_verify = fail_last
        self.controller.submit(request)
        self.idle()
        state = self.controller.state()
        self.assertIn('Invalid exposure', state['error'])
        self.assertEqual(state['current_stage']['id'], request['stage_id'])
        self.assertEqual(state['completed'], 14)
        self.assertFalse(state['current_stage']['finished'])
        self.assertFalse((self.folder / 'completed_stages' / (request['stage_id'] + '.json')).exists())
        self.before_verify = None
        self.controller.submit(self.ready())
        self.idle()
        self.assertEqual(self.controller.state()['current_stage']['id'], 'empty_validation')
        self.assertEqual(len(list((self.folder / 'previous_attempts' / request['stage_id'] / 'attempt-01').glob('empty_training_*'))), 14)

    def test_retry_invalidates_a_stale_completion_marker_before_verification(self):
        stage = self.controller.state()['current_stage']
        marker = self.folder / 'completed_stages' / (stage['id'] + '.json')
        marker.parent.mkdir()
        marker.write_text(json.dumps({'stage_id': stage['id'], 'capture_ids': stage['capture_ids']}))
        # A restored session may retain its old checkpoint but lack raw files.
        self.assertFalse(self.controller.state()['current_stage']['finished'])
        def fail_last(entry):
            if entry['id'] == stage['capture_ids'][-1]:
                raise ValueError('Final capture has invalid evidence')
        self.before_verify = fail_last
        self.controller.submit(self.ready())
        self.idle()
        state = self.controller.state()
        self.assertIn('invalid evidence', state['error'])
        self.assertFalse(marker.exists())
        self.assertEqual(state['current_stage']['id'], stage['id'])
        self.assertFalse(state['current_stage']['finished'])

    def test_preview_is_fresh_but_excluded_from_plan_and_ready_confirmation(self):
        original = (self.folder / 'session.json').read_bytes()
        with mock.patch('preview.render', return_value={'preview_png': 'cG5n', 'caption': 'Preview'}) as render:
            self.controller.submit({'command': 'preview'})
            self.idle()
            self.assertIsNone(self.controller.error)
            self.assertEqual(render.call_args.args[0], self.folder / 'previews/preview-01')
            self.controller.submit({'command': 'preview'})
            self.idle()
            self.assertEqual(render.call_args.args[0], self.folder / 'previews/preview-02')
        self.assertEqual(self.controller.state()['completed'], 0)
        self.assertEqual((self.folder / 'session.json').read_bytes(), original)
        self.assertFalse((self.folder / 'stage_confirmations').exists())
        self.assertEqual(len(self.verified), 0)
        self.assertEqual(len(self.captured), 2)
        self.assertEqual(list((self.folder / 'captures').iterdir()), [])

    def test_connection_check_is_available_without_a_session_and_never_captures(self):
        self.controller.folder = None
        self.connection.return_value = {'ready': False, 'message': 'Eyes is unavailable', 'checks': []}
        self.controller.submit({'command': 'check_connection'})
        self.idle()
        self.assertFalse(self.controller.state()['camera_connection']['ready'])
        self.connection.assert_called_once_with(0)
        self.assertEqual(self.captured, [])

    def test_connection_check_uses_selected_scene_without_session(self):
        self.controller.folder = None
        self.controller.submit({'command': 'check_connection', 'scene_id': 4})
        self.idle()
        self.connection.assert_called_once_with(4)
        self.assertEqual(self.controller.state()['camera_connection']['scene_id'], 4)
        self.assertEqual(self.captured, [])

    def test_connection_for_active_session_cannot_be_changed_by_new_setup_selection(self):
        self.controller.submit({'command': 'check_connection', 'scene_id': 4})
        self.idle()
        self.connection.assert_called_once_with(0)
        self.assertEqual(self.controller.state()['camera_connection']['scene_id'], 0)

    def test_opening_session_rechecks_its_scene(self):
        self.controller.submit({'command': 'select', 'session_path': str(self.folder)})
        self.idle()
        self.connection.assert_called_once_with(0)
        self.assertTrue(self.controller.state()['camera_connection']['ready'])

    def test_host_restart_reopens_a_session_selected_by_its_old_container_path(self):
        session_path = self.folder / 'session.json'
        session = json.loads(session_path.read_text())
        session['runtime_path'] = '/root/data/bin_model/' + self.folder.name
        session_path.write_text(json.dumps(session))
        self.controller.selection_file.write_text(json.dumps({'session_path': session['runtime_path']}))
        resumed = self.new_controller()
        self.assertEqual(resumed.folder, self.folder)
        self.assertEqual(resumed.state()['current_stage']['id'], 'empty_training')
        self.assertEqual(self.captured, [])

    def test_failed_preflight_preserves_existing_partial_capture_and_confirmation(self):
        stage = self.controller.state()['current_stage']
        entry = self.controller.state()['session']['captures'][0]
        existing = self.folder / entry['path']
        existing.mkdir()
        (existing / 'evidence.txt').write_text('original')
        confirmation = self.folder / stage['confirmation_path']
        confirmation.parent.mkdir(parents=True)
        confirmation.write_text('original confirmation')
        self.connection.return_value = {'ready': False, 'message': 'No live Eyes server', 'checks': []}
        self.controller.submit(self.ready())
        self.idle()
        self.assertIn('No live Eyes server', self.controller.error)
        self.assertEqual((existing / 'evidence.txt').read_text(), 'original')
        self.assertEqual(confirmation.read_text(), 'original confirmation')
        self.assertFalse((self.folder / 'previous_attempts').exists())
        self.assertEqual(self.captured, [])

    def test_disconnected_preview_does_not_create_preview_or_capture(self):
        self.connection.side_effect = OSError('Camera container is stopped')
        self.controller.submit({'command': 'preview'})
        self.idle()
        self.assertIn('Camera container is stopped', self.controller.error)
        self.assertFalse((self.folder / 'previews').exists())
        self.assertFalse(self.controller.state()['camera_connection']['ready'])
        self.assertEqual(self.captured, [])


if __name__ == '__main__':
    unittest.main()
