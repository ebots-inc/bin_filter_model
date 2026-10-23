import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as data
from setup_operations import Operations


class WorkerOperationsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = Path(__file__).resolve().parents[2] / 'ebots_system_artifacts/artifacts_cell_4'
        if not source.is_dir():
            self.skipTest('Cell 4 collection helper is not checked out')
        self.artifacts = self.root / 'artifacts'
        for name in ('prepare_session.py', 'recipe.yaml', 'bin_config.yaml'):
            destination = self.artifacts / 'data_saving/bin_model' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / 'data_saving/bin_model' / name, destination)
        recipe = data.read_yaml(self.artifacts / 'data_saving/bin_model/recipe.yaml')
        for key in ('camera_calibration', 'robot_calibration'):
            destination = self.artifacts / recipe[key]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / recipe[key], destination)
        destination = self.artifacts / 'workflow_configs/bin_calibration.yaml'
        destination.parent.mkdir(parents=True)
        shutil.copyfile(source / 'workflow_configs/bin_calibration.yaml', destination)
        self.original_workflow = destination.read_bytes()
        self.operations = Operations(self.artifacts, self.root / 'sessions', self.root / 'models', self.root / 'pipeline.yaml')

    def execute(self, command, **values):
        return self.operations.execute({'command': command, **values}, lambda _: None, lambda: {})

    def test_readable_prepare_catalog_and_no_sequence_mutation(self):
        prepared = self.execute('prepare', name='resistor-baseline')
        folder = Path(prepared['session_path'])
        self.assertEqual(folder.name, 'resistor-baseline')
        self.assertEqual(len(prepared['session']['captures']), 170)
        self.assertEqual(len(prepared['session']['stages']), 26)
        captures = {entry['id']: entry for entry in prepared['session']['captures']}
        loaded = [stage for stage in prepared['session']['stages']
                  if any(item['label'] == 'loaded' for item in stage['bins'].values())]
        self.assertEqual(len(loaded), 24)
        groups = set()
        for stage in loaded:
            self.assertEqual(stage['rounds'], 1)
            self.assertEqual(stage['arrangement_count'], 3)
            self.assertEqual(len(stage['capture_ids']), 5)
            entries = [captures[capture_id] for capture_id in stage['capture_ids']]
            self.assertEqual(sorted(entry['exposure_ms'] for entry in entries), [4, 8, 12, 16, 20])
            self.assertEqual(len({entry['group_id'] for entry in entries}), 1)
            groups.add(entries[0]['group_id'])
            if any(item.get('part_type') == 'resistor' for item in stage['bins'].values()):
                self.assertTrue(all(item['label'] == 'loaded' and item['part_count'] == 1
                                    for item in stage['bins'].values()))
        self.assertEqual(len(groups), len(loaded))
        for name in prepared['session']['bin_ids']:
            walls = [stage['bins'][name] for stage in loaded
                     if stage['bins'][name].get('placement') == 'wall']
            self.assertEqual(len(walls), 12)
            for side in ('left', 'right', 'bottom', 'top'):
                self.assertEqual(sum(item.get('wall_side', 'left') == side for item in walls), 3)
        catalog = self.execute('catalog')
        self.assertIn(str(folder), catalog['sessions'])
        self.assertFalse((folder / 'collect.xml').exists())
        self.assertFalse((self.artifacts / 'sequence').exists())
        self.assertEqual((self.artifacts / 'workflow_configs/bin_calibration.yaml').read_bytes(), self.original_workflow)
        with self.assertRaises(ValueError):
            self.execute('prepare', name='resistor-baseline')

    def test_host_preparation_uses_deployed_calibration_and_identical_eyes_settings(self):
        self.operations = Operations(self.artifacts, self.root / 'sessions', self.root / 'models',
                                     self.root / 'pipeline.yaml', runtime_root='/root/data/bin_model')
        prepared = self.execute('prepare', name='local-pc-session')
        folder = Path(prepared['session_path'])
        session = prepared['session']
        self.assertEqual(folder, self.root / 'sessions/local-pc-session')
        self.assertEqual(session['runtime_path'], '/root/data/bin_model/local-pc-session')
        for exposure in (4, 8, 12, 16, 20):
            local = folder / 'configs' / f'depth_{exposure:02d}ms.yaml'
            settings = data.read_yaml(local)
            parameters = settings['tasks'][0]['steps'][0]['parameters']
            self.assertEqual(parameters['camera_calibration_path'],
                             '/root/artifacts/' + self.operations.recipe['camera_calibration'])
            self.assertEqual(parameters['robot_calibration_path'],
                             '/root/artifacts/' + self.operations.recipe['robot_calibration'])
            shared = self.artifacts / 'config/parts/data_saving/bin_model/sessions' / folder.name / local.name
            self.assertEqual(local.read_bytes(), shared.read_bytes())
        for name in ('camera', 'robot'):
            self.assertEqual(session['calibration'][name],
                             {'artifact_path': self.operations.recipe[name + '_calibration']})
        self.assertFalse((folder / 'calibration').exists())
        self.assertFalse((self.artifacts / 'data_saving/bin_model/sessions' / folder.name / 'calibration').exists())
        self.assertEqual((self.artifacts / 'workflow_configs/bin_calibration.yaml').read_bytes(), self.original_workflow)

    def test_requested_exposures_pair_spotlight_and_depth_for_new_session(self):
        # A station recipe saved by an earlier release must not keep new
        # captures at the old fixed spotlight exposure.
        self.operations.recipe['spotlight_exposure_ms'] = 10.
        prepared = self.execute('prepare', name='paired-exposures', exposures_ms=[4., 7.5, 20.])
        folder = Path(prepared['session_path'])
        session = prepared['session']
        self.assertEqual(sorted({entry['exposure_ms'] for entry in session['captures']}), [4., 7.5, 20.])
        self.assertNotIn('spotlight_exposure_ms', data.read_yaml(folder / 'recipe.yaml'))
        self.assertEqual(self.operations.recipe['spotlight_exposure_ms'], 10.)
        for entry in session['captures']:
            self.assertEqual(entry['spotlight_exposure_ms'], entry['exposure_ms'])
            filename = Path(entry['part_name']).name + '.yaml'
            image = data.read_yaml(folder / 'configs' / filename)['image']
            expected = [entry['exposure_ms']] * 2
            self.assertEqual(image['exposure_time_spot_light'], expected)
            self.assertEqual(image['exposure_time_structure_light'], expected)
            shared = self.artifacts / 'config/parts' / (entry['part_name'] + '.yaml')
            self.assertEqual(data.read_yaml(shared)['image'], image)

    def test_activation_passes_runtime_path_mapping_without_changing_local_paths(self):
        mapper = Mock(side_effect=lambda path: '/root/artifacts/models/' + Path(path).name)
        self.operations = Operations(self.artifacts, self.root / 'sessions', self.root / 'models',
                                     self.root / 'pipeline.yaml', runtime_path=mapper)
        prepared = self.execute('prepare', name='activate-local')
        candidate = Path(prepared['session_path']) / 'candidate'
        reload_runtime = Mock()
        with patch('setup_operations.deployment.activate', return_value={'runtime_verified': True}) as activate:
            result = self.operations.execute({'command': 'activate', 'candidate_path': str(candidate)},
                                             lambda _: None, reload_runtime)
        self.assertTrue(result['runtime_verified'])
        activate.assert_called_once_with(candidate, self.root / 'models', self.root / 'pipeline.yaml',
                                         reload_runtime, runtime_path=mapper)

    def test_quick_check_resolves_configured_runtime_model_to_host_before_validation(self):
        runtime_model = Path('/root/artifacts/models/current/model.json')
        local_model = self.artifacts / 'models/current/model.json'
        local_model.parent.mkdir(parents=True)
        local_model.write_text('{}')
        mapper = Mock(return_value=local_model)
        self.operations = Operations(self.artifacts, self.root / 'sessions', self.root / 'models',
                                     self.root / 'pipeline.yaml', model_path_to_host=mapper)
        self.operations.config.write_text(f'bin_filter_model_path: {runtime_model}\n')
        prepared = self.execute('prepare', quick=True)
        self.operations.collection.verify_session = Mock()
        commands = []
        def run(arguments, progress):
            commands.append(arguments)
            if arguments[0] == 'import-session':
                Path(arguments[arguments.index('--dataset') + 1]).mkdir()
            else:
                destination = Path(arguments[arguments.index('--output') + 1])
                destination.mkdir()
                (destination / 'acceptance.json').write_text('{"purpose":"quick_check","status":"passed"}')
        self.operations.run_cli = run
        result = self.execute('quick_check', session_path=prepared['session_path'])
        mapper.assert_called_once_with(runtime_model)
        validation = next(arguments for arguments in commands if arguments[0] == 'validate-production')
        self.assertEqual(validation[validation.index('--model') + 1], local_model)
        self.assertEqual(result['acceptance']['status'], 'passed')

    def test_quick_check_prepares_only_empty_test_captures_without_fit(self):
        prepared = self.execute('prepare', quick=True)
        self.assertTrue(Path(prepared['session_path']).name.startswith('bin-check-'))
        captures = prepared['session']['captures']
        self.assertEqual(len(captures), 15)
        self.assertTrue(all(entry['split'] == 'test' and set(entry['bins'].values()) == {'empty'} for entry in captures))

    def test_worker_rejects_paths_outside_configured_storage(self):
        with self.assertRaisesRegex(data.DatasetError, 'outside'):
            self.execute('review', session_path=str(self.root))
        with self.assertRaisesRegex(data.DatasetError, 'Unknown'):
            self.execute('shell', command_text='arbitrary command')

    def test_missing_capture_completion_blocks_build_before_spawning_tool(self):
        prepared = self.execute('prepare')
        self.operations.run_cli = lambda *args: self.fail('Build must not run before complete captures')
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.execute('build', session_path=prepared['session_path'])

    def test_rebuild_preserves_candidate_and_selects_latest_result(self):
        prepared = self.execute('prepare', name='rebuild')
        folder = Path(prepared['session_path'])
        self.operations.collection.verify_session = lambda _: None
        def build(arguments, progress):
            candidate = Path(arguments[arguments.index('--output') + 1])
            name = arguments[arguments.index('--model-name') + 1]
            (candidate / 'production_validation').mkdir(parents=True)
            (candidate / 'production_validation/acceptance.json').write_text(json.dumps({'model_name': name}))
        self.operations.run_cli = build
        first = self.execute('build', session_path=str(folder))
        original = (folder / 'candidate/production_validation/acceptance.json').read_bytes()
        second = self.execute('build', session_path=str(folder))
        self.assertNotEqual(first['candidate_path'], second['candidate_path'])
        self.assertEqual(second['acceptance']['model_name'], 'rebuild-02')
        self.assertEqual((folder / 'candidate/production_validation/acceptance.json').read_bytes(), original)
        self.assertEqual(self.operations.candidate(folder), Path(second['candidate_path']))
        self.assertEqual(self.execute('review', session_path=str(folder))['acceptance'], second['acceptance'])

    def test_quick_check_can_retry_after_replay_startup_failure(self):
        prepared = self.execute('prepare', quick=True)
        folder = Path(prepared['session_path'])
        model = self.root / 'active-model.json'
        model.write_text('{}')
        self.operations.config.write_text(f'bin_filter_model_path: {model}\n')
        self.operations.collection.verify_session = lambda _: None
        fail = True
        def replay(arguments, progress):
            if arguments[0] == 'import-session':
                destination = Path(arguments[arguments.index('--dataset') + 1])
                destination.mkdir()
                (destination / 'evidence').write_text('preserved')
            elif fail:
                raise data.DatasetError('Replay unavailable')
            else:
                destination = Path(arguments[arguments.index('--output') + 1])
                destination.mkdir()
                (destination / 'acceptance.json').write_text('{"purpose":"quick_check","status":"passed"}')
        self.operations.run_cli = replay
        with self.assertRaisesRegex(data.DatasetError, 'Replay unavailable'):
            self.execute('quick_check', session_path=str(folder))
        fail = False
        result = self.execute('quick_check', session_path=str(folder))
        self.assertEqual((folder / 'quick_check/attempt-01/dataset/evidence').read_text(), 'preserved')
        self.assertTrue((folder / 'quick_check/attempt-02/results/acceptance.json').is_file())
        self.assertEqual(self.execute('review', session_path=str(folder))['acceptance'], result['acceptance'])


if __name__ == '__main__':
    unittest.main()
