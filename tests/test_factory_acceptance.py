import json
import importlib.util
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as data
import deployment
from production_validation import assess_result, target_bounds, validate_production


class ProductionAcceptanceTests(unittest.TestCase):
    def test_empty_requires_registration_and_expected_empty_outcome(self):
        result = {'registration': {'attempted': True, 'accepted': True}, 'success': False,
                  'outcome': 'empty', 'candidates_robot_mm': []}
        self.assertEqual(assess_result(result, 'empty'), [])
        self.assertTrue(assess_result(dict(result, outcome='failure'), 'empty'))
        self.assertTrue(assess_result(dict(result, registration={'attempted': True, 'accepted': False}), 'empty'))
        self.assertTrue(assess_result(dict(result, success=True, candidates_robot_mm=[[0, 0, 0]]), 'empty'))

    def test_single_resistor_must_have_target_in_prescribed_region(self):
        config = {'xy_bounds_mm': [[425, -499], [630, -305]]}
        region = target_bounds({'target_region_fraction': [.02, .35, .20, .65]}, config)
        result = {'registration': {'attempted': True, 'accepted': True}, 'success': True,
                  'candidates_robot_mm': [[440, -400, -120]]}
        self.assertEqual(assess_result(result, 'loaded', region), [])
        result['candidates_robot_mm'] = [[550, -400, -120]]
        self.assertTrue(assess_result(result, 'loaded', region))
        with self.assertRaises(data.DatasetError):
            target_bounds({'target_region_fraction': [0, 0, float('nan'), 1]}, config)

    def test_normal_fill_does_not_require_an_invented_part_count(self):
        result = {'registration': {'attempted': True, 'accepted': True}, 'success': True,
                  'candidates_robot_mm': [[1, 2, 3]]}
        self.assertEqual(assess_result(result, 'loaded'), [])
        self.assertTrue(assess_result(result | {'success':'false'}, 'loaded'))
        self.assertTrue(assess_result(result | {'candidates_robot_mm':[1]}, 'loaded'))
        self.assertTrue(assess_result(result | {'registration':{'attempted':'true','accepted':'true'}}, 'loaded'))

    def test_rotated_bin_placement_uses_measured_interior_not_wide_lip_bounds(self):
        placement = data.pose_matrix([500.,-600.,-100.,10.,0.,90.])
        tolerance = np.full((20,20),np.nan)
        tolerance[6:14,6:14] = 1.
        geometry = {'xy_bounds_mm':[[-100,-100],[100,100]],
                    'x_centers_mm':np.arange(-100,110,10).tolist(),
                    'y_centers_mm':np.arange(-100,110,10).tolist(),
                    'tolerance_mm':tolerance.tolist()}
        region = target_bounds({'target_region_fraction':[.02,.35,.20,.65]},
                               {'geometry':geometry,'robot_from_bin':placement.tolist()})
        self.assertEqual(region['frame_id'],'bin')
        np.testing.assert_allclose(region['xy_bounds_mm'],[[-38.4,-12.],[-24.,12.]])
        local = np.array([[-30.,0.,5.]])
        candidate = data.transform_points(local,placement).tolist()
        result = {'registration':{'attempted':True,'accepted':True},'success':True,'candidates_robot_mm':candidate}
        self.assertEqual(assess_result(result,'loaded',region),[])
        result['candidates_robot_mm'] = data.transform_points(np.array([[-80.,0.,5.]]),placement).tolist()
        self.assertTrue(assess_result(result,'loaded',region))

    def test_native_error_and_timeout_preserve_failed_acceptance_and_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, program = root/'pipeline.yaml',root/'replay.py'
            config.write_text('voxel_size_mm: 0.5\n')
            program.write_text('unused mock')
            capture = {'id':'heldout','split':'test','bins':{'bin_a':'empty','bin_b':'empty'},'exposure_ms':4.,
                       'depth_images':[{'path':'a.exr'},{'path':'b.exr'}],
                       'source_metadata':{'requested_settings':{'gain':[1,1],'black_level':[0,0]}}}
            manifest = {'scene_id':0,'captures':[capture],'calibration':{
                'camera':{'path':'camera.yaml'},'robot':{'path':'robot.yaml'}},'synthetic':True}
            model = {'model_type':'offline_height_field_bins','model_name':'test',
                     'bins':{name:{'xy_bounds_mm':[[0,0],[10,10]]} for name in ('bin_a','bin_b')}}
            effects = [subprocess.TimeoutExpired('replay',180,output=b'waiting'),
                       subprocess.CompletedProcess('replay',2,stdout='',stderr='camera profile mismatch')]
            with patch('production_validation.data.load_dataset',return_value=(manifest,{})), \
                 patch('production_validation.model_io.load_model',return_value=model), \
                 patch('production_validation.data.checked_file',side_effect=lambda parent,item: parent/item['path']), \
                 patch('production_validation.subprocess.run',side_effect=effects) as run:
                report = validate_production(root,root/'model.json',config,root/'report',program,quick=True)
            self.assertEqual(run.call_args.args[0][:2], [sys.executable, str(program)])
            self.assertEqual(report['status'],'failed')
            self.assertEqual(len(report['results']),2)
            self.assertTrue((root/'report/acceptance.json').is_file())
            self.assertIn('waiting',(root/'report/heldout/bin_a/replay.log').read_text())
            self.assertIn('camera profile mismatch',(root/'report/heldout/bin_b/replay.log').read_text())
            for name in ('bin_a','bin_b'):
                result=json.loads((root/f'report/heldout/{name}/result.json').read_text())
                self.assertEqual(result['outcome'],'execution_error')
                self.assertFalse(result['registration']['accepted'])

    def test_factory_evidence_is_required_for_every_named_bin_in_each_scene(self):
        helper = Path(__file__).resolve().parents[2] / 'ebots_system_artifacts/artifacts_cell_4/data_saving/bin_model/prepare_session.py'
        spec = importlib.util.spec_from_file_location('factory_arrangement_recipe', helper)
        collection = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collection)
        for scene, names in ((1, ['supply-tray']), (4, ['incoming.parts', 'resistors', 'rejects'])):
            with self.subTest(scene=scene), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, program = root / 'pipeline.yaml', root / 'replay.py'
                config.write_text('robot_id: B2\n')
                program.write_text('unused mock')
                recipe = data.read_yaml(helper.parent / 'recipe.yaml')
                recipe.update(bin_ids=names, empty_test_rounds=1)
                recipe['guided']['loaded_arrangements'] = 3
                recipe['guided'].pop('resistor_capture_mode', None)
                stages = [stage for stage in collection.guided_stages(recipe) if stage['split'] == 'test']
                self.assertTrue(any(stage['id'] == f'{names[0]}_resistor_center_arrangement_01' for stage in stages))
                self.assertTrue(all(sum(item['label'] == 'loaded' for item in stage['bins'].values()) == 1
                                    for stage in stages if 'resistor' in stage['id']))
                captures = []
                for stage in stages:
                    stage_id, bins = stage['id'], stage['bins']
                    stage['capture_ids'] = []
                    for exposure in (4., 20.):
                        capture_id = f'{stage_id}_{exposure:g}'
                        stage['capture_ids'].append(capture_id)
                        captures.append({'id': capture_id, 'stage_id': stage_id, 'split': 'test', 'group_id': f'test_{stage_id}',
                            'bins': {name: item['label'] for name, item in bins.items()}, 'exposure_ms': exposure,
                            'depth_images': [{'path': 'a.exr'}, {'path': 'b.exr'}],
                            'source_metadata': {'requested_settings': {'gain': [1, 1], 'black_level': [0, 0]}}})
                manifest = {'scene_id': scene, 'frame_id': 'A1', 'captures': captures, 'stages': stages,
                    'calibration': {'camera': {'path': 'camera.yaml'}, 'robot': {'path': 'robot.yaml'}}}
                model = {'model_type': 'offline_height_field_bins', 'frame_id': 'A1', 'scene_id': scene,
                         'exposure_settings_ms': [4., 20.], 'bins': {name: {'xy_bounds_mm': [[0, 0], [10, 10]]} for name in names}}
                def replay(command, **kwargs):
                    options = dict(zip(command[2::2], command[3::2]))
                    destination = Path(options['--output'])
                    capture = next(item for item in captures if item['id'] == destination.parent.parent.name)
                    loaded = capture['bins'][options['--bin']] == 'loaded'
                    stage = next(item for item in stages if item['id'] == capture['stage_id'])
                    region = stage['bins'][options['--bin']].get('target_region_fraction', [0, 0, 1, 1])
                    candidate = [(region[0] + region[2]) * 5, (region[1] + region[3]) * 5, 3]
                    destination.write_text(json.dumps({'success': loaded, 'outcome': 'success' if loaded else 'empty',
                        'registration': {'attempted': True, 'accepted': True},
                        'candidates_robot_mm': [candidate] if loaded else []}))
                    self.assertEqual(int(options['--scene-id']), scene)
                    return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
                with patch('production_validation.data.load_dataset', return_value=(manifest, {})), \
                     patch('production_validation.model_io.load_model', return_value=model), \
                     patch('production_validation.data.checked_file', side_effect=lambda parent, item: parent / item['path']), \
                     patch('production_validation.subprocess.run', side_effect=replay):
                    report = validate_production(root, root / 'model.json', config, root / 'complete', program)
                    self.assertEqual(report['status'], 'passed')
                    self.assertTrue(any(item['stage_id'] == 'all_normal_arrangement_03' for item in report['results']))
                    self.assertTrue(any(item['stage_id'] == 'all_full_arrangement_03' for item in report['results']))
                    def at_wall(capture, side):
                        stage = next(item for item in stages if item['id'] == capture['stage_id'])
                        item = stage['bins'][names[-1]]
                        return item.get('placement') == 'wall' and item.get('wall_side', 'left') == side
                    # Keep every planned stage. Other walls and repeated
                    # arrangements cannot stand in for a missing wall side.
                    for side in ('left', 'right', 'bottom', 'top'):
                        manifest['captures'] = [capture for capture in captures if not at_wall(capture, side)]
                        report = validate_production(root, root / 'model.json', config, root / f'missing-wall-{side}', program)
                        self.assertEqual(report['status'], 'failed')
                        self.assertEqual(set(report['missing_evidence']),
                                         {f'{names[-1]}:wall:{side} at 4 ms', f'{names[-1]}:wall:{side} at 20 ms'})
                    manifest['captures'] = [capture for capture in captures
                                            if not (at_wall(capture, 'right') and capture['exposure_ms'] == 20.)]
                    report = validate_production(root, root / 'model.json', config, root / 'missing-right-20', program)
                    self.assertEqual(report['status'], 'failed')
                    self.assertEqual(report['missing_evidence'], [f'{names[-1]}:wall:right at 20 ms'])
                    # Frozen left-only plans remain valid and retain their
                    # original acceptance labels without new wall requirements.
                    legacy = [stage for stage in stages if not any(
                        item.get('placement') == 'wall' and item.get('wall_side', 'left') != 'left'
                        for item in stage['bins'].values())]
                    legacy_ids = {stage['id'] for stage in legacy}
                    legacy_captures = [capture for capture in captures if capture['stage_id'] in legacy_ids]
                    manifest.update(stages=legacy, captures=legacy_captures)
                    report = validate_production(root, root / 'model.json', config, root / 'legacy-left-only', program)
                    self.assertEqual(report['status'], 'passed')
                    manifest['captures'] = [capture for capture in legacy_captures
                                            if not (at_wall(capture, 'left') and capture['exposure_ms'] == 20.)]
                    report = validate_production(root, root / 'model.json', config, root / 'legacy-missing-left-20', program)
                    self.assertEqual(report['status'], 'failed')
                    self.assertEqual(report['missing_evidence'], [f'{names[-1]}:wall at 20 ms'])
                    removed = f'{names[-1]}_resistor_corner_arrangement_'
                    manifest['stages'] = [stage for stage in stages if not stage['id'].startswith(removed)]
                    manifest['captures'] = [capture for capture in captures if not capture['stage_id'].startswith(removed)]
                    report = validate_production(root, root / 'model.json', config, root / 'missing', program)
                    self.assertEqual(report['status'], 'failed')
                    self.assertEqual(set(report['missing_evidence']), {f'{names[-1]}:corner at 4 ms', f'{names[-1]}:corner at 20 ms'})
                    manifest['captures'][0]['bins'].pop(names[-1])
                    with self.assertRaises(data.DatasetError):
                        validate_production(root, root / 'model.json', config, root / 'bad-labels', program)

    def test_simultaneous_resistors_require_each_bins_own_target_registration_and_empty_evidence(self):
        helper = Path(__file__).resolve().parents[2] / 'ebots_system_artifacts/artifacts_cell_4/data_saving/bin_model/prepare_session.py'
        spec = importlib.util.spec_from_file_location('simultaneous_acceptance_recipe', helper)
        collection = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collection)
        names = ['incoming.parts', 'resistors', 'rejects']
        recipe = data.read_yaml(helper.parent / 'recipe.yaml')
        recipe.update(bin_ids=names, empty_test_rounds=1)
        stages = [stage for stage in collection.guided_stages(recipe) if stage['split'] == 'test']
        resistor_stages = [stage for stage in stages if 'resistor' in stage['id']]
        self.assertEqual(len(resistor_stages), 18)
        self.assertTrue(all(stage['id'].startswith('all_resistor_') for stage in resistor_stages))
        self.assertTrue(all(item['label'] == 'loaded' and item['part_count'] == 1
                            for stage in resistor_stages for item in stage['bins'].values()))
        captures = []
        for stage in stages:
            stage['capture_ids'] = []
            for exposure in (4., 20.):
                capture_id = f"{stage['id']}_{exposure:g}"
                stage['capture_ids'].append(capture_id)
                captures.append({'id': capture_id, 'stage_id': stage['id'], 'split': 'test',
                    'group_id': 'simultaneous_' + stage['id'], 'exposure_ms': exposure,
                    'bins': {name: item['label'] for name, item in stage['bins'].items()},
                    'depth_images': [{'path': 'a.exr'}, {'path': 'b.exr'}],
                    'source_metadata': {'requested_settings': {'gain': [1, 1], 'black_level': [0, 0]}}})
        manifest = {'scene_id': 4, 'frame_id': 'A1', 'stages': stages, 'captures': captures,
            'calibration': {'camera': {'path': 'camera.yaml'}, 'robot': {'path': 'robot.yaml'}}}
        model = {'model_type': 'offline_height_field_bins', 'frame_id': 'A1', 'scene_id': 4,
            'exposure_settings_ms': [4., 20.],
            'bins': {name: {'xy_bounds_mm': [[index * 20, 0], [index * 20 + 10, 10]]}
                     for index, name in enumerate(names)}}
        failure = None
        failed_capture = 'all_resistor_wall_right_arrangement_01_20'
        def replay(command, **kwargs):
            options = dict(zip(command[2::2], command[3::2]))
            destination = Path(options['--output'])
            capture = next(item for item in captures if item['id'] == destination.parent.parent.name)
            bin_id = options['--bin']
            stage = next(item for item in stages if item['id'] == capture['stage_id'])
            loaded = capture['bins'][bin_id] == 'loaded'
            region = stage['bins'][bin_id].get('target_region_fraction', [0, 0, 1, 1])
            offset = model['bins'][bin_id]['xy_bounds_mm'][0][0]
            candidate = [offset + (region[0] + region[2]) * 5, (region[1] + region[3]) * 5, 3]
            result = {'success': loaded, 'outcome': 'success' if loaded else 'empty',
                'registration': {'attempted': True, 'accepted': True},
                'candidates_robot_mm': [candidate] if loaded else []}
            if bin_id == names[-1] and capture['id'] == failed_capture:
                if failure == 'missing_target':
                    result.update(success=False, outcome='failure', candidates_robot_mm=[])
                elif failure == 'other_bin_target':
                    result['candidates_robot_mm'] = [[candidate[0] - offset, candidate[1], candidate[2]]]
                elif failure == 'registration':
                    result['registration']['accepted'] = False
            destination.write_text(json.dumps(result))
            return subprocess.CompletedProcess(command, 0, stdout='', stderr='')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, program = root / 'pipeline.yaml', root / 'replay.py'
            config.write_text('robot_id: A1\n')
            program.write_text('unused mock')
            with patch('production_validation.data.load_dataset', return_value=(manifest, {})), \
                 patch('production_validation.model_io.load_model', return_value=model), \
                 patch('production_validation.data.checked_file', side_effect=lambda parent, item: parent / item['path']), \
                 patch('production_validation.subprocess.run', side_effect=replay):
                report = validate_production(root, root / 'model.json', config, root / 'complete', program)
                self.assertEqual(report['status'], 'passed')
                self.assertEqual(report['missing_evidence'], [])
                for failure, reason in (('missing_target', 'No pick target'),
                                        ('other_bin_target', 'prescribed resistor placement region'),
                                        ('registration', 'registration was not accepted')):
                    with self.subTest(failure=failure):
                        report = validate_production(root, root / 'model.json', config, root / failure, program)
                        self.assertEqual(report['status'], 'failed')
                        self.assertEqual(report['missing_evidence'], [])
                        failed = [item for item in report['results'] if not item['passed']]
                        self.assertEqual([(item['capture_id'], item['bin_id']) for item in failed],
                                         [(failed_capture, names[-1])])
                        self.assertTrue(any(reason in value for value in failed[0]['reasons']))
                failure = None
                manifest['captures'] = [capture for capture in captures if capture['stage_id'] != 'empty_validation']
                report = validate_production(root, root / 'model.json', config, root / 'missing-empty', program)
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['missing_evidence'], ['empty at 4 ms', 'empty at 20 ms'])


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.candidate = self.root / 'candidate'
        (self.candidate / 'model').mkdir(parents=True)
        (self.candidate / 'model/model.json').write_text('{}')
        (self.candidate / 'production_validation').mkdir()
        self.report = {'status': 'passed', 'purpose': 'factory_acceptance', 'pipeline': 'production_summit',
                       'synthetic': False, 'model_name': 'bin-setup-2026-10-23-01',
                       'missing_evidence': [], 'results': [{'passed': True}]}
        self.report_file = self.candidate / 'production_validation/acceptance.json'
        self.report_file.write_text(json.dumps(self.report))
        self.old_model = self.root / 'old/model.yaml'
        self.old_model.parent.mkdir()
        self.old_model.write_text('old model')
        self.config = self.root / 'pipeline.yaml'
        self.original = ('# Keep station comments\n'
                         f'bin_filter_model_path: {self.old_model}\n'
                         'bin_icp:\n  enabled: true\n  reference_mode: upper_band\n')
        self.config.write_text(self.original)
        self.models = self.root / 'installed'
        self.model = {'model_name': self.report['model_name'], 'model_type': 'calibrated_bin_background'}
        self.loader = patch('deployment.model_io.load_model', return_value=self.model)
        self.loader.start()

    def tearDown(self):
        self.loader.stop()
        self.temporary.cleanup()

    def runtime(self):
        path = data.read_yaml(self.config)['bin_filter_model_path']
        return {'success': True, 'scenes': [{'loaded': True, 'model_path': path,
                'model_name': self.model['model_name'] if Path(path) != self.old_model else 'old'}]}

    def test_activate_and_rollback_verify_runtime_and_preserve_configuration(self):
        replay = self.candidate / 'production_validation/heldout/bin_a/result.json'
        replay.parent.mkdir(parents=True)
        replay.write_text(json.dumps({'model_path': '../../../model/model.json',
                                     'model_path_relative_to': 'result_file'}))
        result = deployment.activate(self.candidate, self.models, self.config, self.runtime)
        self.assertTrue(result['runtime_verified'])
        installed_replay = Path(result['model_path']).parent / 'acceptance/heldout/bin_a/result.json'
        recorded = json.loads(installed_replay.read_text())
        self.assertEqual((installed_replay.parent / recorded['model_path']).resolve(), Path(result['model_path']))
        self.assertIn('reference_mode: measured_rim', self.config.read_text())
        self.assertIn('# Keep station comments', self.config.read_text())
        restored = deployment.rollback(self.models, self.config, self.runtime)
        self.assertEqual(restored['model_path'], str(self.old_model))
        self.assertEqual(self.config.read_text(), self.original)

    def test_wrong_loaded_identity_restores_previous_config_and_runtime(self):
        calls = []
        def runtime():
            calls.append(data.read_yaml(self.config)['bin_filter_model_path'])
            return {'loaded': True, 'model_path': str(self.old_model), 'model_name': 'old'}
        with self.assertRaisesRegex(data.DatasetError, 'previous model restored'):
            deployment.activate(self.candidate, self.models, self.config, runtime)
        self.assertEqual(self.config.read_text(), self.original)
        self.assertEqual(len(calls), 3)
        self.assertFalse((self.models / 'active.json').exists())

    def test_host_install_writes_runtime_path_and_rollback_restores_runtime_identity(self):
        runtime_old = '/root/artifacts/legacy/model.yaml'
        self.original = self.original.replace(str(self.old_model), runtime_old)
        self.config.write_text(self.original)
        mapper = lambda path: Path('/root/artifacts/installed') / path.relative_to(self.models)
        result = deployment.activate(self.candidate, self.models, self.config, self.runtime, runtime_path=mapper)
        host_path = self.models / self.report['model_name'] / 'model.json'
        self.assertEqual(result['model_path'], str(host_path))
        self.assertEqual(result['runtime_model_path'], str(mapper(host_path)))
        self.assertEqual(data.read_yaml(self.config)['bin_filter_model_path'], str(mapper(host_path)))
        self.assertTrue(host_path.is_file())
        self.assertEqual(self.config.stat().st_mode & 0o777, 0o644)
        self.assertEqual((self.models / 'active.json').stat().st_mode & 0o777, 0o644)
        restored = deployment.rollback(self.models, self.config, self.runtime)
        self.assertEqual(restored['runtime_model_path'], runtime_old)
        self.assertEqual(self.config.read_text(), self.original)

    def test_missing_or_synthetic_acceptance_never_changes_active_config(self):
        for changes in ({'synthetic': True}, {'missing_evidence': ['bin_b:corner at 20 ms']},
                        {'pipeline': 'python_filter'}, {'results': []}, {'status': 'failed'}):
            self.report_file.write_text(json.dumps(self.report | changes))
            with self.assertRaises(data.DatasetError):
                deployment.activate(self.candidate, self.models, self.config, self.runtime)
            self.assertEqual(self.config.read_text(), self.original)

    def test_activation_failure_does_not_claim_restore_without_correct_identity(self):
        calls = []
        def wrong_runtime():
            calls.append(None)
            if len(calls) == 1:
                return self.runtime()
            return {'loaded': True, 'model_path': '/wrong/model.json', 'model_name': 'wrong'}
        with self.assertRaisesRegex(data.DatasetError, 'runtime rollback failed'):
            deployment.activate(self.candidate, self.models, self.config, wrong_runtime)
        self.assertEqual(self.config.read_text(), self.original)
        self.assertFalse((self.models / 'active.json').exists())
        # Preserve the failed package at its original path when runtime rollback
        # was not verified: Perception might still be using that directory.
        self.assertTrue((self.models / self.report['model_name']).is_dir())
        self.assertFalse((self.models / 'failed_installations').exists())

    def test_unavailable_preflight_does_not_publish_install_and_same_candidate_can_retry(self):
        def unavailable():
            raise data.DatasetError('Perception reload service is unavailable')
        with self.assertRaisesRegex(data.DatasetError, 'unavailable'):
            deployment.activate(self.candidate, self.models, self.config, unavailable)
        self.assertEqual(self.config.read_text(), self.original)
        self.assertFalse((self.models / self.report['model_name']).exists())
        self.assertFalse((self.models / 'active.json').exists())
        result = deployment.activate(self.candidate, self.models, self.config, self.runtime)
        self.assertTrue(result['runtime_verified'])

    def test_verified_rollback_archives_failed_install_and_same_candidate_can_retry(self):
        def old_runtime():
            return {'loaded': True, 'model_path': str(self.old_model), 'model_name': 'old'}
        for attempt in (1, 2):
            with self.subTest(attempt=attempt), self.assertRaisesRegex(data.DatasetError, 'previous model restored'):
                deployment.activate(self.candidate, self.models, self.config, old_runtime)
            self.assertEqual(self.config.read_text(), self.original)
            self.assertFalse((self.models / self.report['model_name']).exists())
            archive = self.models / 'failed_installations' / self.report['model_name'] / f'attempt-{attempt:02d}'
            self.assertTrue((archive / 'model.json').is_file())
            self.assertEqual((archive / 'acceptance/acceptance.json').read_bytes(), self.report_file.read_bytes())
        result = deployment.activate(self.candidate, self.models, self.config, self.runtime)
        self.assertTrue(result['runtime_verified'])
        self.assertEqual(len(list((self.models / 'failed_installations' / self.report['model_name']).iterdir())), 2)

    def test_parallel_installations_are_rejected(self):
        with deployment.deployment_lock(self.models):
            with self.assertRaisesRegex(data.DatasetError, 'in progress'):
                deployment.activate(self.candidate, self.models, self.config, self.runtime)


if __name__ == '__main__':
    unittest.main()
