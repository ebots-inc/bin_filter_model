"""Scene/layout selection and preparation without cameras or Docker."""
from copy import deepcopy
from pathlib import Path
import shutil
import os
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as data
import station_config
from setup_operations import Operations
import yaml


class StationConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / 'artifacts'
        source = Path(__file__).resolve().parents[2] / 'ebots_system_artifacts/artifacts_cell_4'
        helper = Path(__file__).resolve().parents[1]
        self.recipe = data.read_yaml(helper / 'configs/recipe.yaml')
        for relative in ('workflow_configs/bin_calibration.yaml',
                         self.recipe['camera_calibration'], self.recipe['robot_calibration']):
            target = self.artifacts / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
        self.robot_path = 'calibration/deployment/robot/station_reference.yaml'
        self.write(self.robot_path, {'name': 'A1', 'hand_eye': [
            {'scene_id': scene, 'calibration': [0., 0., 0., 0., 0., 0.]} for scene in (1, 4)]})
        template = data.read_yaml(helper / self.recipe['bin_config'])
        for scene, names in ((1, ['resistors']), (4, ['left', 'middle', 'right'])):
            camera = self.artifacts / f'calibration/deployment/camera/scene{scene}_denali0_params.yaml'
            shutil.copyfile(self.artifacts / self.recipe['camera_calibration'], camera)
            layout = deepcopy(template)
            layout.update(scene_id=scene, frame_id='A1', bins={name: deepcopy(template['bins']['bin_a']) for name in names})
            self.write(f'data_saving/bin_model/layouts/scene{scene}.yaml', layout)
        workflow_path = 'workflow_configs/bin_calibration.yaml'
        workflow = data.read_yaml(self.artifacts / workflow_path)
        for scene in (1, 4):
            workflow['scenes'].append({'scene_id': scene, 'name': 'bin_model_capture', 'ros_action': 'pose',
                'camera': [{'cam_id': 0, 'parameters': [f'artifacts/calibration/deployment/camera/scene{scene}_denali0_params.yaml']}]})
        self.write(workflow_path, workflow)
        self.workflow = workflow

    def write(self, relative, value):
        path = self.artifacts / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value, sort_keys=False))

    def request(self, scene):
        return {'scene_id': scene, 'robot_calibration': self.robot_path,
                'camera_calibration': f'calibration/deployment/camera/scene{scene}_denali0_params.yaml',
                'bin_config': f'data_saving/bin_model/layouts/scene{scene}.yaml'}

    def cad_plan(self, name='curved-bin', **changes):
        directory = 'data_saving/bin_model/cad_references/'
        mesh = self.artifacts / directory / 'curved-bin.stl'
        mesh.parent.mkdir(parents=True, exist_ok=True)
        # Inventory validates paths and YAML metadata, without parsing meshes.
        mesh.write_text('mesh is loaded only when the build starts')
        entry = {'mesh_path': mesh.name, 'scale_to_mm': 1.,
                 'bin_from_cad': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
        plan = {'name': 'Curved bin reference', 'scene_id': 4, 'frame_id': 'A1',
                'bins': {name: deepcopy(entry) for name in ('left', 'middle', 'right')}}
        plan.update(changes)
        relative = directory + name + '.yaml'
        self.write(relative, plan)
        return relative, plan

    def test_inventory_lists_deployed_files_scenes_and_arbitrary_bin_names(self):
        inventory = station_config.inventory(self.artifacts, self.recipe)
        self.assertEqual(inventory['scenes'], [0, 1, 2, 3, 4])
        layouts = {item['scene_id']: item for item in inventory['layouts']}
        self.assertEqual(layouts[1]['bin_ids'], ['resistors'])
        self.assertEqual(layouts[4]['bin_ids'], ['left', 'middle', 'right'])
        self.assertEqual(layouts[4]['frame_id'], 'A1')
        self.assertEqual([item['scene_id'] for item in inventory['eyes_workflow']['scenes']], [0, 1, 4])
        self.assertEqual(inventory['eyes_workflow']['issues'], [])
        self.assertEqual(inventory['cad_references'], [])

    def test_cad_inventory_and_selection_use_the_saved_session(self):
        relative, _ = self.cad_plan()
        inventory = station_config.inventory(self.artifacts, self.recipe)
        self.assertEqual(inventory['cad_references'], [{'path': relative, 'label': 'Curved bin reference',
            'scene_id': 4, 'frame_id': 'A1', 'bin_ids': ['left', 'middle', 'right']}])
        session = {'scene_id': 4, 'frame_id': 'A1', 'bin_ids': ['right', 'left']}
        self.assertEqual(station_config.selected_cad_reference(self.artifacts, relative, session),
                         self.artifacts / relative)
        for selection in ('', None):
            self.assertIsNone(station_config.selected_cad_reference(self.artifacts, selection, session))
        for changed in ({'scene_id': 0}, {'frame_id': 'B2'}, {'bin_ids': ['missing']}):
            with self.subTest(changed=changed), self.assertRaisesRegex(data.DatasetError, 'open session'):
                station_config.selected_cad_reference(self.artifacts, relative, dict(session, **changed))

    def test_invalid_cad_plans_are_reported_without_hiding_valid_ones(self):
        valid, plan = self.cad_plan()
        self.write('data_saving/bin_model/cad_references/not-a-mapping.yaml', [])
        bad = deepcopy(plan)
        del bad['bins']['left']['scale_to_mm']
        self.write('data_saving/bin_model/cad_references/missing-scale.yaml', bad)
        for index, changes in enumerate(({'scene_id': True}, {'frame_id': ''}, {'name': []}, {'bins': []})):
            self.write(f'data_saving/bin_model/cad_references/bad-{index}.yaml', dict(plan, **changes))
        inventory = station_config.inventory(self.artifacts, self.recipe)
        self.assertEqual([item['path'] for item in inventory['cad_references']], [valid])
        self.assertEqual(len(inventory['issues']), 6)
        self.assertTrue(any('scale_to_mm' in issue for issue in inventory['issues']))

    def test_cad_plan_and_mesh_must_remain_inside_artifacts(self):
        relative, plan = self.cad_plan()
        directory = (self.artifacts / relative).parent
        external = self.root / 'external.stl'
        external.write_text('external mesh')
        for mesh_name in (str(external), os.path.relpath(external, directory), 'linked.stl'):
            if mesh_name == 'linked.stl':
                (directory / mesh_name).symlink_to(external)
            plan['bins']['left']['mesh_path'] = mesh_name
            self.write(relative, plan)
            with self.subTest(mesh_name=mesh_name), self.assertRaises(data.DatasetError):
                station_config.cad_reference_info(self.artifacts, relative)
        outside_plan = self.root / 'external.yaml'
        outside_plan.write_text(yaml.safe_dump(plan))
        (directory / 'linked.yaml').symlink_to(outside_plan)
        with self.assertRaises(data.DatasetError):
            station_config.cad_reference_info(self.artifacts, str((directory / 'linked.yaml').relative_to(self.artifacts)))

    def test_one_and_three_bin_setups_preserve_deployment_calibration(self):
        operations = Operations(self.artifacts, self.root / 'sessions', self.root / 'models', self.root / 'pipeline.yaml')
        for scene, count in ((1, 1), (4, 3)):
            with self.subTest(scene=scene):
                request = self.request(scene)
                originals = {key: (self.artifacts / request[key]).read_bytes()
                             for key in ('camera_calibration', 'robot_calibration')}
                result = operations.execute({'command': 'prepare', **request}, lambda _: None, lambda: {})
                folder, session = Path(result['session_path']), result['session']
                self.assertEqual(session['scene_id'], scene)
                self.assertEqual(session['frame_id'], 'A1')
                self.assertEqual(len(session['bin_ids']), count)
                self.assertEqual(len(session['stages']), 26)
                self.assertEqual(len(session['captures']), 170)
                self.assertFalse((folder / 'calibration').exists())
                for key, original in originals.items():
                    self.assertEqual((self.artifacts / request[key]).read_bytes(), original)
                self.assertEqual(session['eyes_config']['artifact_path'], 'workflow_configs/bin_calibration.yaml')
                self.assertFalse((folder / 'eyes_capture.yaml').exists())
                self.assertIn('artifacts/' + request['camera_calibration'],
                              (self.artifacts / session['eyes_config']['artifact_path']).read_text())

    def test_subset_is_explicit_and_empty_or_unknown_selection_is_rejected(self):
        request = dict(self.request(4), bin_ids=['right'])
        selected = station_config.selected_recipe(self.artifacts, self.recipe, request)
        self.assertEqual(selected['bin_ids'], ['right'])
        self.assertNotIn('bin_ids', self.recipe)
        for bins in ([], ['absent'], ['right', 'right']):
            with self.assertRaisesRegex(data.DatasetError, 'Select at least one bin'):
                station_config.selected_recipe(self.artifacts, self.recipe, dict(request, bin_ids=bins))

    def test_recipe_bin_subset_applies_only_to_its_own_layout(self):
        base = dict(self.recipe, bin_ids=['bin_b'])
        self.assertEqual(station_config.inventory(self.artifacts, base)['defaults']['bin_ids'], ['bin_b'])
        self.assertEqual(station_config.selected_recipe(self.artifacts, base, {})['bin_ids'], ['bin_b'])
        self.assertEqual(station_config.selected_recipe(self.artifacts, base, self.request(4))['bin_ids'],
                         ['left', 'middle', 'right'])

    def test_scene_selection_cannot_reuse_another_scene_camera_or_layout(self):
        request = self.request(4)
        with self.assertRaisesRegex(data.DatasetError, 'different scene'):
            station_config.selected_recipe(self.artifacts, self.recipe,
                                           dict(request, camera_calibration=self.recipe['camera_calibration']))
        with self.assertRaisesRegex(data.DatasetError, 'layout measured in this scene'):
            station_config.selected_recipe(self.artifacts, self.recipe, dict(request, bin_config=self.recipe['bin_config']))
        camera = 'calibration/deployment/camera/custom.yaml'
        shutil.copyfile(self.artifacts / self.recipe['camera_calibration'], self.artifacts / camera)
        with self.assertRaisesRegex(data.DatasetError, 'no transform'):
            station_config.selected_recipe(self.artifacts, self.recipe, dict(request, scene_id=3,
                camera_calibration=camera))

    def test_arbitrarily_named_existing_camera_calibration_can_be_selected(self):
        custom = 'calibration/deployment/camera/overhead.yaml'
        shutil.copyfile(self.artifacts / self.recipe['camera_calibration'], self.artifacts / custom)
        selected = station_config.selected_recipe(self.artifacts, self.recipe,
                                                  dict(self.request(4), camera_calibration=custom))
        self.assertEqual(selected['camera_calibration'], custom)

    def test_malformed_layout_is_reported_without_hiding_valid_layouts(self):
        self.write('data_saving/bin_model/layouts/unconfigured.yaml', {'bins': {'left': {}}})
        inventory = station_config.inventory(self.artifacts, self.recipe)
        self.assertEqual(len(inventory['layouts']), 3)
        self.assertIn('scene_id and frame_id', inventory['issues'][0])

    def test_artifact_file_cannot_escape_deployment_directory(self):
        with self.assertRaisesRegex(data.DatasetError, 'relative'):
            station_config.artifact_file(self.artifacts, '../elsewhere.yaml')
        with self.assertRaisesRegex(data.DatasetError, 'invalid artifact'):
            station_config.artifact_file(self.artifacts, self.recipe['bin_config'], 'calibration/deployment/camera')

    def test_invalid_yaml_and_nested_types_are_issues_without_hiding_valid_options(self):
        bad_values = [None, ['not', 'a', 'mapping'], 'text', {'hand_eye': 'not a list'},
                      {'name': [], 'hand_eye': []}, {'name': 'bad', 'hand_eye': ['invalid']},
                      {'name': 'bad', 'hand_eye': [{'scene_id': -1, 'calibration': [0] * 6}]},
                      {'name': 'bad', 'hand_eye': [{'scene_id': 256, 'calibration': [0] * 6}]},
                      {'name': 'bad', 'hand_eye': [{'scene_id': True, 'calibration': [0] * 6}]},
                      {'name': 'bad', 'hand_eye': [{'scene_id': 3, 'calibration': [float('nan')] * 6}]},
                      {'name': 'bad', 'hand_eye': [{'scene_id': 3, 'calibration': [True] * 6}]}]
        for index, value in enumerate(bad_values):
            self.write(f'calibration/deployment/robot/invalid-{index}.yaml', value)
        self.write('calibration/deployment/camera/invalid.yaml', ['invalid'])
        self.write('calibration/deployment/camera/scene999_denali0_params.yaml',
                   data.read_yaml(self.artifacts / self.recipe['camera_calibration']))
        for index, bins in enumerate(({}, {'../escape': {}}, {'ok': None}, {1: {}})):
            self.write(f'data_saving/bin_model/layouts/invalid-{index}.yaml',
                       {'scene_id': 4, 'frame_id': 'A1', 'bins': bins})
        for value in (None, [], {'scenes': 'invalid'}, {'scenes': [None, 4, {'scene_id': True},
                                                                 {'scene_id': 256}, {'scene_id': 7}]}):
            self.write('workcell_manifest.yaml', value)
            inventory = station_config.inventory(self.artifacts, self.recipe)
            self.assertEqual(len(inventory['camera_calibrations']), 3)
            self.assertEqual(len(inventory['robot_calibrations']), 2)
            self.assertEqual(len(inventory['layouts']), 3)
            self.assertNotIn(256, inventory['scenes'])
            self.assertNotIn(999, inventory['scenes'])
            self.assertGreater(len(inventory['issues']), 15)

    def test_workflow_inventory_rejects_duplicate_or_invalid_scenes_and_cameras(self):
        valid = self.workflow['scenes'][1]
        invalid = [None, {'scene_id': True}, {'scene_id': 256},
                   {'scene_id': 3, 'camera': ['bad']},
                   {'scene_id': 3, 'camera': [{'cam_id': 0, 'parameters': 'bad'}]},
                   {'scene_id': 3, 'camera': [{'cam_id': 0, 'parameters': ['artifacts/../external.yaml']}]},
                   {'scene_id': 3, 'camera': [{'cam_id': 0, 'parameters': ['artifacts/' + self.recipe['camera_calibration']]}]}]
        duplicate = deepcopy(valid)
        duplicate['camera'] *= 2
        for item in invalid + [duplicate]:
            self.write('workflow_configs/bin_calibration.yaml', {'scenes': [item]})
            workflow = station_config.inventory(self.artifacts, self.recipe)['eyes_workflow']
            self.assertEqual(workflow['scenes'], [])
            self.assertTrue(workflow['issues'])
        self.write('workflow_configs/bin_calibration.yaml', {'scenes': [valid, deepcopy(valid), self.workflow['scenes'][2]]})
        workflow = station_config.inventory(self.artifacts, self.recipe)['eyes_workflow']
        self.assertEqual([item['scene_id'] for item in workflow['scenes']], [4])
        self.assertTrue(workflow['issues'])
        for value in (None, ['bad'], {'scenes': ['bad']}, {'scenes': {}}):
            self.write('workflow_configs/bin_calibration.yaml', value)
            workflow = station_config.inventory(self.artifacts, self.recipe)['eyes_workflow']
            self.assertEqual(workflow['scenes'], [])
            self.assertTrue(workflow['issues'])

    def test_symlinks_cannot_publish_files_outside_artifacts(self):
        external = self.root / 'external.yaml'
        shutil.copyfile(self.artifacts / self.recipe['camera_calibration'], external)
        link = self.artifacts / 'calibration/deployment/camera/linked.yaml'
        link.symlink_to(external)
        inventory = station_config.inventory(self.artifacts, self.recipe)
        self.assertFalse(any(item['path'].endswith('linked.yaml') for item in inventory['camera_calibrations']))
        self.assertTrue(any('linked.yaml' in issue for issue in inventory['issues']))
        linked_directory = self.artifacts / 'external-camera'
        linked_directory.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(data.DatasetError):
            station_config.artifact_file(self.artifacts, 'external-camera/external.yaml', 'external-camera')


if __name__ == '__main__':
    unittest.main()
