"""Candidate-only CAD overlays and portable source-asset packaging."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bin_filter
import cad_plan
import dataset as data


class CadPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.mesh = self.root / 'original curved bowl 1.STL'
        self.mesh.write_bytes(b'solid original\nendsolid original\n')
        self.plan_path = self.root / 'reference.yaml'
        self.config = {'mesh_path': self.mesh.name, 'scale_to_mm': 1.,
                       'bin_from_cad': np.eye(4).tolist(), 'association_margin_mm': 3.,
                       'min_measured_retained_fraction': .9, 'max_reference_residual_mm': 8.}
        self.plan = {'name': 'curved-bowl', 'scene_id': 3, 'frame_id': 'B2',
                     'bins': {name: deepcopy(self.config) for name in ('bin_left', 'bin_right')}}
        data.write_yaml(self.plan_path, self.plan)
        self.settings = {'model_type': 'calibrated_bin_background', 'scene_id': 3, 'frame_id': 'B2',
                         'bins': {'bin_left': {'kind': 'height_field'}, 'bin_right': {'kind': 'height_field'}}}

    def load(self):
        return cad_plan.load(self.plan_path, scene_id=3, frame_id='B2', bin_ids=self.settings['bins'])

    def test_plan_resolves_mesh_and_overlays_without_changing_source_settings(self):
        frozen_plan = self.plan_path.read_bytes()
        frozen_settings = deepcopy(self.settings)
        plan = self.load()
        changed = cad_plan.apply(self.settings, plan)
        self.assertEqual(changed['bins']['bin_left']['cad_reference']['mesh_path'], str(self.mesh))
        self.assertEqual(self.settings, frozen_settings)
        self.assertEqual(self.plan_path.read_bytes(), frozen_plan)
        changed['bins']['bin_left']['cad_reference']['bin_from_cad'][0][3] = 123
        self.assertEqual(plan['bins']['bin_left']['bin_from_cad'][0][3], 0)

    def test_plan_rejects_wrong_scene_frame_and_missing_bins(self):
        changes = ({'scene_id': 0}, {'scene_id': True}, {'frame_id': 'A1'},
                   {'bins': {'bin_left': self.config}})
        for changeset in changes:
            with self.subTest(changes=changeset):
                data.write_yaml(self.plan_path, self.plan | changeset)
                with self.assertRaises(data.DatasetError):
                    self.load()

    def test_plan_label_and_selected_bins_can_cover_a_larger_station_layout(self):
        data.write_yaml(self.plan_path, self.plan | {'name': '  Curved bin reference  ',
            'bins': self.plan['bins'] | {'extra': self.config}})
        plan = self.load()
        self.assertEqual(plan['name'], 'Curved bin reference')
        self.assertEqual(set(plan['bins']), {'bin_left', 'bin_right'})
        one = cad_plan.load(self.plan_path, scene_id=3, frame_id='B2', bin_ids=['bin_left'])
        self.assertEqual(set(one['bins']), {'bin_left'})

    def test_plan_rejects_implicit_units_bad_transforms_and_invalid_association_settings(self):
        for changes in ({'scale_to_mm': True}, {'scale_to_mm': 0}, {'association_margin_mm': -1},
                        {'min_measured_retained_fraction': 1.1}, {'max_reference_residual_mm': 0},
                        {'bin_from_cad': np.diag([2., 1., 1., 1.]).tolist()},
                        {'alignment': {'enabled': True, 'max_rotation_deg': 90}},
                        {'mesh_path': 'missing.stl'}):
            with self.subTest(changes=changes), self.assertRaises(data.DatasetError):
                cad_plan.resolve_config(self.config | changes, self.root)
        config = deepcopy(self.config)
        del config['scale_to_mm']
        with self.assertRaisesRegex(data.DatasetError, 'scale_to_mm'):
            cad_plan.resolve_config(config, self.root)

    def test_packaged_model_settings_and_plan_survive_directory_move(self):
        plan = self.load()
        settings = cad_plan.apply(self.settings, plan)
        frozen_settings = deepcopy(settings)
        transform = np.eye(4)
        transform[2, 3] = .5
        model = {'model_name': 'cad-candidate', 'bins': {name: {'cad_reference': {
            'mesh_path': str(self.mesh), 'scale_to_mm': 1., 'bin_from_cad': transform.tolist()}}
            for name in plan['bins']}}
        stage = self.root / '.bin-build-transient' / 'model'
        snapshot = cad_plan.package(stage, model, settings, plan)
        self.assertEqual(settings, frozen_settings)
        data.write_yaml(stage / 'settings.yaml', snapshot)
        data.write_json(stage / 'model.json', model)
        final = self.root / 'candidate'
        stage.parent.rename(final)
        for name in plan['bins']:
            reference = model['bins'][name]['cad_reference']
            self.assertFalse(Path(reference['mesh_path']).is_absolute())
            copied = final / 'model' / reference['mesh_path']
            self.assertEqual(copied.read_bytes(), self.mesh.read_bytes())
            self.assertEqual(snapshot['bins'][name]['cad_reference']['bin_from_cad'], transform.tolist())
            self.assertEqual(snapshot['bins'][name]['cad_reference']['alignment'], {'enabled': False})
        restored = cad_plan.load(final / 'model/cad_reference.yaml', scene_id=3, frame_id='B2',
                                 bin_ids=plan['bins'])
        self.assertTrue(all(Path(config['mesh_path']).is_file() for config in restored['bins'].values()))
        text = '\n'.join(path.read_text() for path in (final / 'model').glob('*.yaml'))
        self.assertNotIn(str(self.root), text)
        self.assertNotIn('.bin-build-', text)
        metadata = cad_plan.build_metadata(model)
        self.assertEqual(metadata['plan'], 'model/cad_reference.yaml')
        self.assertEqual(metadata['name'], 'curved-bowl')
        self.assertTrue(all((final / path).is_file() for path in metadata['mesh_paths'].values()))

    def test_converted_model_copies_relative_cad_assets(self):
        plan = self.load()
        settings = cad_plan.apply(self.settings, plan)
        model = {'bins': {name: {'cad_reference': deepcopy(config)} for name, config in plan['bins'].items()}}
        source = self.root / 'source'
        snapshot = cad_plan.package(source, model, settings, plan)
        data.write_yaml(source / 'settings.yaml', snapshot)
        destination = self.root / 'converted'
        args = bin_filter.parser().parse_args(['convert-model', '--model', str(source), '--output', str(destination)])
        with mock.patch.object(bin_filter.model_io, 'load_model', return_value=model), \
                mock.patch.object(bin_filter.model_io, 'write_model') as writer:
            bin_filter.command_convert_model(args)
        converted = writer.call_args.args[1]
        for item in converted['bins'].values():
            self.assertTrue((destination / item['cad_reference']['mesh_path']).is_file())
        self.assertEqual((destination / 'cad_reference.yaml').read_bytes(),
                         (source / 'cad_reference.yaml').read_bytes())

    def test_calibrate_cli_forwards_candidate_overlay_to_modern_fit(self):
        dataset = self.root / 'dataset'
        data.write_yaml(dataset / 'bin_config.yaml', self.settings)
        before = (dataset / 'bin_config.yaml').read_bytes()
        manifest = {'scene_id': 3, 'frame_id': 'B2', 'captures': []}
        args = bin_filter.parser().parse_args(['calibrate', '--dataset', str(dataset),
            '--output', str(self.root / 'model'), '--cad-reference', str(self.plan_path)])
        with mock.patch.object(bin_filter.data, 'load_dataset', return_value=(manifest, {})), \
                mock.patch.object(bin_filter, 'calibrate_modern') as fit:
            bin_filter.command_calibrate(args)
        settings = fit.call_args.args[3]
        self.assertEqual(settings['bins']['bin_left']['cad_reference']['mesh_path'], str(self.mesh))
        self.assertEqual(fit.call_args.kwargs['cad_plan']['name'], 'curved-bowl')
        self.assertEqual((dataset / 'bin_config.yaml').read_bytes(), before)

    def test_build_session_forwards_plan_and_records_portable_metadata(self):
        output = self.root / 'candidate'
        args = bin_filter.parser().parse_args(['build-session', '--session', str(self.root / 'session'),
            '--output', str(output), '--cad-reference', str(self.plan_path), '--pipeline-config', 'pipeline.yaml'])
        capture = {'id': 'empty-01', 'split': 'train', 'bins': {'bin_left': 'empty', 'bin_right': 'empty'},
                   'exposure_ms': 4., 'spotlight_exposure_ms': 4.}
        manifest = {'captures': [capture], 'scene_id': 3, 'frame_id': 'B2'}
        model = {'model_type': 'calibrated_bin_background', 'bins': {'bin_left': {
            'cad_reference': {'mesh_path': 'cad/bin_left/bin.stl'}}},
            'cad_reference_plan': {'name': 'curved-bowl', 'path': 'cad_reference.yaml'}}
        with mock.patch.object(bin_filter.data, 'import_session', return_value=manifest), \
                mock.patch.object(bin_filter, 'command_calibrate') as calibrate, \
                mock.patch.object(bin_filter, 'command_validate'), \
                mock.patch.object(bin_filter.model_io, 'load_model', return_value=model), \
                mock.patch('production_validation.validate_production', return_value={'status': 'passed'}):
            bin_filter.command_build_session(args)
        self.assertEqual(calibrate.call_args.args[0].cad_reference, str(self.plan_path))
        report = json.loads((output / 'build.json').read_text())
        self.assertEqual(report['cad_reference']['plan'], 'model/cad_reference.yaml')
        self.assertEqual(report['activation'], 'not_activated')
        self.assertNotIn('.bin-build-', json.dumps(report))

    def test_no_cad_has_no_additional_build_metadata(self):
        self.assertIsNone(cad_plan.build_metadata({'bins': {'bin_left': {}}}))

    def test_conversion_rejects_asset_and_plan_path_escape(self):
        source = self.root / 'source'
        (source / 'cad').mkdir(parents=True)
        (source / 'cad/escape.stl').symlink_to(self.mesh)
        for relative in ('../original curved bowl 1.STL', str(self.mesh), 'cad/escape.stl'):
            with self.subTest(relative=relative), self.assertRaises(data.DatasetError):
                model = {'bins': {'bin_left': {'cad_reference': {
                    'mesh_path': relative, 'bin_from_cad': np.eye(4).tolist()}}}}
                cad_plan.package(self.root / 'failed', model, {}, source_root=source)
        with self.assertRaises(data.DatasetError):
            cad_plan.package(self.root / 'failed-plan',
                {'bins': {'bin_left': {}}, 'cad_reference_plan': {'path': '../reference.yaml'}},
                {}, source_root=source)


if __name__ == '__main__':
    unittest.main()
