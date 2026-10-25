"""Measured rim, camera error correction and bin-local placement invariants."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as data
import model_io
import modern_model as modern


def settings():
    result = {'model_type': modern.MODEL_TYPE, 'bins': {}}
    for name, offset in [('bin_a', 0.), ('bin_b', 80.)]:
        placement = np.eye(4)
        placement[0, 3] = offset
        result['bins'][name] = {'kind': 'height_field', 'bin_id': name,
            'coordinate_frame': 'bin', 'robot_from_bin': placement.tolist(),
            'xy_bounds_mm': [[-24, -22], [24, 22]], 'z_bounds_mm': [-6, 13],
            'seed_xy_mm': [0, 0], 'max_reference_height_mm': 6,
            'min_capture_support': 3, 'min_points_per_cell': 3,
            'min_tolerance_mm': .2, 'max_tolerance_mm': 4.,
            'registration': {'height_band_mm': [9, 12]},
            'camera_model': {'alignment_margin_mm': .15, 'min_uncertainty_mm': .15}}
    return result


def write_calibration(root, model):
    camera = {'world_r': np.eye(3).ravel().tolist(), 'world_t': [0, 0, 0]}
    camera.update({name: {'rows': 32, 'cols': 40, 'cam_intrinsic': np.eye(3).ravel().tolist(),
                         'cam_extrinsic': np.eye(4).ravel().tolist()} for name in ('cam1', 'cam2')})
    robot = {'name': model['frame_id'], 'hand_eye': [{'scene_id': model['scene_id'], 'calibration': [0] * 6}]}
    for name, value in (('camera', camera), ('robot', robot)):
        data.write_yaml(Path(root) / 'calibration' / f'{name}.yaml', value)
    return model_io.calibration_values(camera, robot, model['scene_id'], model['frame_id'])


def surface(x, y):
    interior = (np.abs(x) < 12) & (np.abs(y) < 10)
    lip = (np.abs(x) < 20) & (np.abs(y) < 18)
    return np.where(interior, .008*x*x+.009*y*y, np.where(lip, 10., -4.))


def captures(exposures=(4., 20.), count=3, seed=31):
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(-24, 24), np.arange(-22, 22))
    origins = np.column_stack((x.ravel(), y.ravel()))
    result = []
    for exposure in exposures:
        for index in range(count):
            views = []
            for camera in (0, 1):
                clouds = []
                for offset in (0., 80.):
                    xy = np.repeat(origins, 12, axis=0) + rng.uniform(.04, .96, (len(origins)*12, 2))
                    bias = (-.5 + .03*exposure) if camera == 0 else (.4 + .06*exposure)
                    z = surface(xy[:, 0], xy[:, 1]) + bias + rng.normal(0, .025, len(xy))
                    clouds.append(np.column_stack((xy[:, 0]+offset, xy[:, 1], z)))
                views.append({'camera_index': camera, 'points': np.concatenate(clouds),
                              'acquisition': {'gain': 1., 'black_level': 0.}})
            result.append({'id': f'e{exposure:g}_r{index}', 'group_id': f'round{index}',
                           'split': 'train', 'bins': {'bin_a': 'empty', 'bin_b': 'empty'},
                           'exposure_ms': exposure, 'requested_settings': {'gain': [1, 1], 'black_level': [0, 0]},
                           'views': views})
    return result


class ModernModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.captures = captures()
        cls.settings = settings()
        cls.model = modern.fit_model(cls.captures, cls.settings, model_name='measured-fixture')

    def test_true_lip_is_separate_from_interior_and_measured_in_all_quadrants(self):
        item = self.model['bins']['bin_a']
        self.assertEqual(item['geometry']['frame_id'], 'bin')
        heights = np.asarray(item['geometry']['surface_height_mm'], dtype=float)
        self.assertLessEqual(np.nanmax(heights), 6)
        rim = item['registration']
        self.assertGreater(len(rim['surface_xyz_mm']), 200)
        self.assertGreater(len(rim['boundary_xyz_mm']), 80)
        self.assertTrue(all(n >= 20 for n in rim['diagnostics']['boundary_quadrants']))
        self.assertGreater(np.min(np.asarray(rim['surface_xyz_mm'])[:, 2]), 9)
        self.assertEqual(item['geometry']['diagnostics']['capture_count'], 6)
        shape = modern.extract_shape(self.model, 'bin_a')
        self.assertNotIn('robot_from_bin', shape)
        self.assertNotIn('camera_profiles', shape)

    def test_camera_bias_is_removed_before_merge_and_pick_points_preserve_provenance(self):
        capture = captures(exposures=(20.,), count=1, seed=19)[0]
        result = modern.classify_views(self.model, 'bin_a', capture['views'], 20.)
        self.assertEqual(result['counts']['kept'], 0)
        profile = self.model['bins']['bin_a']['camera_profiles']
        medians = {(p['camera_index'], p['exposure_ms']): np.nanmedian(np.asarray(p['bias_mm'], dtype=float)) for p in profile}
        self.assertAlmostEqual(medians[1,20.] - medians[0,20.], 1.5, delta=.05)
        views = deepcopy(capture['views'])
        selected = (np.abs(views[0]['points'][:,0]) < 2) & (np.abs(views[0]['points'][:,1]) < 2)
        views[0]['points'][selected,2] += 2
        result = modern.classify_views(self.model, 'bin_a', views, 20.)
        self.assertGreater(result['counts']['kept'], 50)
        original = np.concatenate([view['points'] for view in views])
        np.testing.assert_array_equal(result['original_points'], original)
        np.testing.assert_array_equal(result['points'], result['corrected_points'][result['keep']])
        self.assertGreater(np.max(np.abs(result['points'][:,2] - original[result['keep'],2])), .1)
        np.testing.assert_array_equal(sum(mask.astype(int) for mask in result['masks'].values()), np.ones(len(original)))

    def test_exposure_requires_validated_intervals_and_acquisition_match(self):
        capture = captures(exposures=(12.,), count=1)[0]
        with self.assertRaisesRegex(data.DatasetError, 'held-out'):
            modern.correct_views(self.model, 'bin_a', capture['views'], 12.)
        candidate = modern.correct_views(self.model, 'bin_a', capture['views'], 12., candidate_interval=True)
        self.assertTrue(all(len(view['points']) > 1000 for view in candidate))
        model = deepcopy(self.model)
        model['bins']['bin_a']['validated_intervals_ms'] = [[4.,20.]]
        modern.correct_views(model, 'bin_a', capture['views'], 12.)
        with self.assertRaisesRegex(data.DatasetError, 'extrapolation'):
            modern.correct_views(model, 'bin_a', capture['views'], 21.)
        with self.assertRaisesRegex(data.DatasetError, 'acquisition'):
            modern.correct_views(model, 'bin_a', capture['views'], 20., acquisition={0:{'gain':2,'black_level':0},1:{'gain':1,'black_level':0}})

    def test_invalid_camera_cells_are_excluded_even_when_other_camera_has_geometry(self):
        model = deepcopy(self.model)
        profile = next(p for p in model['bins']['bin_a']['camera_profiles'] if p['camera_index']==0 and p['exposure_ms']==4)
        profile['validity'][21][23] = 0
        profile['bias_mm'][21][23] = None
        profile['uncertainty_mm'][21][23] = None
        views = [{'camera_index': camera, 'points': np.array([[0.,0.,2.]]), 'acquisition': {'gain':1,'black_level':0}} for camera in (0,1)]
        corrected = modern.correct_views(model, 'bin_a', views, 4.)
        self.assertEqual(len(corrected[0]['points']), 0)
        self.assertEqual(len(corrected[1]['points']), 1)

    def test_missing_rim_or_only_missing_data_boundary_is_rejected(self):
        config = deepcopy(self.settings)
        config['bins']['bin_a']['registration']['height_band_mm'] = [11.5,12.5]
        with self.assertRaisesRegex(data.DatasetError, 'Measured rim'):
            modern.fit_model(self.captures, config)
        surface = np.full((44,48), 10.)
        surface[10:30,10:30] = np.nan
        with self.assertRaisesRegex(data.DatasetError, 'observed-boundary'):
            modern._measured_rim(surface, np.full(surface.shape,6), self.settings['bins']['bin_a'], np.array([9.,12.]), {})

    def test_bundle_roundtrip_and_binary_lengths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = model_io.write_model(directory, self.model)
            loaded = model_io.load_model(path.parent)
            self.assertEqual(loaded, self.model)
            document = json.loads(path.read_text())
            rim = document['bins']['bin_a']['registration']['surface_xyz_mm']
            self.assertEqual(set(rim), {'file','dtype','shape'})
            profile = document['bins']['bin_a']['camera_profiles'][0]
            self.assertEqual(profile['validity']['dtype'], 'int64_le')
            (path.parent / rim['file']).write_bytes(b'bad')
            with self.assertRaisesRegex(data.DatasetError, 'byte size'):
                model_io.load_model(path)

    def test_runtime_export_keeps_prediction_inputs_and_omits_training_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = model_io.write_model(root / 'candidate', self.model).parent
            reference = write_calibration(source, self.model)
            (source / 'settings.yaml').write_text('training: only\n')
            (source / 'cad').mkdir()
            (source / 'cad/source.stl').write_text('training only')
            exported = model_io.export_runtime(source, root / 'runtime')
            self.assertEqual(model_io.load_model(exported), self.model | {'calibration_reference': reference})
            self.assertEqual({p.name for p in exported.parent.iterdir()},
                             {'model.json', 'arrays'})
            reexport = model_io.export_runtime(exported, root / 'reexport')
            self.assertEqual(model_io.load_model(exported), model_io.load_model(reexport))
            (source / 'calibration/camera.yaml').unlink()
            with self.assertRaisesRegex(data.DatasetError, 'Missing frozen camera'):
                model_io.export_runtime(source, root / 'incomplete')
            self.assertFalse((root / 'incomplete').exists())

    def test_placement_is_explicit_and_overlapping_world_regions_are_rejected(self):
        model = deepcopy(self.model)
        model['bins']['bin_b']['robot_from_bin'] = np.eye(4).tolist()
        with self.assertRaisesRegex(data.DatasetError, 'overlap'):
            modern.validate_model(model)
        np.testing.assert_array_equal(self.model['bins']['bin_a']['geometry']['xy_bounds_mm'],
                                      self.model['bins']['bin_b']['geometry']['xy_bounds_mm'])
        self.assertEqual(self.model['bins']['bin_b']['robot_from_bin'][0][3], 80.)

    def test_rotated_placement_corrects_in_bin_axes_and_returns_robot_coordinates(self):
        model = deepcopy(self.model)
        placement = data.pose_matrix([5.,-4.,20.,8.,0.,15.])
        model['bins']['bin_a']['robot_from_bin'] = placement.tolist()
        modern.validate_model(model)
        local = np.array([[0.,0.,3.]])
        views = [{'camera_index':camera,'points':data.transform_points(local,placement),
                  'acquisition':{'gain':1.,'black_level':0.}} for camera in (0,1)]
        actual = modern.correct_views(model,'bin_a',views,20.)
        for item in actual:
            np.testing.assert_allclose(item['points'], data.transform_points(item['local_points'],placement), atol=1e-10)
            np.testing.assert_allclose(item['local_points'][:,:2],local[:,:2],atol=1e-10)
            self.assertGreater(abs(item['points'][0,0]-views[item['camera_index']]['points'][0,0]), .001)

    def test_interpolation_needs_both_anchor_masks_and_model_rejects_uncertainty_inflation(self):
        model = deepcopy(self.model)
        item = model['bins']['bin_a']
        item['validated_intervals_ms'] = [[4.,20.]]
        profile = next(p for p in item['camera_profiles'] if p['camera_index']==0 and p['exposure_ms']==20)
        profile['validity'][21][23] = 0
        profile['bias_mm'][21][23] = None
        profile['uncertainty_mm'][21][23] = None
        views = [{'camera_index':0,'points':np.array([[0.,0.,3.]]),'acquisition':{'gain':1.,'black_level':0.}}]
        self.assertEqual(len(modern.correct_views(model,'bin_a',views,4.)[0]['points']),1)
        self.assertEqual(len(modern.correct_views(model,'bin_a',views,12.)[0]['points']),0)
        invalid = deepcopy(self.model)
        profile = invalid['bins']['bin_a']['camera_profiles'][0]
        yy,xx = np.argwhere(np.asarray(profile['validity'],dtype=bool))[0]
        profile['uncertainty_mm'][yy][xx] = 100.
        with self.assertRaisesRegex(data.DatasetError,'error bounds'):
            modern.validate_model(invalid)

    def test_one_and_three_named_bins_fit_roundtrip_and_filter_in_other_scenes(self):
        for scene, names in ((1, ['supply-tray']), (4, ['incoming.parts', 'resistors', 'rejects'])):
            with self.subTest(scene=scene, bins=names):
                config = {'model_type': modern.MODEL_TYPE, 'scene_id': scene, 'frame_id': 'A1', 'bins': {}}
                for index, name in enumerate(names):
                    item = deepcopy(self.settings['bins']['bin_a'])
                    item.update(bin_id=name)
                    item['robot_from_bin'][0][3] = index * 80.
                    config['bins'][name] = item
                records = deepcopy(self.captures)
                for capture in records:
                    capture['bins'] = {name: 'empty' for name in names}
                    for view in capture['views']:
                        first = view['points'][view['points'][:, 0] < 30]
                        view['points'] = np.concatenate([first + [80. * index, 0, 0] for index in range(len(names))])
                model = modern.fit_model(records, config, frame_id='A1', model_name=f'scene-{scene}')
                self.assertEqual(model['frame_id'], 'A1')
                self.assertEqual(model['scene_id'], scene)
                self.assertEqual(set(model['bins']), set(names))
                with tempfile.TemporaryDirectory() as temporary:
                    self.assertEqual(model_io.load_model(model_io.write_model(temporary, model)), model)
                for name in names:
                    result = modern.classify_views(model, name, records[0]['views'], 4.)
                    self.assertEqual(result['counts']['kept'], 0)
                records[0]['bins'].pop(names[-1])
                with self.assertRaises(data.DatasetError):
                    modern.fit_model(records, config, frame_id='A1')

    def test_bin_ids_cannot_escape_bundle_output(self):
        model = deepcopy(self.model)
        for invalid in ('../outside', '/absolute', '', '.hidden'):
            model['bins'] = {invalid: self.model['bins']['bin_a']}
            with self.assertRaises(data.DatasetError):
                modern.validate_model(model)


if __name__ == '__main__':
    unittest.main()
