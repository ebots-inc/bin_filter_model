"""Synthetic IO and transform checks; never access a camera or external drive."""
import copy
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as ds


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.camera_path = self.base / 'camera.yaml'
        self.robot_path = self.base / 'robot.yaml'
        intrinsic = [2., 0., 0., 0., 4., 0., 0., 0., 1.]
        self.camera_yaml = {
            name: {'rows': 2, 'cols': 3, 'cam_intrinsic': intrinsic.copy(),
                   'cam_extrinsic': np.eye(4).ravel().tolist()}
            for name in ('cam1', 'cam2')}
        self.robot_yaml = {'name': 'B2', 'hand_eye': [
            {'scene_id': 0, 'calibration': [0., 0., 0., 0., 0., 0.]}]}
        self.write_calibration()
        self.root = self.base / 'dataset'
        ds.init_dataset(self.root, self.camera_path, self.robot_path)
        _, self.calibration = ds.load_dataset(self.root)

    def write_calibration(self):
        ds.write_yaml(self.camera_path, self.camera_yaml)
        ds.write_yaml(self.robot_path, self.robot_yaml)

    def make_capture(self, name='capture', contour=False, value=100.):
        capture = self.base / name
        capture.mkdir()
        image = np.full((2, 3), value, dtype=np.float32)
        metadata = {'scene_id': 0, 'capture_id': name,
                    'structured_light_exposure_ms': 4.0}
        if contour:
            metadata.update({'depth_images': [], 'spotlight_images': []})
            for key, dirname in [('depth_images', 'contour_depth_images'),
                                 ('spotlight_images', 'contour_spot_light_images')]:
                (capture / dirname).mkdir()
                for index in range(2):
                    relative = f'{dirname}/contour_{name}_Frame{index}.tiff'
                    self.assertTrue(cv2.imwrite(str(capture / relative), image + index))
                    entry = {'path': relative, 'camera_index': index, 'rows': 2,
                             'cols': 3, 'scalar_type': 'float32'}
                    if key == 'depth_images':
                        entry.update({'frame_id': f'camera_{index}',
                                      'length_unit': 'mm', 'depth_axis': 'z'})
                    metadata[key].append(entry)
            ds.write_yaml(capture / 'robot_pose' / f'contour_{name}.yaml', metadata)
        else:
            for dirname in ('depth_map', 'spot_light_images'):
                (capture / dirname).mkdir()
                for index in range(2):
                    self.assertTrue(cv2.imwrite(str(capture / dirname / f'Frame{index}.tiff'),
                                                image + index))
            ds.write_yaml(capture / 'capture_status.yaml', {'status': 'complete'})
            ds.write_yaml(capture / 'capture_metadata.yaml', metadata)
        return capture

    def add(self, capture, capture_id='training', split='train', **kwargs):
        return ds.add_capture(self.root, capture_id, split,
                              {'bin_a': 'empty', 'bin_b': 'loaded'}, 4.0,
                              capture_dir=capture, **kwargs)

    def test_pose_uses_mixed_axis_rotation_vector_degrees(self):
        component = 90. / np.sqrt(2.)
        result = ds.pose_matrix([11., 22., 33., component, component, 0.])
        a = np.sqrt(.5)
        expected = [[.5, .5, a], [.5, .5, -a], [-a, a, 0.]]
        np.testing.assert_allclose(result[:3, :3], expected, atol=1e-12)
        np.testing.assert_array_equal(result[:3, 3], [11., 22., 33.])
        np.testing.assert_array_equal(ds.pose_matrix([0.] * 6), np.eye(4))

    def test_pose_rejects_wrong_size_or_nonfinite_values(self):
        for values in ([0.] * 5, [0.] * 7, [np.nan, 0., 0., 0., 0., 0.],
                       [0., 0., 0., 0., np.inf, 0.]):
            with self.subTest(values=values), self.assertRaises(ds.DatasetError):
                ds.pose_matrix(values)

    def test_optical_z_projection_composes_forward_transforms_once(self):
        camera = copy.deepcopy(self.calibration['cameras'][0])
        camera['vision_from_camera'] = ds.pose_matrix([10., 20., 30., 0., 0., 90.])
        robot = ds.pose_matrix([100., 200., 300., 180., 0., 0.])
        depth = np.array([[0., 4., np.nan], [np.inf, -3., 8.]], dtype=np.float32)
        result = ds.backproject_depth(depth, camera, robot)
        np.testing.assert_allclose(result, [[110., 178., 266.], [108., 172., 262.]],
                                   atol=1e-10)

    def test_stride_keeps_original_pixel_coordinates(self):
        camera = copy.deepcopy(self.calibration['cameras'][0])
        camera.update(rows=3, cols=3)
        depth = np.full((3, 3), 4., dtype=np.float32)
        result = ds.backproject_depth(depth, camera, np.eye(4), stride=2)
        np.testing.assert_array_equal(result, [[0., 0., 4.], [4., 0., 4.],
                                               [0., 2., 4.], [4., 2., 4.]])
        for stride in (0, -1, 1.5):
            with self.subTest(stride=stride), self.assertRaises(ds.DatasetError):
                ds.backproject_depth(depth, camera, np.eye(4), stride)

    def test_calibration_composes_world_rotation_and_translation_before_hand_eye(self):
        world = ds.pose_matrix([10., 20., 30., 0., 0., 90.])
        extrinsic = ds.pose_matrix([3., 4., 5., 0., 0., 0.])
        robot = ds.pose_matrix([100., 200., 300., 180., 0., 0.])
        self.camera_yaml.update(world_r=world[:3,:3].ravel().tolist(), world_t=world[:3,3].tolist())
        self.camera_yaml['cam1']['cam_extrinsic'] = extrinsic.ravel().tolist()
        self.robot_yaml['hand_eye'][0]['calibration'] = [100.,200.,300.,180.,0.,0.]
        self.write_calibration()
        loaded = ds.load_calibration(self.camera_path, self.robot_path)
        np.testing.assert_allclose(loaded['cameras'][0]['vision_from_camera'], world @ extrinsic)
        depth = np.array([[0.,4.,0.],[0.,0.,0.]], dtype=np.float32)
        actual = ds.backproject_depth(depth, loaded['cameras'][0], loaded['robot_from_vision'])
        expected = ds.transform_points(np.array([[2.,0.,4.]]), robot @ world @ extrinsic)
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_depth_tiff_requires_float32_single_channel_matching_calibration(self):
        path = self.base / 'input.tiff'
        camera = self.calibration['cameras'][0]
        for image in (np.ones((2, 3), np.uint16), np.ones((3, 2), np.float32),
                      np.ones((2, 3, 3), np.uint8)):
            self.assertTrue(cv2.imwrite(str(path), image))
            with self.subTest(shape=image.shape, dtype=image.dtype), self.assertRaises(ds.DatasetError):
                ds.read_image(path, camera)
        values = np.array([[0., np.nan, np.inf], [-1., 4.25, 7.5]], np.float32)
        self.assertTrue(cv2.imwrite(str(path), values))
        np.testing.assert_array_equal(ds.read_image(path, camera), values)

    def test_camera_and_robot_calibration_validation(self):
        mutations = [
            lambda c, r: c['cam1'].update(rows=0),
            lambda c, r: c['cam2'].update(cols=2.5),
            lambda c, r: c['cam1']['cam_intrinsic'].__setitem__(0, 0.),
            lambda c, r: c['cam1']['cam_intrinsic'].__setitem__(1, 0.25),
            lambda c, r: c['cam2']['cam_intrinsic'].__setitem__(4, np.inf),
            lambda c, r: c['cam1']['cam_extrinsic'].__setitem__(0, -1.),
            lambda c, r: c['cam1']['cam_extrinsic'].__setitem__(15, 2.),
            lambda c, r: c['cam1']['cam_extrinsic'].__setitem__(3, np.nan),
            lambda c, r: r['hand_eye'].append(copy.deepcopy(r['hand_eye'][0])),
            lambda c, r: r['hand_eye'][0].update(scene_id=1),
            lambda c, r: r['hand_eye'][0]['calibration'].__setitem__(3, np.inf),
        ]
        for index, mutate in enumerate(mutations):
            cameras, robot = copy.deepcopy(self.camera_yaml), copy.deepcopy(self.robot_yaml)
            mutate(cameras, robot)
            ds.write_yaml(self.camera_path, cameras)
            ds.write_yaml(self.robot_path, robot)
            with self.subTest(index=index), self.assertRaises(ds.DatasetError):
                ds.load_calibration(self.camera_path, self.robot_path)

    def test_calibration_is_snapshotted_and_hash_checked(self):
        self.camera_path.write_text('changed original')
        ds.load_dataset(self.root)
        snapshot = self.root / 'calibration' / 'camera.yaml'
        snapshot.write_text(snapshot.read_text() + '\n# changed snapshot\n')
        with self.assertRaisesRegex(ds.DatasetError, 'changed'):
            ds.load_dataset(self.root)

    def test_current_standalone_complete_status_accepted_and_points_reconstructed(self):
        capture = self.make_capture()
        entry = self.add(capture)
        points = ds.load_capture_points(self.root, entry, self.calibration)
        self.assertEqual(points.shape, (12, 3))
        np.testing.assert_array_equal(points[0], [0., 0., 100.])
        np.testing.assert_array_equal(points[6], [0., 0., 101.])

    def test_spotlight_exposure_is_independent_and_verified(self):
        capture = self.make_capture()
        metadata = ds.read_yaml(capture / 'capture_metadata.yaml')
        metadata['spotlight_exposure_ms'] = 12.5
        ds.write_yaml(capture / 'capture_metadata.yaml', metadata)
        with self.assertRaisesRegex(ds.DatasetError, 'spotlight_exposure_ms'):
            self.add(capture, spotlight_exposure_ms=8.)
        entry = self.add(capture)
        self.assertEqual(ds.capture_exposure(entry), 4.)
        self.assertEqual(entry['spotlight_exposure_ms'], 12.5)
        entry['exposure_ms'] = 8.
        with self.assertRaisesRegex(ds.DatasetError, 'exposure differs'):
            ds.capture_exposure(entry)

    def test_legacy_manifest_exposure_uses_and_checks_saved_acquisition_metadata(self):
        entry = {'id': 'legacy', 'exposure_ms': 8., 'source_metadata': {
            'structured_light_exposure_ms': 8., 'spotlight_exposure_ms': 4.}}
        self.assertEqual(ds.capture_exposure(entry), 8.)
        self.assertEqual(ds.capture_spotlight_exposure(entry), 4.)
        entry['exposure_ms'] = 12.
        with self.assertRaisesRegex(ds.DatasetError, 'exposure differs from recorded'):
            ds.capture_exposure(entry)
        entry['spotlight_exposure_ms'] = 8.
        with self.assertRaisesRegex(ds.DatasetError, 'spotlight exposure differs'):
            ds.capture_spotlight_exposure(entry)

    def test_capture_group_cannot_cross_splits_even_with_different_pixels(self):
        self.add(self.make_capture('first'), group_id='layout_1')
        second = self.make_capture('second', value=105.)
        with self.assertRaisesRegex(ds.DatasetError, 'spans train/test'):
            self.add(second, capture_id='second', split='test', group_id='layout_1')
        self.add(second, capture_id='second', group_id='layout_1')
        manifest = ds.read_yaml(self.root / 'dataset.yaml')
        manifest['captures'][1]['split'] = 'test'
        ds.write_yaml(self.root / 'dataset.yaml', manifest)
        with self.assertRaisesRegex(ds.DatasetError, 'spans train/test'):
            ds.load_dataset(self.root)

    def test_camera_identity_and_missing_depth_are_preserved_for_diagnostics(self):
        capture = self.make_capture()
        image = np.array([[100., 0., np.nan], [np.inf, -1., 100.]], dtype=np.float32)
        cv2.imwrite(str(capture / 'depth_map/Frame0.tiff'), image)
        entry = self.add(capture)
        views = ds.load_capture_views(self.root, entry, self.calibration)
        self.assertEqual([view['camera_index'] for view in views], [0, 1])
        self.assertEqual([view['valid_depth_pixels'] for view in views], [2, 6])
        self.assertEqual(views[0]['sampled_depth_pixels'], 6)
        self.assertAlmostEqual(views[0]['valid_depth_fraction'], 1 / 3)
        np.testing.assert_array_equal(ds.load_capture_points(self.root, entry, self.calibration),
                                      np.concatenate([view['points'] for view in views]))

    def test_current_standalone_incomplete_status_rejected(self):
        capture = self.make_capture()
        ds.write_yaml(capture / 'capture_status.yaml', {'status': 'incomplete'})
        with self.assertRaisesRegex(ds.DatasetError, 'incomplete'):
            self.add(capture)

    def test_contour_selection_never_uses_later_inspection_depth(self):
        capture = self.make_capture(contour=True)
        for directory in ('depth_map', 'spot_light_images'):
            (capture / directory).mkdir()
            for index in range(2):
                self.assertTrue(cv2.imwrite(str(capture / directory / f'Frame{index}.tiff'),
                                            np.full((2, 3), 999., dtype=np.float32)))
        entry = self.add(capture)
        self.assertTrue(all('contour_depth_images' in item['path'] for item in entry['depth_images']))
        points = ds.load_capture_points(self.root, entry, self.calibration)
        self.assertLess(points[:, 2].max(), 999.)
        record = capture / 'robot_pose' / 'contour_capture.yaml'
        metadata = ds.read_yaml(record)
        metadata.pop('depth_images')
        ds.write_yaml(record, metadata)
        with self.assertRaisesRegex(ds.DatasetError, 'depth_images'):
            ds.discover_images(capture)

    def test_multiple_contour_records_require_explicit_record(self):
        capture = self.make_capture(contour=True)
        source = capture / 'robot_pose' / 'contour_capture.yaml'
        shutil.copy2(source, capture / 'robot_pose' / 'contour_second.yaml')
        with self.assertRaisesRegex(ds.DatasetError, 'Multiple contour records'):
            ds.discover_images(capture)
        depths, _, _ = ds.discover_images(capture, 'robot_pose/contour_capture.yaml')
        self.assertEqual(depths[0].name, 'contour_capture_Frame0.tiff')

    def test_depth_record_rejects_wrong_units_frames_axis_or_duplicate_camera(self):
        capture = self.make_capture(contour=True)
        record = capture / 'robot_pose' / 'contour_capture.yaml'
        original = ds.read_yaml(record)
        for key, value in [('length_unit', 'm'), ('frame_id', 'robot'), ('depth_axis', 'range'),
                           ('scalar_type', 'uint16'), ('camera_index', 1)]:
            metadata = copy.deepcopy(original)
            metadata['depth_images'][0][key] = value
            ds.write_yaml(record, metadata)
            with self.subTest(key=key), self.assertRaises(ds.DatasetError):
                ds.discover_images(capture)

    def test_duplicate_pixels_cannot_enter_independent_test_split(self):
        capture = self.make_capture(contour=True)
        self.add(capture)
        duplicate = self.base / 'copied_capture'
        shutil.copytree(capture, duplicate)
        with self.assertRaisesRegex(ds.DatasetError, 'already registered'):
            self.add(duplicate, 'holdout', 'test')

    def test_modified_registered_depth_is_rejected(self):
        capture = self.make_capture(contour=True)
        entry = self.add(capture)
        path = ds.resolve(self.root, entry['depth_images'][0]['path'])
        self.assertTrue(cv2.imwrite(str(path), np.full((2, 3), 700., dtype=np.float32)))
        with self.assertRaisesRegex(ds.DatasetError, 'changed'):
            ds.load_capture_points(self.root, entry, self.calibration)

    def test_exposure_mismatch_rejected(self):
        capture = self.make_capture(contour=True)
        with self.assertRaisesRegex(ds.DatasetError, 'Exposure differs'):
            ds.add_capture(self.root, 'different_exposure', 'train',
                           {'bin_a': 'empty', 'bin_b': 'loaded'}, 8.0, capture_dir=capture)

    def test_cloud_requires_explicit_frame_and_applies_hand_eye_only_to_vision(self):
        path = self.base / 'raw.npy'
        points = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        np.save(path, points)
        kwargs = dict(cloud_path=path)
        with self.assertRaisesRegex(ds.DatasetError, 'cloud-frame'):
            self.add(None, **kwargs)
        entry = self.add(None, cloud_frame='vision', **kwargs)
        calibration = copy.deepcopy(self.calibration)
        calibration['robot_from_vision'] = ds.pose_matrix([10., 20., 30., 0., 0., 90.])
        np.testing.assert_allclose(ds.load_capture_points(self.root, entry, calibration),
                                   [[8., 21., 33.], [5., 24., 36.], [2., 27., 39.]])
        entry['cloud']['frame'] = 'robot'
        np.testing.assert_array_equal(ds.load_capture_points(self.root, entry, calibration), points)
        for key, value in [('frame', 'camera'), ('length_unit', 'm'), ('unfiltered', False)]:
            bad = copy.deepcopy(entry)
            bad['cloud'][key] = value
            with self.subTest(key=key), self.assertRaises(ds.DatasetError):
                ds.load_capture_points(self.root, bad, calibration)

    def test_processed_contour_cloud_is_not_raw_filter_input(self):
        path = self.base / 'contour_123_robot_mm.ply'
        path.write_text('not needed: processed contour is rejected before reading')
        with self.assertRaisesRegex(ds.DatasetError, 'already filtered'):
            self.add(None, cloud_path=path, cloud_frame='robot')

    def test_xyz_with_normals_reads_only_xyz_without_scaling(self):
        path = self.base / 'point_cloud.xyz'
        path.write_text('1 2 300 0 0 1\n4 5 600 0 0 1\n7 8 900 0 0 1\n')
        np.testing.assert_array_equal(ds.read_cloud(path), [[1., 2., 300.], [4., 5., 600.],
                                                           [7., 8., 900.]])

    def test_dataset_rejects_wrong_global_frame_or_units(self):
        manifest = ds.read_yaml(self.root / 'dataset.yaml')
        for key, value in [('frame_id', 'vision'), ('length_unit', 'm')]:
            wrong = copy.deepcopy(manifest)
            wrong[key] = value
            ds.write_yaml(self.root / 'dataset.yaml', wrong)
            with self.subTest(key=key), self.assertRaises(ds.DatasetError):
                ds.load_dataset(self.root)


if __name__ == '__main__':
    unittest.main()
