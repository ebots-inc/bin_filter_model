"""Nominal CAD never becomes unobserved measurement evidence."""
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cad_reference as cad
from dataset import DatasetError


def bowl_triangles():
    vertices = np.arange(-5., 6.)
    triangles = []
    for y in vertices[:-1]:
        for x in vertices[:-1]:
            xy = np.array([[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]])
            z = 10 + .03 * np.square(xy).sum(axis=1)
            top = np.column_stack((xy, z))
            bottom = np.column_stack((xy, np.zeros(4)))
            for points in (top, bottom):
                triangles.extend((points[[0, 1, 2]], points[[0, 2, 3]]))
    return np.asarray(triangles)


def binary_stl(path, triangles):
    with path.open('wb') as stream:
        stream.write(b'solid binary header is permitted'.ljust(80, b' '))
        stream.write(struct.pack('<I', len(triangles)))
        for triangle in triangles:
            stream.write(struct.pack('<12fH', *([0.] * 3), *triangle.ravel(), 0))


def ascii_stl(path, triangles):
    lines = ['solid bowl']
    for triangle in triangles:
        lines.extend(['facet normal 0 0 -1', 'outer loop'])
        lines.extend('vertex ' + ' '.join(map(str, vertex)) for vertex in triangle)
        lines.extend(['endloop', 'endfacet'])
    path.write_text('\n'.join(lines + ['endsolid bowl']))


class CadReferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'bowl.stl'
        self.triangles = bowl_triangles()
        binary_stl(self.path, self.triangles)
        self.x = self.y = np.arange(-4.5, 5., 1.)
        self.config = {'mesh_path': str(self.path), 'scale_to_mm': 1., 'bin_from_cad': np.eye(4).tolist()}
        self.nominal = cad.rasterize(self.triangles, np.eye(4), self.x, self.y)
        self.support = np.full(self.nominal.shape, 5)

    def prepare(self, measured=None, config=None, support=None):
        return cad.prepare(config or self.config, self.x, self.y,
                           self.nominal if measured is None else measured,
                           self.support if support is None else support, 3)

    def test_binary_ascii_and_explicit_scale_agree(self):
        actual = cad.read_stl(self.path, 1.)
        np.testing.assert_allclose(actual, self.triangles, atol=1e-6)
        ascii_path = Path(self.directory.name) / 'bowl-ascii.stl'
        ascii_stl(ascii_path, self.triangles / 1000)
        np.testing.assert_allclose(cad.read_stl(ascii_path, 1000), self.triangles)

    def test_top_visible_envelope_excludes_closed_mesh_underside_and_ignores_winding(self):
        self.assertTrue(np.all(self.nominal >= 10))
        reversed_faces = cad.rasterize(self.triangles[:, ::-1], np.eye(4), self.x, self.y)
        np.testing.assert_allclose(reversed_faces, self.nominal)
        self.assertTrue(np.all(cad.rasterize(self.triangles, np.eye(4), [-20., -19.], [-20., -19.]) != 0))
        self.assertTrue(np.isnan(cad.rasterize(self.triangles, np.eye(4), [-20., -19.], [-20., -19.])).all())

    def test_measured_curved_deformation_and_missing_support_are_not_changed(self):
        measured = self.nominal.copy()
        measured[:, 2:5] += 1.7
        measured[0, 0] = np.nan
        support = self.support.copy()
        support[0, 1] = 1
        original = measured.copy()
        counts = support.copy()
        result = self.prepare(measured, support=support)
        np.testing.assert_array_equal(measured, original)
        np.testing.assert_array_equal(support, counts)
        np.testing.assert_allclose(result['nominal_height_mm'], self.nominal, atol=1e-6)
        self.assertEqual(result['diagnostics']['observed_overlap_cells'], measured.size - 2)
        self.assertAlmostEqual(result['diagnostics']['absolute_residual_p95_mm'], 1.7, places=5)
        self.assertIn('unresolved_sensor_error', result['diagnostics']['residual_interpretation'])
        self.assertEqual(result['diagnostics']['alignment']['method'], 'explicit_commissioned_transform')

    def test_sparse_measured_coverage_does_not_become_cad_coverage(self):
        measured = np.full(self.nominal.shape, np.nan)
        measured[:2] = self.nominal[:2]
        with self.assertRaisesRegex(DatasetError, 'overlap'):
            self.prepare(measured)

    def test_axes_and_translation_are_explicit(self):
        transform = np.array([[1, 0, 0, 2], [0, 0, -1, 1], [0, 1, 0, 3], [0, 0, 0, 1]], dtype=float)
        # Native Y vertical; commissioning transform maps it to bin Z.
        inverse = np.linalg.inv(transform)
        native = (self.triangles - transform[:3, 3]) @ inverse[:3, :3].T
        restored = cad.rasterize(native, transform, self.x, self.y)
        np.testing.assert_allclose(restored, self.nominal)

    def test_invalid_units_and_nonrigid_transforms_reject(self):
        for scale in (0, -1, True, np.inf, None):
            with self.subTest(scale=scale), self.assertRaises(DatasetError):
                self.prepare(config=self.config | {'scale_to_mm': scale})
        for key in ('scale_to_mm', 'bin_from_cad'):
            config = self.config.copy()
            del config[key]
            with self.assertRaisesRegex(DatasetError, 'explicit'):
                self.prepare(config=config)
        bad = np.eye(4)
        bad[0, 0] = 2
        with self.assertRaises(DatasetError):
            self.prepare(config=self.config | {'bin_from_cad': bad.tolist()})

    def test_displaced_commissioned_cad_does_not_silently_pass(self):
        transform = np.eye(4)
        transform[0, 3] = 100
        with self.assertRaisesRegex(DatasetError, 'overlap'):
            self.prepare(config=self.config | {'bin_from_cad': transform.tolist()})

    def test_truncated_and_nonfinite_stl_reject(self):
        self.path.write_bytes(self.path.read_bytes()[:-1])
        with self.assertRaises(DatasetError):
            cad.read_stl(self.path, 1)
        invalid = self.triangles.copy()
        invalid[0, 0, 0] = np.nan
        ascii_stl(self.path, invalid)
        with self.assertRaisesRegex(DatasetError, 'finite'):
            cad.read_stl(self.path, 1)

    @staticmethod
    def open3d_stub(transform, fitness=1.):
        result = SimpleNamespace(transformation=transform, fitness=fitness, inlier_rmse=.1)
        registration = SimpleNamespace(registration_icp=lambda *args: result,
            TransformationEstimationPointToPoint=lambda scaling: None,
            ICPConvergenceCriteria=lambda **kwargs: None)
        return SimpleNamespace(geometry=SimpleNamespace(PointCloud=SimpleNamespace),
            utility=SimpleNamespace(Vector3dVector=lambda points: points),
            pipelines=SimpleNamespace(registration=registration))

    def test_alignment_keeps_scale_and_enforces_seed_motion_and_coverage(self):
        config = self.config | {'alignment': {'enabled': True, 'max_translation_mm': 2,
                                             'max_rotation_deg': 5}}
        good = np.eye(4)
        good[2, 3] = .5
        with mock.patch.dict(sys.modules, {'open3d': self.open3d_stub(good)}):
            result = self.prepare(self.nominal + .5, config=config)
        self.assertAlmostEqual(result['bin_from_cad'][2, 3], .5)
        np.testing.assert_allclose(result['bin_from_cad'][:3, :3], np.eye(3))
        bad = np.eye(4)
        bad[2, 3] = 3
        with mock.patch.dict(sys.modules, {'open3d': self.open3d_stub(bad)}):
            with self.assertRaisesRegex(DatasetError, 'limits'):
                self.prepare(config=config)
        angle = np.deg2rad(90)
        bad = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                        [np.sin(angle), np.cos(angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        with mock.patch.dict(sys.modules, {'open3d': self.open3d_stub(bad)}):
            with self.assertRaisesRegex(DatasetError, 'limits'):
                self.prepare(config=config)
        with mock.patch.dict(sys.modules, {'open3d': self.open3d_stub(np.eye(4), .1)}):
            with self.assertRaisesRegex(DatasetError, 'coverage'):
                self.prepare(config=config)

    def test_optional_alignment_dependency_has_actionable_error(self):
        with mock.patch.dict(sys.modules, {'open3d': None}):
            with self.assertRaisesRegex(DatasetError, 'install open3d or commission'):
                self.prepare(config=self.config | {'alignment': {'enabled': True}})


if __name__ == '__main__':
    unittest.main()
