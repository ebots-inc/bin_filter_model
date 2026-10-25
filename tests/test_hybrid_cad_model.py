"""CAD association must preserve measured shape, thin parts and missing support."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cad_guidance
import dataset as data
import model_io
import modern_model as modern
from test_cad_reference import binary_stl
from test_modern_model import captures, settings, surface, write_calibration


def nominal_triangles():
    triangles = []
    # Small faces describe the nominal curved bowl and its flat lip. Heights
    # deliberately omit the repeatable local deformation in the captures.
    for y in np.arange(-18, 18, .5):
        for x in np.arange(-20, 20, .5):
            xy = np.array([[x, y], [x+.5, y], [x+.5, y+.5], [x, y+.5]])
            heights = surface(xy[:, 0], xy[:, 1])
            # The synthetic fixture jumps at its outer edge; retain the top
            # face of the lip at its mathematical boundary.
            heights[(np.abs(xy[:, 0]) == 20) | (np.abs(xy[:, 1]) == 18)] = 10.
            points = np.column_stack((xy, heights))
            triangles.extend((points[[0, 1, 2]], points[[0, 2, 3]]))
    return np.asarray(triangles)


class HybridCadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.mesh = Path(cls.directory.name) / 'nominal.stl'
        binary_stl(cls.mesh, nominal_triangles())
        cls.records = captures()
        # A repeatable 1.4 mm deviation, as a printed surface could have, plus
        # a missing patch. Neither is permission to widen thresholds or fill it.
        for capture in cls.records:
            for view in capture['views']:
                points = view['points']
                local_x = np.where(points[:, 0] > 40, points[:, 0] - 80, points[:, 0])
                deformation = (np.abs(local_x) < 3) & (np.abs(points[:, 1]) < 3)
                points[deformation, 2] += 1.4
                hole = (local_x >= 5) & (local_x < 7) & (points[:, 1] >= 3) & (points[:, 1] < 5)
                view['points'] = points[~hole]
        cls.settings = settings()
        cls.baseline = modern.fit_model(cls.records, cls.settings, model_name='fixture')
        cls.cad_settings = deepcopy(cls.settings)
        for config in cls.cad_settings['bins'].values():
            config['cad_reference'] = {'mesh_path': str(cls.mesh), 'scale_to_mm': 1.,
                                       'bin_from_cad': np.eye(4).tolist()}
        cls.hybrid = modern.fit_model(cls.records, cls.cad_settings, model_name='fixture')

    def test_printed_deviation_preserved_and_thresholds_and_camera_profiles_unchanged(self):
        for name in self.baseline['bins']:
            baseline, hybrid = self.baseline['bins'][name], self.hybrid['bins'][name]
            for key in ('surface_height_mm', 'tolerance_mm', 'capture_support', 'residual_capture_support'):
                np.testing.assert_array_equal(baseline['geometry'][key], hybrid['geometry'][key])
            self.assertEqual(baseline['camera_profiles'], hybrid['camera_profiles'])
            cad = hybrid['cad_reference']
            heights = np.asarray(hybrid['geometry']['surface_height_mm'], dtype=float)
            nominal = np.asarray(cad['nominal_height_mm'], dtype=float)
            deviation = np.asarray(cad['apparent_deviation_mm'], dtype=float)
            self.assertGreater(deviation[22, 24], 1.3)
            known = np.isfinite(heights) & np.isfinite(nominal)
            np.testing.assert_allclose((nominal + deviation)[known], heights[known])
            self.assertIn('unresolved_sensor_error', cad['diagnostics']['residual_interpretation'])

    def test_cad_does_not_fill_unknown_measured_patch(self):
        item = self.hybrid['bins']['bin_a']
        y, x = 25, 29
        self.assertTrue(np.isfinite(item['cad_reference']['nominal_height_mm'][y][x]))
        self.assertIsNone(item['geometry']['surface_height_mm'][y][x])
        self.assertIsNone(item['cad_reference']['apparent_deviation_mm'][y][x])

    def test_thin_part_is_kept_against_actual_deformed_surface(self):
        record = deepcopy(self.records[-1])
        for view in record['views']:
            points = view['points']
            selected = (np.abs(points[:, 0]) < 1.5) & (np.abs(points[:, 1]) < 1.5)
            points[selected, 2] += 1.
        baseline = modern.classify_views(self.baseline, 'bin_a', record['views'], 20.)
        hybrid = modern.classify_views(self.hybrid, 'bin_a', record['views'], 20.)
        self.assertGreater(hybrid['counts']['kept'], 30)
        np.testing.assert_array_equal(baseline['keep'], hybrid['keep'])

    def test_binary_bundle_roundtrip_and_no_cad_backward_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = model_io.write_model(directory, self.hybrid)
            self.assertEqual(model_io.load_model(path), self.hybrid)
            metadata = json.loads(path.read_text())['bins']['bin_a']['cad_reference']
            self.assertEqual(metadata['nominal_height_mm']['dtype'], 'float64_le')
            self.assertEqual(metadata['association_mask']['dtype'], 'int64_le')
        with tempfile.TemporaryDirectory() as directory:
            loaded = model_io.load_model(model_io.write_model(directory, self.baseline))
            self.assertEqual(loaded, self.baseline)
            self.assertNotIn('cad_reference', loaded['bins']['bin_a'])

    def test_runtime_bundle_omits_cad_files_without_changing_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = deepcopy(self.hybrid)
            model['cad_reference_plan'] = {'name': 'training reference', 'path': 'cad_reference.yaml'}
            source = model_io.write_model(root / 'candidate', model)
            write_calibration(source.parent, model)
            runtime = model_io.load_model(model_io.export_runtime(source, root / 'runtime'))
            self.assertNotIn('cad_reference_plan', runtime)
            for name, item in runtime['bins'].items():
                self.assertNotIn('cad_reference', item)
                expected = {key: value for key, value in model['bins'][name].items() if key != 'cad_reference'}
                self.assertEqual(item, expected)
            self.assertFalse(any((root / 'runtime/arrays').rglob('cad_reference')))
            baseline = modern.classify_views(model, 'bin_a', self.records[-1]['views'], 20.)
            exported = modern.classify_views(runtime, 'bin_a', self.records[-1]['views'], 20.)
            np.testing.assert_array_equal(baseline['keep'], exported['keep'])
            np.testing.assert_array_equal(baseline['corrected_points'], exported['corrected_points'])

    def test_wrong_cad_z_rejects_without_changing_measured_model(self):
        config = deepcopy(self.cad_settings['bins']['bin_a']['cad_reference'])
        config['bin_from_cad'][2][3] = 60
        geometry = self.baseline['bins']['bin_a']['geometry']
        reference = self.hybrid['bins']['bin_a']['cad_reference']
        measured = (np.asarray(reference['nominal_height_mm'], dtype=float)
                    + np.asarray(reference['apparent_deviation_mm'], dtype=float))
        with self.assertRaisesRegex(data.DatasetError, 'differs excessively'):
            cad_guidance.guide(geometry, measured, np.asarray(geometry['capture_support']),
                               self.settings['bins']['bin_a'], [9., 12.], {}, config)

    def test_deviation_and_masks_cannot_invent_supported_geometry(self):
        for corruption, message in [('deviation', 'preserve'), ('mask', 'outside')]:
            with self.subTest(corruption=corruption):
                model = deepcopy(self.hybrid)
                reference = model['bins']['bin_a']['cad_reference']
                if corruption == 'deviation':
                    reference['apparent_deviation_mm'][22][24] += 1
                else:
                    reference['association_mask'][22][24] = 0
                with self.assertRaisesRegex(data.DatasetError, message):
                    modern.validate_model(model)

    def test_cad_rim_selection_keeps_lower_neighbor_context(self):
        fit = self.settings['bins']['bin_a']
        x, y = np.meshgrid(np.arange(-24, 24) + .5, np.arange(-22, 22) + .5)
        measured = surface(x, y)
        support = np.full(measured.shape, 6)
        nominal_lip = measured == 10.
        baseline = modern._measured_rim(measured, support, fit, np.array([9.,12.]), {})
        guided = modern._measured_rim(measured, support, fit, np.array([9.,12.]), {}, cad_mask=nominal_lip)
        self.assertEqual(guided['boundary_xyz_mm'], baseline['boundary_xyz_mm'])
        # If lower context had been removed first, no observed drop would exist.
        masked = np.where(nominal_lip, measured, np.nan)
        with self.assertRaisesRegex(data.DatasetError, 'observed-boundary'):
            modern._measured_rim(masked, support, fit, np.array([9.,12.]), {})

    def test_cad_excludes_unrelated_flat_object_from_registration(self):
        fit = self.settings['bins']['bin_a']
        x, y = np.meshgrid(np.arange(-24, 24) + .5, np.arange(-22, 22) + .5)
        measured = surface(x, y)
        cad_lip = measured == 10.
        unrelated = (x > 21) & (np.abs(y) < 4)
        measured[unrelated] = 10.
        support = np.full(measured.shape, 6)
        baseline = modern._measured_rim(measured, support, fit, np.array([9.,12.]), {})
        guided = modern._measured_rim(measured, support, fit, np.array([9.,12.]), {}, cad_mask=cad_lip)
        self.assertTrue(np.any(np.asarray(baseline['surface_xyz_mm'])[:, 0] > 21))
        self.assertFalse(np.any(np.asarray(guided['surface_xyz_mm'])[:, 0] > 21))
        self.assertGreaterEqual(guided['diagnostics']['cad_surface_retained_fraction'], .9)


if __name__ == '__main__':
    unittest.main()
