"""Numerical camera coverage and exposure summaries on controlled observations."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exposure_diagnostics import describe_views, summarize_exposures
from test_exposure_profiles import _field


class ExposureDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.field = _field(2)
        cls.field['tolerance_mm'][3][3] = None
        height = cls.field['surface_height_mm'][1][1]
        cls.views = [
            {'camera_index': 0,
             'points': np.array([[1.5, 1.5, height + .25], [2.5, 2.5, height - .5],
                                 [3.75, 3.75, height + 9], [10, 10, height]]),
             'sampled_depth_pixels': 10, 'valid_depth_pixels': 4, 'valid_depth_fraction': .4},
            {'camera_index': 1,
             'points': np.array([[1.5, 1.5, height + 1], [1.6, 1.6, height + 3]]),
             'sampled_depth_pixels': 10, 'valid_depth_pixels': 2, 'valid_depth_fraction': .2},
        ]

    def test_each_camera_keeps_its_validity_coverage_and_signed_residual_statistics(self):
        results = describe_views(self.views, self.field, 'empty')
        self.assertEqual([item['camera_index'] for item in results], [0, 1])
        self.assertEqual([item['valid_depth_fraction'] for item in results], [.4, .2])
        first, second = results
        self.assertEqual(first['roi_point_count'], 3)
        self.assertEqual(first['known_reference_point_count'], 2)
        self.assertEqual(first['unknown_point_count'], 1)
        self.assertAlmostEqual(first['unknown_fraction_of_roi'], 1 / 3)
        self.assertEqual(first['supported_squares_total'], 35)
        self.assertEqual(first['supported_squares_observed'], 2)
        self.assertAlmostEqual(first['observed_support_fraction'], 2 / 35)
        np.testing.assert_allclose(list(first['signed_residual_quantiles_mm'].values()),
                                   [-.4925, -.125, .2425], atol=1e-10)
        self.assertEqual(second['supported_squares_observed'], 1)
        self.assertAlmostEqual(second['observed_support_fraction'], 1 / 35)
        np.testing.assert_allclose(list(second['signed_residual_quantiles_mm'].values()),
                                   [1.02, 2., 2.98], atol=1e-10)
        self.assertTrue(all(item['residual_interpretation'] == 'empty_background' for item in results))
        self.assertTrue(all('points' not in item for item in results))

    def test_loaded_residuals_are_marked_as_including_foreground(self):
        results = describe_views(self.views, self.field, 'loaded')
        self.assertTrue(all(item['residual_interpretation'] == 'includes_foreground' for item in results))
        self.assertAlmostEqual(results[1]['signed_residual_quantiles_mm']['p50'], 2.)

    def test_no_roi_or_no_known_coverage_produces_defined_zero_and_null_metrics(self):
        no_valid = {'camera_index': 0, 'points': np.empty((0, 3)), 'sampled_depth_pixels': 10,
                    'valid_depth_pixels': 0, 'valid_depth_fraction': 0.0}
        unknown = {'camera_index': None, 'points': np.array([[3.75, 3.75, -95]]),
                   'sampled_depth_pixels': None, 'valid_depth_pixels': None, 'valid_depth_fraction': None}
        first, second = describe_views([no_valid, unknown], self.field, 'empty')
        self.assertEqual(first['roi_point_count'], 0)
        self.assertIsNone(first['unknown_fraction_of_roi'])
        self.assertEqual(second['unknown_fraction_of_roi'], 1.)
        self.assertIsNone(second['valid_depth_fraction'])
        for report in (first, second):
            self.assertEqual(report['supported_squares_observed'], 0)
            self.assertEqual(report['observed_support_fraction'], 0.)
            self.assertEqual(report['signed_residual_quantiles_mm'], {'p01': None, 'p50': None, 'p99': None})

    def test_summary_weights_leakage_by_points_and_keeps_labels_bins_exposures_separate(self):
        def report(exposure, name, label, surface, kept, unknown, mode):
            return {'structured_light_exposure_ms': exposure, 'bin_id': name, 'label': label,
                    'counts': {'surface': surface, 'kept': kept, 'unknown': unknown},
                    'classified_bin_point_count': surface + kept,
                    'exposure_selection': {'mode': mode}}

        reports = [report(10., 'bin_a', 'empty', 1, 1, 2, 'interpolated'),
                   report(10., 'bin_a', 'empty', 27, 3, 4, 'interpolated'),
                   report(10., 'bin_a', 'loaded', 1, 2, 5, 'interpolated'),
                   report(10., 'bin_b', 'empty', 0, 0, 8, 'interpolated'),
                   report(4., 'bin_a', 'empty', 5, 0, 0, 'exact')]
        before = copy.deepcopy(reports)
        results = summarize_exposures(reports)
        self.assertEqual(len(results), 4)
        lookup = {(item['structured_light_exposure_ms'], item['bin_id'], item['label']): item
                  for item in results}
        combined = lookup[(10., 'bin_a', 'empty')]
        self.assertEqual(combined['capture_count'], 2)
        self.assertEqual(combined['classified_bin_point_count'], 32)
        self.assertEqual(combined['counts'], {'surface': 28, 'kept': 4, 'unknown': 6})
        self.assertEqual(combined['empty_leakage_fraction'], .125)
        self.assertEqual(combined['selection_modes'], ['interpolated'])
        self.assertIsNone(lookup[(10., 'bin_a', 'loaded')]['empty_leakage_fraction'])
        self.assertIsNone(lookup[(10., 'bin_b', 'empty')]['empty_leakage_fraction'])
        self.assertEqual(lookup[(4., 'bin_a', 'empty')]['selection_modes'], ['exact'])
        self.assertEqual(reports, before)
        self.assertEqual(summarize_exposures([]), [])


if __name__ == '__main__':
    unittest.main()
