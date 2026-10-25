"""Empty-scene layout proposals use measured depth, never operator coordinates."""
import base64
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import DatasetError
import height_field
import layout_geometry as layout


def measured_bowl(offset=(0., 0., 0.), *, missing_quadrant=False):
    """Known 80 mm round rim, 40 mm bowl depth, and visible outside table."""
    xx, yy = np.meshgrid(np.arange(-49.5, 50), np.arange(-49.5, 50))
    radius = np.hypot(xx, yy)
    zz = np.where(radius < 35, -20 + 40 * (radius / 35) ** 2, np.where(radius < 41, 20., -25.))
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    # Multiple independent camera pixels in each 1 mm raster cell.
    points = np.concatenate([points + [dx, dy, dz] for dx, dy, dz in
                             ((-.1, -.1, -.02), (.1, -.1, .02), (-.1, .1, .01), (.1, .1, -.01))])
    if missing_quadrant:
        points = points[~((points[:, 0] > 0) & (points[:, 1] > 0) & (points[:, 2] > 15))]
    return points + offset


class LayoutGeometryTests(unittest.TestCase):
    def setUp(self):
        self.points = measured_bowl((500., -300., -90.))
        self.bounds = [[450., -350.], [550., -250.]]

    def test_estimates_all_needed_bounds_and_seed_from_observed_empty_bowl(self):
        result = layout.infer_bin(self.points, self.bounds)
        config, details = result['config'], result['diagnostics']
        np.testing.assert_allclose(config['xy_bounds_mm'], self.bounds)
        np.testing.assert_allclose(config['seed_xy_mm'], [500., -300.], atol=2.)
        self.assertAlmostEqual(details['rim_height_mm'], -70., places=2)
        self.assertAlmostEqual(details['height_mm'], 40., delta=1.)
        self.assertLess(config['z_bounds_mm'][0], -110.)
        self.assertGreater(config['z_bounds_mm'][1], -70.)
        self.assertLess(config['max_reference_height_mm'], -70.)
        self.assertGreater(details['rim_points'], 200)
        self.assertGreater(details['boundary_points'], 80)
        self.assertGreaterEqual(min(details['boundary_quadrants']), 20)
        height_field._configuration({k: v for k, v in config.items() if k != 'registration'})

    def test_missing_depth_at_rectangle_center_selects_an_observed_interior_seed(self):
        radius = np.linalg.norm(self.points[:, :2] - [500., -300.], axis=1)
        result = layout.infer_bin(self.points[radius > 7], self.bounds)
        seed = result['config']['seed_xy_mm']
        self.assertGreater(np.linalg.norm(np.asarray(seed) - [500., -300.]), 7.)
        observed = np.asarray(result['diagnostics']['interior_xyz_mm'])
        self.assertTrue(np.isfinite(observed).all())

    def test_flat_table_does_not_become_a_bin(self):
        table = self.points.copy()
        table[:, 2] = -90
        with self.assertRaisesRegex(DatasetError, 'complete top rim'):
            layout.infer_bin(table, self.bounds)

    def test_incomplete_rim_cannot_be_accepted_from_three_sides(self):
        partial = measured_bowl((500., -300., -90.), missing_quadrant=True)
        with self.assertRaisesRegex(DatasetError, 'complete top rim'):
            layout.infer_bin(partial, self.bounds)

    def test_two_pockets_in_one_region_require_separate_marks(self):
        points = np.concatenate((measured_bowl((-50., 0., 0.)), measured_bowl((50., 0., 0.))))
        with self.assertRaisesRegex(DatasetError, 'complete top rim'):
            layout.infer_bin(points, [[-100., -50.], [100., 50.]])

    def test_a_box_cutting_the_interior_is_rejected_even_with_rim_in_every_quadrant(self):
        with self.assertRaisesRegex(DatasetError, 'cuts into the lower interior'):
            layout.infer_bin(self.points, [[450., -350.], [530., -250.]])

    def test_sparse_and_empty_depth_fail_with_operator_action(self):
        for points in (np.empty((0, 3)), self.points[::200], np.full((1000, 3), np.nan)):
            with self.subTest(size=len(points)), self.assertRaisesRegex(DatasetError, 'depth'):
                layout.infer_bin(points, self.bounds)

    def test_scene_frame_bin_count_and_names_are_not_fixed(self):
        shifted = measured_bowl((650., -300., -90.))
        result = layout.infer_layout(np.concatenate((self.points, shifted)), [
            {'bin_id': 'resistors', 'xy_bounds_mm': self.bounds},
            {'bin_id': 'spares-3', 'xy_bounds_mm': [[600., -350.], [700., -250.]]},
        ], 4, 'A1', min_capture_support=3)
        self.assertEqual(result['settings']['scene_id'], 4)
        self.assertEqual(result['settings']['frame_id'], 'A1')
        self.assertEqual(list(result['settings']['bins']), ['resistors', 'spares-3'])
        self.assertEqual(result['settings']['bins']['resistors']['min_capture_support'], 3)

    def test_ambiguous_names_and_overlapping_regions_reject(self):
        for selections in ([{'bin_id': '../escape', 'xy_bounds_mm': self.bounds}],
                           [{'bin_id': 'bin', 'xy_bounds_mm': self.bounds}] * 2,
                           [{'bin_id': name, 'xy_bounds_mm': self.bounds} for name in ('one', 'two')]):
            with self.subTest(selections=selections), self.assertRaises(DatasetError):
                layout.infer_layout(self.points, selections, 0, 'B2')

    def test_pixel_rectangle_round_trip_uses_original_top_view_mapping(self):
        preview = layout.render_preview(self.points, max_size=700)
        # Test an interior selection: the full ROI can extend past observed bounds.
        bounds = np.asarray([[460., -340.], [540., -260.]])
        pixels = layout._pixels(bounds, preview)
        result = layout.pixels_to_bounds(pixels.ravel(), preview)
        np.testing.assert_allclose(result, bounds, atol=1 / preview['pixels_per_mm'])
        image = cv2.imdecode(np.frombuffer(base64.b64decode(preview['preview_png']), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2], (preview['height'], preview['width']))
        with self.assertRaisesRegex(DatasetError, 'inside the preview'):
            layout.pixels_to_bounds([-1, 0, 200, 200], preview)

    def test_review_overlay_preserves_the_coordinate_mapping(self):
        preview = layout.render_preview(self.points, max_size=500)
        result = layout.infer_layout(self.points, [{'bin_id': 'one', 'xy_bounds_mm': self.bounds}], 0, 'B2')
        overlay = layout.render_preview(self.points, viewport=preview, bins=result['bins'])
        for key in ('width', 'height', 'lower_xy_mm', 'upper_xy_mm', 'pixels_per_mm', 'margin_px'):
            self.assertEqual(overlay[key], preview[key])
        self.assertNotEqual(overlay['preview_png'], preview['preview_png'])


if __name__ == '__main__':
    unittest.main()
