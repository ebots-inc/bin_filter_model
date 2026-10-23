"""Synthetic curved surfaces exercise measured references without real data."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from height_field import HeightFieldError, classify, fit_bin, validate_disjoint_bins


def config():
    return {"kind": "height_field", "bin_id": "bin_a", "xy_bounds_mm": [[-16, -14], [16, 14]],
            "z_bounds_mm": [-135, -85], "seed_xy_mm": [0, 0], "max_reference_height_mm": -93,
            "alignment_margin_mm": 0.5, "min_tolerance_mm": 1.0, "max_tolerance_mm": 4.0}


def bowl(x, y):
    return -100 + 0.025 * x * x + 0.02 * y * y


def captures(count=7, inconsistent=False):
    rng = np.random.default_rng(87)
    results = []
    x, y = np.meshgrid(np.arange(-16, 16), np.arange(-14, 14))
    origins = np.column_stack((x.ravel(), y.ravel()))
    for capture in range(count):
        xy = np.repeat(origins, 12, axis=0) + rng.uniform(0.03, 0.97, (len(origins) * 12, 2))
        inner = (np.abs(xy[:, 0]) < 12) & (np.abs(xy[:, 1]) < 10)
        rim = (np.abs(xy[:, 0]) < 14) & (np.abs(xy[:, 1]) < 12)
        z = np.where(inner, bowl(xy[:, 0], xy[:, 1]), np.where(rim, -88.0, -125.0))
        z += rng.normal(0, 0.025, len(z)) + rng.choice([-0.1, 0.1], len(z))
        if inconsistent and capture == count - 1:
            region = (xy[:, 0] > -7) & (xy[:, 0] < -5) & (xy[:, 1] > 4) & (xy[:, 1] < 6)
            z[region] += 8
        # A genuinely unobserved hole remains missing across all captures.
        hole = (xy[:, 0] > 2) & (xy[:, 0] < 4) & (xy[:, 1] > 2) & (xy[:, 1] < 4)
        results.append(np.column_stack((xy[~hole], z[~hole])))
    return results


class HeightFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = config()
        cls.clouds = captures()
        cls.model = fit_bin(cls.clouds, cls.config)

    def test_curved_reference_serialization_and_empirical_support(self):
        model = json.loads(json.dumps(self.model, allow_nan=False))
        self.assertNotIn('schema_version', model)
        self.assertEqual(model["diagnostics"]["capture_count"], 7)
        self.assertGreater(model["diagnostics"]["interpolation_squares_reliable"], 300)
        self.assertLess(model["diagnostics"]["reference_cells_connected"],
                        model["diagnostics"]["reference_cells_below_rim"])
        self.assertTrue(all(item["residual_point_count"] > 3000 for item in model["diagnostics"]["captures"]))

    def test_thin_part_high_part_holes_table_other_bin_and_invalid(self):
        points = np.array([[0, 0, bowl(0, 0)], [0, 0, bowl(0, 0) + 2], [1, -1, -86],
                           [3, 3, bowl(3, 3) + 5], [15, 0, -125], [50, 0, -95],
                           [0, 0, -110], [np.nan, 0, -95], [0, np.inf, -95]])
        keep, masks = classify(points, self.model)
        np.testing.assert_array_equal(keep, [False, True, True, False, False, False, False, False, False])
        self.assertTrue(masks["unknown"][[3, 4]].all())
        self.assertTrue(masks["outside"][[5, 6]].all())
        self.assertTrue(masks["surface"][0])
        np.testing.assert_array_equal(sum(mask.astype(int) for mask in masks.values()), np.ones(len(points)))

    def test_independent_empty_capture_has_no_foreground(self):
        held_out = captures(count=8)[-1]
        keep, masks = classify(held_out, self.model)
        self.assertEqual(int(keep.sum()), 0)
        self.assertGreater(int(masks["unknown"].sum()), 0)

    def test_inconsistent_surface_becomes_unknown_without_inflating_tolerance(self):
        model = fit_bin(captures(inconsistent=True), self.config)
        self.assertGreater(model["diagnostics"]["interpolation_squares_excessive_tolerance"], 0)
        _, masks = classify(np.array([[-6, 5, bowl(-6, 5) + 2]]), model)
        self.assertTrue(masks["unknown"][0])
        finite = np.asarray(model["tolerance_mm"], dtype=float)
        self.assertLessEqual(float(np.nanmax(finite)), 4.0)

    def test_missing_support_or_invalid_configuration_is_rejected(self):
        with self.assertRaisesRegex(HeightFieldError, "at least 5"):
            fit_bin(self.clouds[:4], self.config)
        for change in ({"cell_size_mm": 0}, {"min_capture_support": 1},
                       {"cell_size_mm": 1e-308},
                       {"min_points_per_cell": 1.5}, {"residual_quantile": 0.5},
                       {"xy_bounds_mm": [[1, 1], [0, 0]]}, {"seed_xy_mm": [100, 0]},
                       {"max_tolerance_mm": 0.1}, {"not_supported": 1}):
            with self.subTest(change=change), self.assertRaises(HeightFieldError):
                fit_bin(self.clouds, self.config | change)
        shifted = [points + [0, 0, 10 * (index % 2)] for index, points in enumerate(self.clouds)]
        with self.assertRaisesRegex(HeightFieldError, "unknown/unreliable"):
            fit_bin(shifted, self.config)

    def test_model_integrity_and_named_bin_exclusivity(self):
        invalid = copy.deepcopy(self.model)
        invalid["max_height_above_surface_mm"] = 200
        with self.assertRaises(HeightFieldError):
            classify(np.empty((0, 3)), invalid)
        invalid = copy.deepcopy(self.model)
        heights = np.asarray(invalid["surface_height_mm"], dtype=float)
        y, x = np.argwhere(np.isfinite(heights))[0]
        invalid["surface_height_mm"][y][x] = -200
        with self.assertRaisesRegex(HeightFieldError, "fitting Z"):
            classify(np.empty((0, 3)), invalid)
        with self.assertRaisesRegex(HeightFieldError, "overlapping XY bounds"):
            validate_disjoint_bins({"bin_a": self.model, "bin_b": self.model})
        other = copy.deepcopy(self.model)
        for row in other["xy_bounds_mm"]:
            row[0] += 32
        other["seed_xy_mm"][0] += 32
        other["x_centers_mm"] = [x + 32 for x in other["x_centers_mm"]]
        report = validate_disjoint_bins({"bin_a": self.model, "bin_b": other})
        self.assertEqual(report["status"], "disjoint_xy_interiors")


if __name__ == "__main__":
    unittest.main()
