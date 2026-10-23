"""Synthetic geometry checks; no ROS, cameras, filesystem captures, or GPU."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import GeometryFitError, classify, fit_bin, validate_disjoint_bins


def configuration():
    angle = np.deg2rad(23)
    rotation = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [523, -419, -125]
    return {
        "bin_id": "bin_a", "robot_from_bin_seed": transform.tolist(),
        "interior_polygon_mm": [[-50, -40], [50, -40], [50, 40], [-50, 40]],
        "floor_search_band_mm": 1.0, "wall_search_band_mm": 4.0,
        "wall_height_range_mm": [3.0, 35.0], "max_height_mm": 45.0,
        "fit": {"iterations": 100, "min_points_per_surface": 35,
                "ransac_distance_mm": 0.2, "max_tolerance_mm": 0.8,
                "alignment_margin_mm": 0.1, "random_seed": 21},
    }


def to_robot(points, config):
    transform = np.asarray(config["robot_from_bin_seed"])
    return points @ transform[:3, :3].T + transform[:3, 3]


def empty_clouds(config, count=4, missing_wall=None, curvature=0.0, partial_floor=False):
    x, y = np.meshgrid(np.linspace(-50, 50, 31), np.linspace(-40, 40, 25))
    floor = np.column_stack((x.ravel(), y.ravel(), curvature * (x.ravel() / 50) ** 2))
    if partial_floor:
        floor = floor[(np.abs(floor[:, 0]) < 12) & (np.abs(floor[:, 1]) < 12)]
    z, along = np.meshgrid(np.linspace(0, 40, 25), np.linspace(-1, 1, 51))
    z, along = z.ravel(), along.ravel()
    surfaces = [floor,
                np.column_stack((50 * along, np.full_like(z, -40), z)),
                np.column_stack((np.full_like(z, 50), 40 * along, z)),
                np.column_stack((50 * along, np.full_like(z, 40), z)),
                np.column_stack((-50 + 0.05 * z, 40 * along, z))]
    rng = np.random.default_rng(391)
    captures = []
    for capture in range(count):
        used = [surface for index, surface in enumerate(surfaces)
                if not (capture == count - 1 and index == missing_wall)]
        points = np.concatenate(used)
        points = points + rng.normal(0, 0.025, points.shape)
        captures.append(to_robot(points, config))
    return captures


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = configuration()
        cls.clouds = empty_clouds(cls.config)
        cls.model = fit_bin(cls.clouds, cls.config)

    def test_plane_fit_serialization_and_repeated_capture_diagnostics(self):
        model = json.loads(json.dumps(self.model, allow_nan=False))
        self.assertEqual(model["bin_id"], "bin_a")
        self.assertEqual(model["diagnostics"]["capture_count"], 4)
        self.assertAlmostEqual(np.linalg.norm(model["floor"]["normal"]), 1)
        expected = np.array([1, 0, -0.05])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(model["walls"][3]["normal"], expected, atol=0.002)
        for plane in [model["floor"], *model["walls"]]:
            self.assertEqual(len(plane["diagnostics"]["per_capture"]), 4)
            for capture in plane["diagnostics"]["per_capture"]:
                self.assertGreater(capture["inlier_fraction"], 0.95)
                self.assertGreater(min(capture["coverage_fraction"]), 0.5)
            self.assertLess(plane["tolerance_mm"], 0.3)

    def test_thin_parts_tilted_wall_other_bin_and_invalid_points(self):
        # Last wall slopes inward by 0.05 mm per mm height.
        local = np.array([[0, 0, 0], [0, 0, 1], [-49, 0, 20], [-48, 0, 20],
                          [0, -40, 20], [0, 170, 8], [0, 0, 46], [0, 0, -2],
                          [np.nan, 0, 1], [0, np.inf, 1]])
        robot = np.full_like(local, np.nan)
        robot[:8] = to_robot(local[:8], self.config)
        robot[8:] = local[8:]
        keep, reasons = classify(robot, self.model)
        np.testing.assert_array_equal(keep, [False, True, False, True, False, False, False, False, False, False])
        self.assertTrue(reasons["floor"][0])
        self.assertTrue(reasons["wall"][[2, 4]].all())
        self.assertTrue(reasons["outside"][[5, 6, 7]].all())
        self.assertTrue(reasons["invalid"][[8, 9]].all())
        np.testing.assert_array_equal(sum(mask.astype(int) for mask in reasons.values()), np.ones(10))
        empty, masks = classify(np.empty((0, 3)), self.model)
        self.assertEqual(empty.shape, (0,))
        self.assertTrue(all(mask.shape == (0,) for mask in masks.values()))

    def test_held_out_empty_capture_is_removed(self):
        # Different noise realization; the calibration cloud is not the test input.
        held_out = empty_clouds(self.config, count=5)[-1]
        keep, _ = classify(held_out, self.model)
        self.assertEqual(int(keep.sum()), 0)

    def test_each_capture_must_support_every_wall(self):
        with self.assertRaisesRegex(GeometryFitError, "wall 3: capture 3"):
            fit_bin(empty_clouds(self.config, missing_wall=4), self.config)

    def test_spatially_concentrated_floor_is_rejected(self):
        with self.assertRaisesRegex(GeometryFitError, "floor: capture 0.*coverage"):
            fit_bin(empty_clouds(self.config, partial_floor=True), self.config)

    def test_curved_floor_cannot_be_silently_replaced_by_plane(self):
        config = copy.deepcopy(self.config)
        config["floor_search_band_mm"] = 6
        with self.assertRaisesRegex(GeometryFitError, "floor:.*planar support"):
            fit_bin(empty_clouds(config, curvature=5), config)

    def test_floor_and_wall_uncertainty_are_measured_separately(self):
        config = copy.deepcopy(self.config)
        config["fit"].update({"ransac_distance_mm": 0.5, "min_tolerance_mm": 0.05})
        captures = empty_clouds(config)
        transform = np.asarray(config["robot_from_bin_seed"])
        rng = np.random.default_rng(915)
        for index, robot in enumerate(captures):
            local = (robot - transform[:3, 3]) @ transform[:3, :3]
            floor = (np.abs(local[:, 2]) < 1) & (np.abs(local[:, 0]) < 44) & (np.abs(local[:, 1]) < 34)
            local[floor, 2] += rng.normal(0, 0.12, int(floor.sum()))
            captures[index] = to_robot(local, config)
        model = fit_bin(captures, config)
        self.assertGreater(model["floor"]["tolerance_mm"],
                           max(wall["tolerance_mm"] for wall in model["walls"]) + 0.1)

    def test_excessive_measured_tolerance_is_rejected_not_clipped(self):
        config = copy.deepcopy(self.config)
        config["fit"].update({"alignment_margin_mm": 0.19, "max_tolerance_mm": 0.2})
        with self.assertRaisesRegex(GeometryFitError, "measured tolerance.*exceeds maximum"):
            fit_bin(self.clouds, config)

    def test_invalid_configuration_rejected(self):
        changes = [
            ("robot_from_bin_seed", np.diag([1, 1, -1, 1]).tolist()),
            ("robot_from_bin_seed", np.diag([2, 1, 1, 1]).tolist()),
            ("interior_polygon_mm", [[-50, -40], [-50, 40], [50, 40], [50, -40]]),
            ("interior_polygon_mm", [[-50, -40], [50, -40], [0, 0], [50, 40], [-50, 40]]),
            ("wall_search_band_mm", 100), ("wall_height_range_mm", [0, 30]),
            ("floor_search_band_mm", float("nan")), ("max_height_mm", 5),
            ("fit", {"unsupported_typo": 1}), ("fit", {"iterations": 0}),
            ("fit", {"residual_quantile": 0.1}),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.config)
                config[key] = value
                with self.assertRaises(GeometryFitError):
                    fit_bin(self.clouds, config)
        with self.assertRaisesRegex(GeometryFitError, "at least two"):
            fit_bin(self.clouds[:1], self.config)

    def test_invalid_model_is_not_a_pass_through(self):
        model = copy.deepcopy(self.model)
        model["walls"][0]["normal"] = [0, -1, 0]
        with self.assertRaises(GeometryFitError):
            classify(np.array([[0, 0, 1]]), model)
        model = copy.deepcopy(self.model)
        model["floor"]["tolerance_mm"] = 100
        with self.assertRaises(GeometryFitError):
            classify(np.array([[0, 0, 1]]), model)

    def test_overlapping_bin_models_are_rejected(self):
        with self.assertRaisesRegex(GeometryFitError, "bin_a.*bin_b.*overlapping usable volumes"):
            validate_disjoint_bins({"bin_a": self.model, "bin_b": copy.deepcopy(self.model)})

    def test_disjoint_models_and_exact_face_contact(self):
        # Analytic boxes allow exact usable-boundary contact despite 0.2 mm
        # surface tolerances. Their common frame is rotated/translated in robot.
        model = copy.deepcopy(self.model)
        model["floor"].update(normal=[0, 0, 1], offset_mm=0.0, tolerance_mm=0.2)
        for wall, normal, offset in zip(model["walls"], [[0, 1, 0], [-1, 0, 0], [0, -1, 0], [1, 0, 0]],
                                       [40.0, 50.0, 40.0, 50.0]):
            wall.update(normal=normal, offset_mm=offset, tolerance_mm=0.2)
        def shifted(distance):
            other = copy.deepcopy(model)
            transform = np.asarray(other["robot_from_bin"])
            transform[:3, 3] += transform[:3, :3] @ np.array([distance, 0, 0])
            other["robot_from_bin"] = transform.tolist()
            return other
        separated = validate_disjoint_bins({"bin_a": model, "bin_b": shifted(200.0)})
        self.assertEqual(separated["pairs"][0]["relation"], "disjoint")
        touching = validate_disjoint_bins({"bin_a": model, "bin_b": shifted(99.6)})
        self.assertEqual(touching["pairs"][0]["relation"], "boundary_contact")
        self.assertEqual(touching["pairs"][0]["intersection_dimension"], 2)
        with self.assertRaisesRegex(GeometryFitError, "overlapping usable volumes"):
            validate_disjoint_bins({"bin_a": model, "bin_b": shifted(99.5)})


if __name__ == "__main__":
    unittest.main()
