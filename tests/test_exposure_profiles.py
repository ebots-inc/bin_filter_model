"""Measured exposure selection and conservative interpolation, without hardware."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exposure_profiles import EXPOSURE_ATOL_MS, MODEL_TYPE, select_model, validate_bundle
from height_field import HeightFieldError, classify, fit_bin


def _training(label, count):
    return [{"id": f"{label}_{index}",
             "input_signature": hashlib.sha256(f"{label}_signature_{index}".encode()).hexdigest()}
            for index in range(count)]


def _field(capture_count, shift=0.0):
    x, y = np.meshgrid(np.arange(7), np.arange(7))
    xy = np.concatenate([np.column_stack((x.ravel(), y.ravel())) + offset
                         for offset in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))])
    clouds = [np.column_stack((xy, np.full(len(xy), -100 + shift + index * 0.005)))
              for index in range(capture_count)]
    return fit_bin(clouds, {
        "kind": "height_field", "bin_id": "bin_a", "xy_bounds_mm": [[0, 0], [7, 7]],
        "z_bounds_mm": [-135, -85], "seed_xy_mm": [1.5, 1.5],
        "max_reference_height_mm": -90, "min_capture_support": 2,
        "min_tolerance_mm": 1.0, "max_tolerance_mm": 4.0,
    })


def _bins(field):
    other = copy.deepcopy(field)
    other["bin_id"] = "bin_b"
    for row in other["xy_bounds_mm"]:
        row[0] += 20
    other["seed_xy_mm"][0] += 20
    other["x_centers_mm"] = [value + 20 for value in other["x_centers_mm"]]
    return {"bin_a": field, "bin_b": other}


def _mask_tolerance(field, mask):
    values = np.asarray(field["tolerance_mm"], dtype=object)
    values[~mask] = None
    field["tolerance_mm"] = values.tolist()


class ExposureProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        low = {"exposure_ms": 4.0, "training_captures": _training("low", 2), "bins": _bins(_field(2))}
        high = {"exposure_ms": 20.0, "training_captures": _training("high", 3), "bins": _bins(_field(3, 2))}
        cls.bundle = {
            "model_type": MODEL_TYPE, "frame_id": "B2", "length_unit": "mm",
            "scene_id": 0, "calibration_sha256": {"camera": "a" * 64, "robot": "b" * 64},
            "training_captures": copy.deepcopy(low["training_captures"] + high["training_captures"]),
            "interpolation": {"method": "linear_height_max_tolerance", "margin_mm": 0.5},
            "profiles": [low, high], "validation_status": "requires_held_out_review",
        }

    def test_exact_anchors_and_legacy_preserve_bins_and_do_not_mutate_source(self):
        before = json.dumps(self.bundle, sort_keys=True, allow_nan=False)
        for profile in self.bundle["profiles"]:
            with self.subTest(exposure=profile["exposure_ms"]):
                selected = select_model(self.bundle, profile["exposure_ms"])
                self.assertNotIn('schema_version', selected)
                self.assertEqual(selected["model_type"], "offline_height_field_bins")
                self.assertEqual(selected["bins"], profile["bins"])
                self.assertEqual(selected["training_captures"], profile["training_captures"])
                self.assertEqual(selected["exposure_selection"]["mode"], "exact")
                validate_bundle(selected)
                legacy = select_model(selected, profile["exposure_ms"])
                self.assertEqual(legacy["bins"], profile["bins"])
                self.assertEqual(legacy["exposure_selection"]["mode"], "single")
                with self.assertRaisesRegex(HeightFieldError, "exposure differs"):
                    select_model(selected, 10)
                selected["bins"]["bin_a"]["surface_height_mm"][0][0] = -110
        self.assertEqual(json.dumps(self.bundle, sort_keys=True, allow_nan=False), before)

    def test_arbitrary_intermediate_values_blend_height_and_use_max_tolerance_plus_margin(self):
        bundle = copy.deepcopy(self.bundle)
        bundle["profiles"][1]["bins"]["bin_a"]["tolerance_mm"][2][2] = 2.0
        lower = np.asarray(bundle["profiles"][0]["bins"]["bin_a"]["surface_height_mm"])
        upper = np.asarray(bundle["profiles"][1]["bins"]["bin_a"]["surface_height_mm"])
        for exposure in (4.0001, 5.37, 10.0, 13.123456, 19.9999):
            selected = select_model(bundle, exposure)
            weight = (exposure - 4.0) / 16.0
            field = selected["bins"]["bin_a"]
            np.testing.assert_allclose(field["surface_height_mm"], (1 - weight) * lower + weight * upper)
            self.assertEqual(field["tolerance_mm"][2][2], 2.5)
            self.assertEqual(field["tolerance_mm"][1][1], 1.5)
            self.assertEqual(selected["exposure_selection"], {
                "requested_exposure_ms": exposure, "lower_exposure_ms": 4.0,
                "upper_exposure_ms": 20.0, "weight": weight, "mode": "interpolated",
            })
            np.testing.assert_array_equal(field["capture_support"], np.full((7, 7), 2))
            np.testing.assert_array_equal(field["residual_capture_support"], np.full((6, 6), 2))
            json.dumps(selected, allow_nan=False)

    def test_selection_uses_adjacent_anchors_in_a_larger_bundle(self):
        bundle = copy.deepcopy(self.bundle)
        middle = {"exposure_ms": 8.0, "training_captures": _training("middle", 2),
                  "bins": _bins(_field(2, 0.25))}
        bundle["profiles"].insert(1, middle)
        bundle["training_captures"].extend(copy.deepcopy(middle["training_captures"]))
        for exposure, lower, upper in ((6, 4, 8), (10, 8, 20)):
            selected = select_model(bundle, exposure)
            self.assertEqual(selected["exposure_selection"]["lower_exposure_ms"], lower)
            self.assertEqual(selected["exposure_selection"]["upper_exposure_ms"], upper)
            self.assertEqual(selected["exposure_selection"]["weight"], (exposure - lower) / (upper - lower))
        self.assertEqual(select_model(bundle, 8)["bins"], middle["bins"])

    def test_optional_bin_ids_do_not_change_compatible_geometry(self):
        bundle = copy.deepcopy(self.bundle)
        bundle["profiles"][1]["bins"]["bin_a"].pop("bin_id")
        self.assertEqual(select_model(bundle, 10)["exposure_selection"]["mode"], "interpolated")

    def test_float_roundoff_matching_and_outside_rejection_have_no_relative_tolerance(self):
        for exposure in (4.0 - 0.5 * EXPOSURE_ATOL_MS, 4.0 + 0.5 * EXPOSURE_ATOL_MS,
                         20.0 - 0.5 * EXPOSURE_ATOL_MS, 20.0 + 0.5 * EXPOSURE_ATOL_MS):
            self.assertEqual(select_model(self.bundle, exposure)["exposure_selection"]["mode"], "exact")
        for exposure in (3.9, 20.1, 4.0 - 2 * EXPOSURE_ATOL_MS, 20.0 + 2 * EXPOSURE_ATOL_MS):
            with self.subTest(exposure=exposure), self.assertRaisesRegex(HeightFieldError, "no extrapolation"):
                select_model(self.bundle, exposure)
        for exposure in (True, None, "10", 0, -4, np.nan, np.inf):
            with self.subTest(exposure=exposure), self.assertRaises(HeightFieldError):
                select_model(self.bundle, exposure)

    def test_unknown_vertices_and_squares_at_either_endpoint_remain_unknown(self):
        bundle = copy.deepcopy(self.bundle)
        low, high = [profile["bins"]["bin_a"] for profile in bundle["profiles"]]
        low["tolerance_mm"][3][3] = None
        high["surface_height_mm"][5][5] = None
        for y in (4, 5):
            for x in (4, 5):
                high["tolerance_mm"][y][x] = None
        selected = select_model(bundle, 9.25)
        field = selected["bins"]["bin_a"]
        self.assertIsNone(field["surface_height_mm"][5][5])
        self.assertIsNone(field["tolerance_mm"][3][3])
        self.assertIsNone(field["tolerance_mm"][4][4])
        _, masks = classify(np.array([[4, 4, -95], [5.5, 5.5, -95]]), field)
        np.testing.assert_array_equal(masks["unknown"], [True, True])

    def test_excessive_intermediate_tolerance_becomes_unknown_and_seed_loss_is_rejected(self):
        bundle = copy.deepcopy(self.bundle)
        high = bundle["profiles"][1]["bins"]["bin_a"]
        high["tolerance_mm"][3][3] = 3.75
        selected = select_model(bundle, 11.1)
        self.assertIsNone(selected["bins"]["bin_a"]["tolerance_mm"][3][3])
        self.assertEqual(selected["bins"]["bin_a"]["diagnostics"]["interpolation_squares_excessive_tolerance"], 1)
        self.assertEqual(select_model(bundle, 20)["bins"]["bin_a"]["tolerance_mm"][3][3], 3.75)
        high["tolerance_mm"][1][1] = 3.75
        with self.assertRaisesRegex(HeightFieldError, "unknown/unreliable interpolated"):
            select_model(bundle, 12)
        self.assertEqual(select_model(bundle, 20)["exposure_selection"]["mode"], "exact")

    def test_common_coverage_keeps_only_the_seed_connected_component(self):
        bundle = copy.deepcopy(self.bundle)
        low_mask, high_mask = np.zeros((6, 6), bool), np.zeros((6, 6), bool)
        low_mask[1, 1:5] = True
        low_mask[1:5, 4] = True
        high_mask[1:5, 1] = True
        high_mask[4, 1:5] = True
        _mask_tolerance(bundle["profiles"][0]["bins"]["bin_a"], low_mask)
        _mask_tolerance(bundle["profiles"][1]["bins"]["bin_a"], high_mask)
        selected = select_model(bundle, 10)
        finite = np.isfinite(np.asarray(selected["bins"]["bin_a"]["tolerance_mm"], dtype=float))
        self.assertEqual(int(finite.sum()), 1)
        self.assertTrue(finite[1, 1])
        self.assertFalse(finite[4, 4])

    def test_intermediate_surface_rejection_and_thin_foreground_retention(self):
        field = select_model(self.bundle, 10)["bins"]["bin_a"]
        height = field["surface_height_mm"][1][1]
        points = np.array([[1.75, 1.75, height], [1.75, 1.75, height + 2],
                           [1.75, 1.75, height - 3], [40, 40, height + 2]])
        kept, masks = classify(points, field)
        np.testing.assert_array_equal(kept, [False, True, False, False])
        self.assertTrue(masks["surface"][0])
        np.testing.assert_array_equal(sum(mask.astype(int) for mask in masks.values()), np.ones(4))

    def test_malformed_bundle_and_provenance_are_rejected_even_at_an_exact_anchor(self):
        changes = [
            ("frame", lambda m: m.update(frame_id="A1")),
            ("unit", lambda m: m.update(length_unit="m")),
            ("scene", lambda m: m.update(scene_id=True)),
            ("calibration", lambda m: m.update(calibration_sha256={"camera": "wrong"})),
            ("uppercase calibration", lambda m: m["calibration_sha256"].update(camera="A" * 64)),
            ("method", lambda m: m["interpolation"].update(method="linear_tolerance")),
            ("negative margin", lambda m: m["interpolation"].update(margin_mm=-0.1)),
            ("infinite margin", lambda m: m["interpolation"].update(margin_mm=np.inf)),
            ("one anchor", lambda m: m.update(profiles=m["profiles"][:1])),
            ("unsorted", lambda m: m["profiles"].reverse()),
            ("too close", lambda m: m["profiles"][1].update(exposure_ms=4 + EXPOSURE_ATOL_MS)),
            ("invalid exposure", lambda m: m["profiles"][1].update(exposure_ms=False)),
            ("missing bin", lambda m: m["profiles"][1]["bins"].pop("bin_b")),
            ("bin id", lambda m: m["profiles"][1]["bins"]["bin_b"].update(bin_id="bin_a")),
            ("corrupt height", lambda m: m["profiles"][1]["bins"]["bin_a"]["surface_height_mm"][0].pop()),
            ("excess support", lambda m: m["profiles"][1]["bins"]["bin_a"]["capture_support"][0].__setitem__(0, 4)),
            ("missing fit count", lambda m: m["profiles"][1]["bins"]["bin_a"]["diagnostics"].pop("capture_count")),
            ("fraction support", lambda m: m["profiles"][1]["bins"]["bin_a"]["residual_capture_support"][0].__setitem__(0, 2.5)),
            ("unknown root capture", lambda m: m["training_captures"].append({"id": "extra", "input_signature": "e" * 64})),
            ("omitted root capture", lambda m: m["training_captures"].pop()),
            ("changed group metadata", lambda m: m["training_captures"][0].update(layout_group="wrong")),
            ("duplicate id", lambda m: m["profiles"][1]["training_captures"][0].update(id="low_0")),
            ("duplicate input", lambda m: m["profiles"][1]["training_captures"][0].update(input_signature=m["profiles"][0]["training_captures"][0]["input_signature"])),
            ("invalid signature", lambda m: m["profiles"][1]["training_captures"][0].update(input_signature="not-a-hash")),
            ("profile calibration", lambda m: m["profiles"][1].update(calibration_sha256={"camera": "c" * 64, "robot": "b" * 64})),
            ("profile frame", lambda m: m["profiles"][1].update(frame_id="A1")),
        ]
        for label, mutate in changes:
            with self.subTest(change=label):
                bundle = copy.deepcopy(self.bundle)
                mutate(bundle)
                with self.assertRaises(HeightFieldError):
                    select_model(bundle, 4)

    def test_incompatible_valid_geometry_seed_and_fit_parameters_are_rejected(self):
        for change in ("seed", "fit_parameters", "geometry"):
            with self.subTest(change=change):
                bundle = copy.deepcopy(self.bundle)
                field = bundle["profiles"][1]["bins"]["bin_a"]
                if change == "seed":
                    field["seed_xy_mm"] = [2.5, 2.5]
                elif change == "fit_parameters":
                    field["fit_parameters"]["residual_quantile"] = 0.95
                else:
                    field["xy_bounds_mm"] = [[0.1, 0], [7.1, 7]]
                    field["x_centers_mm"] = [value + 0.1 for value in field["x_centers_mm"]]
                with self.assertRaisesRegex(HeightFieldError, "incompatible"):
                    select_model(bundle, 4)


class PooledExposureTests(unittest.TestCase):
    def setUp(self):
        self.model = {
            "model_type": "offline_height_field_bins",
            "frame_id": "B2", "length_unit": "mm", "scene_id": 0,
            "calibration_sha256": {"camera": "a" * 64, "robot": "b" * 64},
            "training_captures": _training("pooled", 3),
            "exposure_settings_ms": [4.0, 20.0], "bins": _bins(_field(3)),
        }

    def test_pooled_grid_is_unchanged_for_any_or_missing_capture_exposure(self):
        before = copy.deepcopy(self.model)
        for exposure in (None, 4.0, 12.5, 20.0, 0.1, 50.0):
            selected = select_model(self.model, exposure)
            self.assertEqual(selected["bins"], self.model["bins"])
            self.assertNotIn("exposure_ms", selected)
            self.assertEqual(selected["exposure_selection"], {
                "mode": "pooled", "exposure_settings_ms": [4.0, 20.0],
            })
            selected["bins"]["bin_a"]["surface_height_mm"][0][0] = -110
        self.assertEqual(self.model, before)

    def test_invalid_pooled_provenance_and_ambiguous_model_are_rejected(self):
        for settings in (None, [], [4.0], [20.0, 4.0], [4.0, 4.0], [4.0, False], [4.0, float('nan')]):
            with self.subTest(settings=settings), self.assertRaises(HeightFieldError):
                validate_bundle(self.model | {"exposure_settings_ms": settings})
        for key, value in (("exposure_ms", 4.0), ("profiles", []), ("interpolation", {})):
            with self.subTest(key=key), self.assertRaisesRegex(HeightFieldError, "one shared height field"):
                validate_bundle(self.model | {key: value})

    def test_pooled_selection_preserves_geometry_and_provenance_checks(self):
        self.model["bins"]["bin_a"]["capture_support"][0][0] = 4
        with self.assertRaisesRegex(HeightFieldError, "exceeds profile training capture count"):
            select_model(self.model)


if __name__ == "__main__":
    unittest.main()
