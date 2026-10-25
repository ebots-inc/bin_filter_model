"""Offline tests for session isolation, exposure sweeps and incomplete capture rejection."""

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np
import OpenEXR

import prepare_session as collection


def write_exr(path, image):
    OpenEXR.File({}, {"Y": image}).write(str(path))


class CollectionSessionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bin_collection_test_")
        self.root = Path(self.temporary.name)
        self.artifacts = self.root / "artifacts"
        self.recipe = collection.read_yaml(collection.DEFAULT_RECIPE)
        config_source = collection.resolve_bin_config(collection.ARTIFACTS, self.recipe["bin_config"])
        self.recipe["bin_config"] = "test-bin-config.yaml"
        for key in ("camera_calibration", "robot_calibration", "bin_config"):
            relative = self.recipe[key]
            destination = self.artifacts / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config_source if key == "bin_config" else collection.ARTIFACTS / relative, destination)
        destination = self.artifacts / collection.EYES_WORKFLOW
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(collection.ARTIFACTS / collection.EYES_WORKFLOW, destination)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, mode="empty", **kwargs):
        return collection.prepare_session(self.artifacts, self.root / "sessions", Path("/root/data/bin_model"),
                                          self.recipe, mode=mode, **kwargs)

    def read_session(self, folder):
        return json.loads((folder / "session.json").read_text())

    def complete(self, folder, capture_ids=None):
        session = self.read_session(folder)
        captures = [entry for entry in session["captures"] if capture_ids is None or entry["id"] in capture_ids]
        for stage in session.get("stages", []):
            if not any(entry.get("stage_id") == stage["id"] for entry in captures):
                continue
            path = folder / stage["confirmation_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"session_id": session["session_id"], "stage_id": stage["id"],
                                       "gate_token": stage["id"] + "/confirmed", "confirmed_at": "2026-10-23T12:00:00Z"}))
        for capture in captures:
            directory = folder / capture["path"]
            directory.mkdir(parents=True)
            manifest = {"status": "complete", "simulated": False,
                        "scene_id": session["scene_id"], "frame_id": session["frame_id"], "length_unit": "mm", "depth_axis": "z",
                        "part_name": capture["part_name"], "task_name": "capture", "calibration": {},
                        "requested_settings": {"path": "requested_settings.yaml", "values_are_requested_not_sensor_verified": True},
                        "images": {}, "files": []}
            for kind in ("depth", "spotlight"):
                (directory / kind).mkdir()
                manifest["images"][kind] = []
                for camera in range(2):
                    write_exr(directory / kind / f"Frame{camera}.exr", np.ones((5, 7), np.float32))
                    manifest["images"][kind].append({"camera_index": camera, "height": 5, "width": 7, "dtype": "float32"})
            (directory / "calibration").mkdir()
            for name in ("camera", "robot"):
                relative = f"calibration/{name}.yaml"
                shutil.copyfile(collection.resolve_calibration(folder, session, name), directory / relative)
                manifest["calibration"][name] = {"path": relative}
            settings = {"structured_light_exposure_ms": [capture["exposure_ms"]] * 2,
                        "spotlight_exposure_ms": [capture["spotlight_exposure_ms"]] * 2,
                        "gain": self.recipe["gain"], "black_level": self.recipe["black_level"]}
            (directory / "requested_settings.yaml").write_text(collection.dump_yaml(settings))
            for file in sorted(directory.rglob("*")):
                if file.is_file():
                    manifest["files"].append({"path": file.relative_to(directory).as_posix(), "size_bytes": file.stat().st_size})
            (directory / "capture_manifest.yaml").write_text(collection.dump_yaml(manifest))
        return folder / session["captures"][0]["path"]

    def refresh_size(self, directory, relative):
        path = directory / "capture_manifest.yaml"
        manifest = collection.read_yaml(path)
        for entry in manifest["files"]:
            if entry["path"] == relative:
                entry["size_bytes"] = (directory / relative).stat().st_size
        path.write_text(collection.dump_yaml(manifest))

    def test_empty_plan_uses_existing_deployment_calibration(self):
        folder = self.prepare()
        session = self.read_session(folder)
        self.assertEqual(len(session["captures"]), 50)
        self.assertEqual(sum(c["split"] == "train" for c in session["captures"]), 35)
        self.assertEqual({c["exposure_ms"] for c in session["captures"]}, {4, 8, 12, 16, 20})
        self.assertEqual(len({c["path"] for c in session["captures"]}), 50)
        self.assertTrue(all(c["path"].startswith("captures/") for c in session["captures"]))
        shared = self.artifacts / "data_saving/bin_model/sessions" / folder.name
        self.assertEqual(session["artifacts_root"], str(self.artifacts))
        self.assertFalse((shared / "calibration").exists())
        self.assertFalse((folder / "calibration").exists())
        for name in ("camera", "robot"):
            self.assertEqual(session["calibration"][name], {"artifact_path": self.recipe[f"{name}_calibration"]})
            self.assertEqual(collection.resolve_calibration(folder, session, name),
                             self.artifacts / self.recipe[f"{name}_calibration"])
        for exposure in self.recipe["exposure_ms"]:
            part = collection.read_yaml(folder / "configs" / f"{collection.exposure_name(exposure)}.yaml")
            self.assertEqual(part["image"]["exposure_time_structure_light"], [exposure, exposure])
            self.assertEqual(part["image"]["exposure_time_spot_light"], [exposure, exposure])
            self.assertEqual(part["image"]["random_exposure_time"], [])
            self.assertEqual(part["tasks"][0]["steps"][0]["step"], "capture_data")
            self.assertEqual(part["tasks"][0]["steps"][0]["parameters"]["camera_calibration_path"],
                             "/root/artifacts/" + self.recipe["camera_calibration"])
        self.assertEqual(session["collection_interface"], "standalone")
        self.assertFalse((folder / "eyes_capture.yaml").exists())
        self.assertEqual(session["eyes_config"], {
            "artifact_path": collection.EYES_WORKFLOW,
            "shared_path": "/root/artifacts/" + collection.EYES_WORKFLOW})

    def test_reprepare_cannot_reuse_cached_part_names_or_capture_paths(self):
        first, second = self.prepare(), self.prepare()
        a, b = self.read_session(first), self.read_session(second)
        self.assertNotEqual(first, second)
        self.assertTrue({c["part_name"] for c in a["captures"]}.isdisjoint({c["part_name"] for c in b["captures"]}))
        self.assertTrue((self.artifacts / "config/parts" / (a["captures"][0]["part_name"] + ".yaml")).is_file())

    def test_readable_names_are_unique_and_explicit_names_do_not_overwrite(self):
        first, second = self.prepare(), self.prepare()
        self.assertRegex(first.name, r"^bin-setup-\d{4}-\d{2}-\d{2}-01$")
        self.assertTrue(second.name.endswith("-02"))
        named = self.prepare(name="resistor-baseline")
        self.assertEqual(named.name, "resistor-baseline")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare(name="resistor-baseline")
        for name in ("../escape", "bad name", ""):
            with self.assertRaises(ValueError):
                self.prepare(name=name)

    def test_default_guided_plan_has_operator_stages_and_loaded_scenes_are_test_only(self):
        folder = collection.prepare_session(self.artifacts, self.root / "sessions",
                                             Path("/root/data/bin_model"), self.recipe)
        session = self.read_session(folder)
        self.assertEqual(len(session["captures"]), 170)
        self.assertEqual(len(session["stages"]), 26)
        self.assertEqual(sum(c["split"] == "train" for c in session["captures"]), 35)
        self.assertTrue(all(c["split"] == "test" for c in session["captures"] if "loaded" in c["bins"].values()))
        self.assertEqual(session["collection_interface"], "standalone")
        self.assertNotIn("workflow_calibration", session)
        self.assertFalse((folder / "collect.xml").exists())
        self.assertFalse((folder / "workflow.yaml").exists())
        self.assertFalse((self.artifacts / "sequence").exists())
        self.assertEqual(list((self.artifacts / "workflow_configs").iterdir()),
                         [self.artifacts / collection.EYES_WORKFLOW])
        shared = self.artifacts / "data_saving/bin_model/sessions" / folder.name
        self.assertEqual(session["artifacts_root"], str(self.artifacts))
        self.assertFalse((shared / "calibration").exists())
        self.assertFalse((folder / "calibration").exists())
        for name in ("camera", "robot"):
            self.assertEqual(session["calibration"][name], {"artifact_path": self.recipe[f"{name}_calibration"]})
            self.assertEqual(collection.resolve_calibration(folder, session, name),
                             self.artifacts / self.recipe[f"{name}_calibration"])
        self.assertFalse((folder / "eyes_capture.yaml").exists())
        self.assertFalse((shared / "eyes_capture.yaml").exists())
        eyes = collection.read_yaml(self.artifacts / collection.EYES_WORKFLOW)
        self.assertEqual(eyes["scenes"][0]["ros_action"], "pose")
        self.assertEqual(eyes["scenes"][0]["camera"][0]["parameters"],
                         ["artifacts/" + self.recipe["camera_calibration"]])
        self.assertNotIn("robot", eyes)
        self.assertNotIn("workflow_calibration", session)
        by_id = {entry["id"]: entry for entry in session["captures"]}
        for stage in session["stages"]:
            self.assertEqual(stage["confirmation_path"], f"stage_confirmations/{stage['id']}.json")
            self.assertTrue(stage["instructions"])
            self.assertTrue(all(by_id[key]["stage_id"] == stage["id"] for key in stage["capture_ids"]))
        for exposure in self.recipe["exposure_ms"]:
            local = folder / "configs" / f"{collection.exposure_name(exposure)}.yaml"
            installed = self.artifacts / "config/parts/data_saving/bin_model/sessions" / folder.name / local.name
            self.assertEqual(installed.read_bytes(), local.read_bytes())
        resistor = [stage for stage in session["stages"] if "resistor" in stage["id"]]
        self.assertEqual(len(resistor), 18)
        for stage in resistor:
            self.assertEqual(sorted(value["part_count"] for value in stage["bins"].values()), [1, 1])
            loaded = next(value for value in stage["bins"].values() if value["label"] == "loaded")
            self.assertEqual(loaded["part_type"], "resistor")
            self.assertEqual(len(loaded["target_region_fraction"]), 4)
        self.assertIsNone(session["acceptance"]["minimum_part_thickness_mm"])

    def test_loaded_arrangements_have_independent_confirmations_and_exposure_groups(self):
        folder = self.prepare(mode="guided")
        session = self.read_session(folder)
        recipe = collection.read_yaml(folder / "recipe.yaml")
        old_recipe = copy.deepcopy(recipe)
        old_recipe["guided"].pop("loaded_arrangements")
        self.assertEqual(collection.guided_stages(recipe)[:2], collection.guided_stages(old_recipe)[:2])
        stages = session["stages"][2:]
        self.assertEqual(len(stages), 24)
        self.assertEqual(len({stage["confirmation_path"] for stage in stages}), 24)
        groups = set()
        by_id = {capture["id"]: capture for capture in session["captures"]}
        for stage in stages:
            self.assertEqual(stage["rounds"], 1)
            index = stage["arrangement_index"]
            self.assertEqual(stage["arrangement_count"], 3)
            self.assertTrue(stage["id"].endswith(f"_arrangement_{index:02d}"))
            self.assertIn(f"arrangement {index}/3", stage["title"])
            captures = [by_id[capture_id] for capture_id in stage["capture_ids"]]
            self.assertEqual([capture["exposure_ms"] for capture in captures], recipe["exposure_ms"])
            group_ids = {capture["group_id"] for capture in captures}
            self.assertEqual(len(group_ids), 1)
            self.assertTrue(groups.isdisjoint(group_ids))
            groups.update(group_ids)
            if "resistor" in stage["id"]:
                if index > 1:
                    self.assertIn("Rotate or reposition", stage["instructions"])
                    self.assertIn("same highlighted", stage["instructions"])
                self.assertEqual(sum(bin["part_count"] for bin in stage["bins"].values()), len(stage["bins"]))
            else:
                self.assertIn(stage["fill"], ("normal", "full"))
                if index > 1:
                    self.assertIn("Rearrange the contents", stage["instructions"])
                    self.assertIn(f"same {stage['fill']} fill level", stage["instructions"])
        first, second = stages[:2]
        self.complete(folder, set(first["capture_ids"]))
        self.assertTrue((folder / first["confirmation_path"]).is_file())
        self.assertFalse((folder / second["confirmation_path"]).exists())
        for capture_id in first["capture_ids"]:
            collection.verify_capture(folder, session, by_id[capture_id])
        with self.assertRaisesRegex(ValueError, "Missing operator confirmation"):
            collection.verify_capture(folder, session, by_id[second["capture_ids"][0]])

    def test_new_preparation_normalizes_old_rounds_to_arrangements_without_mutating_input(self):
        self.recipe["guided"].pop("loaded_arrangements")
        self.recipe["guided"].pop("wall_sides")
        self.recipe["guided"].pop("resistor_capture_mode")
        self.recipe["loaded_test_rounds"] = 2
        session = self.read_session(self.prepare(mode="guided"))
        self.assertNotIn("loaded_arrangements", self.recipe["guided"])
        self.assertNotIn("wall_sides", self.recipe["guided"])
        self.assertNotIn("resistor_capture_mode", self.recipe["guided"])
        self.assertEqual(len(session["stages"]), 18)
        self.assertEqual(len(session["captures"]), 130)
        self.assertTrue(all(stage["arrangement_count"] == 2 and stage["rounds"] == 1
                            for stage in session["stages"][2:]))
        for value in (0, -1, True, 1.5):
            self.recipe["guided"]["loaded_arrangements"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "loaded_arrangements"):
                self.prepare(mode="guided")

    def test_simultaneous_resistors_cover_each_bin_with_one_ready_per_arrangement(self):
        self.configure_scene(4, ["left_tray", "middle_tray", "right_tray"])
        folder = self.prepare(mode="guided")
        session = self.read_session(folder)
        recipe = collection.read_yaml(folder / "recipe.yaml")
        self.assertEqual(recipe["guided"]["resistor_capture_mode"], "simultaneous")
        by_id = {capture["id"]: capture for capture in session["captures"]}
        resistor_stages = [stage for stage in session["stages"] if stage["id"].startswith("all_resistor_")]
        self.assertEqual(len(resistor_stages), 18)
        self.assertEqual({stage["id"].rsplit("_arrangement_", 1)[0] for stage in resistor_stages},
                         {"all_resistor_center", "all_resistor_wall", "all_resistor_wall_right",
                          "all_resistor_wall_bottom", "all_resistor_wall_top", "all_resistor_corner"})
        for stage in resistor_stages:
            self.assertEqual(stage["rounds"], 1)
            self.assertEqual(len(stage["capture_ids"]), 5)
            self.assertIn("exactly 1 resistor in EACH selected bin", stage["instructions"])
            self.assertNotIn("empty", stage["instructions"])
            if stage["arrangement_index"] > 1:
                self.assertIn("Rotate or reposition the resistor in EACH selected bin", stage["instructions"])
            targets = list(stage["bins"].values())
            self.assertTrue(all(item == targets[0] for item in targets))
            self.assertEqual(targets[0]["part_count"], 1)
            self.assertEqual(targets[0]["part_type"], "resistor")
            self.assertEqual(targets[0]["label"], "loaded")
            for capture_id in stage["capture_ids"]:
                self.assertEqual(set(by_id[capture_id]["bins"].values()), {"loaded"})
        sequential = copy.deepcopy(recipe)
        sequential["guided"].pop("resistor_capture_mode")
        preserved = {stage["id"]: stage for stage in collection.guided_stages(sequential)
                     if "resistor" not in stage["id"]}
        actual = {stage["id"]: stage for stage in collection.guided_stages(recipe)
                  if "resistor" not in stage["id"]}
        self.assertEqual(actual, preserved)

    def test_explicit_sequential_mode_remains_available_and_unknown_modes_fail(self):
        self.recipe["exposure_ms"] = [4]
        self.recipe["guided"]["resistor_capture_mode"] = "sequential"
        folder = self.prepare(mode="guided")
        session = self.read_session(folder)
        self.assertEqual(len(session["stages"]), 44)
        self.assertEqual(len(session["captures"]), 52)
        for stage in session["stages"]:
            if "resistor" in stage["id"]:
                self.assertEqual(sum(item["part_count"] for item in stage["bins"].values()), 1)
        # An older all-wall session has exactly this plan without a mode field.
        recipe = collection.read_yaml(folder / "recipe.yaml")
        recipe["guided"].pop("resistor_capture_mode")
        (folder / "recipe.yaml").write_text(collection.dump_yaml(recipe))
        self.complete(folder)
        self.assertEqual(collection.verify_session(folder), 52)
        for value in (None, "all", True, []):
            self.recipe["guided"]["resistor_capture_mode"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "resistor_capture_mode"):
                self.prepare(mode="guided")

    def test_all_walls_have_distinct_regions_and_preserve_existing_left_stages(self):
        self.recipe["guided"]["resistor_capture_mode"] = "sequential"
        folder = self.prepare(mode="guided")
        session = self.read_session(folder)
        recipe = collection.read_yaml(folder / "recipe.yaml")
        old_recipe = copy.deepcopy(recipe)
        old_recipe["guided"].pop("wall_sides")
        old_stages = {stage["id"]: stage for stage in collection.guided_stages(old_recipe)}
        new_stages = {stage["id"]: stage for stage in collection.guided_stages(recipe)}
        for stage_id, old in old_stages.items():
            self.assertEqual(new_stages[stage_id], old)
        regions = {"left": [0.02, 0.35, 0.20, 0.65], "right": [0.80, 0.35, 0.98, 0.65],
                   "bottom": [0.35, 0.02, 0.65, 0.20], "top": [0.35, 0.80, 0.65, 0.98]}
        for bin_id in recipe["bin_ids"]:
            wall_stages = [stage for stage in session["stages"] if stage["id"].startswith(f"{bin_id}_resistor_wall")]
            self.assertEqual(len(wall_stages), 12)
            self.assertEqual([stage["bins"][bin_id].get("wall_side", "left") for stage in wall_stages],
                             [side for side in regions for _ in range(3)])
            for stage in wall_stages:
                target = stage["bins"][bin_id]
                side = target.get("wall_side", "left")
                suffix = "wall" if side == "left" else f"wall_{side}"
                self.assertEqual(stage["id"], f"{bin_id}_resistor_{suffix}_arrangement_{stage['arrangement_index']:02d}")
                self.assertEqual(target["placement"], "wall")
                self.assertEqual(target["target_region_fraction"], regions[side])
                self.assertIn(side, target["placement_instruction"])
                self.assertEqual(target["part_count"], 1)
                self.assertTrue(all(item["label"] == "empty" for name, item in stage["bins"].items() if name != bin_id))
                if side == "left":
                    self.assertNotIn("wall_side", target)

    def test_wall_sides_are_configurable_and_invalid_lists_are_rejected(self):
        self.recipe["guided"]["wall_sides"] = ["right"]
        session = self.read_session(self.prepare(mode="guided"))
        self.assertEqual(len(session["stages"]), 17)
        self.assertEqual(len(session["captures"]), 125)
        for stage in session["stages"]:
            if "_resistor_wall" in stage["id"]:
                selected = next(item for item in stage["bins"].values() if item["label"] == "loaded")
                self.assertEqual(selected["wall_side"], "right")
        for sides in ([], ["left", "left"], ["front"], "left", [False]):
            self.recipe["guided"]["wall_sides"] = sides
            with self.subTest(sides=sides), self.assertRaisesRegex(ValueError, "wall_sides"):
                self.prepare(mode="guided")

    def test_session_inputs_are_readable_by_camera_process_with_another_uid(self):
        folder = self.prepare(mode="guided")
        shared = self.artifacts / "data_saving/bin_model/sessions" / folder.name
        parts = self.artifacts / "config/parts/data_saving/bin_model/sessions" / folder.name
        for root in (folder, shared, parts):
            for path in root.rglob("*"):
                if path.is_file():
                    self.assertEqual(path.stat().st_mode & 0o777, 0o644, str(path))
        # Replacing a previously written public file must keep its sharing mode.
        local = folder / "configs/depth_04ms.yaml"
        collection.write_atomic(local, local.read_text())
        self.assertEqual(local.stat().st_mode & 0o777, 0o644)

    def test_guided_verification_requires_matching_operator_confirmations(self):
        self.recipe["exposure_ms"] = [4]
        folder = self.prepare(mode="guided")
        self.complete(folder)
        self.assertEqual(collection.verify_session(folder), 34)
        session = self.read_session(folder)
        confirmation_path = folder / session["stages"][0]["confirmation_path"]
        confirmation = json.loads(confirmation_path.read_text())
        confirmation["session_id"] = "another-session"
        confirmation_path.write_text(json.dumps(confirmation))
        with self.assertRaisesRegex(ValueError, "operator confirmation"):
            collection.verify_session(folder)

    def test_historical_display_wording_verifies_without_rewriting_evidence(self):
        self.recipe["exposure_ms"] = [4]
        folder = self.prepare(mode="guided")
        session = self.read_session(folder)
        stage = next(stage for stage in session["stages"] if "resistor_corner" in stage["id"])
        stage["title"] = "All selected bins: one resistor each, corner — arrangement 1/3"
        stage["instructions"] = ("Place exactly 1 resistor in EACH selected bin on the floor beside "
                                 "the low-X/low-Y corner shown in the bin diagram. Press Ready.")
        for item in stage["bins"].values():
            item["placement_instruction"] = "on the floor beside the low-X/low-Y corner shown in the bin diagram"
        (folder / "session.json").write_text(json.dumps(session))
        self.complete(folder)
        before = {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}
        self.assertEqual(collection.verify_session(folder), 34)
        self.assertEqual(before, {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()})

    def test_display_compatibility_keeps_stage_and_placement_invariants(self):
        self.recipe["exposure_ms"] = [4]
        folder = self.prepare(mode="guided")
        original = self.read_session(folder)
        index = next(index for index, stage in enumerate(original["stages"])
                     if "resistor_wall_right" in stage["id"])
        original["stages"][index]["instructions"] = "Historical wall placement instructions."
        alterations = (
            ("id", "another-stage"), ("split", "train"), ("rounds", 2),
            ("arrangement_index", 2), ("arrangement_count", 4),
            ("capture_ids", []), ("confirmation_path", "stage_confirmations/another-stage.json"),
            ("unexpected_metadata", True),
        )
        bin_alterations = (
            ("label", "empty"), ("part_count", 2), ("part_type", "capacitor"),
            ("placement", "center"), ("wall_side", "left"),
            ("target_region_fraction", [0., 0., 1., 1.]),
            ("unexpected_metadata", True),
        )
        for target, changes in (("stage", alterations), ("bin", bin_alterations)):
            for key, value in changes:
                with self.subTest(target=target, key=key):
                    session = copy.deepcopy(original)
                    item = session["stages"][index]
                    if target == "bin":
                        item = next(iter(item["bins"].values()))
                    item[key] = value
                    (folder / "session.json").write_text(json.dumps(session))
                    with self.assertRaisesRegex(ValueError, "metadata differs from recipe"):
                        collection.verify_session(folder)
        # Presentation remains required text, rather than a bypass for malformed stages.
        for key in ("title", "instructions"):
            session = copy.deepcopy(original)
            session["stages"][index][key] = None
            (folder / "session.json").write_text(json.dumps(session))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "nonempty text"):
                collection.verify_session(folder)

    def test_quick_check_standalone_is_empty_test_only_without_training(self):
        folder = self.prepare(mode="check")
        session = self.read_session(folder)
        self.assertTrue(folder.name.startswith("bin-check-"))
        self.assertEqual(len(session["captures"]), 15)
        self.assertEqual({capture["split"] for capture in session["captures"]}, {"test"})
        self.assertTrue(all(set(capture["bins"].values()) == {"empty"} for capture in session["captures"]))
        self.assertEqual(len(session["stages"]), 1)
        self.assertEqual(session["stages"][0]["id"], "empty_check")
        self.assertFalse((folder / "collect.xml").exists())
        self.complete(folder)
        self.assertEqual(collection.verify_session(folder), 15)

    def test_preparation_preserves_existing_workflows_and_verifies_one_capture(self):
        paths = [self.artifacts / "sequence/operator_workflow.xml",
                 self.artifacts / "workflow_configs/operator_workflow.yaml"]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Existing operator workflow\n")
        original = {path: path.read_bytes() for path in paths}
        folder = self.prepare(mode="guided")
        session = self.read_session(folder)
        first = session["captures"][0]
        self.complete(folder, capture_ids={first["id"]})
        self.assertEqual(collection.verify_capture(folder, session, first)["status"], "complete")
        self.assertEqual(original, {path: path.read_bytes() for path in paths})
        self.assertFalse((folder / session["stages"][1]["confirmation_path"]).exists())
        with self.assertRaises(FileNotFoundError):
            collection.verify_session(folder)
        (folder / session["stages"][0]["confirmation_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "operator confirmation"):
            collection.verify_capture(folder, session, first)
        unknown = dict(first, id="not-planned")
        with self.assertRaisesRegex(ValueError, "not in this session"):
            collection.verify_capture(folder, session, unknown)

    def test_custom_exposures(self):
        self.recipe["exposure_ms"] = [4, 6, 10.5, 20]
        self.assertEqual(len(self.read_session(self.prepare())["captures"]), 40)

    def test_new_sessions_pair_light_exposures_in_configs_plan_and_capture_metadata(self):
        self.recipe["exposure_ms"] = [4, 6, 8, 20]
        self.recipe["spotlight_exposure_ms"] = 10.0
        folder = self.prepare()
        session = self.read_session(folder)
        self.assertEqual(self.recipe["spotlight_exposure_ms"], 10.0)
        self.assertNotIn("spotlight_exposure_ms", collection.read_yaml(folder / "recipe.yaml"))
        self.assertTrue(all(entry["spotlight_exposure_ms"] == entry["exposure_ms"]
                            for entry in session["captures"]))
        for exposure in self.recipe["exposure_ms"]:
            local = folder / "configs" / f"{collection.exposure_name(exposure)}.yaml"
            shared = self.artifacts / "config/parts/data_saving/bin_model/sessions" / folder.name / local.name
            self.assertEqual(local.read_bytes(), shared.read_bytes())
            image = collection.read_yaml(local)["image"]
            self.assertEqual(image["exposure_time_spot_light"], [exposure, exposure])
            self.assertEqual(image["exposure_time_structure_light"], [exposure, exposure])
        self.complete(folder)
        self.assertEqual(collection.verify_session(folder), 40)
        for capture in session["captures"]:
            metadata = collection.read_yaml(folder / capture["path"] / "requested_settings.yaml")
            self.assertEqual(metadata["spotlight_exposure_ms"], [capture["exposure_ms"]] * 2)
            self.assertEqual(metadata["structured_light_exposure_ms"], metadata["spotlight_exposure_ms"])

    def configure_scene(self, scene_id, names, frame_id="fixture_frame"):
        config_path = self.artifacts / self.recipe["bin_config"]
        config = collection.read_yaml(config_path)
        template = next(iter(config["bins"].values()))
        config.update(scene_id=scene_id, frame_id=frame_id,
                      bins={name: copy.deepcopy(template) for name in names})
        config_path.write_text(collection.dump_yaml(config))
        robot_path = self.artifacts / self.recipe["robot_calibration"]
        robot = collection.read_yaml(robot_path)
        robot["name"] = frame_id
        robot["hand_eye"] = [dict(robot["hand_eye"][0], scene_id=scene_id)]
        robot_path.write_text(collection.dump_yaml(robot))
        self.recipe.update(scene_id=scene_id, frame_id=frame_id)
        workflow_path = self.artifacts / collection.EYES_WORKFLOW
        workflow = collection.read_yaml(workflow_path)
        if not any(scene["scene_id"] == scene_id for scene in workflow["scenes"]):
            workflow["scenes"].append({"scene_id": scene_id, "name": "bin_model_capture", "ros_action": "pose",
                "camera": [{"cam_id": 0, "parameters": ["artifacts/" + self.recipe["camera_calibration"]]}]})
        workflow_path.write_text(collection.dump_yaml(workflow))

    def test_one_bin_scene_one_and_three_bins_scene_four(self):
        for scene_id, names in ((1, ["tray"]), (4, ["left", "middle", "right"])):
            with self.subTest(scene_id=scene_id):
                self.configure_scene(scene_id, names)
                folder = self.prepare(mode="guided")
                session = self.read_session(folder)
                self.assertEqual(session["bin_ids"], names)
                self.assertEqual(session["scene_id"], scene_id)
                self.assertEqual(session["frame_id"], "fixture_frame")
                self.assertEqual(len(session["captures"]), 170)
                self.assertEqual(len(session["stages"]), 26)
                self.assertEqual({stage["id"] for stage in session["stages"] if stage.get("fill") in ("normal", "full")},
                                 {f"all_{kind}_arrangement_{index:02d}"
                                  for kind in ("normal", "full") for index in range(1, 4)})
                for stage in session["stages"]:
                    self.assertEqual(list(stage["bins"]), names)
                    if "resistor" in stage["id"]:
                        self.assertEqual(sum(entry["part_count"] for entry in stage["bins"].values()), len(names))
                    self.assertNotIn("both", stage["instructions"].lower())
                    self.assertNotIn("Bin A", stage["instructions"])
                eyes = collection.read_yaml(self.artifacts / session["eyes_config"]["artifact_path"])
                self.assertEqual(sum(scene["scene_id"] == scene_id for scene in eyes["scenes"]), 1)
                capture = session["captures"][0]
                self.complete(folder, {capture["id"]})
                self.assertEqual(collection.verify_capture(folder, session, capture)["scene_id"], scene_id)

    def test_selected_bins_and_custom_runtime_artifacts(self):
        self.configure_scene(4, ["left", "middle", "right"])
        self.recipe["bin_ids"] = ["right", "left"]
        folder = self.prepare(mode="guided", runtime_artifacts_root=Path("/opt/workcell/artifacts"))
        session = self.read_session(folder)
        self.assertEqual(session["bin_ids"], ["right", "left"])
        self.assertEqual(list(collection.read_yaml(folder / "bin_config.yaml")["bins"]), ["right", "left"])
        self.assertEqual(len(session["captures"]), 170)
        config = collection.read_yaml(folder / "configs/depth_04ms.yaml")
        self.assertEqual(config["tasks"][0]["steps"][0]["parameters"]["camera_calibration_path"],
                         "/opt/workcell/artifacts/" + self.recipe["camera_calibration"])

    def test_unsupported_stereo_pair_is_rejected_before_writes(self):
        for denali_id in (1, -1, True):
            self.recipe["denali_id"] = denali_id
            with self.subTest(denali_id=denali_id), self.assertRaisesRegex(ValueError, "supports Denali 0"):
                self.prepare()
        self.assertFalse((self.root / "sessions").exists())

    def test_shared_eyes_workflow_is_preserved_and_must_match_selected_calibration(self):
        path = self.artifacts / collection.EYES_WORKFLOW
        original = path.read_bytes()
        folder = self.prepare(mode="guided")
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((folder / "eyes_capture.yaml").exists())
        workflow = collection.read_yaml(path)
        camera = workflow["scenes"][0]["camera"][0]
        camera["parameters"] = ["/root/artifacts/" + self.recipe["camera_calibration"]]
        path.write_text(collection.dump_yaml(workflow))
        with self.assertRaisesRegex(ValueError, "relative to EBOTS_ROOT"):
            self.prepare()
        other = "calibration/deployment/camera/other.yaml"
        shutil.copyfile(self.artifacts / self.recipe["camera_calibration"], self.artifacts / other)
        camera["parameters"] = ["artifacts/" + other]
        path.write_text(collection.dump_yaml(workflow))
        with self.assertRaisesRegex(ValueError, "differs from"):
            self.prepare()
        camera["parameters"] = ["artifacts/" + self.recipe["camera_calibration"]]
        workflow["scenes"][0]["scene_id"] = 1
        path.write_text(collection.dump_yaml(workflow))
        with self.assertRaisesRegex(ValueError, "exactly one entry for scene 0"):
            self.prepare()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing Eyes workflow"):
            self.prepare()

    def test_mismatched_layout_scene_or_frame_and_invalid_bin_ids_are_rejected(self):
        for key, value in (("scene_id", 1), ("frame_id", "A1")):
            config_path = self.artifacts / self.recipe["bin_config"]
            config = collection.read_yaml(config_path)
            config[key] = value
            config_path.write_text(collection.dump_yaml(config))
            with self.assertRaisesRegex(ValueError, "bin_config"):
                self.prepare()
            config[key] = self.recipe[key]
            config_path.write_text(collection.dump_yaml(config))
        for names in ([], ["missing"], ["bin_a", "bin_a"], ["../escape"], ["bin a"]):
            self.recipe["bin_ids"] = names
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.prepare()
        self.assertFalse((self.root / "sessions").exists())

    def test_deployment_reference_cannot_leave_its_camera_or_robot_directory(self):
        folder = self.prepare()
        session = self.read_session(folder)
        for relative in ("../outside.yaml", self.recipe["robot_calibration"], "data_saving/bin_model/recipe.yaml"):
            session["calibration"]["camera"] = {"artifact_path": relative}
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                collection.resolve_calibration(folder, session, "camera")
        link = self.artifacts / "calibration/deployment/camera/escaped.yaml"
        link.symlink_to(self.artifacts / self.recipe["robot_calibration"])
        session["calibration"]["camera"] = {"artifact_path": str(link.relative_to(self.artifacts))}
        with self.assertRaisesRegex(ValueError, "camera calibration must"):
            collection.resolve_calibration(folder, session, "camera")

    def test_old_saved_guided_session_keeps_legacy_stages_and_calibration_snapshots(self):
        self.recipe["exposure_ms"] = [4]
        folder = self.prepare(mode="guided")
        session = self.read_session(folder)
        recipe = collection.read_yaml(folder / "recipe.yaml")
        recipe.pop("bin_ids")
        recipe["guided"].pop("loaded_arrangements")
        recipe["guided"].pop("wall_sides")
        recipe["guided"].pop("resistor_capture_mode")
        recipe["spotlight_exposure_ms"] = 10.0
        (folder / "recipe.yaml").write_text(collection.dump_yaml(recipe))
        session.pop("bin_ids")
        session.pop("artifacts_root")
        (folder / "calibration").mkdir()
        for name in ("camera", "robot"):
            relative = f"calibration/{name}.yaml"
            shutil.copyfile(self.artifacts / self.recipe[f"{name}_calibration"], folder / relative)
            session["calibration"][name] = {"path": relative}
        session["captures"] = collection.capture_plan(recipe, session["session_id"], "guided", None, None)
        session["stages"] = collection.guided_stages(recipe)
        for stage in session["stages"]:
            stage["capture_ids"] = [entry["id"] for entry in session["captures"] if entry["stage_id"] == stage["id"]]
            stage["confirmation_path"] = f"stage_confirmations/{stage['id']}.json"
        self.assertIn("both_normal", [stage["id"] for stage in session["stages"]])
        self.assertTrue(all(capture["exposure_ms"] == 4.0 and capture["spotlight_exposure_ms"] == 10.0
                            for capture in session["captures"]))
        legacy_config = collection.dump_yaml(collection.part_config(4, recipe))
        collection.write_atomic(folder / "configs/depth_04ms.yaml", legacy_config)
        collection.write_atomic(self.artifacts / "config/parts/data_saving/bin_model/sessions" /
                                folder.name / "depth_04ms.yaml", legacy_config)
        (folder / "session.json").write_text(json.dumps(session))
        self.complete(folder)
        self.assertEqual(collection.verify_session(folder), 34)
        first = folder / session["captures"][0]["path"]
        settings_path = first / "requested_settings.yaml"
        settings = collection.read_yaml(settings_path)
        self.assertEqual(settings["spotlight_exposure_ms"], [10.0, 10.0])
        settings["spotlight_exposure_ms"] = [4.0, 4.0]
        settings_path.write_text(collection.dump_yaml(settings))
        self.refresh_size(first, "requested_settings.yaml")
        with self.assertRaisesRegex(ValueError, "Wrong spotlight_exposure_ms"):
            collection.verify_session(folder)

    def test_old_generic_sessions_keep_left_only_walls_and_original_loaded_grouping(self):
        self.recipe["exposure_ms"] = [4]
        for arrangements in (False, True):
            with self.subTest(arrangements=arrangements):
                folder = self.prepare(mode="guided")
                session = self.read_session(folder)
                recipe = collection.read_yaml(folder / "recipe.yaml")
                recipe["guided"].pop("wall_sides")
                recipe["guided"].pop("resistor_capture_mode")
                if not arrangements:
                    recipe["guided"].pop("loaded_arrangements")
                (folder / "recipe.yaml").write_text(collection.dump_yaml(recipe))
                session["captures"] = collection.capture_plan(recipe, session["session_id"], "guided", None, None)
                session["stages"] = collection.guided_stages(recipe)
                for stage in session["stages"]:
                    stage["capture_ids"] = [entry["id"] for entry in session["captures"] if entry["stage_id"] == stage["id"]]
                    stage["confirmation_path"] = f"stage_confirmations/{stage['id']}.json"
                self.assertEqual(len(session["stages"]), 26 if arrangements else 10)
                loaded = session["stages"][2:]
                self.assertTrue(all(stage["rounds"] == (1 if arrangements else 3) for stage in loaded))
                self.assertEqual(len({capture["group_id"] for capture in session["captures"]
                                      if "loaded" in capture["bins"].values()}), 24 if arrangements else 8)
                self.assertFalse(any("wall_right" in stage["id"] or "wall_bottom" in stage["id"] or "wall_top" in stage["id"]
                                     for stage in loaded))
                (folder / "session.json").write_text(json.dumps(session))
                self.complete(folder)
                snapshots = {name: (folder / name).read_bytes() for name in ("session.json", "recipe.yaml")}
                self.assertEqual(collection.verify_session(folder), 34)
                self.assertEqual(snapshots, {name: (folder / name).read_bytes() for name in snapshots})

    def test_invalid_exposure_or_insufficient_training_fails_before_writes(self):
        for exposures in ([3, 4], [4, 21], [4, 4], [8, 4], [float("nan")]):
            with self.subTest(exposures=exposures):
                self.recipe["exposure_ms"] = exposures
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse((self.root / "sessions").exists())
        self.recipe["exposure_ms"] = [4]
        self.recipe["empty_train_rounds"] = 1
        with self.assertRaisesRegex(ValueError, "minimum capture support"):
            self.prepare()

    def test_runtime_root_must_be_absolute_without_traversal(self):
        for runtime in ("relative/bin_model", "/", "/root/../outside/bin_model"):
            with self.subTest(runtime=runtime), self.assertRaises(ValueError):
                collection.prepare_session(self.artifacts, self.root / "sessions", Path(runtime), self.recipe)
        self.assertFalse((self.root / "sessions").exists())

    def test_loaded_validation_never_trains_and_keeps_arrangement_group(self):
        session = self.read_session(self.prepare(mode="loaded", labels={"bin_a": "loaded", "bin_b": "empty"},
                                                group_id="thin_parts_floor_01"))
        self.assertEqual(len(session["captures"]), 15)
        self.assertEqual({c["split"] for c in session["captures"]}, {"test"})
        self.assertEqual({c["group_id"] for c in session["captures"]}, {"thin_parts_floor_01"})
        with self.assertRaises(ValueError):
            self.prepare(mode="loaded", labels={"bin_a": "loaded"}, group_id="arrangement")
        with self.assertRaises(ValueError):
            self.prepare(mode="loaded", labels={"bin_a": "loaded", "bin_b": "empty"})

    def test_missing_capture_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            collection.verify_session(self.prepare())

    def test_complete_float_exr_session_needs_no_hash_fields(self):
        self.recipe["exposure_ms"] = [4]
        folder = self.prepare()
        self.complete(folder)
        self.assertEqual(collection.verify_session(folder), 10)

    def test_wrong_exposure_and_calibration_are_rejected_without_hashes(self):
        self.recipe["exposure_ms"] = [4]
        folder = self.prepare()
        first = self.complete(folder)
        path = first / "requested_settings.yaml"
        original = path.read_text()
        path.write_text(original.replace("- 4.0", "- 8.0"))
        self.refresh_size(first, "requested_settings.yaml")
        with self.assertRaisesRegex(ValueError, "structured_light_exposure_ms"):
            collection.verify_session(folder)
        path.write_text(original)
        self.refresh_size(first, "requested_settings.yaml")
        robot_path = first / "calibration/robot.yaml"
        robot = collection.read_yaml(robot_path)
        robot["name"] = "A1"
        robot_path.write_text(collection.dump_yaml(robot))
        self.refresh_size(first, "calibration/robot.yaml")
        with self.assertRaisesRegex(ValueError, "calibration differs"):
            collection.verify_session(folder)

    def test_in_progress_and_wrong_image_shape_are_rejected(self):
        self.recipe["exposure_ms"] = [4]
        folder = self.prepare()
        first = self.complete(folder)
        lock = first / ".capture_in_progress"
        lock.mkdir()
        with self.assertRaisesRegex(ValueError, "in progress"):
            collection.verify_session(folder)
        lock.rmdir()
        write_exr(first / "depth/Frame0.exr", np.ones((8, 9), np.float32))
        self.refresh_size(first, "depth/Frame0.exr")
        with self.assertRaisesRegex(ValueError, "shape differs"):
            collection.verify_session(folder)

    def test_modified_split_and_path_escape_are_rejected(self):
        folder = self.prepare()
        session = self.read_session(folder)
        session["captures"][0]["split"] = "test"
        (folder / "session.json").write_text(json.dumps(session))
        with self.assertRaisesRegex(ValueError, "plan differs"):
            collection.verify_session(folder)
        with self.assertRaisesRegex(ValueError, "Unsafe relative path"):
            collection.safe_path(folder, "../outside")


if __name__ == "__main__":
    unittest.main()
