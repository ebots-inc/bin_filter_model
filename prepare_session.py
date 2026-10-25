#!/usr/bin/env python3
"""Prepare and verify bin-model capture sessions; never connect to hardware."""

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import yaml


HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"
DEFAULT_RECIPE = CONFIGS / "recipe.yaml"
ARTIFACTS = HERE.parent / "ebots_system_artifacts/artifacts_cell_4"
LEGACY_BINS = ("bin_a", "bin_b")
EYES_WORKFLOW = "workflow_configs/bin_calibration.yaml"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def safe_path(root, relative):
    require(isinstance(relative, str) and relative, "Expected a relative file path")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe relative path: {relative}")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()), f"Path leaves session: {relative}")
    return resolved


def absolute_path(value):
    path = Path(value)
    require(path.is_absolute() and ".." not in path.parts,
            f"Expected an absolute non-root path without '..': {value}")
    require(path != Path(path.anchor), "A filesystem root is not a collection directory")
    return path


def write_atomic(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        stream.write(contents)
        # Session recipes and metadata are shared with Eyes under another UID.
        # NamedTemporaryFile defaults to 0600; rename retains that unreadable mode.
        os.fchmod(stream.fileno(), 0o644)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def dump_yaml(value):
    return yaml.safe_dump(value, sort_keys=False)


def exposure_name(value):
    text = f"{value:g}".replace(".", "p")
    if value == int(value):
        text = f"{int(value):02d}"
    return f"depth_{text}ms"


def spotlight_exposure(recipe, exposure):
    """New sessions pair exposures; old saved recipes retain their recorded setting."""
    return float(recipe.get("spotlight_exposure_ms", exposure))


def validate_bin_ids(bin_ids):
    require(isinstance(bin_ids, list) and bin_ids and all(
        isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
        for name in bin_ids), "bin_ids must be a nonempty list of names using letters, digits, '.', '_' or '-'")
    require(len(set(bin_ids)) == len(bin_ids), "bin_ids must be unique")


def deployment_calibration(artifacts, relative, name):
    require(name in ("camera", "robot"), "Unknown calibration type")
    artifacts = Path(artifacts).resolve()
    path = safe_path(artifacts, relative)
    deployment = artifacts / "calibration/deployment" / name
    require(path.is_relative_to(deployment.resolve()) and
            deployment.resolve().is_relative_to(artifacts),
            f"{name} calibration must be in artifacts/calibration/deployment/{name}/")
    require(path.is_file() and path.stat().st_size, f"Missing {name} calibration: {path}")
    return path


def resolve_calibration(folder, session, name, artifacts=None):
    """Resolve deployment references while retaining old saved-session support."""
    entry = session["calibration"][name]
    if "artifact_path" in entry:
        artifact_root = artifacts if artifacts is not None else session.get("artifacts_root")
        require(artifact_root is not None, "Deployment calibration requires artifacts_root")
        return deployment_calibration(absolute_path(artifact_root), entry["artifact_path"], name)
    path = safe_path(Path(folder), entry["path"])
    require(path.is_file() and path.stat().st_size, f"Missing session calibration: {path}")
    return path


def validate_eyes_workflow(artifacts, recipe):
    """Check the shared Eyes workflow without generating or modifying it."""
    workflow_path = safe_path(Path(artifacts), EYES_WORKFLOW)
    require(workflow_path.is_file(), f"Missing Eyes workflow: {workflow_path}")
    workflow = read_yaml(workflow_path)
    require(isinstance(workflow, dict) and isinstance(workflow.get("scenes"), list),
            f"{EYES_WORKFLOW} must define a scenes list")
    scenes = [scene for scene in workflow["scenes"] if isinstance(scene, dict) and
              type(scene.get("scene_id")) is int and scene["scene_id"] == recipe["scene_id"]]
    require(len(scenes) == 1, f"{EYES_WORKFLOW} must contain exactly one entry for scene {recipe['scene_id']}")
    camera_list = scenes[0].get("camera")
    require(isinstance(camera_list, list), f"Eyes workflow scene {recipe['scene_id']} needs a camera list")
    cameras = [camera for camera in camera_list if isinstance(camera, dict) and
               type(camera.get("cam_id")) is int and camera["cam_id"] == recipe["denali_id"]]
    require(len(cameras) == 1, f"Eyes workflow scene {recipe['scene_id']} needs exactly one Denali 0 camera")
    parameters = cameras[0].get("parameters")
    require(isinstance(parameters, list) and parameters and isinstance(parameters[0], str),
            "Eyes workflow camera needs a deployment calibration parameter")
    parameter = Path(parameters[0])
    require(not parameter.is_absolute() and ".." not in parameter.parts and
            parameter.parts and parameter.parts[0] == "artifacts",
            "Eyes workflow calibration must use artifacts/calibration/deployment/camera/... relative to EBOTS_ROOT")
    configured = deployment_calibration(artifacts, str(parameter.relative_to("artifacts")), "camera")
    selected = deployment_calibration(artifacts, recipe["camera_calibration"], "camera")
    require(configured == selected,
            f"Selected camera calibration differs from {EYES_WORKFLOW} for scene {recipe['scene_id']}")
    return workflow_path


def resolve_bin_config(artifacts, value):
    """Resolve local fitting settings; retain legacy artifact-relative recipes."""
    require(isinstance(value, str) and value, "bin_config must name a settings file")
    path = Path(value)
    if path.is_absolute():
        require(".." not in path.parts, "bin_config must not contain '..'")
        return path.resolve()
    if path.parts and path.parts[0] == "configs":
        return safe_path(HERE, value)
    return safe_path(Path(artifacts), value)


def validate_recipe(recipe, artifacts):
    require(isinstance(recipe, dict), "Recipe must be a mapping")
    require(type(recipe.get("scene_id")) is int and 0 <= recipe["scene_id"] <= 255,
            "scene_id must be an integer from 0 to 255")
    require(type(recipe.get("denali_id")) is int and recipe["denali_id"] == 0,
            "This capture client supports Denali 0 (two sensors) within the selected scene")
    require(isinstance(recipe.get("frame_id"), str) and recipe["frame_id"] and
            recipe.get("length_unit") == "mm", "Bin bounds require a named robot frame in millimetres")
    exposures = recipe.get("exposure_ms")
    require(isinstance(exposures, list) and exposures and all(
        type(value) in (float, int) and math.isfinite(value) and 4 <= value <= 20 for value in exposures),
        "exposure_ms must be a nonempty list of finite values from 4 to 20 ms")
    require(exposures == sorted(set(exposures)), "exposure_ms must be sorted and unique")
    require(len({exposure_name(value) for value in exposures}) == len(exposures), "Exposure names collide")
    if "spotlight_exposure_ms" in recipe:
        require(type(recipe["spotlight_exposure_ms"]) in (int, float) and
                math.isfinite(recipe["spotlight_exposure_ms"]) and recipe["spotlight_exposure_ms"] > 0,
                "Legacy spotlight_exposure_ms must be positive and finite")
    for key in ("empty_train_rounds", "empty_test_rounds", "loaded_test_rounds"):
        require(type(recipe.get(key)) is int and recipe[key] > 0, f"{key} must be a positive integer")
    guided = recipe.get("guided", {})
    require(isinstance(guided, dict), "guided must be a mapping")
    if "loaded_arrangements" in guided:
        require(type(guided["loaded_arrangements"]) is int and guided["loaded_arrangements"] > 0,
                "guided.loaded_arrangements must be a positive integer")
    if "wall_sides" in guided:
        sides = guided["wall_sides"]
        require(isinstance(sides, list) and sides and all(
            isinstance(side, str) and side in ("left", "right", "bottom", "top") for side in sides)
            and len(set(sides)) == len(sides),
            "guided.wall_sides must be a nonempty unique list of left, right, bottom, top")
    require(guided.get("resistor_capture_mode", "sequential") in ("sequential", "simultaneous"),
            "guided.resistor_capture_mode must be sequential or simultaneous")
    for key in ("gain", "black_level"):
        values = recipe.get(key)
        require(isinstance(values, list) and len(values) == 2 and all(
            type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in values),
            f"{key} requires two finite nonnegative camera values")
    require(all(type(value) is int and value <= 65535 for value in recipe["black_level"]),
            "black_level requires two uint16 camera values")
    sources = {name: deployment_calibration(artifacts, recipe[f"{name}_calibration"], name)
               for name in ("camera", "robot")}
    sources["settings"] = resolve_bin_config(artifacts, recipe["bin_config"])
    for name, path in sources.items():
        require(path.is_file() and path.stat().st_size, f"Missing {name} source: {path}")
    robot = read_yaml(sources["robot"])
    require(robot.get("name") == recipe["frame_id"], "Robot calibration frame does not match recipe")
    require(sum(item.get("scene_id") == recipe["scene_id"] for item in robot.get("hand_eye", [])) == 1,
            "Robot calibration must contain exactly one hand-eye transform for this scene")
    camera = read_yaml(sources["camera"])
    require(all(key in camera for key in ("cam1", "cam2", "world_r", "world_t")),
            "Camera calibration must contain both Denali cameras and world transform")
    config = read_yaml(sources["settings"])
    require(isinstance(config, dict) and isinstance(config.get("bins"), dict) and config["bins"],
            "bin_config must contain at least one configured bin")
    configured = list(config["bins"])
    validate_bin_ids(configured)
    selected = recipe.get("bin_ids", configured)
    validate_bin_ids(selected)
    require(set(selected) <= set(configured), "Every selected bin_id must exist in bin_config")
    for key in ("scene_id", "frame_id"):
        require(key not in config or config[key] == recipe[key],
                f"bin_config {key} does not match recipe; choose bounds measured for this scene and frame")
    for name in selected:
        settings = config["bins"][name]
        require(recipe["empty_train_rounds"] >= settings["min_capture_support"],
                f"Training rounds cannot meet {name}'s minimum capture support")
    validate_eyes_workflow(artifacts, recipe)
    return sources


def part_config(exposure, recipe, runtime_artifacts_root=Path("/root/artifacts")):
    return {
        "segmentation": {"class_id": 999},
        "image": {"exposure_time": [2.5], "random_exposure_time": [],
                  "exposure_time_spot_light": [spotlight_exposure(recipe, exposure)] * 2,
                  "exposure_time_structure_light": [float(exposure)] * 2,
                  "gain": list(recipe["gain"]), "black_level": list(recipe["black_level"])},
        "tasks": [{"name": "capture", "node": "perception", "steps": [{
            "step": "capture_data", "cam_id": recipe["denali_id"], "parameters": {
                "denali_id": recipe["denali_id"], "frame_id": recipe["frame_id"],
                "camera_calibration_path": str(Path(runtime_artifacts_root) / recipe["camera_calibration"]),
                "robot_calibration_path": str(Path(runtime_artifacts_root) / recipe["robot_calibration"]),
            }}]}],
    }


def capture_plan(recipe, session_id, mode, labels, group_id):
    bin_ids = recipe.get("bin_ids", LEGACY_BINS)
    if mode in ("guided", "check"):
        captures = []
        for stage in guided_stages(recipe, mode):
            split = stage["split"]
            for round_index in range(1, stage["rounds"] + 1):
                for exposure in recipe["exposure_ms"]:
                    capture_id = f"{stage['id']}_{round_index:02d}_{exposure_name(exposure)}"
                    captures.append({"id": capture_id, "path": f"captures/{capture_id}", "split": split,
                        "bins": {name: item["label"] for name, item in stage["bins"].items()},
                        "exposure_ms": float(exposure), "spotlight_exposure_ms": spotlight_exposure(recipe, exposure),
                        "group_id": f"{session_id}_{stage['id']}", "stage_id": stage["id"],
                        "part_name": f"data_saving/bin_model/sessions/{session_id}/{exposure_name(exposure)}"})
        return captures
    if mode == "empty":
        require(labels is None and group_id is None, "Empty collection labels and groups are generated automatically")
        bins = dict.fromkeys(bin_ids, "empty")
        splits = (("train", recipe["empty_train_rounds"]), ("test", recipe["empty_test_rounds"]))
    else:
        require(labels is not None and set(labels) == set(bin_ids) and set(labels.values()) <= {"empty", "loaded"},
                "Loaded validation requires an explicit empty or loaded label for every selected bin")
        require("loaded" in labels.values(), "Loaded validation requires at least one loaded bin")
        require(group_id and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", group_id),
                "Loaded validation requires a stable --group-id for this unchanged part arrangement")
        bins = labels
        splits = (("test", recipe["loaded_test_rounds"]),)
    captures = []
    for split, rounds in splits:
        for round_index in range(1, rounds + 1):
            for exposure in recipe["exposure_ms"]:
                capture_id = f"{split}_{round_index:02d}_{exposure_name(exposure)}"
                captures.append({"id": capture_id, "path": f"captures/{capture_id}", "split": split,
                                 "bins": dict(bins), "exposure_ms": float(exposure),
                                 "spotlight_exposure_ms": spotlight_exposure(recipe, exposure),
                                 "group_id": group_id or f"{session_id}_{split}_{round_index:02d}",
                                 "part_name": f"data_saving/bin_model/sessions/{session_id}/{exposure_name(exposure)}"})
    return captures


def guided_stages(recipe, mode="guided"):
    """Guide any configured number of bins, retaining old saved stage text."""
    if "bin_ids" not in recipe:
        return legacy_guided_stages(recipe, mode)
    bin_ids = recipe["bin_ids"]
    validate_bin_ids(bin_ids)
    empty = {name: {"label": "empty", "part_count": 0, "placement": "empty",
                    "placement_instruction": "Remove every part and loose item."} for name in bin_ids}
    empty_description = " ".join(f"{name}: 0 parts." for name in bin_ids)
    unchanged = "Leave every bin unchanged during the exposure sweep."
    if mode == "check":
        return [{"id": "empty_check", "title": "Quick check — empty bins", "split": "test",
                 "rounds": recipe["empty_test_rounds"], "bins": empty,
                 "instructions": f"{empty_description} Empty every selected bin, then press Ready to check the installed model. "
                                 "This check collects fresh test data only; it does not train or replace the model."}]
    stages = []
    for stage_id, title, split, rounds in (
            ("empty_training", "Empty bins — training", "train", recipe["empty_train_rounds"]),
            ("empty_validation", "Empty bins — fresh validation", "test", recipe["empty_test_rounds"])):
        stages.append({"id": stage_id, "title": title, "split": split, "rounds": rounds, "bins": empty,
                       "instructions": f"{empty_description} Remove all parts, keep the camera view clear, then press Ready. {unchanged}"})
    simultaneous = recipe["guided"].get("resistor_capture_mode", "sequential") == "simultaneous"
    selected_groups = [bin_ids] if simultaneous else [[name] for name in bin_ids]
    for selected_bins in selected_groups:
        selected = selected_bins[0]
        others = " ".join(f"{name}: 0 parts, completely empty." for name in bin_ids if name not in selected_bins)
        walls = {
            "left": ("left (low-X)", [0.02, 0.35, 0.20, 0.65]),
            "right": ("right (high-X)", [0.80, 0.35, 0.98, 0.65]),
            "bottom": ("bottom (low-Y)", [0.35, 0.02, 0.65, 0.20]),
            "top": ("top (high-Y)", [0.35, 0.80, 0.65, 0.98]),
        }
        placements = [("center", "near the center of the lower bowl", [0.35, 0.35, 0.65, 0.65], None)]
        for side in recipe["guided"].get("wall_sides", ["left"]):
            description, region = walls[side]
            placements.append(("wall", f"on the lower bowl toward the {description} side shown in the bin diagram, off the rim and steep wall", region, side))
        placements.append(("corner", "on the lower bowl toward the bottom-left (low-X/low-Y) diagonal shown in the bin diagram", [0.02, 0.02, 0.20, 0.20], None))
        for placement, description, region, side in placements:
            bins = {name: dict(value) for name, value in empty.items()}
            for name in selected_bins:
                bins[name] = {"label": "loaded", "part_count": 1, "part_type": "resistor", "placement": placement,
                              "placement_instruction": description, "target_region_fraction": list(region)}
            suffix, title = placement, "bottom-left diagonal" if placement == "corner" else placement
            if side not in (None, "left"):
                for name in selected_bins:
                    bins[name]["wall_side"] = side
                suffix, title = f"wall_{side}", f"{side} wall"
            if simultaneous:
                stage_id = f"all_resistor_{suffix}"
                stage_title = f"All selected bins: one resistor each, {title}"
                instructions = (f"Place exactly 1 resistor in EACH selected bin ({', '.join(bin_ids)}) {description}. "
                                "Use the thinnest production resistor to check retention. Press Ready when every bin is prepared; "
                                "leave every resistor unchanged during the exposure sweep.")
            else:
                stage_id, stage_title = f"{selected}_resistor_{suffix}", f"{selected}: one resistor, {title}"
                instructions = f"{selected}: exactly 1 resistor {description}. "
                if others:
                    instructions += others + " "
                instructions += "Use the thinnest production resistor to check retention. Press Ready when placed; do not move it during the exposure sweep."
            stages.append({"id": stage_id, "title": stage_title,
                           "split": "test", "rounds": recipe["loaded_test_rounds"], "bins": bins,
                           "instructions": instructions})
    for kind in ("normal", "full"):
        description = recipe["guided"][f"{kind}_fill_instruction"]
        bins = {name: {"label": "loaded", "part_count": None, "placement": kind,
                       "placement_instruction": description} for name in bin_ids}
        stages.append({"id": f"all_{kind}", "title": f"All selected bins: {kind} fill", "split": "test",
                       "rounds": recipe["loaded_test_rounds"], "bins": bins,
                       "instructions": f"For each selected bin ({', '.join(bin_ids)}): {description} "
                                       f"No exact part count is prescribed. Press Ready when arranged. {unchanged}"})
    arrangements = recipe["guided"].get("loaded_arrangements")
    if arrangements is None:
        return stages
    return loaded_arrangement_stages(stages, arrangements, simultaneous_resistors=simultaneous)


def loaded_arrangement_stages(stages, arrangements, simultaneous_resistors=False):
    """Give each changed loaded arrangement its own confirmation and exposure sweep."""
    result = []
    for stage in stages:
        loaded = [entry for entry in stage["bins"].values() if entry["label"] == "loaded"]
        if not loaded:
            result.append(stage)
            continue
        resistor = any(entry.get("part_type") == "resistor" for entry in loaded)
        placement = loaded[0]["placement"]
        display_placement = "bottom-left diagonal" if placement == "corner" else placement
        for index in range(1, arrangements + 1):
            item = copy.deepcopy(stage)
            item.update(id=f"{stage['id']}_arrangement_{index:02d}",
                        title=f"{stage['title']} — arrangement {index}/{arrangements}",
                        rounds=1, arrangement_index=index, arrangement_count=arrangements)
            introduction = f"Arrangement {index} of {arrangements}. "
            if index == 1:
                introduction += "Prepare the first arrangement as described below. "
            elif resistor and simultaneous_resistors:
                introduction += (f"Rotate or reposition the resistor in EACH selected bin to a different pose within the same highlighted {display_placement} direction, "
                                 "keeping every resistor on the lower bowl, off the rim and steep wall. ")
            elif resistor:
                introduction += (f"Rotate or reposition the resistor to a different pose within the same highlighted {display_placement} direction, "
                                 "keeping it on the lower bowl, off the rim and steep wall. Keep every other selected bin empty. ")
            else:
                introduction += ("Rearrange the contents in every selected bin to create a different layout "
                                 f"while keeping the same {placement} fill level. ")
            item["instructions"] = introduction + stage["instructions"]
            if not resistor:
                item["fill"] = placement
            result.append(item)
    return result


def legacy_guided_stages(recipe, mode="guided"):
    """Each stage is one unchanged arrangement and needs its own ready acknowledgement."""
    stages = []
    empty = {name: {"label": "empty", "part_count": 0, "placement": "empty", "placement_instruction": "Remove every part and loose item."}
             for name in LEGACY_BINS}
    if mode == "check":
        return [{"id": "empty_check", "title": "Quick check — empty bins", "split": "test",
                 "rounds": recipe["empty_test_rounds"], "bins": empty,
                 "instructions": "Bin A: 0 parts. Bin B: 0 parts. Empty both bins, then press Ready to check the installed model. This check collects fresh test data only; it does not train or replace the model."}]
    for stage_id, title, split, rounds in (
            ("empty_training", "Empty bins — training", "train", recipe["empty_train_rounds"]),
            ("empty_validation", "Empty bins — fresh validation", "test", recipe["empty_test_rounds"])):
        stages.append({"id": stage_id, "title": title, "split": split, "rounds": rounds, "bins": empty,
                       "instructions": "Bin A: 0 parts. Bin B: 0 parts. Remove all parts from both bins, keep the camera view clear, then press Ready. Leave both bins unchanged during the exposure sweep."})
    for selected in LEGACY_BINS:
        other = next(name for name in LEGACY_BINS if name != selected)
        for placement, description, region in (
                ("center", "near the center of the lower bowl", [0.35, 0.35, 0.65, 0.65]),
                ("wall", "on the lower bowl toward the left (low-X) side shown in the bin diagram, off the rim and steep wall", [0.02, 0.35, 0.20, 0.65]),
                ("corner", "on the lower bowl toward the bottom-left (low-X/low-Y) diagonal shown in the bin diagram", [0.02, 0.02, 0.20, 0.20])):
            bins = {name: dict(value) for name, value in empty.items()}
            bins[selected] = {"label": "loaded", "part_count": 1, "part_type": "resistor", "placement": placement,
                              "placement_instruction": description,
                              "target_region_fraction": region}
            selected_name, other_name = selected.replace("_", " ").title(), other.replace("_", " ").title()
            stages.append({"id": f"{selected}_resistor_{placement}", "title": f"{selected_name}: one resistor, {placement}",
                "split": "test", "rounds": recipe["loaded_test_rounds"], "bins": bins,
                "instructions": f"{selected_name}: exactly 1 resistor {description}. {other_name}: 0 parts, completely empty. Use the thinnest production resistor to check retention. Press Ready when placed; do not move it during the exposure sweep."})
    for kind in ("normal", "full"):
        description = recipe["guided"][f"{kind}_fill_instruction"]
        bins = {name: {"label": "loaded", "part_count": None, "placement": kind,
                       "placement_instruction": description} for name in LEGACY_BINS}
        stages.append({"id": f"both_{kind}", "title": f"Both bins: {kind} fill", "split": "test",
            "rounds": recipe["loaded_test_rounds"], "bins": bins,
            "instructions": f"Bin A and Bin B: {description} No exact part count is prescribed. Press Ready when both are arranged; leave the contents unchanged during the exposure sweep."})
    return stages


def prepare_session(artifacts, output_root, runtime_root, recipe, mode="guided", labels=None, group_id=None,
                    name=None, runtime_artifacts_root=Path("/root/artifacts")):
    """Prepare settings and deployment calibration references for standalone collection."""
    artifacts, output_root = Path(artifacts).resolve(), absolute_path(output_root).resolve()
    runtime_root = absolute_path(runtime_root)
    runtime_artifacts_root = absolute_path(runtime_artifacts_root)
    recipe = copy.deepcopy(recipe)
    # A new collection always sweeps both lighting modes together. Retain the
    # legacy field only when reading an already saved session, never when preparing.
    recipe.pop("spotlight_exposure_ms", None)
    if mode == "guided":
        require(isinstance(recipe.get("guided"), dict), "Guided setup requires guided instructions")
        recipe["guided"].setdefault("loaded_arrangements", recipe.get("loaded_test_rounds"))
        recipe["guided"].setdefault("wall_sides", ["left", "right", "bottom", "top"])
        recipe["guided"].setdefault("resistor_capture_mode", "simultaneous")
    sources = validate_recipe(recipe, artifacts)
    recipe.setdefault("bin_ids", list(read_yaml(sources["settings"])["bins"]))
    require(mode in ("empty", "loaded", "guided", "check"), "Unknown collection mode")
    if mode in ("guided", "check"):
        require(labels is None and group_id is None, "Guided setup generates arrangement labels and groups")
        guided_stages(recipe, mode)
    capture_plan(recipe, "preview", mode, labels, group_id)
    if name is not None:
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name), "Session name must use 1–80 letters, digits, '_' or '-'")
    prefix = datetime.now(timezone.utc).strftime(("bin-check" if mode == "check" else "bin-setup") + "-%Y-%m-%d")
    output_root.mkdir(parents=True, exist_ok=True)
    for counter in range(1, 100000):
        session_id = name or f"{prefix}-{counter:02d}"
        folder = output_root / session_id
        shared = artifacts / "data_saving/bin_model/sessions" / session_id
        parts = artifacts / "config/parts/data_saving/bin_model/sessions" / session_id
        if folder.exists() or shared.exists() or parts.exists():
            require(name is None, f"Session name already exists: {session_id}")
            continue
        try:
            folder.mkdir(exist_ok=False)
            break
        except FileExistsError:
            require(name is None, f"Session name already exists: {session_id}")
    else:
        raise ValueError("No unused session name available")
    runtime_session = runtime_root / session_id
    captures = capture_plan(recipe, session_id, mode, labels, group_id)
    (folder / "captures").mkdir()
    session = {"session_id": session_id, "mode": mode,
               "artifacts_root": str(artifacts), "bin_ids": list(recipe["bin_ids"]),
               "bin_labels": dict(captures[0]["bins"]), "group_id": group_id,
               "scene_id": recipe["scene_id"], "frame_id": recipe["frame_id"], "length_unit": "mm",
               "depth_axis": "z", "exposure_evidence": "requested_not_sensor_verified",
               "collection_interface": "standalone",
               "runtime_path": str(runtime_session),
               "calibration": {}, "captures": captures}
    if mode in ("guided", "check"):
        session["stages"] = guided_stages(recipe, mode)
        for stage in session["stages"]:
            stage["capture_ids"] = [capture["id"] for capture in captures if capture["stage_id"] == stage["id"]]
            stage["confirmation_path"] = f"stage_confirmations/{stage['id']}.json"
        session["acceptance"] = dict(recipe["guided"].get("acceptance", {}))
    settings = read_yaml(sources["settings"])
    settings["bins"] = {name: settings["bins"][name] for name in recipe["bin_ids"]}
    write_atomic(folder / "bin_config.yaml", dump_yaml(settings))
    session["settings"] = {"path": "bin_config.yaml"}
    session["calibration"] = {name: {"artifact_path": recipe[f"{name}_calibration"]}
                              for name in ("camera", "robot")}
    write_atomic(folder / "recipe.yaml", dump_yaml(recipe))
    session["recipe"] = {"path": "recipe.yaml"}
    for exposure in recipe["exposure_ms"]:
        relative = f"config/parts/data_saving/bin_model/sessions/{session_id}/{exposure_name(exposure)}.yaml"
        contents = dump_yaml(part_config(exposure, recipe, runtime_artifacts_root))
        # Eyes resolves depth settings by part_name from the shared artifact tree.
        # The native capture client also receives the identical session-local file.
        write_atomic(artifacts / relative, contents)
        write_atomic(folder / "configs" / f"{exposure_name(exposure)}.yaml", contents)
    session["eyes_config"] = {"artifact_path": EYES_WORKFLOW,
                              "shared_path": str(runtime_artifacts_root / EYES_WORKFLOW)}
    write_atomic(folder / "session.json", json.dumps(session, indent=2) + "\n")
    return folder


def stage_semantics(stages):
    """Compare the recorded plan without tying it to today's display wording.

    Keep the original text in the saved session. Only the three presentation
    fields below may vary; acquisition, labels, placement and confirmation
    metadata (including unknown added fields) must still match the recipe.
    """
    require(isinstance(stages, list), "Session stages must be a list")
    result = []
    for stage in stages:
        require(isinstance(stage, dict), "Session stages must be mappings")
        for key in ("title", "instructions"):
            require(isinstance(stage.get(key), str) and stage[key].strip(),
                    f"Stage {key} must be nonempty text")
        bins = stage.get("bins")
        require(isinstance(bins, dict), "Stage bins must be a mapping")
        semantic = {key: value for key, value in stage.items()
                    if key not in ("title", "instructions", "bins")}
        semantic["bins"] = {}
        for name, item in bins.items():
            require(isinstance(item, dict), "Stage bin metadata must be a mapping")
            require(isinstance(item.get("placement_instruction"), str)
                    and item["placement_instruction"].strip(),
                    "Stage placement_instruction must be nonempty text")
            semantic["bins"][name] = {key: value for key, value in item.items()
                                      if key != "placement_instruction"}
        result.append(semantic)
    return result


def verify_session(folder, decode_images=True):
    """Reject incomplete, altered, mislabeled or differently configured captures."""
    folder = Path(folder).resolve()
    session = json.loads((folder / "session.json").read_text())
    for entry in [session["settings"], session["recipe"]]:
        require(safe_path(folder, entry["path"]).is_file(), "Missing session snapshot")
    for name in ("camera", "robot"):
        resolve_calibration(folder, session, name)
    recipe = read_yaml(safe_path(folder, session["recipe"]["path"]))
    if "bin_ids" in recipe:
        require(session.get("bin_ids") == recipe["bin_ids"], "Session bins differ from recipe")
    captures = session["captures"]
    require(captures and len({c["id"] for c in captures}) == len(captures) and
            len({c["path"] for c in captures}) == len(captures), "Duplicate/empty capture plan")
    expected_plan = capture_plan(recipe, session["session_id"], session["mode"],
                                 session["bin_labels"] if session["mode"] == "loaded" else None,
                                 session.get("group_id"))
    require(captures == expected_plan, "Capture plan differs from recipe, labels or arrangement groups")
    if session["mode"] in ("guided", "check"):
        expected_stages = guided_stages(recipe, session["mode"])
        for stage in expected_stages:
            stage["capture_ids"] = [capture["id"] for capture in captures if capture["stage_id"] == stage["id"]]
            stage["confirmation_path"] = f"stage_confirmations/{stage['id']}.json"
        require(stage_semantics(session.get("stages")) == stage_semantics(expected_stages),
                "Stage arrangement, placement or confirmation metadata differs from recipe")
    for capture in captures:
        verify_capture(folder, session, capture, decode_images=decode_images)
    return len(captures)


def verify_capture(folder, session, capture, decode_images=True):
    """Verify one completed capture without requiring future stages to exist.

    The standalone controller calls this after each acquisition. Whole-session
    verification additionally checks the complete plan against the frozen recipe.
    """
    folder = Path(folder).resolve()
    require(capture in session["captures"], "Capture is not in this session's plan")
    recipe = read_yaml(safe_path(folder, session["recipe"]["path"]))
    if session["mode"] in ("guided", "check"):
        matches = [stage for stage in session.get("stages", []) if stage["id"] == capture.get("stage_id")]
        require(len(matches) == 1 and capture["id"] in matches[0]["capture_ids"],
                "Capture does not belong to its operator stage")
        stage = matches[0]
        confirmation_path = safe_path(folder, stage["confirmation_path"])
        require(confirmation_path.is_file(), f"Missing operator confirmation: {stage['id']}")
        confirmation = json.loads(confirmation_path.read_text())
        require(confirmation.get("session_id") == session["session_id"] and
                confirmation.get("stage_id") == stage["id"] and confirmation.get("confirmed_at") and
                confirmation.get("gate_token"), f"Missing operator confirmation: {stage['id']}")
    require(capture["split"] in ("train", "test") and set(capture["bins"]) == set(recipe.get("bin_ids", LEGACY_BINS)),
            "Capture split/bin labels are invalid")
    require(set(capture["bins"].values()) <= {"empty", "loaded"} and
            not (capture["split"] == "train" and "loaded" in capture["bins"].values()),
            "Loaded scenes cannot be used as empty-bin training data")
    path = safe_path(folder, capture["path"])
    require(not (path / ".capture_in_progress").exists(), f"Capture is still in progress: {capture['id']}")
    manifest = read_yaml(path / "capture_manifest.yaml")
    require(manifest.get("status") == "complete",
            f"Capture has no complete manifest: {capture['id']}")
    require(manifest.get("simulated") is False, "Simulation captures cannot validate a physical bin")
    require(manifest.get("scene_id") == session["scene_id"] and manifest.get("frame_id") == session["frame_id"]
            and manifest.get("length_unit") == "mm" and manifest.get("depth_axis") == "z",
            "Capture scene/frame/units do not match session")
    require(manifest.get("part_name") == capture["part_name"] and manifest.get("task_name") == "capture",
            "Capture used the wrong part configuration or task")
    entries = manifest.get("files", [])
    files = {entry["path"]: entry for entry in entries}
    required = {f"{kind}/Frame{camera}.exr" for kind in ("depth", "spotlight") for camera in range(2)}
    required |= {"requested_settings.yaml", "calibration/camera.yaml", "calibration/robot.yaml"}
    require(len(files) == len(entries) and required <= files.keys(), "Capture file manifest is incomplete")
    for relative, entry in files.items():
        item = safe_path(path, relative)
        require(item.is_file() and item.stat().st_size == entry["size_bytes"] > 0, f"Missing/altered capture file: {item}")
    for name in session["calibration"]:
        actual = manifest["calibration"][name]
        require(read_yaml(safe_path(path, actual["path"])) == read_yaml(resolve_calibration(folder, session, name)),
                f"Capture {name} calibration differs from session")
    settings = read_yaml(path / "requested_settings.yaml")
    require(manifest.get("requested_settings", {}).get("values_are_requested_not_sensor_verified") is True,
            "Capture must identify exposure metadata as requested settings")
    for key, expected in (("structured_light_exposure_ms", capture["exposure_ms"]),
                          ("spotlight_exposure_ms", capture["spotlight_exposure_ms"])):
        values = settings.get(key)
        require(isinstance(values, list) and len(values) == 2 and all(
            type(value) in (float, int) and math.isfinite(value) and abs(value - expected) < 1e-6
            for value in values), f"Wrong {key} in {capture['id']}")
    for key in ("gain", "black_level"):
        require(settings.get(key) == recipe[key], f"Wrong {key} in {capture['id']}")
    if decode_images:
        verify_images(path, manifest)
    return manifest


def verify_images(folder, manifest):
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2
    import numpy as np
    for kind in ("depth", "spotlight"):
        sizes = []
        for camera in range(2):
            path = folder / kind / f"Frame{camera}.exr"
            try:
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            except cv2.error:
                image = None
            if image is None:
                image = read_exr(path)
            require(image is not None and image.dtype == np.float32 and image.ndim == 2,
                    f"Invalid Float32 single-channel EXR: {folder}/{kind}/Frame{camera}.exr")
            metadata = manifest["images"][kind][camera]
            require(tuple(image.shape) == (metadata["height"], metadata["width"]), "Image shape differs from manifest")
            require(bool(np.any(np.isfinite(image) & (image > 0))), f"No valid positive pixels: {kind}/{camera}")
            sizes.append(image.shape)
        require(sizes[0] == sizes[1], f"The two {kind} images have different shapes")


def read_exr(path):
    """Decode original channel precision when OpenCV has no EXR codec."""
    import numpy as np
    try:
        import OpenEXR
    except ImportError as error:
        raise ValueError("EXR verification needs OpenEXR; install bin_filter_offline/requirements.txt") from error
    source = OpenEXR.InputFile(str(path))
    try:
        header = source.header()
        channels = header["channels"]
        require(len(channels) == 1, f"Expected a single EXR channel: {path}")
        name = next(iter(channels))
        channel = channels[name]
        require(channel.type.v == 2 and channel.xSampling == 1 and channel.ySampling == 1,
                f"Expected an original Float32 EXR channel: {path}")
        window = header["dataWindow"]
        shape = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)
        return np.frombuffer(source.channel(name), dtype=np.float32).reshape(shape).copy()
    finally:
        source.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Create a unique session and its exposure configurations")
    prepare.add_argument("--output-root", type=Path, required=True, help="Absolute session root on this host")
    prepare.add_argument("--runtime-root", type=Path, required=True, help="Session root as seen by the native capture process")
    prepare.add_argument("--artifacts-root", type=Path, default=ARTIFACTS, help="Artifact tree to update; use a copy for dry runs")
    prepare.add_argument("--runtime-artifacts-root", type=Path, default=Path("/root/artifacts"),
                         help="Artifact tree as seen by Eyes and the native capture process")
    prepare.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    prepare.add_argument("--exposures", type=float, nargs="+", help="Sorted unique depth exposures from 4 to 20 ms")
    prepare.add_argument("--mode", choices=("guided", "check"), default="guided")
    prepare.add_argument("--name", help="Optional readable unique session name (letters, digits, '_' and '-')")
    subparsers.add_parser("check", help="Validate the default recipe and calibration references without writing")
    verify = subparsers.add_parser("verify", help="Check every planned capture, settings, calibration and EXRs")
    verify.add_argument("session", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            recipe = read_yaml(DEFAULT_RECIPE)
            validate_recipe(recipe, ARTIFACTS)
            recipe.setdefault("bin_ids", list(read_yaml(resolve_bin_config(ARTIFACTS, recipe["bin_config"]))["bins"]))
            count = len(capture_plan(recipe, "preview", "guided", None, None))
            stage_count = len(guided_stages(recipe))
            check_count = len(capture_plan(recipe, "preview", "check", None, None))
            print(f"Recipe valid. Scene {recipe['scene_id']}, {len(recipe['bin_ids'])} bins: "
                  f"guided setup {count} captures across {stage_count} confirmations; quick check {check_count}.")
        elif args.command == "prepare":
            recipe = read_yaml(args.recipe)
            if args.exposures is not None:
                recipe["exposure_ms"] = args.exposures
            folder = prepare_session(args.artifacts_root, args.output_root, args.runtime_root,
                                     recipe, mode=args.mode, name=args.name,
                                     runtime_artifacts_root=args.runtime_artifacts_root)
            print(f"Session: {folder}\nRuntime path: {args.runtime_root / folder.name}")
            print("Open this session in the standalone Bin Setup UI.")
            print("Eyes and the capture process use the existing deployment calibration under the artifact tree.")
            print(f"Verify: python3 {Path(__file__).resolve()} verify {folder}")
        else:
            count = verify_session(args.session)
            print(f"PASS: {count} captures complete and verified. Exposure values are requested settings, not sensor readback.")
    except (ValueError, KeyError, OSError, TypeError, ImportError, yaml.YAMLError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
