"""Discover deployed calibration and configured bin layouts for the local UI."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import math
import numpy as np
import yaml

import dataset as data
import prepare_session as collection


CAD_REFERENCE_DIRECTORY = 'data_saving/bin_model/cad_references'  # Legacy setup locations remain readable.
LOCAL_CAD_REFERENCE_DIRECTORY = 'configs/cad_references'


def artifact_file(artifacts, relative, directory=None):
    artifacts = Path(artifacts).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise data.DatasetError('Select a file relative to the artifacts directory')
    path = (artifacts / relative).resolve()
    boundary = (artifacts / directory).resolve() if directory else artifacts
    if not path.is_relative_to(artifacts) or not path.is_relative_to(boundary) or not path.is_file():
        raise data.DatasetError(f'Missing or invalid artifact file: {relative}')
    return path


def cad_reference_file(artifacts, relative):
    relative = Path(relative)
    if relative.parts[:2] == ('configs', 'cad_references'):
        path = collection.safe_path(collection.HERE, str(relative))
        if not path.is_relative_to((collection.HERE / LOCAL_CAD_REFERENCE_DIRECTORY).resolve()) or not path.is_file():
            raise data.DatasetError('Select an existing CAD reference in bin_filter_offline/configs/cad_references')
        return path, collection.HERE.resolve()
    return artifact_file(artifacts, relative, CAD_REFERENCE_DIRECTORY), Path(artifacts).resolve()


def cad_reference_info(artifacts, relative):
    """Read a commissioned plan without loading meshes during UI polling."""
    artifacts = Path(artifacts).resolve()
    path, reference_root = cad_reference_file(artifacts, relative)
    if path.suffix != '.yaml':
        raise data.DatasetError('Select a CAD reference YAML plan')
    value = data.read_yaml(path)
    scene, frame, name = value.get('scene_id'), value.get('frame_id'), value.get('name')
    if (type(scene) is not int or not 0 <= scene <= 255
            or not isinstance(frame, str) or not frame.strip()
            or not isinstance(name, str) or not name.strip()):
        raise data.DatasetError('A CAD reference needs name, scene_id and frame_id')
    bins = value.get('bins')
    data.validate_bin_ids(bins)
    for bin_id, entry in bins.items():
        if not isinstance(entry, dict):
            raise data.DatasetError(f'CAD reference {bin_id} must be a mapping')
        mesh_name = entry.get('mesh_path')
        if not isinstance(mesh_name, str) or not mesh_name.strip() or Path(mesh_name).is_absolute():
            raise data.DatasetError(f'CAD reference {bin_id} needs mesh_path relative to its plan')
        mesh = (path.parent / mesh_name).resolve()
        if not mesh.is_relative_to(reference_root) or not mesh.is_file() or mesh.suffix.lower() != '.stl':
            raise data.DatasetError(f'CAD reference {bin_id} needs an STL file within its setup directory')
        scale = entry.get('scale_to_mm')
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise data.DatasetError(f'CAD reference {bin_id} needs explicit positive scale_to_mm')
        data.validate_transform(entry.get('bin_from_cad'), f'CAD reference {bin_id}.bin_from_cad')
    return {'path': str(Path(relative)), 'label': name, 'scene_id': scene,
            'frame_id': frame, 'bin_ids': list(bins)}


def selected_cad_reference(artifacts, relative, session):
    """CAD is a build option matched against the saved session, never a new setup."""
    if relative == '' or relative is None:
        return None
    if not isinstance(relative, str):
        raise data.DatasetError('Select a CAD reference plan from the station setup')
    info = cad_reference_info(artifacts, relative)
    if info['scene_id'] != session['scene_id'] or info['frame_id'] != session['frame_id']:
        raise data.DatasetError('The CAD reference must match the open session scene and reference frame')
    if not set(session['bin_ids']) <= set(info['bin_ids']):
        raise data.DatasetError('The CAD reference must include every bin in the open session')
    return cad_reference_file(artifacts, relative)[0]


def inventory(artifacts, recipe, *, layouts_root=None):
    artifacts = Path(artifacts).resolve()
    cameras, robots, layouts, cad_references, issues = [], [], [], [], []
    scene_ids = set(range(5))
    errors = (ValueError, TypeError, KeyError, OSError, OverflowError, RuntimeError, yaml.YAMLError)

    def scene_id(value):
        if type(value) is not int or not 0 <= value <= 255:
            raise data.DatasetError('Scene ID must be an integer from 0 to 255')
        return value

    try:
        scene_ids.add(scene_id(recipe['scene_id']))
    except errors as error:
        issues.append(f'Default recipe: {error}')
    manifest = artifacts / 'workcell_manifest.yaml'
    if manifest.is_file():
        try:
            value = data.read_yaml(artifact_file(artifacts, 'workcell_manifest.yaml'))
            entries = value.get('scenes', [])
            if not isinstance(entries, list):
                raise data.DatasetError('scenes must be a list of scene mappings')
            for item in entries:
                try:
                    if not isinstance(item, dict):
                        raise data.DatasetError('Each scene must be a mapping')
                    scene_ids.add(scene_id(item.get('scene_id')))
                except errors as error:
                    issues.append(f'workcell_manifest.yaml: {error}')
        except errors as error:
            issues.append(f'workcell_manifest.yaml: {error}')
    for path in sorted((artifacts / 'calibration/deployment/camera').glob('*.yaml')):
        relative = str(path.relative_to(artifacts))
        try:
            value = data.read_yaml(artifact_file(artifacts, relative, 'calibration/deployment/camera'))
            for name in ('cam1', 'cam2'):
                camera = value.get(name)
                if not isinstance(camera, dict):
                    raise data.DatasetError(f'{name} must be a camera calibration mapping')
                if any(type(camera.get(key)) is not int or camera[key] <= 0 for key in ('rows', 'cols')):
                    raise data.DatasetError(f'{name} needs positive integer rows and cols')
                intrinsic = np.asarray(camera.get('cam_intrinsic'), dtype=float).reshape(3, 3)
                if (not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                        or not np.allclose(intrinsic[2], [0, 0, 1])):
                    raise data.DatasetError(f'{name} has invalid camera intrinsics')
                data.validate_transform(camera.get('cam_extrinsic'), f'{name}.cam_extrinsic')
            match = re.match(r'scene(\d+)(?:_|\b)', path.stem)
            scene = scene_id(int(match.group(1))) if match else None
            if 'scene_id' in value:
                declared = scene_id(value['scene_id'])
                if scene is not None and declared != scene:
                    raise data.DatasetError('Camera filename and scene_id disagree')
                scene = declared
            if scene is not None:
                scene_ids.add(scene)
            cameras.append({'path': relative, 'label': path.name, 'scene_id': scene})
        except errors as error:
            issues.append(f'{path.name}: {error}')
    for path in sorted((artifacts / 'calibration/deployment/robot').glob('*.yaml')):
        try:
            relative = str(path.relative_to(artifacts))
            value = data.read_yaml(artifact_file(artifacts, relative, 'calibration/deployment/robot'))
            if not isinstance(value.get('name'), str) or not value['name'].strip():
                raise data.DatasetError('Robot calibration needs a nonempty frame name')
            entries = value.get('hand_eye')
            if not isinstance(entries, list) or not entries:
                raise data.DatasetError('hand_eye must contain scene calibration mappings')
            scenes = []
            for item in entries:
                if not isinstance(item, dict):
                    raise data.DatasetError('Each hand_eye entry must be a mapping')
                selected = scene_id(item.get('scene_id'))
                if selected in scenes:
                    raise data.DatasetError('hand_eye must contain each scene only once')
                calibration = item.get('calibration')
                if (not isinstance(calibration, list) or len(calibration) != 6
                        or any(type(number) not in (int, float) for number in calibration)):
                    raise data.DatasetError('hand_eye calibration must contain six finite numbers')
                data.pose_matrix(calibration)
                scenes.append(selected)
            robots.append({'path': relative, 'label': value['name'], 'scene_ids': scenes})
            scene_ids.update(scenes)
        except errors as error:
            issues.append(f'{path.name}: {error}')
    default_layout, candidates = None, set()
    try:
        default_layout = collection.resolve_bin_config(artifacts, recipe['bin_config'])
        if not default_layout.is_file():
            raise data.DatasetError(f'Missing bin layout: {default_layout}')
        candidates.add(default_layout)
    except errors as error:
        issues.append(f'Default bin layout: {error}')
    layout_root = artifacts / 'data_saving/bin_model/layouts'
    if layout_root.is_dir():
        candidates.update(layout_root.rglob('*.yaml'))
    candidates.update((collection.CONFIGS / 'layouts').rglob('*.yaml'))
    saved_layouts = Path(layouts_root).resolve() if layouts_root is not None else None
    if saved_layouts is not None:
        candidates.update(saved_layouts.glob('*.yaml'))
    for path in sorted(candidates):
        try:
            resolved = path.resolve()
            if resolved == default_layout:
                relative = recipe['bin_config']
            elif resolved.is_relative_to(collection.CONFIGS.resolve()):
                relative = str(resolved.relative_to(collection.HERE))
            elif saved_layouts is not None and resolved.parent == saved_layouts:
                relative = str(resolved)
            else:
                resolved = artifact_file(artifacts, path.relative_to(artifacts), 'data_saving/bin_model/layouts')
                relative = str(path.relative_to(artifacts))
            value = data.read_yaml(resolved)
            bins = value.get('bins')
            data.validate_bin_ids(bins)
            if any(not isinstance(item, dict) for item in bins.values()):
                raise data.DatasetError('Each configured bin must be a mapping')
            scene = value.get('scene_id', recipe['scene_id'] if resolved == default_layout else None)
            frame = value.get('frame_id', recipe['frame_id'] if resolved == default_layout else None)
            if type(scene) is not int or not 0 <= scene <= 255 or not isinstance(frame, str) or not frame.strip():
                raise data.DatasetError('A layout needs scene_id and frame_id')
            label = value.get('name', path.stem)
            if not isinstance(label, str) or not label.strip():
                raise data.DatasetError('A layout name must be nonempty text')
            scene_ids.add(scene)
            layouts.append({'path': relative, 'label': label,
                            'scene_id': scene, 'frame_id': frame, 'bin_ids': list(bins)})
        except errors as error:
            issues.append(f'{path.name}: {error}')
    for root, directory in ((collection.HERE, LOCAL_CAD_REFERENCE_DIRECTORY),
                            (artifacts, CAD_REFERENCE_DIRECTORY)):
        for path in sorted((root / directory).glob('*.yaml')):
            try:
                cad_references.append(cad_reference_info(artifacts, path.relative_to(root)))
            except errors as error:
                issues.append(f'{path.name}: {error}')
    workflow = _workflow_inventory(artifacts, cameras)
    scene_ids.update(item['scene_id'] for item in workflow['scenes'])
    issues.extend(workflow['issues'])
    return {'scenes': sorted(scene_ids), 'camera_calibrations': cameras, 'robot_calibrations': robots,
            'layouts': layouts, 'cad_references': cad_references, 'issues': issues, 'eyes_workflow': workflow,
            'defaults': {key: recipe.get(key) for key in
                         ('scene_id', 'camera_calibration', 'robot_calibration', 'bin_config', 'bin_ids')}}


def _workflow_inventory(artifacts, cameras):
    """Report only unambiguous scene/camera entries from the existing Eyes file."""
    relative = 'workflow_configs/bin_calibration.yaml'
    result = {'path': relative, 'scenes': [], 'issues': []}
    errors = (ValueError, TypeError, KeyError, OSError, OverflowError, RuntimeError, yaml.YAMLError)
    try:
        config = data.read_yaml(artifact_file(artifacts, relative, 'workflow_configs'))
        entries = config.get('scenes')
        if not isinstance(entries, list) or not entries:
            raise data.DatasetError('scenes must contain the scene/camera mappings used by Eyes')
        counts = {}
        for item in entries:
            if isinstance(item, dict) and type(item.get('scene_id')) is int:
                counts[item['scene_id']] = counts.get(item['scene_id'], 0) + 1
        available = {item['path']: item for item in cameras}
        for item in entries:
            try:
                if not isinstance(item, dict):
                    raise data.DatasetError('Each workflow scene must be a mapping')
                scene = item.get('scene_id')
                if type(scene) is not int or not 0 <= scene <= 255:
                    raise data.DatasetError('Scene ID must be an integer from 0 to 255')
                if counts[scene] != 1:
                    raise data.DatasetError(f'Scene {scene} is repeated; provide one configuration per scene')
                configured = item.get('camera')
                if (not isinstance(configured, list) or not configured
                        or any(not isinstance(camera, dict) for camera in configured)):
                    raise data.DatasetError(f'Scene {scene}: camera must contain camera mappings')
                denali = [camera for camera in configured if type(camera.get('cam_id')) is int and camera['cam_id'] == 0]
                if len(denali) != 1:
                    raise data.DatasetError(f'Scene {scene}: configure exactly one Denali camera with cam_id: 0')
                parameters = denali[0].get('parameters')
                if (not isinstance(parameters, list) or not parameters or not isinstance(parameters[0], str)
                        or not parameters[0].startswith('artifacts/')):
                    raise data.DatasetError(f'Scene {scene}: first camera parameter must be artifacts/calibration/deployment/camera/FILE.yaml')
                calibration = parameters[0][len('artifacts/'):]
                artifact_file(artifacts, calibration, 'calibration/deployment/camera')
                if calibration not in available:
                    raise data.DatasetError(f'Scene {scene}: camera calibration is unavailable or invalid: {calibration}')
                if available[calibration]['scene_id'] not in (None, scene):
                    raise data.DatasetError(f'Scene {scene}: camera calibration identifies another scene')
                result['scenes'].append({'scene_id': scene, 'camera_calibration': calibration, 'denali_id': 0})
            except errors as error:
                result['issues'].append(f'{relative}: {error}')
    except errors as error:
        result['issues'].append(f'{relative}: {error}')
    return result


def selected_recipe(artifacts, base, request, *, layouts_root=None):
    """Resolve a new setup explicitly, without borrowing another scene's bounds."""
    recipe = deepcopy(base)
    scene = request.get('scene_id', recipe['scene_id'])
    if type(scene) is not int or not 0 <= scene <= 255:
        raise data.DatasetError('Scene ID must be an integer from 0 to 255')
    recipe['scene_id'] = scene
    config = inventory(artifacts, base, layouts_root=layouts_root)
    for field, category in (('camera_calibration', 'camera_calibrations'),
                              ('robot_calibration', 'robot_calibrations'), ('bin_config', 'layouts')):
        selected = request.get(field, base[field])
        found = next((item for item in config[category] if item['path'] == selected), None)
        if found is None:
            raise data.DatasetError(f'Select an available {field.replace("_", " ")} from the station setup')
        if field == 'camera_calibration' and found['scene_id'] not in (None, scene):
            raise data.DatasetError('The camera calibration filename identifies a different scene')
        if field == 'robot_calibration':
            if scene not in found['scene_ids']:
                raise data.DatasetError('The reference-frame calibration has no transform for this scene')
            recipe['frame_id'] = found['label']
        if field == 'bin_config':
            if found['scene_id'] != scene or found['frame_id'] != recipe['frame_id']:
                raise data.DatasetError('Select a bin layout measured in this scene and reference frame')
            default_bins = base.get('bin_ids', found['bin_ids']) if selected == base['bin_config'] else found['bin_ids']
            selected_bins = request.get('bin_ids', default_bins)
            if (not isinstance(selected_bins, list) or not selected_bins or
                    any(not isinstance(value, str) for value in selected_bins) or
                    len(set(selected_bins)) != len(selected_bins) or not set(selected_bins) <= set(found['bin_ids'])):
                raise data.DatasetError('Select at least one bin from the chosen layout')
            recipe['bin_ids'] = selected_bins
        recipe[field] = selected
    if request.get('exposures_ms') is not None:
        recipe['exposure_ms'] = request['exposures_ms']
    # A visually measured layout is tied to the calibration that produced its
    # coordinates. Continue to read the shared station files, without copies in
    # artifacts or silently reusing old bounds after a calibration change.
    layout = data.read_yaml(collection.resolve_bin_config(artifacts, recipe['bin_config']))
    if 'calibration_reference' in layout:
        from model_io import calibration_values
        expected = layout['calibration_reference']
        if not isinstance(expected, dict) or not all(key in expected for key in ('camera', 'robot')):
            raise data.DatasetError('The saved layout calibration is incomplete; create a new visual layout')
        current = calibration_values(
            data.read_yaml(artifact_file(artifacts, recipe['camera_calibration'], 'calibration/deployment/camera')),
            data.read_yaml(artifact_file(artifacts, recipe['robot_calibration'], 'calibration/deployment/robot')),
            scene, recipe['frame_id'])
        expected = calibration_values(expected['camera'], expected['robot'], scene, recipe['frame_id'])
        if current != expected:
            raise data.DatasetError('Station calibration changed since this layout was measured. Create a new visual layout.')
    return recipe
