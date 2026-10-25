"""Capture a calibrated scene for drawing bin layouts before a session exists.

Preview data stays on this PC. Only the camera settings that Eyes resolves by
part name are published temporarily into the shared artifacts tree.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import base64
import json
import math
from pathlib import Path

import numpy as np

import dataset as data
import model_io
import prepare_session as collection


def _selected_recipe(artifacts, base, request):
    recipe = deepcopy(base)
    scene = request.get('scene_id', base.get('scene_id'))
    if type(scene) is not int or not 0 <= scene <= 255:
        raise data.DatasetError('Select a camera scene before capturing the bin layout')
    recipe.update(scene_id=scene, denali_id=0, length_unit='mm')
    paths = {}
    for name in ('camera', 'robot'):
        value = request.get(f'{name}_calibration', base.get(f'{name}_calibration'))
        paths[name] = collection.deployment_calibration(artifacts, value, name)
        recipe[f'{name}_calibration'] = value
    calibration = data.load_calibration(paths['camera'], paths['robot'], scene)
    recipe['frame_id'] = calibration['frame_id']
    if request.get('frame_id', calibration['frame_id']) != calibration['frame_id']:
        raise data.DatasetError('Selected reference frame does not match its calibration')
    collection.validate_eyes_workflow(artifacts, recipe)
    exposure = request.get('exposure_ms', 8.0)
    if type(exposure) not in (int, float) or not math.isfinite(exposure) or not 4 <= exposure <= 20:
        raise data.DatasetError('Layout preview exposure must be between 4 and 20 ms')
    for name in ('gain', 'black_level'):
        values = recipe.get(name)
        if (not isinstance(values, list) or len(values) != 2 or
                any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values)):
            raise data.DatasetError(f'Camera recipe requires two finite nonnegative {name} values')
    # Match structured and spotlight exposure even for older recipes.
    recipe.pop('spotlight_exposure_ms', None)
    return recipe, paths, float(exposure)


def _new_directory(data_root, artifacts):
    root = Path(data_root).resolve() / 'layout_previews'
    root.mkdir(parents=True, exist_ok=True)
    if not root.resolve().is_relative_to(Path(data_root).resolve()):
        raise data.DatasetError('Layout preview storage must stay under the configured data directory')
    prefix = datetime.now(timezone.utc).strftime('layout-preview-%Y-%m-%d')
    for index in range(1, 100000):
        name = f'{prefix}-{index:02d}'
        folder = root / name
        published = Path(artifacts) / 'config/parts/data_saving/bin_model/sessions' / name
        if not published.resolve().is_relative_to(Path(artifacts).resolve()):
            raise data.DatasetError('Temporary Eyes settings must stay inside the station artifacts')
        if published.exists():
            continue
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        try:
            published.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            folder.rmdir()
            continue
        return folder, published
    raise data.DatasetError('No unused layout preview name is available')


def reconstruct_points(capture, calibration):
    """Keep every observed depth sample; the geometry fitter needs rim density."""
    clouds, counts = [], []
    for index, camera in enumerate(calibration['cameras']):
        image = data.read_image(Path(capture) / f'depth/Frame{index}.exr', camera)
        points = data.backproject_depth(image, camera, calibration['robot_from_vision'])
        points = points[np.isfinite(points).all(axis=1)].astype(np.float32)
        clouds.append(points)
        counts.append(len(points))
    points = np.concatenate(clouds)
    if not len(points):
        raise data.DatasetError('No valid depth was captured. Check the camera view and try the layout preview again.')
    return points, counts


def _render(points):
    import layout_geometry
    return layout_geometry.render_preview(points)


def capture(artifacts, data_root, recipe, request, capture_runtime, progress):
    """Capture an empty scene without requiring or creating a collection session."""
    if request.get('empty_confirmed') is not True:
        raise data.DatasetError('Empty all bins and confirm the camera view is clear before capturing the layout')
    if not callable(capture_runtime):
        raise data.DatasetError('A layout preview requires the camera capture service')
    artifacts = Path(artifacts).resolve()
    selected, paths, exposure = _selected_recipe(artifacts, recipe, request)
    before = {name: data.read_yaml(path) for name, path in paths.items()}
    folder, published = _new_directory(data_root, artifacts)
    part = collection.exposure_name(exposure)
    session = {'session_id': folder.name, 'scene_id': selected['scene_id'],
               'frame_id': selected['frame_id'], 'length_unit': 'mm', 'artifacts_root': str(artifacts),
               'calibration': {name: {'artifact_path': selected[f'{name}_calibration']}
                               for name in ('camera', 'robot')}}
    entry = {'id': 'layout_preview', 'exposure_ms': exposure,
             'part_name': f'data_saving/bin_model/sessions/{folder.name}/{part}'}
    record = {'session': session, 'capture': entry, 'purpose': 'layout_preview_excluded_from_training'}
    collection.write_atomic(folder / 'capture_request.json', json.dumps(record, indent=2) + '\n')
    contents = collection.dump_yaml(collection.part_config(exposure, selected))
    destination = folder / 'capture'
    try:
        collection.write_atomic(folder / 'configs' / f'{part}.yaml', contents)
        collection.write_atomic(published / f'{part}.yaml', contents)
        progress('Capturing the empty scene for a bin layout. This preview is excluded from training and validation.')
        capture_runtime(folder, session, entry, destination, progress)
    finally:
        # The capture transport returns only after its native caller has finished
        # and released the camera buffers. Never remove another session's files.
        (published / f'{part}.yaml').unlink(missing_ok=True)
        published.rmdir()
    manifest = data.read_yaml(destination / 'capture_manifest.yaml')
    if any(manifest.get(key) != value for key, value in {
            'status': 'complete', 'scene_id': selected['scene_id'], 'frame_id': selected['frame_id'],
            'length_unit': 'mm', 'depth_axis': 'z'}.items()):
        raise data.DatasetError('Layout capture metadata does not match the selected scene and reference frame')
    captured_paths = {name: destination / f'calibration/{name}.yaml' for name in paths}
    for name, source in paths.items():
        if data.read_yaml(source) != before[name] or data.read_yaml(captured_paths[name]) != before[name]:
            raise data.DatasetError('Calibration changed during the layout capture. Capture a new preview before drawing bins.')
    calibration = data.load_calibration(captured_paths['camera'], captured_paths['robot'], selected['scene_id'])
    progress('Creating a top-down view from the captured depth and existing station calibration.')
    points, counts = reconstruct_points(destination, calibration)
    rendered = _render(points)
    (folder / 'preview.png').write_bytes(base64.b64decode(rendered['preview_png'], validate=True))
    width = rendered.get('width', rendered.get('width_px'))
    height = rendered.get('height', rendered.get('height_px'))
    # No object arrays or Python serialization: ordinary float32 coordinates.
    np.save(folder / 'points.npy', points, allow_pickle=False)
    metadata = {key: value for key, value in rendered.items() if key != 'preview_png'}
    metadata.update(path=str(folder), preview_id=folder.name, scene_id=selected['scene_id'],
                    frame_id=selected['frame_id'], length_unit='mm', exposure_ms=exposure,
                    camera_calibration=selected['camera_calibration'], robot_calibration=selected['robot_calibration'],
                    camera_points=counts, width=width, height=height, width_px=width, height_px=height,
                    xy_bounds_mm=[rendered['lower_xy_mm'], rendered['upper_xy_mm']],
                    caption=f"Scene {selected['scene_id']} in {selected['frame_id']}, viewed from above. "
                            'Draw one box around the entire outside edge of each empty bin, including its flat top rim.')
    collection.write_atomic(folder / 'preview.json', json.dumps(metadata, indent=2) + '\n')
    return {**metadata, 'preview_png': rendered['preview_png']}


def load_preview(data_root, preview_path):
    """Load only a completed preview owned by this application's data directory."""
    data_root = Path(data_root).resolve()
    root = (data_root / 'layout_previews').resolve()
    if not root.is_relative_to(data_root):
        raise data.DatasetError('Layout preview storage must stay under the configured data directory')
    path = Path(preview_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.parent != root or not path.name.startswith('layout-preview-'):
        raise data.DatasetError('Choose a captured layout preview from this Bin Setup application')
    for name in ('preview.json', 'points.npy'):
        item = path / name
        if not item.is_file() or item.resolve().parent != path:
            raise data.DatasetError('The layout preview is missing or incomplete; capture a new preview')
    metadata = json.loads((path / 'preview.json').read_text())
    if metadata.get('path') != str(path):
        raise data.DatasetError('The layout preview location changed; capture a new preview')
    points = np.load(path / 'points.npy', allow_pickle=False)
    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise data.DatasetError('The layout preview contains invalid depth coordinates')
    return points, metadata


def validate_preview_calibration(artifacts, preview_path, metadata):
    """Reject stale drawing coordinates when station geometry calibration changed.

    The returned small numeric reference can accompany a saved layout. This
    compares the values used for reconstruction, without checksums or copying
    calibration files into the artifact deployment.
    """
    folder = Path(preview_path).resolve()
    if metadata.get('path') != str(folder) or not folder.name.startswith('layout-preview-'):
        raise data.DatasetError('Choose a captured layout preview from this Bin Setup application')
    scene, frame = metadata.get('scene_id'), metadata.get('frame_id')
    if type(scene) is not int or not 0 <= scene <= 255 or not isinstance(frame, str) or not frame.strip():
        raise data.DatasetError('The layout preview is missing its calibrated scene and reference frame')
    current, captured = {}, {}
    for name in ('camera', 'robot'):
        source = collection.deployment_calibration(artifacts, metadata.get(f'{name}_calibration'), name)
        snapshot = folder / f'capture/calibration/{name}.yaml'
        if not snapshot.is_file() or not snapshot.resolve().is_relative_to(folder):
            raise data.DatasetError('The layout preview calibration is missing; capture a new preview')
        current[name], captured[name] = data.read_yaml(source), data.read_yaml(snapshot)
    reference = model_io.calibration_values(captured['camera'], captured['robot'], scene, frame)
    present = model_io.calibration_values(current['camera'], current['robot'], scene, frame)
    if present != reference:
        raise data.DatasetError('Station calibration changed since this layout preview. Capture a new preview and mark the bins again.')
    return reference
