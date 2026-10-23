"""Recorded-data IO; all geometry is in millimetres. No ROS/hardware access."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

# OpenCV disables its EXR codec by default in some binary distributions.
# Captures here are trusted factory artifacts; an explicit user override wins.
os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2
import numpy as np
import yaml


class DatasetError(ValueError):
    pass


EXPOSURE_ATOL_MS = 1e-5


def validate_bin_ids(bins):
    """Bin names also identify output directories and must stay within them."""
    if (not isinstance(bins, dict) or not bins or any(
            not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name)
            for name in bins)):
        raise DatasetError('Provide at least one named bin using letters, digits, underscore, dot or hyphen')
    return set(bins)


def validate_bin_labels(labels, expected=None):
    names = validate_bin_ids(labels)
    if expected is not None and names != set(expected):
        raise DatasetError('Capture labels must include every configured bin exactly once')
    if any(value not in ('empty', 'loaded') for value in labels.values()):
        raise DatasetError('Specify empty/loaded labels for every configured bin')
    return names


def positive_exposure(value):
    if isinstance(value, (bool, np.bool_)):
        raise DatasetError('Record a finite positive exposure in ms')
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise DatasetError('Record a finite positive exposure in ms') from error
    if not np.isfinite(value) or value <= 0:
        raise DatasetError('Record a finite positive exposure in ms')
    return value


def capture_exposure(entry):
    """Legacy exposure_ms is the structured-light exposure, never spotlight."""
    value = positive_exposure(entry['exposure_ms'])
    explicit = positive_exposure(entry.get('structured_light_exposure_ms', value))
    if abs(explicit - value) > EXPOSURE_ATOL_MS:
        raise DatasetError(f"{entry.get('id', 'capture')}: exposure differs between depth exposure fields")
    recorded = entry.get('source_metadata', {}).get('structured_light_exposure_ms')
    if recorded is not None and abs(positive_exposure(recorded) - value) > EXPOSURE_ATOL_MS:
        raise DatasetError(f"{entry.get('id', 'capture')}: exposure differs from recorded structured-light metadata")
    return value


def capture_spotlight_exposure(entry):
    recorded = entry.get('source_metadata', {}).get('spotlight_exposure_ms')
    value = entry.get('spotlight_exposure_ms', recorded)
    if value is None:
        return positive_exposure(recorded) if recorded is not None else None
    value = positive_exposure(value)
    if recorded is not None and abs(positive_exposure(recorded) - value) > EXPOSURE_ATOL_MS:
        raise DatasetError(f"{entry.get('id', 'capture')}: spotlight exposure differs from recorded metadata")
    return value


def validate_capture_groups(captures):
    groups = {}
    for entry in captures:
        group = entry.get('group_id')
        if group is None:
            continue
        if not isinstance(group, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', group):
            raise DatasetError('group_id must contain letters, digits, underscore or hyphen')
        if group in groups and groups[group] != entry['split']:
            raise DatasetError(f'Capture group {group!r} spans train/test; keep a layout/session in one split')
        groups[group] = entry['split']


def read_yaml(path):
    with Path(path).open() as stream:
        # Profile bundles contain several full grids. LibYAML avoids repeatedly
        # paying the pure-Python parsing cost while retaining safe constructors.
        value = yaml.load(stream, Loader=getattr(yaml, 'CSafeLoader', yaml.SafeLoader))
    if not isinstance(value, dict):
        raise DatasetError(f"Expected a YAML mapping: {path}")
    return value


def write_yaml(path, value):
    """Publish a whole YAML file, without leaving partially written manifests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            yaml.safe_dump(value, stream, sort_keys=False)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root, path):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def validate_transform(value, name='transform'):
    try:
        matrix = np.asarray(value, dtype=float).reshape(4, 4)
    except (ValueError, TypeError) as error:
        raise DatasetError(f"{name} must contain 16 numeric values") from error
    rotation = matrix[:3, :3]
    if (not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or
            not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise DatasetError(f"{name} must be a finite proper rigid transform (translation in mm)")
    return matrix


def pose_matrix(pose):
    """Match C++ PoseVecToMat: XYZ mm + rotation vector degrees, NOT Euler."""
    values = np.asarray(pose, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise DatasetError('Hand-eye calibration must have six finite numbers')
    rotation, _ = cv2.Rodrigues(np.deg2rad(values[3:]))
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, values[:3]
    return matrix


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def load_calibration(camera_path, robot_path, scene_id=0):
    if type(scene_id) is not int or scene_id < 0:
        raise DatasetError('Scene ID must be a nonnegative integer')
    cameras, robot = read_yaml(camera_path), read_yaml(robot_path)
    entries = [item for item in robot.get('hand_eye', []) if item.get('scene_id') == scene_id]
    if len(entries) != 1:
        raise DatasetError(f'Expected exactly one hand_eye entry for scene {scene_id}')
    if not isinstance(robot.get('name'), str) or not robot['name'].strip():
        raise DatasetError('Robot calibration must identify a nonempty robot frame')
    result = {'robot_from_vision': pose_matrix(entries[0]['calibration']), 'cameras': [],
              'frame_id': robot['name']}
    world = np.eye(4)
    if 'world_r' in cameras or 'world_t' in cameras:
        try:
            world[:3, :3] = np.asarray(cameras['world_r'], dtype=float).reshape(3, 3)
            world[:3, 3] = np.asarray(cameras['world_t'], dtype=float).reshape(3)
        except (KeyError, ValueError, TypeError) as error:
            raise DatasetError('Camera world_r/world_t must contain a 3x3 rotation and XYZ translation') from error
        validate_transform(world, 'camera world transform')
    for name in ('cam1', 'cam2'):
        item = cameras[name]
        intrinsic = np.asarray(item['cam_intrinsic'], dtype=float).reshape(3, 3)
        if (not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or
                not np.allclose(intrinsic[2], [0, 0, 1]) or
                abs(intrinsic[0, 1]) > 1e-8 or abs(intrinsic[1, 0]) > 1e-8):
            raise DatasetError(f'{name}: invalid or skewed intrinsics; match the contour pinhole convention')
        rows, cols = item['rows'], item['cols']
        if (type(rows) is not int or type(cols) is not int or rows <= 0 or cols <= 0):
            raise DatasetError(f'{name}: invalid rows/cols')
        result['cameras'].append({'intrinsic': intrinsic, 'rows': rows, 'cols': cols,
                                 'vision_from_camera': world @ validate_transform(item['cam_extrinsic'], name)})
    return result


def _read_exr(path):
    """Use the reference codec when OpenCV was built without OpenEXR support."""
    try:
        import OpenEXR
    except ImportError as error:
        raise DatasetError('EXR decoding requires OpenEXR: install bin_filter_offline/requirements.txt') from error
    source = OpenEXR.InputFile(str(path))
    try:
        header = source.header()
        channels = header['channels']
        if len(channels) != 1:
            raise DatasetError(f'{path}: expected one EXR channel, got {list(channels)}')
        name = next(iter(channels))
        channel = channels[name]
        dtype = {0: np.uint32, 1: np.float16, 2: np.float32}.get(channel.type.v)
        if dtype is None or channel.xSampling != 1 or channel.ySampling != 1:
            raise DatasetError(f'{path}: unsupported EXR channel encoding')
        window = header['dataWindow']
        shape = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)
        return np.frombuffer(source.channel(name), dtype=dtype).reshape(shape).copy()
    finally:
        source.close()


def read_image(path, camera, depth=True):
    image = None
    try:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    except cv2.error as error:
        if Path(path).suffix.lower() != '.exr':
            raise DatasetError(f'Cannot decode {path}: {error}') from error
    if image is None and Path(path).suffix.lower() == '.exr':
        image = _read_exr(path)
    if image is None:
        raise DatasetError(f'Cannot read image: {path}')
    if image.shape != (camera['rows'], camera['cols']):
        raise DatasetError(f"{path}: expected single-channel {camera['rows']}x{camera['cols']}; got {image.shape}")
    if depth and image.dtype != np.float32:
        raise DatasetError(f'{path}: depth must be original float32 optical-Z millimetres, not {image.dtype}')
    return image


def backproject_depth(depth, camera, robot_from_vision, stride=1):
    if not isinstance(stride, int) or stride < 1:
        raise DatasetError('Depth stride must be a positive integer')
    if depth.shape != (camera['rows'], camera['cols']) or depth.dtype != np.float32:
        raise DatasetError('Depth shape/type does not match calibration (float32 HxW required)')
    sampled = depth[::stride, ::stride]
    rows, cols = np.nonzero(np.isfinite(sampled) & (sampled > 0))
    z = sampled[rows, cols].astype(float)
    intrinsic = camera['intrinsic']
    points = np.column_stack(((cols * stride - intrinsic[0, 2]) * z / intrinsic[0, 0],
                              (rows * stride - intrinsic[1, 2]) * z / intrinsic[1, 1], z))
    return transform_points(points, robot_from_vision @ camera['vision_from_camera'])


def file_entry(path, root):
    path = Path(path).resolve()
    try:
        text = str(path.relative_to(Path(root).resolve()))
    except ValueError:
        text = str(path)
    return {'path': text, 'sha256': sha256(path)}


def checked_file(root, entry):
    path = resolve(root, entry['path'])
    if sha256(path) != entry['sha256']:
        raise DatasetError(f'File changed since it was registered: {path}')
    return path


def init_dataset(root, camera_path, robot_path, scene_id=0):
    calibration = load_calibration(camera_path, robot_path, scene_id)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'calibration').mkdir()
    for source, target in [(camera_path, 'camera.yaml'), (robot_path, 'robot.yaml')]:
        shutil.copy2(source, root / 'calibration' / target)
    manifest = {'scene_id': scene_id, 'frame_id': calibration['frame_id'], 'length_unit': 'mm',
                'calibration': {name: file_entry(root / 'calibration' / f'{name}.yaml', root)
                                for name in ('camera', 'robot')}, 'captures': []}
    write_yaml(root / 'dataset.yaml', manifest)
    return manifest


def load_dataset(root):
    root = Path(root)
    manifest = read_yaml(root / 'dataset.yaml')
    if (not isinstance(manifest.get('frame_id'), str) or not manifest['frame_id'].strip()
            or manifest.get('length_unit') != 'mm'):
        raise DatasetError('Unsupported dataset frame/unit')
    validate_capture_groups(manifest['captures'])
    calibration = manifest['calibration']
    cameras = checked_file(root, calibration['camera'])
    robot = checked_file(root, calibration['robot'])
    loaded = load_calibration(cameras, robot, manifest['scene_id'])
    if loaded['frame_id'] != manifest['frame_id']:
        raise DatasetError('Dataset frame differs from robot calibration')
    expected = None
    for entry in manifest['captures']:
        expected = validate_bin_labels(entry.get('bins'), expected)
    return manifest, loaded


def get_capture(manifest, capture_id):
    entries = [entry for entry in manifest['captures'] if entry['id'] == capture_id]
    if len(entries) != 1:
        raise DatasetError(f'Expected exactly one capture named {capture_id!r}')
    return entries[0]



def _within(root, relative):
    """Resolve archive references without allowing absolute/traversing paths."""
    if not isinstance(relative, str) or not relative:
        raise DatasetError('Capture/session file path must be nonempty and relative')
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise DatasetError('Capture/session file path must stay inside its root')
    result = (Path(root) / path).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise DatasetError('Capture/session file path resolves outside its root')
    return result


def _frozen_files(root, entries):
    if not isinstance(entries, dict) or set(entries) != {'camera', 'robot'}:
        raise DatasetError('Calibration must contain frozen camera and robot files')
    result = {}
    for name, entry in entries.items():
        path = _within(root, entry['path'])
        read_yaml(path)
        result[name] = path
    return result


def session_calibration_paths(root, session):
    """Use deployment calibration for current sessions and archived copies for old ones."""
    entries = session.get('calibration')
    if not isinstance(entries, dict) or set(entries) != {'camera', 'robot'}:
        raise DatasetError('Session calibration must identify camera and robot files')
    result = {}
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            raise DatasetError('Session calibration entries must be mappings')
        if 'artifact_path' in entry:
            artifacts = session.get('artifacts_root')
            if not isinstance(artifacts, str) or not Path(artifacts).is_absolute():
                raise DatasetError('Deployment calibration requires an absolute artifacts_root')
            path = _within(artifacts, entry['artifact_path'])
            if not path.is_relative_to((Path(artifacts) / 'calibration/deployment' / name).resolve()):
                raise DatasetError(f'{name}: use a deployment calibration file under artifacts/calibration/deployment/{name}')
        else:
            path = _within(root, entry.get('path'))
        read_yaml(path)
        result[name] = path
    return result


def _discover_completed_capture(root):
    manifest = read_yaml(root / 'capture_manifest.yaml')
    if manifest.get('status') != 'complete':
        raise DatasetError(f'{root}: incomplete capture manifest')
    for key, expected in (('length_unit', 'mm'), ('depth_axis', 'z')):
        if key in manifest and manifest[key] != expected:
            raise DatasetError(f'Capture {key} must be {expected}')
    if (root / '.capture_in_progress').exists():
        raise DatasetError(f'{root}: capture is still in progress')
    listed = {}
    for entry in manifest.get('files', []):
        path = _within(root, entry['path'])
        if entry['path'] in listed:
            raise DatasetError('Capture manifest contains duplicate file paths')
        if type(entry.get('size_bytes')) is not int or entry['size_bytes'] <= 0 or path.stat().st_size != entry['size_bytes']:
            raise DatasetError(f'{path}: capture file size mismatch')
        listed[entry['path']] = path
    required = ['requested_settings.yaml'] + [f'{kind}/Frame{index}.exr' for kind in ('depth', 'spotlight') for index in range(2)]
    if any(name not in listed for name in required):
        raise DatasetError('Completed capture must list requested_settings and both depth/spotlight EXR pairs')
    for kind in ('depth', 'spotlight'):
        images = manifest.get('images', {}).get(kind, [])
        # yaml-cpp indexing can emit either a sequence or an integer-key mapping.
        if isinstance(images, dict) and set(images) == {0, 1}:
            images = [images[0], images[1]]
        if not isinstance(images, list) or len(images) != 2:
            raise DatasetError(f'Capture must describe both {kind} camera images')
        for item in images:
            if (item.get('dtype') != 'float32' or type(item.get('height')) is not int
                    or type(item.get('width')) is not int or item['height'] <= 0 or item['width'] <= 0):
                raise DatasetError(f'Capture {kind} image metadata must describe positive float32 images')
    requested = read_yaml(listed['requested_settings.yaml'])
    if requested.get('values_are_requested_not_sensor_verified') is not True:
        raise DatasetError('Capture must explicitly identify requested settings as not sensor verified')
    def same_camera_exposure(value):
        if isinstance(value, list):
            if len(value) != 2:
                raise DatasetError('Expected exposure values for both cameras')
            values = [positive_exposure(item) for item in value]
            if abs(values[0] - values[1]) > EXPOSURE_ATOL_MS:
                raise DatasetError('Asymmetric camera exposure is unsupported; use matching settings for both cameras')
            return values[0]
        return positive_exposure(value)
    metadata = dict(manifest)
    metadata.update(structured_light_exposure_ms=same_camera_exposure(requested['structured_light_exposure_ms']),
                    spotlight_exposure_ms=same_camera_exposure(requested['spotlight_exposure_ms']),
                    exposure_evidence='requested_not_sensor_verified',
                    requested_settings=requested)
    if 'calibration' in manifest:
        snapshots = _frozen_files(root, manifest['calibration'])
        metadata['calibration_content'] = {name: read_yaml(path) for name, path in snapshots.items()}
    return ([listed[f'depth/Frame{i}.exr'] for i in range(2)],
            [listed[f'spotlight/Frame{i}.exr'] for i in range(2)], metadata)


def import_session(session_path, dataset_root):
    """Import all completed captures, using immutable recipe labels and settings.

    The new dataset is staged and published only after every capture passes.
    Source images remain referenced in their original session directory.
    """
    session_path = Path(session_path).resolve()
    if session_path.is_dir():
        session_path /= 'session.json'
    session_root = session_path.parent
    with session_path.open() as stream:
        session = json.load(stream)
    if (not isinstance(session.get('frame_id'), str) or not session['frame_id'].strip()
            or session.get('length_unit') != 'mm' or type(session.get('scene_id')) is not int
            or session['scene_id'] < 0):
        raise DatasetError('Unsupported session frame/units/scene')
    calibration = session_calibration_paths(session_root, session)
    if load_calibration(calibration['camera'], calibration['robot'], session['scene_id'])['frame_id'] != session['frame_id']:
        raise DatasetError('Session frame differs from robot calibration')
    settings = _within(session_root, session['settings']['path'])
    configured_bins = read_yaml(settings).get('bins', {})
    recipe = read_yaml(_within(session_root, session['recipe']['path'])) if 'recipe' in session else {}
    captures = session.get('captures')
    if not isinstance(captures, list) or not captures:
        raise DatasetError('Session contains no capture requests')
    # Older recorded sessions sometimes omitted fitting regions. Labels still
    # must agree across every capture, and guided sessions require real regions.
    expected_bins = validate_bin_ids(configured_bins) if configured_bins else validate_bin_labels(captures[0].get('bins'))
    for entry in captures:
        validate_bin_labels(entry.get('bins'), expected_bins)
    validate_capture_groups(captures)
    stages = session.get('stages', [])
    confirmations = {}
    if session.get('mode') in ('guided', 'check'):
        if not isinstance(session.get('session_id'), str) or not session['session_id'].strip():
            raise DatasetError('Guided collection requires a session ID')
        if not isinstance(stages, list) or not stages:
            raise DatasetError('Guided collection requires operator stages')
        by_stage = {}
        assigned = set()
        for item in stages:
            stage_id = item.get('id')
            if (not isinstance(stage_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', stage_id)
                    or stage_id in by_stage):
                raise DatasetError('Guided stages require unique IDs')
            by_stage[stage_id] = item
            if validate_bin_ids(item.get('bins')) != expected_bins:
                raise DatasetError(f'{stage_id}: stage must describe every configured bin')
            confirmation = _within(session_root, item['confirmation_path'])
            if not confirmation.is_file():
                raise DatasetError(f'{stage_id}: operator confirmation is missing')
            with confirmation.open() as stream:
                confirmed = json.load(stream)
            if (confirmed.get('session_id') != session.get('session_id')
                    or confirmed.get('stage_id') != stage_id
                    or not isinstance(confirmed.get('confirmed_at'), str)
                    or not confirmed['confirmed_at'].strip()):
                raise DatasetError(f'{stage_id}: operator confirmation does not match session/stage')
            confirmations[stage_id] = confirmed
            ids = item.get('capture_ids', [])
            if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or assigned.intersection(ids):
                raise DatasetError('Each guided capture must belong to exactly one operator stage')
            assigned.update(ids)
        if assigned != {entry['id'] for entry in captures}:
            raise DatasetError('Guided stage capture lists differ from the session captures')
        for entry in captures:
            item = by_stage.get(entry.get('stage_id'))
            if item is None or entry['id'] not in item['capture_ids']:
                raise DatasetError(f"{entry['id']}: capture is assigned to the wrong operator stage")
            labels = {name: values['label'] for name, values in item['bins'].items()}
            if entry['split'] != item['split'] or entry['bins'] != labels:
                raise DatasetError(f"{entry['id']}: capture labels/split disagree with its operator stage")
    # Check completion and calibration for the entire session before staging.
    expected_calibration = {name: read_yaml(path) for name, path in calibration.items()}
    for entry in captures:
        capture_root = _within(session_root, entry['path'])
        if not (capture_root / 'capture_manifest.yaml').is_file():
            raise DatasetError(f"{entry['id']}: capture is not complete; missing capture_manifest.yaml")
        _, _, metadata = _discover_completed_capture(capture_root)
        if metadata.get('scene_id') != session['scene_id'] or metadata.get('frame_id') != session['frame_id']:
            raise DatasetError(f"{entry['id']}: capture scene/frame differs from session")
        if 'part_name' in entry and (metadata.get('part_name') != entry['part_name'] or metadata.get('task_name') != 'capture'):
            raise DatasetError(f"{entry['id']}: capture used the wrong part configuration or task")
        for key in ('gain', 'black_level'):
            if key in recipe and metadata['requested_settings'].get(key) != recipe[key]:
                raise DatasetError(f"{entry['id']}: requested {key} differs from collection recipe")
        if metadata.get('calibration_content') != expected_calibration:
            raise DatasetError(f"{entry['id']}: capture calibration differs from session deployment calibration")
        if metadata.get('simulated') is not False:
            raise DatasetError(f"{entry['id']}: factory collection requires explicitly non-simulated captures")
    dataset_root = Path(dataset_root).resolve()
    if dataset_root.exists():
        raise DatasetError(f'Refusing to overwrite existing dataset: {dataset_root}')
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    stage_parent = Path(tempfile.mkdtemp(prefix='.bin-session-', dir=dataset_root.parent))
    stage = stage_parent / 'dataset'
    try:
        init_dataset(stage, calibration['camera'], calibration['robot'], session['scene_id'])
        fitting_settings = read_yaml(settings)
        fitting_settings.pop('schema_version', None)
        write_yaml(stage / 'bin_config.yaml', fitting_settings)
        archived_session = dict(session)
        archived_session.pop('schema_version', None)
        write_json(stage / 'session.json', archived_session)
        if recipe:
            write_yaml(stage / 'recipe.yaml', recipe)
        if confirmations:
            (stage / 'stage_confirmations').mkdir()
        for stage_id, confirmed in confirmations.items():
            write_json(stage / 'stage_confirmations' / f'{stage_id}.json', confirmed)
        for entry in captures:
            add_capture(stage, entry['id'], entry['split'], entry['bins'], entry['exposure_ms'],
                        capture_dir=_within(session_root, entry['path']),
                        spotlight_exposure_ms=entry['spotlight_exposure_ms'], group_id=entry['group_id'],
                        note='Imported from frozen collection session; requested exposure is not hardware verified')
        manifest = read_yaml(stage / 'dataset.yaml')
        original_entries = {entry['id']: entry for entry in captures}
        for entry in manifest['captures']:
            original = original_entries[entry['id']]
            if 'stage_id' in original:
                entry['stage_id'] = original['stage_id']
        if stages:
            manifest['stages'] = stages
        if 'acceptance' in session:
            manifest['acceptance'] = session['acceptance']
        manifest['collection_mode'] = session.get('mode', 'legacy')
        if session.get('synthetic') is True:
            manifest['synthetic'] = True
        manifest['operator_confirmations'] = confirmations
        manifest['collection_session'] = {'path': str(session_path),
                                          'session_id': session.get('session_id'),
                                          'exposure_evidence': 'requested_not_sensor_verified'}
        write_yaml(stage / 'dataset.yaml', manifest)
        stage.rename(dataset_root)
    finally:
        shutil.rmtree(stage_parent)
    return manifest

def discover_images(capture_dir, record=None):
    """Never pair the later inspection images with an earlier contour request."""
    root = Path(capture_dir).resolve()
    if record is None and (root / 'capture_manifest.yaml').exists():
        return _discover_completed_capture(root)
    if record is None and ((root / 'depth').exists() or (root / 'spotlight').exists()):
        raise DatasetError(f'{root}: EXR capture has no completed capture_manifest.yaml')
    records = sorted((root / 'robot_pose').glob('contour_*.yaml'))
    if record is None and records:
        if len(records) != 1:
            raise DatasetError('Multiple contour records: select the exact one using --record')
        record = records[0]
    if record is not None:
        record = resolve(root, record)
        metadata = read_yaml(record)
        result = []
        for key in ('depth_images', 'spotlight_images'):
            entries = sorted(metadata.get(key, []), key=lambda item: item['camera_index'])
            if len(entries) != 2 or [item['camera_index'] for item in entries] != [0, 1]:
                raise DatasetError(f'{record}: expected camera indices 0 and 1 in {key}')
            if key == 'depth_images':
                for index, item in enumerate(entries):
                    if (item.get('length_unit') != 'mm' or item.get('depth_axis') != 'z' or
                            item.get('frame_id') != f'camera_{index}' or item.get('scalar_type') != 'float32'):
                        raise DatasetError(f'{record}: depth frame/units/type mismatch')
            result.append([resolve(root, item['path']) for item in entries])
        return result[0], result[1], metadata
    status_path = root / 'capture_status.yaml'
    if status_path.exists() and read_yaml(status_path).get('status') != 'complete':
        raise DatasetError(f'{root}: capture status is incomplete')
    depths = [root / 'depth_map' / f'Frame{i}.tiff' for i in range(2)]
    spots = [root / 'spot_light_images' / f'Frame{i}.tiff' for i in range(2)]
    metadata = read_yaml(root / 'capture_metadata.yaml') if (root / 'capture_metadata.yaml').exists() else {}
    if status_path.exists():
        metadata['capture_status'] = read_yaml(status_path)
    return depths, spots, metadata


def add_capture(root, capture_id, split, labels, exposure_ms, capture_dir=None, record=None,
                cloud_path=None, cloud_frame=None, note='', spotlight_exposure_ms=None, group_id=None):
    manifest, calibration = load_dataset(root)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', capture_id):
        raise DatasetError('Capture ID must contain letters, digits, underscore or hyphen')
    if any(entry['id'] == capture_id for entry in manifest['captures']):
        raise DatasetError(f'Capture ID already registered: {capture_id}')
    if split not in ('train', 'test'):
        raise DatasetError('Specify train/test for each capture')
    configured = read_yaml(Path(root) / 'bin_config.yaml').get('bins') if (Path(root) / 'bin_config.yaml').is_file() else None
    expected = configured or (manifest['captures'][0]['bins'] if manifest['captures'] else None)
    validate_bin_labels(labels, expected)
    exposure_ms = positive_exposure(exposure_ms)
    if spotlight_exposure_ms is not None:
        spotlight_exposure_ms = positive_exposure(spotlight_exposure_ms)
    entry = {'id': capture_id, 'split': split, 'bins': labels, 'exposure_ms': exposure_ms,
             'structured_light_exposure_ms': exposure_ms, 'note': note}
    if group_id is not None:
        entry['group_id'] = group_id
        validate_capture_groups([*manifest['captures'], entry])
    if cloud_path:
        if cloud_frame not in ('robot', 'vision'):
            raise DatasetError('Cloud import requires --cloud-frame robot or vision, in mm')
        cloud_path = Path(cloud_path)
        if cloud_path.name.startswith('contour_'):
            raise DatasetError('Contour PLY files are already filtered; import the raw depth record instead')
        points = read_cloud(cloud_path)
        if len(points) < 3:
            raise DatasetError('Unfiltered cloud is empty or has too few finite points')
        entry['cloud'] = dict(file_entry(cloud_path, root), frame=cloud_frame, length_unit='mm', unfiltered=True)
        signature = entry['cloud']['sha256']
    else:
        depths, spots, metadata = discover_images(capture_dir, record)
        recorded_calibration = metadata.get('calibration_content')
        expected_calibration = {key: read_yaml(resolve(root, item['path']))
                                for key, item in manifest['calibration'].items()}
        if recorded_calibration is not None and recorded_calibration != expected_calibration:
            raise DatasetError('Capture calibration differs from frozen dataset calibration')
        if metadata.get('frame_id', metadata.get('robot_id', manifest['frame_id'])) != manifest['frame_id']:
            raise DatasetError('Capture robot_id differs from dataset frame')
        recorded_depth = metadata.get('structured_light_exposure_ms')
        if recorded_depth is not None and abs(positive_exposure(recorded_depth) - exposure_ms) > EXPOSURE_ATOL_MS:
            raise DatasetError(f'Exposure differs from structured_light_exposure_ms in capture metadata: {recorded_depth} ms')
        recorded_spot = metadata.get('spotlight_exposure_ms')
        if recorded_spot is not None:
            recorded_spot = positive_exposure(recorded_spot)
            if spotlight_exposure_ms is not None and abs(recorded_spot - spotlight_exposure_ms) > EXPOSURE_ATOL_MS:
                raise DatasetError(f'Exposure differs from spotlight_exposure_ms in capture metadata: {recorded_spot} ms')
            spotlight_exposure_ms = recorded_spot
        if 'scene_id' in metadata and metadata['scene_id'] != manifest['scene_id']:
            raise DatasetError('Capture scene_id differs from dataset')
        for i, camera in enumerate(calibration['cameras']):
            for kind in ('depth', 'spotlight'):
                description = metadata.get('images', {}).get(kind)
                if description is not None and (description[i]['height'] != camera['rows']
                                                or description[i]['width'] != camera['cols']):
                    raise DatasetError(f'{kind}: capture image dimensions differ from calibration')
            depth = read_image(depths[i], camera)
            if not np.any(np.isfinite(depth) & (depth > 0)):
                raise DatasetError(f'{depths[i]}: no valid positive depths')
            spotlight = read_image(spots[i], camera, depth=False)
            if 'images' in metadata and spotlight.dtype != np.float32:
                raise DatasetError(f'{spots[i]}: captured spotlight must retain original float32 pixels')
        entry['depth_images'] = [file_entry(path, root) for path in depths]
        entry['spotlight_images'] = [file_entry(path, root) for path in spots]
        entry['acquisition'] = metadata.get('acquisition', {})
        entry['source_metadata'] = metadata
        if 'exposure_evidence' in metadata:
            entry['exposure_evidence'] = metadata['exposure_evidence']
        signature = hashlib.sha256(''.join(item['sha256'] for item in entry['depth_images']).encode()).hexdigest()
    if any(item['input_signature'] == signature for item in manifest['captures']):
        raise DatasetError('These inputs are already registered; duplicates cannot be independent test captures')
    entry['input_signature'] = signature
    # Absent metadata is unknown; do not invent a spotlight exposure for clouds
    # or legacy acquisitions that recorded only structured light.
    entry['spotlight_exposure_ms'] = spotlight_exposure_ms
    manifest['captures'].append(entry)
    write_yaml(Path(root) / 'dataset.yaml', manifest)
    return entry


def read_cloud(path):
    path = Path(path)
    if path.suffix.lower() in ('.xyz', '.xyzn', '.txt'):
        points = np.loadtxt(path, usecols=(0, 1, 2), ndmin=2)
    elif path.suffix.lower() == '.npy':
        points = np.load(path, allow_pickle=False)
    else:
        import open3d as o3d
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise DatasetError(f'Expected Nx3 cloud (XYZ mm): {path}')
    return points[np.isfinite(points).all(axis=1)].astype(float)


def load_capture_views(root, entry, calibration, stride=1):
    """Preserve camera identity and sampled depth validity for exposure QA."""
    if 'cloud' in entry:
        item = entry['cloud']
        if item.get('unfiltered') is not True or item.get('length_unit') != 'mm':
            raise DatasetError('Cloud inputs must explicitly be unfiltered and in millimetres')
        points = read_cloud(checked_file(root, item))
        if item['frame'] == 'vision':
            points = transform_points(points, calibration['robot_from_vision'])
        elif item['frame'] != 'robot':
            raise DatasetError('Cloud frame must be robot or vision')
        return [{'camera_index': None, 'points': points, 'sampled_depth_pixels': None,
                 'valid_depth_pixels': None, 'valid_depth_fraction': None}]
    pairs = entry['depth_images']
    if len(pairs) != 2:
        raise DatasetError('Two depth images are required')
    views = []
    for i, item in enumerate(pairs):
        camera = calibration['cameras'][i]
        depth = read_image(checked_file(root, item), camera)
        points = backproject_depth(depth, camera, calibration['robot_from_vision'], stride)
        count = depth[::stride, ::stride].size
        acquisition = {}
        settings = entry.get('source_metadata', {}).get('requested_settings', {})
        for key in ('gain', 'black_level'):
            value = settings.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                value = value[i]
            if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                acquisition[key] = float(value)
        views.append({'camera_index': i, 'points': points, 'sampled_depth_pixels': count,
                      'acquisition': acquisition,
                      'valid_depth_pixels': len(points), 'valid_depth_fraction': len(points) / count})
    return views


def load_capture_points(root, entry, calibration, stride=1):
    return np.concatenate([view['points'] for view in load_capture_views(root, entry, calibration, stride)])


def write_ply(path, points, colors=None, frame_id='robot'):
    """Binary PLY keeps full-scale captures compact; RGB colors are uint8."""
    points = np.asarray(points)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    fields = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
    if colors is not None:
        fields += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        colors = np.asarray(colors, dtype=np.uint8)[valid]
    data = np.empty(len(points), dtype=fields)
    for i, name in enumerate(('x', 'y', 'z')):
        data[name] = points[:, i]
    header = ['ply', 'format binary_little_endian 1.0', f'comment frame {frame_id} units mm',
              f'element vertex {len(points)}', 'property float x', 'property float y', 'property float z']
    if colors is not None:
        for i, name in enumerate(('red', 'green', 'blue')):
            data[name] = colors[:, i]
            header.append(f'property uchar {name}')
    with Path(path).open('xb') as stream:
        stream.write(('\n'.join(header) + '\nend_header\n').encode())
        data.tofile(stream)


def cloud_summary(points, frame_id='robot'):
    return {'point_count': int(len(points)), 'frame_id': frame_id, 'length_unit': 'mm',
            'min_mm': points.min(axis=0).tolist() if len(points) else None,
            'max_mm': points.max(axis=0).tolist() if len(points) else None}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
