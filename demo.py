"""Small analytic stereo renderer for a reproducible, hardware-free example.

These are invented bins/cameras, never a calibration for the real workcell.
The rendered TIFFs exercise the SAME importer/reconstruction/fitter as real data.
"""
from pathlib import Path

import cv2
import numpy as np

import dataset as data


def camera_at(origin, target):
    z = np.asarray(target, dtype=float) - origin
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack((x, y, z))
    matrix[:3, 3] = origin
    return {'rows': 240, 'cols': 320,
            'cam_intrinsic': [370.0, 0, 159.5, 0, 370.0, 119.5, 0, 0, 1],
            'cam_extrinsic': matrix.ravel().tolist()}


def surfaces_for_box(lo, hi, label=0, floor=True):
    # Each plane is clipped to a finite rectangle. Open tops remain open.
    surfaces = []
    if floor:
        surfaces.append((2, lo[2], lo, hi, label))
    for axis in (0, 1):
        for value in (lo[axis], hi[axis]):
            surfaces.append((axis, value, lo, hi, label))
    return surfaces


def render(camera, surfaces, rng):
    k = np.asarray(camera['cam_intrinsic']).reshape(3, 3)
    transform = np.asarray(camera['cam_extrinsic']).reshape(4, 4)
    rows, cols = np.indices((camera['rows'], camera['cols']))
    rays = np.stack(((cols - k[0, 2]) / k[0, 0], (rows - k[1, 2]) / k[1, 1],
                     np.ones_like(rows)), axis=-1).reshape(-1, 3)
    directions = rays @ transform[:3, :3].T
    origin = transform[:3, 3]
    depth = np.full(len(rays), np.inf)
    truth = np.zeros(len(rays), dtype=np.uint8)
    for axis, value, lo, hi, label in surfaces:
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (value - origin[axis]) / directions[:, axis]
            intersections = origin + directions * t[:, None]
        axes = [i for i in range(3) if i != axis]
        hit = (np.isfinite(t) & (t > 0) & (t < depth) &
               np.all(intersections[:, axes] >= np.asarray(lo)[axes] - 1e-7, axis=1) &
               np.all(intersections[:, axes] <= np.asarray(hi)[axes] + 1e-7, axis=1))
        depth[hit], truth[hit] = t[hit], label
    valid = np.isfinite(depth)
    depth[valid] += rng.normal(0, 0.025, valid.sum())
    depth[~valid] = 0
    spotlight = np.where(valid, 60.0 + truth * 50.0, 0).astype(np.float32)
    shape = (camera['rows'], camera['cols'])
    return depth.reshape(shape).astype(np.float32), spotlight.reshape(shape), truth.reshape(shape)


def generate_demo(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    calibrations = root / 'synthetic_calibration'
    calibrations.mkdir()
    cameras = {'cam1': camera_at([-180, -210, 650], [0, 0, 110]),
               'cam2': camera_at([180, 210, 650], [0, 0, 110])}
    robot = {'name': 'B2', 'hand_eye': [{'scene_id': 0, 'calibration': [0, 0, 0, 0, 0, 0]}]}
    data.write_yaml(calibrations / 'camera.yaml', cameras)
    data.write_yaml(calibrations / 'robot.yaml', robot)
    dataset = root / 'dataset'
    data.init_dataset(dataset, calibrations / 'camera.yaml', calibrations / 'robot.yaml')
    settings = {'bins': {}}
    for name, x in [('bin_a', -70.0), ('bin_b', 70.0)]:
        transform = np.eye(4)
        transform[:3, 3] = [x, 0, 100]
        settings['bins'][name] = {
            'robot_from_bin_seed': transform.tolist(),
            'interior_polygon_mm': [[-50, -40], [50, -40], [50, 40], [-50, 40]],
            'floor_search_band_mm': 2.0, 'wall_search_band_mm': 3.0,
            'wall_height_range_mm': [5.0, 40.0], 'max_height_mm': 90.0,
            'fit': {'ransac_distance_mm': 0.2, 'max_tolerance_mm': 0.8,
                    'alignment_margin_mm': 0.15, 'iterations': 100},
        }
    data.write_yaml(dataset / 'bin_config.yaml', settings)
    # Smaller than the recommended real dataset, but contains independent
    # train/test empty captures and held-out 3 mm thin objects in both bins.
    entries = [(f'empty_{i:02d}', 'train' if i < 4 else 'test', False, False) for i in range(6)]
    entries += [('loaded_train', 'train', True, True), ('loaded_a_test', 'test', True, False),
                ('loaded_b_test', 'test', False, True), ('loaded_both_test', 'test', True, True)]
    rng = np.random.default_rng(123)
    for capture_id, split, has_a, has_b in entries:
        directory = dataset / 'captures' / capture_id
        for subdirectory in ('depth_map', 'spot_light_images', 'truth'):
            (directory / subdirectory).mkdir(parents=True, exist_ok=False)
        surfaces = []
        for bin_index, (x, loaded) in enumerate([(-70, has_a), (70, has_b)]):
            surfaces.extend(surfaces_for_box([x - 50, -40, 100], [x + 50, 40, 150]))
            if loaded:
                offset = 12 if capture_id == 'loaded_train' else -8
                lo, hi = [x + offset - 12, -9, 100], [x + offset + 12, 9, 103]
                surfaces.append((2, 103, lo, hi, bin_index + 1))
                surfaces.extend(surfaces_for_box(lo, hi, label=bin_index + 1, floor=False))
        for index, camera in enumerate(cameras.values()):
            depth, spotlight, truth = render(camera, surfaces, rng)
            for subdirectory, value in [('depth_map', depth), ('spot_light_images', spotlight)]:
                if not cv2.imwrite(str(directory / subdirectory / f'Frame{index}.tiff'), value):
                    raise RuntimeError('Could not write synthetic TIFF')
            np.save(directory / 'truth' / f'Frame{index}.npy', truth)
        data.write_yaml(directory / 'capture_status.yaml', {'status': 'complete', 'synthetic': True})
        data.write_yaml(directory / 'capture_metadata.yaml', {'structured_light_exposure_ms': 8.0,
                                                           'spotlight_exposure_ms': 8.0, 'synthetic': True})
        data.add_capture(dataset, capture_id, split,
                         {'bin_a': 'loaded' if has_a else 'empty', 'bin_b': 'loaded' if has_b else 'empty'},
                         8.0, capture_dir=directory, note='SYNTHETIC: analytic bins, not a workcell calibration')
    manifest = data.read_yaml(dataset / 'dataset.yaml')
    manifest['synthetic'] = True
    data.write_yaml(dataset / 'dataset.yaml', manifest)
    (root / 'SYNTHETIC_ONLY.txt').write_text(
        'Invented camera/bin geometry for software verification only. Never deploy these models to a robot.\n'
        'The truth/ arrays label rendered object pixels (1=bin A object, 2=bin B object, 0=background).\n')
    return root
