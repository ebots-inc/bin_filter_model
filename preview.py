"""Small labeled preview from a fresh raw capture, in robot XY coordinates."""
import base64
import json
from pathlib import Path

import cv2
import numpy as np

import dataset as data


def robot_regions(settings):
    regions = {}
    for name, config in settings['bins'].items():
        lo, hi = np.asarray(config['xy_bounds_mm'], dtype=float)
        vertices = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
        if config.get('coordinate_frame') == 'bin':
            placement = data.validate_transform(config['robot_from_bin'])
            xyz = np.array([[x, y, z] for z in config['z_bounds_mm'] for x, y in vertices])
            projected = data.transform_points(xyz, placement)[:, :2].astype(np.float32)
            vertices = cv2.convexHull(projected).reshape(-1, 2)
        regions[name] = vertices
    return regions


def render(capture, session):
    capture, session = Path(capture), Path(session)
    settings = data.read_yaml(session / 'bin_config.yaml')
    recorded = json.loads((session / 'session.json').read_text())
    paths = data.session_calibration_paths(session, recorded)
    calibration = data.load_calibration(paths['camera'], paths['robot'], recorded['scene_id'])
    regions = robot_regions(settings)
    vertices = np.concatenate(list(regions.values()))
    lower, upper = vertices.min(axis=0) - 15, vertices.max(axis=0) + 15
    scale = min(680 / (upper - lower))
    width, height = np.ceil((upper - lower) * scale).astype(int) + 60
    image = np.full((height, width, 3), 32, dtype=np.uint8)

    def pixels(xy):
        p = (np.asarray(xy) - lower) * scale + 30
        p[..., 1] = height - p[..., 1]
        return np.rint(p).astype(int)

    counts = []
    for index, camera in enumerate(calibration['cameras']):
        depth = data.read_image(capture / f'depth/Frame{index}.exr', camera)
        points = data.backproject_depth(depth, camera, calibration['robot_from_vision'], 2)
        points = points[np.all((points[:, :2] >= lower) & (points[:, :2] <= upper), axis=1)]
        counts.append(len(points))
        if len(points):
            points = points[np.argsort(points[:, 2])]
            locations = pixels(points[:, :2])
            valid = np.all((locations >= [0, 0]) & (locations < [width, height]), axis=1)
            z_lower, z_upper = np.quantile(points[:, 2], [.01, .99])
            levels = np.clip((points[:, 2] - z_lower) / max(z_upper - z_lower, 1.) * 255, 0, 255).astype(np.uint8)
            colors = cv2.applyColorMap(levels[:, None], cv2.COLORMAP_TURBO)[:, 0]
            image[locations[valid, 1], locations[valid, 0]] = colors[valid]
    for name, vertices in regions.items():
        polygon = pixels(vertices)
        cv2.polylines(image, [polygon], True, (255, 255, 255), 2)
        origin = polygon.min(axis=0)
        cv2.putText(image, name.replace('_', ' ').upper(), (int(origin[0]) + 8, int(origin[1]) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
    cv2.putText(image, '+X right / +Y up', (12, height - 8), cv2.FONT_HERSHEY_SIMPLEX, .4, (220, 220, 220), 1)
    success, png = cv2.imencode('.png', image)
    if not success:
        raise data.DatasetError('Could not encode the bin preview')
    return {'preview_png': base64.b64encode(png.tobytes()).decode('ascii'),
            'camera_points': counts, 'caption': f"Scene {recorded['scene_id']} in {calibration['frame_id']}: fresh observed depth, viewed from above. White outlines are the configured bin regions."}
