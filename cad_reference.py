"""Optional nominal CAD reference, commissioned against empty-bin measurements.

STL has no units or coordinate convention: both scale and placement are explicit.
The reference is a top-visible height envelope, never a replacement for observed
support. Its residual contains physical shape differences *and* unresolved sensor
error; it is not manufacturing metrology.
"""
from __future__ import annotations

from pathlib import Path
import math
import re

import numpy as np

from dataset import DatasetError, validate_transform


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise DatasetError(f'{name} must be finite and positive')
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise DatasetError(f'{name} must be finite and positive') from error
    if not np.isfinite(number) or number <= 0:
        raise DatasetError(f'{name} must be finite and positive')
    return number


def _fraction(value, name):
    number = _positive(value, name)
    if number > 1:
        raise DatasetError(f'{name} must be within (0, 1]')
    return number


def read_stl(path, scale_to_mm):
    """Return finite triangle vertices in millimetres from binary or ASCII STL."""
    scale = _positive(scale_to_mm, 'CAD scale_to_mm')
    path = Path(path)
    if path.stat().st_size > 128 * 1024 * 1024:
        raise DatasetError('CAD STL exceeds the 128 MiB offline import limit')
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], 'little') if len(raw) >= 84 else 0
    if count and len(raw) == 84 + 50 * count:
        records = np.frombuffer(raw, offset=84, count=count,
            dtype=np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]))
        triangles = records['vertices'].astype(np.float64)
    else:
        try:
            text = raw.decode('ascii')
        except UnicodeDecodeError as error:
            raise DatasetError('CAD is not a complete binary or ASCII STL') from error
        if not re.match(r'\s*solid(?:\s|$)', text, re.IGNORECASE):
            raise DatasetError('CAD is not a complete binary or ASCII STL')
        vertices = re.findall(r'^\s*vertex\s+(\S+)\s+(\S+)\s+(\S+)\s*$', text,
                              re.MULTILINE | re.IGNORECASE)
        if not vertices or len(vertices) % 3:
            raise DatasetError('ASCII STL must contain complete triangle vertices')
        try:
            triangles = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
        except ValueError as error:
            raise DatasetError('ASCII STL vertices must be numeric') from error
    with np.errstate(over='ignore', invalid='ignore'):
        triangles *= scale
    if not len(triangles) or not np.isfinite(triangles).all():
        raise DatasetError('CAD STL requires finite triangle coordinates')
    if not np.any(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1) > 1e-12):
        raise DatasetError('CAD STL has no nondegenerate triangles')
    return triangles


def _centers(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or len(result) < 2 or not np.isfinite(result).all() or np.any(np.diff(result) <= 0):
        raise DatasetError(f'CAD {name} must be finite ascending grid centers')
    return result


def rasterize(triangles_mm, bin_from_cad, x_centers_mm, y_centers_mm):
    """Rasterize the highest triangle intersection at each XY grid center.

    Selecting the highest intersection excludes the underside of a closed bowl.
    Face winding and unreliable STL stored normals do not determine visibility.
    Vertical triangles have zero projected area and supply no invented heights.
    """
    transform = validate_transform(bin_from_cad, 'CAD bin_from_cad')
    triangles = np.asarray(triangles_mm, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
        raise DatasetError('CAD triangles must be finite Nx3x3 coordinates')
    x, y = _centers(x_centers_mm, 'x_centers_mm'), _centers(y_centers_mm, 'y_centers_mm')
    if len(x) * len(y) > 1_000_000:
        raise DatasetError('CAD raster exceeds one million cells')
    transformed = triangles @ transform[:3, :3].T + transform[:3, 3]
    heights = np.full((len(y), len(x)), -np.inf)
    for triangle in transformed:
        a, b, c = triangle
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) <= 1e-12:
            continue
        lo, hi = triangle[:, :2].min(axis=0), triangle[:, :2].max(axis=0)
        x0, x1 = np.searchsorted(x, lo[0], 'left'), np.searchsorted(x, hi[0], 'right')
        y0, y1 = np.searchsorted(y, lo[1], 'left'), np.searchsorted(y, hi[1], 'right')
        if x0 == x1 or y0 == y1:
            continue
        xx, yy = np.meshgrid(x[x0:x1], y[y0:y1])
        u = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
        v = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
        inside = (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
        z = u * a[2] + v * b[2] + (1 - u - v) * c[2]
        destination = heights[y0:y1, x0:x1]
        np.maximum(destination, np.where(inside, z, -np.inf), out=destination)
    heights[~np.isfinite(heights)] = np.nan
    return heights


def _grid_points(surface, x, y, mask):
    yy, xx = np.nonzero(mask)
    return np.column_stack((x[xx], y[yy], surface[yy, xx]))


def _bounded_alignment(triangles, initial, x, y, measured, observed, settings):
    """Refine a commissioned placement; never discover orientation or scale."""
    allowed = {'enabled', 'max_translation_mm', 'max_rotation_deg',
               'max_correspondence_distance_mm', 'max_iterations', 'min_overlap_fraction'}
    if set(settings) - allowed:
        raise DatasetError('CAD alignment contains unsupported parameters')
    translation_limit = _positive(settings.get('max_translation_mm', 5), 'CAD max_translation_mm')
    rotation_limit = _positive(settings.get('max_rotation_deg', 5), 'CAD max_rotation_deg')
    if rotation_limit >= 45:
        raise DatasetError('CAD max_rotation_deg must be below 45 to preserve the seeded symmetric orientation')
    distance = _positive(settings.get('max_correspondence_distance_mm', 5), 'CAD max_correspondence_distance_mm')
    iterations = _positive(settings.get('max_iterations', 50), 'CAD max_iterations')
    if iterations != int(iterations) or iterations > 1000:
        raise DatasetError('CAD max_iterations must be an integer no greater than 1000')
    minimum = _fraction(settings.get('min_overlap_fraction', .5), 'CAD alignment min_overlap_fraction')
    try:
        import open3d as o3d
    except ImportError as error:
        raise DatasetError('Optional CAD alignment requires Open3D in this Python environment; '
                           'install open3d or commission bin_from_cad explicitly with alignment disabled') from error
    nominal = rasterize(triangles, initial, x, y)
    # Use only independently observed XY cells. Missing measurements cannot
    # supply correspondences or increase registration support.
    common = observed & np.isfinite(nominal)
    if np.count_nonzero(common) < 20:
        raise DatasetError('CAD alignment needs at least 20 supported overlapping empty-bin cells')
    source_points = _grid_points(nominal, x, y, common)
    target_points = _grid_points(measured, x, y, observed)
    # Deterministic bounded subsampling keeps setup cost independent of raw camera density.
    source_points = source_points[::max(1, math.ceil(len(source_points) / 20000))]
    target_points = target_points[::max(1, math.ceil(len(target_points) / 50000))]
    source, target = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    target.points = o3d.utility.Vector3dVector(target_points)
    result = o3d.pipelines.registration.registration_icp(source, target, distance, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iterations)))
    delta = validate_transform(result.transformation, 'CAD alignment correction')
    # Translation is displacement at the commissioned CAD origin, not at world
    # zero; this remains meaningful when a CAD uses a distant native origin.
    final = delta @ initial
    translation = float(np.linalg.norm(final[:3, 3] - initial[:3, 3]))
    angle = math.degrees(math.acos(float(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))))
    if translation > translation_limit or angle > rotation_limit:
        raise DatasetError(f'CAD alignment exceeds commissioned limits: {translation:.3f} mm, {angle:.3f} deg')
    if not np.isfinite(result.fitness) or result.fitness < minimum:
        raise DatasetError('CAD alignment has insufficient correspondence coverage')
    return final, {'method': 'bounded_rigid_point_to_point_empty_training',
                   'translation_mm': translation, 'rotation_deg': angle,
                   'correspondence_fraction': float(result.fitness),
                   'correspondence_rmse_mm': float(result.inlier_rmse)}


def prepare(config, x_centers_mm, y_centers_mm, measured_surface, support, min_capture_support):
    """Prepare a nominal reference without modifying measurements or support.

    Callers must keep their measured reliability mask and intersect any CAD
    selection with it. ``footprint`` describes only CAD geometry, not evidence.
    """
    if not isinstance(config, dict):
        raise DatasetError('CAD reference configuration must be a mapping')
    allowed = {'mesh_path', 'scale_to_mm', 'bin_from_cad', 'alignment', 'min_overlap_fraction'}
    if set(config) - allowed:
        raise DatasetError('CAD reference contains unsupported parameters')
    if any(key not in config for key in ('mesh_path', 'scale_to_mm', 'bin_from_cad')):
        raise DatasetError('CAD requires mesh_path, explicit scale_to_mm, and bin_from_cad')
    transform = validate_transform(config['bin_from_cad'], 'CAD bin_from_cad')
    x, y = _centers(x_centers_mm, 'x_centers_mm'), _centers(y_centers_mm, 'y_centers_mm')
    measured = np.asarray(measured_surface, dtype=float)
    counts = np.asarray(support, dtype=float)
    minimum_support = _positive(min_capture_support, 'CAD min_capture_support')
    if minimum_support != int(minimum_support) or minimum_support < 2:
        raise DatasetError('CAD min_capture_support must be an integer of at least two')
    if (measured.shape != (len(y), len(x)) or counts.shape != measured.shape
            or np.isinf(measured).any() or not np.isfinite(counts).all()
            or np.any(counts < 0) or np.any(counts != np.floor(counts))):
        raise DatasetError('CAD reference requires matching measured height and capture support grids')
    observed = np.isfinite(measured) & (counts >= minimum_support)
    if not observed.any():
        raise DatasetError('CAD reference has no independently supported empty-bin measurements')
    triangles = read_stl(config['mesh_path'], config['scale_to_mm'])
    alignment = config.get('alignment', {})
    if not isinstance(alignment, dict) or type(alignment.get('enabled', False)) is not bool:
        raise DatasetError('CAD alignment must be a mapping with boolean enabled')
    if alignment.get('enabled', False):
        transform, alignment_report = _bounded_alignment(triangles, transform, x, y, measured, observed, alignment)
    else:
        if set(alignment) - {'enabled'}:
            raise DatasetError('Disabled CAD alignment accepts only enabled: false')
        alignment_report = {'method': 'explicit_commissioned_transform'}
    nominal = rasterize(triangles, transform, x, y)
    footprint = np.isfinite(nominal)
    common = observed & footprint
    # Observations outside the footprint can be table/background. Therefore the
    # denominator is CAD footprint cells, not all pixels in the rectangular ROI.
    overlap = float(common.sum() / max(1, footprint.sum()))
    minimum = _fraction(config.get('min_overlap_fraction', .5), 'CAD min_overlap_fraction')
    if not footprint.any() or overlap < minimum:
        raise DatasetError(f'CAD placement has insufficient measured footprint overlap ({overlap:.1%}); '
                           'check units, axes, commissioned transform, and empty-bin coverage')
    residual = measured[common] - nominal[common]
    diagnostics = {'triangle_count': int(len(triangles)), 'nominal_cells': int(footprint.sum()),
        'observed_overlap_cells': int(common.sum()), 'observed_overlap_fraction': overlap,
        'signed_residual_quantiles_mm': np.quantile(residual, [0, .5, .95, 1]).tolist(),
        'absolute_residual_p95_mm': float(np.quantile(np.abs(residual), .95)),
        'residual_interpretation': 'measured_minus_nominal_includes_physical_shape_and_unresolved_sensor_error',
        'alignment': alignment_report}
    return {'nominal_height_mm': nominal, 'footprint': footprint,
            'bin_from_cad': transform, 'diagnostics': diagnostics}
