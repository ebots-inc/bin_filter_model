"""Measured bin-local shape, rim registration and camera/exposure error models.

The geometric reference is an empirical consensus of independent empty captures;
absolute sensor bias cannot be identified without independent metrology. Camera
biases below are relative to that fixed consensus, not claimed absolute accuracy.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
import warnings

import numpy as np

import dataset as data
import height_field as hf

MODEL_TYPE = 'calibrated_bin_background'
FIT_KEYS = set(hf.DEFAULTS) | {'kind', 'bin_id', 'xy_bounds_mm', 'z_bounds_mm', 'seed_xy_mm'}
REGISTRATION_DEFAULTS = {
    'max_slope': .35, 'edge_height_drop_mm': .8, 'edge_search_radius_mm': 3.,
    'min_surface_points': 200, 'min_boundary_points': 80, 'min_quadrant_points': 20,
}
CAMERA_DEFAULTS = {
    'min_capture_support': 5, 'min_points_per_cell': 3, 'residual_quantile': .99,
    'min_uncertainty_mm': .2, 'max_uncertainty_mm': 4., 'alignment_margin_mm': .3,
    'max_bias_mm': 10., 'min_profile_cells': 100, 'interpolation_margin_mm': .2,
}


def _nullable(array):
    return hf._nullable(np.asarray(array, dtype=float))


def _views(capture):
    views = capture['views']() if callable(capture['views']) else capture['views']
    by_camera = {view['camera_index']: view for view in views}
    if set(by_camera) != {0, 1} or len(views) != 2:
        raise data.DatasetError('Modern fitting requires the two separate original camera views')
    return by_camera


def _acquisition(capture, camera):
    settings = capture.get('requested_settings', capture.get('source_metadata', {}).get('requested_settings', {}))
    result = {}
    for key in ('gain', 'black_level'):
        value = settings.get(key, capture.get(key))
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                raise data.DatasetError(f'{key} requires one value for each camera')
            value = value[camera]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise data.DatasetError(f'{key} must be finite and nonnegative')
        result[key] = float(value)
    return result


def _parameters(settings, defaults, name):
    if not isinstance(settings, dict) or set(settings) - set(defaults):
        raise data.DatasetError(f'{name} contains unsupported parameters')
    result = defaults | settings
    for key, value in result.items():
        nonnegative = key in ('alignment_margin_mm', 'interpolation_margin_mm')
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                or (value < 0 if nonnegative else value <= 0)):
            raise data.DatasetError(f'{name}.{key} must be finite and positive')
        if key.startswith('min_') and key.endswith(('points', 'cells', 'support')):
            if value != int(value):
                raise data.DatasetError(f'{name}.{key} must be an integer')
            result[key] = int(value)
    if result.get('min_capture_support', 2) < 2:
        raise data.DatasetError('Camera calibration needs at least two independent captures')
    return result


def _local_configuration(config, captures):
    fit = {key: deepcopy(value) for key, value in config.items() if key in FIT_KEYS}
    fit['frame_id'] = 'bin'
    frame = config.get('coordinate_frame', 'robot')
    if frame not in ('robot', 'bin'):
        raise data.DatasetError('coordinate_frame must be robot or bin')
    if 'robot_from_bin' in config:
        placement = data.validate_transform(config['robot_from_bin'], 'robot_from_bin')
    else:
        if frame == 'bin':
            raise data.DatasetError('Bin-local fitting settings require robot_from_bin')
        bounds = np.asarray(fit['xy_bounds_mm'], dtype=float)
        z_bounds = np.asarray(fit['z_bounds_mm'], dtype=float)
        first = np.concatenate([view['points'] for view in _views(captures[0]).values()])
        valid = hf._inside(first, bounds) & (first[:, 2] >= z_bounds[0]) & (first[:, 2] <= z_bounds[1])
        if np.count_nonzero(valid) < 20:
            raise data.DatasetError('Cannot establish bin-local origin from empty measured points')
        seed = fit.get('seed_xy_mm', bounds.mean(axis=0))
        placement = np.eye(4)
        placement[:3, 3] = [*seed, float(np.quantile(first[valid, 2], .05))]
    band = config.get('registration', {}).get('height_band_mm')
    if band is None:
        band = [fit.get('max_reference_height_mm', hf.DEFAULTS['max_reference_height_mm']), fit['z_bounds_mm'][1]]
    band = np.asarray(band, dtype=float)
    if band.shape != (2,) or not np.isfinite(band).all() or band[0] >= band[1]:
        raise data.DatasetError('registration.height_band_mm must be an ascending finite pair')
    if frame == 'robot':
        if not np.allclose(placement[:3, :3], np.eye(3), atol=1e-10):
            raise data.DatasetError('Rotated robot_from_bin requires coordinate_frame: bin and local fitting bounds')
        offset = placement[:3, 3]
        fit['xy_bounds_mm'] = (np.asarray(fit['xy_bounds_mm']) - offset[:2]).tolist()
        fit['seed_xy_mm'] = (np.asarray(fit.get('seed_xy_mm', np.mean(config['xy_bounds_mm'], axis=0))) - offset[:2]).tolist()
        fit['z_bounds_mm'] = (np.asarray(fit['z_bounds_mm']) - offset[2]).tolist()
        fit['max_reference_height_mm'] = fit.get('max_reference_height_mm', hf.DEFAULTS['max_reference_height_mm']) - offset[2]
        band -= offset[2]
    hf._configuration(fit)
    return fit, placement, band


def _crop_views(captures, fit, placement):
    bounds = np.asarray(fit['xy_bounds_mm'], dtype=float)
    inverse = np.linalg.inv(placement)
    local = []
    for capture in captures:
        views = {}
        for camera, view in _views(capture).items():
            points = data.transform_points(np.asarray(view['points'], dtype=float), inverse)
            views[camera] = points[hf._inside(points, bounds)]
        local.append({'capture': capture, 'views': views})
    return local


def _full_reference(clouds, fit):
    params, bounds, z_bounds, _, shape = hf._configuration(fit)
    count = int(np.prod(shape))
    medians = []
    for points in clouds:
        inside = hf._inside(points, bounds) & (points[:, 2] >= z_bounds[0]) & (points[:, 2] <= z_bounds[1])
        selected = points[inside]
        cell = np.floor((selected[:, :2] - bounds[0]) / params['cell_size_mm']).astype(int)
        ids = cell[:, 1] * shape[1] + cell[:, 0]
        medians.append(hf._group_quantile(ids, selected[:, 2], count, .5, params['min_points_per_cell'])[0])
    medians = np.asarray(medians)
    support = np.isfinite(medians).sum(axis=0).reshape(shape)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        surface = np.nanmedian(medians, axis=0).reshape(shape)
    surface[support < params['min_capture_support']] = np.nan
    return surface, support


def _fit_profile(records, camera, exposure, surface, fit, parameters):
    _, bounds, _, _, shape = hf._configuration(fit)
    square_shape = (shape[0] - 1, shape[1] - 1)
    count = int(np.prod(square_shape))
    residuals, residual_ids, medians = [], [], []
    acquisition = None
    for record in records:
        settings = _acquisition(record['capture'], camera)
        if acquisition is None:
            acquisition = settings
        elif acquisition != settings:
            raise data.DatasetError(f'Camera {camera} at {exposure:g} ms changes gain/black_level across training captures')
        points = record['views'][camera]
        prediction, ids = hf._interpolate(points, bounds, fit.get('cell_size_mm', 1.), surface)
        known = np.isfinite(prediction)
        residual = points[known, 2] - prediction[known]
        ids = ids[known]
        residuals.append(residual)
        residual_ids.append(ids)
        medians.append(hf._group_quantile(ids, residual, count, .5, parameters['min_points_per_cell'])[0])
    medians = np.asarray(medians)
    support = np.isfinite(medians).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        bias = np.nanmedian(medians, axis=0)
    quantiles = []
    for ids, residual in zip(residual_ids, residuals):
        quantiles.append(hf._group_quantile(ids, np.abs(residual - bias[ids]), count,
                                            parameters['residual_quantile'], parameters['min_points_per_cell'])[0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        uncertainty = np.nanmax(quantiles, axis=0) + parameters['alignment_margin_mm']
    uncertainty = np.maximum(uncertainty, parameters['min_uncertainty_mm'])
    validity = ((support >= parameters['min_capture_support']) & np.isfinite(bias) & np.isfinite(uncertainty)
                & (np.abs(bias) <= parameters['max_bias_mm']) & (uncertainty <= parameters['max_uncertainty_mm']))
    if np.count_nonzero(validity) < parameters['min_profile_cells']:
        raise data.DatasetError(f'Camera {camera} at {exposure:g} ms has insufficient independently supported surface coverage')
    bias[~validity] = np.nan
    uncertainty[~validity] = np.nan
    return {'camera_index': camera, 'exposure_ms': float(exposure), 'acquisition': acquisition or {},
            'bias_mm': _nullable(bias.reshape(square_shape)), 'uncertainty_mm': _nullable(uncertainty.reshape(square_shape)),
            'validity': validity.reshape(square_shape).astype(int).tolist(),
            'capture_support': support.reshape(square_shape).tolist(),
            'training_capture_ids': [item['capture']['id'] for item in records],
            'diagnostics': {'supported_cells': int(validity.sum()), 'total_cells': count,
                            'bias_quantiles_mm': np.quantile(bias[validity], [0, .5, 1]).tolist(),
                            'uncertainty_quantiles_mm': np.quantile(uncertainty[validity], [0, .5, 1]).tolist()}}


def _profile_arrays(bin_model, camera, exposure, acquisition=None, candidate_interval=False):
    exposure = data.positive_exposure(exposure)
    profiles = sorted((item for item in bin_model['camera_profiles'] if item['camera_index'] == camera),
                      key=lambda item: item['exposure_ms'])
    anchors = np.array([item['exposure_ms'] for item in profiles])
    if not len(anchors):
        raise data.DatasetError(f'No camera {camera} error model')
    exact = np.flatnonzero(np.isclose(anchors, exposure, atol=data.EXPOSURE_ATOL_MS, rtol=0))
    if len(exact):
        lower = upper = profiles[int(exact[0])]
        weight = 0.
    else:
        index = int(np.searchsorted(anchors, exposure))
        if index == 0 or index == len(anchors):
            raise data.DatasetError('Exposure outside measured camera-profile range; extrapolation is forbidden')
        if not candidate_interval and not any(lo <= exposure <= hi for lo, hi in bin_model.get('validated_intervals_ms', [])):
            raise data.DatasetError('Intermediate exposure interval has not passed held-out validation')
        lower, upper = profiles[index - 1], profiles[index]
        weight = (exposure - lower['exposure_ms']) / (upper['exposure_ms'] - lower['exposure_ms'])
    conditions = lower.get('acquisition', {})
    if conditions != upper.get('acquisition', {}):
        raise data.DatasetError('Cannot interpolate camera profiles with different acquisition conditions')
    if conditions:
        if acquisition is None or any(key not in acquisition or not np.isclose(acquisition[key], value, atol=1e-8, rtol=0)
                                      for key, value in conditions.items()):
            raise data.DatasetError(f'Camera {camera} acquisition settings do not match its calibrated profile')
    a, b = np.asarray(lower['bias_mm'], dtype=float), np.asarray(upper['bias_mm'], dtype=float)
    ua, ub = np.asarray(lower['uncertainty_mm'], dtype=float), np.asarray(upper['uncertainty_mm'], dtype=float)
    validity = np.asarray(lower['validity'], dtype=bool) & np.asarray(upper['validity'], dtype=bool)
    bias = (1 - weight) * a + weight * b
    uncertainty = np.maximum(ua, ub)
    if lower is not upper:
        uncertainty += bin_model.get('interpolation_margin_mm', .2)
    validity &= np.isfinite(bias) & np.isfinite(uncertainty)
    return bias, uncertainty, validity


def _correct_local(points, geometry, arrays):
    bias, uncertainty, validity = arrays
    bounds = np.asarray(geometry['xy_bounds_mm'], dtype=float)
    size = float(geometry['cell_size_mm'])
    selected = np.flatnonzero(hf._inside(points, bounds))
    cell = np.floor((points[selected, :2] - bounds[0]) / size - .5).astype(int)
    ny, nx = bias.shape
    supported = (cell[:, 0] >= 0) & (cell[:, 0] < nx) & (cell[:, 1] >= 0) & (cell[:, 1] < ny)
    selected, cell = selected[supported], cell[supported]
    supported = validity[cell[:, 1], cell[:, 0]]
    selected, cell = selected[supported], cell[supported]
    corrected = points[selected].copy()
    corrected[:, 2] -= bias[cell[:, 1], cell[:, 0]]
    return corrected, selected, uncertainty[cell[:, 1], cell[:, 0]]


def _measured_rim(surface, support, fit, band, settings):
    params = _parameters({key: value for key, value in settings.items() if key != 'height_band_mm'}, REGISTRATION_DEFAULTS, 'registration')
    cell_size = float(fit.get('cell_size_mm', 1.))
    observed = np.isfinite(surface)
    # A flat-lip sample must have measured neighbors. Missing data never creates an edge.
    gradient_y, gradient_x = np.gradient(surface, cell_size)
    slope = np.hypot(gradient_x, gradient_y)
    flat = observed & np.isfinite(slope) & (surface >= band[0]) & (surface <= band[1]) & (slope <= params['max_slope'])
    distance = np.full(flat.shape, np.inf)
    radius = max(1, int(math.ceil(params['edge_search_radius_mm'] / cell_size)))
    ny, nx = flat.shape
    for dy, dx in [(0, offset) for offset in range(-radius, radius + 1) if offset] + [(offset, 0) for offset in range(-radius, radius + 1) if offset]:
        y0, y1 = max(0, -dy), min(ny, ny - dy)
        x0, x1 = max(0, -dx), min(nx, nx - dx)
        current = surface[y0:y1, x0:x1]
        neighbor = surface[y0+dy:y1+dy, x0+dx:x1+dx]
        # Inner/outer physical drops are inferred only from observed lower geometry.
        drop = np.isfinite(neighbor) & (current - neighbor >= params['edge_height_drop_mm'])
        distances = distance[y0:y1, x0:x1]
        distances[drop] = np.minimum(distances[drop], abs(dy) + abs(dx))
    distance[~flat] = np.inf
    boundary = np.isfinite(distance)
    # Keep the nearest stable measured lip row, rather than a thick band of
    # interchangeable planar samples. Ties retain the observed edge support.
    for dy, dx in ((-1,0), (1,0), (0,-1), (0,1)):
        y0, y1 = max(0,-dy), min(ny,ny-dy)
        x0, x1 = max(0,-dx), min(nx,nx-dx)
        boundary[y0:y1,x0:x1] &= distance[y0:y1,x0:x1] <= distance[y0+dy:y1+dy,x0+dx:x1+dx]
    bounds = np.asarray(fit['xy_bounds_mm'], dtype=float)
    seed = np.asarray(fit['seed_xy_mm'], dtype=float)
    def measured(mask):
        yy, xx = np.nonzero(mask)
        return np.column_stack((bounds[0, 0] + (xx + .5) * cell_size,
                                bounds[0, 1] + (yy + .5) * cell_size, surface[mask]))
    points, edges = measured(flat), measured(boundary)
    def quadrant_count(values):
        q = (values[:, 0] >= seed[0]).astype(int) + 2 * (values[:, 1] >= seed[1]).astype(int)
        return np.bincount(q, minlength=4)
    if len(points) < params['min_surface_points'] or len(edges) < params['min_boundary_points']:
        raise data.DatasetError(f'Measured rim is insufficient: {len(points)} flat-lip and {len(edges)} observed-boundary points; '
                                'capture the physical rim or review its measured height band')
    q_surface, q_boundary = quadrant_count(points), quadrant_count(edges)
    if min(q_surface.min(), q_boundary.min()) < params['min_quadrant_points']:
        raise data.DatasetError('Measured rim/boundary lacks independent support in all four quadrants')
    return {'surface_xyz_mm': points.tolist(), 'boundary_xyz_mm': edges.tolist(),
            'source': 'measured_empty_captures',
            'method': 'flat_lip_nearest_observed_height_drop',
            'height_band_mm': band.tolist(), 'parameters': params,
            'diagnostics': {'surface_points': len(points), 'boundary_points': len(edges),
                            'surface_quadrants': q_surface.tolist(), 'boundary_quadrants': q_boundary.tolist(),
                            'measured_min_mm': points.min(axis=0).tolist(), 'measured_max_mm': points.max(axis=0).tolist(),
                            'minimum_capture_support': int(support[flat].min()),
                            'boundary_interpretation': 'measured lip samples next to an observed lower surface; no edges synthesized from missing data'}}


def fit_model(captures, settings, frame_id='B2', model_name='bin-model'):
    """Fit the production model from camera-separated all-empty training captures."""
    if not isinstance(model_name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', model_name):
        raise data.DatasetError('model_name must be a readable filesystem-safe name')
    expected_bins = data.validate_bin_ids(settings.get('bins'))
    for capture in captures:
        data.validate_bin_labels(capture.get('bins'), expected_bins)
    if 'frame_id' in settings and settings['frame_id'] != frame_id:
        raise data.DatasetError('Bin configuration frame differs from the calibrated robot frame')
    train = [capture for capture in captures if capture['split'] == 'train' and
             all(label == 'empty' for label in capture['bins'].values())]
    if len(train) < 3:
        raise data.DatasetError('At least three independent all-empty camera captures are required')
    if len({capture['id'] for capture in train}) != len(train):
        raise data.DatasetError('Duplicate training capture IDs')
    exposures = sorted({data.capture_exposure(capture) for capture in train})
    model = {'model_type': MODEL_TYPE, 'model_name': model_name, 'frame_id': frame_id, 'length_unit': 'mm',
             'scene_id': settings.get('scene_id', 0), 'bins': {},
             'training_captures': [{key: capture[key] for key in ('id', 'group_id', 'input_signature') if key in capture}
                                   for capture in train],
             'validation_status': 'requires_held_out_production_review',
             'geometry_reference': 'fixed_measured_empty_capture_consensus; sensor bias is relative, not absolute metrology'}
    for name, config in settings['bins'].items():
        fit, placement, band = _local_configuration(config, train)
        local = _crop_views(train, fit, placement)
        merged = [np.concatenate(list(record['views'].values())) for record in local]
        full_surface, _ = _full_reference(merged, fit)
        camera_config = {'min_capture_support': fit.get('min_capture_support', 5),
                         'min_points_per_cell': fit.get('min_points_per_cell', 3)} | config.get('camera_model', {})
        camera_parameters = _parameters(camera_config, CAMERA_DEFAULTS, 'camera_model')
        if not .9 <= camera_parameters['residual_quantile'] <= 1:
            raise data.DatasetError('camera_model.residual_quantile must be within [0.9,1]')
        if camera_parameters['min_uncertainty_mm'] > camera_parameters['max_uncertainty_mm']:
            raise data.DatasetError('Minimum camera uncertainty exceeds the maximum')
        profiles = []
        for camera in (0, 1):
            for exposure in exposures:
                records = [item for item in local if abs(data.capture_exposure(item['capture']) - exposure) <= data.EXPOSURE_ATOL_MS]
                if len(records) < camera_parameters['min_capture_support']:
                    raise data.DatasetError(f'{name}: exposure {exposure:g} ms lacks independent camera calibration repeats')
                profiles.append(_fit_profile(records, camera, exposure, full_surface, fit, camera_parameters))
        geometry_lookup = {'xy_bounds_mm': fit['xy_bounds_mm'], 'cell_size_mm': fit.get('cell_size_mm', 1.)}
        provisional = {'geometry': geometry_lookup, 'camera_profiles': profiles, 'supported_exposure_ms': [exposures[0], exposures[-1]],
                       'validated_intervals_ms': [], 'interpolation_margin_mm': camera_parameters['interpolation_margin_mm']}
        corrected = []
        for record in local:
            exposure = data.capture_exposure(record['capture'])
            corrected.append(np.concatenate([_correct_local(points, geometry_lookup,
                _profile_arrays(provisional, camera, exposure, _acquisition(record['capture'], camera)))[0]
                for camera, points in record['views'].items()]))
        geometry = hf.fit_bin(corrected, fit)
        # Even after camera bias correction, discretizing a curved reference can
        # introduce measurable residuals. Keep their empirical envelope instead
        # of replacing it with an assumed zero-error geometric surface.
        geometry['tolerance_role'] = 'measured_post_camera_correction_geometry_residual; runtime threshold is max with camera uncertainty'
        corrected_surface, support = _full_reference(corrected, fit)
        registration = _measured_rim(corrected_surface, support, fit, band, config.get('registration', {}))
        model['bins'][name] = provisional | {'geometry': geometry, 'robot_from_bin': placement.tolist(),
            'shape_name': config.get('shape_name', f'{model_name}-{name}-measured'), 'registration': registration,
            'camera_model_parameters': camera_parameters}
    validate_model(model)
    return model


def validate_model(model):
    if not isinstance(model, dict) or model.get('model_type') != MODEL_TYPE or model.get('length_unit') != 'mm':
        raise data.DatasetError('Unsupported calibrated bin model')
    if not isinstance(model.get('frame_id'), str) or not model['frame_id'].strip():
        raise data.DatasetError('Calibrated model requires a robot frame')
    if not isinstance(model.get('model_name'), str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', model['model_name']):
        raise data.DatasetError('Calibrated model requires a readable model_name')
    if type(model.get('scene_id')) is not int or model['scene_id'] < 0:
        raise data.DatasetError('Calibrated model requires a nonnegative scene_id')
    training = model.get('training_captures')
    if not isinstance(training,list) or len(training)<3 or any(not isinstance(item,dict) or not isinstance(item.get('id'),str)
                                                              or not item['id'] for item in training):
        raise data.DatasetError('Calibrated model requires independent training capture identities')
    training_ids = {item['id'] for item in training}
    if len(training_ids) != len(training):
        raise data.DatasetError('Calibrated model training capture IDs must be unique')
    data.validate_bin_ids(model.get('bins'))
    robot_bounds = []
    for name, item in model['bins'].items():
        placement = data.validate_transform(item['robot_from_bin'], f'{name}.robot_from_bin')
        geometry = item['geometry']
        if geometry.get('frame_id') != 'bin':
            raise data.DatasetError('Modern geometry must be bin-local')
        _, bounds, _, _ = hf._validated_model(geometry)
        if geometry.get('diagnostics', {}).get('capture_count') != len(training):
            raise data.DatasetError('Geometry support count differs from model training captures')
        shape = np.asarray(geometry['tolerance_mm'], dtype=float).shape
        camera_parameters = _parameters(item.get('camera_model_parameters', {}), CAMERA_DEFAULTS, 'camera_model')
        profiles = item.get('camera_profiles', [])
        seen = set()
        anchors = {0: [], 1: []}
        for profile in profiles:
            camera = profile['camera_index']
            exposure = data.positive_exposure(profile['exposure_ms'])
            if type(camera) is not int or camera not in (0, 1) or (camera, exposure) in seen:
                raise data.DatasetError('Camera profiles need unique camera/exposure pairs')
            seen.add((camera, exposure)); anchors[camera].append(exposure)
            conditions = profile.get('acquisition', {})
            if not isinstance(conditions, dict) or (conditions and set(conditions) != {'gain', 'black_level'}):
                raise data.DatasetError('Camera acquisition must record both gain and black_level, or explicitly remain unknown')
            if any(isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value) or value < 0
                   for value in conditions.values()):
                raise data.DatasetError('Camera acquisition requires finite nonnegative gain/black_level')
            bias = np.asarray(profile['bias_mm'], dtype=float)
            uncertainty = np.asarray(profile['uncertainty_mm'], dtype=float)
            validity = np.asarray(profile['validity'])
            support = np.asarray(profile['capture_support'], dtype=float)
            if any(values.shape != shape for values in (bias, uncertainty, validity, support)):
                raise data.DatasetError('Camera profile grid differs from geometry')
            if np.isinf(bias).any() or np.isinf(uncertainty).any() or not np.isin(validity, [0, 1]).all():
                raise data.DatasetError('Camera profile contains invalid bias/uncertainty/validity')
            valid = validity.astype(bool)
            if not valid.any() or not np.isfinite(bias[valid]).all() or not np.isfinite(uncertainty[valid]).all() or np.any(uncertainty[valid] <= 0):
                raise data.DatasetError('Supported camera profile cells require finite bias and positive uncertainty')
            if (np.any(np.abs(bias[valid]) > camera_parameters['max_bias_mm'])
                    or np.any(uncertainty[valid] < camera_parameters['min_uncertainty_mm'])
                    or np.any(uncertainty[valid] > camera_parameters['max_uncertainty_mm'])):
                raise data.DatasetError('Supported camera profile cells violate configured error bounds')
            if not np.isfinite(support).all() or np.any(support < 0) or np.any(support != np.floor(support)):
                raise data.DatasetError('Camera profile capture support must be nonnegative integers')
            ids = profile.get('training_capture_ids')
            if not isinstance(ids,list) or not all(isinstance(value,str) and value for value in ids) or len(ids) != len(set(ids)):
                raise data.DatasetError('Camera profile requires unique independent training capture IDs')
            if not set(ids) <= training_ids:
                raise data.DatasetError('Camera profile names captures outside the model training set')
            minimum = camera_parameters['min_capture_support']
            if np.any(support[valid] < minimum) or np.any(support > len(ids)):
                raise data.DatasetError('Camera profile support violates independent training support')
        if not anchors[0] or sorted(anchors[0]) != sorted(anchors[1]):
            raise data.DatasetError('Both cameras must have the same measured exposure anchors')
        if np.any(np.diff(sorted(anchors[0])) <= data.EXPOSURE_ATOL_MS):
            raise data.DatasetError('Camera exposure anchors must be separated by more than the exposure matching tolerance')
        expected_range = [min(anchors[0]), max(anchors[0])]
        if item.get('supported_exposure_ms') != expected_range:
            raise data.DatasetError('Supported exposure range must match the measured anchors')
        for interval in item.get('validated_intervals_ms', []):
            if (not isinstance(interval, list) or len(interval) != 2 or not np.isfinite(interval).all()
                    or interval[0] >= interval[1] or interval[0] < expected_range[0] or interval[1] > expected_range[1]):
                raise data.DatasetError('Validated exposure intervals must lie within measured anchors')
        registration = item['registration']
        registration_parameters = _parameters(registration.get('parameters', {}), REGISTRATION_DEFAULTS, 'registration')
        if registration.get('source') != 'measured_empty_captures':
            raise data.DatasetError('Registration reference must use measured empty captures')
        band = np.asarray(registration.get('height_band_mm'), dtype=float)
        if band.shape != (2,) or not np.isfinite(band).all() or band[0] >= band[1]:
            raise data.DatasetError('Registration requires a finite ascending local height band')
        for field in ('surface_xyz_mm', 'boundary_xyz_mm'):
            points = np.asarray(registration[field], dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
                raise data.DatasetError(f'{field} must contain finite measured local XYZ points')
            minimum = registration_parameters['min_surface_points' if field == 'surface_xyz_mm' else 'min_boundary_points']
            if (len(points) < minimum or not hf._inside(points,bounds).all()
                    or np.any(points[:,2] < band[0]) or np.any(points[:,2] > band[1])):
                raise data.DatasetError('Measured registration reference violates point-count or local bounds')
            seed = np.asarray(geometry['seed_xy_mm'])
            quadrants = (points[:,0] >= seed[0]).astype(int) + 2*(points[:,1] >= seed[1]).astype(int)
            if np.bincount(quadrants,minlength=4).min() < registration_parameters['min_quadrant_points']:
                raise data.DatasetError('Measured registration reference lacks four-quadrant coverage')
        # Disjoint world XY envelopes prevent silently assigning one observation to two bins.
        z0, z1 = geometry['z_bounds_mm']
        corners = np.array([[x, y, z] for x in bounds[:, 0] for y in bounds[:, 1] for z in (z0, z1)])
        world = data.transform_points(corners, placement)
        robot_bounds.append((world[:, :2].min(axis=0), world[:, :2].max(axis=0)))
    for index, (alo, ahi) in enumerate(robot_bounds):
        for blo, bhi in robot_bounds[index + 1:]:
            if np.all(np.minimum(ahi, bhi) - np.maximum(alo, blo) > 1e-6):
                raise data.DatasetError('Placed bin XY envelopes overlap; revise measured placements/bounds')
    return model


def correct_views(model, bin_id, views, exposure_ms, acquisition=None, candidate_interval=False):
    """Correct each camera before merging; retain original coordinates and indices."""
    item = model['bins'][bin_id]
    placement = data.validate_transform(item['robot_from_bin'])
    inverse = np.linalg.inv(placement)
    result = []
    for view in views:
        camera = view['camera_index']
        conditions = acquisition.get(camera, {}) if isinstance(acquisition, dict) else None
        if conditions is None and 'acquisition' in view:
            conditions = view['acquisition']
        arrays = _profile_arrays(item, camera, exposure_ms, conditions, candidate_interval)
        original = np.asarray(view['points'], dtype=float)
        local = data.transform_points(original, inverse)
        roi = hf._inside(local, np.asarray(item['geometry']['xy_bounds_mm'], dtype=float))
        corrected, indices, uncertainty = _correct_local(local, item['geometry'], arrays)
        result.append({**{key: value for key, value in view.items() if key != 'points'},
                       'points': data.transform_points(corrected, placement), 'local_points': corrected,
                       'original_points': original[indices], 'source_indices': indices,
                       'input_roi_mask': roi,
                       'source_point_count': len(original), 'uncertainty_mm': uncertainty,
                       'invalid_measurement_count': int(roi.sum()) - len(indices)})
    return result


def classify_views(model, bin_id, views, exposure_ms, acquisition=None, candidate_interval=False):
    """Classify and return bias-corrected measurements, preserving provenance."""
    corrected = correct_views(model, bin_id, views, exposure_ms, acquisition, candidate_interval)
    geometry = model['bins'][bin_id]['geometry']
    parameters, bounds, heights, tolerance = hf._validated_model(geometry)
    results = []
    for view in corrected:
        points = view['local_points']
        prediction, ids = hf._interpolate(points, bounds, parameters['cell_size_mm'], heights)
        thresholds = np.full(len(points), np.nan)
        addressable = ids >= 0
        thresholds[addressable] = tolerance.ravel()[ids[addressable]]
        thresholds = np.maximum(thresholds, view['uncertainty_mm'])
        known = np.isfinite(prediction) & np.isfinite(thresholds)
        clearance = points[:, 2] - prediction
        keep = known & (clearance > thresholds) & (clearance <= parameters['max_height_above_surface_mm'])
        surface = known & (np.abs(clearance) <= thresholds)
        results.append(view | {'keep': keep, 'surface': surface, 'unknown': ~known,
                               'clearance_mm': clearance, 'threshold_mm': thresholds,
                               'kept_original_points': view['original_points'][keep]})
    kept = np.concatenate([view['points'][view['keep']] for view in results]) if results else np.empty((0, 3))
    # Public masks follow concatenated ORIGINAL input order, including invalid
    # camera measurements. Only diagnostic arrays inside views use valid subsets.
    masks = {name: [] for name in ('kept', 'surface', 'unknown', 'outside', 'invalid')}
    corrected_all = []
    original_all = []
    for view, result in zip(views, results):
        original = np.asarray(view['points'], dtype=float)
        current = {name: np.zeros(len(original), dtype=bool) for name in masks}
        current['unknown'][:] = result['input_roi_mask']
        current['outside'][:] = np.isfinite(original).all(axis=1) & ~result['input_roi_mask']
        indices = result['source_indices']
        aligned = np.full(original.shape, np.nan)
        aligned[indices] = result['points']
        corrected_all.append(aligned)
        original_all.append(original)
        current['kept'][indices] = result['keep']
        current['surface'][indices] = result['surface']
        current['unknown'][indices] = result['unknown']
        current['outside'][indices] = ~(result['keep'] | result['surface'] | result['unknown'])
        invalid = ~np.isfinite(original).all(axis=1)
        current['invalid'][invalid] = True
        current['unknown'][invalid] = False
        for name in masks:
            masks[name].append(current[name])
    masks = {name: np.concatenate(values) if values else np.empty(0, dtype=bool) for name, values in masks.items()}
    return {'points': kept, 'views': results, 'keep': masks['kept'], 'masks': masks,
            'corrected_points': np.concatenate(corrected_all) if corrected_all else np.empty((0,3)),
            'original_points': np.concatenate(original_all) if original_all else np.empty((0,3)),
            'camera_diagnostics': [{'camera_index': view['camera_index'],
                                    'input_points': view['source_point_count'],
                                    'input_roi_points': int(view['input_roi_mask'].sum()),
                                    'corrected_points': len(view['points']),
                                    'invalid_measurement_count': view['invalid_measurement_count'],
                                    'foreground_points': int(view['keep'].sum())} for view in results],
            'counts': {'kept': len(kept), 'surface': sum(int(view['surface'].sum()) for view in results),
                       'unknown': sum(int(view['unknown'].sum()) + view['invalid_measurement_count'] for view in results)}}


def extract_shape(model, bin_id):
    """Export reusable local geometry without camera/site-specific calibration."""
    item = model['bins'][bin_id]
    return deepcopy({key: item[key] for key in ('shape_name', 'geometry', 'registration')})
