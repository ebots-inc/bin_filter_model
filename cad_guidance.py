"""Constrain measured bin associations with an optional nominal CAD reference.

CAD never supplies missing observations or raises the foreground threshold.
Apparent shape differences include unresolved sensor bias as well as the bin's
physical deviation; these are diagnostics, not manufacturing measurements.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

import cad_reference as cad
from dataset import DatasetError, validate_transform
import height_field as hf


DEFAULTS = {'association_margin_mm': 3., 'min_measured_retained_fraction': .9,
            'max_reference_residual_mm': 10.}
REFERENCE_KEYS = {'mesh_path', 'scale_to_mm', 'bin_from_cad', 'alignment', 'min_overlap_fraction'}


def _parameters(config):
    if not isinstance(config, dict) or set(config) - REFERENCE_KEYS - set(DEFAULTS):
        raise DatasetError('CAD reference contains unsupported parameters')
    result = {key: config.get(key, value) for key, value in DEFAULTS.items()}
    for key, value in result.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                or (value < 0 if key == 'association_margin_mm' else value <= 0)):
            raise DatasetError(f'CAD {key} must be finite and positive (association margin may be zero)')
    if result['min_measured_retained_fraction'] > 1:
        raise DatasetError('CAD min_measured_retained_fraction must be within (0, 1]')
    return result


def _dilate(mask, radius):
    """Conservative XY association margin; it is never a Z detection tolerance."""
    result = np.zeros_like(mask, dtype=bool)
    ny, nx = mask.shape
    for dy in range(-min(radius, ny - 1), min(radius, ny - 1) + 1):
        for dx in range(-min(radius, nx - 1), min(radius, nx - 1) + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            y0, y1 = max(0, -dy), min(ny, ny - dy)
            x0, x1 = max(0, -dx), min(nx, nx - dx)
            result[y0:y1, x0:x1] |= mask[y0+dy:y1+dy, x0+dx:x1+dx]
    return result


def _retained(before, after, minimum, name):
    fraction = float(np.count_nonzero(after) / max(1, np.count_nonzero(before)))
    if fraction < minimum:
        raise DatasetError(f'CAD association retains only {fraction:.1%} of measured {name}; '
                           'review the mesh, units, placement, and association margin')
    return fraction


def guide(geometry, surface, support, fit, band, registration_settings, config):
    """Return measured geometry, CAD diagnostics and a nominal rim association mask."""
    parameters = _parameters(config)
    result = cad.prepare({key: value for key, value in config.items() if key in REFERENCE_KEYS},
                         geometry['x_centers_mm'], geometry['y_centers_mm'], surface, support,
                         geometry['fit_parameters']['min_capture_support'])
    diagnostics = result['diagnostics']
    if diagnostics['absolute_residual_p95_mm'] > parameters['max_reference_residual_mm']:
        raise DatasetError('CAD reference differs excessively from empty-bin measurements; '
                           'check units, axes, placement, and whether this is the correct bin CAD')
    cell_size = geometry['cell_size_mm']
    radius = int(np.ceil(parameters['association_margin_mm'] / cell_size))
    # Bound work for accidental metre/mm errors in a margin, independently of
    # the actual CAD size and without imposing detection-tolerance assumptions.
    if radius > 50:
        raise DatasetError('CAD association margin exceeds 50 grid cells; review its units')
    association = _dilate(result['footprint'], radius)
    nominal = result['nominal_height_mm']
    gy, gx = np.gradient(nominal, cell_size)
    nominal_lip = (np.isfinite(nominal) & (np.hypot(gx, gy) <= registration_settings.get('max_slope', .35))
                   & (nominal >= band[0] - parameters['association_margin_mm'])
                   & (nominal <= band[1] + parameters['association_margin_mm']))
    rim_mask = _dilate(nominal_lip, radius)

    guided = deepcopy(geometry)
    heights = np.asarray(geometry['surface_height_mm'], dtype=float)
    known_before = np.isfinite(heights)
    seed = (np.asarray(geometry['seed_xy_mm']) - np.asarray(geometry['xy_bounds_mm'])[0]) / cell_size
    vertex = np.floor(seed).astype(int)
    known = hf._connected(known_before & association, (int(vertex[1]), int(vertex[0])), 'CAD-associated reference cells')
    tolerance = np.asarray(geometry['tolerance_mm'], dtype=float)
    reliable_before = np.isfinite(tolerance)
    corners = known[:-1, :-1] & known[:-1, 1:] & known[1:, :-1] & known[1:, 1:]
    square = np.floor(seed - .5).astype(int)
    reliable = hf._connected(reliable_before & corners, (int(square[1]), int(square[0])), 'CAD-associated interpolation squares')
    minimum = parameters['min_measured_retained_fraction']
    diagnostics['measured_geometry_retained_fraction'] = _retained(known_before, known, minimum, 'geometry cells')
    diagnostics['measured_reliable_area_retained_fraction'] = _retained(reliable_before, reliable, minimum, 'reliable area')
    heights[~known] = np.nan
    tolerance[~reliable] = np.nan
    guided['surface_height_mm'] = hf._nullable(heights)
    guided['tolerance_mm'] = hf._nullable(tolerance)
    guided['diagnostics'].update(reference_cells_connected=int(known.sum()),
        interpolation_squares_four_corners=int(corners.sum()),
        interpolation_squares_reliable=int(reliable.sum()),
        reliable_area_mm2=float(reliable.sum() * cell_size ** 2),
        reliable_fraction_of_roi=float(reliable.sum() / reliable.size),
        tolerance_quantiles_mm=np.quantile(tolerance[reliable], [0, .5, .95, 1]).tolist())
    deviation = np.asarray(surface, dtype=float) - nominal
    deviation[np.asarray(support) < geometry['fit_parameters']['min_capture_support']] = np.nan
    record = {'mesh_path': str(config['mesh_path']), 'scale_to_mm': float(config['scale_to_mm']),
              'seed_bin_from_cad': deepcopy(config['bin_from_cad']),
              'bin_from_cad': result['bin_from_cad'].tolist(),
              'role': 'nominal_association_with_measured_surface_and_rim',
              'parameters': parameters, 'diagnostics': diagnostics,
              'nominal_height_mm': hf._nullable(nominal),
              'apparent_deviation_mm': hf._nullable(deviation),
              'association_mask': association.astype(int).tolist()}
    return guided, record, rim_mask


def validate(record, geometry):
    """Validate optional stored arrays without requiring CAD or Open3D at runtime."""
    if not isinstance(record, dict) or record.get('role') != 'nominal_association_with_measured_surface_and_rim':
        raise DatasetError('Unsupported CAD reference role')
    validate_transform(record.get('bin_from_cad'), 'CAD bin_from_cad')
    validate_transform(record.get('seed_bin_from_cad'), 'CAD seed_bin_from_cad')
    cad._positive(record.get('scale_to_mm'), 'CAD scale_to_mm')
    path = record.get('mesh_path')
    if not isinstance(path, str) or not path or Path(path).suffix.lower() != '.stl':
        raise DatasetError('CAD reference must name its STL source')
    params = _parameters(record.get('parameters', {}))
    heights = np.asarray(geometry['surface_height_mm'], dtype=float)
    try:
        nominal = np.asarray(record['nominal_height_mm'], dtype=float)
        deviation = np.asarray(record['apparent_deviation_mm'], dtype=float)
        association = np.asarray(record['association_mask'])
    except (KeyError, ValueError, TypeError) as error:
        raise DatasetError('CAD reference requires nominal, deviation and association grids') from error
    if any(array.shape != heights.shape for array in (nominal, deviation, association)):
        raise DatasetError('CAD reference grids differ from the measured geometry')
    if (np.isinf(nominal).any() or np.isinf(deviation).any() or not np.isfinite(nominal).any()
            or not np.isin(association, [0, 1]).all()):
        raise DatasetError('CAD reference grids contain invalid heights or association values')
    supported = np.asarray(geometry['capture_support']) >= geometry['fit_parameters']['min_capture_support']
    if np.any(np.isfinite(deviation) & (~np.isfinite(nominal) | ~supported)):
        raise DatasetError('CAD apparent deviation requires both nominal and measured support')
    if np.any(np.isfinite(heights) & ~association.astype(bool)):
        raise DatasetError('Measured geometry lies outside the CAD association mask')
    common = np.isfinite(heights) & np.isfinite(nominal)
    if not np.allclose((nominal + deviation)[common], heights[common], atol=1e-8, rtol=0):
        raise DatasetError('CAD apparent deviation must preserve the measured surface heights')
    if not np.isfinite(deviation).any() or np.nanquantile(np.abs(deviation), .95) > params['max_reference_residual_mm']:
        raise DatasetError('CAD apparent deviation exceeds its setup compatibility limit')
