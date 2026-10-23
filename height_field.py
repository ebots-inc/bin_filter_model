"""Offline measured height fields for fixed curved bins; CPU NumPy only.

Surface vertices are at XY cell centers in the declared coordinate frame. Interpolation
requires four observed vertices. Tolerances belong to interpolation squares,
are measured on independent empty TRAIN captures, and never fill missing data.
This is background classification, not CAD registration or grasp validation.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from itertools import combinations
import math
import warnings

import numpy as np


class HeightFieldError(ValueError):
    pass


DEFAULTS = {
    "cell_size_mm": 1.0,
    "min_capture_support": 5,
    "min_points_per_cell": 3,
    "residual_quantile": 0.99,
    "alignment_margin_mm": 0.5,
    "min_tolerance_mm": 1.0,
    "max_tolerance_mm": 4.0,
    "max_reference_height_mm": -82.0,
    "max_height_above_surface_mm": 150.0,
}


def _number(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise HeightFieldError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise HeightFieldError(f"{name} must be a finite number") from error
    if not math.isfinite(value) or (positive and value <= 0):
        raise HeightFieldError(f"{name} must be {'positive and ' if positive else ''}finite")
    return value


def _array(value, shape, name, allow_unknown=False):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise HeightFieldError(f"{name} must contain numeric values") from error
    if result.shape != shape:
        raise HeightFieldError(f"{name} must have shape {shape}; got {result.shape}")
    if np.isinf(result).any() or (not allow_unknown and not np.isfinite(result).all()):
        raise HeightFieldError(f"{name} contains invalid values")
    return result


def _cloud(points):
    try:
        points = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise HeightFieldError("Cloud must be an Nx3 numeric array in robot millimetres") from error
    if points.ndim != 2 or points.shape[1] != 3:
        raise HeightFieldError("Cloud must be an Nx3 numeric array in robot millimetres")
    return points


def _configuration(config):
    if not isinstance(config, Mapping):
        raise HeightFieldError("Height-field configuration must be a dictionary")
    allowed = set(DEFAULTS) | {"kind", "bin_id", "frame_id", "xy_bounds_mm", "z_bounds_mm", "seed_xy_mm"}
    if set(config) - allowed:
        raise HeightFieldError(f"Unsupported height-field settings: {sorted(set(config) - allowed)}")
    if not isinstance(config.get('frame_id', 'B2'), str) or not config.get('frame_id', 'B2').strip():
        raise HeightFieldError("Height-field frame must be a nonempty coordinate frame")
    if config.get("kind", "height_field") != "height_field":
        raise HeightFieldError("Expected kind: height_field")
    if "xy_bounds_mm" not in config or "z_bounds_mm" not in config:
        raise HeightFieldError("xy_bounds_mm and background z_bounds_mm are required")
    parameters = DEFAULTS | {key: config[key] for key in DEFAULTS if key in config}
    for name in parameters:
        parameters[name] = _number(parameters[name], name,
                                    positive=name not in ("max_reference_height_mm", "alignment_margin_mm"))
    for name in ("min_capture_support", "min_points_per_cell"):
        if parameters[name] != int(parameters[name]):
            raise HeightFieldError(f"{name} must be a positive integer")
        parameters[name] = int(parameters[name])
    if parameters["min_capture_support"] < 2:
        raise HeightFieldError("At least two independent captures must support each surface cell")
    if not 0.9 <= parameters["residual_quantile"] <= 1:
        raise HeightFieldError("residual_quantile must be in [0.9, 1]")
    if not 0 <= parameters["alignment_margin_mm"] < parameters["max_tolerance_mm"]:
        raise HeightFieldError("alignment_margin_mm must be nonnegative and below max_tolerance_mm")
    if parameters["min_tolerance_mm"] > parameters["max_tolerance_mm"]:
        raise HeightFieldError("Minimum tolerance exceeds maximum tolerance")
    if parameters["max_height_above_surface_mm"] <= parameters["max_tolerance_mm"]:
        raise HeightFieldError("Maximum foreground height must exceed maximum surface tolerance")
    bounds = _array(config["xy_bounds_mm"], (2, 2), "xy_bounds_mm")
    z_bounds = _array(config["z_bounds_mm"], (2,), "z_bounds_mm")
    if np.any(bounds[1] <= bounds[0]) or z_bounds[1] <= z_bounds[0]:
        raise HeightFieldError("Bounds must be [minimum, maximum] in ascending order")
    if not z_bounds[0] < parameters["max_reference_height_mm"] <= z_bounds[1]:
        raise HeightFieldError("max_reference_height_mm must lie above the minimum and within background Z bounds")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        shape_xy = np.ceil((bounds[1] - bounds[0]) / parameters["cell_size_mm"])
    if (not np.isfinite(shape_xy).all() or np.any(shape_xy < 3)
            or np.any(shape_xy > 1000000) or float(shape_xy[0]) * float(shape_xy[1]) > 1000000):
        raise HeightFieldError("Grid must be at least 3x3 and at most one million cells")
    shape_xy = shape_xy.astype(int)
    seed = _array(config.get("seed_xy_mm", bounds.mean(axis=0)), (2,), "seed_xy_mm")
    if np.any(seed < bounds[0]) or np.any(seed >= bounds[1]):
        raise HeightFieldError("seed_xy_mm must lie inside the XY bounds")
    if "bin_id" in config and (not isinstance(config["bin_id"], str) or not config["bin_id"].strip()):
        raise HeightFieldError("bin_id must be a nonempty string")
    return parameters, bounds, z_bounds, seed, (int(shape_xy[1]), int(shape_xy[0]))


def _group_quantile(ids, values, count, quantile, minimum):
    """Grouped exact linear quantiles, without one Python loop per raster cell."""
    result = np.full(count, np.nan)
    point_counts = np.zeros(count, dtype=np.int64)
    if not len(ids):
        return result, point_counts
    order = np.lexsort((values, ids))
    sorted_ids, sorted_values = ids[order], values[order]
    groups, starts, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    point_counts[groups] = counts
    supported = counts >= minimum
    groups, starts, counts = groups[supported], starts[supported], counts[supported]
    positions = (counts - 1) * quantile
    lower = np.floor(positions).astype(np.int64)
    fraction = positions - lower
    low = sorted_values[starts + lower]
    high = sorted_values[starts + np.ceil(positions).astype(np.int64)]
    result[groups] = low + fraction * (high - low)
    return result, point_counts


def _connected(mask, seed_yx, name):
    y, x = seed_yx
    if not 0 <= y < mask.shape[0] or not 0 <= x < mask.shape[1] or not mask[y, x]:
        raise HeightFieldError(f"Bowl seed lies in unknown/unreliable {name}; inspect training coverage and choose an observed interior seed")
    selected = np.zeros_like(mask, dtype=bool)
    selected[y, x] = True
    frontier = deque([(y, x)])
    while frontier:
        y, x = frontier.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not selected[yy, xx]:
                selected[yy, xx] = True
                frontier.append((yy, xx))
    return selected


def _inside(points, bounds):
    return (np.isfinite(points).all(axis=1) & np.all(points[:, :2] >= bounds[0], axis=1)
            & np.all(points[:, :2] < bounds[1], axis=1))


def _interpolate(points, bounds, cell_size, heights):
    """Return prediction and lower-left square index; prediction NaN is unknown."""
    prediction = np.full(len(points), np.nan)
    ids = np.full(len(points), -1, dtype=np.int64)
    selected = np.flatnonzero(_inside(points, bounds))
    if not len(selected):
        return prediction, ids
    coordinates = (points[selected, :2] - bounds[0]) / cell_size - 0.5
    lower = np.floor(coordinates).astype(np.int64)
    nx, ny = heights.shape[1], heights.shape[0]
    valid = (lower[:, 0] >= 0) & (lower[:, 0] < nx - 1) & (lower[:, 1] >= 0) & (lower[:, 1] < ny - 1)
    selected, coordinates, lower = selected[valid], coordinates[valid], lower[valid]
    x, y = lower[:, 0], lower[:, 1]
    fraction = coordinates - lower
    corners = np.column_stack((heights[y, x], heights[y, x + 1], heights[y + 1, x], heights[y + 1, x + 1]))
    supported = np.isfinite(corners).all(axis=1)
    wx, wy = fraction[:, 0], fraction[:, 1]
    weights = np.column_stack(((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy))
    prediction[selected[supported]] = np.sum(corners[supported] * weights[supported], axis=1)
    ids[selected] = y * (nx - 1) + x
    return prediction, ids


def _nullable(values):
    array = np.asarray(values, dtype=object)
    array[~np.isfinite(np.asarray(values, dtype=float))] = None
    return array.tolist()


def fit_bin(clouds: list[np.ndarray], config: dict) -> dict:
    """Fit measured curved background from empty TRAIN captures only.

    Per-capture cell medians are combined by median. Background Z/rim gates apply
    only to this reference, never to live foreground heights. Residual quantiles
    use ALL finite training points over observed XY squares, including points
    outside the reference fitting Z band. The maximum qualifying per-capture
    residual quantile sets each square's tolerance. Excessive tolerance, missing
    support, and disconnected regions are unknown, never clipped or filled.
    """
    parameters, bounds, z_bounds, seed, shape = _configuration(config)
    if not isinstance(clouds, (list, tuple)) or len(clouds) < parameters["min_capture_support"]:
        raise HeightFieldError(f"Need at least {parameters['min_capture_support']} independent empty training captures")
    cell_size = parameters["cell_size_mm"]
    point_minimum, capture_minimum = parameters["min_points_per_cell"], parameters["min_capture_support"]
    cell_count = int(np.prod(shape))
    medians = np.full((len(clouds), cell_count), np.nan)
    capture_diagnostics = []
    for index, value in enumerate(clouds):
        points = _cloud(value)
        selected = points[_inside(points, bounds) & (points[:, 2] >= z_bounds[0]) & (points[:, 2] <= z_bounds[1])]
        indices = np.floor((selected[:, :2] - bounds[0]) / cell_size).astype(np.int64)
        ids = indices[:, 1] * shape[1] + indices[:, 0]
        medians[index], _ = _group_quantile(ids, selected[:, 2], cell_count, 0.5, point_minimum)
        capture_diagnostics.append({"capture_index": index, "input_point_count": len(points),
                                    "background_fit_point_count": len(selected),
                                    "supported_reference_cells": int(np.isfinite(medians[index]).sum())})
    capture_support = np.isfinite(medians).sum(axis=0).reshape(shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        heights = np.nanmedian(medians, axis=0).reshape(shape)
    observed = (capture_support >= capture_minimum) & np.isfinite(heights)
    below_rim = observed & (heights <= parameters["max_reference_height_mm"])
    seed_vertex = np.floor((seed - bounds[0]) / cell_size).astype(int)
    connected_vertices = _connected(below_rim, (int(seed_vertex[1]), int(seed_vertex[0])), "reference cells")
    heights[~connected_vertices] = np.nan
    square_shape = (shape[0] - 1, shape[1] - 1)
    square_count = int(np.prod(square_shape))
    residuals = np.full((len(clouds), square_count), np.nan)
    for index, value in enumerate(clouds):
        points = _cloud(value)
        prediction, ids = _interpolate(points, bounds, cell_size, heights)
        selected = np.isfinite(prediction)
        residuals[index], _ = _group_quantile(ids[selected], np.abs(points[selected, 2] - prediction[selected]),
                                             square_count, parameters["residual_quantile"], point_minimum)
        capture_diagnostics[index]["residual_point_count"] = int(selected.sum())
        capture_diagnostics[index]["supported_residual_squares"] = int(np.isfinite(residuals[index]).sum())
    residual_support = np.isfinite(residuals).sum(axis=0).reshape(square_shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        measured = np.nanmax(residuals, axis=0).reshape(square_shape) + parameters["alignment_margin_mm"]
    measured = np.maximum(measured, parameters["min_tolerance_mm"])
    corner_support = (np.isfinite(heights[:-1, :-1]) & np.isfinite(heights[:-1, 1:])
                      & np.isfinite(heights[1:, :-1]) & np.isfinite(heights[1:, 1:]))
    sufficient = corner_support & (residual_support >= capture_minimum) & np.isfinite(measured)
    excessive = sufficient & (measured > parameters["max_tolerance_mm"])
    reliable = sufficient & ~excessive
    seed_square = np.floor((seed - bounds[0]) / cell_size - 0.5).astype(int)
    connected_squares = _connected(reliable, (int(seed_square[1]), int(seed_square[0])), "interpolation squares")
    tolerance = measured.copy()
    tolerance[~connected_squares] = np.nan
    reliable_tolerances = tolerance[np.isfinite(tolerance)]
    if not len(reliable_tolerances):
        raise HeightFieldError("No reliable connected surface remains after measured-uncertainty checks")
    model = {
        "kind": "height_field", "frame_id": config.get("frame_id", "B2"), "length_unit": "mm",
        "xy_bounds_mm": bounds.tolist(), "z_bounds_mm": z_bounds.tolist(), "seed_xy_mm": seed.tolist(),
        "cell_size_mm": cell_size,
        "x_centers_mm": (bounds[0, 0] + (np.arange(shape[1]) + 0.5) * cell_size).tolist(),
        "y_centers_mm": (bounds[0, 1] + (np.arange(shape[0]) + 0.5) * cell_size).tolist(),
        "surface_height_mm": _nullable(heights), "tolerance_mm": _nullable(tolerance),
        "capture_support": capture_support.tolist(), "residual_capture_support": residual_support.tolist(),
        "max_reference_height_mm": parameters["max_reference_height_mm"],
        "max_height_above_surface_mm": parameters["max_height_above_surface_mm"],
        "fit_parameters": parameters,
        "diagnostics": {
            "capture_count": len(clouds), "grid_shape_yx": list(shape),
            "reference_cells_total": cell_count, "reference_cells_observed": int(observed.sum()),
            "reference_cells_below_rim": int(below_rim.sum()),
            "reference_cells_connected": int(connected_vertices.sum()),
            "interpolation_squares_total": square_count,
            "interpolation_squares_four_corners": int(corner_support.sum()),
            "interpolation_squares_sufficient_support": int(sufficient.sum()),
            "interpolation_squares_excessive_tolerance": int(excessive.sum()),
            "interpolation_squares_reliable_before_component": int(reliable.sum()),
            "interpolation_squares_reliable": int(connected_squares.sum()),
            "reliable_area_mm2": float(connected_squares.sum() * cell_size ** 2),
            "reliable_fraction_of_roi": float(connected_squares.sum() / square_count),
            "tolerance_quantiles_mm": np.quantile(reliable_tolerances, [0, 0.5, 0.95, 1]).tolist(),
            "captures": capture_diagnostics,
            "status": "offline_model_requires_held_out_review",
        },
    }
    if "bin_id" in config:
        model["bin_id"] = config["bin_id"]
    return model


def _validated_model(model):
    if (not isinstance(model, Mapping) or model.get("kind") != "height_field"
            or not isinstance(model.get('frame_id'), str) or not model['frame_id'].strip()
            or model.get("length_unit") != "mm"):
        raise HeightFieldError("Unsupported height-field model kind/frame/units")
    try:
        config = dict(model["fit_parameters"], xy_bounds_mm=model["xy_bounds_mm"],
                      z_bounds_mm=model["z_bounds_mm"], seed_xy_mm=model["seed_xy_mm"])
        parameters, bounds, z_bounds, seed, shape = _configuration(config)
        if model["cell_size_mm"] != parameters["cell_size_mm"]:
            raise HeightFieldError("Model cell size differs from its fit settings")
        if (model["max_reference_height_mm"] != parameters["max_reference_height_mm"]
                or model["max_height_above_surface_mm"] != parameters["max_height_above_surface_mm"]):
            raise HeightFieldError("Model height bounds differ from its fit settings")
        cell_size = parameters["cell_size_mm"]
        x = _array(model["x_centers_mm"], (shape[1],), "x_centers_mm")
        y = _array(model["y_centers_mm"], (shape[0],), "y_centers_mm")
        if (not np.allclose(x, bounds[0, 0] + (np.arange(shape[1]) + 0.5) * cell_size, atol=1e-8, rtol=0)
                or not np.allclose(y, bounds[0, 1] + (np.arange(shape[0]) + 0.5) * cell_size, atol=1e-8, rtol=0)):
            raise HeightFieldError("Model centers do not match the configured XY grid")
        heights = _array(model["surface_height_mm"], shape, "surface_height_mm", allow_unknown=True)
        square_shape = (shape[0] - 1, shape[1] - 1)
        tolerance = _array(model["tolerance_mm"], square_shape, "tolerance_mm", allow_unknown=True)
        support = _array(model["capture_support"], shape, "capture_support")
        residual_support = _array(model["residual_capture_support"], square_shape, "residual_capture_support")
        for name, counts in (("capture_support", support), ("residual_capture_support", residual_support)):
            if np.any(counts < 0) or np.any(counts != np.floor(counts)):
                raise HeightFieldError(f"{name} must contain nonnegative integer counts")
        known = np.isfinite(heights)
        if (np.any(heights[known] < z_bounds[0])
                or np.any(heights[known] > parameters["max_reference_height_mm"])
                or np.any(support[known] < parameters["min_capture_support"])):
            raise HeightFieldError("Reference heights violate fitting Z, rim, or capture-support bounds")
        corners = known[:-1, :-1] & known[:-1, 1:] & known[1:, :-1] & known[1:, 1:]
        reliable = np.isfinite(tolerance)
        if (not reliable.any() or np.any(reliable & ~corners)
                or np.any(tolerance[reliable] < parameters["min_tolerance_mm"])
                or np.any(tolerance[reliable] > parameters["max_tolerance_mm"])
                or np.any(residual_support[reliable] < parameters["min_capture_support"])):
            raise HeightFieldError("Reliable interpolation squares violate measured-support/tolerance requirements")
        seed_square = np.floor((seed - bounds[0]) / cell_size - 0.5).astype(int)
        connected = _connected(reliable, (int(seed_square[1]), int(seed_square[0])), "model interpolation squares")
        if not np.array_equal(connected, reliable):
            raise HeightFieldError("Model contains reliable surfaces disconnected from the bin seed")
    except (KeyError, TypeError) as error:
        raise HeightFieldError("Incomplete height-field model") from error
    return parameters, bounds, heights, tolerance


def classify(points_robot, model):
    """Partition observations into invalid/outside/unknown/surface/kept masks.

    Unknown never means foreground. Live points may rise above the empty rim;
    they are retained when their observed XY reference is reliable and height
    above that reference exceeds tolerance but stays within the allowed maximum.
    """
    points = _cloud(points_robot)
    parameters, bounds, heights, tolerance = _validated_model(model)
    finite = np.isfinite(points).all(axis=1)
    inside = _inside(points, bounds)
    prediction, ids = _interpolate(points, bounds, parameters["cell_size_mm"], heights)
    thresholds = np.full(len(points), np.nan)
    addressable = ids >= 0
    thresholds[addressable] = tolerance.ravel()[ids[addressable]]
    known = inside & np.isfinite(prediction) & np.isfinite(thresholds)
    residual = np.zeros(len(points))
    residual[known] = points[known, 2] - prediction[known]
    outside = finite & ~inside
    outside |= known & ((residual < -thresholds) | (residual > parameters["max_height_above_surface_mm"]))
    surface = known & ~outside & (residual <= thresholds)
    keep = known & ~outside & ~surface
    unknown = inside & ~known
    return keep, {"invalid": ~finite, "outside": outside, "unknown": unknown, "surface": surface, "kept": keep}


def validate_disjoint_bins(models: dict) -> dict:
    """Conservatively require disjoint robot-XY ROI interiors for named bins.

    Unknown holes do not justify intersecting ROI definitions. This guarantees
    unique membership of accepted points but says nothing about robot collision
    clearance or correctness of the measured bin model.
    """
    if not isinstance(models, Mapping) or not models:
        raise HeightFieldError("Provide a nonempty dictionary of named bin models")
    if any(not isinstance(name, str) or not name.strip() for name in models):
        raise HeightFieldError("Bin names must be nonempty strings")
    bounds = {name: _validated_model(model)[1] for name, model in models.items()}
    pairs = []
    for left, right in combinations(bounds, 2):
        overlap = np.minimum(bounds[left][1], bounds[right][1]) - np.maximum(bounds[left][0], bounds[right][0])
        if np.all(overlap > 1e-6):
            raise HeightFieldError(f"Bins {left!r} and {right!r} have overlapping XY bounds; define separate named-bin regions")
        pairs.append({"bins": [left, right], "relation": "disjoint_xy_interiors", "overlap_extent_mm": overlap.tolist()})
    return {"status": "disjoint_xy_interiors", "scope": "model_membership_only",
            "numerical_tolerance_mm": 1e-6, "pairs": pairs}
