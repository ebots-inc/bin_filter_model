"""Offline, CPU-only calibration of convex bins with planar floors and walls.

All coordinates and distances are millimetres. T_AB maps coordinates in B to A;
``robot_from_bin_seed`` is a rigid transform, not a request to estimate CAD pose.
Only empty training captures belong in ``fit_bin``. Every capture must support
every surface. Unsupported geometry is an error, never an implicit pass-through.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import combinations

import numpy as np


class GeometryFitError(ValueError):
    """The supplied captures/configuration cannot support a valid bin model."""


FIT_DEFAULTS = {
    "ransac_distance_mm": 0.5,
    "max_normal_angle_deg": 15.0,
    "iterations": 250,
    "min_points_per_surface": 50,
    "max_points_per_capture": 200000,
    "residual_quantile": 0.99,
    "alignment_margin_mm": 0.2,
    "min_tolerance_mm": 0.2,
    "max_tolerance_mm": 2.0,
    "min_inlier_fraction": 0.9,
    "min_coverage_fraction": 0.5,
    "random_seed": 0,
}

_CONFIG_KEYS = {
    "bin_id", "robot_from_bin_seed", "interior_polygon_mm",
    "floor_search_band_mm", "wall_search_band_mm", "wall_height_range_mm",
    "max_height_mm", "fit",
}


def _number(value, name, *, minimum=None, strict=False):
    if isinstance(value, (bool, np.bool_)):
        raise GeometryFitError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise GeometryFitError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise GeometryFitError(f"{name} must be finite")
    if minimum is not None and (result <= minimum if strict else result < minimum):
        operator = ">" if strict else ">="
        raise GeometryFitError(f"{name} must be {operator} {minimum}")
    return result


def _integer(value, name, minimum):
    result = _number(value, name, minimum=minimum)
    if result != int(result):
        raise GeometryFitError(f"{name} must be an integer")
    return int(result)


def _array(value, name):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise GeometryFitError(f"{name} must contain numeric values") from error
    if not np.isfinite(result).all():
        raise GeometryFitError(f"{name} contains non-finite values")
    return result


def _transform(value):
    transform = _array(value, "robot_from_bin_seed")
    if transform.shape != (4, 4):
        raise GeometryFitError("robot_from_bin_seed must be 4x4")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0)):
        raise GeometryFitError("robot_from_bin_seed must be a proper rigid transform (no scale/reflection)")
    return transform


def _polygon(value):
    polygon = _array(value, "interior_polygon_mm")
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise GeometryFitError("interior_polygon_mm must have at least three XY vertices")
    edges = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= 1e-6):
        raise GeometryFitError("Polygon vertices must be distinct; do not repeat the first vertex")
    normals = np.column_stack((-edges[:, 1], edges[:, 0])) / lengths[:, None]
    offsets = -np.einsum("ij,ij->i", normals, polygon)
    # Requiring every non-edge vertex strictly inside every edge also rejects
    # self-intersecting polygons and collinear vertices, not just wrong winding.
    for edge in range(len(polygon)):
        other = [i for i in range(len(polygon)) if i not in (edge, (edge + 1) % len(polygon))]
        if np.any(polygon[other] @ normals[edge] + offsets[edge] <= 1e-6):
            raise GeometryFitError("interior_polygon_mm must be strictly convex and counterclockwise")
    return polygon, normals, offsets, edges / lengths[:, None], lengths


def _inset_polygon(normals, offsets, inset):
    vertices = []
    for index in range(len(normals)):
        matrix = np.stack((normals[index - 1], normals[index]))
        try:
            vertex = np.linalg.solve(matrix, inset - offsets[[index - 1, index]])
        except np.linalg.LinAlgError as error:
            raise GeometryFitError("Wall search band leaves no central floor footprint") from error
        vertices.append(vertex)
    vertices = np.asarray(vertices)
    if np.any(vertices @ normals.T + offsets < inset - 1e-6):
        raise GeometryFitError("Wall search band leaves no central floor footprint")
    if np.any(np.ptp(vertices, axis=0) <= 1e-6):
        raise GeometryFitError("Central floor footprint is degenerate")
    return vertices


def _configuration(config):
    if not isinstance(config, Mapping):
        raise GeometryFitError("Bin configuration must be a dictionary")
    unknown = set(config) - _CONFIG_KEYS
    required = _CONFIG_KEYS - {"bin_id", "fit"}
    missing = required - set(config)
    if missing or unknown:
        raise GeometryFitError(f"Invalid bin configuration: missing={sorted(missing)}, unknown={sorted(unknown)}")
    supplied_fit = config.get("fit", {})
    if not isinstance(supplied_fit, Mapping) or set(supplied_fit) - set(FIT_DEFAULTS):
        raise GeometryFitError("fit contains unsupported parameters")
    parameters = FIT_DEFAULTS | dict(supplied_fit)
    integer_keys = {"iterations": 1, "min_points_per_surface": 3,
                    "max_points_per_capture": 3, "random_seed": 0}
    for key, minimum in integer_keys.items():
        parameters[key] = _integer(parameters[key], f"fit.{key}", minimum)
    for key in set(parameters) - set(integer_keys):
        parameters[key] = _number(parameters[key], f"fit.{key}", minimum=0,
                                  strict=key != "alignment_margin_mm")
    if not 0 < parameters["max_normal_angle_deg"] < 45:
        raise GeometryFitError("fit.max_normal_angle_deg must be between 0 and 45 degrees")
    for key in ("residual_quantile", "min_inlier_fraction", "min_coverage_fraction"):
        if not 0 < parameters[key] <= 1:
            raise GeometryFitError(f"fit.{key} must be in (0, 1]")
    if parameters["residual_quantile"] < 0.9:
        raise GeometryFitError("fit.residual_quantile must be at least 0.9")
    if parameters["min_inlier_fraction"] < 0.5:
        raise GeometryFitError("fit.min_inlier_fraction must be at least 0.5")
    if parameters["min_tolerance_mm"] > parameters["max_tolerance_mm"]:
        raise GeometryFitError("Minimum tolerance exceeds maximum tolerance")
    if parameters["ransac_distance_mm"] > parameters["max_tolerance_mm"]:
        raise GeometryFitError("RANSAC distance exceeds maximum allowable surface tolerance")
    if parameters["max_points_per_capture"] < parameters["min_points_per_surface"]:
        raise GeometryFitError("Sampling cap is below minimum surface support")
    transform = _transform(config["robot_from_bin_seed"])
    polygon, normals, offsets, directions, lengths = _polygon(config["interior_polygon_mm"])
    floor_band = _number(config["floor_search_band_mm"], "floor_search_band_mm", minimum=0, strict=True)
    wall_band = _number(config["wall_search_band_mm"], "wall_search_band_mm", minimum=0, strict=True)
    height_range = _array(config["wall_height_range_mm"], "wall_height_range_mm")
    if height_range.shape != (2,) or not 0 < height_range[0] < height_range[1]:
        raise GeometryFitError("wall_height_range_mm must be [positive_min, larger_max]")
    max_height = _number(config["max_height_mm"], "max_height_mm", minimum=0, strict=True)
    if max_height < height_range[1]:
        raise GeometryFitError("max_height_mm must include the wall calibration height range")
    if np.any(lengths <= 2 * wall_band):
        raise GeometryFitError("Wall search band consumes an entire polygon edge")
    inset = _inset_polygon(normals, offsets, wall_band)
    return (parameters, transform, polygon, normals, offsets, directions, lengths,
            floor_band, wall_band, height_range, max_height, inset)


def _cloud(value, name):
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise GeometryFitError(f"{name} must be an Nx3 numeric array") from error
    if points.ndim != 2 or points.shape[1] != 3:
        raise GeometryFitError(f"{name} must be an Nx3 numeric array")
    return points


def _least_squares_plane(points, nominal_normal):
    center = points.mean(axis=0)
    centered = points - center
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(points))
    if eigenvalues[1] <= 1e-10:
        raise GeometryFitError("Surface support is collinear or coincident")
    normal = eigenvectors[:, 0]
    if normal @ nominal_normal < 0:
        normal = -normal
    return normal, -float(normal @ center)


def _fit_plane(captures, nominal_normal, coverage_coordinates, expected_spans, parameters, rng, name):
    minimum = parameters["min_points_per_surface"]
    for index, points in enumerate(captures):
        if len(points) < minimum:
            raise GeometryFitError(f"{name}: capture {index} has {len(points)} seed points; needs {minimum}")
    points = np.concatenate(captures)
    evaluation = points
    if len(points) > 10000:
        evaluation = points[rng.choice(len(points), 10000, replace=False)]
    cosine = math.cos(math.radians(parameters["max_normal_angle_deg"]))
    distance = parameters["ransac_distance_mm"]
    best_count, best_plane, best_error = -1, None, math.inf
    for _ in range(parameters["iterations"]):
        triple = evaluation[rng.choice(len(evaluation), 3, replace=False)]
        normal = np.cross(triple[1] - triple[0], triple[2] - triple[0])
        norm = np.linalg.norm(normal)
        if norm <= 1e-10:
            continue
        normal /= norm
        if normal @ nominal_normal < 0:
            normal = -normal
        if normal @ nominal_normal < cosine:
            continue
        offset = -float(normal @ triple[0])
        residuals = np.abs(evaluation @ normal + offset)
        inliers = residuals <= distance
        count = int(inliers.sum())
        error = float(np.median(residuals[inliers])) if count else math.inf
        if count > best_count or (count == best_count and error < best_error):
            best_count, best_plane, best_error = count, (normal, offset), error
    if best_plane is None or best_count < 3:
        raise GeometryFitError(f"{name}: no supported plane within the expected normal constraint")
    normal, offset = best_plane
    for _ in range(4):
        inliers = np.abs(points @ normal + offset) <= distance
        if int(inliers.sum()) < minimum:
            raise GeometryFitError(f"{name}: insufficient support during plane refinement")
        normal, offset = _least_squares_plane(points[inliers], nominal_normal)
        if normal @ nominal_normal < cosine:
            raise GeometryFitError(f"{name}: fitted surface exceeds its allowed normal angle")
    per_capture = []
    residual_bounds = []
    for index, capture in enumerate(captures):
        residuals = np.abs(capture @ normal + offset)
        inliers = residuals <= distance
        count = int(inliers.sum())
        fraction = count / len(capture)
        if count < minimum or fraction < parameters["min_inlier_fraction"]:
            raise GeometryFitError(
                f"{name}: capture {index} lacks planar support ({count}/{len(capture)} inliers); "
                "check bin pose, seed bands, curvature, or incomplete empty capture")
        coordinates = coverage_coordinates(capture[inliers])
        spans = np.quantile(coordinates, 0.98, axis=0) - np.quantile(coordinates, 0.02, axis=0)
        coverage = spans / expected_spans
        if np.any(coverage < parameters["min_coverage_fraction"]):
            raise GeometryFitError(f"{name}: capture {index} has insufficient spatial coverage {coverage.tolist()}")
        quantile = float(np.quantile(residuals, parameters["residual_quantile"]))
        residual_bounds.append(quantile)
        per_capture.append({
            "capture_index": index, "seed_point_count": len(capture),
            "inlier_point_count": count, "inlier_fraction": fraction,
            "coverage_fraction": coverage.tolist(), "residual_quantile_mm": quantile,
            "inlier_rms_mm": float(np.sqrt(np.mean(residuals[inliers] ** 2))),
            "signed_residual_median_mm": float(np.median(capture @ normal + offset)),
        })
    tolerance = max(parameters["min_tolerance_mm"],
                    max(residual_bounds) + parameters["alignment_margin_mm"])
    if tolerance > parameters["max_tolerance_mm"]:
        raise GeometryFitError(
            f"{name}: measured tolerance {tolerance:.4g} mm exceeds maximum "
            f"{parameters['max_tolerance_mm']:.4g} mm; no tolerance was silently clipped")
    return {
        "normal": normal.tolist(), "offset_mm": offset, "tolerance_mm": tolerance,
        "diagnostics": {
            "normal_angle_deg": math.degrees(math.acos(float(np.clip(normal @ nominal_normal, -1, 1)))),
            "point_count": len(points), "per_capture": per_capture,
        },
    }


def _check_volume(floor, walls, max_height):
    floor_normal = np.asarray(floor["normal"])
    normals = np.asarray([wall["normal"] for wall in walls])
    offsets = np.asarray([wall["offset_mm"] for wall in walls])
    tolerance = np.asarray([wall["tolerance_mm"] for wall in walls])
    # Check both the physical volume and the usable volume after tolerances.
    # For planar halfspaces, valid bounded cross-sections at both ends establish
    # a nondegenerate volume throughout the intervening height interval.
    for inset in (np.zeros(len(walls)), tolerance):
        for height in (floor["tolerance_mm"], max_height):
            vertices = []
            for index in range(len(walls)):
                matrix = np.stack((floor_normal, normals[index - 1], normals[index]))
                if np.linalg.cond(matrix) > 1e8:
                    raise GeometryFitError("Fitted walls do not define a stable bounded bin volume")
                rhs = [height - floor["offset_mm"],
                       inset[index - 1] - offsets[index - 1], inset[index] - offsets[index]]
                vertex = np.linalg.solve(matrix, rhs)
                if not np.isfinite(vertex).all():
                    raise GeometryFitError("Fitted bin coordinates exceed the numerical range")
                if np.any(normals @ vertex + offsets < inset - 1e-6):
                    raise GeometryFitError("Fitted walls intersect inconsistently or close the usable bin volume")
                vertices.append(vertex)
            vertices = np.asarray(vertices)
            area = 0.5 * abs(np.sum(vertices[:, 0] * np.roll(vertices[:, 1], -1)
                                    - vertices[:, 1] * np.roll(vertices[:, 0], -1)))
            if not math.isfinite(area) or area <= 1e-6:
                raise GeometryFitError("Fitted bin volume is degenerate")


def fit_bin(clouds: list[np.ndarray], config: dict) -> dict:
    """Fit one bin using multiple unfiltered, robot-frame EMPTY training clouds.

    ``floor_search_band_mm`` bounds |local Z| around the seed floor. Wall seeds
    use distance from the polygon edges and the local-Z ``wall_height_range_mm``.
    Floor seeds are inset by the wall search band; wall endpoints are similarly
    excluded to avoid adjacent surfaces. Sampling caps apply to each whole cloud.
    ``max_height_mm`` is signed normal height above the fitted floor. Wall planes
    extend to that height: this version assumes a convex, planar-sided cavity;
    ledges, ribs, curved surfaces, and changing wall slope need another model.
    """
    if not isinstance(clouds, (list, tuple)) or len(clouds) < 2:
        raise GeometryFitError("Provide at least two independent empty training captures")
    (parameters, transform, polygon, normals, offsets, directions, lengths,
     floor_band, wall_band, height_range, max_height, inset) = _configuration(config)
    rng = np.random.default_rng(parameters["random_seed"])
    local_clouds, capture_diagnostics = [], []
    for index, value in enumerate(clouds):
        cloud = _cloud(value, f"capture {index}")
        finite = np.isfinite(cloud).all(axis=1)
        selected = cloud[finite]
        finite_count = len(selected)
        if len(selected) > parameters["max_points_per_capture"]:
            selected = selected[rng.choice(len(selected), parameters["max_points_per_capture"], replace=False)]
        local_clouds.append((selected - transform[:3, 3]) @ transform[:3, :3])
        capture_diagnostics.append({"capture_index": index, "input_point_count": len(cloud),
                                    "finite_point_count": finite_count, "sampled_point_count": len(selected)})
    floor_seeds = [points[(np.abs(points[:, 2]) <= floor_band)
                          & np.all(points[:, :2] @ normals.T + offsets >= wall_band, axis=1)]
                   for points in local_clouds]
    floor = _fit_plane(floor_seeds, np.array([0.0, 0.0, 1.0]), lambda p: p[:, :2],
                       np.ptp(inset, axis=0), parameters, rng, "floor")
    if floor["tolerance_mm"] >= max_height:
        raise GeometryFitError("Floor tolerance consumes the entire allowed height")
    walls = []
    floor_normal = np.asarray(floor["normal"])
    for edge, (normal_xy, offset, direction, length) in enumerate(zip(normals, offsets, directions, lengths)):
        seeds = []
        for points in local_clouds:
            along = (points[:, :2] - polygon[edge]) @ direction
            floor_height = points @ floor_normal + floor["offset_mm"]
            selected = ((np.abs(points[:, :2] @ normal_xy + offset) <= wall_band)
                        & (along >= wall_band) & (along <= length - wall_band)
                        & (points[:, 2] >= height_range[0]) & (points[:, 2] <= height_range[1])
                        & (floor_height > floor["tolerance_mm"] + parameters["ransac_distance_mm"]))
            seeds.append(points[selected])
        def coordinates(points, origin=polygon[edge], direction=direction):
            return np.column_stack(((points[:, :2] - origin) @ direction, points[:, 2]))
        wall = _fit_plane(seeds, np.r_[normal_xy, 0.0], coordinates,
                          np.array([length - 2 * wall_band, height_range[1] - height_range[0]]),
                          parameters, rng, f"wall {edge}")
        wall["edge_index"] = edge
        walls.append(wall)
    _check_volume(floor, walls, max_height)
    model = {
        "kind": "convex_planar_bin", "length_unit": "mm",
        "robot_from_bin": transform.tolist(), "interior_polygon_mm": polygon.tolist(),
        "max_height_mm": max_height, "floor": floor, "walls": walls,
        "fit_parameters": parameters,
        "seed_parameters": {"floor_search_band_mm": floor_band, "wall_search_band_mm": wall_band,
                            "wall_height_range_mm": height_range.tolist()},
        "diagnostics": {"capture_count": len(clouds), "captures": capture_diagnostics},
    }
    if "bin_id" in config:
        if not isinstance(config["bin_id"], str) or not config["bin_id"].strip():
            raise GeometryFitError("bin_id must be a nonempty string")
        model["bin_id"] = config["bin_id"]
    return model


def _validated_model(model):
    if not isinstance(model, Mapping) or model.get("kind") != "convex_planar_bin":
        raise GeometryFitError("Unsupported bin model format/kind")
    if model.get("length_unit") != "mm":
        raise GeometryFitError("Bin model must use millimetres")
    try:
        transform = _transform(model["robot_from_bin"])
        polygon, expected_normals, _, _, _ = _polygon(model["interior_polygon_mm"])
        floor, walls = model["floor"], model["walls"]
        max_height = _number(model["max_height_mm"], "model.max_height_mm", minimum=0, strict=True)
        if not isinstance(walls, list) or len(walls) != len(polygon):
            raise GeometryFitError("Bin model wall count differs from its polygon")
        for index, plane in enumerate([floor, *walls]):
            normal = _array(plane["normal"], "plane.normal")
            if normal.shape != (3,) or not np.isclose(np.linalg.norm(normal), 1, atol=1e-6, rtol=0):
                raise GeometryFitError("Plane normals must be normalized 3-vectors")
            _number(plane["offset_mm"], "plane.offset_mm")
            _number(plane["tolerance_mm"], "plane.tolerance_mm", minimum=0, strict=True)
            nominal = np.array([0, 0, 1]) if index == 0 else np.r_[expected_normals[index - 1], 0]
            if normal @ nominal <= math.sqrt(0.5):
                raise GeometryFitError("Plane orientation is inconsistent with its inward seed normal")
        if floor["tolerance_mm"] >= max_height:
            raise GeometryFitError("Floor tolerance consumes the entire allowed height")
        _check_volume(floor, walls, max_height)
    except (KeyError, TypeError) as error:
        raise GeometryFitError("Incomplete or invalid bin model") from error
    return transform, floor, walls, max_height


def classify(points_robot, model):
    """Return retained-point mask and disjoint diagnostic reason masks.

    Reasons partition all input rows: ``invalid``, ``outside``, ``floor``,
    ``wall``, ``kept`` (in that priority). This classifies geometry only, not a
    grasp target or collision-free motion. Invalid observations remain excluded.
    """
    points = _cloud(points_robot, "points_robot")
    transform, floor, walls, max_height = _validated_model(model)
    valid = np.isfinite(points).all(axis=1)
    local = np.zeros_like(points)
    with np.errstate(over="ignore", invalid="ignore"):
        local[valid] = (points[valid] - transform[:3, 3]) @ transform[:3, :3]
    valid &= np.isfinite(local).all(axis=1)
    local[~valid] = 0
    floor_height = local @ np.asarray(floor["normal"]) + floor["offset_mm"]
    wall_distance = local @ np.asarray([wall["normal"] for wall in walls]).T
    wall_distance += np.asarray([wall["offset_mm"] for wall in walls])
    wall_tolerance = np.asarray([wall["tolerance_mm"] for wall in walls])
    outside = valid & ((floor_height < -floor["tolerance_mm"]) | (floor_height > max_height)
                       | np.any(wall_distance < -wall_tolerance, axis=1))
    remaining = valid & ~outside
    floor_mask = remaining & (floor_height <= floor["tolerance_mm"])
    wall_mask = remaining & ~floor_mask & np.any(wall_distance <= wall_tolerance, axis=1)
    keep = remaining & ~floor_mask & ~wall_mask
    return keep, {"invalid": ~valid, "outside": outside, "floor": floor_mask, "wall": wall_mask, "kept": keep}


def _world_halfspaces(model):
    """Usable closed halfspaces: normal_robot @ point_robot + offset >= 0."""
    transform, floor, walls, max_height = _validated_model(model)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    normals, offsets = [], []
    for plane in [floor, *walls]:
        normal = rotation @ np.asarray(plane["normal"])
        offset = plane["offset_mm"] - normal @ translation - plane["tolerance_mm"]
        normals.append(normal)
        offsets.append(offset)
    floor_normal = rotation @ np.asarray(floor["normal"])
    normals.append(-floor_normal)
    offsets.append(max_height - floor["offset_mm"] + floor_normal @ translation)
    return np.asarray(normals), np.asarray(offsets)


def validate_disjoint_bins(models: dict) -> dict:
    """Reject positive-volume overlap between fitted usable bin memberships.

    This compares the complete bounded convex models, not only observed points.
    Shared faces/edges are permitted. Boundaries are compared at 1e-6 mm numerical
    tolerance. This is model-membership consistency, not collision validation or
    proof that fitted models match the actual bins.
    """
    if not isinstance(models, Mapping) or not models:
        raise GeometryFitError("Provide a nonempty dictionary of named bin models")
    if any(not isinstance(name, str) or not name.strip() for name in models):
        raise GeometryFitError("Bin model names must be nonempty strings")
    halfspaces = {name: _world_halfspaces(model) for name, model in models.items()}
    tolerance = 1e-6
    diagnostics = []
    for left, right in combinations(halfspaces, 2):
        normals = np.concatenate((halfspaces[left][0], halfspaces[right][0]))
        offsets = np.concatenate((halfspaces[left][1], halfspaces[right][1]))
        vertices = []
        for indices in combinations(range(len(normals)), 3):
            selected = list(indices)
            matrix = normals[selected]
            if np.linalg.cond(matrix) > 1e8:
                continue
            try:
                vertex = np.linalg.solve(matrix, -offsets[selected])
            except np.linalg.LinAlgError:
                continue
            if not np.isfinite(vertex).all() or np.any(normals @ vertex + offsets < -tolerance):
                continue
            if any(np.linalg.norm(vertex - existing) <= tolerance for existing in vertices):
                continue
            vertices.append(vertex)
        dimension = -1
        if vertices:
            vertices = np.asarray(vertices)
            dimension = int(np.linalg.matrix_rank(vertices - vertices[0], tol=tolerance))
            if dimension == 3:
                raise GeometryFitError(
                    f"Bins {left!r} and {right!r} have overlapping usable volumes; "
                    "check separate bin placements, interior boundaries, slopes, and max_height_mm")
        diagnostics.append({"bins": [left, right],
                            "relation": "boundary_contact" if dimension >= 0 else "disjoint",
                            "intersection_dimension": dimension, "intersection_vertex_count": len(vertices)})
    return {"status": "no_positive_volume_overlap", "numerical_tolerance_mm": tolerance,
            "scope": "fitted_model_membership_only", "pairs": diagnostics}
