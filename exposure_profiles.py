"""Select exposure profiles or use one field pooled across capture settings.

Interpolation is an assumption about depth bias between measured anchors, not a
claim of camera accuracy. Validate independent empty and loaded recordings at
interior exposures before deploying an interval. No extrapolation or missing
surface filling is performed, and a single measured exposure stays exact-only.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re

import numpy as np

from height_field import HeightFieldError, _connected, _nullable, _validated_model, validate_disjoint_bins


EXPOSURE_ATOL_MS = 1e-5
MODEL_TYPE = "offline_height_field_exposure_profiles"
LEGACY_HEIGHT_FIELD_TYPE = "offline_height_field_bins"
LEGACY_PLANAR_TYPE = "offline_convex_planar_bins"
_GEOMETRY_KEYS = (
    "kind", "frame_id", "length_unit", "xy_bounds_mm",
    "z_bounds_mm", "seed_xy_mm", "cell_size_mm", "x_centers_mm", "y_centers_mm",
    "max_reference_height_mm", "max_height_above_surface_mm", "fit_parameters",
)


def _finite(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise HeightFieldError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise HeightFieldError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _training(records, name):
    if not isinstance(records, list) or not records:
        raise HeightFieldError(f"{name} must contain independent training captures")
    by_id, signatures = {}, set()
    for record in records:
        if not isinstance(record, Mapping):
            raise HeightFieldError(f"{name} entries must be mappings")
        capture_id, signature = record.get("id"), record.get("input_signature")
        if (not isinstance(capture_id, str) or not capture_id.strip()
                or not isinstance(signature, str) or re.fullmatch(r"[0-9a-f]{64}", signature) is None):
            raise HeightFieldError(f"{name} entries require nonempty id and lowercase SHA-256 input_signature")
        if capture_id in by_id or signature in signatures:
            raise HeightFieldError(f"{name} contains duplicate training ids or input signatures")
        by_id[capture_id] = record
        signatures.add(signature)
    return by_id


def _provenance(model):
    if (not isinstance(model.get('frame_id'), str) or not model['frame_id'].strip()
            or model.get("length_unit") != "mm"
            or isinstance(model.get("scene_id"), bool)
            or not isinstance(model.get("scene_id"), int) or model["scene_id"] < 0):
        raise HeightFieldError("Model frame/units/scene are invalid")
    calibration = model.get("calibration_sha256")
    if (not isinstance(calibration, Mapping) or set(calibration) != {"camera", "robot"}
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in calibration.values())):
        raise HeightFieldError("Model calibration_sha256 must contain camera and robot SHA-256 digests")


def _bins(bins, records, legacy_planar=False, frame_id=None):
    if (not isinstance(bins, Mapping) or not bins or any(not isinstance(name, str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) for name in bins)):
        raise HeightFieldError("Model must contain at least one filesystem-safe named bin")
    if legacy_planar:
        from geometry import validate_disjoint_bins as validate_planar_bins
        validate_planar_bins(bins)
        return
    validate_disjoint_bins(bins)
    for name, model in bins.items():
        if frame_id is not None and model.get('frame_id') != frame_id:
            raise HeightFieldError(f'{name}: height-field frame differs from model frame')
        if "bin_id" in model and model["bin_id"] != name:
            raise HeightFieldError(f"{name}: bin_id does not match its model key")
        parameters = model["fit_parameters"]
        if len(records) < parameters["min_capture_support"]:
            raise HeightFieldError(f"{name}: insufficient independent training captures")
        for key in ("capture_support", "residual_capture_support"):
            if np.any(np.asarray(model[key], dtype=float) > len(records)):
                raise HeightFieldError(f"{name}: {key} exceeds profile training capture count")
        diagnostics = model.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            raise HeightFieldError(f"{name}: diagnostics must be a mapping")
        if (isinstance(diagnostics.get("capture_count"), bool)
                or not isinstance(diagnostics.get("capture_count"), int)
                or diagnostics["capture_count"] != len(records)):
            raise HeightFieldError(f"{name}: diagnostics capture count differs from profile training captures")


def validate_bundle(model):
    """Validate profile anchors, pooled fields, and legacy single-exposure fields.

    Returns the input without modifying it. Unusable intermediate coverage is
    rejected by ``select_model`` when that interval is requested; exact measured
    anchors remain usable even if their common reliable area loses the seed.
    """
    if not isinstance(model, Mapping):
        raise HeightFieldError("Exposure model must be a mapping")
    model_type = model.get("model_type")
    if model_type not in (MODEL_TYPE, LEGACY_HEIGHT_FIELD_TYPE, LEGACY_PLANAR_TYPE):
        raise HeightFieldError(f"Unsupported model type: {model_type}")
    _provenance(model)
    root_training = _training(model.get("training_captures"), "Model training_captures")
    if "exposure_settings_ms" in model:
        if model_type != LEGACY_HEIGHT_FIELD_TYPE or any(
                key in model for key in ("exposure_ms", "profiles", "interpolation")):
            raise HeightFieldError("Pooled exposure settings require one shared height field without exposure selection")
        settings = model["exposure_settings_ms"]
        if not isinstance(settings, list) or len(settings) < 2:
            raise HeightFieldError("exposure_settings_ms must contain at least two captured settings")
        settings = [_finite(value, "exposure_settings_ms", positive=True) for value in settings]
        if any(upper - lower <= EXPOSURE_ATOL_MS for lower, upper in zip(settings, settings[1:])):
            raise HeightFieldError("exposure_settings_ms must be sorted with distinct settings")
        _bins(model.get("bins"), root_training, frame_id=model['frame_id'])
        return model
    if model_type != MODEL_TYPE:
        _finite(model.get("exposure_ms"), "exposure_ms", positive=True)
        _bins(model.get("bins"), root_training, legacy_planar=model_type == LEGACY_PLANAR_TYPE, frame_id=model['frame_id'])
        return model
    policy = model.get("interpolation")
    if (not isinstance(policy, Mapping) or policy.get("method") != "linear_height_max_tolerance"
            or set(policy) != {"method", "margin_mm"}):
        raise HeightFieldError("Unsupported interpolation policy; expected linear_height_max_tolerance and margin_mm")
    if _finite(policy["margin_mm"], "interpolation margin_mm") < 0:
        raise HeightFieldError("interpolation margin_mm must be nonnegative")
    profiles = model.get("profiles")
    if not isinstance(profiles, list) or len(profiles) < 2:
        raise HeightFieldError("An exposure bundle requires at least two measured profiles")
    combined = []
    previous_exposure = None
    reference = None
    for profile in profiles:
        if not isinstance(profile, Mapping):
            raise HeightFieldError("Exposure profiles must be mappings")
        exposure = _finite(profile.get("exposure_ms"), "profile exposure_ms", positive=True)
        if previous_exposure is not None and exposure - previous_exposure <= 2 * EXPOSURE_ATOL_MS:
            raise HeightFieldError("Exposure profiles must be sorted and separated by more than 2e-5 ms")
        previous_exposure = exposure
        for key in ("frame_id", "length_unit", "scene_id", "calibration_sha256"):
            if key in profile and profile[key] != model[key]:
                raise HeightFieldError(f"Profile {exposure:g} ms has incompatible {key} provenance")
        records = _training(profile.get("training_captures"), f"Profile {exposure:g} ms training_captures")
        _bins(profile.get("bins"), records, frame_id=model['frame_id'])
        combined.extend(records.values())
        if reference is None:
            reference = profile["bins"]
        else:
            if set(profile['bins']) != set(reference):
                raise HeightFieldError('All exposure profiles must contain the same named bins')
            for name, field in profile["bins"].items():
                for key in _GEOMETRY_KEYS:
                    if field.get(key) != reference[name].get(key):
                        raise HeightFieldError(f"{name}: incompatible {key} across exposure profiles")
    all_training = _training(combined, "Combined profiles")
    if root_training != all_training:
        raise HeightFieldError("Model training_captures must exactly match the union of profile training captures")
    return model


def _blend_field(lower, upper, weight, margin):
    parameters, bounds, lower_heights, lower_tolerance = _validated_model(lower)
    _, _, upper_heights, upper_tolerance = _validated_model(upper)
    heights = (1.0 - weight) * lower_heights + weight * upper_heights
    tolerance = np.maximum(lower_tolerance, upper_tolerance) + margin
    known = np.isfinite(heights)
    corners = known[:-1, :-1] & known[:-1, 1:] & known[1:, :-1] & known[1:, 1:]
    supported = corners & np.isfinite(tolerance)
    excessive = supported & (tolerance > parameters["max_tolerance_mm"])
    reliable = supported & ~excessive
    seed_square = np.floor((np.asarray(lower["seed_xy_mm"]) - bounds[0])
                           / parameters["cell_size_mm"] - 0.5).astype(int)
    connected = _connected(reliable, (int(seed_square[1]), int(seed_square[0])),
                           "interpolated exposure squares")
    tolerance[~connected] = np.nan
    result = deepcopy(lower)
    result["surface_height_mm"] = _nullable(heights)
    result["tolerance_mm"] = _nullable(tolerance)
    for key in ("capture_support", "residual_capture_support"):
        result[key] = np.minimum(np.asarray(lower[key]), np.asarray(upper[key])).tolist()
    # The interpolated surface was never fitted to captures. Do not retain the
    # lower anchor's fit diagnostics as if they described this derived field.
    result["diagnostics"] = {
        "status": "interpolated_model_requires_held_out_interval_review",
        "interpolation_squares_total": int(tolerance.size),
        "interpolation_squares_common_supported": int(supported.sum()),
        "interpolation_squares_excessive_tolerance": int(excessive.sum()),
        "interpolation_squares_reliable_before_component": int(reliable.sum()),
        "interpolation_squares_reliable": int(connected.sum()),
        "reliable_area_mm2": float(connected.sum() * parameters["cell_size_mm"] ** 2),
        "reliable_fraction_of_roi": float(connected.sum() / tolerance.size),
        "tolerance_quantiles_mm": np.quantile(tolerance[connected], [0, 0.5, 0.95, 1]).tolist(),
    }
    _validated_model(result)
    return result


def select_model(model, exposure_ms=None):
    """Return the shared pooled field, or select a measured/interior exposure.

    Pooled models ignore the requested exposure and need no acquisition metadata.
    Heights are blended only on common observed vertices. Square tolerances are
    the endpoint maximum plus the configured interpolation margin. A tolerance
    over the original fitting cap makes the square unknown, never clipped.
    Derived intermediate wrappers are for classification, not saved calibration:
    their diagnostics describe coverage and deliberately contain no fitted
    capture count. Validate and store the original bundle instead.
    """
    validate_bundle(model)
    if "exposure_settings_ms" in model:
        # The capture setting is provenance only: pooling fits the same geometry
        # and residual envelope for every request, including unspecified settings.
        result = deepcopy(model)
        result["exposure_selection"] = {
            "mode": "pooled", "exposure_settings_ms": deepcopy(model["exposure_settings_ms"]),
        }
        return result
    exposure = _finite(exposure_ms, "Requested exposure_ms", positive=True)
    if model["model_type"] != MODEL_TYPE:
        measured = float(model["exposure_ms"])
        if not np.isclose(exposure, measured, atol=EXPOSURE_ATOL_MS, rtol=0):
            raise HeightFieldError(f"Requested exposure differs from model ({measured:g} ms); no extrapolation is supported")
        result = deepcopy(model)
        result["exposure_selection"] = {
            "requested_exposure_ms": exposure, "lower_exposure_ms": measured,
            "upper_exposure_ms": measured, "weight": 0.0, "mode": "single",
        }
        return result
    profiles = model["profiles"]
    anchors = np.asarray([profile["exposure_ms"] for profile in profiles], dtype=float)
    matches = np.flatnonzero(np.isclose(anchors, exposure, atol=EXPOSURE_ATOL_MS, rtol=0))
    result = {key: deepcopy(value) for key, value in model.items()
              if key not in ("profiles", "interpolation")}
    result["model_type"] = LEGACY_HEIGHT_FIELD_TYPE
    if len(matches):
        index = int(matches[0])
        lower = upper = float(anchors[index])
        weight, mode = 0.0, "exact"
        result["bins"] = deepcopy(profiles[index]["bins"])
        result["training_captures"] = deepcopy(profiles[index]["training_captures"])
        result["exposure_ms"] = lower
    else:
        index = int(np.searchsorted(anchors, exposure))
        if index == 0 or index == len(anchors):
            raise HeightFieldError(f"Requested exposure {exposure:g} ms is outside measured profile range "
                                   f"[{anchors[0]:g}, {anchors[-1]:g}] ms; no extrapolation is supported")
        lower, upper = float(anchors[index - 1]), float(anchors[index])
        weight, mode = (exposure - lower) / (upper - lower), "interpolated"
        margin = float(model["interpolation"]["margin_mm"])
        result["bins"] = {name: _blend_field(profiles[index - 1]["bins"][name],
                                             profiles[index]["bins"][name], weight, margin)
                          for name in profiles[index]['bins']}
        result["training_captures"] = deepcopy(profiles[index - 1]["training_captures"]
                                                + profiles[index]["training_captures"])
        result["exposure_ms"] = exposure
        result["validation_status"] = "requires_held_out_interval_review"
    result["exposure_selection"] = {
        "requested_exposure_ms": exposure, "lower_exposure_ms": lower,
        "upper_exposure_ms": upper, "weight": weight, "mode": mode,
    }
    return result
