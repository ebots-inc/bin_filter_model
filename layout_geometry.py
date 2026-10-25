"""Reviewable layout estimates from an empty, calibrated scene.

An operator marks each complete bin in a metric top view. The measured cloud
supplies all numeric fitting bounds; no dimensions are inferred from a picture
without calibration. This is setup evidence, never training or acceptance data.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import math
import re

import cv2
import numpy as np

from dataset import DatasetError
import height_field as hf
import modern_model


def _cloud(points):
    try:
        result = np.asarray(points, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise DatasetError('The empty-scene preview needs calibrated XYZ depth points') from error
    if result.ndim != 2 or result.shape[1] != 3:
        raise DatasetError('The empty-scene preview needs calibrated XYZ depth points')
    result = result[np.isfinite(result).all(axis=1)]
    if len(result) < 500:
        raise DatasetError('Too little valid depth. Clear the camera view and capture the empty bins again.')
    return result


def _bounds(value):
    try:
        bounds = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise DatasetError('Draw a rectangle around the complete bin, including its top rim.') from error
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
        raise DatasetError('Draw a rectangle around the complete bin, including its top rim.')
    extent = bounds[1] - bounds[0]
    if np.any(extent < 8) or np.prod(np.ceil(extent)) > 1000000:
        raise DatasetError('The marked region is too small or too large. Mark one complete bin.')
    return bounds


def pixels_to_bounds(rectangle, preview):
    """Convert original-image pixel corners to robot XY millimetres."""
    try:
        rectangle = np.asarray(rectangle, dtype=float).reshape(2, 2)
        width, height = int(preview['width']), int(preview['height'])
        scale, margin = float(preview['pixels_per_mm']), float(preview['margin_px'])
        lower = np.asarray(preview['lower_xy_mm'], dtype=float)
    except (ValueError, TypeError, KeyError) as error:
        raise DatasetError('Capture an empty-scene preview before marking bins.') from error
    if (not np.isfinite(rectangle).all() or not math.isfinite(scale) or scale <= 0
            or np.any(rectangle < 0) or np.any(rectangle > [width - 1, height - 1])):
        raise DatasetError('Keep each bin rectangle inside the preview.')
    xy = rectangle.copy()
    xy[:, 0] = (rectangle[:, 0] - margin) / scale + lower[0]
    xy[:, 1] = (height - 1 - rectangle[:, 1] - margin) / scale + lower[1]
    return _bounds([xy.min(axis=0), xy.max(axis=0)]).tolist()


def _pixels(xy, viewport):
    result = (np.asarray(xy, dtype=float) - viewport['lower_xy_mm']) * viewport['pixels_per_mm'] + viewport['margin_px']
    result[..., 1] = viewport['height'] - 1 - result[..., 1]
    return np.rint(result).astype(int)


def render_preview(points, max_size=1000, *, viewport=None, bins=None):
    """Render measured height in a top view; optional overlays share its mapping.

    The tiny extreme tail of XY observations is omitted from the initial view to
    prevent isolated bad-depth points from shrinking the useful scene to a dot.
    This changes only the display, never the saved cloud or selected-bin fit.
    """
    points = _cloud(points)
    if viewport is None:
        lower, upper = np.quantile(points[:, :2], [.0005, .9995], axis=0)
        span = upper - lower
        if np.any(span < 1):
            raise DatasetError('The depth preview has no usable surface. Capture the empty bins again.')
        margin = 30
        scale = (max_size - 2 * margin - 1) / max(span)
        width, height = np.ceil(span * scale).astype(int) + 2 * margin + 1
        viewport = {'width': int(width), 'height': int(height), 'lower_xy_mm': lower.tolist(),
                    'upper_xy_mm': upper.tolist(), 'pixels_per_mm': float(scale), 'margin_px': margin}
    else:
        viewport = {key: deepcopy(viewport[key]) for key in
                    ('width', 'height', 'lower_xy_mm', 'upper_xy_mm', 'pixels_per_mm', 'margin_px')}
    image = np.full((viewport['height'], viewport['width'], 3), 27, dtype=np.uint8)
    # Display one representative height per image pixel, rather than drawing all
    # 4+ million observations or allowing one bad high point to cover a pixel.
    pixels = _pixels(points[:, :2], viewport)
    valid = np.all((pixels >= [0, 0]) & (pixels < [viewport['width'], viewport['height']]), axis=1)
    pixels, heights = pixels[valid], points[valid, 2]
    count = viewport['width'] * viewport['height']
    median, _ = hf._group_quantile(pixels[:, 1] * viewport['width'] + pixels[:, 0], heights, count, .5, 1)
    selected = np.isfinite(median)
    if not selected.any():
        raise DatasetError('The preview contains no visible depth. Capture the empty bins again.')
    z_low, z_high = np.quantile(median[selected], [.01, .99])
    values = np.clip((median[selected] - z_low) / max(z_high - z_low, 1) * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(values[:, None], cv2.COLORMAP_TURBO)[:, 0]
    image.reshape(-1, 3)[selected] = colors
    if bins:
        # Distinguish review evidence from the original height color ramp.
        # Otherwise orange high surfaces are easily mistaken for detected rim.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor((gray * .45).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for item in bins or []:
        for key, color in (('interior_xyz_mm', (255, 220, 20)), ('rim_xyz_mm', (0, 150, 255))):
            values = np.asarray(item.get(key, []), dtype=float)
            if not len(values):
                continue
            locations = _pixels(values[:, :2], viewport)
            for offset in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)):
                marks = locations + offset
                keep = np.all((marks >= [0, 0]) & (marks < [viewport['width'], viewport['height']]), axis=1)
                image[marks[keep, 1], marks[keep, 0]] = color
        corners = _pixels(item['xy_bounds_mm'], viewport)
        lo, hi = corners.min(axis=0), corners.max(axis=0)
        cv2.rectangle(image, tuple(lo), tuple(hi), (245, 245, 245), 2)
        cv2.putText(image, item['bin_id'], (int(lo[0]) + 4, int(lo[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 2)
        seed = _pixels(item['seed_xy_mm'], viewport)
        cv2.drawMarker(image, tuple(seed), (255, 255, 255), cv2.MARKER_CROSS, 13, 2)
    cv2.putText(image, '+X right / +Y up', (8, viewport['height'] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, .4, (240, 240, 240), 1)
    success, encoded = cv2.imencode('.png', image)
    if not success:
        raise DatasetError('Could not create the empty-scene preview')
    return viewport | {'preview_png': base64.b64encode(encoded.tobytes()).decode('ascii'),
                       'caption': 'Calibrated top view: height colors show measured depth. Mark the complete rim of each empty bin.'}


def _surface(points, bounds):
    selected = points[hf._inside(points, bounds)]
    if len(selected) < 1500:
        raise DatasetError('Too little depth inside this rectangle. Mark the whole bin or capture again.')
    shape = np.ceil(bounds[1] - bounds[0]).astype(int)
    cell = np.floor(selected[:, :2] - bounds[0]).astype(int)
    values, count = hf._group_quantile(cell[:, 1] * shape[0] + cell[:, 0], selected[:, 2], int(np.prod(shape)), .5, 3)
    surface = values.reshape(tuple(shape[::-1]))
    if np.isfinite(surface).sum() < 500:
        raise DatasetError('The bin depth is too sparse. Clear the view and capture the empty bin again.')
    return surface, count.reshape(surface.shape)


def infer_bin(points, xy_bounds_mm, *, min_capture_support=5):
    """Estimate one region using a measured flat lip surrounding lower geometry.

    The native model's actual flat-lip/boundary checks are reused. Empty table
    patches and cropped rims therefore cannot pass simply because they have a
    plausible height. The result remains a draft requiring operator review.
    """
    points, bounds = _cloud(points), _bounds(xy_bounds_mm)
    surface, support = _surface(points, bounds)
    yy, xx = np.indices(surface.shape)
    x = bounds[0, 0] + xx + .5
    y = bounds[0, 1] + yy + .5
    center, extent = bounds.mean(axis=0), bounds[1] - bounds[0]
    radius = np.maximum(abs((x - center[0]) / (extent[0] / 2)), abs((y - center[1]) / (extent[1] / 2)))
    gy, gx = np.gradient(surface)
    flat = np.isfinite(surface) & (np.hypot(gx, gy) <= .35)
    peripheral = flat & (radius >= .55)
    values = np.sort(surface[peripheral])
    if len(values) < 200:
        raise DatasetError('No complete flat top rim was found. Include the whole rim and remove any parts or hands.')
    # Separate high flat lip from the lower table/floor. A small number of high
    # noise cells cannot create a supported candidate height band.
    groups = np.split(values, np.flatnonzero(np.diff(values) > 2.) + 1)
    groups = [group for group in groups if len(group) >= 200]
    diagnostics = []
    for group in reversed(groups):
        rim_low, rim_high = np.quantile(group, [.005, .995])
        if rim_high - rim_low > max(12., .12 * max(extent)):
            continue
        # Exclude the rim from background fitting with a modest measured-height
        # separation, while retaining its physical neighboring drop for ICP.
        lower_band, upper_band = float(rim_low - 3.), float(rim_high + 3.)
        interior = np.isfinite(surface) & (surface < lower_band)
        eligible = interior & (radius < .6)
        if np.count_nonzero(eligible) < 100:
            continue
        interior_depth = float(np.median(surface[eligible]))
        if float(np.median(group)) - interior_depth < 5.:
            continue
        # A measured interior patch with observed neighbors makes a reliable
        # seed; a geometric rectangle center might instead fall in a depth hole.
        distances = cv2.distanceTransform(interior.astype(np.uint8), cv2.DIST_L2, 5)
        distances[~eligible] = 0
        sy, sx = np.unravel_index(np.argmax(distances), surface.shape)
        if distances[sy, sx] < 3:
            continue
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(interior.astype(np.uint8), 4)
        label = labels[sy, sx]
        main = labels == label
        main_size = int(main.sum())
        if max(main[0].sum(), main[-1].sum(), main[:, 0].sum(), main[:, -1].sum()) >= 3:
            diagnostics.append('The rectangle cuts into the lower interior. Draw just outside the complete top rim.')
            continue
        # Two substantial lower pockets inside one rectangle are ambiguous.
        # Background outside the rim sits at the rectangle border and is ignored.
        separate = []
        for other in range(1, len(stats)):
            if other == label or stats[other, cv2.CC_STAT_AREA] < max(300, main_size * .2):
                continue
            left, top, width, height = stats[other, :4]
            if left == 0 or top == 0 or left + width == surface.shape[1] or top + height == surface.shape[0]:
                continue
            cx, cy = centroids[other]
            if (.12 * surface.shape[1] < cx < .88 * surface.shape[1]
                    and .12 * surface.shape[0] < cy < .88 * surface.shape[0]):
                separate.append(other)
        if separate:
            diagnostics.append('This rectangle contains more than one lower pocket. Mark each bin separately.')
            continue
        seed = [float(x[sy, sx]), float(y[sy, sx])]
        z_bottom = float(np.quantile(surface[main], .001) - 3.)
        z_top = upper_band
        config = {key: deepcopy(value) for key, value in hf.DEFAULTS.items()}
        config.update(kind='height_field', xy_bounds_mm=bounds.tolist(), seed_xy_mm=seed,
                      z_bounds_mm=[z_bottom, z_top], min_capture_support=min_capture_support,
                      max_reference_height_mm=lower_band,
                      registration={'height_band_mm': [lower_band, upper_band],
                                    'max_slope': .35, 'min_surface_points': 200,
                                    'min_boundary_points': 80, 'min_quadrant_points': 20})
        try:
            hf._configuration({key: value for key, value in config.items() if key != 'registration'})
            measured = modern_model._measured_rim(surface, support, config, np.array([lower_band, upper_band]), config['registration'])
        except (DatasetError, hf.HeightFieldError) as error:
            diagnostics.append(str(error))
            continue
        # Highlight only genuinely observed interior cells in review, not an
        # imagined solid polygon. Downsample overlays to bound response size.
        iy, ix = np.nonzero(main)
        stride = max(1, math.ceil(len(iy) / 6000))
        interior_points = np.column_stack((x[iy, ix], y[iy, ix], surface[iy, ix]))[::stride]
        return {'config': config, 'diagnostics': {
            'xy_bounds_mm': bounds.tolist(), 'seed_xy_mm': seed,
            'width_mm': float(extent[0]), 'depth_mm': float(extent[1]),
            'height_mm': float(np.median(group) - np.quantile(surface[main], .01)),
            'rim_height_mm': float(np.median(group)), 'rim_points': measured['diagnostics']['surface_points'],
            'cell_size_mm': config['cell_size_mm'], 'rim_height_band_mm': [lower_band, upper_band],
            'boundary_points': measured['diagnostics']['boundary_points'], 'interior_points': main_size,
            'rim_quadrants': measured['diagnostics']['surface_quadrants'],
            'boundary_quadrants': measured['diagnostics']['boundary_quadrants'],
            'rim_xyz_mm': measured['surface_xyz_mm'], 'interior_xyz_mm': interior_points.tolist(),
            'notes': ['Draft estimated from measured empty-scene depth; review the highlighted rim and interior before saving.',
                      'Training and held-out validation still determine whether the finished model can be used.']}}
    detail = (' ' + diagnostics[0]) if diagnostics else ''
    raise DatasetError('Could not identify a complete top rim around a lower bin interior. '
                       'Empty the bin, include the whole rim, exclude neighboring bins, and capture again.' + detail)


def infer_layout(points, selections, scene_id, frame_id, *, min_capture_support=5):
    if (isinstance(scene_id, bool) or not isinstance(scene_id, int) or not 0 <= scene_id <= 255
            or not isinstance(frame_id, str) or not frame_id.strip()):
        raise DatasetError('Select the calibrated scene and robot frame before creating a layout.')
    if (isinstance(min_capture_support, bool) or not isinstance(min_capture_support, int)
            or min_capture_support < 2):
        raise DatasetError('Layout fitting requires support from at least two training captures.')
    if not isinstance(selections, list) or not 1 <= len(selections) <= 32:
        raise DatasetError('Mark at least one bin (up to 32 bins per scene).')
    seen, rectangles = set(), []
    for item in selections:
        if not isinstance(item, dict) or not isinstance(item.get('bin_id'), str):
            raise DatasetError('Give each marked bin a unique name.')
        name = item['bin_id']
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', name) or name in seen:
            raise DatasetError('Use unique bin names starting with a letter, followed by letters, numbers, underscores or hyphens.')
        bounds = _bounds(item.get('xy_bounds_mm'))
        for previous in rectangles:
            overlap = np.maximum(0, np.minimum(bounds[1], previous[1]) - np.maximum(bounds[0], previous[0]))
            if np.prod(overlap) > .01 * min(np.prod(bounds[1] - bounds[0]), np.prod(previous[1] - previous[0])):
                raise DatasetError('Bin rectangles overlap. Mark each physical bin separately.')
        seen.add(name)
        rectangles.append(bounds)
    bins, summaries = {}, []
    points = _cloud(points)
    for item, bounds in zip(selections, rectangles):
        try:
            result = infer_bin(points, bounds, min_capture_support=min_capture_support)
        except DatasetError as error:
            raise DatasetError(f"{item['bin_id']}: {error}") from error
        bins[item['bin_id']] = result['config']
        summaries.append({'bin_id': item['bin_id']} | result['diagnostics'])
    return {'settings': {'model_type': modern_model.MODEL_TYPE, 'scene_id': scene_id,
                         'frame_id': frame_id, 'bins': bins}, 'bins': summaries,
            'notes': ['Orange: measured flat rim. Cyan: observed lower interior. White cross: automatically chosen interior seed.',
                      'Confirm every rectangle encloses one empty bin, including the whole physical rim.']}
