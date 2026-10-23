"""Recorded-depth quality diagnostics; no acceptance or part-truth inference."""
from __future__ import annotations

import numpy as np

import height_field


def describe_views(views, model, label):
    parameters, bounds, heights, tolerance = height_field._validated_model(model)
    total_squares = int(np.isfinite(tolerance).sum())
    result = []
    for view in views:
        points = view['points']
        inside = height_field._inside(points, bounds)
        roi = points[inside]
        prediction, ids = height_field._interpolate(roi, bounds, parameters['cell_size_mm'], heights)
        thresholds = np.full(len(roi), np.nan)
        valid_ids = ids >= 0
        thresholds[valid_ids] = tolerance.ravel()[ids[valid_ids]]
        known = np.isfinite(prediction) & np.isfinite(thresholds)
        residual = roi[known, 2] - prediction[known]
        quantiles = np.quantile(residual, [.01, .5, .99]).tolist() if len(residual) else [None] * 3
        observed = len(np.unique(ids[known]))
        result.append({key: value for key, value in view.items() if key != 'points'} | {
            'roi_point_count': len(roi), 'known_reference_point_count': int(known.sum()),
            'unknown_point_count': int((~known).sum()),
            'unknown_fraction_of_roi': float((~known).mean()) if len(roi) else None,
            'supported_squares_total': total_squares, 'supported_squares_observed': observed,
            'observed_support_fraction': observed / total_squares,
            'signed_residual_quantiles_mm': dict(zip(('p01', 'p50', 'p99'), quantiles)),
            'residual_interpretation': 'empty_background' if label == 'empty' else 'includes_foreground',
        })
    return result


def summarize_exposures(reports):
    groups = {}
    for report in reports:
        key = (report['structured_light_exposure_ms'], report['bin_id'], report['label'])
        if key not in groups:
            groups[key] = {'structured_light_exposure_ms': key[0], 'bin_id': key[1], 'label': key[2],
                           'capture_count': 0, 'counts': {}, 'classified_bin_point_count': 0,
                           'selection_modes': set()}
        group = groups[key]
        group['capture_count'] += 1
        group['classified_bin_point_count'] += report['classified_bin_point_count']
        for reason, count in report['counts'].items():
            group['counts'][reason] = group['counts'].get(reason, 0) + count
        group['selection_modes'].add(report['exposure_selection']['mode'])
    result = []
    for key in sorted(groups):
        group = groups[key]
        considered = group['classified_bin_point_count']
        group['empty_leakage_fraction'] = (group['counts']['kept'] / considered
                                           if considered and group['label'] == 'empty' else None)
        group['selection_modes'] = sorted(group['selection_modes'])
        result.append(group)
    return result
