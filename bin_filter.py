#!/usr/bin/env python3
"""Calibrate and replay named bin filters on recorded data; never moves hardware."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

import dataset as data
import exposure_profiles
import model_io

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent / 'ebots_system_artifacts/artifacts_cell_4'
COLORS = {'invalid': [0, 0, 0], 'outside': [90, 90, 90], 'floor': [60, 120, 230],
          'wall': [230, 150, 30], 'surface': [60, 120, 230],
          'unknown': [190, 70, 210], 'kept': [40, 220, 60]}


def model_backend(model_type):
    if model_type == 'offline_convex_planar_bins':
        import geometry
        return geometry
    if model_type == 'offline_height_field_bins':
        import height_field
        return height_field
    raise data.DatasetError(f'Unsupported model type: {model_type}')


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def seed_config(bin_ids=('bin_a', 'bin_b')):
    """Crop centers provide only rough XY locations, never interior dimensions."""
    config = {'bins': {}}
    data.validate_bin_ids(dict.fromkeys(bin_ids))
    for name in bin_ids:
        config['bins'][name] = {
            'robot_from_bin_seed': [[1, 0, 0, None], [0, 1, 0, None], [0, 0, 1, None], [0, 0, 0, 1]],
            'interior_polygon_mm': None,
            'floor_search_band_mm': 8.0,
            'wall_search_band_mm': 8.0,
            'wall_height_range_mm': [8.0, 60.0],
            'max_height_mm': 180.0,
        }
    return config


def command_init(args):
    data.init_dataset(args.dataset, args.camera_calibration, args.robot_calibration, args.scene_id)
    data.write_yaml(Path(args.dataset) / 'bin_config.yaml', seed_config(getattr(args, 'bins', None) or ['bin_a', 'bin_b']))
    print(f'Created {args.dataset}. Register captures, inspect geometry, and fill the null seed values in bin_config.yaml.')


def command_add(args):
    if args.cloud and not args.unfiltered:
        raise data.DatasetError('Cloud import requires --unfiltered: confirm it is before bin/part filtering')
    if args.record and not args.capture_dir:
        raise data.DatasetError('--record requires --capture-dir')
    labels = {}
    for item in getattr(args, 'bin_labels', None) or []:
        name, separator, label = item.partition('=')
        if not separator or name in labels:
            raise data.DatasetError('Use --bin-label BIN=empty or BIN=loaded once for each bin')
        labels[name] = label
    for name, label in (('bin_a', getattr(args, 'bin_a', None)), ('bin_b', getattr(args, 'bin_b', None))):
        if label is not None:
            if name in labels:
                raise data.DatasetError(f'Duplicate label for {name}')
            labels[name] = label
    result = data.add_capture(args.dataset, args.id, args.split, labels, args.exposure_ms,
                              capture_dir=args.capture_dir, record=args.record,
                              cloud_path=args.cloud, cloud_frame=args.cloud_frame, note=args.note,
                              spotlight_exposure_ms=args.spotlight_exposure_ms, group_id=args.group_id)
    print(f"Registered {result['id']} ({result['split']}); inputs are referenced, not copied.")


def load_points(args):
    manifest, calibration = data.load_dataset(args.dataset)
    entry = data.get_capture(manifest, args.capture)
    points = data.load_capture_points(args.dataset, entry, calibration, args.stride)
    if not len(points):
        raise data.DatasetError('Capture contains no valid reconstructed points')
    return manifest, entry, points


def optional_view(points, colors=None, mesh=None, pick=False):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors) / 255.0)
    if pick:
        viewer = o3d.visualization.VisualizerWithEditing()
        if not viewer.create_window(window_name='Shift+left click: pick; Q: close and print robot XYZ mm'):
            raise data.DatasetError('Cannot open a window; use exported PLY in a point-cloud viewer')
        viewer.add_geometry(cloud)
        viewer.run()
        selected = viewer.get_picked_points()
        viewer.destroy_window()
        for index in selected:
            print(f'Picked robot XYZ mm: {points[index].tolist()}')
    else:
        o3d.visualization.draw_geometries([cloud] + ([mesh] if mesh is not None else []))


def command_inspect(args):
    manifest, entry, points = load_points(args)
    if bool(args.cad) != bool(args.cad_transform):
        raise data.DatasetError('CAD overlay needs BOTH --cad mesh and --cad-transform YAML')
    if not np.isfinite(args.cad_scale_to_mm) or args.cad_scale_to_mm <= 0:
        raise data.DatasetError('--cad-scale-to-mm must be finite and positive')
    output = new_output(args.output)
    data.write_ply(output / 'raw_robot_mm.ply', points, frame_id=manifest['frame_id'])
    report = dict(data.cloud_summary(points, manifest['frame_id']), capture_id=entry['id'], depth_stride=args.stride)
    mesh = None
    if args.cad:
        import open3d as o3d
        if Path(args.cad).suffix.lower() not in ('.stl', '.obj', '.ply'):
            raise data.DatasetError('Export CAD as STL/OBJ/PLY first; STEP is not read by this tool')
        mesh = o3d.io.read_triangle_mesh(args.cad)
        if mesh.is_empty() or not mesh.has_triangles():
            raise data.DatasetError('CAD file has no triangle mesh')
        matrix = data.validate_transform(data.read_yaml(args.cad_transform)['robot_from_cad'], 'robot_from_cad')
        mesh.scale(args.cad_scale_to_mm, center=np.zeros(3))
        mesh.transform(matrix)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.8, 0.35, 0.2])
        if not o3d.io.write_triangle_mesh(str(output / 'cad_robot_mm.ply'), mesh):
            raise data.DatasetError('Could not export CAD overlay')
        report['cad'] = {'path': str(Path(args.cad).resolve()), 'sha256': data.sha256(args.cad),
                         'scale_to_mm': args.cad_scale_to_mm, 'robot_from_cad': matrix.tolist(),
                         'role': 'visual reference only; no automatic CAD registration'}
    data.write_json(output / 'inspection.json', report)
    print(f"{entry['id']}: {len(points)} points; robot bounds {report['min_mm']} to {report['max_mm']} mm")
    print(f'Wrote {output}')
    if args.show or args.pick:
        optional_view(points, mesh=mesh, pick=args.pick)


def training_subset(points, config, max_points):
    if config.get('kind') == 'height_field':
        bounds = np.asarray(config['xy_bounds_mm'], dtype=float)
        z = np.asarray(config['z_bounds_mm'], dtype=float)
        if bounds.shape != (2, 2) or z.shape != (2,):
            raise data.DatasetError('Height field needs xy_bounds_mm [[xmin,ymin],[xmax,ymax]] and z_bounds_mm [min,max]')
        # Residual/support estimates need the actual observed density. Do not
        # apply the planar fitter's random sample cap to height-field inputs.
        return points[np.isfinite(points).all(axis=1) &
                      np.all((points[:, :2] >= bounds[0]) & (points[:, :2] < bounds[1]), axis=1)]
    matrix = data.validate_transform(config['robot_from_bin_seed'], 'robot_from_bin_seed (fill null floor Z)')
    polygon = np.asarray(config['interior_polygon_mm'], dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise data.DatasetError('Fill interior_polygon_mm with actual convex inner-wall XY coordinates')
    local = data.transform_points(points, np.linalg.inv(matrix))
    margin = float(config['wall_search_band_mm'])
    lo = np.r_[polygon.min(axis=0) - margin, -float(config['floor_search_band_mm'])]
    hi = np.r_[polygon.max(axis=0) + margin, float(config['max_height_mm'])]
    points = points[np.all((local >= lo) & (local <= hi), axis=1)]
    if len(points) > max_points:
        indices = np.random.default_rng(42).choice(len(points), max_points, replace=False)
        points = points[indices]
    return points


def command_calibrate(args):
    manifest, calibration = data.load_dataset(args.dataset)
    settings_path = Path(args.config) if args.config else Path(args.dataset) / 'bin_config.yaml'
    settings = data.read_yaml(settings_path)
    settings.pop('schema_version', None)  # Ignore archived configuration version tags.
    configured_bins = data.validate_bin_ids(settings.get('bins'))
    for entry in manifest['captures']:
        data.validate_bin_labels(entry.get('bins'), configured_bins)
    for key in ('frame_id', 'scene_id'):
        if key in settings and settings[key] != manifest[key]:
            raise data.DatasetError(f'Bin configuration {key} differs from dataset')
    cad_plan = None
    if getattr(args, 'cad_reference', None):
        if settings.get('model_type') != 'calibrated_bin_background':
            raise data.DatasetError('--cad-reference requires the calibrated bin-background model')
        import cad_plan as cad_inputs
        cad_plan = cad_inputs.load(args.cad_reference, scene_id=manifest['scene_id'],
                                  frame_id=manifest['frame_id'], bin_ids=configured_bins)
        settings = cad_inputs.apply(settings, cad_plan)
    elif any('cad_reference' in config for config in settings['bins'].values()):
        if settings.get('model_type') != 'calibrated_bin_background':
            raise data.DatasetError('CAD reference settings require the calibrated bin-background model')
        import cad_plan as cad_inputs
        for config in settings['bins'].values():
            if 'cad_reference' in config:
                config['cad_reference'] = cad_inputs.resolve_config(config['cad_reference'], settings_path.parent)
    if settings.get('model_type') == 'calibrated_bin_background':
        return calibrate_modern(args, manifest, calibration, settings, cad_plan=cad_plan)
    kinds = {config.get('kind', 'planar') for config in settings['bins'].values()}
    if len(kinds) != 1 or not kinds.issubset({'planar', 'height_field'}):
        raise data.DatasetError('All bins must use the same supported kind: planar or height_field')
    model_type = 'offline_height_field_bins' if kinds == {'height_field'} else 'offline_convex_planar_bins'
    backend = model_backend(model_type)
    train = [entry for entry in manifest['captures'] if entry['split'] == 'train' and
             all(label == 'empty' for label in entry['bins'].values())]
    if len(train) < 3:
        raise data.DatasetError('At least 3 independent all-empty training captures are required (7 recommended)')
    profiles_enabled = getattr(args, 'exposure_profiles', False)
    pool_exposures = getattr(args, 'pool_exposures', False)
    requested_exposure = getattr(args, 'exposure_ms', None)
    if pool_exposures and (profiles_enabled or requested_exposure is not None):
        raise data.DatasetError('--pool-exposures cannot be combined with --exposure-profiles or --exposure-ms')
    if requested_exposure is not None:
        requested_exposure = data.positive_exposure(requested_exposure)
        train = [entry for entry in train if abs(data.capture_exposure(entry) - requested_exposure) <= data.EXPOSURE_ATOL_MS]
        if not train:
            raise data.DatasetError('No all-empty training captures at the selected exposure')
    groups = []
    for entry in sorted(train, key=data.capture_exposure):
        exposure = data.capture_exposure(entry)
        if not groups or abs(exposure - groups[-1][0]) > data.EXPOSURE_ATOL_MS:
            groups.append((exposure, []))
        groups[-1][1].append(entry)
    if profiles_enabled:
        if kinds != {'height_field'} or len(groups) < 2:
            raise data.DatasetError('--exposure-profiles requires height fields and at least two independently captured exposures')
    elif pool_exposures:
        if kinds != {'height_field'} or len(groups) < 2:
            raise data.DatasetError('--pool-exposures requires height fields and at least two captured exposures')
    elif len(groups) != 1:
        raise data.DatasetError('Empty training captures must use one exposure; select --exposure-ms, --pool-exposures, or --exposure-profiles')
    margin = getattr(args, 'interpolation_margin_mm', 0.5)
    if not np.isfinite(margin) or margin < 0:
        raise data.DatasetError('--interpolation-margin-mm must be finite and nonnegative')
    if args.max_fit_points < 100:
        raise data.DatasetError('--max-fit-points must be at least 100')
    def provenance(entries):
        return [{key: entry[key] for key in ('id', 'input_signature', 'group_id') if key in entry}
                for entry in entries]

    exposure_settings = [exposure for exposure, _ in groups]
    fit_groups = [(None, train)] if pool_exposures else groups
    profiles = []
    for exposure, captures in fit_groups:
        fit_label = 'pooled exposures' if pool_exposures else f'{exposure:g} ms'
        if len(captures) < 3:
            raise data.DatasetError(f'{fit_label} needs at least 3 independent all-empty training captures')
        fitted = {}
        for name, config in settings['bins'].items():
            if kinds == {'height_field'}:
                config = dict(config, frame_id=manifest['frame_id'])
            clouds = []
            for entry in captures:
                points = data.load_capture_points(args.dataset, entry, calibration, args.stride)
                clouds.append(training_subset(points, config, args.max_fit_points))
                print(f"{fit_label} {name}: loaded {entry['id']} ({len(clouds[-1])} fit points)", flush=True)
            fitted[name] = backend.fit_bin(clouds, config)
            print(f'Fitted {name} at {fit_label} using {len(captures)} empty recordings.', flush=True)
        exposure_provenance = ({'exposure_settings_ms': exposure_settings} if pool_exposures
                               else {'exposure_ms': exposure})
        profiles.append({**exposure_provenance, 'training_captures': provenance(captures),
                         'bins': fitted, 'bin_membership_check': backend.validate_disjoint_bins(fitted)})
    implementation_files = ('bin_filter.py', 'dataset.py', 'model_io.py', Path(backend.__file__).name, 'exposure_profiles.py')
    model = {'model_type': model_type,
             'frame_id': manifest['frame_id'], 'length_unit': 'mm', 'scene_id': manifest['scene_id'],
             'created_utc': datetime.now(timezone.utc).isoformat(),
             'synthetic': manifest.get('synthetic', False),
             'online_integration': 'cpp_height_field' if kinds == {'height_field'} else 'not_implemented',
             'validation_status': 'requires_held_out_review',
             'calibration_sha256': {key: item['sha256'] for key, item in manifest['calibration'].items()},
             'settings_sha256': data.sha256(settings_path),
             'implementation_sha256': {name: data.sha256(HERE / name)
                                       for name in implementation_files},
             'depth_stride': args.stride,
             'max_fit_points_per_capture_bin': None if kinds == {'height_field'} else args.max_fit_points,
             'training_captures': provenance(train)}
    if profiles_enabled:
        model.update(model_type=exposure_profiles.MODEL_TYPE, profiles=profiles,
                     interpolation={'method': 'linear_height_max_tolerance', 'margin_mm': margin})
        exposure_profiles.validate_bundle(model)
    else:
        model.update(profiles[0])
        if pool_exposures:
            exposure_profiles.validate_bundle(model)
    output = new_output(args.output)
    shutil.copytree(Path(args.dataset) / 'calibration', output / 'calibration')
    data.write_yaml(output / 'settings.yaml', settings)
    model_path = model_io.write_model(output, model, getattr(args, 'output_format', 'bundle'))
    print(f'Saved {model_path}. Next: validate on held-out recordings; this is not a ROS parameter file.')


def calibrate_modern(args, manifest, calibration, settings, cad_plan=None):
    import modern_model
    training = [entry for entry in manifest['captures'] if entry['split'] == 'train'
                and all(label == 'empty' for label in entry['bins'].values())]
    if not training:
        raise data.DatasetError('Rim-model fitting requires independent all-empty training captures')
    records = []
    for entry in training:
        views = data.load_capture_views(args.dataset, entry, calibration, args.stride)
        for view in views:
            selected = np.zeros(len(view['points']), dtype=bool)
            for config in settings['bins'].values():
                if config.get('coordinate_frame') == 'bin':
                    selected[:] = True
                    break
                bounds = np.asarray(config['xy_bounds_mm'])
                selected |= np.all((view['points'][:, :2] >= bounds[0] - 15)
                                   & (view['points'][:, :2] <= bounds[1] + 15), axis=1)
            view['points'] = view['points'][selected]
        records.append(dict(entry, views=views,
                            acquisition=entry.get('source_metadata', {}).get('requested_settings', {})))
        print(f"Loaded independent camera measurements: {entry['id']}", flush=True)
    model = modern_model.fit_model(records, settings, frame_id=manifest['frame_id'],
                                   model_name=getattr(args, 'model_name', None) or Path(args.output).name)
    model.update(scene_id=manifest['scene_id'], synthetic=manifest.get('synthetic', False),
                 created_utc=datetime.now(timezone.utc).isoformat(),
                 training_captures=[{key: entry[key] for key in ('id', 'group_id') if key in entry}
                                    for entry in training],
                 calibration={'camera': 'calibration/camera.yaml', 'robot': 'calibration/robot.yaml'})
    output = new_output(args.output)
    shutil.copytree(Path(args.dataset) / 'calibration', output / 'calibration')
    if any('cad_reference' in item for item in model['bins'].values()):
        import cad_plan as cad_inputs
        settings = cad_inputs.package(output, model, settings, cad_plan)
    data.write_yaml(output / 'settings.yaml', settings)
    path = model_io.write_model(output, model, getattr(args, 'output_format', 'bundle'))
    print(f'Saved rim, interior and camera measurement profiles to {path}', flush=True)


def load_model(args, manifest):
    model = model_io.load_model(args.model)
    if model.get('model_type') == 'calibrated_bin_background':
        import modern_model
        modern_model.validate_model(model)
        if model.get('frame_id') != manifest['frame_id'] or model.get('scene_id') != manifest['scene_id']:
            raise data.DatasetError('Model frame/scene does not match the dataset')
        folder = Path(args.model) if Path(args.model).is_dir() else Path(args.model).parent
        if 'calibration_reference' in model:
            snapshots = {name: data.read_yaml(data.resolve(args.dataset, manifest['calibration'][name]['path']))
                         for name in ('camera', 'robot')}
            current = model_io.calibration_values(snapshots['camera'], snapshots['robot'],
                                                  manifest['scene_id'], manifest['frame_id'])
            if model_io.calibration_reference(args.model, model) != current:
                raise data.DatasetError('Model calibration differs from the dataset snapshots')
        else:
            # Offline candidates retain full capture-time snapshots. Runtime
            # exports carry only the relevant values in model.json instead.
            for name, entry in manifest['calibration'].items():
                if data.read_yaml(folder / f'calibration/{name}.yaml') != data.read_yaml(data.resolve(args.dataset, entry['path'])):
                    raise data.DatasetError('Model calibration differs from the dataset snapshots')
        return model
    is_bundle = model.get('model_type') == exposure_profiles.MODEL_TYPE
    if (model.get('model_type') not in (
            'offline_convex_planar_bins', 'offline_height_field_bins', exposure_profiles.MODEL_TYPE) or
            model.get('frame_id') != manifest['frame_id'] or model.get('length_unit') != 'mm' or
            model.get('scene_id') != manifest['scene_id']):
        raise data.DatasetError('Model frame/units/scene do not match dataset')
    expected = {key: entry['sha256'] for key, entry in manifest['calibration'].items()}
    if model.get('calibration_sha256') != expected:
        raise data.DatasetError('Model calibration differs from this dataset; recalibrate or use matching snapshots')
    if is_bundle or 'exposure_settings_ms' in model:
        exposure_profiles.validate_bundle(model)
    else:
        data.validate_bin_ids(model.get('bins'))
        model_backend(model['model_type']).validate_disjoint_bins(model['bins'])
    return model


def filter_output(output, points, model, bin_id, capture, max_preview=200000, views=None):
    if bin_id not in model['bins'] or bin_id not in capture['bins']:
        raise data.DatasetError(f'Unknown bin {bin_id!r} in model or capture')
    modern_diagnostics = None
    foreground = None
    if model['model_type'] == 'calibrated_bin_background':
        import modern_model
        if views is None:
            raise data.DatasetError('Camera-separated measurements are required for this model')
        classified = modern_model.classify_views(model, bin_id, views, data.capture_exposure(capture))
        keep, masks = classified['keep'], classified['masks']
        modern_diagnostics = classified.get('camera_diagnostics')
        foreground = classified['points']
    else:
        keep, masks = model_backend(model['model_type']).classify(points, model['bins'][bin_id])
    output.mkdir(parents=True, exist_ok=False)
    data.write_ply(output / 'foreground_robot_mm.ply', points[keep] if foreground is None else foreground,
                   frame_id=model['frame_id'])
    preview_indices = np.arange(len(points))
    if len(points) > max_preview:
        preview_indices = np.random.default_rng(42).choice(len(points), max_preview, replace=False)
    colors = np.zeros((len(preview_indices), 3), dtype=np.uint8)
    for reason, mask in masks.items():
        if reason in COLORS:
            colors[mask[preview_indices]] = COLORS[reason]
    data.write_ply(output / 'classified_preview_robot_mm.ply', points[preview_indices], colors,
                   frame_id=model['frame_id'])
    counts = {reason: int(np.count_nonzero(mask)) for reason, mask in masks.items()}
    considered = sum(counts.get(key, 0) for key in ('floor', 'wall', 'surface', 'kept'))
    report = {'capture_id': capture['id'], 'bin_id': bin_id, 'label': capture['bins'][bin_id],
              'structured_light_exposure_ms': data.capture_exposure(capture),
              'spotlight_exposure_ms': data.capture_spotlight_exposure(capture),
              'group_id': capture.get('group_id'), 'exposure_selection': model.get('exposure_selection'),
              'counts': counts, 'preview_points': len(preview_indices), 'colors_rgb': COLORS,
              'classified_bin_point_count': considered,
              'unknown_point_count': counts.get('unknown', 0),
              'foreground_point_count': int(keep.sum()),
              'empty_leakage_fraction': float(keep.sum() / considered) if considered and capture['bins'][bin_id] == 'empty' else None,
              'review_required': True,
              'interpretation': 'Geometry-only counts. No summit, grasp-success, or part-retention accuracy is inferred.'}
    if views is not None and model['model_type'] == 'offline_height_field_bins':
        from exposure_diagnostics import describe_views
        report['camera_diagnostics'] = describe_views(views, model['bins'][bin_id], capture['bins'][bin_id])
    if modern_diagnostics is not None:
        report['camera_diagnostics'] = modern_diagnostics
    data.write_json(output / 'report.json', report)
    return report


def check_exposure(entry, model):
    data.capture_spotlight_exposure(entry)
    if model.get('model_type') == 'calibrated_bin_background':
        return dict(model, exposure_selection={'mode': 'per_camera_measured_profile',
                                                'exposure_ms': data.capture_exposure(entry)})
    return exposure_profiles.select_model(model, data.capture_exposure(entry))


def command_filter(args):
    manifest, calibration = data.load_dataset(args.dataset)
    entry = data.get_capture(manifest, args.capture)
    model = check_exposure(entry, load_model(args, manifest))
    views = data.load_capture_views(args.dataset, entry, calibration, args.stride)
    points = np.concatenate([view['points'] for view in views])
    if not len(points):
        raise data.DatasetError('Capture contains no valid reconstructed points')
    report = filter_output(Path(args.output), points, model, args.bin, entry, views=views)
    print(f"{entry['id']} {args.bin}: kept {report['foreground_point_count']} / {len(points)} points. Wrote {args.output}")


def command_validate(args):
    manifest, calibration = data.load_dataset(args.dataset)
    model = load_model(args, manifest)
    entries = [entry for entry in manifest['captures'] if entry['split'] == args.split]
    if not entries:
        raise data.DatasetError(f'No {args.split} captures registered')
    trained = {entry['input_signature'] for entry in model['training_captures'] if 'input_signature' in entry}
    if args.split == 'test' and any(entry['input_signature'] in trained for entry in entries):
        raise data.DatasetError('Test set contains inputs used for fitting; collect independent test recordings')
    trained_groups = {entry['group_id'] for entry in model['training_captures'] if entry.get('group_id')}
    if args.split == 'test' and any(entry.get('group_id') in trained_groups for entry in entries):
        raise data.DatasetError('Test set contains a capture group used for fitting; hold out complete layouts/sessions')
    # Validate every requested exposure before creating any outputs. Keep only
    # one selected model in memory while replaying full-resolution captures.
    checked_exposures = set()
    expected_bins = model.get('bins') or model.get('profiles', [{}])[0].get('bins')
    for entry in entries:
        data.validate_bin_labels(entry.get('bins'), expected_bins)
        data.capture_spotlight_exposure(entry)
        exposure = data.capture_exposure(entry)
        if exposure not in checked_exposures:
            check_exposure(entry, model)
            checked_exposures.add(exposure)
    output = new_output(args.output)
    reports, overlaps = [], []
    for entry in entries:
        selected = check_exposure(entry, model)
        views = data.load_capture_views(args.dataset, entry, calibration, args.stride)
        points = np.concatenate([view['points'] for view in views])
        if selected['model_type'] == 'calibrated_bin_background':
            import modern_model
            kept = [modern_model.classify_views(selected, name, views, data.capture_exposure(entry))['keep']
                    for name in selected['bins']]
        else:
            classify = model_backend(selected['model_type']).classify
            kept = [classify(points, item)[0] for item in selected['bins'].values()]
        overlaps.append({'capture_id': entry['id'], 'kept_by_multiple_bins': int(np.count_nonzero(np.sum(kept, axis=0) > 1))})
        for name in selected['bins']:
            report = filter_output(output / entry['id'] / name, points, selected, name, entry, views=views)
            reports.append(report)
            print(f"{entry['id']} {name}: kept {report['foreground_point_count']} points ({report['label']})", flush=True)
    from exposure_diagnostics import summarize_exposures
    report = {'split': args.split, 'capture_count': len(entries),
              'depth_stride': args.stride, 'model_sha256': data.sha256(
                  Path(args.model) / 'model.json' if Path(args.model).is_dir() else args.model),
              'status': 'manual_review_required', 'results': reports,
              'exposure_summary': summarize_exposures(reports),
              'bin_membership_overlap': overlaps,
              'review': ['Empty bins: inspect any green retained surfaces.',
                         'Height fields: purple unknown regions are excluded; check missing surface coverage.',
                         'Loaded bins: compare green points with raw cloud and spotlight images; do real parts survive?',
                         'Check each bin separately, particularly walls, thin parts, and any shared boundary.',
                         'These counts alone do not measure false targets, part retention, or pick success.']}
    data.write_json(output / 'validation.json', report)
    print(f'Wrote {output / "validation.json"}; inspect the colored PLYs before accepting the model.')


def command_convert_model(args):
    model = model_io.load_model(args.model)
    output = new_output(args.output)
    source = Path(args.model) if Path(args.model).is_dir() else Path(args.model).parent
    if (source / 'calibration').is_dir():
        shutil.copytree(source / 'calibration', output / 'calibration')
    settings = (model_io._without_versions(data.read_yaml(source / 'settings.yaml'))
                if (source / 'settings.yaml').is_file() else None)
    if any('cad_reference' in item for item in model.get('bins', {}).values()):
        import cad_plan as cad_inputs
        settings = cad_inputs.package(output, model, settings or {}, source_root=source)
    if settings is not None:
        data.write_yaml(output / 'settings.yaml', settings)
    print(f'Saved {model_io.write_model(output, model, args.output_format)}')


def command_import_session(args):
    result = data.import_session(args.session, args.dataset)
    print(f"Registered {len(result['captures'])} captures in {args.dataset}; exposure values are requested, not sensor verified.")



def command_build_session(args):
    """Build a reviewable candidate from one complete collection session."""
    output = Path(args.output).resolve()
    if output.exists():
        raise data.DatasetError(f'Refusing to overwrite existing output: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bin-build-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'result'
        stage.mkdir()
        manifest = data.import_session(args.session, stage / 'dataset')
        training = [entry for entry in manifest['captures'] if entry['split'] == 'train'
                    and all(label == 'empty' for label in entry['bins'].values())]
        exposures = {data.capture_exposure(entry) for entry in training}
        common = dict(dataset=str(stage / 'dataset'), stride=args.stride)
        command_calibrate(argparse.Namespace(**common, config=None, output=str(stage / 'model'),
                          max_fit_points=200000, pool_exposures=len(exposures) > 1,
                          output_format='bundle', model_name=getattr(args, 'model_name', None) or output.name,
                          cad_reference=getattr(args, 'cad_reference', None)))
        command_validate(argparse.Namespace(**common, model=str(stage / 'model/model.json'),
                         output=str(stage / 'validation'), split='test'))
        acceptance = None
        model = model_io.load_model(stage / 'model')
        if model.get('model_type') == 'calibrated_bin_background':
            from production_validation import validate_production
            config = getattr(args, 'pipeline_config', None)
            if not config:
                raise data.DatasetError('Rim-model building requires --pipeline-config for production replay')
            acceptance = validate_production(stage / 'dataset', stage / 'model/model.json', config,
                                              stage / 'production_validation',
                                              getattr(args, 'replay_executable', None))
        build_report = {
            'status': acceptance['status'] if acceptance else 'manual_review_required',
            'dataset': 'dataset/dataset.yaml', 'model': 'model/model.json',
            'validation': 'validation/validation.json',
            'training_exposures_ms': sorted(exposures),
            'exposure_evidence': 'requested_not_sensor_verified',
            'activation': 'not_activated',
            'production_acceptance': 'production_validation/acceptance.json' if acceptance else None,
            'loaded_validation_captures': sum(entry['split'] == 'test' and
                any(label == 'loaded' for label in entry['bins'].values()) for entry in manifest['captures']),
        }
        from cad_plan import build_metadata
        cad_metadata = build_metadata(model)
        if cad_metadata is not None:
            build_report['cad_reference'] = cad_metadata
        data.write_json(stage / 'build.json', build_report)
        stage.rename(output)
    print(f'Built {output / "model/model.json"}; review {output / "validation/validation.json"}. '
          'A successful build is not factory acceptance and does not activate the model.')


def command_production_validate(args):
    from production_validation import validate_production
    report = validate_production(args.dataset, args.model, args.pipeline_config, args.output,
                                 args.replay_executable, quick=args.quick)
    print(f"Production {'check' if args.quick else 'acceptance'}: {report['status']}; {args.output}/acceptance.json")


def runtime_reload():
    from runtime_reload_client import reload_model
    return reload_model()


def command_activate(args):
    import deployment
    result = deployment.activate(args.candidate, args.models_root, args.pipeline_config, runtime_reload,
                                 state_root=args.deployment_root)
    print(f"Activated and verified {result['model_path']}")


def command_rollback(args):
    import deployment
    result = deployment.rollback(args.models_root, args.pipeline_config, runtime_reload,
                                 state_root=args.deployment_root)
    print(f"Restored and verified {result['model_path']}")

def command_demo(args):
    from demo import generate_demo
    root = generate_demo(Path(args.output))
    common = dict(dataset=str(root / 'dataset'), stride=1)
    command_calibrate(argparse.Namespace(**common, config=None, output=str(root / 'model'), max_fit_points=200000))
    command_validate(argparse.Namespace(**common, model=str(root / 'model/model.json'),
                                        output=str(root / 'validation'), split='test'))
    print(f'Synthetic-only demo complete: {root}. These models are NOT Cell 4 calibrations.')


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='Create a new dataset with frozen calibration copies')
    init.add_argument('--dataset', required=True)
    init.add_argument('--camera-calibration', default=str(ARTIFACTS / 'calibration/deployment/camera/scene0_denali0_params.yaml'))
    init.add_argument('--robot-calibration', default=str(ARTIFACTS / 'calibration/deployment/robot/B2_params.yaml'))
    init.add_argument('--scene-id', type=int, default=0)
    init.add_argument('--bins', nargs='+', help='Bin IDs to configure; defaults to bin_a bin_b for existing setups')
    init.set_defaults(function=command_init)
    add = commands.add_parser('add', help='Register one recorded stationary scene; references source files')
    add.add_argument('--dataset', required=True)
    source = add.add_mutually_exclusive_group(required=True)
    source.add_argument('--capture-dir')
    source.add_argument('--cloud')
    add.add_argument('--record', help='Exact contour metadata YAML, relative to capture-dir or absolute')
    add.add_argument('--cloud-frame', choices=['robot', 'vision'])
    add.add_argument('--unfiltered', action='store_true')
    add.add_argument('--id', required=True)
    add.add_argument('--split', required=True, choices=['train', 'test'])
    add.add_argument('--bin-label', dest='bin_labels', action='append', metavar='BIN=empty|loaded',
                     help='Repeat for every configured bin')
    add.add_argument('--bin-a', choices=['empty', 'loaded'], help='Compatibility alias for --bin-label bin_a=LABEL')
    add.add_argument('--bin-b', choices=['empty', 'loaded'], help='Compatibility alias for --bin-label bin_b=LABEL')
    add.add_argument('--exposure-ms', type=float, required=True, help='Recorded structured-light exposure in ms')
    add.add_argument('--spotlight-exposure-ms', type=float, help='Optional recorded spotlight exposure; otherwise read metadata')
    add.add_argument('--group-id', help='Layout/session ID; every exposure/repeat in a group must share one split')
    add.add_argument('--note', default='')
    add.set_defaults(function=command_add)
    inspect = commands.add_parser('inspect', help='Export raw robot cloud, print bounds, optionally view/pick/CAD overlay')
    filt = commands.add_parser('filter', help='Replay geometric filtering for one capture and one bin')
    calibrate = commands.add_parser('calibrate', help='Fit models from all-empty training captures')
    validate = commands.add_parser('validate', help='Export geometry diagnostics on a chosen split')
    for command in (inspect, filt, calibrate, validate):
        command.add_argument('--dataset', required=True)
        command.add_argument('--output', required=True, help='New output directory; never overwrite results')
        command.add_argument('--stride', type=int, default=1, help='Depth pixel stride, default 1 (full resolution)')
    for command in (inspect, filt):
        command.add_argument('--capture', required=True)
    for command in (filt, validate):
        command.add_argument('--model', required=True)
    inspect.add_argument('--show', action='store_true')
    inspect.add_argument('--pick', action='store_true', help='Shift-click points to print robot XYZ after closing viewer')
    inspect.add_argument('--cad')
    inspect.add_argument('--cad-transform', help='YAML containing robot_from_cad: a 4x4 rigid matrix')
    inspect.add_argument('--cad-scale-to-mm', type=float, default=1.0)
    inspect.set_defaults(function=command_inspect)
    filt.add_argument('--bin', required=True)
    filt.set_defaults(function=command_filter)
    calibrate.add_argument('--config', help='Default: DATASET/bin_config.yaml')
    calibrate.add_argument('--model-name', help='Readable identity for a new rim model')
    calibrate.add_argument('--cad-reference', help='Optional commissioned CAD plan YAML; collection data remains unchanged')
    calibrate.add_argument('--output-format', choices=['bundle', 'yaml'], default='bundle',
                           help='Default: checked binary arrays + model.json; yaml is legacy interchange')
    selection = calibrate.add_mutually_exclusive_group()
    selection.add_argument('--exposure-ms', type=float, help='Fit a legacy single-exposure model from this group only')
    selection.add_argument('--exposure-profiles', action='store_true', help='Fit independent height fields by exposure with bounded interpolation')
    selection.add_argument('--pool-exposures', action='store_true',
                           help='Fit one shared height field from all recorded exposures; no runtime exposure selection')
    calibrate.add_argument('--interpolation-margin-mm', type=float, default=0.5,
                           help='Extra tolerance at intermediate exposures (default 0.5 mm); validate on held-out intermediate captures')
    calibrate.add_argument('--max-fit-points', type=int, default=200000,
                           help='Planar fitter sample limit; height fields use all points within their fit bounds')
    calibrate.set_defaults(function=command_calibrate)
    validate.add_argument('--split', choices=['train', 'test'], default='test')
    validate.set_defaults(function=command_validate)
    convert = commands.add_parser('convert-model', help='Convert legacy YAML or a checked model bundle without refitting')
    convert.add_argument('--model', required=True)
    convert.add_argument('--output', required=True, help='New directory; never overwrite')
    convert.add_argument('--output-format', choices=['bundle', 'yaml'], default='bundle')
    convert.set_defaults(function=command_convert_model)
    session = commands.add_parser('import-session', help='Register a completed bin-collection session using frozen labels and calibration')
    session.add_argument('--session', required=True, help='Session directory or session.json')
    session.add_argument('--dataset', required=True, help='New dataset directory; never overwrite')
    session.set_defaults(function=command_import_session)
    build = commands.add_parser('build-session', help='Import, fit, and replay a completed session into one candidate package')
    build.add_argument('--session', required=True)
    build.add_argument('--output', required=True, help='New result directory; published only after all stages succeed')
    build.add_argument('--stride', type=int, default=1)
    build.add_argument('--model-name')
    build.add_argument('--cad-reference', help='Optional commissioned CAD plan YAML for this candidate')
    build.add_argument('--pipeline-config', help='Production Summit configuration; required for rim-model acceptance')
    build.add_argument('--replay-executable', help='Path to the production summit_replay executable')
    build.set_defaults(function=command_build_session)
    production = commands.add_parser('validate-production', help='Replay real held-out depth through the production Summit pipeline')
    production.add_argument('--dataset', required=True)
    production.add_argument('--model', required=True)
    production.add_argument('--pipeline-config', required=True)
    production.add_argument('--output', required=True)
    production.add_argument('--replay-executable')
    production.add_argument('--quick', action='store_true', help='Check existing model using fresh held-out empty captures without refitting')
    production.set_defaults(function=command_production_validate)
    activate = commands.add_parser('activate', help='Install a passing physical candidate and verify live Perception reload')
    rollback = commands.add_parser('rollback', help='Restore the previous installed model and verify live Perception reload')
    activate.add_argument('--candidate', required=True)
    for command in (activate, rollback):
        command.add_argument('--models-root', required=True)
        command.add_argument('--deployment-root', required=True, help='History directory outside artifacts, e.g. data/bin_model/deployment')
        command.add_argument('--pipeline-config', required=True)
    activate.set_defaults(function=command_activate)
    rollback.set_defaults(function=command_rollback)
    demo = commands.add_parser('demo', help='Generate synthetic data, calibrate, and replay; no cameras or ROS')
    demo.add_argument('--output', required=True)
    demo.set_defaults(function=command_demo)
    return cli


def main():
    args = parser().parse_args()
    try:
        if getattr(args, 'stride', 1) < 1:
            raise data.DatasetError('--stride must be positive')
        args.function(args)
    except (ValueError, OSError, KeyError, TypeError, ImportError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
