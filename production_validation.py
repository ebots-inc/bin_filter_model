"""Replay held-out captures through the production Summit executable.

Acceptance is based on measured empty scenes and operator-confirmed resistor
placements. No part thickness, filled-bin count or hardware readback is invented.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

import dataset as data
import model_io


def replay_program(value=None):
    if value:
        path = Path(value)
        if path.is_file():
            return str(path.resolve())
        found = shutil.which(str(value))
        if found:
            return found
    else:
        found = shutil.which('summit_replay')
        if found:
            return found
        for prefix in ('/root/perception/workspace/install', '/root/perception/install', '/root/install'):
            path = Path(prefix) / 'perception_node/lib/perception_node/summit_replay'
            if path.is_file():
                return str(path)
    raise data.DatasetError('Production replay is unavailable. Build perception_node/summit_replay and set --replay-executable.')


def _stages(manifest, dataset_root):
    stages = manifest.get('stages', [])
    if not stages and (Path(dataset_root) / 'session.json').is_file():
        stages = json.loads((Path(dataset_root) / 'session.json').read_text()).get('stages', [])
    if not isinstance(stages, list) or any(not isinstance(stage, dict) or not stage.get('id') for stage in stages):
        raise data.DatasetError('Validation stages require named mappings')
    result = {stage['id']: stage for stage in stages}
    if len(result) != len(stages):
        raise data.DatasetError('Validation stage IDs must be unique')
    return result


def target_bounds(stage_bin, bin_config):
    region = stage_bin.get('target_region_fraction')
    if region is None:
        return None
    if len(region) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in region):
        raise data.DatasetError('Target region must contain four finite fractions within [0,1]')
    if not (region[0] < region[2] and region[1] < region[3]):
        raise data.DatasetError('Target region must have positive area')
    geometry = bin_config.get('geometry', bin_config)
    if 'tolerance_mm' in geometry:
        reliable = np.isfinite(np.asarray(geometry['tolerance_mm'], dtype=float))
        yy, xx = np.nonzero(reliable)
        if not len(xx):
            raise data.DatasetError('Cannot define a resistor placement without a measured reliable interior')
        x, y = np.asarray(geometry['x_centers_mm']), np.asarray(geometry['y_centers_mm'])
        lo, hi = [x[xx.min()],y[yy.min()]], [x[xx.max()+1],y[yy.max()+1]]
    else:
        lo, hi = geometry['xy_bounds_mm']
    bounds = [[lo[i] + (hi[i] - lo[i]) * region[i] for i in range(2)],
              [lo[i] + (hi[i] - lo[i]) * region[i + 2] for i in range(2)]]
    if 'geometry' in bin_config:
        # Wide lip/table fitting bounds are not valid resistor placement bounds.
        # Compare in local bin axes, including tilted/rotated site placements.
        placement = data.validate_transform(bin_config['robot_from_bin'])
        if not any(x[column+1] >= bounds[0][0] and x[column] <= bounds[1][0]
                   and y[row+1] >= bounds[0][1] and y[row] <= bounds[1][1] for row,column in zip(yy,xx)):
            raise data.DatasetError('The prescribed resistor region has no measured reliable interior support')
        return {'frame_id':'bin', 'xy_bounds_mm':bounds, 'robot_from_bin':placement.tolist(),
                'reference':'measured_reliable_interior'}
    return bounds


def assess_result(result, label, region=None):
    reasons = []
    registration = result.get('registration', result.get('bin_registration', result.get('icp', {})))
    if (not isinstance(registration,dict) or registration.get('attempted') is not True
            or registration.get('accepted') is not True):
        reasons.append('Bin registration was not accepted')
    candidates = result.get('candidates_robot_mm', result.get('summit_candidates_robot_mm', []))
    if not isinstance(candidates, list) or any(not isinstance(p,(list,tuple)) or len(p) != 3
            or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in p) for p in candidates):
        return ['Replay returned malformed candidate positions']
    if type(result.get('success')) is not bool:
        reasons.append('Replay did not return a boolean success state')
    if label == 'empty':
        if result.get('success') or candidates:
            reasons.append('An empty bin produced a pick target')
        if not (result.get('empty_scene') is True or result.get('outcome') == 'empty'):
            reasons.append('The pipeline did not finish with an expected empty-scene result')
    elif label == 'loaded':
        if result.get('success') is not True or not candidates:
            reasons.append('No pick target was detected in the loaded bin')
        elif region:
            if isinstance(region, dict):
                points = data.transform_points(np.asarray(candidates), np.linalg.inv(data.validate_transform(region['robot_from_bin'])))
                bounds = region['xy_bounds_mm']
            else:
                points, bounds = candidates, region
            if not any(all(bounds[0][i] <= point[i] <= bounds[1][i] for i in range(2)) for point in points):
                reasons.append('No target was detected inside the prescribed resistor placement region')
    else:
        reasons.append('Unknown expected bin contents')
    return reasons


def validate_production(dataset, model_path, pipeline_config, output, executable=None, *, quick=False):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    manifest, _ = data.load_dataset(dataset)
    model = model_io.load_model(model_path)
    expected_bins = data.validate_bin_ids(model.get('bins'))
    for key in ('frame_id', 'scene_id'):
        if key in model and model[key] != manifest.get(key):
            raise data.DatasetError(f'Production model {key} differs from dataset')
    if model.get('model_type') == 'calibrated_bin_background':
        for name, item in model['bins'].items():
            for profile in item['camera_profiles']:
                if not {'gain', 'black_level'} <= set(profile.get('acquisition', {})):
                    raise data.DatasetError(f'{name}: model has unknown camera gain/black level and cannot establish factory acceptance')
    program = replay_program(executable)
    config = Path(pipeline_config).resolve()
    if not config.is_file():
        raise data.DatasetError(f'Missing production pipeline configuration: {config}')
    entries = [entry for entry in manifest['captures'] if entry['split'] == 'test']
    if not entries:
        raise data.DatasetError('Production validation requires held-out test captures')
    training_ids = {item['id'] for item in model.get('training_captures', [])}
    training_groups = {item.get('group_id') for item in model.get('training_captures', []) if item.get('group_id')}
    if any(entry['id'] in training_ids or entry.get('group_id') in training_groups for entry in entries):
        raise data.DatasetError('Production validation cannot reuse training captures or arrangements')
    stages = _stages(manifest, dataset)
    capture_stages = {}
    for stage in stages.values():
        if data.validate_bin_ids(stage.get('bins')) != expected_bins:
            raise data.DatasetError(f"{stage['id']}: stage must describe every model bin")
        data.validate_bin_labels({name: item.get('label') for name, item in stage['bins'].items()}, expected_bins)
        for capture_id in stage.get('capture_ids', []):
            if capture_id in capture_stages:
                raise data.DatasetError(f'{capture_id}: validation capture belongs to multiple stages')
            capture_stages[capture_id] = stage
    for entry in entries:
        data.validate_bin_labels(entry.get('bins'), expected_bins)
        stage = stages.get(entry.get('stage_id'), capture_stages.get(entry['id']))
        if stages and (stage is None or capture_stages.get(entry['id']) is not stage):
            raise data.DatasetError(f"{entry['id']}: validation capture must belong to its declared stage")
        if stage and (entry['bins'] != {name: item['label'] for name, item in stage['bins'].items()}
                      or stage.get('split') != entry['split']):
            raise data.DatasetError(f"{entry['id']}: validation labels/split differ from its stage")
    output.mkdir(parents=True, exist_ok=False)
    # Copy the exact configuration used for replay into the reviewable report.
    shutil.copyfile(config, output / 'pipeline_config.yaml')
    results = []
    stage_exposures = {}
    for entry in entries:
        if len(entry.get('depth_images', [])) != 2:
            raise data.DatasetError(f"{entry['id']}: production replay requires the two original depth images")
        stage = stages.get(entry.get('stage_id'), capture_stages.get(entry['id'], {}))
        stage_id = stage.get('id', entry.get('stage_id', 'unlabelled'))
        exposure = data.capture_exposure(entry)
        stage_exposures.setdefault(stage_id, set()).add(exposure)
        depths = [data.checked_file(dataset, item) for item in entry['depth_images']]
        for bin_id, label in entry['bins'].items():
            destination = output / entry['id'] / bin_id
            destination.mkdir(parents=True)
            report_path = destination / 'result.json'
            launcher = [sys.executable, program] if Path(program).suffix == '.py' else [program]
            command = [*launcher, '--model', str(Path(model_path).resolve()),
                       '--camera-calibration', str(data.resolve(dataset, manifest['calibration']['camera']['path'])),
                       '--robot-calibration', str(data.resolve(dataset, manifest['calibration']['robot']['path'])),
                       '--config', str(config), '--depth0', str(depths[0]), '--depth1', str(depths[1]),
                       '--exposure-ms', str(exposure), '--scene-id', str(manifest['scene_id']),
                       '--bin', bin_id, '--output', str(report_path)]
            acquisition = entry.get('source_metadata', {}).get('requested_settings', {})
            for field, option in (('gain', '--gain'), ('black_level', '--black-level')):
                values = acquisition.get(field)
                if not isinstance(values, list) or len(values) != 2:
                    raise data.DatasetError(f"{entry['id']}: production validation requires recorded requested {field} for both cameras")
                command += [option, ','.join(map(str, values))]
            print(f"Production replay: {entry['id']} / {bin_id} ({label}, {exposure:g} ms)", flush=True)
            execution_error = None
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
                log = completed.stdout + completed.stderr
                if completed.returncode:
                    execution_error = f'Production replay exited with status {completed.returncode}'
                elif not report_path.is_file():
                    execution_error = 'Production replay did not produce its result'
            except subprocess.TimeoutExpired as error:
                def output_text(value):
                    return value.decode(errors='replace') if isinstance(value,bytes) else (value or '')
                log = output_text(error.stdout) + output_text(error.stderr)
                execution_error = 'Production replay exceeded the 180-second limit'
            except OSError as error:
                log = str(error)
                execution_error = f'Production replay could not start: {error}'
            (destination / 'replay.log').write_text(log)
            if execution_error is None:
                try:
                    def invalid_constant(value):
                        raise ValueError(f'Nonfinite JSON value: {value}')
                    result = json.loads(report_path.read_text(),parse_constant=invalid_constant)
                    if not isinstance(result,dict):
                        raise ValueError('Expected a result mapping')
                except (ValueError,OSError) as error:
                    execution_error = f'Production replay returned an unreadable result: {error}'
            if execution_error:
                result = {'success':False, 'outcome':'execution_error', 'message':execution_error,
                          'registration':{'attempted':False, 'accepted':False}, 'candidates_robot_mm':[]}
            # A build is fitted and replayed under an atomic staging directory.
            # Relative package references survive publication of that directory.
            result['model_path'] = os.path.relpath(Path(model_path).resolve(), report_path.parent)
            result['model_path_relative_to'] = 'result_file'
            data.write_json(report_path, result)
            region = target_bounds(stage.get('bins', {}).get(bin_id, {}), model['bins'][bin_id])
            reasons = assess_result(result, label, region)
            if execution_error:
                reasons.insert(0,execution_error + '; inspect replay.log')
            results.append({'capture_id': entry['id'], 'stage_id': stage_id, 'bin_id': bin_id,
                            'exposure_ms': exposure, 'expected': label, 'target_region': region,
                            'passed': not reasons, 'reasons': reasons,
                            'result': str(report_path.relative_to(output))})
    missing = []
    exposures = set(model.get('supported_exposure_ms', model.get('exposure_settings_ms', [])))
    if not exposures and model.get('bins'):
        for item in model['bins'].values():
            exposures.update(profile['exposure_ms'] for profile in item.get('camera_profiles', []))
            if not item.get('camera_profiles'):
                exposures.update(item.get('supported_exposure_ms', []))
    if not exposures:
        exposures = {data.capture_exposure(entry) for entry in entries}
    # Require every planned wall side independently. Saved left-only plans keep
    # their original acceptance requirements and report labels.
    wall_sides = {bin_id: set() for bin_id in expected_bins}
    for stage in stages.values():
        if stage.get('split') != 'test':
            continue
        for bin_id, item in stage['bins'].items():
            if item.get('placement') == 'wall':
                side = item.get('wall_side', 'left')
                if side not in ('left', 'right', 'bottom', 'top'):
                    raise data.DatasetError(f"{stage['id']}: unknown resistor wall side {side!r}")
                wall_sides[bin_id].add(side)
    wall_keys = {bin_id: {side: f'{bin_id}:wall' if sides == {'left'} else f'{bin_id}:wall:{side}'
                          for side in sorted(sides)} for bin_id, sides in wall_sides.items()}
    # Require an empty validation scene and each bin's resistor placements at
    # every supported setting; normal/full are qualitative operator labels.
    required = {'empty': set(), 'normal': set(), 'full': set()}
    if not quick:
        required.update({f'{bin_id}:{placement}': set() for bin_id in expected_bins
                         for placement in ('center', 'corner')})
        for bin_id, keys in wall_keys.items():
            required.update({key: set() for key in keys.values()} or {f'{bin_id}:wall': set()})
    for stage_id, stage in stages.items():
        values = stage_exposures.get(stage_id, set())
        bins = stage.get('bins', {})
        if set(bins) == expected_bins and all(item.get('label') == 'empty' for item in bins.values()):
            required['empty'].update(values)
        for kind in ('normal', 'full'):
            if (stage.get('fill') == kind or stage_id.endswith('_' + kind)) and all(
                    item.get('label') == 'loaded' for item in bins.values()):
                required[kind].update(values)
        for bin_id, item in bins.items():
            key = f"{bin_id}:{item.get('placement')}"
            if item.get('placement') == 'wall':
                key = wall_keys[bin_id].get(item.get('wall_side', 'left'), key)
            if (type(item.get('part_count')) is int and item['part_count'] == 1 and key in required
                    and item.get('label') == 'loaded' and item.get('target_region_fraction') is not None):
                required[key].update(values)
    if quick:
        required = {'empty': required['empty']}
        if not stages and all(all(label == 'empty' for label in entry['bins'].values()) for entry in entries):
            required['empty'] = {data.capture_exposure(entry) for entry in entries}
    for kind, seen in required.items():
        for exposure in sorted(exposures - seen):
            missing.append(f'{kind} at {exposure:g} ms')
    passed = bool(results) and all(item['passed'] for item in results) and not missing
    report = {'status': 'passed' if passed else 'failed', 'purpose': 'quick_check' if quick else 'factory_acceptance',
              'model_name': model.get('model_name', Path(model_path).parent.name),
              'model_type': model.get('model_type'), 'synthetic': manifest.get('synthetic', False),
              'pipeline': 'production_summit', 'capture_count': len(entries), 'results': results,
              'missing_evidence': missing, 'exposures_ms': sorted(exposures),
              'minimum_part': 'operator-placed resistor', 'filled_amount': 'operator-described normal/full',
              'exposure_evidence': 'requested_not_sensor_verified'}
    data.write_json(output / 'acceptance.json', report)
    return report
