"""Apply optional CAD commissioning plans without changing collected sessions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import shutil

import numpy as np

import dataset as data


CONFIG_KEYS = {'mesh_path', 'scale_to_mm', 'bin_from_cad', 'alignment',
               'min_overlap_fraction', 'association_margin_mm',
               'min_measured_retained_fraction', 'max_reference_residual_mm'}
ALIGNMENT_KEYS = {'enabled', 'max_translation_mm', 'max_rotation_deg',
                  'max_correspondence_distance_mm', 'max_iterations', 'min_overlap_fraction'}


def _number(value, name, *, minimum=0, inclusive=False, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise data.DatasetError(f'CAD {name} must be numeric')
    value = float(value)
    if (not np.isfinite(value) or (value < minimum if inclusive else value <= minimum)
            or (maximum is not None and value > maximum)):
        raise data.DatasetError(f'CAD {name} is outside its supported range')
    return value


def resolve_config(config, base_path):
    """Validate one bin's inputs, resolving mesh paths against its document."""
    if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
        raise data.DatasetError('CAD bin configuration contains unsupported parameters')
    if any(key not in config for key in ('mesh_path', 'scale_to_mm', 'bin_from_cad')):
        raise data.DatasetError('Each CAD bin needs mesh_path, scale_to_mm, and bin_from_cad')
    result = deepcopy(config)
    if not isinstance(result['mesh_path'], str) or not result['mesh_path'].strip():
        raise data.DatasetError('CAD mesh_path must name an STL file')
    path = Path(result['mesh_path']).expanduser()
    if not path.is_absolute():
        path = Path(base_path) / path
    path = path.resolve()
    if path.suffix.lower() != '.stl' or not path.is_file():
        raise data.DatasetError(f'CAD mesh_path must name an existing STL file: {path}')
    result['mesh_path'] = str(path)
    result['scale_to_mm'] = _number(result['scale_to_mm'], 'scale_to_mm')
    result['bin_from_cad'] = data.validate_transform(result['bin_from_cad'], 'CAD bin_from_cad').tolist()
    for key in ('min_overlap_fraction', 'min_measured_retained_fraction'):
        if key in result:
            result[key] = _number(result[key], key, maximum=1)
    if 'association_margin_mm' in result:
        result['association_margin_mm'] = _number(result['association_margin_mm'], 'association_margin_mm', inclusive=True)
    if 'max_reference_residual_mm' in result:
        result['max_reference_residual_mm'] = _number(result['max_reference_residual_mm'], 'max_reference_residual_mm')
    if 'alignment' in result:
        alignment = result['alignment']
        if (not isinstance(alignment, dict) or set(alignment) - ALIGNMENT_KEYS
                or type(alignment.get('enabled', False)) is not bool):
            raise data.DatasetError('CAD alignment requires a mapping with boolean enabled and supported parameters')
        if not alignment.get('enabled', False) and set(alignment) - {'enabled'}:
            raise data.DatasetError('Disabled CAD alignment accepts only enabled: false')
        for key in set(alignment) - {'enabled'}:
            alignment[key] = _number(alignment[key], f'alignment.{key}',
                                     maximum=1 if key == 'min_overlap_fraction' else None)
        if alignment.get('max_rotation_deg', 5) >= 45:
            raise data.DatasetError('CAD alignment max_rotation_deg must be below 45')
        if 'max_iterations' in alignment:
            iterations = alignment['max_iterations']
            if iterations != int(iterations) or iterations > 1000:
                raise data.DatasetError('CAD alignment max_iterations must be an integer no greater than 1000')
            alignment['max_iterations'] = int(iterations)
    return result


def load(path, *, scene_id, frame_id, bin_ids):
    """Read a scene plan and select every requested bin without changing the plan."""
    path = Path(path).resolve()
    document = data.read_yaml(path)
    if not isinstance(document, dict) or set(document) != {'name', 'scene_id', 'frame_id', 'bins'}:
        raise data.DatasetError('CAD plan requires name, scene_id, frame_id, and bins')
    name = document['name']
    if not isinstance(name, str) or not name.strip():
        raise data.DatasetError('CAD plan name must be a nonempty label')
    if type(document['scene_id']) is not int or document['scene_id'] != scene_id:
        raise data.DatasetError('CAD plan scene_id differs from the requested dataset')
    if not isinstance(document['frame_id'], str) or document['frame_id'] != frame_id:
        raise data.DatasetError('CAD plan frame_id differs from the requested dataset')
    actual = data.validate_bin_ids(document['bins'])
    selected = set(bin_ids)
    if not selected or not selected.issubset(actual):
        raise data.DatasetError('CAD plan must include every configured bin ID')
    resolved = {bin_id: resolve_config(config, path.parent) for bin_id, config in document['bins'].items()}
    return dict(document, name=name.strip(), bins={bin_id: config for bin_id, config in resolved.items()
                                                  if bin_id in selected})


def apply(settings, plan):
    """Return candidate-only fitting settings; never mutate frozen collection data."""
    if data.validate_bin_ids(settings.get('bins')) != set(plan['bins']):
        raise data.DatasetError('CAD plan differs from the configured bin IDs')
    result = deepcopy(settings)
    for name, config in plan['bins'].items():
        result['bins'][name]['cad_reference'] = deepcopy(config)
    return result


def _asset_name(path):
    stem = re.sub(r'[^A-Za-z0-9_.-]+', '-', path.stem).strip('.-') or 'bin'
    return stem[:120] + '.stl'


def _packaged_source(root, relative, *, cad_asset=False):
    """Resolve portable model references without permitting path/symlink escape."""
    path = Path(relative)
    if (path.is_absolute() or '..' in path.parts or not path.parts
            or (cad_asset and path.parts[0] != 'cad')):
        raise data.DatasetError('Packaged CAD paths must remain relative to the model directory')
    root = Path(root).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise data.DatasetError('Packaged CAD path resolves outside the model directory')
    return resolved


def package(output, model, settings, plan=None, *, source_root=None):
    """Copy CAD inputs into a portable candidate and return its settings snapshot.

    The freshly fitted model's CAD mesh paths are updated in place. Large numeric
    arrays are not duplicated. Fitting settings are copied before modification.
    """
    output = Path(output)
    snapshot = deepcopy(settings)
    model_bins = model.get('bins', {})
    data.validate_bin_ids(model_bins)
    selected = {name: item['cad_reference'] for name, item in model_bins.items() if 'cad_reference' in item}
    for name, reference in selected.items():
        path = Path(reference['mesh_path'])
        if source_root is not None:
            path = _packaged_source(source_root, path, cad_asset=True)
        elif not path.is_absolute():
            raise data.DatasetError('CAD packaging requires a resolved mesh path or source model directory')
        if not path.is_file():
            raise data.DatasetError(f'CAD mesh is unavailable for packaging: {path}')
        relative = Path('cad') / name / _asset_name(path)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise data.DatasetError(f'Refusing to overwrite packaged CAD: {destination}')
        shutil.copyfile(path, destination)
        reference['mesh_path'] = relative.as_posix()
        config = snapshot.get('bins', {}).get(name, {}).get('cad_reference')
        if config is not None:
            config['mesh_path'] = relative.as_posix()
            # Freeze the fitted placement for a portable reproducible snapshot;
            # alignment diagnostics in the model retain how it was established.
            config['bin_from_cad'] = np.asarray(reference['bin_from_cad']).tolist()
            config['alignment'] = {'enabled': False}
    if plan is not None:
        if set(selected) != set(plan['bins']):
            raise data.DatasetError('Fitted model lacks one or more requested CAD references')
        portable_plan = {key: deepcopy(plan[key]) for key in ('name', 'scene_id', 'frame_id')}
        portable_plan['bins'] = {name: deepcopy(snapshot['bins'][name]['cad_reference']) for name in selected}
        data.write_yaml(output / 'cad_reference.yaml', portable_plan)
        model['cad_reference_plan'] = {'name': plan['name'], 'path': 'cad_reference.yaml'}
    elif source_root is not None and model.get('cad_reference_plan'):
        source_plan = _packaged_source(source_root, model['cad_reference_plan']['path'])
        if source_plan.is_file():
            # Converted bundles retain the same relative CAD asset layout.
            shutil.copyfile(source_plan, output / 'cad_reference.yaml')
            model['cad_reference_plan']['path'] = 'cad_reference.yaml'
    return snapshot


def build_metadata(model):
    """Return optional build metadata with paths relative to the candidate root."""
    selected = {name: item['cad_reference']['mesh_path'] for name, item in model.get('bins', {}).items()
                if 'cad_reference' in item}
    if not selected:
        return None
    plan = model.get('cad_reference_plan', {})
    return {'name': plan.get('name'), 'plan': f"model/{plan['path']}" if plan.get('path') else None,
            'mesh_paths': {name: f'model/{path}' for name, path in selected.items()},
            'role': 'nominal_geometric_association_with_measured_surface_and_support'}
