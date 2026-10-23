"""Portable bin-model storage: small JSON metadata plus checked raw numeric arrays.

Binary payloads are C-order little-endian 64-bit arrays. NaN means unknown only
for nullable height, tolerance, bias and uncertainty fields. YAML remains an
explicit interchange/legacy option.
The storage layout does not change the geometry or exposure behavior.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np

from dataset import DatasetError, read_yaml, write_yaml, validate_bin_ids

ARRAY_FIELDS = {
    'x_centers_mm': ('float64_le', 1, False),
    'y_centers_mm': ('float64_le', 1, False),
    'surface_height_mm': ('float64_le', 2, True),
    'tolerance_mm': ('float64_le', 2, True),
    'capture_support': ('int64_le', 2, False),
    'residual_capture_support': ('int64_le', 2, False),
}
PROFILE_FIELDS = {'bias_mm': ('float64_le', 2, True),
                  'uncertainty_mm': ('float64_le', 2, True),
                  'validity': ('int64_le', 2, False),
                  'capture_support': ('int64_le', 2, False)}
REGISTRATION_FIELDS = {name: ('float64_le', 2, False)
                       for name in ('surface_xyz_mm', 'boundary_xyz_mm')}
DTYPES = {'float64_le': np.dtype('<f8'), 'int64_le': np.dtype('<i8')}
MAX_ARRAY_BYTES = 512 * 1024 * 1024


def _fields(model):
    if model.get('model_type') == 'calibrated_bin_background':
        bins = model.get('bins')
        if not isinstance(bins, dict):
            raise DatasetError('Model bins must contain mappings')
        for name, field in bins.items():
            if not isinstance(field, dict):
                raise DatasetError('Model bins must contain mappings')
            yield f'bins/{name}/geometry', field.get('geometry'), ARRAY_FIELDS
            yield f'bins/{name}/registration', field.get('registration'), REGISTRATION_FIELDS
            profiles = field.get('camera_profiles')
            if not isinstance(profiles, list):
                raise DatasetError('Camera profiles must be a list')
            for index, profile in enumerate(profiles):
                yield f'bins/{name}/camera_profiles/{index}', profile, PROFILE_FIELDS
        return
    if model.get('model_type') == 'offline_height_field_exposure_profiles':
        profiles = model.get('profiles')
        if not isinstance(profiles, list):
            raise DatasetError('Exposure profiles must be a list')
        containers = [(f'profiles/{index}/bins', profile.get('bins'))
                      for index, profile in enumerate(profiles) if isinstance(profile, dict)]
        if len(containers) != len(profiles):
            raise DatasetError('Exposure profiles must contain mappings')
    else:
        containers = [('bins', model.get('bins'))]
    for prefix, bins in containers:
        if not isinstance(bins, dict) or any(not isinstance(field, dict) for field in bins.values()):
            raise DatasetError('Model bins must contain mappings')
        for name, field in bins.items():
            if field.get('kind') == 'height_field':
                yield f'{prefix}/{name}', field, ARRAY_FIELDS


def _validate(model):
    import exposure_profiles
    if not isinstance(model, dict):
        raise DatasetError('Model must be a mapping')
    if model.get('model_type') == 'calibrated_bin_background':
        from modern_model import validate_model
        validate_model(model)
        return
    if (not isinstance(model.get('frame_id'), str) or not model['frame_id'].strip()
            or model.get('length_unit') != 'mm'):
        raise DatasetError('Unsupported model frame/units')
    kind = model.get('model_type')
    if kind == exposure_profiles.MODEL_TYPE or 'exposure_settings_ms' in model:
        exposure_profiles.validate_bundle(model)
    elif kind in ('offline_height_field_bins', 'offline_convex_planar_bins'):
        validate_bin_ids(model.get('bins'))
        if kind == 'offline_height_field_bins':
            import height_field as backend
        else:
            import geometry as backend
        backend.validate_disjoint_bins(model['bins'])
    else:
        raise DatasetError(f'Unsupported model type: {kind}')


def _array(value, name, specification=None):
    dtype, rank, nullable = specification or ARRAY_FIELDS[name]
    try:
        numeric = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise DatasetError(f'{name}: expected a rectangular numeric array') from error
    if numeric.ndim != rank or any(size < 1 for size in numeric.shape):
        raise DatasetError(f'{name}: invalid array shape')
    if numeric.nbytes > MAX_ARRAY_BYTES:
        raise DatasetError(f'{name}: array exceeds supported size')
    if np.isinf(numeric).any() or (not nullable and not np.isfinite(numeric).all()):
        raise DatasetError(f'{name}: invalid nonfinite array values')
    if dtype == 'int64_le':
        if np.any(numeric < 0) or np.any(numeric >= 2**63) or np.any(numeric != np.floor(numeric)):
            raise DatasetError(f'{name}: support must contain nonnegative int64 counts')
        # Convert from the original values, preserving integers above 2**53.
        return np.asarray(value, dtype=DTYPES[dtype])
    return np.asarray(value, dtype=DTYPES[dtype])


def _lists(array, nullable):
    if nullable:
        values = array.astype(object)
        values[np.isnan(array)] = None
        return values.tolist()
    return array.tolist()


def _payload_path(root, value):
    if not isinstance(value, str):
        raise DatasetError('Array descriptor file must be a relative path')
    relative = Path(value)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] != 'arrays':
        raise DatasetError('Array payload must stay inside the bundle arrays directory')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise DatasetError('Array payload resolves outside the model bundle')
    return path


def _without_versions(value):
    # Ignore old version tags when converting archived YAML files.
    if isinstance(value, dict):
        return {key: _without_versions(item) for key, item in value.items()
                if key not in ('schema_version', 'storage_format')}
    if isinstance(value, list):
        return [_without_versions(item) for item in value]
    return value


def load_model(path):
    """Read a bundle directory/model.json or a legacy YAML model, validating it."""
    path = Path(path)
    if path.is_dir():
        path /= 'model.json'
    if path.suffix.lower() in ('.yaml', '.yml'):
        model = read_yaml(path)
    else:
        def invalid_constant(value):
            raise DatasetError(f'Nonstandard JSON number: {value}')
        with path.open() as stream:
            model = json.load(stream, parse_constant=invalid_constant)
        if not isinstance(model, dict):
            raise DatasetError('Model metadata must be a mapping')
        for _, field, fields in _fields(model):
            if not isinstance(field, dict):
                raise DatasetError('Array container must be a mapping')
            for name, specification in fields.items():
                dtype, rank, nullable = specification
                descriptor = field.get(name)
                if not isinstance(descriptor, dict) or set(descriptor) != {'file', 'dtype', 'shape'}:
                    raise DatasetError(f'{name}: expected a binary array descriptor')
                shape = descriptor['shape']
                if (descriptor['dtype'] != dtype or not isinstance(shape, list) or len(shape) != rank
                        or any(type(size) is not int or size < 1 for size in shape)):
                    raise DatasetError(f'{name}: invalid array dtype/shape')
                size = math.prod(shape) * 8
                if size > MAX_ARRAY_BYTES:
                    raise DatasetError(f'{name}: array exceeds supported size')
                payload = _payload_path(path.parent, descriptor['file'])
                if payload.stat().st_size != size:
                    raise DatasetError(f'{name}: array byte size does not match shape')
                array = np.fromfile(payload, dtype=DTYPES[dtype]).reshape(shape)
                _array(array, name, specification)
                field[name] = _lists(array, nullable)
    model = _without_versions(model)
    _validate(model)
    return model


def write_model(output, model, output_format='bundle'):
    """Publish model files in an existing output directory; never replace files."""
    if output_format not in ('bundle', 'yaml'):
        raise DatasetError('Output format must be bundle or yaml')
    _validate(model)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() or (output / name).is_symlink()
           for name in ('model.json', 'model.yaml', 'arrays')):
        raise DatasetError(f'Refusing to overwrite model files in {output}')
    if output_format == 'yaml':
        target = output / 'model.yaml'
        write_yaml(target, _without_versions(model))
        return target
    document = _without_versions(model)
    stage = Path(tempfile.mkdtemp(prefix='.model-', dir=output))
    try:
        for prefix, field, fields in _fields(document):
            for name, specification in fields.items():
                dtype, _, _ = specification
                array = _array(field[name], name, specification)
                relative = f'arrays/{prefix}/{name}.bin'
                payload = stage / relative
                payload.parent.mkdir(parents=True, exist_ok=True)
                array.tofile(payload)
                field[name] = {'file': relative, 'dtype': dtype, 'shape': list(array.shape)}
        manifest = stage / 'model.json'
        manifest.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
        if (stage / 'arrays').exists():
            (stage / 'arrays').rename(output / 'arrays')
        # Manifest is the completion marker, published after all binary payloads.
        manifest.rename(output / 'model.json')
    finally:
        shutil.rmtree(stage)
    return output / 'model.json'
