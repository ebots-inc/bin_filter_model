"""Install reviewed bin models and verify the model loaded by Perception.

The caller supplies a ROS reload/status function. Filesystem changes are rolled
back if the running service cannot load the candidate. No hardware is operated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from contextlib import contextmanager
import fcntl

import dataset as data
import model_io


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False, encoding='utf-8') as stream:
        temporary = Path(stream.name)
        stream.write(text)
        os.fchmod(stream.fileno(), 0o644)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _config_text(text, model_path, modern=True):
    import yaml
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict) or 'bin_filter_model_path' not in parsed:
        raise data.DatasetError('Deployment configuration is missing bin_filter_model_path')
    value = json.dumps(str(model_path))
    changed, count = re.subn(r'^bin_filter_model_path:.*$', 'bin_filter_model_path: ' + value, text, flags=re.M)
    if count != 1:
        raise data.DatasetError('Deployment configuration must have one top-level model path')
    if modern:
        icp = parsed.get('bin_icp', {})
        if not icp.get('enabled'):
            raise data.DatasetError('Rim-model deployment requires bin_icp.enabled: true')
        changed, count = re.subn(r'^(\s+)reference_mode:.*$', r'\1reference_mode: measured_rim', changed, flags=re.M)
        if count != 1:
            raise data.DatasetError('Deployment configuration must contain one bin_icp.reference_mode')
    return changed


def verify_identity(reply, expected_path, expected_name=None):
    if not isinstance(reply, dict) or not reply.get('success', True):
        raise data.DatasetError('Perception did not acknowledge model loading')
    scenes = reply.get('scenes', [reply])
    if isinstance(scenes, dict):
        scenes = list(scenes.values())
    identities = [item.get('model', item) for item in scenes]
    expected = Path(expected_path).resolve()
    matching = [item for item in identities if item.get('loaded') and item.get('model_path')
                and Path(item['model_path']).resolve() == expected]
    if not matching or (expected_name and any(item.get('model_name') != expected_name for item in matching)):
        raise data.DatasetError('Perception loaded a different model; deployment was not verified')
    return reply


def _state(root):
    path = Path(root) / 'active.json'
    return json.loads(path.read_text()) if path.is_file() else {'history': []}


def _activate(candidate, models_root, config_path, reload_runtime, runtime_path, state_root):
    candidate, models_root, config_path = map(lambda p: Path(p).resolve(), (candidate, models_root, config_path))
    report_path = candidate / 'production_validation/acceptance.json'
    report = json.loads(report_path.read_text())
    if (report.get('status') != 'passed' or report.get('purpose') != 'factory_acceptance'
            or report.get('pipeline') != 'production_summit' or report.get('synthetic') is not False
            or report.get('missing_evidence') or not report.get('results')
            or not all(item.get('passed') for item in report['results'])):
        raise data.DatasetError('Use model requires passing physical production replay for every prescribed scene')
    model = model_io.load_model(candidate / 'model')
    name = model.get('model_name')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', name):
        raise data.DatasetError('Model needs a readable name containing letters, digits, hyphens or underscores')
    if (report.get('model_name') != name or model.get('model_type') != 'calibrated_bin_background'
            or model.get('synthetic', False)):
        raise data.DatasetError('Acceptance report does not identify this rim model')
    if not callable(reload_runtime):
        raise data.DatasetError('Activation requires a running Perception reload/status connection')
    models_root.mkdir(parents=True, exist_ok=True)
    destination = models_root / name
    if destination.exists():
        raise data.DatasetError(f'Model already installed: {destination}; use a new readable model name')
    model_path = destination / 'model.json'
    runtime_model_path = Path(runtime_path(model_path)) if runtime_path else model_path
    if not runtime_model_path.is_absolute() or '..' in runtime_model_path.parts:
        raise data.DatasetError('Installed model must have an absolute runtime path')
    original = config_path.read_text()
    state = _state(state_root)
    previous = {'config': original, 'model_path': data.read_yaml(config_path)['bin_filter_model_path'],
                'name': state.get('name')}
    if state.get('model_path') and state.get('runtime_model_path', state['model_path']) == previous['model_path']:
        previous['host_model_path'] = state['model_path']
        for key in ('acceptance', 'candidate_path'):
            if key in state:
                previous[key] = state[key]
    # Confirm that the service is reachable and the current model can be reloaded
    # before creating an installed directory or changing the active configuration.
    verify_identity(reload_runtime(), previous['model_path'], previous.get('name'))
    with tempfile.TemporaryDirectory(prefix='.install-', dir=models_root) as temporary:
        stage = Path(temporary) / name
        model_io.export_runtime(candidate / 'model', stage)
        stage.rename(destination)
    try:
        atomic_text(config_path, _config_text(original, runtime_model_path))
        identity = verify_identity(reload_runtime(), runtime_model_path, name)
    except Exception as error:
        atomic_text(config_path, original)
        try:
            verify_identity(reload_runtime(), previous['model_path'], previous.get('name'))
        except Exception as rollback_error:
            raise data.DatasetError(f'Candidate activation failed ({error}); configuration restored, '
                                    f'but runtime rollback failed ({rollback_error}). Stop picking and reload Perception.') from error
        # Keep the failed package for diagnosis while freeing this readable name
        # for a retry after the runtime problem has been corrected. Only do this
        # after the previous runtime identity has been verified above.
        attempts = state_root / 'failed_installations' / name
        try:
            attempts.mkdir(parents=True, exist_ok=True)
            counter = 1
            while (attempts / f'attempt-{counter:02d}').exists():
                counter += 1
            archive = attempts / f'attempt-{counter:02d}'
            shutil.move(str(destination), str(archive))
            atomic_text(archive / 'failure.json', json.dumps({
                'error': str(error), 'candidate_path': str(candidate),
                'acceptance': str(report_path)}, indent=2) + '\n')
        except OSError as archive_error:
            raise data.DatasetError(f'Candidate activation failed; previous model restored, '
                                    f'but failed installation could not be archived at {destination}: '
                                    f'{archive_error}') from error
        raise data.DatasetError(f'Candidate activation failed; previous model restored: {error}') from error
    result = {'name': name, 'model_path': str(model_path), 'config_path': str(config_path),
              'runtime_model_path': str(runtime_model_path),
              'candidate_path': str(candidate), 'acceptance': str(report_path),
              'runtime_verified': True, 'identity': identity,
              'history': [*state.get('history', []), previous]}
    atomic_text(state_root / 'active.json', json.dumps(result, indent=2) + '\n')
    return result


def _rollback(models_root, config_path, reload_runtime, state_root):
    models_root, config_path = Path(models_root).resolve(), Path(config_path).resolve()
    state = _state(state_root)
    if not state.get('history'):
        raise data.DatasetError('No previous bin model is available for rollback')
    previous = state['history'][-1]
    current = config_path.read_text()
    try:
        atomic_text(config_path, previous['config'])
        identity = verify_identity(reload_runtime(), previous['model_path'], previous.get('name'))
    except Exception as error:
        atomic_text(config_path, current)
        try:
            verify_identity(reload_runtime(), data.read_yaml(config_path)['bin_filter_model_path'], state.get('name'))
        except Exception as restore_error:
            raise data.DatasetError(f'Rollback failed ({error}); could not restore runtime ({restore_error})') from error
        raise data.DatasetError(f'Rollback failed; current model restored: {error}') from error
    result = {'name': previous.get('name'), 'model_path': previous.get('host_model_path', previous['model_path']),
              'runtime_model_path': previous['model_path'], 'config_path': str(config_path),
              'runtime_verified': True, 'identity': identity, 'history': state['history'][:-1]}
    for key in ('acceptance', 'candidate_path'):
        if key in previous:
            result[key] = previous[key]
    atomic_text(state_root / 'active.json', json.dumps(result, indent=2) + '\n')
    return result


@contextmanager
def deployment_lock(state_root):
    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.deployment.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise data.DatasetError('Another bin model installation or rollback is in progress') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _state_directory(models_root, state_root):
    root = Path(state_root).resolve()
    if root.is_relative_to(Path(models_root).resolve()):
        raise data.DatasetError('Deployment history must be outside the runtime model directory')
    return root


def activate(candidate, models_root, config_path, reload_runtime, runtime_path=None, *, state_root):
    state_root = _state_directory(models_root, state_root)
    with deployment_lock(state_root):
        return _activate(candidate, models_root, config_path, reload_runtime, runtime_path, state_root)


def rollback(models_root, config_path, reload_runtime, *, state_root):
    state_root = _state_directory(models_root, state_root)
    with deployment_lock(state_root):
        return _rollback(models_root, config_path, reload_runtime, state_root)
