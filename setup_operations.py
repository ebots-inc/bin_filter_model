#!/usr/bin/env python3
"""File and model operations used by the standalone Bin Setup application."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import shutil

import dataset as data
import deployment
import station_config
import cad_upload


class Operations:
    def __init__(self, artifacts, data_root, models_root, config, replay=None,
                 runtime_root=None, runtime_path=None, model_path_to_host=None, recipe_path=None):
        self.artifacts = Path(artifacts).resolve()
        self.data_root = Path(data_root).resolve()
        self.models_root = Path(models_root).resolve()
        self.deployment_state_root = self.data_root / 'deployment'
        self.layouts_root = self.data_root / 'layouts'
        self.config = Path(config).resolve()
        self.replay = replay
        # Session files live on this PC; capture settings may name the same
        # files through a different path inside the camera runtime.
        self.runtime_root = Path(runtime_root) if runtime_root is not None else self.data_root
        self.runtime_path = runtime_path
        self.model_path_to_host = model_path_to_host
        helper = Path(__file__).with_name('prepare_session.py')
        spec = importlib.util.spec_from_file_location('bin_collection', helper)
        self.collection = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.collection)
        self.recipe = data.read_yaml(recipe_path or helper.parent / 'configs/recipe.yaml')

    def inside(self, value, root=None):
        path = Path(value).resolve()
        root = root or self.data_root
        if not path.is_relative_to(root) or path == root:
            raise data.DatasetError('Selected path is outside the configured bin-setup storage')
        return path

    def run_cli(self, arguments, progress):
        command = [sys.executable, '-u', str(Path(__file__).with_name('bin_filter.py')), *map(str, arguments)]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            tail = []
            for line in process.stdout:
                message = line.rstrip()
                tail = (tail + [message])[-20:]
                progress(message)
            if process.wait():
                raise data.DatasetError('\n'.join(tail) or 'Model operation failed')

    def candidate(self, folder):
        folder = self.inside(folder)
        selection = folder / 'selected_candidate.json'
        path = folder / (json.loads(selection.read_text())['directory'] if selection.is_file() else 'candidate')
        return self.inside(path, folder)

    def review_path(self, folder):
        folder = self.inside(folder)
        session = json.loads((folder / 'session.json').read_text())
        if session.get('mode') == 'check':
            selection = folder / 'selected_check.json'
            relative = json.loads(selection.read_text())['directory'] if selection.is_file() else 'quick_check'
            return self.inside(folder / relative, folder) / 'acceptance.json'
        return self.candidate(folder) / 'production_validation/acceptance.json'

    def execute(self, request, progress, reload_runtime, capture_runtime=None):
        command = request['command']
        if command == 'catalog':
            self.data_root.mkdir(parents=True, exist_ok=True)
            return {'configuration': station_config.inventory(self.artifacts, self.recipe, layouts_root=self.layouts_root),
                    'sessions': [str(path.parent) for path in sorted(self.data_root.glob('*/session.json'))],
                    'active': deployment._state(self.deployment_state_root)}
        if command == 'prepare':
            mode = 'check' if request.get('quick') else 'guided'
            name = request.get('name') or None
            kwargs = {} if name is None else {'name': name}
            recipe = station_config.selected_recipe(self.artifacts, self.recipe, request, layouts_root=self.layouts_root)
            folder = self.collection.prepare_session(self.artifacts, self.data_root, self.runtime_root,
                                                       recipe, mode=mode, **kwargs)
            session = json.loads((folder / 'session.json').read_text())
            return {'session_path': str(folder), 'session': session}
        if command in ('layout_preview', 'layout_analyze', 'layout_save'):
            import layout_workflow
            return layout_workflow.execute(self.artifacts, self.data_root, self.recipe,
                                           request, capture_runtime, progress)
        if command == 'preview':
            import preview
            folder = self.inside(request['session_path'])
            session = json.loads((folder / 'session.json').read_text())
            if not callable(capture_runtime):
                raise data.DatasetError('Preview requires the capture service')
            root = folder / 'previews'
            root.mkdir(exist_ok=True)
            index = 1
            while (root / f'preview-{index:02d}').exists():
                index += 1
            destination = root / f'preview-{index:02d}'
            progress('Capturing a fresh preview; this image is excluded from training and validation.')
            capture_runtime(folder, session, session['captures'][0], destination, progress)
            result = preview.render(destination, folder)
            result['storage_free_gb'] = round(shutil.disk_usage(folder).free / 1e9, 1)
            return result
        if command == 'build':
            folder = self.inside(request['session_path'])
            session = json.loads((folder / 'session.json').read_text())
            selection = request.get('cad_reference')
            cad_reference = (cad_upload.selected_plan(folder, selection, session)
                if isinstance(selection, str) and selection.startswith('session:') else
                station_config.selected_cad_reference(self.artifacts, selection, session))
            self.collection.verify_session(folder)
            candidate = folder / 'candidate'
            index = 1
            while candidate.exists():
                index += 1
                candidate = folder / f'candidate-{index:02d}'
            name = folder.name if index == 1 else f'{folder.name}-{index:02d}'
            arguments = ['build-session', '--session', folder, '--output', candidate,
                         '--model-name', name, '--pipeline-config', self.config]
            if cad_reference is not None:
                arguments += ['--cad-reference', cad_reference]
            if self.replay:
                arguments += ['--replay-executable', self.replay]
            self.run_cli(arguments, progress)
            acceptance = json.loads((candidate / 'production_validation/acceptance.json').read_text())
            deployment.atomic_text(folder / 'selected_candidate.json', json.dumps({'directory': candidate.name}))
            return {'candidate_path': str(candidate), 'acceptance': acceptance}
        if command in ('cad_upload', 'cad_preview', 'cad_confirm'):
            folder = self.inside(request['session_path'])
            session = json.loads((folder / 'session.json').read_text())
            if session.get('mode') != 'guided':
                raise data.DatasetError('CAD references require a model collection session')
            if command == 'cad_upload':
                return cad_upload.upload(folder, session, request['filename'], request['payload'], request.get('target', 'all'))
            if command == 'cad_confirm':
                if request.get('confirmed') is not True:
                    raise data.DatasetError('Confirm that you reviewed the CAD size and orientation')
                return cad_upload.confirm(folder, session, request['preview_path'])
            self.collection.verify_session(folder, decode_images=False)
            # A candidate may already supply the measured empty-training fit.
            # Match provenance/settings; CAD-guided candidates are not baselines.
            candidates = [self.candidate(folder) / 'model/model.json',
                          *sorted(folder.glob('candidate*/model/model.json'), reverse=True),
                          *sorted(folder.glob('cad_baselines/baseline-*/model/model.json'), reverse=True)]
            baseline = next((model for path in candidates
                             if (model := cad_upload.matching_baseline(path, folder, session)) is not None), None)
            if baseline is None:
                progress('Fitting an empty-training baseline for CAD placement. Held-out captures are excluded from fitting.')
                destination = cad_upload._new_directory(folder / 'cad_baselines', 'baseline')
                self.run_cli(['import-session', '--session', folder, '--dataset', destination / 'dataset'], progress)
                self.run_cli(['calibrate', '--dataset', destination / 'dataset', '--output', destination / 'model',
                              '--model-name', 'cad-empty-training-baseline'], progress)
                baseline = cad_upload.matching_baseline(destination / 'model/model.json', folder, session)
                if baseline is None:
                    raise data.DatasetError('CAD baseline does not match this session’s empty training data')
            return cad_upload.preview(folder, session, baseline, request.get('selections'), progress)
        if command == 'review':
            report = self.review_path(request['session_path'])
            return {'acceptance': json.loads(report.read_text())}
        if command == 'activate':
            candidate = self.inside(request['candidate_path'])
            mapping = {} if self.runtime_path is None else {'runtime_path': self.runtime_path}
            return deployment.activate(candidate, self.models_root, self.config, reload_runtime,
                                       state_root=self.deployment_state_root, **mapping)
        if command == 'rollback':
            return deployment.rollback(self.models_root, self.config, reload_runtime,
                                       state_root=self.deployment_state_root)
        if command == 'quick_check':
            folder = self.inside(request['session_path'])
            self.collection.verify_session(folder)
            active_path = Path(data.read_yaml(self.config)['bin_filter_model_path'])
            if self.model_path_to_host is not None:
                active_path = Path(self.model_path_to_host(active_path))
            if not active_path.is_file():
                raise data.DatasetError('The configured model is unavailable')
            index = 1
            attempt = folder / 'quick_check' / f'attempt-{index:02d}'
            while attempt.exists():
                index += 1
                attempt = folder / 'quick_check' / f'attempt-{index:02d}'
            attempt.mkdir(parents=True)
            self.run_cli(['import-session', '--session', folder, '--dataset', attempt / 'dataset'], progress)
            args = ['validate-production', '--dataset', attempt / 'dataset', '--model', active_path,
                    '--pipeline-config', self.config, '--output', attempt / 'results', '--quick']
            if self.replay:
                args += ['--replay-executable', self.replay]
            self.run_cli(args, progress)
            acceptance = json.loads((attempt / 'results/acceptance.json').read_text())
            deployment.atomic_text(folder / 'selected_check.json',
                                   json.dumps({'directory': str((attempt / 'results').relative_to(folder))}))
            return {'acceptance': acceptance}
        raise data.DatasetError(f'Unknown Bin Setup operation: {command}')
