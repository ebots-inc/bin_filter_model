#!/usr/bin/env python3
"""Standalone local Bin Setup UI: guide, acquire, fit and validate saved scenes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
from urllib.parse import urlsplit, unquote

import dataset as data
import deployment
import cad_upload
from camera_runtime import HostCameraCapture, json_report
from docker_runtime import DockerRuntime, discover_container
from setup_operations import Operations


class Controller:
    def __init__(self, operations, capture, reload_runtime, check_camera=None):
        self.ops, self.capture, self.reload_runtime = operations, capture, reload_runtime
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.busy = False
        self.active_command = None
        self.stop_requested = threading.Event()
        self.folder = None
        self.token = secrets.token_urlsafe(24)
        self.logs = []
        self.error = None
        self.result = None
        self.check_camera = check_camera or getattr(capture, 'check', None)
        self.camera_connection = {'ready': None, 'checking': False, 'checked_at': None, 'checks': []}
        self.selection_file = self.ops.data_root / 'selected_session.json'
        if self.selection_file.is_file():
            try:
                selected = Path(json.loads(self.selection_file.read_text())['session_path'])
                try:
                    self.folder = self.ops.inside(selected)
                except data.DatasetError:
                    # Older servers stored their container path. Reopen only
                    # the local session whose frozen metadata names that path.
                    local = self.ops.inside(self.ops.data_root / selected.name)
                    saved = json.loads((local / 'session.json').read_text())
                    if saved.get('runtime_path') != str(selected):
                        raise data.DatasetError('Saved session belongs to another data root')
                    self.folder = local
                self._session()
            except (ValueError, OSError, KeyError):
                self.folder = None

    def log(self, message):
        with self.lock:
            self.logs = (self.logs + [message])[-100:]

    def _session(self):
        return json.loads((self.folder / 'session.json').read_text()) if self.folder else None

    def _stage_state(self, session):
        stages = []
        by_id = {entry['id']: entry for entry in session['captures']}
        for stage in session.get('stages', []):
            captures = [by_id[key] for key in stage['capture_ids']]
            complete = sum((self.folder / entry['path'] / 'capture_manifest.yaml').is_file() and
                           not (self.folder / entry['path'] / '.capture_in_progress').exists() for entry in captures)
            finished = complete == len(captures) and (self.folder / 'completed_stages' / (stage['id'] + '.json')).is_file()
            stages.append(dict(stage, completed=complete, total=len(captures), finished=finished))
        return stages

    def state(self):
        with self.lock:
            session = self._session()
            stages = self._stage_state(session) if session else []
            current = next((stage for stage in stages if not stage['finished']), None)
            return {'busy': self.busy, 'active_command': self.active_command, 'stop_requested': self.stop_requested.is_set(),
                    'session_path': str(self.folder) if self.folder else None, 'session': session,
                    'stages': stages, 'current_stage': current, 'ready_token': self.token,
                    'completed': sum(stage['completed'] for stage in stages),
                    'total': sum(stage['total'] for stage in stages), 'logs': list(self.logs),
                    'error': self.error, 'result': self.result,
                    'camera_connection': dict(self.camera_connection),
                    'storage_free_gb': round(shutil.disk_usage(self.ops.data_root).free / 1e9, 1),
                    'candidate_exists': bool(self.folder and (self.ops.candidate(self.folder) / 'production_validation/acceptance.json').is_file()),
                    'review_exists': bool(self.folder and self.ops.review_path(self.folder).is_file()),
                    'cad': cad_upload.state(self.folder),
                    'catalog': self.ops.execute({'command': 'catalog'}, self.log, self.reload_runtime)}

    def submit(self, request):
        with self.lock:
            command = request.get('command')
            if command == 'stop':
                if self.active_command != 'capture_stage':
                    raise data.DatasetError('There is no capture sweep to stop')
                self.stop_requested.set()
                self.log('Stop requested. The current capture will finish; no further exposure will start.')
                return
            if self.busy:
                raise data.DatasetError('An operation is still running')
            allowed = {'prepare', 'select', 'preview', 'capture_stage', 'build', 'review', 'activate', 'rollback', 'quick_check', 'check_connection',
                       'cad_upload', 'cad_preview', 'cad_confirm', 'layout_preview', 'layout_analyze', 'layout_save'}
            if command not in allowed:
                raise data.DatasetError('Unknown Bin Setup operation')
            if command == 'capture_stage':
                state = self.state()
                if (request.get('ready_token') != self.token or not state['current_stage'] or
                        request.get('stage_id') != state['current_stage']['id']):
                    raise data.DatasetError('The scene changed. Review the current instructions before confirming Ready.')
                # Consume readiness before starting the worker, including on failure.
                self.token = secrets.token_urlsafe(24)
            elif command == 'layout_preview':
                if request.get('empty_confirmed') is not True:
                    raise data.DatasetError('Empty all bins, remove your hands, and confirm the scene is ready before taking a layout photo.')
                if type(request.get('scene_id')) is not int or not 0 <= request['scene_id'] <= 255:
                    raise data.DatasetError('Select the calibrated scene before taking a layout photo.')
                if request.get('ready_token') != self.token:
                    raise data.DatasetError('The scene changed. Review the empty-bin instructions before taking a new layout photo.')
                self.token = secrets.token_urlsafe(24)
            self.busy = True
            self.active_command = command
            self.error = None
            self.result = None
            self.stop_requested.clear()
            self.pool.submit(self._perform, dict(request))

    def _check_connection(self, required=False, scene_id=None):
        with self.lock:
            self.camera_connection = dict(self.camera_connection, checking=True)
        try:
            session = self._session()
            scene = scene_id if scene_id is not None else (session['scene_id'] if session else self.ops.recipe['scene_id'])
            if type(scene) is not int or not 0 <= scene <= 255:
                raise data.DatasetError('Scene ID must be an integer from 0 to 255')
            if self.check_camera is None:
                raise data.DatasetError('No camera readiness checker is configured')
            report = self.check_camera(scene)
            report = dict(report, scene_id=scene)
        except Exception as error:
            report = {'ready': False, 'message': str(error), 'checks': [
                {'id': 'client', 'label': 'Camera connection', 'ready': False, 'message': str(error),
                 'action': 'Check the camera container and native helper installation.'}]}
        with self.lock:
            self.camera_connection = dict(report, checking=False, checked_at=datetime.now(timezone.utc).isoformat())
        if required and not report.get('ready'):
            reasons = report.get('details', {}).get('missing', [])
            raise data.DatasetError('Camera is not ready. ' + ('; '.join(reasons) or report.get('message', 'Click Check connection.')))
        return self.camera_connection

    def _select(self, folder):
        folder = self.ops.inside(folder)
        session = json.loads((folder / 'session.json').read_text())
        if session.get('mode') not in ('guided', 'check') or not session.get('stages'):
            raise data.DatasetError('Open a guided Bin Setup or Bin Check session')
        with self.lock:
            self.folder = folder
            self.token = secrets.token_urlsafe(24)
            self.camera_connection = {'ready': None, 'checking': False, 'checked_at': None, 'checks': [],
                                      'scene_id': session['scene_id']}
            deployment.atomic_text(self.selection_file, json.dumps({'session_path': str(folder)}))

    def _capture_stage(self, request):
        # A stale UI must not archive captures or record a confirmation before
        # checking that actual camera servers are still available.
        self._check_connection(required=True)
        session = self._session()
        stage = next(item for item in session['stages'] if item['id'] == request['stage_id'])
        captures = [item for item in session['captures'] if item['id'] in stage['capture_ids']]
        # Retake the whole interrupted arrangement. Mixing an old partial sweep
        # with a newly prepared scene would invalidate the exposure comparison.
        existing = [self.folder / item['path'] for item in captures if (self.folder / item['path']).exists()]
        confirmation = self.folder / stage['confirmation_path']
        checkpoint = self.folder / 'completed_stages' / (stage['id'] + '.json')
        if existing or confirmation.exists() or checkpoint.exists():
            attempts = self.folder / 'previous_attempts' / stage['id']
            attempts.mkdir(parents=True, exist_ok=True)
            index = 1
            while (attempts / f'attempt-{index:02d}').exists():
                index += 1
            archive = attempts / f'attempt-{index:02d}'
            archive.mkdir()
            for path in existing:
                path.rename(archive / path.name)
            for entry in captures:
                log = self.folder / 'logs' / (Path(entry['path']).name + '.log')
                if log.is_file():
                    (archive / 'logs').mkdir(exist_ok=True)
                    log.rename(archive / 'logs' / log.name)
            if confirmation.exists():
                confirmation.rename(archive / 'confirmation.json')
            if checkpoint.exists():
                checkpoint.rename(archive / 'completion.json')
            self.log('Preserved the interrupted stage under previous_attempts; recapturing this scene in full.')
        deployment.atomic_text(confirmation, json.dumps({'session_id': session['session_id'], 'stage_id': stage['id'],
            'gate_token': request['ready_token'], 'confirmed_at': datetime.now(timezone.utc).isoformat(),
            'source': 'standalone_bin_setup'}, indent=2) + '\n')
        for entry in captures:
            if self.stop_requested.is_set():
                self.log('Stopped between captures. Prepare the same scene and click Ready to retake this stage.')
                return {'stopped': True}
            destination = self.folder / entry['path']
            self.capture(self.folder, session, entry, destination, self.log)
            self.ops.collection.verify_capture(self.folder, session, entry)
            self.log(f"Saved {entry['id']}")
        deployment.atomic_text(self.folder / 'completed_stages' / (stage['id'] + '.json'),
                               json.dumps({'stage_id': stage['id'], 'capture_ids': stage['capture_ids']}))
        self.log('Scene complete. Prepare the next scene and confirm Ready when finished.')
        return {'stage_complete': stage['id']}

    def _perform(self, request):
        try:
            command = request['command']
            if command == 'check_connection':
                result = self._check_connection(scene_id=request.get('scene_id'))
            elif command == 'prepare':
                result = self.ops.execute(request, self.log, self.reload_runtime, self.capture)
                self._select(result['session_path'])
                self._check_connection()
            elif command == 'select':
                self._select(request['session_path'])
                self._check_connection()
                result = {'opened': str(self.folder)}
            elif command == 'capture_stage':
                result = self._capture_stage(request)
            elif command in ('layout_preview', 'layout_analyze', 'layout_save'):
                # Layout commissioning precedes session creation. An unrelated
                # open collection session remains selected and unchanged.
                if command == 'layout_preview':
                    self._check_connection(required=True, scene_id=request['scene_id'])
                result = self.ops.execute(request, self.log, self.reload_runtime, self.capture)
            else:
                if command != 'rollback' and self.folder is None:
                    raise data.DatasetError('Prepare or open a session first')
                if self.folder:
                    request['session_path'] = str(self.folder)
                    request['candidate_path'] = str(self.ops.candidate(self.folder))
                if command == 'preview':
                    self._check_connection(required=True)
                result = self.ops.execute(request, self.log, self.reload_runtime, self.capture)
            with self.lock:
                self.result = result
        except Exception as error:
            with self.lock:
                self.error = str(error)
            self.log(str(error))
        finally:
            with self.lock:
                self.busy = False
                self.active_command = None
                self.token = secrets.token_urlsafe(24)


def make_server(controller, host='127.0.0.1', port=8765):
    token = secrets.token_urlsafe(32)
    page = Path(__file__).with_name('bin_setup.html').read_text().replace('__API_TOKEN__', token)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, value, status=200, content_type='application/json'):
            body = value.encode() if isinstance(value, str) else json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('X-Frame-Options', 'DENY')
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            # This UI is local to the station. Also reject browser DNS rebinding.
            hostname = urlsplit('http://' + self.headers.get('Host', '')).hostname
            return hostname in ('127.0.0.1', 'localhost', '::1')

        def do_GET(self):
            if not self.valid_host():
                return self.send({'error': 'Local access only'}, 403)
            if self.path == '/':
                return self.send(page, content_type='text/html; charset=utf-8')
            if self.path == '/guide':
                return self.send(Path(__file__).with_name('guide.html').read_text(),
                                 content_type='text/html; charset=utf-8')
            guide_assets = {'/guide/assets/capture-cycle.svg': 'capture-cycle.svg',
                            '/guide/assets/resistor-placements.svg': 'resistor-placements.svg',
                            '/guide/assets/layout-setup.svg': 'layout-setup.svg'}
            if self.path in guide_assets:
                asset = Path(__file__).parent / 'docs' / guide_assets[self.path]
                return self.send(asset.read_text(), content_type='image/svg+xml')
            if self.path == '/api/state':
                return self.send(controller.state())
            self.send({'error': 'Not found'}, 404)

        def do_POST(self):
            if not self.valid_host() or self.headers.get('X-Bin-Setup-Token') != token:
                return self.send({'error': 'Refresh the local Bin Setup page'}, 403)
            if self.path not in ('/api/command', '/api/cad-upload'):
                return self.send({'error': 'Not found'}, 404)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if self.path == '/api/cad-upload':
                    if not 0 < size <= cad_upload.MAX_UPLOAD_BYTES:
                        raise ValueError('Choose an STL no larger than 16 MiB')
                    if self.headers.get('Content-Type') != 'application/octet-stream':
                        raise ValueError('Expected an STL file upload')
                    payload = self.rfile.read(size)
                    if len(payload) != size:
                        raise ValueError('The CAD upload was incomplete')
                    controller.submit({'command': 'cad_upload', 'payload': payload,
                        'filename': unquote(self.headers.get('X-Cad-Filename', '')),
                        'target': self.headers.get('X-Cad-Target', 'all')})
                    return self.send({'accepted': True}, 202)
                if not 0 < size <= 16384:
                    raise ValueError('Invalid request length')
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError('Expected an operation')
                controller.submit(request)
                self.send({'accepted': True}, 202)
            except (ValueError, KeyError, OSError) as error:
                self.send({'error': str(error)}, 409)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    workspace = Path(__file__).resolve().parent.parent
    parser.add_argument('--artifacts-root', default=str(workspace / 'ebots_system_artifacts/artifacts_cell_4'))
    parser.add_argument('--data-root', default=str(workspace / 'data/bin_model'))
    parser.add_argument('--models-root')
    parser.add_argument('--pipeline-config')
    parser.add_argument('--recipe', help='Collection recipe YAML; defaults to bin_filter_offline/configs/recipe.yaml')
    parser.add_argument('--capture-executable')
    parser.add_argument('--replay-executable')
    parser.add_argument('--camera-container', default='auto', help='Eyes container name, or auto to find the station container')
    parser.add_argument('--replay-container', default='auto', help='Container holding summit_replay; only used for validation/activation')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    root = Path(args.data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # One collector owns a session root. A browser refresh does not release it.
    with (root / '.bin_setup.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Bin Setup is already running for this data root')
        artifacts = Path(args.artifacts_root).resolve()
        capture = HostCameraCapture(artifacts, args.camera_container, args.capture_executable)
        os.environ['BIN_SETUP_ARTIFACTS_ROOT'] = str(artifacts)
        os.environ['BIN_SETUP_REPLAY_CONTAINER'] = args.replay_container
        if args.replay_executable:
            os.environ['BIN_SETUP_REPLAY_EXECUTABLE'] = args.replay_executable
        replay = str(Path(__file__).with_name('host_replay.py'))

        def replay_runtime():
            name = discover_container('perception', artifacts) if args.replay_container == 'auto' else args.replay_container
            return DockerRuntime(name)

        def runtime_reload():
            runtime = replay_runtime()
            script = Path(__file__).with_name('runtime_reload_client.py').read_text()
            result = subprocess.run(runtime.command(['timeout', '--kill-after=5s', '75s', 'python3', '-c', script]),
                                    capture_output=True, text=True, timeout=85)
            if result.returncode:
                raise data.DatasetError(result.stderr.strip() or result.stdout.strip() or 'Model reload failed')
            return json_report(result.stdout)

        operations = Operations(artifacts, root,
            args.models_root or str(Path(args.artifacts_root) / 'calibration/deployment/bin'),
            args.pipeline_config or str(Path(args.artifacts_root) / 'config/pose_estimation/contour_summit_pose_params.yaml'),
            replay, runtime_root=capture.runtime_root,
            runtime_path=lambda path: replay_runtime().to_runtime(path),
            model_path_to_host=lambda path: replay_runtime().to_host(path), recipe_path=args.recipe)
        controller = Controller(operations, capture, runtime_reload)
        server = make_server(controller, port=args.port)
        controller.submit({'command': 'check_connection'})
        print(f'Bin Setup: http://localhost:{server.server_port} — no capture starts until Scene ready is pressed.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            controller.stop_requested.set()
        finally:
            server.server_close()
            controller.pool.shutdown(wait=True)


if __name__ == '__main__':
    main()
