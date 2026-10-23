"""Camera acquisition transport for the workstation Bin Setup application."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import dataset as data
from docker_runtime import DockerRuntime, discover_container


def json_report(output):
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise data.DatasetError('The camera helper returned no readiness report. Rebuild bin_filter_offline/native.')


def connection_report(report, runtime):
    status = report['camera_status']
    hardware = (status['received'] and status['node_status'] == 1 and status['hardware'] and
                status['age_seconds'] is not None and status['age_seconds'] <= 3 and
                len(status['denali']) == 1 and status['denali'][0]['sensor_en'] == 3)
    checks = [{'id': 'camera', 'label': f"Camera hardware — scene {report['scene_id']}", 'ready': bool(hardware),
               'message': 'Both Denali sensors are ready.' if hardware else '; '.join(report['missing']),
               'action': '' if hardware else 'Start the camera node in hardware mode on this ROS domain.'}]
    for key, label, action in [('denali', 'Denali acquisition service', 'Start the camera node.'),
                               ('eyes_depth', 'Eyes depth reconstruction',
                                'Start Eyes with artifacts/workflow_configs/bin_calibration.yaml.')]:
        service = report['services'][key]
        checks.append({'id': key, 'label': label, 'ready': service['ready'], 'message': service['name'],
                       'action': '' if service['ready'] else action})
    return {'ready': report['ready'], 'checking': False, 'checked_at': datetime.now(timezone.utc).isoformat(),
            'runtime': f"{runtime} · ROS domain {report['ros_domain_id']} · {report['rmw_implementation']}",
            'message': 'Camera and Eyes services are ready. Calibration is checked when capturing.' if report['ready'] else
                       'Start the missing camera services, then click Check connection.',
            'checks': checks, 'details': report}


class HostCameraCapture:
    runtime_root = '/root/artifacts/data_saving/bin_model/sessions'

    def __init__(self, artifacts, container='auto', program=None):
        self.artifacts = Path(artifacts).resolve()
        self.container, self.program = container, program

    def runtime(self):
        name = discover_container('eyes', self.artifacts) if self.container == 'auto' else self.container
        runtime = DockerRuntime(name)
        if runtime.to_runtime(self.artifacts) != '/root/artifacts':
            raise data.DatasetError('The camera container must share this station’s artifacts at /root/artifacts')
        return runtime

    def program_path(self, runtime):
        return self.program or runtime.to_runtime(self.artifacts / 'data_saving/bin_model/tools/native/bin_capture')

    def check(self, scene_id=0):
        runtime = self.runtime()
        command = runtime.command([self.program_path(runtime), '--check', '--scene-id', str(scene_id),
                                   '--ready-timeout-sec', '5'])
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        if result.returncode not in (0, 2):
            raise data.DatasetError(result.stderr.strip() or result.stdout.strip() or
                                    'Camera helper unavailable; build bin_filter_offline/native first.')
        try:
            report = json_report(result.stdout)
        except data.DatasetError as error:
            raise data.DatasetError(result.stderr.strip() or str(error)) from error
        return connection_report(report, runtime.container)

    def calibration_paths(self, folder, session, shared):
        entries = session.get('calibration', {})
        referenced = [name for name in ('camera', 'robot') if 'artifact_path' in entries.get(name, {})]
        if referenced:
            if len(referenced) != 2:
                raise data.DatasetError('Camera and robot calibration must both reference the station artifacts')
            paths = {}
            for name in referenced:
                reference = Path(entries[name]['artifact_path'])
                path = (self.artifacts / reference).resolve()
                deployment = (self.artifacts / 'calibration/deployment' / name).resolve()
                if reference.is_absolute() or '..' in reference.parts or not path.is_relative_to(deployment):
                    raise data.DatasetError(f'{name} calibration must reference a file in artifacts/calibration/deployment/{name}')
                if not path.is_file():
                    raise data.DatasetError(f'The station {name} calibration is missing: {path}')
                paths[name] = path
            return paths, False
        # Sessions created before deployment references used local snapshots.
        # Preserve those sessions without rewriting any saved calibration.
        paths = {}
        for name in ('camera', 'robot'):
            frozen = folder / f'calibration/{name}.yaml'
            published = shared / f'calibration/{name}.yaml'
            if data.read_yaml(frozen) != data.read_yaml(published):
                raise data.DatasetError(f'The shared {name} calibration differs from this session')
            paths[name] = published
        return paths, True

    def __call__(self, folder, session, capture, destination, progress):
        folder, destination = Path(folder), Path(destination)
        if destination.exists():
            raise data.DatasetError(f'Capture destination already exists: {destination}')
        runtime = self.runtime()
        part_name = Path(capture['part_name'])
        expected = Path('data_saving/bin_model/sessions') / session['session_id']
        if part_name.parent != expected or '..' in part_name.parts:
            raise data.DatasetError('Capture recipe is not registered to this Bin Setup session')
        shared = self.artifacts / expected
        part = self.artifacts / 'config/parts' / part_name.with_suffix('.yaml')
        frozen_part = folder / 'configs' / part.name
        if part.read_bytes() != frozen_part.read_bytes():
            raise data.DatasetError('Registered camera settings differ from this session’s frozen settings')
        calibration, relocated = self.calibration_paths(folder, session, shared)
        transfers = shared / 'transfers'
        transfers.mkdir(parents=True, exist_ok=True)
        # Keep failed attempts for diagnosis. A new attempt never overwrites them.
        index = 1
        attempt = transfers / f'{destination.name}-attempt-{index:02d}'
        while attempt.exists():
            index += 1
            attempt = transfers / f'{destination.name}-attempt-{index:02d}'
        attempt.mkdir()
        staged = attempt / 'capture'
        runtime_output = runtime.to_runtime(staged)
        arguments = [self.program_path(runtime), '--part-config', runtime.to_runtime(part),
                     '--part-name', str(part_name), '--camera-calibration', runtime.to_runtime(calibration['camera']),
                     '--robot-calibration', runtime.to_runtime(calibration['robot']),
                     '--scene-id', str(session['scene_id']), '--output', runtime_output]
        if relocated:
            arguments.append('--relocated-session')
        # The native client owns CUDA buffers until Eyes releases its imports.
        # A host deadline must not kill that owner while the request is active.
        command = runtime.command(arguments, outputs=[runtime_output])
        log = folder / 'logs' / (destination.name + '.log')
        log.parent.mkdir(exist_ok=True)
        progress(f"Capturing {capture['exposure_ms']:g} ms — {destination.name}")
        with log.open('w') as stream:
            # Ctrl+C stops the UI between captures; keep its active Docker client
            # outside the terminal process group so cleanup can finish safely.
            with subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                  start_new_session=True) as process:
                try:
                    returncode = process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    progress('Waiting for camera/Eyes to release buffers; no further captures will start '
                             f'until this request finishes. Log: {log}')
                    returncode = process.wait()
        if returncode:
            tail = '\n'.join(log.read_text(errors='replace').splitlines()[-12:])
            raise data.DatasetError(f'Capture failed. Log: {log}\n{tail}\nTemporary data: {attempt}')
        if not (staged / 'capture_manifest.yaml').is_file() or (staged / '.capture_in_progress').exists():
            raise data.DatasetError(f'Capture did not finish writing its manifest: {staged}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        incoming = Path(tempfile.mkdtemp(prefix='.incoming-', dir=destination.parent))
        try:
            shutil.copytree(staged, incoming, dirs_exist_ok=True)
            incoming.rename(destination)
        except Exception:
            shutil.rmtree(incoming)
            raise
        shutil.rmtree(attempt)
