"""Local Docker access for camera/replay clients; never starts robot nodes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import dataset as data


def docker_json(arguments):
    try:
        result = subprocess.run(['docker', *arguments], capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise data.DatasetError(f'Cannot contact the local Docker runtime: {error}') from error
    if result.returncode:
        raise data.DatasetError(result.stderr.strip() or 'Docker request failed')
    return json.loads(result.stdout)


class DockerRuntime:
    def __init__(self, container, *, inspection=None):
        self.container = container
        self.info = inspection or docker_json(['inspect', container])[0]
        if not self.info.get('State', {}).get('Running'):
            raise data.DatasetError(f'Container {container} is stopped. Start the station camera environment first.')
        self.mounts = [(Path(item['Source']).resolve(), Path(item['Destination']))
                       for item in self.info.get('Mounts', []) if item.get('Type') == 'bind']

    def to_runtime(self, path):
        path = Path(path).resolve()
        for source, destination in sorted(self.mounts, key=lambda item: len(item[0].parts), reverse=True):
            if path.is_relative_to(source):
                return str(destination / path.relative_to(source))
        raise data.DatasetError(f'{path} is not shared with {self.container}')

    def to_host(self, path):
        path = Path(path)
        if not path.is_absolute() or '..' in path.parts:
            raise data.DatasetError('Runtime paths must be absolute')
        for source, destination in sorted(self.mounts, key=lambda item: len(item[1].parts), reverse=True):
            if path.is_relative_to(destination):
                return str(source / path.relative_to(destination))
        raise data.DatasetError(f'{path} has no host bind mount in {self.container}')

    def command(self, argv, *, outputs=()):
        # Only this constant setup program is interpreted by Bash. All caller
        # arguments remain distinct positional arguments through exec "$@".
        setup = '''set -e
umask 022
declare -A bin_setup_ros=()
for bin_setup_key in ROS_DOMAIN_ID ROS_LOCALHOST_ONLY RMW_IMPLEMENTATION; do
  if [ -v "$bin_setup_key" ]; then bin_setup_ros["$bin_setup_key"]="${!bin_setup_key}"; fi
done
source /opt/ros/humble/setup.bash
for setup_file in /root/workspace/install/setup.bash /root/eyes/workspace/install/setup.bash /root/perception/workspace/install/setup.bash; do
  if [ -f "$setup_file" ]; then source "$setup_file"; fi
done
for bin_setup_key in "${!bin_setup_ros[@]}"; do export "$bin_setup_key=${bin_setup_ros[$bin_setup_key]}"; done
owner="$1"
count="$2"
shift 2
outputs=()
for ((i=0; i<count; i++)); do outputs+=("$1"); shift; done
handoff() {
  for output in "${outputs[@]}"; do
    if [ -e "$output" ]; then chown -R -- "$owner" "$output"; fi
  done
}
if [ "$count" -gt 0 ]; then
  trap handoff EXIT
  "$@"
else
  exec "$@"
fi
'''
        command = ['docker', 'exec', '-e', 'PYTHONDONTWRITEBYTECODE=1']
        # Explicit workstation ROS settings take precedence; otherwise retain
        # the camera container's own ROS domain and middleware configuration.
        for key in ('ROS_DOMAIN_ID', 'ROS_LOCALHOST_ONLY', 'RMW_IMPLEMENTATION'):
            if key in os.environ:
                command += ['-e', f'{key}={os.environ[key]}']
        for output in outputs:
            self.to_host(output)  # Ownership handoff is limited to shared outputs.
        return [*command, self.container, 'bash', '-c', setup, 'bin-setup-client',
                f'{os.getuid()}:{os.getgid()}', str(len(outputs)), *map(str, outputs), *map(str, argv)]


def discover_container(kind, artifacts):
    # Container names are local deployment choices. Match the shared station
    # artifacts and an installed executable instead of a developer's naming
    # convention. This probe never starts a ROS node or sends a camera request.
    programs = {
        'eyes': ['/root/workspace/install/ebots_eyes/lib/ebots_eyes/ebots_eyes',
                 '/root/eyes/workspace/install/ebots_eyes/lib/ebots_eyes/ebots_eyes'],
        'perception': ['/root/perception/workspace/install/perception_node/lib/perception_node/summit_replay',
                       '/root/workspace/install/perception_node/lib/perception_node/summit_replay'],
    }
    if kind not in programs:
        raise data.DatasetError(f'Unknown container capability: {kind}')
    try:
        result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise data.DatasetError(f'Cannot contact the local Docker runtime: {error}') from error
    if result.returncode:
        raise data.DatasetError(result.stderr.strip() or 'Cannot list camera containers')
    candidates = []
    for name in result.stdout.splitlines():
        try:
            runtime = DockerRuntime(name)
            if runtime.to_runtime(artifacts) != '/root/artifacts':
                continue
            probe = subprocess.run(['docker', 'exec', name, 'sh', '-c',
                'for program do if [ -x "$program" ]; then exit 0; fi; done; exit 1',
                'bin-setup-discovery', *programs[kind]],
                capture_output=True, text=True, timeout=10)
            if probe.returncode == 0:
                candidates.append(name)
        except (data.DatasetError, subprocess.TimeoutExpired):
            continue
    if len(candidates) != 1:
        option = 'camera' if kind == 'eyes' else 'replay'
        details = ', '.join(candidates) if candidates else 'none'
        raise data.DatasetError(f'Found {len(candidates)} running containers with {kind} tools and this station’s '
                               f'artifacts ({details}). Select one with --{option}-container NAME.')
    return candidates[0]
