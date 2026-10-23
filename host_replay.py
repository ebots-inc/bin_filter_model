#!/usr/bin/env python3
"""Run the production replay executable in its installed container from the host."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PATH_OPTIONS = {'--model', '--camera-calibration', '--robot-calibration', '--config',
                '--depth0', '--depth1', '--output', '--filtered-cloud'}
VALUE_OPTIONS = {'--exposure-ms', '--gain', '--black-level', '--scene-id', '--bin'}
DEFAULT_EXECUTABLE = '/root/perception/workspace/install/perception_node/lib/perception_node/summit_replay'


def mapped_arguments(runtime, arguments):
    if len(arguments) % 2:
        raise ValueError('Replay options require a value')
    converted, seen = [], set()
    for option, value in zip(arguments[::2], arguments[1::2]):
        if option not in PATH_OPTIONS | VALUE_OPTIONS or option in seen:
            raise ValueError(f'Unknown or repeated replay option: {option}')
        seen.add(option)
        converted.extend([option, str(runtime.to_runtime(Path(value))) if option in PATH_OPTIONS else value])
    return converted


def main(arguments=None, runtime_factory=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    try:
        container = os.environ.get('BIN_SETUP_REPLAY_CONTAINER')
        if not container or container == 'auto':
            from docker_runtime import discover_container
            artifacts = os.environ.get('BIN_SETUP_ARTIFACTS_ROOT')
            if not artifacts:
                raise ValueError('Automatic replay discovery requires BIN_SETUP_ARTIFACTS_ROOT')
            container = discover_container('perception', Path(artifacts))
        if runtime_factory is None:
            from docker_runtime import DockerRuntime
            runtime_factory = DockerRuntime
        runtime = runtime_factory(container)
        program = os.environ.get('BIN_SETUP_REPLAY_EXECUTABLE', DEFAULT_EXECUTABLE)
        if not program.startswith('/') or '..' in Path(program).parts:
            raise ValueError('BIN_SETUP_REPLAY_EXECUTABLE must be an absolute container path')
        # Bound the process inside the container too. Killing only docker exec's
        # host client can leave a native replay writing its result after timeout.
        mapped = mapped_arguments(runtime, arguments)
        outputs = [value for option, value in zip(mapped[::2], mapped[1::2])
                   if option in ('--output', '--filtered-cloud')]
        command = runtime.command(['timeout', '--kill-after=5s', '150s', program, *mapped],
                                  outputs=outputs)
        return subprocess.run(command, check=False).returncode
    except (ValueError, OSError) as error:
        print(f'host_replay: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
