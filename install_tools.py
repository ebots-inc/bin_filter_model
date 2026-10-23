#!/usr/bin/env python3
"""Stage Bin Setup Python tools while preserving the installed camera helper."""
import argparse
from pathlib import Path
import shutil
import tempfile


def install(artifacts):
    source = Path(__file__).resolve().parent
    parent = Path(artifacts).resolve() / 'data_saving/bin_model'
    if not (parent / 'prepare_session.py').is_file():
        raise ValueError('Expected an artifacts tree containing the bin collection helper')
    destination = parent / 'tools'
    previous = parent / 'tools.previous'
    with tempfile.TemporaryDirectory(prefix='.bin-tools-', dir=parent) as temporary:
        stage = Path(temporary) / 'tools'
        stage.mkdir()
        for path in sorted(source.glob('*.py')):
            shutil.copyfile(path, stage / path.name)
        shutil.copyfile(source / 'requirements.txt', stage / 'requirements.txt')
        shutil.copyfile(source / 'bin_setup.html', stage / 'bin_setup.html')
        if (destination / 'native').is_dir():
            shutil.copytree(destination / 'native', stage / 'native')
        if previous.exists():
            shutil.rmtree(previous)
        if destination.exists():
            destination.rename(previous)
        try:
            stage.rename(destination)
        except Exception:
            if previous.exists():
                previous.rename(destination)
            raise
    return destination


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--artifacts-root', required=True)
    args = cli.parse_args()
    print(install(args.artifacts_root))
