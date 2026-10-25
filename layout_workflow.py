"""Save reviewed layouts from camera previews in local setup storage."""
from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import re

import dataset as data
import layout_geometry as geometry
import layout_setup
import prepare_session as collection


def _image(folder, name):
    path = folder / name
    if not path.is_file() or path.resolve().parent != folder:
        raise data.DatasetError('The layout review image is missing; capture a new preview')
    return base64.b64encode(path.read_bytes()).decode('ascii')


def _new_analysis(preview):
    for index in range(1, 100000):
        folder = preview / f'analysis-{index:02d}'
        try:
            folder.mkdir()
        except FileExistsError:
            continue
        return folder
    raise data.DatasetError('No unused layout review name is available')


def _analyze(artifacts, storage, recipe, request, progress):
    points, preview = layout_setup.load_preview(storage, request['preview_path'])
    folder = Path(preview['path'])
    reference = layout_setup.validate_preview_calibration(artifacts, folder, preview)
    selections = request.get('bins')
    if not isinstance(selections, list) or not 1 <= len(selections) <= 32:
        raise data.DatasetError('Draw and name each bin before finding the interior and rim')
    if any(not isinstance(item, dict) for item in selections):
        raise data.DatasetError('Draw and name each bin before finding the interior and rim')
    rectangles = [{'bin_id': item.get('bin_id'), 'rectangle': item.get('rectangle')}
                  for item in selections]
    bounds = [{'bin_id': item['bin_id'],
               'xy_bounds_mm': geometry.pixels_to_bounds(item['rectangle'], preview)}
              for item in rectangles]
    progress('Estimating the interior and flat top rim of each marked bin from measured depth.')
    training_rounds = int(recipe['empty_train_rounds'])
    if training_rounds < 2:
        raise data.DatasetError('The collection recipe needs at least two empty training rounds per exposure')
    result = geometry.infer_layout(points, bounds, preview['scene_id'], preview['frame_id'],
                                   min_capture_support=min(5, training_rounds))
    rendered = geometry.render_preview(points, viewport=preview, bins=result['bins'])
    # Coordinate arrays are only needed to draw the overlay, not for UI polling.
    summaries = [{key: value for key, value in item.items()
                  if key not in ('rim_xyz_mm', 'interior_xyz_mm')} for item in result['bins']]
    settings = result['settings']
    settings['calibration_reference'] = reference
    analysis = _new_analysis(folder)
    review = {'path': str(analysis), 'preview_path': str(folder),
              'scene_id': preview['scene_id'], 'frame_id': preview['frame_id'],
              'selections': rectangles, 'bins': summaries, 'notes': result['notes']}
    collection.write_atomic(analysis / 'layout.yaml', collection.dump_yaml(settings))
    (analysis / 'review.png').write_bytes(base64.b64decode(rendered['preview_png'], validate=True))
    collection.write_atomic(analysis / 'review.json', json.dumps(review, indent=2) + '\n')
    progress('Review the highlighted interior and top rim for every bin, then save the layout.')
    return {'layout_analysis': {**review, 'preview_png': rendered['preview_png'],
                                'preview': {**preview, 'preview_png': _image(folder, 'preview.png')}}}


def _save(artifacts, storage, request):
    if request.get('confirmed') is not True:
        raise data.DatasetError('Review every highlighted bin interior and top rim before saving the layout')
    name = request.get('name')
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 100:
        raise data.DatasetError('Give the layout a short, readable name')
    name = name.strip()
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')
    if not slug:
        raise data.DatasetError('Include letters or numbers in the layout name')
    analysis = Path(request['analysis_path']).resolve()
    if not re.fullmatch(r'analysis-\d+', analysis.name):
        raise data.DatasetError('Select a completed layout review before saving')
    _, preview = layout_setup.load_preview(storage, analysis.parent)
    for filename in ('review.json', 'layout.yaml', 'review.png'):
        path = analysis / filename
        if not path.is_file() or path.resolve().parent != analysis:
            raise data.DatasetError('The layout review is incomplete; find the bin surfaces again')
    review = json.loads((analysis / 'review.json').read_text())
    if review.get('path') != str(analysis) or review.get('preview_path') != preview['path']:
        raise data.DatasetError('The layout review location changed; find the bin surfaces again')
    reference = layout_setup.validate_preview_calibration(artifacts, analysis.parent, preview)
    settings = data.read_yaml(analysis / 'layout.yaml')
    if (settings.get('scene_id') != preview['scene_id'] or settings.get('frame_id') != preview['frame_id']
            or settings.get('calibration_reference') != reference):
        raise data.DatasetError('The layout review does not match its calibrated preview; find the surfaces again')
    data.validate_bin_ids(settings.get('bins'))
    settings = deepcopy(settings)
    settings['name'] = name
    settings['layout_review'] = str(analysis.relative_to(storage))
    root = storage / 'layouts'
    if root.resolve().parent != storage:
        raise data.DatasetError('Saved layouts must stay under the configured data directory')
    root.mkdir(exist_ok=True)
    contents = collection.dump_yaml(settings)
    # Exclusive creation leaves earlier layouts and existing sessions unchanged.
    for index in range(1, 100000):
        filename = f'{slug}.yaml' if index == 1 else f'{slug}-{index:02d}.yaml'
        path = root / filename
        try:
            with path.open('x') as stream:
                stream.write(contents)
        except FileExistsError:
            continue
        context = {key: preview[key] for key in
                   ('scene_id', 'frame_id', 'camera_calibration', 'robot_calibration')}
        return {'layout_saved': {'path': str(path), 'label': name, **context}}
    raise data.DatasetError('No unused layout name is available')


def execute(artifacts, data_root, recipe, request, capture_runtime, progress):
    artifacts, storage = Path(artifacts).resolve(), Path(data_root).resolve()
    command = request['command']
    if command == 'layout_preview':
        return {'layout_preview': layout_setup.capture(artifacts, storage, recipe, request,
                                                       capture_runtime, progress)}
    if command == 'layout_analyze':
        return _analyze(artifacts, storage, recipe, request, progress)
    if command == 'layout_save':
        return _save(artifacts, storage, request)
    raise data.DatasetError(f'Unknown layout operation: {command}')
