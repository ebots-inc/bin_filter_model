"""Session-local CAD uploads and operator-reviewed placement from empty training.

STL scale and orientation are explicit. Placement fits only measured empty-bin
geometry; it does not read held-out captures or change foreground thresholds.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

import cad_plan
import cad_reference
import dataset as data
import deployment
import model_io

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_TRIANGLES = 100000
SCALES = {'mm': 1., 'cm': 10., 'm': 1000., 'in': 25.4}
AXES = ('+X', '-X', '+Y', '-Y', '+Z', '-Z')


def _inside(folder, relative, prefix):
    folder = Path(folder).resolve()
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != prefix:
        raise data.DatasetError('CAD files must remain inside this session')
    result = (folder / path).resolve()
    if not result.is_relative_to(folder / prefix):
        raise data.DatasetError('CAD path escapes this session')
    return result


def _new_directory(parent, stem):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
        path = parent / f'{stem}-{index:02d}'
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise data.DatasetError('Too many CAD uploads or previews in this session')


def upload(folder, session, filename, payload, target='all'):
    if session.get('mode') != 'guided':
        raise data.DatasetError('CAD upload requires a model collection session')
    if (not isinstance(filename, str) or len(filename) > 200 or '/' in filename or '\\' in filename
            or Path(filename).suffix.lower() != '.stl'):
        raise data.DatasetError('Choose an STL file with a simple filename')
    if not isinstance(payload, bytes) or not 0 < len(payload) <= MAX_UPLOAD_BYTES:
        raise data.DatasetError('CAD upload must be an STL no larger than 16 MiB')
    if target != 'all' and target not in session['bin_ids']:
        raise data.DatasetError('Choose one of the open session bins')
    folder = Path(folder)
    destination = _new_directory(_inside(folder, 'cad_uploads', 'cad_uploads'), 'reference')
    try:
        mesh = destination / 'reference.stl'
        mesh.write_bytes(payload)
        triangles = cad_reference.read_stl(mesh, 1.)
        if len(triangles) > MAX_TRIANGLES:
            raise data.DatasetError('Simplify this STL to at most 100,000 triangles before uploading')
        bounds = np.ptp(triangles.reshape(-1, 3), axis=0)
        if np.count_nonzero(bounds > 1e-6) != 3:
            raise data.DatasetError('Upload the complete three-dimensional bin, including the top rim')
        info = {'id': str(destination.relative_to(folder)), 'filename': filename,
                'triangles': len(triangles), 'native_dimensions': bounds.tolist()}
        data.write_json(destination / 'upload.json', info)
        selection_path = folder / 'cad_uploads/selection.json'
        selection = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
        for name in session['bin_ids'] if target == 'all' else [target]:
            selection[name] = {'upload': info['id'], 'units': 'mm', 'up_axis': '+Z', 'yaw_deg': 0}
        deployment.atomic_text(selection_path, json.dumps(selection, indent=2))
        return {'cad_uploaded': info, 'cad_selection': selection}
    except Exception:
        shutil.rmtree(destination)
        raise


def state(folder):
    if folder is None:
        return {'uploads': [], 'selection': {}, 'references': []}
    folder = Path(folder)
    uploads = [json.loads(path.read_text()) for path in sorted((folder / 'cad_uploads').glob('reference-*/upload.json'))]
    selection_path = folder / 'cad_uploads/selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
    references = []
    for marker in sorted((folder / 'cad_previews').glob('preview-*/confirmed.json')):
        plan = marker.parent / 'plan.yaml'
        item = data.read_yaml(plan)
        references.append({'path': 'session:' + str(plan.relative_to(folder)), 'label': item['name'],
                           'scene_id': item['scene_id'], 'frame_id': item['frame_id'], 'bin_ids': list(item['bins'])})
    return {'uploads': uploads, 'selection': selection, 'references': references}


def selected_plan(folder, value, session):
    if not isinstance(value, str) or not value.startswith('session:'):
        raise data.DatasetError('Select a confirmed uploaded CAD reference')
    path = _inside(folder, value[len('session:'):], 'cad_previews')
    if path.name != 'plan.yaml' or not (path.parent / 'confirmed.json').is_file():
        raise data.DatasetError('Preview and confirm this CAD placement before building')
    _validate_review(folder, session, path)
    if max(path.stat().st_mtime_ns, (path.parent / 'review.json').stat().st_mtime_ns) > (path.parent / 'confirmed.json').stat().st_mtime_ns:
        raise data.DatasetError('CAD placement changed after review; preview and confirm it again')
    cad_plan.load(path, scene_id=session['scene_id'], frame_id=session['frame_id'], bin_ids=session['bin_ids'])
    return path


def _training_ids(session):
    return {item['id'] for item in session['captures'] if item['split'] == 'train'
            and set(item['bins'].values()) == {'empty'}}


def _validate_review(folder, session, plan):
    """Ordinary file freshness guards prevent reusing a review after recollection."""
    review_path = plan.parent / 'review.json'
    review = json.loads(review_path.read_text())
    if set(review['training_capture_ids']) != _training_ids(session):
        raise data.DatasetError('CAD preview belongs to different training captures')
    inputs = [Path(folder) / 'session.json', Path(folder) / 'bin_config.yaml']
    inputs.extend(Path(folder) / item['path'] / 'capture_manifest.yaml' for item in session['captures']
                  if item['id'] in _training_ids(session))
    if any(not path.is_file() or path.stat().st_mtime_ns > review_path.stat().st_mtime_ns for path in inputs):
        raise data.DatasetError('Empty training data or settings changed; preview and confirm CAD again')


def matching_baseline(path, folder, session):
    """Reuse only an unchanged session's measured-only, empty-trained geometry."""
    try:
        path = Path(path)
        model = model_io.load_model(path)
        if (model.get('model_type') != 'calibrated_bin_background'
                or model.get('scene_id') != session['scene_id'] or model.get('frame_id') != session['frame_id']
                or set(model['bins']) != set(session['bin_ids'])
                or {item['id'] for item in model['training_captures']} != _training_ids(session)
                or any('cad_reference' in item for item in model['bins'].values())
                or data.read_yaml(path.parent / 'settings.yaml') != data.read_yaml(Path(folder) / 'bin_config.yaml')):
            return None
        for capture in session['captures']:
            if capture['id'] in _training_ids(session):
                manifest = Path(folder) / capture['path'] / 'capture_manifest.yaml'
                if not manifest.is_file() or manifest.stat().st_mtime_ns > path.stat().st_mtime_ns:
                    return None
        return model
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _rotation(up, yaw):
    if up not in AXES or type(yaw) not in (int, float) or yaw not in (0, 90, 180, 270):
        raise data.DatasetError('Choose the upward CAD axis and a 0/90/180/270 degree rotation')
    # Right-handed mapping: +Y up gives X right and native -Z forward.
    rotations = {
        '+Z': np.eye(3), '-Z': np.diag([1., -1., -1.]),
        '+Y': np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]]),
        '-Y': np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]]),
        '+X': np.array([[0, 0, -1.], [0, 1., 0], [1., 0, 0]]),
        '-X': np.array([[0, 0, 1.], [0, 1., 0], [-1., 0, 0]])}
    angle = np.deg2rad(yaw)
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]]) @ rotations[up]


def _measured(item):
    geometry = item['geometry']
    x, y = (np.asarray(geometry[key], dtype=float) for key in ('x_centers_mm', 'y_centers_mm'))
    measured = np.asarray(geometry['surface_height_mm'], dtype=float).copy()
    support = np.asarray(geometry['capture_support'])
    observed = np.isfinite(measured) & (support >= geometry['fit_parameters']['min_capture_support'])
    measured[~observed] = np.nan
    # The fitted interior excludes the rim. Reuse saved measured rim samples,
    # only where the original independent capture support is sufficient.
    points = np.asarray(item['registration']['surface_xyz_mm'], dtype=float)
    if len(points):
        ix = np.rint((points[:, 0] - x[0]) / (x[1] - x[0])).astype(int)
        iy = np.rint((points[:, 1] - y[0]) / (y[1] - y[0])).astype(int)
        valid = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
        ix, iy, points = ix[valid], iy[valid], points[valid]
        valid = support[iy, ix] >= geometry['fit_parameters']['min_capture_support']
        measured[iy[valid], ix[valid]] = points[valid, 2]
    return x, y, measured


def fit_placement(triangles, item, up, yaw):
    """Fixed-scale/yaw placement with bounded XY search and robust Z fit.

    No nearest-neighbour/ML dependencies are needed. Small physical deviations
    remain measured; this placement only supplies nominal association geometry.
    """
    x, y, measured = _measured(item)
    rotation = _rotation(up, yaw)
    rotated = triangles @ rotation.T
    mesh_low, mesh_high = rotated.reshape(-1, 3).min(axis=0), rotated.reshape(-1, 3).max(axis=0)
    observed = np.isfinite(measured)
    rows, columns = np.nonzero(observed)
    if len(rows) < 100:
        raise data.DatasetError('CAD preview needs at least 100 supported empty-training cells')
    measured_size = np.array([np.ptp(x[columns]), np.ptp(y[rows])])
    ratio = (mesh_high - mesh_low)[:2] / measured_size
    if np.any(ratio < .7) or np.any(ratio > 1.4):
        raise data.DatasetError('CAD dimensions do not match the measured bin; check units and upward axis')
    transform = np.eye(4)
    transform[:3, :3] = rotation
    center = np.array([(x[columns].min() + x[columns].max()) / 2,
                       (y[rows].min() + y[rows].max()) / 2])
    transform[:2, 3] = center - (mesh_low[:2] + mesh_high[:2]) / 2
    # Raster once at native grid spacing; candidate XY shifts are exact grid
    # translations. A robust median removes global Z before scoring shape.
    nominal = cad_reference.rasterize(triangles, transform, x, y)
    cell = float(item['geometry']['cell_size_mm'])
    limit = min(20, max(1, int(np.ceil(10 / cell))))
    stride = max(1, int(np.ceil(len(rows) / 6000)))
    rows, columns = rows[::stride], columns[::stride]
    best = None
    for dy in range(-limit, limit + 1):
        for dx in range(-limit, limit + 1):
            cy, cx = rows - dy, columns - dx
            valid = (cy >= 0) & (cy < len(y)) & (cx >= 0) & (cx < len(x))
            residual = measured[rows[valid], columns[valid]] - nominal[cy[valid], cx[valid]]
            residual = residual[np.isfinite(residual)]
            if len(residual) < .8 * len(rows):
                continue
            dz = float(np.median(residual))
            score = float(np.quantile(np.abs(residual - dz), .8))
            candidate = (score, abs(dx) + abs(dy), dx, dy, dz)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
    if best is None:
        raise data.DatasetError('CAD does not overlap the measured bin sufficiently; check axis and orientation')
    transform[:3, 3] += [best[2] * cell, best[3] * cell, best[4]]
    nominal = cad_reference.rasterize(triangles, transform, x, y)
    common = np.isfinite(measured) & np.isfinite(nominal)
    overlap = float(common.sum() / max(1, np.isfinite(nominal).sum()))
    p95 = float(np.quantile(np.abs(measured[common] - nominal[common]), .95))
    retained = float(common.sum() / observed.sum())
    if overlap < .5 or retained < .9 or p95 > 10:
        raise data.DatasetError(f'CAD placement needs review: {overlap:.0%} footprint observed, '
                               f'{retained:.0%} measured surface overlap, p95 difference {p95:.2f} mm. '
                               'Check the STL, units, upward axis and rotation; try another quarter turn.')
    return transform, {'observed_overlap_fraction': overlap, 'measured_overlap_fraction': retained,
        'absolute_residual_p95_mm': p95, 'dimensions_mm': (mesh_high - mesh_low).tolist(),
        'method': 'fixed_scale_and_yaw_bounded_xy_empty_training_height_fit'}, (x, y, measured, nominal)


def _picture(name, surfaces):
    """Actual top outlines and central height sections, no decorative mock data."""
    x, y, measured, nominal = surfaces
    canvas = np.full((410, 1060, 3), 255, np.uint8)
    blue, orange, gray = (180, 100, 25), (15, 130, 230), (95, 95, 95)
    cv2.putText(canvas, f'{name}: blue = measured empty bin; orange = CAD', (18, 26),
                cv2.FONT_HERSHEY_SIMPLEX, .63, gray, 1, cv2.LINE_AA)
    for index, (mask, color) in enumerate(((np.isfinite(measured), blue), (np.isfinite(nominal), orange))):
        tile = cv2.resize(mask.astype(np.uint8), (310, 310), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(tile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour[:, 0, 0] += 15
            contour[:, 0, 1] = 370 - contour[:, 0, 1]
            cv2.drawContours(canvas, [contour], -1, color, 2)
    cv2.putText(canvas, 'Top outline (+X right, +Y up)', (15, 395), cv2.FONT_HERSHEY_SIMPLEX, .5, gray, 1)
    low = float(np.nanmin([np.nanmin(measured), np.nanmin(nominal)]))
    high = float(np.nanmax([np.nanmax(measured), np.nanmax(nominal)]))
    for axis, left in ((0, 370), (1, 720)):
        mid = np.argmin(np.abs(y if axis == 0 else x))
        abscissa = x if axis == 0 else y
        for surface, color in ((measured, blue), (nominal, orange)):
            section = surface[mid, :] if axis == 0 else surface[:, mid]
            xx = left + (abscissa - abscissa.min()) / np.ptp(abscissa) * 310
            yy = 370 - (section - low) / max(1., high - low) * 300
            for i in range(len(section) - 1):
                if np.isfinite(section[i:i+2]).all():
                    cv2.line(canvas, (int(xx[i]), int(yy[i])), (int(xx[i+1]), int(yy[i+1])), color, 2)
        cv2.putText(canvas, ('X' if axis == 0 else 'Y') + ' center height section', (left, 395),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, gray, 1)
    ok, encoded = cv2.imencode('.png', canvas)
    if not ok:
        raise data.DatasetError('Could not render CAD preview')
    return encoded.tobytes()


def preview(folder, session, model, selections, progress):
    if not isinstance(selections, dict) or set(selections) != set(session['bin_ids']):
        raise data.DatasetError('Choose a CAD file and orientation for every selected bin')
    folder = Path(folder)
    output = _new_directory(_inside(folder, 'cad_previews', 'cad_previews'), 'preview')
    plan = {'name': 'Uploaded CAD ' + output.name, 'scene_id': session['scene_id'],
            'frame_id': session['frame_id'], 'bins': {}}
    reports = []
    try:
        for name, selection in selections.items():
            if not isinstance(selection, dict) or set(selection) != {'upload', 'units', 'up_axis', 'yaw_deg'}:
                raise data.DatasetError('CAD placement requires file, units, upward axis and rotation')
            scale = SCALES.get(selection['units'])
            if scale is None:
                raise data.DatasetError('Choose STL units: mm, cm, m or inches')
            source = _inside(folder, selection['upload'], 'cad_uploads')
            info = json.loads((source / 'upload.json').read_text())
            triangles = cad_reference.read_stl(_inside(folder, source.relative_to(folder) / 'reference.stl', 'cad_uploads'), scale)
            progress(f'{name}: matching CAD to supported empty-training measurements; scale and quarter turn are fixed.')
            transform, diagnostics, surfaces = fit_placement(triangles, model['bins'][name],
                                                            selection['up_axis'], selection['yaw_deg'])
            mesh = output / f'{name}.stl'
            shutil.copyfile(source / 'reference.stl', mesh)
            plan['bins'][name] = {'mesh_path': mesh.name, 'scale_to_mm': scale,
                                  'bin_from_cad': transform.tolist(), 'alignment': {'enabled': False}}
            png = _picture(name, surfaces)
            (output / f'{name}.png').write_bytes(png)
            reports.append({'bin_id': name, 'filename': info['filename'], **diagnostics,
                            'preview_png': base64.b64encode(png).decode('ascii')})
        data.write_yaml(output / 'plan.yaml', plan)
        data.write_json(output / 'review.json', {'selections': selections, 'bins': [
            {key: value for key, value in item.items() if key != 'preview_png'} for item in reports],
            'training_capture_ids': sorted(_training_ids(session)), 'source_model_name': model['model_name']})
        deployment.atomic_text(folder / 'cad_uploads/selection.json', json.dumps(selections, indent=2))
        return {'cad_preview': {'path': str((output / 'plan.yaml').relative_to(folder)), 'bins': reports,
                              'note': 'Review the CAD orientation and size. Differences include real shape and sensor error. '
                                      'Only empty-training measurements were used; held-out validation still must pass.'}}
    except Exception:
        shutil.rmtree(output)
        raise


def confirm(folder, session, relative):
    path = _inside(folder, relative, 'cad_previews')
    if path.name != 'plan.yaml':
        raise data.DatasetError('Choose a generated CAD preview')
    cad_plan.load(path, scene_id=session['scene_id'], frame_id=session['frame_id'], bin_ids=session['bin_ids'])
    _validate_review(folder, session, path)
    deployment.atomic_text(path.parent / 'confirmed.json', json.dumps({
        'confirmed_at': datetime.now(timezone.utc).isoformat(), 'operator_reviewed_size_and_orientation': True}))
    return {'cad_confirmed': 'session:' + str(path.relative_to(Path(folder)))}
