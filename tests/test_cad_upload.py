"""Real STL placement, upload confinement and reviewed-plan integration."""
from pathlib import Path
import json
import os
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cad_reference
import cad_upload
import dataset as data


def bowl():
    axis = np.linspace(-10, 10, 11)
    points = np.array([[[x, y, x*x/30 + y*y/50] for x in axis] for y in axis])
    triangles = []
    for row in range(10):
        for col in range(10):
            a, b, c, d = points[row, col], points[row, col+1], points[row+1, col], points[row+1, col+1]
            triangles.extend([[a, b, c], [b, d, c]])
    return np.asarray(triangles)


def stl_bytes(triangles):
    return bytes(80) + struct.pack('<I', len(triangles)) + b''.join(
        struct.pack('<12fH', 0., 0., 0., *triangle.ravel(), 0) for triangle in triangles)


def model():
    axis = np.arange(-16., 17.)
    transform = np.eye(4)
    transform[:3, 3] = [3., -2., 7.]
    surface = cad_reference.rasterize(bowl(), transform, axis, axis)
    return {'model_name': 'empty-model', 'bins': {'dish': {'geometry': {
        'x_centers_mm': axis, 'y_centers_mm': axis, 'surface_height_mm': surface,
        'capture_support': np.where(np.isfinite(surface), 5, 0), 'cell_size_mm': 1.,
        'fit_parameters': {'min_capture_support': 5}}, 'registration': {'surface_xyz_mm': []}}}}


class CadUploadTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.session = {'mode': 'guided', 'scene_id': 4, 'frame_id': 'station', 'bin_ids': ['dish'],
                        'captures': [{'id': 'empty-01', 'split': 'train', 'bins': {'dish': 'empty'}, 'path': 'captures/empty-01'}]}
        (self.folder / 'session.json').write_text(json.dumps(self.session))
        (self.folder / 'bin_config.yaml').write_text('bins: {}\n')
        capture = self.folder / 'captures/empty-01'
        capture.mkdir(parents=True)
        (capture / 'capture_manifest.yaml').write_text('status: complete\n')

    def upload(self, **kwargs):
        return cad_upload.upload(self.folder, self.session, 'Curved bin.STL', stl_bytes(bowl()), **kwargs)

    def test_upload_generates_readable_unique_names_and_preserves_original(self):
        first, second = self.upload(), self.upload()
        self.assertEqual(first['cad_uploaded']['id'], 'cad_uploads/reference-01')
        self.assertEqual(second['cad_uploaded']['id'], 'cad_uploads/reference-02')
        self.assertEqual((self.folder / 'cad_uploads/reference-01/reference.stl').read_bytes(), stl_bytes(bowl()))
        self.assertEqual(len(cad_upload.state(self.folder)['uploads']), 2)
        self.assertFalse((self.folder / 'artifacts').exists())

    def test_upload_rejects_traversal_invalid_stl_wrong_mode_and_oversize(self):
        for name, payload in [('../bin.stl', stl_bytes(bowl())), ('bin.txt', b'invalid'),
                              ('bin.stl', b'invalid'), ('bin.stl', bytes(cad_upload.MAX_UPLOAD_BYTES + 1))]:
            with self.subTest(name=name, size=len(payload)), self.assertRaises(ValueError):
                cad_upload.upload(self.folder, self.session, name, payload)
        self.assertEqual(cad_upload.state(self.folder)['uploads'], [])
        self.session['mode'] = 'check'
        with self.assertRaisesRegex(ValueError, 'collection session'):
            self.upload()

    def test_symlink_upload_directory_cannot_write_outside_session(self):
        outside = self.folder / 'outside'
        outside.mkdir()
        session = self.folder / 'session'
        session.mkdir()
        (session / 'cad_uploads').symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            cad_upload.upload(session, self.session, 'bin.stl', stl_bytes(bowl()))
        self.assertEqual(list(outside.iterdir()), [])

    def test_fixed_scale_translation_is_recovered_from_supported_empty_geometry(self):
        actual = model()['bins']['dish']
        pose, report, _ = cad_upload.fit_placement(bowl(), actual, '+Z', 0)
        np.testing.assert_allclose(pose[:3, 3], [3, -2, 7], atol=1e-6)
        np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-6)
        self.assertLess(report['absolute_residual_p95_mm'], 1e-6)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            cad_upload.fit_placement(bowl()*1000, actual, '+Z', 0)
        with self.assertRaisesRegex(ValueError, 'upward CAD axis'):
            cad_upload.fit_placement(bowl(), actual, '+Z', 7)

    def test_preview_requires_confirmation_and_saves_portable_plan_and_actual_picture(self):
        self.upload()
        selection = cad_upload.state(self.folder)['selection']
        selection['dish']['up_axis'] = '+Z'
        result = cad_upload.preview(self.folder, self.session, model(), selection, lambda _: None)
        relative = result['cad_preview']['path']
        self.assertTrue(result['cad_preview']['bins'][0]['preview_png'])
        with self.assertRaisesRegex(ValueError, 'confirm'):
            cad_upload.selected_plan(self.folder, 'session:' + relative, self.session)
        response = cad_upload.confirm(self.folder, self.session, relative)
        plan = cad_upload.selected_plan(self.folder, response['cad_confirmed'], self.session)
        self.assertEqual(plan, self.folder / relative)
        self.assertEqual(data.read_yaml(plan)['bins']['dish']['mesh_path'], 'dish.stl')
        self.assertEqual(len(cad_upload.state(self.folder)['references']), 1)
        # A changed collection cannot inherit an old CAD review.
        self.session['captures'][0]['id'] = 'different-training'
        with self.assertRaisesRegex(ValueError, 'different training'):
            cad_upload.confirm(self.folder, self.session, relative)

    def test_preview_has_no_heldout_capture_loader_and_each_bin_requires_settings(self):
        self.upload()
        with self.assertRaisesRegex(ValueError, 'every selected bin'):
            cad_upload.preview(self.folder, self.session, model(), {}, lambda _: None)
        with patch('dataset.load_capture_views', side_effect=AssertionError('No capture loading in placement')):
            cad_upload.preview(self.folder, self.session, model(), cad_upload.state(self.folder)['selection'], lambda _: None)

    def test_recollected_training_invalidates_confirmed_plan_even_with_same_capture_ids(self):
        self.upload()
        result = cad_upload.preview(self.folder, self.session, model(), cad_upload.state(self.folder)['selection'], lambda _: None)
        relative = result['cad_preview']['path']
        selected = cad_upload.confirm(self.folder, self.session, relative)['cad_confirmed']
        plan = self.folder / relative
        manifest = self.folder / self.session['captures'][0]['path'] / 'capture_manifest.yaml'
        later = (plan.parent / 'review.json').stat().st_mtime_ns + 1000000000
        os.utime(manifest, ns=(later, later))
        with self.assertRaisesRegex(ValueError, 'changed'):
            cad_upload.selected_plan(self.folder, selected, self.session)
        with self.assertRaisesRegex(ValueError, 'changed'):
            cad_upload.confirm(self.folder, self.session, relative)


if __name__ == '__main__':
    unittest.main()
