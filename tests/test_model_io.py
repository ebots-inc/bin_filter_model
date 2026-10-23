"""Binary bundles retain classifier results and reject damaged payloads."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as ds
import height_field
import model_io
from test_height_field_cli import empty_cloud


class ModelIoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bins = {}
        clouds = [empty_cloud(seed) for seed in range(3)]
        for name, shift in [('bin_a', 0), ('bin_b', 40)]:
            config = {'kind': 'height_field', 'xy_bounds_mm': [[shift - 10, -10], [shift + 10, 10]],
                      'z_bounds_mm': [-135, -85], 'seed_xy_mm': [shift, 0],
                      'max_reference_height_mm': -93, 'min_capture_support': 3,
                      'min_tolerance_mm': .7, 'max_tolerance_mm': 3., 'alignment_margin_mm': .3}
            bins[name] = height_field.fit_bin(clouds, config)
        cls.model = {'model_type': 'offline_height_field_bins',
                     'frame_id': 'B2', 'length_unit': 'mm', 'scene_id': 0, 'bins': bins,
                     'exposure_settings_ms': [4., 8., 12., 16., 20.],
                     'calibration_sha256': {'camera': 'a'*64, 'robot': 'b'*64},
                     'training_captures': [{'id': f'train_{i}', 'input_signature': f'{i:064x}'} for i in range(3)]}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_binary_roundtrip_is_exact_and_preserves_classifier(self):
        path = model_io.write_model(self.root / 'bundle', self.model)
        document = json.loads(path.read_text())
        descriptor = document['bins']['bin_a']['surface_height_mm']
        self.assertEqual(descriptor['dtype'], 'float64_le')
        self.assertNotIn('storage_format', document)
        loaded = model_io.load_model(path.parent)
        self.assertEqual(loaded, self.model)
        probes = np.array([[0, 0, -100], [0, 0, -96], [3, 3, -90]])
        np.testing.assert_array_equal(height_field.classify(probes, loaded['bins']['bin_a'])[0],
                                      height_field.classify(probes, self.model['bins']['bin_a'])[0])
        self.assertLess(path.stat().st_size, 20000)
        with self.assertRaisesRegex(ds.DatasetError, 'overwrite'):
            model_io.write_model(path.parent, self.model)

    def test_profile_bundle_roundtrip_and_explicit_legacy_export(self):
        model = copy.deepcopy(self.model)
        model.pop('exposure_settings_ms')
        bins = model.pop('bins')
        model.update(model_type='offline_height_field_exposure_profiles',
                     profiles=[{'exposure_ms': exposure, 'bins': copy.deepcopy(bins),
                                'training_captures': [{'id': f'train_{j*3+i}', 'input_signature': f'{j*3+i:064x}'}
                                                      for i in range(3)]}
                               for j, exposure in enumerate((4., 20.))],
                     interpolation={'method': 'linear_height_max_tolerance', 'margin_mm': .2})
        model['training_captures'] = [item for profile in model['profiles'] for item in profile['training_captures']]
        path = model_io.write_model(self.root / 'profiles', model)
        loaded = model_io.load_model(path)
        self.assertEqual(loaded, model)
        legacy = model_io.write_model(self.root / 'legacy', loaded, 'yaml')
        self.assertEqual(model_io.load_model(legacy), model)
        cli = Path(__file__).resolve().parents[1] / 'bin_filter.py'
        result = subprocess.run([sys.executable, str(cli), 'convert-model', '--model', str(legacy),
                                 '--output', str(self.root / 'converted')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(model_io.load_model(self.root / 'converted'), model)

    def test_archived_version_tags_are_ignored_and_not_reexported(self):
        legacy = copy.deepcopy(self.model)
        legacy['schema_version'] = 1
        for field in legacy['bins'].values():
            field['schema_version'] = 2
        source = self.root / 'legacy.yaml'
        ds.write_yaml(source, legacy)
        loaded = model_io.load_model(source)
        self.assertEqual(loaded, self.model)
        path = model_io.write_model(self.root / 'converted_legacy', loaded)
        self.assertNotIn('schema_version', path.read_text())
        self.assertNotIn('storage_format', path.read_text())

    def test_payload_damage_and_descriptor_escapes_are_rejected(self):
        for mutation in ('truncate', 'shape', 'dtype', 'escape', 'infinity', 'negative_support'):
            with self.subTest(mutation=mutation):
                path = model_io.write_model(self.root / mutation, self.model)
                doc = json.loads(path.read_text())
                key = 'capture_support' if mutation == 'negative_support' else 'surface_height_mm'
                descriptor = doc['bins']['bin_a'][key]
                payload = path.parent / descriptor['file']
                if mutation == 'truncate':
                    payload.write_bytes(payload.read_bytes()[:-1])
                elif mutation == 'shape':
                    descriptor['shape'] = [1, 1]
                elif mutation == 'dtype':
                    descriptor['dtype'] = 'float32_le'
                elif mutation == 'escape':
                    descriptor['file'] = '../' + descriptor['file']
                else:
                    dtype = '<i8' if mutation == 'negative_support' else '<f8'
                    values = np.fromfile(payload, dtype=dtype)
                    values[0] = -1 if mutation == 'negative_support' else np.inf
                    values.tofile(payload)
                path.write_text(json.dumps(doc))
                with self.assertRaises(ValueError):
                    model_io.load_model(path)


if __name__ == '__main__':
    unittest.main()
