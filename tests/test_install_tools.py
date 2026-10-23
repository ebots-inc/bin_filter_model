from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import install_tools


class InstallToolsTests(unittest.TestCase):
    def test_python_refresh_preserves_native_binary_and_previous_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Path(temporary)
            parent = artifacts / 'data_saving/bin_model'
            native = parent / 'tools/native/bin_capture'
            native.parent.mkdir(parents=True)
            native.write_bytes(b'existing native executable')
            native.chmod(0o755)
            (parent / 'prepare_session.py').write_text('existing helper')
            (parent / 'tools/old_python_file.py').write_text('previous')
            for _ in range(2):
                installed = install_tools.install(artifacts)
                self.assertEqual((installed / 'native/bin_capture').read_bytes(), b'existing native executable')
                self.assertEqual((installed / 'native/bin_capture').stat().st_mode & 0o777, 0o755)
                self.assertTrue((installed / 'host_replay.py').is_file())
                self.assertTrue((parent / 'tools.previous/native/bin_capture').is_file())
            self.assertFalse((installed / 'old_python_file.py').exists())


if __name__ == '__main__':
    unittest.main()
