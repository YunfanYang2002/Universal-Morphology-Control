"""HD2C evidence boundaries; no optimizer, checkpoint assets, or simulator needed."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import run_hyperdistill_hd2c as runner


class Hd2cRunnerTests(unittest.TestCase):
    def test_package_keeps_native_manifests_and_epoch_history_not_old_failure(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / 'tmp') as td:
            root = Path(td)
            old = root / 'hd2b'
            old.mkdir()
            (old / 'failure.txt').write_text('old SIGTERM')
            output = root / 'hd2c'
            (output / 'teacher_reference').mkdir(parents=True)
            (output / 'teacher_reference/manifest.json').write_text('{"teacher": "bound"}')
            (output / 'teacher_reference/arrays.npz').write_bytes(b'raw excluded')
            (output / 'epoch_metrics.jsonl').write_text('{"epoch": 150}\n')
            archive = runner.package(output, {'status': {'HD2_FINAL': 'NOT_RUN'}})
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(set(bundle.namelist()), {
                    'teacher_reference/manifest.json', 'epoch_metrics.jsonl', 'manifest.json'})
                manifest = json.loads(bundle.read('manifest.json'))
            excluded = next(row for row in manifest['artifacts'] if row['relative_path'].endswith('.npz'))
            self.assertFalse(excluded['included_in_package'])
            self.assertEqual(manifest['status']['HD2_FINAL'], 'NOT_RUN')

    def test_missing_completed_run_fails_before_any_subprocess_and_packages(self):
        before = set((runner.ROOT / 'tmp').glob('hyperdistill_hd2c_ood6_*_audit.zip'))
        missing = runner.ROOT / 'tmp/hd2c_test_absent_completed_run'
        self.assertFalse(missing.exists())
        with patch('sys.argv', ['hd2c', '--hd2b-run', str(missing)]), \
                patch.object(runner.platform, 'node', return_value='test-host'), \
                patch.object(runner.subprocess, 'Popen', side_effect=AssertionError('must not start rollout')):
            self.assertEqual(runner.main(), 1)
        created = set((runner.ROOT / 'tmp').glob('hyperdistill_hd2c_ood6_*_audit.zip')) - before
        self.assertEqual(len(created), 1)
        with zipfile.ZipFile(created.pop()) as bundle:
            self.assertIn('failure.txt', bundle.namelist())
            status = json.loads(bundle.read('status.json'))
            self.assertEqual(status['HD2C_CHECKPOINT_5'], 'NOT_RUN')
            self.assertEqual(status['HD2_FINAL'], 'NOT_RUN')
            self.assertEqual(status['OOD_TRAINING_SAMPLES'], 0)


if __name__ == '__main__':
    unittest.main()
