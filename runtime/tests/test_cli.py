"""Check process isolation, saved outputs and failure propagation."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest


class CLIContracts(unittest.TestCase):
    def call(self, *arguments):
        return subprocess.run([sys.executable, '-m', 'bpupga', *map(str, arguments)],
                              text=True, capture_output=True)

    def test_methods_and_bundled_resources(self):
        result = self.call('methods')
        self.assertEqual(result.returncode, 0, result.stderr)
        methods = json.loads(result.stdout)
        self.assertEqual(len(methods), 12)
        self.assertEqual(len({m['identifier'] for m in methods}), 12)
        self.assertTrue(all(m['devices'] == ['cpu', 'cuda'] for m in methods))
        check = self.call('doctor')
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertTrue(all(json.loads(check.stdout)['bundled_files'].values()))

    def test_graph_only_child_and_no_output_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'example'
            run = self.call('demo', '--method', 'grampa', '--device', 'cpu', '--output', output)
            self.assertEqual(run.returncode, 0, run.stderr)
            fit = output / 'grampa' / 'fit'
            packet = json.loads((fit / 'packet.json').read_text())
            self.assertEqual(set(packet), {'method', 'parameters', 'seed', 'device', 'family'})
            self.assertFalse((fit / 'Alignment.json').exists())
            record = json.loads((output / 'grampa/result.json').read_text())
            self.assertEqual(record['status'], 'SUCCESS')
            self.assertIn('pair', record['variants']['native'])
            repeat = self.call('demo', '--method', 'grampa', '--output', output)
            self.assertNotEqual(repeat.returncode, 0)
            invalid = self.call('run', '--data', output / 'inputs', '--method', 'grampa',
                                '--parameters', '{"ground_truth": {}}', '--output', output / 'invalid')
            self.assertNotEqual(invalid.returncode, 0)
            failure = json.loads((output / 'invalid/result.json').read_text())
            self.assertEqual(failure['status'], 'FAILED')
            self.assertNotIn('variants', failure)


if __name__ == '__main__':
    unittest.main()
