"""The bounded metric command is mandatory and fails through existing CI tiers."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from tests.ci.topology import check

ROOT = Path(__file__).resolve().parents[2]


class MetricsRegistration(unittest.TestCase):
    def test_canonical_producer_cannot_disappear(self):
        path = 'scripts/build.sh'
        original = (ROOT / path).read_text()
        with self.assertRaises(AssertionError):
            check(ROOT, {path: original.replace('/usr/bin/bash "${PROJECT_ROOT}/scripts/build-e3-metrics.sh"', ':')})

    def test_failed_producer_stops_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / 'scripts'
            scripts.mkdir()
            for name in ('common.sh', 'ci-build-prerequisites.sh'):
                (scripts / name).write_text((ROOT / 'scripts' / name).read_text())
            for name in ('clean', 'configure', 'build-k1', 'build-a2', 'build-l2', 'build-l3s', 'build-e3-metrics', 'build-i1'):
                status = 37 if name == 'build-e3-metrics' else 0
                (scripts / f'{name}.sh').write_text(f'printf "%s\\n" "{name}" >> "$E3_REGISTRATION_LOG"\nexit {status}\n')
            log = root / 'calls'
            result = subprocess.run(['/usr/bin/bash', str(scripts / 'ci-build-prerequisites.sh'), 'native-numerics'],
                                    env={**os.environ, 'E3_REGISTRATION_LOG': str(log)}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 37, result.stderr)
            self.assertEqual(log.read_text().splitlines(), ['clean', 'configure', 'build-k1', 'build-a2', 'build-l2', 'build-l3s', 'build-e3-metrics'])

    def test_failed_metric_test_stops_each_tier(self):
        for target in ('test-after-build', 'test-ci-core-after-build', 'test-acceptance-predecessors-after-build'):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'Makefile').write_text((ROOT / 'Makefile').read_text())
                scripts = root / 'scripts'
                scripts.mkdir()
                preceding = ('test', 'check_a0_api_contract', 'test-k1', 'test-a2', 'test-l2', 'test-l3s')
                for name in preceding:
                    (scripts / f'{name}.sh').write_text('exit 0\n')
                (scripts / 'test-e3-metrics.sh').write_text('printf "metrics rejection\\n" >&2\nexit 37\n')
                result = subprocess.run(['make', target], cwd=root, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('metrics rejection', result.stderr)
                self.assertNotIn('test-i1.sh', result.stdout)
                self.assertNotIn('test-e1.sh', result.stdout)


if __name__ == '__main__':
    unittest.main()
