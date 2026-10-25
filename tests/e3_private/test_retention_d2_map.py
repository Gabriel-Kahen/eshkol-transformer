"""Host contracts for the separate D2/E3 error-mapping diagnostic."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PARENT = '8fec18f2dba9c1f5a766b2bc9836ed42fbc690f1'
DRIVER = ROOT / 'tests/e3_private/retention_arena_diagnostic_driver.esk'
RUNTIME = ROOT / 'tests/e3_private/retention_arena_diagnostic_runtime.esk'
PAIR = ROOT / 'scripts/e3-retention-d2-map-pair.sh'


def old(relative):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                    f'{PARENT}:{relative}'])


class D2MapProbeTest(unittest.TestCase):
    def test_production_and_predecessor_diagnostics_are_unchanged(self):
        for relative in ('internal/d2/lib/d2_dataset.esk',
                         'native/e3_private_extension.esk',
                         'native/e3_errors_extension.esk',
                         'native/m3_model_extension.esk',
                         'tests/e3_private/retention_arena_diagnostic_bridge.c'):
            self.assertEqual((ROOT / relative).read_bytes(), old(relative))
        previous = old('tests/e3_private/retention_arena_diagnostic_driver.esk').decode()
        current = DRIVER.read_text()
        marker = ';;; Test-only mapper probe.'
        self.assertEqual(previous[:previous.index('(define (e3-test-retention-run')],
                         current[:current.index(marker)])
        prior_runtime = old('tests/e3_private/retention_arena_diagnostic_runtime.esk').decode()
        inherited = '(if d2-only?'
        self.assertEqual(prior_runtime[prior_runtime.index(inherited):],
                         RUNTIME.read_text()[RUNTIME.read_text().index(inherited):])

    def test_exact_promotion_and_mapper_seams(self):
        d2 = (ROOT / 'internal/d2/lib/d2_dataset.esk').read_text()
        self.assertIn('(let ((promoted (region-close region caught)))', d2)
        self.assertIn('(vector-set! escape 0 promoted)', d2)
        self.assertIn('(raise (vector-ref escape 0))', d2)
        e3 = (ROOT / 'native/e3_private_extension.esk').read_text()
        self.assertIn('(e3-note-caught-error! entry stage caught inspect-args)', e3)
        self.assertIn('(vector-set! entry 11 (vector-ref e3-error-table index))', e3)
        body = DRIVER.read_text()
        start = body.index('(define (e3-test-retention-diagnose-d2-map')
        end = body.index('(define (e3-test-retention-run', start)
        probe = body[start:end]
        for anchor in ('(if (= index 512)', '(e3-test-read-fail-stage 1)',
                       '(d2-token-dataset-next-batch dataset)',
                       "'d2-map-escape-identity", '(e3-note-caught-error! entry 5 caught inspect-args)',
                       '(e3-error-table-ref 5 \'io)',
                       '(e3-test-retention-probe 23)', '(e3-test-retention-probe 24)',
                       '(e3-test-retention-call! model dataset frame destinations cursor modes'):
            self.assertIn(anchor, probe)
        before = '(vector-set! samples 3 (__arena-used))'
        mapper = '(e3-note-caught-error! entry 5 caught inspect-args)'
        after = '(vector-set! samples 4 (__arena-used))'
        self.assertIn(before + '\n                ' + mapper + '\n                ' + after,
                      probe)
        self.assertLess(probe.index('(vector-set! samples 2 (__arena-used))'),
                        probe.index(before))
        self.assertLess(probe.index(after),
                        probe.index('(vector-set! samples 5 (__arena-used))'))

    def test_fresh_control_runs_after_map_failure(self):
        subprocess.run(['bash', '-n', str(PAIR)], check=True)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / 'corpus').mkdir()
            runtime = root / 'fake-runtime'
            runtime.write_text('''#!/usr/bin/env bash
printf '%s %s\\n' "$2" "$$" >> "$CALL_LOG"
if [[ "$2" == d2-map-fault ]]; then exit 23; fi
printf 'E3-RETENTION-D2-ARENA mode=%s before=1 after-arm=1 after-call=2 after-checks=2 after-retry=2\\n' "$2"
''')
            runtime.chmod(0o755)
            time_tool = root / 'fake-time'
            time_tool.write_text('#!/usr/bin/env bash\nshift 4\nexec "$@"\n')
            time_tool.chmod(0o755)
            env = dict(os.environ, CALL_LOG=str(root / 'calls'))
            result = subprocess.run(['bash', str(PAIR), str(root), str(runtime),
                                     str(root / 'corpus'), 'normal-a', '0', '10',
                                     str(time_tool)], capture_output=True,
                                    text=True, env=env)
            self.assertEqual(result.returncode, 1, result.stderr)
            calls = [line.split() for line in (root / 'calls').read_text().splitlines()]
            self.assertEqual([row[0] for row in calls],
                             ['d2-fault', 'd2-map-fault', 'd2-control'])
            self.assertEqual(len({row[1] for row in calls}), 3)
            self.assertEqual(result.stdout.splitlines(),
                             ['normal-a\td2-fault\t0\t1',
                              'normal-a\td2-map-fault\t23\t0',
                              'normal-a\td2-control\t0\t1'])

    def test_map_marker_requires_separate_pre_map_sample(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / 'corpus').mkdir()
            runtime = root / 'fake-runtime'
            runtime.write_text('''#!/usr/bin/env bash
if [[ "$2" == d2-map-fault ]]; then
  printf 'E3-RETENTION-D2-MAP before=1 after-arm=1 caught=2 after-map=3 after-call=3 after-checks=3 after-retry=3\\n'
else
  printf 'E3-RETENTION-D2-ARENA mode=%s before=1 after-arm=1 after-call=2 after-checks=2 after-retry=2\\n' "$2"
fi
''')
            runtime.chmod(0o755)
            time_tool = root / 'fake-time'
            time_tool.write_text('#!/usr/bin/env bash\nshift 4\nexec "$@"\n')
            time_tool.chmod(0o755)
            result = subprocess.run(['bash', str(PAIR), str(root), str(runtime),
                                     str(root / 'corpus'), 'normal-a', '0', '10',
                                     str(time_tool)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(result.stdout.splitlines(),
                             ['normal-a\td2-fault\t0\t1',
                              'normal-a\td2-map-fault\t0\t0',
                              'normal-a\td2-control\t0\t1'])


if __name__ == '__main__':
    unittest.main()
