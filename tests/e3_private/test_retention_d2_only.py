"""Host contracts for the isolated direct-D2 retention diagnostic."""
from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'tests/e3_private/retention_arena_diagnostic_driver.esk'
RUNTIME = ROOT / 'tests/e3_private/retention_arena_diagnostic_runtime.esk'
PAIR = ROOT / 'scripts/e3-retention-d2-arena-pair.sh'
PARENT = 'ef1260332e6a8ae75da60ee13de919ce3eae94bc'


class D2ArenaDiagnosticTest(unittest.TestCase):
    def test_original_failure_and_report_assertions_are_unchanged(self):
        old = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                       f'{PARENT}:tests/e3_private/retention_arena_diagnostic_driver.esk'],
                                      text=True)
        new = DRIVER.read_text()
        start = '(define (e3-test-retention-fail-retry!'
        end = '(define (e3-test-retention-reconcile!'
        self.assertEqual(old[old.index(start):old.index(end)],
                         new[new.index(start):new.index(end)])
        start = '(define (e3-test-retention-report'
        end = '(define (e3-test-retention-teardown'
        self.assertEqual(old[old.index(start):old.index(end)],
                         new[new.index(start):new.index(end)])
        self.assertEqual(subprocess.check_output(
            ['git', '-C', str(ROOT), 'show',
             f'{PARENT}:native/e3_private_extension.esk']),
            (ROOT / 'native/e3_private_extension.esk').read_bytes())

    def test_direct_fault_is_matched_and_retry_is_real(self):
        body = DRIVER.read_text()
        start = body.index('(define (e3-test-retention-diagnose-d2')
        end = body.index('(define (e3-test-retention-run', start)
        probe = body[start:end]
        for text in ('(if (= index 512)', '(e3-test-read-fail-stage (if fault? 1 0))',
                     '(d2-token-dataset-next-batch dataset)',
                     '(d2-token-batch-release! batch)',
                     '(d2-token-dataset-seek! dataset cursor)',
                     "'token-dataset-next-batch", "'d2-direct-outcome",
                     "'d2-direct-cursor", "'d2-direct-no-live-batch-or-borrow",
                     '(e3-test-retention-call! model dataset frame destinations cursor modes'):
            self.assertIn(text, probe)
        self.assertLess(probe.index('(d2-token-dataset-next-batch dataset)'),
                        probe.index('(vector-set! samples 2 (__arena-used))'))
        self.assertLess(probe.index('(vector-set! samples 2 (__arena-used))'),
                        probe.index("'d2-direct-cursor"))
        self.assertLess(probe.index("'d2-direct-cursor"),
                        probe.rindex('(e3-test-retention-call!'))
        runtime = RUNTIME.read_text()
        self.assertIn('(exit 0)))', runtime)
        self.assertLess(runtime.index('(exit 0)))'),
                        runtime.index('(e3-retention-test-run diagnose input output)'))
        self.assertIn('(e3-retention-test-run report input output)', runtime)
        self.assertIn('(e3-retention-test-run teardown input output)', runtime)

    def test_native_d2_live_selector_negative(self):
        native = (ROOT / 'src/eshkol_transformer/e3_frame.c').read_text()
        self.assertIn('case 23: return et_d2_batch_test_live_count_v1();', native)
        self.assertIn('case 24: return et_d2_batch_test_borrow_count_v1();', native)
        body = DRIVER.read_text()
        start = body.index("'d2-direct-no-live-batch-or-borrow")
        end = body.index('(vector-set! samples 3', start)
        selectors = [int(value) for value in re.findall(
            r'\(= \(e3-test-retention-probe ([0-9]+)\) 0\)', body[start:end])]
        self.assertEqual(selectors, [23, 24])
        for leaked in (23, 24):
            observed = {23: 0, 24: 0}
            observed[leaked] = 1
            self.assertFalse(all(observed[index] == 0 for index in selectors))

    def test_pair_runs_control_after_fault_failure_in_fresh_process(self):
        subprocess.run(['bash', '-n', str(PAIR)], check=True)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / 'corpus').mkdir()
            runtime = root / 'fake-runtime'
            runtime.write_text('''#!/usr/bin/env bash
printf '%s %s\\n' "$2" "$$" >> "$CALL_LOG"
if [[ "$2" == d2-fault ]]; then exit 23; fi
printf 'E3-RETENTION-D2-ARENA mode=d2-control before=1 after-arm=1 after-call=1 after-checks=1 after-retry=1\\n'
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
            self.assertEqual([row[0] for row in calls], ['d2-fault', 'd2-control'])
            self.assertEqual(len({row[1] for row in calls}), 2)
            self.assertEqual(result.stdout.splitlines(),
                             ['normal-a\td2-fault\t23\t0',
                              'normal-a\td2-control\t0\t1'])


if __name__ == '__main__':
    unittest.main()
