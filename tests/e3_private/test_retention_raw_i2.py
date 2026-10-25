"""Host-only contract for the diagnostic I2 raw probe successor."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'tests/e3_private/retention_arena_diagnostic_driver.esk'
RUNTIME = ROOT / 'tests/e3_private/retention_arena_diagnostic_runtime.esk'
PAIR = ROOT / 'scripts/e3-retention-arena-raw-pair.sh'
BASE = '4ab378a373ecad342907ca8f66d02a29a16d2537'


def old(path: Path) -> bytes:
    return subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', f'{BASE}:{path.relative_to(ROOT)}'])


class RetentionRawI2Tests(unittest.TestCase):
    def test_report_prints_exact_native_snapshot_words_before_preserved_assertion(self):
        previous = old(DRIVER).decode()
        current = DRIVER.read_text()
        report = '(define (e3-test-retention-report horizon present?)'
        teardown = '(define (e3-test-retention-teardown horizon present?)'
        reconcile = '(define (e3-test-retention-reconcile! before after)'
        prepare = '(define (e3-test-retention-prepare horizon present? corpus-path)'
        self.assertEqual(previous[:previous.index(report)], current[:current.index(report)])
        run = '(define (e3-test-retention-run phase horizon present? corpus-path)'
        self.assertEqual(previous[previous.index(teardown):previous.index(run)],
                         current[current.index(teardown):current.index(';;; Diagnostic-only matched D2 call')])
        self.assertEqual(previous[previous.index(reconcile):previous.index(prepare)],
                         current[current.index(reconcile):current.index(prepare)])
        body = current[current.index(report):current.index(teardown)]
        start = body.index('(after (e3-test-retention-probes))')
        marker = body.index('(display "E3-RETENTION-I2-RAW before-begin=")')
        flush = body.index('(flush-output-port (current-output-port))')
        assertion = body.index('(e3-test-retention-reconcile! before after)')
        self.assertLess(start, marker)
        self.assertLess(marker, flush)
        self.assertLess(flush, assertion)
        for side in ('before', 'after'):
            for index in (14, 15, 16):
                self.assertEqual(body.count(f'(vector-ref {side} {index})'), 1)
        self.assertIn("'balanced-i2-views-pins", current)
        runtime = RUNTIME.read_text()
        inherited = '(e3-retention-test-run diagnose input output)'
        previous_runtime = old(RUNTIME).decode()
        self.assertEqual(previous_runtime[previous_runtime.index(inherited):],
                         runtime[runtime.index(inherited):])
        for rel in ('tests/e3_private/retention_arena_diagnostic_bridge.c',
                    'tests/e3_private/retention_arena_diagnostic_driver_root.esk',
                    'tests/e3_private/retention_arena.cpp'):
            path = ROOT / rel
            self.assertEqual(path.read_bytes(), old(path))

    def run_pair(self, runtime_body: str):
        with tempfile.TemporaryDirectory(prefix='e3-raw-i2-host-') as raw:
            root = Path(raw)
            corpus = root / 'corpus'
            corpus.mkdir()
            runtime = root / 'fake-runtime'
            runtime.write_text('#!/usr/bin/env bash\n' + runtime_body)
            runtime.chmod(0o755)
            time_tool = root / 'fake-time'
            time_tool.write_text('#!/usr/bin/env bash\nshift 4\nexec "$@"\n')
            time_tool.chmod(0o755)
            calls = root / 'calls'
            result = subprocess.run(
                ['bash', str(PAIR), str(root), str(runtime), str(corpus),
                 'normal-a', '0', '10', str(time_tool)], capture_output=True,
                text=True, env={**os.environ, 'CALL_LOG': str(calls)})
            return result, calls.read_text().splitlines()

    def test_fault_exit_preserves_raw_and_runs_fresh_control(self):
        result, calls = self.run_pair('''
printf '%s %s\\n' "$2" "$$" >> "$CALL_LOG"
printf 'E3-RETENTION-ARENA-DIAGNOSTIC mode=%s prepared=1 before=2 after-arm=3 after-call=4 after-checks=5 after-retry=6\\n' "$2"
printf 'E3-RETENTION-I2-RAW before-begin=1 before-view=2 before-end=3 after-begin=4 after-view=5 after-end=6\\n'
if [[ "$2" == fault ]]; then exit 23; fi
printf 'E3-RETENTION-ARENA-POSTCHECK-PASS mode=%s\\n' "$2"
''')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual([row.split()[0] for row in calls], ['fault', 'control'])
        self.assertEqual(len({row.split()[1] for row in calls}), 2)
        self.assertEqual(result.stdout.splitlines(),
                         ['normal-a\tfault\t23\t1\t1\t0',
                          'normal-a\tcontrol\t0\t1\t1\t1'])

    def test_missing_raw_is_nonaccepting_and_control_still_runs(self):
        result, calls = self.run_pair('''
printf '%s\\n' "$2" >> "$CALL_LOG"
printf 'E3-RETENTION-ARENA-DIAGNOSTIC mode=%s prepared=1 before=2 after-arm=3 after-call=4 after-checks=5 after-retry=6\\n' "$2"
printf 'E3-RETENTION-ARENA-POSTCHECK-PASS mode=%s\\n' "$2"
''')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(calls, ['fault', 'control'])
        self.assertEqual(result.stdout.splitlines(),
                         ['normal-a\tfault\t0\t1\t0\t1',
                          'normal-a\tcontrol\t0\t1\t0\t1'])

    def test_pair_shell_parses_and_old_recipe_is_unmodified(self):
        subprocess.run(['bash', '-n', str(PAIR)], check=True)
        old_pair = ROOT / 'scripts/e3-retention-arena-phase-pair.sh'
        old_gate = ROOT / 'scripts/test-e3-retention-arena-phase-v2.sh'
        self.assertEqual(old_pair.read_bytes(), old(old_pair))
        self.assertEqual(old_gate.read_bytes(), old(old_gate))


if __name__ == '__main__':
    unittest.main()
